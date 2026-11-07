"""Auditable detector-only WSP adaptation. Phases never silently run a fresh test."""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

POOLS = ["last", "mean_answer", "mean_turn"]
SEED = 20261107
ALPHAS = [0.01, 0.02, 0.03, 0.04, 0.05]
COUNTS = {"score_calls": 0, "score_rows": 0, "node_logit_calls": 0, "node_logit_rows": 0, "sklearn_predict_calls": 0, "sklearn_predict_rows": 0}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_theta(path, theta):
    np.savez(path, **theta)


def load_theta(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


class Data:
    def __init__(self, root, names):
        root = Path(root)
        meta = pd.read_csv(root / "meta.csv", dtype={"row_id": str, "group_id": str})
        idx = np.flatnonzero(meta.partition.isin(names).to_numpy())
        self.meta = meta.iloc[idx].reset_index(drop=True)
        self.X = np.stack([np.asarray(np.load(root / f"acts_{p}.npy", mmap_mode="r", allow_pickle=False)[idx], dtype=np.float64) for p in POOLS], axis=1)
        self.y = self.meta.label.to_numpy(dtype=int)
        assert np.isfinite(self.X).all()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def node_logits(th, X):
    COUNTS["node_logit_calls"] += 1
    COUNTS["node_logit_rows"] += len(X)
    return np.einsum("npld,pld->npl", (X - th["mu"]) / th["sd"], th["W"]) + th["b"]


def score(th, X):
    COUNTS["score_calls"] += 1
    COUNTS["score_rows"] += len(X)
    return np.sum(sigmoid(node_logits(th, X)) * th["a"], axis=(1, 2))


def validate(th):
    assert set(th) == {"W", "b", "a", "mu", "sd"}
    assert th["W"].shape == th["mu"].shape == th["sd"].shape
    assert th["b"].shape == th["a"].shape == th["W"].shape[:2]
    assert np.all(th["sd"] > 0) and np.all(th["a"] >= 0)
    assert abs(th["a"].sum() - 1) < 1e-10
    assert all(np.isfinite(v).all() for v in th.values())


def threshold(neg, alpha):
    ordered = np.sort(np.asarray(neg))
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    return float(ordered[rank - 1]) if rank <= len(ordered) else float("inf")


def metrics(meta, s, fixed_threshold=None):
    y, fam = meta.label.to_numpy(), meta.family.to_numpy()
    per_auc = {f: float(roc_auc_score(y[fam == f], s[fam == f])) for f in sorted(set(fam)) if len(set(y[fam == f])) == 2}
    result = {"macro_auroc": float(np.mean(list(per_auc.values()))) if per_auc else None, "per_family_auroc": per_auc}
    if fixed_threshold is not None:
        alarm = s > fixed_threshold
        result.update(tp=int(np.sum(alarm & (y == 1))), fn=int(np.sum(~alarm & (y == 1))),
                      fp=int(np.sum(alarm & (y == 0))), tn=int(np.sum(~alarm & (y == 0))),
                      recall=float(alarm[y == 1].mean()) if np.any(y == 1) else None,
                      fpr=float(alarm[y == 0].mean()) if np.any(y == 0) else None)
        return result
    recalls = []
    for alpha in ALPHAS:
        cutoff = threshold(s[y == 0], alpha)
        vals = [float((s[(fam == f) & (y == 1)] > cutoff).mean()) for f in sorted(set(fam)) if np.any((fam == f) & (y == 1))]
        result[f"macro_recall_{alpha:.2f}"] = float(np.mean(vals))
        recalls.append(np.mean(vals))
    result["selection_score"] = float(np.mean(recalls))
    return result


def fit(th, D, active):
    out = copy.deepcopy(th)
    references = []
    for p, l in active:
        x = (D.X[:, p, l] - out["mu"][p, l]) / out["sd"][p, l]
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(x, D.y)
        assert clf.n_iter_[0] < 2000, "ordinary fit did not converge"
        out["W"][p, l], out["b"][p, l] = clf.coef_[0], clf.intercept_[0]
        references.append(clf.predict_proba(x)[:, 1] * out["a"][p, l])
        COUNTS["sklearn_predict_calls"] += 1
        COUNTS["sklearn_predict_rows"] += len(x)
    replay_error = float(np.max(np.abs(score(out, D.X) - np.sum(references, axis=0))))
    assert replay_error < 1e-6, replay_error
    return out, replay_error


def audit(D, th, root, capture):
    s, z = score(th, D.X), node_logits(th, D.X)
    cutoff = threshold(s[D.y == 0], 0.05)
    table = D.meta.copy()
    table["score"], table["threshold"] = s, cutoff
    table["prediction"] = (s > cutoff).astype(int)
    table["error"] = np.where((D.y == 0) & (s > cutoff), "FP", np.where((D.y == 1) & (s <= cutoff), "FN", np.where(D.y == 0, "TN", "TP")))
    table.to_csv(root / "development_predictions.csv", index=False)
    rows = []
    for by in ["family", "quadrant"]:
        for name, block in table.groupby(by):
            rows.append({"grouping": by, "value": name, **metrics(block, block.score.to_numpy(), cutoff)})
    write_json(root / "error_audit.json", rows)
    cases = pd.read_json(Path(capture) / "cases_inspect.jsonl", lines=True)
    case_map = cases.set_index("row_id").to_dict(orient="index")
    selected = []
    # Stable sample over different groups and families, not only extreme scores.
    for kind, count in [("FP", 2), ("FN", 2), ("TN", 1), ("TP", 1)]:
        block = table[table.error == kind].drop_duplicates("group_id").sort_values(["family", "row_id"])
        taken, used_families = [], set()
        for r in block.itertuples():
            if r.family in used_families:
                continue
            taken.append(r)
            used_families.add(r.family)
            if len(taken) == count:
                break
        for r in taken:
            original = case_map[r.row_id]
            selected.append({"row_id": r.row_id, "family": r.family, "quadrant": r.quadrant, "label": int(r.label), "error": kind,
                             "score": float(r.score), "messages_json": original["messages_json"], "completion": original["completion"]})
    write_json(root / "six_case_audit.json", selected)
    layer_info = []
    for l in range(D.X.shape[2]):
        p = sigmoid(z[:, 0, l])
        w = th["W"][0, l]
        top = np.argsort(np.abs(w))[-12:][::-1]
        layer_info.append({"layer_position": l, "weight_norm": float(np.linalg.norm(w)), "auroc": float(roc_auc_score(D.y, p)),
                           "mean_prob_by_quadrant": {q: float(p[D.meta.quadrant == q].mean()) for q in sorted(D.meta.quadrant.unique())},
                           "top_weights": [[int(j), float(w[j])] for j in top]})
    write_json(root / "weight_inspection.json", {"layers": layer_info, "weight_cosines": (th["W"][0] @ th["W"][0].T / np.outer(np.linalg.norm(th["W"][0], axis=1), np.linalg.norm(th["W"][0], axis=1))).tolist()})
    print("DEVELOPMENT_AUDIT", json.dumps({"metrics": metrics(D.meta, s, cutoff), "errors": table.error.value_counts().to_dict(), "layers": layer_info}), flush=True)
    print("FULL_CONTEXT_AUDIT", json.dumps(selected, ensure_ascii=False), flush=True)


def prepare(a):
    root = Path(a.run)
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "theta0.npz").exists(), "refuse to overwrite starting checkpoint"
    D = Data(a.capture, ["fit"])
    shape = D.X.shape[1:]
    mu, sd = D.X.mean(0), np.maximum(D.X.std(0), 1e-8)
    th = {"mu": np.repeat(mu[0:1], 3, axis=0), "sd": np.repeat(sd[0:1], 3, axis=0),
          "W": np.zeros(shape), "b": np.zeros(shape[:2]), "a": np.zeros(shape[:2])}
    th["a"][0] = 1 / shape[1]
    th, replay = fit(th, D, [(0, l) for l in range(shape[1])])
    for p in [1, 2]:
        th["W"][p], th["b"][p] = th["W"][0], th["b"][0]
    validate(th)
    save_theta(root / "theta0.npz", th)
    np.savez(root / "unlabelled_fit_stats.npz", mu=mu, sd=sd)
    # Exact affine transport negative control: changing coordinates must preserve logits.
    identity = copy.deepcopy(th)
    identity["W"] = th["W"] * sd / th["sd"]
    identity["b"] = th["b"] + np.sum(th["W"] * (mu - th["mu"]) / th["sd"], axis=-1)
    identity["mu"], identity["sd"] = mu, sd
    identity_error = float(np.max(np.abs(node_logits(identity, D.X) - node_logits(th, D.X))))
    assert identity_error < 1e-6
    manifest = {"seed": SEED, "created_unix": time.time(), "capture_manifest_sha256": file_hash(Path(a.capture) / "capture_manifest.json"),
                "theta0_sha256": file_hash(root / "theta0.npz"), "sklearn_replay_max_abs_error": replay,
                "affine_identity_max_abs_logit_error": identity_error, "attempt_limit": 10, "completed_attempts": 0,
                "prediction_accounting_note": "score calls invoke node-logit calls; counts are nested, not additive. Internal sklearn optimizer evaluations are not enumerated.",
                "executing_agent": "Codex in this conversation; no separate paid Claude API", "python": platform.python_version()}
    write_json(root / "manifest.json", manifest)
    audit(Data(a.capture, ["inspect"]), th, root, a.capture)
    manifest["prepare_prediction_counts"] = COUNTS.copy()
    write_json(root / "manifest.json", manifest)
    print("PREPARED", json.dumps(manifest), flush=True)


def edit_candidate(base, stats, spec, D):
    th = copy.deepcopy(base)
    th["a"][:] = 0
    for branch in spec["branches"]:
        p = POOLS.index(branch["pool"])
        layers = branch.get("layers", list(range(base["W"].shape[1])))
        weights = np.asarray(branch.get("weights", np.ones(len(layers))), dtype=float)
        weights = weights / weights.sum() * branch.get("mix", 1.0)
        th["a"][p, layers] += weights
        if branch.get("normalization") == "pool_fit":
            th["mu"][p], th["sd"][p] = stats["mu"][p], stats["sd"][p]
    for op in spec.get("operations", []):
        if op["kind"] == "clip_weight_quantile":
            for p, l in np.argwhere(th["a"] > 0):
                cap = np.quantile(np.abs(th["W"][p, l]), op["quantile"])
                th["W"][p, l] = np.clip(th["W"][p, l], -cap, cap)
        elif op["kind"] == "scale_logits":
            th["W"] *= op["factor"]
            th["b"] *= op["factor"]
        elif op["kind"] == "remove_length_direction":
            length = np.log1p(D.meta.answer_tokens.to_numpy(dtype=float))
            length -= length.mean()
            for p, l in np.argwhere(th["a"] > 0):
                X = (D.X[:, p, l] - th["mu"][p, l]) / th["sd"][p, l]
                direction = (X - X.mean(0)).T @ length
                direction /= max(np.linalg.norm(direction), 1e-12)
                th["W"][p, l] -= op["strength"] * np.dot(th["W"][p, l], direction) * direction
        else:
            raise ValueError("unsupported edit; add reviewed explicit code, not an opaque optimizer")
    validate(th)
    return th


def active_count(th):
    return int(np.count_nonzero(th["a"])) * (th["W"].shape[-1] + 1)


def choose(records):
    best = records[0]
    for r in records[1:]:
        delta = r["metrics"]["selection_score"] - best["metrics"]["selection_score"]
        if delta > 0.005 or (abs(delta) <= 0.005 and (r["metrics"]["macro_auroc"], -r["capacity"]) > (best["metrics"]["macro_auroc"], -best["capacity"])):
            best = r
    return best


def attempt(a):
    root = Path(a.run)
    assert not (root / "FINAL_DONE.json").exists(), "test was already opened"
    manifest = json.loads((root / "manifest.json").read_text())
    n = manifest["completed_attempts"] + 1
    assert n <= 10
    spec = json.loads(Path(a.spec).read_text())
    assert spec["attempt"] == n and all(spec.get(k) for k in ["hypothesis", "falsifying_prediction", "rationale", "category", "branches"])
    folder = root / f"attempt_{n:02d}"
    folder.mkdir(exist_ok=False)
    # Hypothesis is written durably before this function accesses selection scores.
    write_json(folder / "spec.json", spec)
    base = load_theta(root / "theta0.npz")
    with np.load(root / "unlabelled_fit_stats.npz", allow_pickle=False) as z:
        stats = {k: z[k] for k in z.files}
    inspect = Data(a.capture, ["inspect"])
    candidate = edit_candidate(base, stats, spec, inspect)
    save_theta(folder / "wsp.npz", candidate)
    train = Data(a.capture, ["fit", "inspect"])
    active = [tuple(x) for x in np.argwhere(candidate["a"] > 0)]
    ordinary_start = time.monotonic()
    ordinary, replay = fit(candidate, train, active)
    ordinary_seconds = time.monotonic() - ordinary_start
    save_theta(folder / "ordinary.npz", ordinary)
    # Random weight delta control on the identical structural readout and normalization.
    structural = copy.deepcopy(candidate)
    structural["W"], structural["b"] = base["W"].copy(), base["b"].copy()
    random = copy.deepcopy(structural)
    rng = np.random.default_rng(SEED + n)
    for k in ["W", "b"]:
        for p, l in active:
            delta = candidate[k][p, l] - structural[k][p, l]
            noise = rng.standard_normal(np.shape(delta))
            random[k][p, l] += noise * np.linalg.norm(delta) / max(np.linalg.norm(noise), 1e-12)
    save_theta(folder / "random.npz", random)
    selection = Data(a.capture, ["selection"])
    outcomes = {}
    for name, theta in [("wsp", candidate), ("ordinary", ordinary), ("random", random), ("structural", structural)]:
        # Only WSP and ordinary steer independent ten-candidate winner selection.
        s = score(theta, selection.X)
        sm = metrics(selection.meta, s)
        ds = score(theta, inspect.X)
        dm = metrics(inspect.meta, ds)
        outcomes[name] = {"id": f"attempt_{n:02d}/{name}", "metrics": sm, "development_metrics": dm, "capacity": active_count(theta), "checkpoint_sha256": file_hash(folder / f"{name}.npz") if name != "structural" else None}
        table = selection.meta.copy()
        table["score"] = s
        table.to_csv(folder / f"selection_{name}.csv", index=False)
        table = inspect.meta.copy()
        table["score"] = ds
        table.to_csv(folder / f"development_{name}.csv", index=False)
    write_json(folder / "feedback.json", outcomes)
    records = []
    initial_s = score(base, selection.X)
    initial = {"id": "theta0", "metrics": metrics(selection.meta, initial_s), "capacity": active_count(base)}
    for arm in ["wsp", "ordinary"]:
        options = [initial] + [json.loads(p.read_text())[arm] for p in sorted(root.glob("attempt_*/feedback.json"))]
        records.append({"arm": arm, "winner": choose(options)["id"]})
    write_json(folder / "keep_revert.json", {"winners_after_attempt": records, "rule": "selection score then macro-AUROC then active capacity; threshold 0.005", "ordinary_replay_error": replay,
                                              "ordinary_fit_seconds": ordinary_seconds, "prediction_counts": COUNTS.copy()})
    manifest["completed_attempts"] = n
    manifest["last_attempt"] = outcomes
    write_json(root / "manifest.json", manifest)
    print("ATTEMPT_RESULT", json.dumps({"attempt": n, "hypothesis": spec["hypothesis"], "wsp": outcomes["wsp"],
                                        "wsp_winner": records[0], "control_feedback": "saved; withheld until all ten candidates are frozen"}), flush=True)


def freeze(a):
    root = Path(a.run)
    assert not (root / "FROZEN.json").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["completed_attempts"] == 10, "exactly ten real attempts required"
    selection = Data(a.capture, ["selection"])
    base = load_theta(root / "theta0.npz")
    initial = {"id": "theta0", "metrics": metrics(selection.meta, score(base, selection.X)), "capacity": active_count(base)}
    winners = {"original": "theta0.npz"}
    for arm in ["wsp", "ordinary"]:
        options = [initial] + [json.loads(p.read_text())[arm] for p in sorted(root.glob("attempt_*/feedback.json"))]
        picked = choose(options)["id"]
        winners[arm] = "theta0.npz" if picked == "theta0" else picked + ".npz"
    if winners["wsp"] != "theta0.npz":
        winners["random_control"] = str(Path(winners["wsp"]).parent / "random.npz")
    artifact = {"created_unix": time.time(), "winners": winners, "hashes": {k: file_hash(root / v) for k, v in winners.items()}, "selection_queries_per_arm": 10}
    write_json(root / "FROZEN.json", artifact)
    print("FROZEN", json.dumps(artifact), flush=True)


def sample_group_indices(meta, rng):
    codes, groups = pd.factorize(meta.group_id, sort=True)
    positions = [np.flatnonzero(codes == i) for i in range(len(groups))]
    return np.concatenate([positions[i] for i in rng.integers(0, len(groups), len(groups))])


def final(a):
    root = Path(a.run)
    assert not (root / "FINAL_DONE.json").exists(), "final cannot be repeated after inspection"
    frozen = json.loads((root / "FROZEN.json").read_text())
    for k, v in frozen["winners"].items():
        assert file_hash(root / v) == frozen["hashes"][k]
    # Mark the opening before any scores: an interrupted final remains disclosed.
    write_json(root / "FINAL_OPENED.json", {"opened_unix": time.time()})
    test, calib = Data(a.capture, ["test"]), Data(a.capture, ["calib"])
    assert (calib.y == 0).all()
    assert not set(test.meta.group_id) & set(calib.meta.group_id)
    names = list(frozen["winners"])
    scores, cal_scores, rows, cuts = {}, {}, {}, {}
    for name in names:
        th = load_theta(root / frozen["winners"][name])
        s, c = score(th, test.X), score(th, calib.X)
        scores[name], cal_scores[name] = s, c
        cut = threshold(c, 0.05)
        assert np.isfinite(cut), "not enough calibration negatives"
        cuts[name] = cut
        rows[name] = metrics(test.meta, s, cut)
        rows[name]["target_fpr_achieved"] = rows[name]["fpr"] <= 0.05
        rows[name]["at_1pct"] = metrics(test.meta, s, threshold(c, 0.01))
        for part, data, vector in [("test", test, s), ("calib", calib, c)]:
            table = data.meta.copy()
            table["score"], table["threshold_5pct"] = vector, cut
            table["prediction_5pct"] = (vector > cut).astype(int)
            table.to_csv(root / f"final_{part}_{name}.csv", index=False)
    write_json(root / "thresholds.json", cuts)
    rng = np.random.default_rng(SEED)
    draws = {name: {"recall": [], "fpr": []} for name in names}
    # Paired across arms; resample calibration too, because threshold estimation is uncertain.
    for _ in range(2000):
        ti, ci = sample_group_indices(test.meta, rng), sample_group_indices(calib.meta, rng)
        yy = test.y[ti]
        for name in names:
            cutoff = threshold(cal_scores[name][ci], 0.05)
            alarm = scores[name][ti] > cutoff
            draws[name]["recall"].append(float(alarm[yy == 1].mean()))
            draws[name]["fpr"].append(float(alarm[yy == 0].mean()))
    intervals = {}
    for name in names:
        for endpoint in ["recall", "fpr"]:
            intervals[name + ":" + endpoint] = np.quantile(draws[name][endpoint], [0.025, 0.975]).tolist()
    for other in ["original", "ordinary"]:
        for endpoint in ["recall", "fpr"]:
            delta = np.asarray(draws["wsp"][endpoint]) - np.asarray(draws[other][endpoint])
            intervals["wsp-minus-" + other + ":" + endpoint] = np.quantile(delta, [0.025, 0.975]).tolist()
    result = {"rows": rows, "intervals": intervals, "bootstrap": "2000 paired group draws, re-calibrating in each draw; conditional on these familiar target families",
              "historical_independence": "not established", "test_rows": len(test.y), "calibration_rows": len(calib.y)}
    write_json(root / "FINAL_DONE.json", result)
    print("FINAL_RESULT", json.dumps(result), flush=True)


def replay(a):
    root = Path(a.run)
    saved = json.loads((root / "FINAL_DONE.json").read_text())
    for name, expected in saved["rows"].items():
        table = pd.read_csv(root / f"final_test_{name}.csv")
        found = metrics(table, table.score.to_numpy(), table.threshold_5pct.iloc[0])
        for key in ["tp", "fp", "tn", "fn", "recall", "fpr", "macro_auroc"]:
            assert abs(found[key] - expected[key]) < 1e-12, (name, key)
    print("REPLAY_OK", len(saved["rows"]), "arms from saved per-example scores", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "attempt", "freeze", "final", "replay"])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--spec")
    args = ap.parse_args()
    globals()[args.phase](args)
