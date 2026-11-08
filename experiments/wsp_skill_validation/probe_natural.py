"""Execute the saved source-offset hypothesis and its falsification controls."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def main():
    ap = argparse.ArgumentParser()
    for name in ["capture", "baseline", "hypothesis", "audit-script", "eval-script", "out"]:
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    root, out = Path(args.capture), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    assert (root / "CAPTURE_DONE.json").exists()
    hypothesis = json.loads(Path(args.hypothesis).read_text())
    (out / "HYPOTHESIS.json").write_bytes(Path(args.hypothesis).read_bytes())
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    audit, ev = module("audit", args.audit_script), module("ev", args.eval_script)
    t = audit.load(args.baseline)
    assert hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest() == "3d9dfe5dc4142bf876f800fa710a5e2c0f65afcb9977d34255b2bd328e439a89"
    meta = pd.read_csv(root / "meta.csv").fillna("")
    is_transfer = meta.group_id.map(lambda g: int(hashlib.sha256(("diagnostic-transfer:" + g).encode()).hexdigest()[:12], 16) % 3 == 0)
    role_indices = {"discovery": np.flatnonzero(((meta.partition == "diagnostic") & ~is_transfer).to_numpy()),
                    "transfer": np.flatnonzero(((meta.partition == "diagnostic") & is_transfer).to_numpy())}
    def get_role(role):
        ii = role_indices[role]
        # Slice before scoring; other roles are never passed to this computation.
        return meta.iloc[ii].reset_index(drop=True), np.load(root / "acts_last.npy", mmap_mode="r")[ii].astype(np.float64)
    md, xd = get_role("discovery")
    qd = (xd - t["mu"]) / t["sd"]
    family_means = {}
    for f in sorted(set(md.family)):
        family_means[f] = np.mean([qd[(md.family == f).to_numpy() & (md.label == y).to_numpy()].mean(0)
                                   for y in [0, 1]], axis=0)
    game_families = {"among", "werewolf"}
    contrast = np.mean([v for f, v in family_means.items() if f in game_families], axis=0) - np.mean([v for f, v in family_means.items() if f not in game_families], axis=0)
    u = contrast / np.linalg.norm(contrast, axis=1, keepdims=True)
    np.savez_compressed(out / "direction.npz", u=u, contrast=contrast)
    def patch(alpha, probe=None):
        c = copy.deepcopy(t)
        for l in range(len(t["W"])):
            if probe is None or l == probe:
                c["W"][l] -= alpha * u[l] * (u[l] @ t["W"][l])
        return c
    variants = {"original": t, "primary_a1": patch(1), "dose_a05": patch(.5),
                "dose_a15": patch(1.5), "wrong_direction": patch(-1)}
    for l in range(len(t["W"])):
        variants[f"isolate_{l}"] = patch(1, l)
    rng = np.random.default_rng(202611082)
    null = copy.deepcopy(t)
    for l in range(len(t["W"])):
        d = variants["primary_a1"]["W"][l] - t["W"][l]
        noise = rng.normal(size=d.shape)
        noise -= u[l] * (noise @ u[l])
        null["W"][l] += noise / np.linalg.norm(noise) * np.linalg.norm(d)
    variants["orthogonal_random"] = null
    for name, theta in variants.items():
        np.savez_compressed(out / (name + ".npz"), **theta)
    # Candidate and all controls are now fixed before transfer scoring.
    (out / "CANDIDATES_FROZEN.json").write_text(json.dumps({"hypothesis_sha256": hashlib.sha256(Path(args.hypothesis).read_bytes()).hexdigest(),
        "variants": {name: hashlib.sha256((out / (name + ".npz")).read_bytes()).hexdigest() for name in variants}}, indent=2))
    mt, xt = get_role("transfer")
    scores = {}
    ledger = []
    for role, m, x in [("discovery", md, xd), ("transfer", mt, xt)]:
        for name, theta in variants.items():
            z = np.einsum("nld,ld->nl", (x - theta["mu"]) / theta["sd"], theta["W"]) + theta["b"]
            s = expit(z).mean(1)
            scores[role, name] = s
            m.assign(score=s).to_csv(out / f"scores_{role}_{name}.csv", index=False)
            ledger.append({"role": role, "variant": name, "rows": len(m)})
    thresholds = {name: ev.cutoff(scores["discovery", name][md.label == 0], .025) for name in variants}
    results = {}
    game = mt.family.isin(game_families).to_numpy()
    labels = mt.label.to_numpy()
    for name in variants:
        s = scores["transfer", name]
        results[name] = {"overall": ev.counts(labels, s, thresholds[name]),
                         "game": ev.counts(labels[game], s[game], thresholds[name]),
                         "nongame": ev.counts(labels[~game], s[~game], thresholds[name]),
                         "threshold": thresholds[name],
                         "game_mean_score_by_label": {str(y): float(s[game & (labels == y)].mean()) for y in [0, 1]}}
    original, primary = results["original"], results["primary_a1"]
    p = hypothesis["predictions"]
    checks = {"game_score_each_label": all(primary["game_mean_score_by_label"][str(y)] - original["game_mean_score_by_label"][str(y)] >= p["transfer_game_mean_score_gain_each_label_min"] for y in [0, 1]),
              "game_recall": primary["game"]["recall"] - original["game"]["recall"] >= p["transfer_game_recall_gain_min"],
              "nongame_preservation": original["nongame"]["recall"] - primary["nongame"]["recall"] <= p["transfer_nongame_recall_loss_max"],
              "false_alarm_preservation": primary["overall"]["fpr"] - original["overall"]["fpr"] <= p["transfer_overall_fpr_increase_max"]}
    form = audit.inspect(t, variants["primary_a1"])
    checks["edit_form"] = form["allowed"]
    result = {"scope": "diagnostic transfer only; observational source-direction hypothesis, not semantic causal proof",
              "discovery_rows": len(md), "transfer_rows": len(mt), "score_batches": ledger,
              "direction_dot_original_weights": np.einsum("ld,ld->l", u, t["W"]).tolist(),
              "variants": results, "primary_form_audit": form, "checks": checks,
              "qualifies_for_validation": bool(all(checks.values())),
              "decision": "evaluate_primary_once_on_validation" if all(checks.values()) else "reject_mechanism_keep_original"}
    (out / "PROBE_RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
