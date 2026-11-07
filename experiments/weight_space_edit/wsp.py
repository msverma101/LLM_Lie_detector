"""Weight-space programming (WSP) arm: Claude edits the trained probe's weights directly.

    code-search arm (research_loop/): theta_new = Train(Claude's new probe.py, D)
    this arm:                          theta_new = theta_old + delta_theta_Claude   (no gradient step on Claude's side)

Phases (run in order, each writes into --run):

  prepare  freeze a family-disjoint partition manifest, train the LinearEnsemble probe ONCE on
           dev_fit, export its weights as an editable checkpoint theta0.npz, check numerical replay
  gd       ordinary optimization of the SAME parameters (gradient fine-tuning from theta0 on dev,
           early stopping on selection, same evaluation budget) + a full refit on all dev data
  claude   WSP arm: Claude Opus 5.5 inspects theta0 and dev errors, proposes direct weight edits via
           tools, gets only aggregate selection scores back (bounded budget)
  random   control: random edits with the same per-block size as Claude's chosen edit
  final    freeze all arms, set thresholds on the separate CALIBRATION honest answers, unseal TEST once

Partitions (by FIBS family, so no paired answers / paraphrases leak across):
  dev_fit     10 of the 14 research_loop train families  -> theta0 is fit here
  dev_inspect  4 train families                          -> Claude's error analysis, GD fine-tuning
  selection    4 of the 7 research_loop val families     -> aggregate scores for candidate choice
  calib        3 val families, honest answers only       -> alarm threshold, used only in `final`
  test         the 7 research_loop test families         -> sealed; only `final` loads them

Probe (LinearEnsemble, parameters theta = {mu, sd, W, b, a}):
  z_l(x) = ((x_l - mu_l) / sd_l) . W_l + b_l          per layer l
  s(x)   = sum_l a_l * sigmoid(z_l(x))                 a_l = 1/L initially
Claude may edit W, b, a. mu/sd (scaler statistics) stay fixed in this version.

Selection rule (fixed before the search): maximize selection TPR at 5% FPR (threshold from the
selection honest answers); ties within 0.005 -> higher selection macro-AUROC -> smaller edit.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "research_loop"))
from eval import POOLS, TEST_FAMILIES, VAL_FAMILIES, load  # noqa: E402

POOL = "last"
C = 0.01
SEED = 0
REPLAY_TOL = 1e-6          # max |sklearn score - numpy score| on dev_fit, fixed before any search
EVAL_BUDGET = 20           # evaluate() calls per arm on selection
TIE = 0.005
MODEL = "claude-opus-5-5"


# ----------------------------------------------------------------------------- data
def partitions(meta, seed=SEED):
    fam = meta.family.values
    fams = sorted(set(fam))
    train_f = [f for f in fams if f not in VAL_FAMILIES and f not in TEST_FAMILIES]
    rng = np.random.RandomState(seed)
    tr = list(rng.permutation(train_f))
    va = list(rng.permutation(sorted(f for f in VAL_FAMILIES if f in fams)))
    n_insp = max(1, round(len(tr) * 4 / 14))
    n_cal = max(1, round(len(va) * 3 / 7))
    return {
        "dev_fit": sorted(tr[n_insp:]), "dev_inspect": sorted(tr[:n_insp]),
        "selection": sorted(va[n_cal:]), "calib": sorted(va[:n_cal]),
        "test": sorted(f for f in TEST_FAMILIES if f in fams),
    }


class Data:
    """Loads only the partitions asked for. The test partition is loaded only by `final`."""

    def __init__(self, acts, parts, names):
        self.meta, self.info, arrays = load(acts)
        self.parts = {}
        fam = self.meta.family.values
        for n in names:
            idx = np.where(np.isin(fam, parts[n]))[0]
            if n == "calib":  # calibration = honest answers only
                idx = idx[self.meta.label.values[idx] == 0]
            X = np.nan_to_num(np.asarray(arrays[POOL][idx], dtype=np.float32))
            self.parts[n] = dict(idx=idx, X=X, y=self.meta.label.values[idx].astype(int), fam=fam[idx])

    def __getitem__(self, n):
        return self.parts[n]


def idx_hash(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- probe
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def layer_logits(theta, X):
    Xs = (X - theta["mu"][None]) / theta["sd"][None]
    return np.einsum("nld,ld->nl", Xs, theta["W"]) + theta["b"][None]


def score(theta, X):
    return sigmoid(layer_logits(theta, X)) @ theta["a"]


def fit_theta0(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    L, d = X.shape[1], X.shape[2]
    th = {k: np.zeros((L, d)) for k in ("mu", "sd", "W")}
    th["b"], th["a"] = np.zeros(L), np.full(L, 1.0 / L)
    sk = []
    for j in range(L):
        sc = StandardScaler().fit(X[:, j])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[:, j]), y)
        th["mu"][j], th["sd"][j] = sc.mean_, sc.scale_
        th["W"][j], th["b"][j] = clf.coef_[0], clf.intercept_[0]
        sk.append((sc, clf))
    ref = np.mean([c.predict_proba(s.transform(X[:, j]))[:, 1] for j, (s, c) in enumerate(sk)], axis=0)
    return th, ref


def save_theta(path, th):
    np.savez(path, **th)


def load_theta(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def theta_hash(th):
    h = hashlib.sha256()
    for k in sorted(th):
        h.update(np.ascontiguousarray(th[k], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def edit_size(th, th0):
    out = {}
    for k in ("W", "b", "a"):
        out[k] = float(np.linalg.norm(th[k] - th0[k]) / (np.linalg.norm(th0[k]) + 1e-12))
    out["total"] = out["W"] + out["b"] + out["a"]
    return out


# ----------------------------------------------------------------------------- metrics
def macro_auroc(s, y, fam):
    per = {f: float(roc_auc_score(y[fam == f], s[fam == f])) for f in sorted(set(fam))
           if len(set(y[fam == f])) == 2}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def thr_at(s_honest, fpr):
    return float(np.quantile(s_honest, 1.0 - fpr, method="higher"))


def tpr_fpr(s, y, thr):
    f = s > thr
    return float(f[y == 1].mean()), float(f[y == 0].mean()) if (y == 0).any() else float("nan")


def selection_metrics(th, part):
    s = score(th, part["X"])
    y = part["y"]
    au, per = macro_auroc(s, y, part["fam"])
    out = {"sel_macro_auroc": round(au, 4)}
    for t in (0.01, 0.05):
        tpr, _ = tpr_fpr(s, y, thr_at(s[y == 0], t))
        out[f"sel_tpr_at_{int(t * 100)}fpr"] = round(tpr, 4)
    out["sel_per_family_auroc"] = {k: round(v, 4) for k, v in per.items()}
    return out


def better(a, b):
    """Pre-registered selection rule. a, b = (metrics, edit_total). True if a beats b."""
    (ma, ea), (mb, eb) = a, b
    da = ma["sel_tpr_at_5fpr"] - mb["sel_tpr_at_5fpr"]
    if abs(da) > TIE:
        return da > 0
    if abs(ma["sel_macro_auroc"] - mb["sel_macro_auroc"]) > 1e-4:
        return ma["sel_macro_auroc"] > mb["sel_macro_auroc"]
    return ea < eb


# ----------------------------------------------------------------------------- phases
def cmd_prepare(a):
    os.makedirs(a.run, exist_ok=True)
    meta, info, _ = load(a.acts)
    parts = partitions(meta)
    D = Data(a.acts, parts, ["dev_fit", "dev_inspect", "selection", "calib"])
    for n in ("dev_fit", "dev_inspect", "selection"):
        assert len(set(D[n]["y"])) == 2, f"{n} needs both classes"
    th0, ref = fit_theta0(D["dev_fit"]["X"], D["dev_fit"]["y"])
    replay = float(np.max(np.abs(score(th0, D["dev_fit"]["X"]) - ref)))
    assert replay < REPLAY_TOL, f"numpy replay differs from sklearn by {replay}"
    save_theta(os.path.join(a.run, "theta0.npz"), th0)
    fam = meta.family.values
    man = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "activation_root": os.path.abspath(a.acts), "info": info,
        "source_model": info.get("model"), "pool": POOL, "probe": f"LinearEnsemble (per-layer logreg C={C}, mean of probs)",
        "probe_origin": "own LinearEnsemble trained here on dev_fit (no compatible published EFC checkpoint: the cache "
                        "keeps only pooled activations, EFC needs token-level activations/covariances)",
        "theta0_hash": theta_hash(th0), "replay_max_abs_diff": replay, "replay_tol": REPLAY_TOL,
        "seed": SEED, "eval_budget_per_arm": EVAL_BUDGET, "selection_rule": __doc__.split("Selection rule")[1].strip(),
        "llm_model": MODEL, "partitions": {},
    }
    for n, fs in parts.items():
        idx = np.where(np.isin(fam, fs))[0]
        if n == "calib":
            idx = idx[meta.label.values[idx] == 0]
        man["partitions"][n] = {"families": fs, "n": int(len(idx)), "n_lie": int(meta.label.values[idx].sum()),
                                "row_hash": idx_hash(idx)}
    json.dump(man, open(os.path.join(a.run, "run_manifest.json"), "w"), indent=2)
    m0 = selection_metrics(th0, D["selection"])
    json.dump({"arm": "original", "theta": "theta0.npz", "selection": m0}, open(os.path.join(a.run, "arm_original.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in man.items() if k in ("partitions", "theta0_hash", "replay_max_abs_diff")}, indent=1))
    print("original selection:", m0)


def cmd_gd(a):
    import torch
    th0 = load_theta(os.path.join(a.run, "theta0.npz"))
    parts = json.load(open(os.path.join(a.run, "run_manifest.json")))
    parts = {k: v["families"] for k, v in parts["partitions"].items()}
    D = Data(a.acts, parts, ["dev_fit", "dev_inspect", "selection"])
    e0 = edit_size(th0, th0)["total"]
    best = (selection_metrics(th0, D["selection"]), e0, th0, 0)

    # (1) gradient fine-tuning of W, b, a from theta0 on all dev data, L2 pull toward theta0
    Xd = np.concatenate([D["dev_fit"]["X"], D["dev_inspect"]["X"]])
    yd = np.concatenate([D["dev_fit"]["y"], D["dev_inspect"]["y"]]).astype(np.float32)
    insp = np.r_[np.zeros(len(D["dev_fit"]["y"])), np.ones(len(D["dev_inspect"]["y"]))]
    wgt = np.where(insp == 1, 0.5 / insp.sum(), 0.5 / (1 - insp).sum()).astype(np.float32)  # half weight each
    Xs = torch.tensor(((Xd - th0["mu"][None]) / th0["sd"][None]).astype(np.float32))
    y_t, w_t = torch.tensor(yd), torch.tensor(wgt)
    P0 = {k: torch.tensor(th0[k], dtype=torch.float32) for k in ("W", "b", "a")}
    P = {k: v.clone().requires_grad_(True) for k, v in P0.items()}
    opt = torch.optim.Adam(P.values(), lr=a.lr)
    log = []
    steps_per_eval = a.steps // EVAL_BUDGET
    for e in range(EVAL_BUDGET):
        for _ in range(steps_per_eval):
            z = torch.einsum("nld,ld->nl", Xs, P["W"]) + P["b"]
            s = (torch.sigmoid(z) * P["a"]).sum(1).clamp(1e-6, 1 - 1e-6)
            loss = -(w_t * (y_t * torch.log(s) + (1 - y_t) * torch.log(1 - s))).sum()
            loss = loss + a.l2 * sum(((P[k] - P0[k]) ** 2).sum() for k in P)
            opt.zero_grad()
            loss.backward()
            opt.step()
        th = dict(th0, **{k: P[k].detach().double().numpy() for k in P})
        m = selection_metrics(th, D["selection"])
        es = edit_size(th, th0)["total"]
        log.append({"eval": e + 1, "step": (e + 1) * steps_per_eval, "loss": float(loss), **{k: v for k, v in m.items() if k != "sel_per_family_auroc"}})
        print(log[-1], flush=True)
        if better((m, es), best[:2]):
            best = (m, es, th, e + 1)
    save_theta(os.path.join(a.run, "theta_gd.npz"), best[2])
    json.dump({"arm": "gd_finetune", "theta": "theta_gd.npz", "selection": best[0], "chosen_eval": best[3],
               "edit_size": edit_size(best[2], th0), "lr": a.lr, "l2": a.l2, "steps": a.steps, "log": log},
              open(os.path.join(a.run, "arm_gd_finetune.json"), "w"), indent=2)

    # (2) plain refit of the same architecture on all dev data (1 evaluation)
    th_r, _ = fit_theta0(Xd, yd.astype(int))
    th_r["mu"], th_r["sd"] = th_r["mu"], th_r["sd"]  # refit uses its own scaler, still the same parameter family
    save_theta(os.path.join(a.run, "theta_refit.npz"), th_r)
    json.dump({"arm": "refit_all_dev", "theta": "theta_refit.npz", "selection": selection_metrics(th_r, D["selection"])},
              open(os.path.join(a.run, "arm_refit_all_dev.json"), "w"), indent=2)
    print("gd best:", best[0], "\nrefit:", selection_metrics(th_r, D["selection"]))


# ----------------------------------------------------------------------------- the WSP lab
class Lab:
    def __init__(self, run, D, th0, budget):
        self.run, self.D, self.th0, self.budget = run, D, th0, budget
        self.cands = {"c0": {"theta": th0, "parent": None, "ops": [], "rationale": "original probe theta0"}}
        self.evals, self.n = [], 0
        os.makedirs(os.path.join(run, "candidates"), exist_ok=True)
        meta = D.meta
        self.text = {n: meta.answer.fillna("").values[D[n]["idx"]] if "answer" in meta else None for n in ("dev_inspect",)}
        self.quad = meta.quadrant.values[D["dev_inspect"]["idx"]] if "quadrant" in meta else None
        self.src = meta.source.values[D["dev_inspect"]["idx"]] if "source" in meta else None
        ins = D["dev_inspect"]
        self.Zs = (ins["X"] - th0["mu"][None]) / th0["sd"][None]  # standardized dev_inspect activations

    # -- read-only tools ------------------------------------------------------
    def overview(self, candidate="c0"):
        th = self.cands[candidate]["theta"]
        ins = self.D["dev_inspect"]
        Lg = layer_logits(th, ins["X"])
        W = th["W"]
        cos = (W @ W.T) / np.outer(np.linalg.norm(W, axis=1), np.linalg.norm(W, axis=1))
        layers = []
        for j, l in enumerate(self.D.info["layers"]):
            layers.append({"index": j, "block": l, "a": round(float(th["a"][j]), 4), "b": round(float(th["b"][j]), 4),
                           "W_norm": round(float(np.linalg.norm(W[j])), 4),
                           "dev_inspect_auroc": round(float(roc_auc_score(ins["y"], Lg[:, j])), 4),
                           "dev_inspect_mean_logit_honest": round(float(Lg[ins["y"] == 0, j].mean()), 3),
                           "dev_inspect_mean_logit_lie": round(float(Lg[ins["y"] == 1, j].mean()), 3)})
        s = score(th, ins["X"])
        au, per = macro_auroc(s, ins["y"], ins["fam"])
        return {"candidate": candidate, "shape_W": list(W.shape), "layers": layers,
                "cosine_between_layer_W": np.round(cos, 3).tolist(),
                "dev_inspect_macro_auroc": round(au, 4), "dev_inspect_per_family": {k: round(v, 4) for k, v in per.items()},
                "dev_inspect_families": sorted(set(ins["fam"])), "n_dev_inspect": int(len(ins["y"]))}

    def inspect_weights(self, layer, top_k=25, candidate="c0"):
        th = self.cands[candidate]["theta"]
        w = th["W"][layer]
        order = np.argsort(-np.abs(w))[:top_k]
        q = np.quantile(np.abs(w), [0.5, 0.9, 0.99, 1.0])
        frac = float((np.cumsum(np.sort(w ** 2)[::-1]) / (w ** 2).sum() < 0.5).sum() + 1)
        return {"layer": layer, "bias": float(th["b"][layer]), "a": float(th["a"][layer]),
                "abs_quantiles_50_90_99_max": np.round(q, 5).tolist(),
                "n_dims_holding_half_of_squared_norm": int(frac),
                "top_dims": [{"dim": int(i), "w": round(float(w[i]), 5)} for i in order]}

    def feature_stats(self, layer, dims):
        ins = self.D["dev_inspect"]
        Z = self.Zs[:, layer, :][:, dims]
        out = []
        for k, d in enumerate(dims):
            z = Z[:, k]
            per = {}
            for f in sorted(set(ins["fam"])):
                m = ins["fam"] == f
                if len(set(ins["y"][m])) == 2:
                    per[f] = round(float(z[m & (ins["y"] == 1)].mean() - z[m & (ins["y"] == 0)].mean()), 3)
            out.append({"dim": int(d), "w": round(float(self.th0["W"][layer, d]), 5),
                        "mean_diff_lie_minus_honest_std_units": round(float(z[ins["y"] == 1].mean() - z[ins["y"] == 0].mean()), 3),
                        "per_family_diff": per})
        return {"layer": layer, "partition": "dev_inspect", "dims": out}

    def inspect_errors(self, n=12, candidate="c0"):
        th = self.cands[candidate]["theta"]
        ins = self.D["dev_inspect"]
        s = score(th, ins["X"])
        y = ins["y"]
        thr = thr_at(s[y == 0], 0.05)
        Lg = layer_logits(th, ins["X"])
        miss = np.where((y == 1) & (s <= thr))[0]
        fa = np.where((y == 0) & (s > thr))[0]
        rng = np.random.RandomState(0)

        def row(i):
            r = {"family": str(ins["fam"][i]), "score": round(float(s[i]), 3),
                 "layer_logits": np.round(Lg[i], 2).tolist()}
            if self.quad is not None:
                r["quadrant"], r["source"] = str(self.quad[i]), str(self.src[i])
            if self.text["dev_inspect"] is not None:
                r["answer_snippet"] = str(self.text["dev_inspect"][i])[:160]
            return r

        def by_fam(ix):
            return {f: int((ins["fam"][ix] == f).sum()) for f in sorted(set(ins["fam"]))}

        return {"partition": "dev_inspect", "threshold_rule": "5% FPR on dev_inspect honest",
                "tpr": round(float((s[y == 1] > thr).mean()), 4), "n_missed": int(len(miss)), "n_false_alarms": int(len(fa)),
                "missed_by_family": by_fam(miss), "false_alarms_by_family": by_fam(fa),
                "missed_examples": [row(i) for i in rng.permutation(miss)[:n]],
                "false_alarm_examples": [row(i) for i in rng.permutation(fa)[:n]]}

    # -- edits ----------------------------------------------------------------
    def apply_edit(self, parent, ops, rationale):
        th = {k: v.copy() for k, v in self.cands[parent]["theta"].items()}
        L, d = th["W"].shape
        for op in ops:
            t = op["op"]
            if t == "set_layer_weights":
                v = np.asarray(op["values"], float)
                assert v.shape == (L,) and (v >= 0).all(), "a must be L non-negative values"
                th["a"] = v / v.sum()
            elif t == "scale_layer":
                th["W"][op["layer"]] *= op["factor"]
                th["b"][op["layer"]] *= op.get("bias_factor", 1.0)
            elif t == "set_bias":
                th["b"][op["layer"]] = op["value"]
            elif t in ("set_dims", "add_dims", "scale_dims"):
                dims = np.asarray(op["dims"], int)
                v = np.asarray(op["values"], float) if "values" in op else float(op["factor"])
                if t == "set_dims":
                    th["W"][op["layer"], dims] = v
                elif t == "add_dims":
                    th["W"][op["layer"], dims] += v
                else:
                    th["W"][op["layer"], dims] *= v
            elif t == "keep_top_k":
                w = th["W"][op["layer"]]
                cut = np.argsort(-np.abs(w))[op["k"]:]
                w[cut] *= op.get("shrink_rest", 0.0)
            elif t == "blend_layers":  # W[target] <- (1-alpha) W[target] + alpha W[source]
                al = op["alpha"]
                th["W"][op["target"]] = (1 - al) * th["W"][op["target"]] + al * th["W"][op["source"]]
            elif t == "add_dev_mean_diff":  # add alpha * (lie - honest) mean difference on dev_inspect, unit norm, scaled to |W|
                ins = self.D["dev_inspect"]
                Z = self.Zs[:, op["layer"], :]
                if "family" in op:
                    m = ins["fam"] == op["family"]
                    dv = Z[m & (ins["y"] == 1)].mean(0) - Z[m & (ins["y"] == 0)].mean(0)
                else:
                    dv = Z[ins["y"] == 1].mean(0) - Z[ins["y"] == 0].mean(0)
                dv = dv / (np.linalg.norm(dv) + 1e-12) * np.linalg.norm(th["W"][op["layer"]])
                th["W"][op["layer"]] += op["alpha"] * dv
            else:
                raise ValueError(f"unknown op {t}")
        assert all(np.isfinite(v).all() for v in th.values())
        self.n += 1
        cid = f"c{self.n}"
        self.cands[cid] = {"theta": th, "parent": parent, "ops": ops, "rationale": rationale}
        es = edit_size(th, self.th0)
        save_theta(os.path.join(self.run, "candidates", f"{cid}.npz"), th)
        json.dump({"id": cid, "parent": parent, "ops": ops, "rationale": rationale, "edit_size": es, "time": time.time()},
                  open(os.path.join(self.run, "candidates", f"{cid}.json"), "w"), indent=1)
        return {"candidate": cid, "edit_size_relative_to_theta0": {k: round(v, 4) for k, v in es.items()}}

    def evaluate(self, candidate):
        if len(self.evals) >= self.budget:
            return {"error": f"evaluation budget of {self.budget} used up; call finish"}
        th = self.cands[candidate]["theta"]
        m = selection_metrics(th, self.D["selection"])
        es = edit_size(th, self.th0)["total"]
        self.evals.append({"candidate": candidate, "edit_total": es, **m})
        js = os.path.join(self.run, "candidates", f"{candidate}.json")
        if os.path.exists(js):
            rec = json.load(open(js))
            rec.setdefault("selection", []).append(m)
            json.dump(rec, open(js, "w"), indent=1)
        return {**{k: v for k, v in m.items()}, "evaluations_left": self.budget - len(self.evals)}

    def choose(self):
        # c0 (unedited) always competes, free of budget: an edit must beat the original to be chosen
        m0 = selection_metrics(self.th0, self.D["selection"])
        best = (((m0, 0.0)), "c0")
        for e in self.evals:
            cand = (e, e["edit_total"])
            if best is None or better(cand, best[0]):
                best = (cand, e["candidate"])
        return best[1] if best else "c0"


TOOLS = [
    {"name": "overview", "description": "Per-layer summary of a candidate probe: layer mixing weights a, biases, |W|, per-layer AUROC and mean logits on DEV_INSPECT, cosine similarity between layer weight vectors, per-family dev_inspect AUROC.",
     "input_schema": {"type": "object", "properties": {"candidate": {"type": "string"}}, "required": []}},
    {"name": "inspect_weights", "description": "Raw weights of one layer's logistic-regression vector W[layer] (standardized-input space): bias, mixing weight, |w| quantiles, concentration, and the top_k dimensions by |w| with values.",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "top_k": {"type": "integer"}, "candidate": {"type": "string"}}, "required": ["layer"]}},
    {"name": "feature_stats", "description": "For given dims of one layer: weight in theta0 and the mean standardized activation difference (lie minus honest) on DEV_INSPECT, overall and per family. Use it to test whether a dimension's weight makes sense.",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "dims": {"type": "array", "items": {"type": "integer"}}}, "required": ["layer", "dims"]}},
    {"name": "inspect_errors", "description": "Error analysis on DEV_INSPECT at a 5% FPR threshold: missed lies and false alarms by family, with example rows (family, quadrant, source, per-layer logits, answer snippet).",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}, "candidate": {"type": "string"}}, "required": []}},
    {"name": "apply_edit", "description": (
        "Create a new candidate = parent theta + your edit. Write the hypothesis in `rationale` BEFORE you evaluate. ops is a list, applied in order. Supported ops:\n"
        "{op:'set_layer_weights', values:[L non-negative floats]} (renormalized to sum 1)\n"
        "{op:'scale_layer', layer, factor, bias_factor?}\n{op:'set_bias', layer, value}\n"
        "{op:'set_dims'|'add_dims', layer, dims:[...], values:[...]}  {op:'scale_dims', layer, dims, factor}\n"
        "{op:'keep_top_k', layer, k, shrink_rest?}  (sparsify: other dims multiplied by shrink_rest, default 0)\n"
        "{op:'blend_layers', target, source, alpha}  (W[target] <- (1-alpha)W[target] + alpha W[source])\n"
        "{op:'add_dev_mean_diff', layer, alpha, family?}  (add alpha*|W| * unit lie-minus-honest mean direction from DEV_INSPECT, optionally one family)"),
     "input_schema": {"type": "object", "properties": {"parent": {"type": "string"}, "ops": {"type": "array", "items": {"type": "object"}}, "rationale": {"type": "string"}}, "required": ["parent", "ops", "rationale"]}},
    {"name": "evaluate", "description": "Aggregate scores of a candidate on the SELECTION families (macro-AUROC, TPR at 1%/5% FPR with the threshold set on selection honest answers, per-family AUROC). Counts against your evaluation budget. No per-example output.",
     "input_schema": {"type": "object", "properties": {"candidate": {"type": "string"}}, "required": ["candidate"]}},
    {"name": "finish", "description": "Stop the search. The final candidate is chosen by the fixed rule (best selection TPR@5%FPR, ties -> AUROC -> smaller edit) among evaluated candidates.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

SYSTEM = """You are a researcher doing weight-space programming on a trained lie-detection probe.

The detector reads cached residual-stream activations of Qwen3.5-2B at the last token of a final answer, from 6 layers. Per layer l: z_l = ((x_l - mu_l)/sd_l) . W_l + b_l, and the score is s = sum_l a_l * sigmoid(z_l). It was trained once (logistic regression per layer) on 10 FIBS deception families. Labels: 1 = the final answer is a knowing/contextual falsehood (FIBS definition).

Your job: improve the detector by EDITING ITS WEIGHTS DIRECTLY (W, b, a) with the tools, without retraining. Goal: catch more lies on unseen families at a controlled false-alarm rate, i.e. higher TPR at 5% FPR (macro-AUROC is secondary). The final threshold will later be set on separate honest calibration data you never see, and the result is measured once on sealed test families you never see. Generalizing to new kinds of lies matters more than fitting the selection families.

Data you can use: DEV_INSPECT (4 held-out training-side families) for error analysis and feature statistics; SELECTION (other families) only through `evaluate`, which returns aggregates and counts against a budget. Work like a scientist: inspect, form a hypothesis, write it in the rationale, make a targeted edit, evaluate, keep or go back to an earlier candidate (any candidate can be a parent). Prefer edits you can justify; small, interpretable edits are better than large ones at equal performance. When you are done or the budget is used, call `finish`."""


def cmd_claude(a):
    import anthropic
    th0 = load_theta(os.path.join(a.run, "theta0.npz"))
    man = json.load(open(os.path.join(a.run, "run_manifest.json")))
    parts = {k: v["families"] for k, v in man["partitions"].items()}
    D = Data(a.acts, parts, ["dev_inspect", "selection"])  # no calib, no test
    lab = Lab(a.run, D, th0, EVAL_BUDGET)
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": (
        f"Start. The original probe is candidate 'c0'. You have {EVAL_BUDGET} evaluate calls. "
        f"Layers (decoder blocks): {D.info['layers']}. W has shape {list(th0['W'].shape)}. "
        f"DEV_INSPECT families: {parts['dev_inspect']}. SELECTION families: {parts['selection']}.")}]
    transcript = open(os.path.join(a.run, "claude_transcript.md"), "w")
    usage = {"input": 0, "output": 0}
    finished = None
    for turn in range(a.max_turns):
        kw = dict(model=MODEL, max_tokens=32000, system=SYSTEM, tools=TOOLS, messages=msgs,
                  thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": a.effort})
        try:
            with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw) as s:
                msg = s.get_final_message()
        except TypeError:
            with client.messages.stream(**kw) as s:
                msg = s.get_final_message()
        usage["input"] += msg.usage.input_tokens
        usage["output"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            transcript.write("\n**refusal**\n")
            break
        msgs.append({"role": "assistant", "content": msg.content})
        results = []
        for b in msg.content:
            if b.type == "thinking" and b.thinking:
                transcript.write(f"\n> _thinking:_ {b.thinking[:2000]}\n")
            elif b.type == "text" and b.text.strip():
                transcript.write(f"\n{b.text}\n")
                print(b.text[:500], flush=True)
            elif b.type == "tool_use":
                inp = b.input if isinstance(b.input, dict) else json.loads(b.input)
                try:
                    if b.name == "finish":
                        finished = inp.get("summary", "")
                        out = {"ok": True}
                    else:
                        out = getattr(lab, b.name)(**inp)
                    err = False
                except Exception as e:  # noqa: BLE001 - report tool errors back to Claude
                    out, err = {"error": f"{type(e).__name__}: {e}"}, True
                txt = json.dumps(out)
                transcript.write(f"\n`{b.name}({json.dumps(inp)[:1500]})`\n\n```\n{txt[:3000]}\n```\n")
                print(f"[turn {turn}] {b.name} -> {txt[:300]}", flush=True)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": txt, "is_error": err})
        transcript.flush()
        if finished is not None or msg.stop_reason != "tool_use":
            break
        msgs.append({"role": "user", "content": results})
    chosen = lab.choose()
    th = lab.cands[chosen]["theta"]
    save_theta(os.path.join(a.run, "theta_wsp.npz"), th)
    json.dump({"arm": "wsp_claude", "theta": "theta_wsp.npz", "chosen_candidate": chosen,
               "selection": selection_metrics(th, D["selection"]), "edit_size": edit_size(th, th0),
               "n_candidates": len(lab.cands) - 1, "n_evaluations": len(lab.evals), "evals": lab.evals,
               "finish_summary": finished, "usage_tokens": usage, "model": MODEL, "effort": a.effort,
               "lineage": lineage(lab, chosen)},
              open(os.path.join(a.run, "arm_wsp_claude.json"), "w"), indent=2)
    print("chosen", chosen, selection_metrics(th, D["selection"]), "usage", usage)


def lineage(lab, cid):
    out = []
    while cid is not None:
        c = lab.cands[cid]
        out.append({"id": cid, "ops": c["ops"], "rationale": c["rationale"]})
        cid = c["parent"]
    return out[::-1]


def cmd_random(a):
    th0 = load_theta(os.path.join(a.run, "theta0.npz"))
    thw = load_theta(os.path.join(a.run, "theta_wsp.npz"))
    man = json.load(open(os.path.join(a.run, "run_manifest.json")))
    parts = {k: v["families"] for k, v in man["partitions"].items()}
    D = Data(a.acts, parts, ["selection"])
    rng = np.random.RandomState(SEED)
    best, log = None, []
    for e in range(EVAL_BUDGET):
        th = dict(th0)
        for k in ("W", "b"):  # same norm of change per layer as Claude's chosen edit, random direction
            dl = thw[k] - th0[k]
            r = rng.randn(*dl.shape)
            if k == "W":
                r *= (np.linalg.norm(dl, axis=1) / (np.linalg.norm(r, axis=1) + 1e-12))[:, None]
            else:
                r *= np.abs(dl) / (np.abs(r) + 1e-12)
            th[k] = th0[k] + r
        th["a"] = th0["a"] + rng.permutation(thw["a"] - th0["a"])
        th["a"] = np.clip(th["a"], 0, None) / np.clip(th["a"], 0, None).sum()
        m = selection_metrics(th, D["selection"])
        es = edit_size(th, th0)["total"]
        log.append({k: v for k, v in m.items() if k != "sel_per_family_auroc"})
        if best is None or better((m, es), best[:2]):
            best = (m, es, th)
    save_theta(os.path.join(a.run, "theta_random.npz"), best[2])
    json.dump({"arm": "random_edit_control", "theta": "theta_random.npz", "selection": best[0], "log": log},
              open(os.path.join(a.run, "arm_random_edit_control.json"), "w"), indent=2)
    print("random best:", best[0])


def cmd_final(a):
    lock = os.path.join(a.run, "FINAL_DONE")
    exploratory = os.path.exists(lock)
    if exploratory and not a.exploratory:
        sys.exit("test was already unsealed for this run; rerun with --exploratory (numbers will be labelled exploratory)")
    man = json.load(open(os.path.join(a.run, "run_manifest.json")))
    parts = {k: v["families"] for k, v in man["partitions"].items()}
    D = Data(a.acts, parts, ["calib", "test"])
    arms = sorted(f for f in os.listdir(a.run) if f.startswith("arm_") and f.endswith(".json"))
    th0 = load_theta(os.path.join(a.run, "theta0.npz"))
    T = D["test"]
    rows, scores = [], {}
    for f in arms:
        info = json.load(open(os.path.join(a.run, f)))
        th = load_theta(os.path.join(a.run, info["theta"]))
        s_cal, s_t = score(th, D["calib"]["X"]), score(th, T["X"])
        scores[info["arm"]] = (s_cal, s_t)
        au, per = macro_auroc(s_t, T["y"], T["fam"])
        r = {"arm": info["arm"], "sel_tpr5": info["selection"]["sel_tpr_at_5fpr"], "sel_auroc": info["selection"]["sel_macro_auroc"],
             "test_macro_auroc": au, "test_per_family": per, "edit_total": edit_size(th, th0)["total"]}
        for t in (0.01, 0.05):
            thr = thr_at(s_cal, t)
            r[f"thr{int(t*100)}"] = thr
            r[f"test_tpr{int(t*100)}"], r[f"test_fpr{int(t*100)}"] = tpr_fpr(s_t, T["y"], thr)
        rows.append(r)

    # paired bootstrap over test FAMILIES (groups), thresholds fixed from calibration
    fams = sorted(set(T["fam"]))
    rng = np.random.RandomState(SEED)
    B = 2000
    boot = {}
    for arm in scores:
        boot[arm] = {"tpr5": [], "fpr5": [], "auroc": []}
    samples = [rng.choice(fams, len(fams), replace=True) for _ in range(B)]
    for arm, (s_cal, s_t) in scores.items():
        thr = thr_at(s_cal, 0.05)
        flag = s_t > thr
        per_f = {f: (flag[(T["fam"] == f) & (T["y"] == 1)], flag[(T["fam"] == f) & (T["y"] == 0)]) for f in fams}
        au_f = {f: (roc_auc_score(T["y"][T["fam"] == f], s_t[T["fam"] == f]) if len(set(T["y"][T["fam"] == f])) == 2 else np.nan) for f in fams}
        for smp in samples:
            pos = np.concatenate([per_f[f][0] for f in smp])
            neg = np.concatenate([per_f[f][1] for f in smp])
            boot[arm]["tpr5"].append(pos.mean() if len(pos) else np.nan)
            boot[arm]["fpr5"].append(neg.mean() if len(neg) else np.nan)
            boot[arm]["auroc"].append(np.nanmean([au_f[f] for f in smp]))

    def ci(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return f"{np.mean(x):+.3f} [{np.quantile(x, .025):+.3f}, {np.quantile(x, .975):+.3f}]"

    diffs = []
    for a_, b_ in [("wsp_claude", "original"), ("wsp_claude", "gd_finetune"), ("wsp_claude", "refit_all_dev"),
                   ("wsp_claude", "random_edit_control"), ("gd_finetune", "original")]:
        if a_ in boot and b_ in boot:
            diffs.append((f"{a_} − {b_}", *(ci(np.array(boot[a_][k]) - np.array(boot[b_][k])) for k in ("tpr5", "fpr5", "auroc"))))

    tag = " (EXPLORATORY: test was already unsealed before)" if exploratory else ""
    L = [f"# WSP run results{tag}", "",
         f"Run `{os.path.basename(os.path.abspath(a.run))}`, source model `{man.get('source_model')}`, probe theta0 `{man['theta0_hash']}`, "
         f"replay max |diff| {man['replay_max_abs_diff']:.1e} (tol {man['replay_tol']}).", "",
         "Partitions (family-disjoint): " + "; ".join(f"**{k}** {v['n']} rows ({v['n_lie']} lies): {', '.join(v['families'])}" for k, v in man["partitions"].items()), "",
         "Thresholds are set on **calibration** honest answers only; **test** is evaluated once.", "",
         "| arm | sel TPR@5% | sel AUROC | test TPR@5% | test FPR@5% | test TPR@1% | test FPR@1% | test macro-AUROC | edit size |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['sel_tpr5']:.3f} | {r['sel_auroc']:.3f} | {r['test_tpr5']:.3f} | {r['test_fpr5']:.3f} | "
                 f"{r['test_tpr1']:.3f} | {r['test_fpr1']:.3f} | {r['test_macro_auroc']:.3f} | {r['edit_total']:.3f} |")
    L += ["", "Paired differences, bootstrap over the 7 test families (mean [95% interval]), threshold from calibration at 5% FPR:", "",
          "| comparison | Δ test TPR@5% | Δ test FPR@5% | Δ test macro-AUROC |", "|---|---|---|---|"]
    L += [f"| {d[0]} | {d[1]} | {d[2]} | {d[3]} |" for d in diffs]
    L += ["", "Per-family test AUROC:", "", "| family | " + " | ".join(r["arm"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for f in fams:
        L.append(f"| {f} | " + " | ".join(f"{r['test_per_family'].get(f, float('nan')):.3f}" for r in rows) + " |")
    open(os.path.join(a.run, "results_final.md" if not exploratory else "results_final_exploratory.md"), "w").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.join(a.run, "final_rows.json"), "w"), indent=2, default=float)
    open(lock, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "gd", "claude", "random", "final"])
    ap.add_argument("--acts", default="../research_loop/acts")
    ap.add_argument("--run", default="runs_wsp/run1")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max_turns", type=int, default=60)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "gd": cmd_gd, "claude": cmd_claude, "random": cmd_random, "final": cmd_final}[a.phase](a)


if __name__ == "__main__":
    main()
