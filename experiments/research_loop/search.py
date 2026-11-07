"""Comparison without an LLM: ordinary hyperparameter search with the SAME budget and rules.

  python search.py --acts acts --iters 15        -> runs_search/results.tsv (same columns as runs/)

Same harness (eval.py), same split, same selection rule as loop.py: a trial is kept only if
val macro-AUROC beats the current best by --min_gain. Only val scores drive selection; test
scores are logged for the final comparison. If Claude's loop beats this on TEST, the LLM adds
something beyond plain tuning; if not, that is a result too.

Search space (random search, seeded): pool, regularization C, which layers, how layers are
combined (average of per-layer probes vs. one probe on concatenated layers), family-balanced
sample weights.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loop import COLUMNS, run_eval  # noqa: E402

RUNS = os.path.join(HERE, "runs_search")
TSV = os.path.join(RUNS, "results.tsv")

TEMPLATE = '''"""Search trial: {desc}"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POOL = {pool!r}
C = {C}
LAYERS = {layers}          # indices into the cached layers
COMBINE = {combine!r}      # "mean": one probe per layer, average probs; "concat": one probe on all layers
BALANCE = {balance}        # family-balanced sample weights


def _weights(train):
    if not BALANCE:
        return None
    fam = np.asarray(train.family)
    _, inv, counts = np.unique(fam, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * len(w) / w.sum()


def _views(X):
    X = X[:, LAYERS]
    return [X.reshape(len(X), -1)] if COMBINE == "concat" else [X[:, j] for j in range(X.shape[1])]


def fit(train):
    w = _weights(train)
    models = []
    for V in _views(train.X[POOL]):
        sc = StandardScaler().fit(V)
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(V), train.y, sample_weight=w)
        models.append((sc, clf))
    return models


def predict(models, data):
    return np.mean([clf.predict_proba(sc.transform(V))[:, 1]
                    for (sc, clf), V in zip(models, _views(data.X[POOL]))], axis=0)
'''


def sample_config(rng, n_layers):
    k = int(rng.integers(1, n_layers + 1))
    layers = sorted(rng.choice(n_layers, size=k, replace=False).tolist())
    cfg = dict(pool=str(rng.choice(["last", "mean_answer", "mean_turn"])),
               C=float(10 ** rng.uniform(-4, 0)), layers=layers,
               combine=str(rng.choice(["mean", "concat"])), balance=bool(rng.integers(0, 2)))
    cfg["desc"] = (f"pool={cfg['pool']} C={cfg['C']:.4g} layers={layers} "
                   f"combine={cfg['combine']} balance={cfg['balance']}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--min_gain", type=float, default=0.002)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    acts = os.path.abspath(args.acts)
    os.makedirs(RUNS, exist_ok=True)
    n_layers = len(json.load(open(os.path.join(acts, "info.json")))["layers"])
    rng = np.random.default_rng(args.seed)

    # trial 0 = the same baseline as loop.py (runs/iter_000.py if present, else probe.py)
    base = os.path.join(HERE, "runs", "iter_000.py")
    base = base if os.path.exists(base) else os.path.join(HERE, "probe.py")
    shutil.copy(base, os.path.join(RUNS, "iter_000.py"))
    rows = []
    best = None
    for it in range(args.iters + 1):
        path = os.path.join(RUNS, f"iter_{it:03d}.py")
        if it == 0:
            desc = "baseline: FIBS LinearEnsemble on last token"
        else:
            cfg = sample_config(rng, n_layers)
            desc = cfg["desc"]
            open(path, "w").write(TEMPLATE.format(**cfg))
        res, secs = run_eval(path, acts, args.timeout)
        json.dump(res, open(os.path.join(RUNS, f"iter_{it:03d}.json"), "w"), indent=2)
        val = res.get("val_auroc", float("nan"))
        if "error" in res or val != val:
            status, keep = "crash", False
        elif best is None:
            status, keep = "baseline", True
        else:
            keep = val > best + args.min_gain
            status = "keep" if keep else "discard"
        if keep:
            best = val
        rows.append({"iter": it, "val_auroc": val, "test_auroc": res.get("test_auroc", float("nan")),
                     "train_auroc": res.get("train_auroc", float("nan")), "kept": keep, "status": status,
                     "seconds": round(secs, 1), "hypothesis": desc})
        pd.DataFrame(rows, columns=COLUMNS).to_csv(TSV, sep="\t", index=False)
        print(f"[search {it}] val={val:.4f} {status.upper()}  {desc}", flush=True)
    print(f"\nDone. Best val AUROC {best:.4f}. Results in {TSV}", flush=True)


if __name__ == "__main__":
    main()
