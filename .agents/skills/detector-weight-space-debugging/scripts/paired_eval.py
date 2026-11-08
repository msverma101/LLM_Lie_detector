"""Calibrated paired evaluation of fixed detector scores with source-group bootstrap.

Input CSVs have row_id, group_id, family, label, score. Calibration contains only
honest rows. Candidate and original files must be aligned by IDs, never position.
This helper does not grant holdout access or pick a candidate after final opening.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def cutoff(scores, alpha=.025):
    a = np.sort(np.asarray(scores, dtype=float))
    if not 0 < alpha < 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty honest calibration and 0<alpha<1 required")
    rank = math.ceil((len(a) + 1) * (1 - alpha))
    return float(a[rank - 1]) if rank <= len(a) else float("inf")


def counts(labels, scores, threshold):
    y = np.asarray(labels)
    p = np.asarray(scores) > threshold
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Binary labels required")
    pos, neg = y == 1, y == 0
    return {"tp": int(np.sum(p & pos)), "fn": int(np.sum(~p & pos)),
            "fp": int(np.sum(p & neg)), "tn": int(np.sum(~p & neg)),
            "recall": float(p[pos].mean()) if pos.any() else None,
            "fpr": float(p[neg].mean()) if neg.any() else None}


def align(original, candidate):
    required = ["row_id", "group_id", "family", "label", "score"]
    for d in [original, candidate]:
        if not set(required) <= set(d) or d.row_id.duplicated().any() or d[required].isna().any().any():
            raise ValueError("Missing fields, missing values or duplicate row IDs")
        if not np.isfinite(d.score).all():
            raise ValueError("Nonfinite score")
    if set(original.row_id) != set(candidate.row_id):
        raise ValueError("Unpaired row IDs")
    a = original[required].sort_values("row_id").reset_index(drop=True)
    b = candidate[required].sort_values("row_id").reset_index(drop=True)
    if not a.drop(columns="score").equals(b.drop(columns="score")):
        raise ValueError("Labels or group/family assignments differ between arms")
    return a.rename(columns={"score": "original"}).assign(candidate=b.score.to_numpy())


def blocks(d):
    # A recorded group can span families: resample it once, not once per family.
    return [np.asarray(ix, dtype=int) for ix in d.groupby("group_id", sort=True).indices.values()]


def draw(group_blocks, rng):
    return np.concatenate([group_blocks[i] for i in rng.integers(0, len(group_blocks), len(group_blocks))])


def compare(calibration, evaluation, alpha=.025, draws=2000, seed=202611082):
    if set(calibration.label) != {0}:
        raise ValueError("Calibration must be honest only")
    if set(calibration.group_id) & set(evaluation.group_id) or set(calibration.row_id) & set(evaluation.row_id):
        raise ValueError("Calibration/evaluation overlap")
    if set(evaluation.label) != {0, 1}:
        raise ValueError("Evaluation needs both labels")
    out = {"alpha": alpha, "calibration_rows": len(calibration), "evaluation_rows": len(evaluation),
           "calibration_groups": calibration.group_id.nunique(), "evaluation_groups": evaluation.group_id.nunique(),
           "threshold_comparison": "separately calibrated; observed FPRs can differ", "arms": {}}
    predictions = {}
    for arm in ["original", "candidate"]:
        t = cutoff(calibration[arm], alpha)
        metric = counts(evaluation.label, evaluation[arm], t)
        metric["threshold"] = t if np.isfinite(t) else None
        metric["threshold_is_infinite"] = not np.isfinite(t)
        metric["auroc"] = float(roc_auc_score(evaluation.label, evaluation[arm]))
        metric["per_family"] = {str(f): counts(g.label, g[arm], t) for f, g in evaluation.groupby("family")}
        out["arms"][arm] = metric
        predictions[arm] = evaluation[arm].to_numpy() > t
    y = evaluation.label.to_numpy() == 1
    out["recall_difference"] = out["arms"]["candidate"]["recall"] - out["arms"]["original"]["recall"]
    out["gained_detections"] = int(np.sum(y & predictions["candidate"] & ~predictions["original"]))
    out["lost_detections"] = int(np.sum(y & ~predictions["candidate"] & predictions["original"]))
    rng = np.random.default_rng(seed)
    cb, eb = blocks(calibration), blocks(evaluation)
    values, invalid = [], 0
    for _ in range(draws):
        ci, ei = draw(cb, rng), draw(eb, rng)
        c, e = calibration.iloc[ci], evaluation.iloc[ei]
        if e.label.nunique() != 2:
            invalid += 1
            continue
        metrics = [counts(e.label, e[arm], cutoff(c[arm], alpha)) for arm in ["original", "candidate"]]
        values.append([metrics[1]["recall"] - metrics[0]["recall"], metrics[1]["fpr"], metrics[0]["fpr"]])
    if len(values) < max(100, draws * .9):
        raise ValueError("Insufficient valid bootstrap draws")
    interval = np.quantile(values, [.025, .975], axis=0)
    out["joint_group_bootstrap"] = {"draws": draws, "valid": len(values), "invalid": invalid,
        "seed": seed, "family_stratified": False, "scope": "recorded groups; familiar-family mixture",
        "recall_difference_95": interval[:, 0].tolist(), "candidate_fpr_95": interval[:, 1].tolist(),
        "original_fpr_95": interval[:, 2].tolist()}
    out["confirmed_improvement"] = bool(interval[0, 0] > 0 and interval[1, 1] <= .05)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ["calib-original", "calib-candidate", "eval-original", "eval-candidate", "out"]:
        ap.add_argument("--" + field, required=True)
    ap.add_argument("--alpha", type=float, default=.025)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    c = align(pd.read_csv(args.calib_original), pd.read_csv(args.calib_candidate))
    e = align(pd.read_csv(args.eval_original), pd.read_csv(args.eval_candidate))
    result = compare(c, e, args.alpha, args.draws)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
