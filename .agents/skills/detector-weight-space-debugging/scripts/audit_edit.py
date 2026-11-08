"""Check fixed six-probe NPZ contracts and distinguish calibration from repair.

This checks tensors, not whether a causal hypothesis or optimizer-free provenance
is convincing. Each W row is a distinct conceptual logistic-probe tensor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def inspect(parent, candidate, purpose="repair"):
    issues, probes = [], []
    required = {"W", "b", "mu", "sd"}
    if set(parent) != required or set(candidate) != required:
        return {"allowed": False, "issues": ["checkpoint_keys"], "purpose": purpose}
    for k in sorted(required):
        if candidate[k].shape != parent[k].shape:
            issues.append("changed_shape:" + k)
        if candidate[k].dtype != parent[k].dtype:
            issues.append("changed_dtype:" + k)
        if not np.isfinite(candidate[k]).all() or not np.isfinite(parent[k]).all():
            issues.append("nonfinite:" + k)
    if issues:
        return {"allowed": False, "issues": issues, "purpose": purpose}
    W, b = parent["W"], parent["b"]
    if W.ndim != 2 or b.shape != (len(W),) or parent["mu"].shape != W.shape or parent["sd"].shape != W.shape:
        issues.append("invalid_probe_shape")
    if np.any(parent["sd"] <= 0) or np.any(candidate["sd"] <= 0):
        issues.append("nonpositive_scale")
    for k in ["mu", "sd"]:
        if array_hash(parent[k]) != array_hash(candidate[k]):
            issues.append("changed_representation:" + k)
    if issues:
        return {"allowed": False, "issues": issues, "purpose": purpose}
    changed = 0
    calibration_only = []
    for l, row in enumerate(W):
        new = candidate["W"][l]
        dw = new - row
        db = float(candidate["b"][l] - b[l])
        n = int(np.count_nonzero(dw))
        if not n and db == 0:
            continue
        changed += n + int(db != 0)
        denom = float(row @ row)
        scalar = float(new @ row / denom) if denom > 0 else None
        residual = float(np.linalg.norm(new - scalar * row) / max(np.linalg.norm(new), np.linalg.norm(row), 1e-30)) if scalar is not None else None
        # Bias can differ: any affine whole-logit change remains a global knob.
        is_global = n == 0 or (residual is not None and residual <= 1e-10)
        if is_global:
            calibration_only.append(l)
        probes.append({"probe": l, "changed_W_entries": n, "delta_W_l2": float(np.linalg.norm(dw)),
                       "delta_b": db, "best_scalar": scalar, "scalar_residual": residual,
                       "global_calibration": is_global})
    if not changed and purpose == "repair":
        issues.append("no_weight_edit")
    if calibration_only and purpose == "repair":
        issues.append("whole_probe_calibration:" + ",".join(map(str, calibration_only)))
    return {"allowed": not issues, "purpose": purpose, "issues": issues,
            "changed_entries": changed, "probes": probes,
            "classification": "unchanged" if not changed else "contains_global_calibration" if calibration_only else "localized_or_directional",
            "mechanism_validated": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent")
    ap.add_argument("candidate")
    ap.add_argument("--purpose", choices=["repair", "calibration-control", "ablation-control", "unchanged-control"], default="repair")
    ap.add_argument("--out")
    args = ap.parse_args()
    result = inspect(load(args.parent), load(args.candidate), args.purpose)
    result["files"] = [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                       for p in [args.parent, args.candidate]]
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    raise SystemExit(0 if result["allowed"] else 2)


if __name__ == "__main__":
    main()
