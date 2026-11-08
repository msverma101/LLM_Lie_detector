"""Freeze selection before final score access; report rejected hypotheses honestly."""
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--previous-run", required=True)
    ap.add_argument("--control-contract", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    root, previous = Path(args.root), Path(args.previous_run)
    out = root / "final"
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("ev", root / "paired_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    spec = importlib.util.spec_from_file_location("audit", root / "audit_edit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    capture = root / "capture"
    assert (capture / "CAPTURE_DONE.json").exists()
    original = audit.load(previous / "theta0.npz")
    meta = pd.read_csv(capture / "meta.csv").fillna("")
    assert meta.groupby("group_id").partition.nunique().max() == 1
    def role_data(role):
        idx = np.flatnonzero((meta.partition == role).to_numpy())
        return meta.iloc[idx].reset_index(drop=True), np.load(capture / "acts_last.npy", mmap_mode="r")[idx].astype(np.float64)
    score_queries = []
    def score(t, m, x, role, arm):
        score_queries.append({"role": role, "arm": arm, "rows": len(x), "unix": time.time()})
        z = np.einsum("nld,ld->nl", (x - t["mu"]) / t["sd"], t["W"]) + t["b"]
        return m.assign(score=expit(z).mean(1))
    probe = json.loads((root / "natural_probe/PROBE_RESULT.json").read_text())
    selected, selection = original, {"decision": "retain_original", "reason": "natural intervention failed registered diagnostic transfer gate", "validation_opened": False}
    if probe["qualifies_for_validation"]:
        candidate = audit.load(root / "natural_probe/primary_a1.npz")
        mc, xc = role_data("val_calib")
        me, xe = role_data("validation")
        ca = score(original, mc, xc, "val_calib", "original")
        cb = score(candidate, mc, xc, "val_calib", "candidate")
        ea = score(original, me, xe, "validation", "original")
        eb = score(candidate, me, xe, "validation", "candidate")
        val = ev.compare(ev.align(ca, cb), ev.align(ea, eb), draws=args.draws)
        adopted = val["joint_group_bootstrap"]["recall_difference_95"][0] > 0 and val["arms"]["candidate"]["fpr"] <= .05
        if adopted:
            selected = candidate
        selection = {"decision": "adopt_primary_a1" if adopted else "retain_original", "validation_opened": True, "validation": val}
    write(out / "SELECTION.json", selection)
    arms = {"original": original, "selected": selected}
    for name in ["primary_a1", "dose_a05", "dose_a15", "wrong_direction", "orthogonal_random"] + [f"isolate_{l}" for l in range(6)]:
        arms["diagnostic_" + name] = audit.load(root / "natural_probe" / (name + ".npz"))
    winners = json.loads((previous / "ordinary/WINNERS.json").read_text())
    arms["historical_ordinary_extra"] = audit.load(previous / winners["extra_data"]["path"])
    contract = json.loads(Path(args.control_contract).read_text())
    corrupt = copy.deepcopy(original)
    j = contract["fault_coordinate"]
    corrupt["W"][:, j] += 25 * np.max(np.abs(original["W"]), axis=1)
    repair = copy.deepcopy(corrupt)
    repair["W"][:, j] = 0
    for name, theta in [("engineered_corrupt", corrupt), ("engineered_repair", repair)]:
        for k in theta:
            assert audit.array_hash(theta[k]) == contract["arrays"][name][k], (name, k, "independent agent artifact differs")
        arms[name] = theta
    (out / "checkpoints").mkdir()
    for name, theta in arms.items():
        np.savez_compressed(out / "checkpoints" / (name + ".npz"), **theta)
    write(out / "CONTROL_CONTRACT.json", contract)
    frozen = {"unix": time.time(), "selection": selection, "alpha": .025,
              "checkpoint_sha256": {name: sha(out / "checkpoints" / (name + ".npz")) for name in arms},
              "protocol_sha256": sha(root / "PROTOCOL.md"), "hypothesis_sha256": sha(root / "NATURAL_HYPOTHESIS.json"),
              "capture_done_sha256": sha(capture / "CAPTURE_DONE.json"),
              "code_sha256": {name: sha(root / name) for name in ["finalize_study.py", "paired_eval.py", "audit_edit.py", "probe_natural.py"]},
              "interpretation": "Only selected is the primary natural submission; other doses/rejected edits are diagnostics, engineered arms test an injected fault, historical ordinary fitting is not newly budget-matched"}
    write(out / "FROZEN.json", frozen)
    write(out / "FINAL_OPENED.json", {"unix": time.time(), "frozen_sha256": sha(out / "FROZEN.json")})
    mc, xc = role_data("calib")
    mt, xt = role_data("test")
    calibration, evaluation, summaries = {}, {}, {}
    for name, theta in arms.items():
        c = score(theta, mc, xc, "calib", name)
        e = score(theta, mt, xt, "test", name)
        threshold = ev.cutoff(c.score, .025)
        c.assign(prediction=(c.score > threshold).astype(int)).to_csv(out / ("calib_" + name + ".csv"), index=False)
        e.assign(prediction=(e.score > threshold).astype(int)).to_csv(out / ("test_" + name + ".csv"), index=False)
        calibration[name], evaluation[name] = c, e
        summary = ev.counts(e.label, e.score, threshold)
        summary["threshold"] = threshold
        summary["per_family"] = {str(f): ev.counts(g.label, g.score, threshold) for f, g in e.groupby("family")}
        summary["fpr_binomial_upper_95_descriptive"] = float(beta.ppf(.95, summary["fp"] + 1, summary["tn"])) if summary["tn"] else 1.
        summaries[name] = summary
        print("FROZEN_FINAL_ARM", name, json.dumps({k: summary[k] for k in ["tp", "fn", "fp", "tn", "recall", "fpr"]}), flush=True)
    comparisons = {}
    for a, b, name in [("original", "selected", "primary_selected_vs_original"),
                        ("original", "diagnostic_primary_a1", "rejected_primary_vs_original"),
                        ("engineered_corrupt", "engineered_repair", "engineered_repair_vs_corrupt"),
                        ("original", "engineered_repair", "engineered_repair_vs_uncorrupted"),
                        ("original", "historical_ordinary_extra", "historical_ordinary_vs_original")]:
        comparisons[name] = ev.compare(ev.align(calibration[a], calibration[b]), ev.align(evaluation[a], evaluation[b]), draws=args.draws)
        print("PAIRED_COMPARISON_DONE", name, flush=True)
    result = {"date": "2026-11-08", "alpha": .025, "selection": selection,
              "capture_rows": len(meta), "counts": meta.groupby(["partition", "family", "label"]).size().rename("n").reset_index().to_dict("records"),
              "calibration_rows": len(mc), "calibration_groups": int(mc.group_id.nunique()),
              "test_rows": len(mt), "test_groups": int(mt.group_id.nunique()),
              "arms": summaries, "comparisons": comparisons, "score_queries": score_queries,
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
              "scope": "Fresh recorded groups from familiar families; no proven natural improvement, engineered recovery is a separate test; no monitored-LLM edits"}
    write(out / "FINAL_DONE.json", result)
    print("FINAL_DONE", json.dumps({"selection": selection["decision"], "test_rows": len(mt), "calibration_rows": len(mc)}), flush=True)


if __name__ == "__main__":
    main()
