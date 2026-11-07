"""Secondary analyses of PLAN-0001, run AFTER final-test (descriptive; the primary endpoint is already fixed).
Baselines are fit on TRAIN only, thresholds certified on CALIBRATION negatives, as for the arms."""
import json, sys
import numpy as np
sys.path.insert(0, "../../src")
from liedet import evalkit as ev, wsp
from liedet.pipeline import _load, _mask
from liedet.probes import TextBaseline
from liedet.cli import load_facts
from pathlib import Path

VAR = "followup"
rows, feats, margins, split = _load(Path("features"), Path("data"))
y = np.array([r.label for r in rows]); fid = np.array([r.fact_id for r in rows])
tr, ca, te = (_mask(rows, margins, split, p, VAR) for p in ("train", "calib", "test"))
sc = dict(np.load("dev_followup/published_scores.npz"))
methods = {f"published_{k}": v for k, v in sc.items()}
methods["baseline_neutral_logprob"] = -feats["lp_neutral"].astype(float)
methods["baseline_length"] = np.array([len(r.assistant) for r in rows], float)
texts = [f"{r.system} || {r.user} || {r.assistant}" for r in rows]
tb = TextBaseline().fit([t for t, m in zip(texts, tr) if m], y[tr])
methods["baseline_text_tfidf"] = tb.score(texts)
out = {}
for n, s in methods.items():
    cneg = s[ca & (y == 0)]
    try: thr = ev.certified_threshold(cneg, 0.05, 0.05); cert = True
    except ev.CannotCertify: thr = float(np.quantile(cneg, .95)); cert = False
    pos, neg = s[te & (y == 1)], s[te & (y == 0)]
    fl = (pos > thr).astype(float); lo, hi = ev.cluster_bootstrap_mean(fl, fid[te & (y == 1)], 10000, 0)
    k = int((neg > thr).sum()); flo, fhi = ev.clopper_pearson(k, len(neg))
    out[n] = {"auroc": ev.auroc(pos, neg), "recall": float(fl.mean()), "recall_ci95": [lo, hi], "fpr": k / len(neg), "fpr_ci95": [flo, fhi], "certified": cert}
# how far did ordinary optimisation (A1) move the weights?
z0, z1 = np.load("arms_followup/state_A0_published.npz"), np.load("arms_followup/state_A1_finetune.npz")
out["_A1_weight_change"] = {"rel_frobenius": float(np.linalg.norm(z1["W"] - z0["W"]) / np.linalg.norm(z0["W"])),
                            "max_abs_delta": float(np.abs(z1["W"] - z0["W"]).max())}
# accepted edits of A2/A3
for a in ("A2_random", "A3_wsp_llm"):
    h = json.load(open(f"arms_followup/history_{a}.json"))
    out[f"_{a}_accepted_edits"] = [x["edit"] for x in h if x["accepted"]]
json.dump(out, open("secondary_test_analysis.json", "w"), indent=2)
for n, v in out.items():
    if n.startswith("_"): print(n, json.dumps(v)); continue
    print(f"{n:26s} auroc={v['auroc']:.3f} recall={v['recall']:.3f} ({v['recall_ci95'][0]:.2f}-{v['recall_ci95'][1]:.2f}) fpr={v['fpr']:.3f} cert={v['certified']}")
