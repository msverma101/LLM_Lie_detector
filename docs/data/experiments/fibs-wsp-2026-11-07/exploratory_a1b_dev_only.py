"""EXPLORATORY, DEV ONLY (deviation from PLAN-0001, labelled as such): is the preregistered A1 simply too weak?
Reads train+calibration only. Does not touch the test split and does not change the primary endpoint."""
import sys, json
import numpy as np
sys.path.insert(0, "../../src")
from pathlib import Path
from liedet import fibs, wsp
from liedet.pipeline import _load, _arm_data
rows, feats, margins, split = _load(Path("features"), Path("data"))
layers = list(json.load(open("features/meta.json"))["layers"])
probe = fibs.load_published("linear"); s0, _ = wsp.state_from_published(probe)
data, Z, y, tr, ca = _arm_data(rows, feats, margins, split, probe, layers, "followup")
out = {}
for lam in (1.0, 0.1, 0.01, 0.0):
    for lr, steps in ((1e-3, 100), (1e-2, 300)):
        s = wsp.finetune(s0, data.Z_tr, data.y_tr, lam=lam, lr=lr, steps=steps)
        out[f"lam={lam},lr={lr},steps={steps}"] = {"train": wsp.recall_at_fpr(s.score(data.Z_tr), data.y_tr),
            "calib": wsp.recall_at_fpr(s.score(data.Z_ca), data.y_ca), "rel_change": float(np.linalg.norm(s.W - s0.W) / np.linalg.norm(s0.W))}
out["A0_reference"] = {"train": wsp.recall_at_fpr(s0.score(data.Z_tr), data.y_tr), "calib": wsp.recall_at_fpr(s0.score(data.Z_ca), data.y_ca)}
json.dump(out, open("exploratory_a1b_dev.json", "w"), indent=2)
for k, v in out.items(): print(k, {a: round(b, 3) for a, b in v.items()})
