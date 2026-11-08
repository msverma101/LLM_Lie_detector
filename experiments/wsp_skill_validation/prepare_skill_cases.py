"""Prepare blinded skill cases on real old development activations, not efficacy data.

The evaluator receives a checkpoint and diagnostic examples, not this constructor
or the answer key. This engineered test is separate from natural-detector efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    previous, root = Path(args.previous), Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    with np.load(previous / "run/theta0.npz", allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    meta = pd.read_csv(previous / "capture/meta.csv").fillna("")
    idx = np.flatnonzero((meta.partition == "inspect").to_numpy())
    X = np.load(previous / "capture/acts_last.npy", mmap_mode="r")[idx]
    m = meta.iloc[idx].reset_index(drop=True)
    raw = pd.read_json(previous / "capture/cases_inspect.jsonl", lines=True).fillna("")
    rng = np.random.default_rng(482917)
    j = int(rng.integers(base["W"].shape[1]))
    corrupt = {k: v.copy() for k, v in base.items()}
    # A strong, sparse fault added to all probe vectors; no activation/label fitting.
    corrupt["W"][:, j] += 25 * np.max(np.abs(base["W"]), axis=1)
    packet_map = {"case_a": corrupt, "case_b": base}
    for name, t in packet_map.items():
        packet = root / name
        packet.mkdir()
        np.savez_compressed(packet / "detector.npz", **t)
        np.save(packet / "activations.npy", X)
        m.to_csv(packet / "diagnostic_meta.csv", index=False)
        raw.to_json(packet / "diagnostic_cases.jsonl", orient="records", lines=True, force_ascii=False)
        (packet / "TASK.md").write_text("""Use the detector-weight-space-debugging skill to investigate this saved detector.
The checkpoint contains W,b,mu,sd; score=mean sigmoid(sum(((x-mu)/sd)*W)+b).
The monitored LLM and representation are frozen. Files are diagnostic data only.
Determine whether a direct weight repair is justified. If so, save HYPOTHESIS.json
before scoring your candidate, an inspectable edit.py and candidate.npz; otherwise
save a documented no-edit decision. Include correct controls and potential harms.
Do not train/fine-tune/fit a classifier. Work only in this packet, the provided
skill and your output subdirectory. Do not access other packets or parent files.
Report diagnostic results as development, not final detector efficacy.
""")
    answer = {"type": "engineered sparse-fault positive control and unchanged negative control",
              "case_a": {"fault": "additive coefficient anomaly", "coordinate": j,
                         "probes": list(range(len(base["W"]))), "scale": "25 * per-probe maxabs(original W)"},
              "case_b": {"fault": None},
              "rows": len(m), "groups": int(m.group_id.nunique()),
              "original_checkpoint_sha256": hashlib.sha256((previous / "run/theta0.npz").read_bytes()).hexdigest(),
              "data_scope": "previous diagnostic activations, not independent natural efficacy confirmation"}
    (root / "ANSWER_KEY.json").write_text(json.dumps(answer, indent=2))
    print("PREPARED_SKILL_CASES", len(m), "real diagnostic rows; answer key outside evaluator packets")


if __name__ == "__main__":
    main()
