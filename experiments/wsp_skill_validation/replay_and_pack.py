"""Verify saved tensors/scores and optionally package the portable completed study."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(root):
    capture, final = root / "capture", root / "final"
    capture_done = json.loads((capture / "CAPTURE_DONE.json").read_text())
    for name, digest in capture_done["sha256"].items():
        assert sha(capture / name) == digest, (name, "capture changed")
    frozen = json.loads((final / "FROZEN.json").read_text())
    result = json.loads((final / "FINAL_DONE.json").read_text())
    meta = pd.read_csv(capture / "meta.csv").fillna("")
    activations = np.load(capture / "acts_last.npy", mmap_mode="r")
    records = []
    for arm, digest in frozen["checkpoint_sha256"].items():
        path = final / "checkpoints" / (arm + ".npz")
        assert sha(path) == digest, (arm, "checkpoint changed")
        with np.load(path, allow_pickle=False) as z:
            theta = {k: z[k] for k in z.files}
        raw_scores = {}
        for role in ["calib", "test"]:
            ii = np.flatnonzero((meta.partition == role).to_numpy())
            x = activations[ii].astype(np.float64)
            z = np.einsum("nld,ld->nl", (x - theta["mu"]) / theta["sd"], theta["W"]) + theta["b"]
            scores = expit(z).mean(1)
            raw_scores[role] = scores
            saved = pd.read_csv(final / (role + "_" + arm + ".csv"))
            assert saved.row_id.tolist() == meta.iloc[ii].row_id.tolist()
            error = float(np.max(np.abs(scores - saved.score.to_numpy())))
            assert error < 1e-10, (arm, role, error)
            threshold = result["arms"][arm]["threshold"]
            assert np.array_equal((scores > threshold).astype(int), saved.prediction.to_numpy())
            records.append({"arm": arm, "role": role, "max_score_error": error})
    verification = {"scope": "raw activation and saved-checkpoint replay; no independent GPU generation or agent replication",
                    "capture_hashes_verified": len(capture_done["sha256"]),
                    "checkpoint_arms_verified": len(frozen["checkpoint_sha256"]), "records": records}
    (root / "REPLAY_VERIFICATION.json").write_text(json.dumps(verification, indent=2))
    print("RAW_ACTIVATION_REPLAY_OK", verification["checkpoint_arms_verified"], "arms", flush=True)
    return verification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    replay(root)
    if args.pack:
        archive = root / "wsp_skill_validation_complete.tar.xz"
        with tarfile.open(archive, "w:xz", preset=5) as tf:
            for path in sorted(root.iterdir()):
                if path.name == archive.name or path.name.startswith("transfer_") or path.name == "EXPORT_MANIFEST.json":
                    continue
                tf.add(path, arcname=path.name)
        manifest = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
                    "raw_activations_included": True, "complete_contexts_included": True}
        (root / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print("ARCHIVE_READY", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
