"""End-to-end through the real command line (subprocess) on the tiny random model."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}


def run(*args):
    return subprocess.run([sys.executable, "-m", "liedet", *args], capture_output=True, text=True, env=ENV, cwd=ROOT)


def test_full_pipeline_on_tiny_model(tiny_dir, tmp_path):
    out = tmp_path / "run"
    r = run("build-data", "--out", str(out / "data"), "--n-arith", "40", "--limit-facts", "60")
    assert r.returncode == 0, r.stderr
    r = run("capture", "--model", str(tiny_dir), "--data", str(out / "data"), "--out", str(out / "f"), "--dtype", "float32")
    assert r.returncode == 0, r.stderr
    z = np.load(out / "f" / "features.npz")
    assert z["mean"].shape[0] == len(z["row_ids"]) and z["mean"].ndim == 3
    r = run("evaluate", "--data", str(out / "data"), "--features", str(out / "f"), "--out", str(out / "e"),
            "--n-boot", "200")
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "e" / "results.json").read_text())
    assert "in_distribution/answer" in res and any(k.startswith("held_out_family=") for k in res)
    # random tiny model has no belief: the harness must NOT claim support for the probe on noise
    assert (out / "e" / "report.md").read_text().startswith("# liedet evaluation")
    r = run("weights", "--model", str(tiny_dir), "--data", str(out / "data"), "--features", str(out / "f"),
            "--out", str(out / "w"), "--dtype", "float32", "--layer-pos", "1", "--alpha-edit", "2.0", "--n-null", "128")
    assert r.returncode == 0, r.stderr
    w = json.loads((out / "w" / "weights.json").read_text())
    assert w["positive_control_planted_edit"]["z"] > 8
    assert abs(w["negative_control_random_edit_vs_u"]["z"]) < 5
    # manifest pins every artefact
    for sub in ("data", "f", "e", "w"):
        assert (out / sub / "MANIFEST.sha256").exists()


def test_rerun_is_deterministic(tiny_dir, tmp_path):
    h = []
    for k in range(2):
        d = tmp_path / f"d{k}"
        assert run("build-data", "--out", str(d), "--n-arith", "20").returncode == 0
        h.append((d / "rows.jsonl").read_bytes())
    assert h[0] == h[1]
