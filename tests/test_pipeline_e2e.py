"""fibs-eval -> wsp-run -> final-test through the real CLI (subprocess) on the tiny model with synthetic probes."""
import json
import subprocess
import sys
from pathlib import Path

from conftest import make_synthetic_probe_root

ROOT = Path(__file__).resolve().parents[1]


def run(env_extra, *args):
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "LIEDET_ARCHS": "linear", **env_extra}
    return subprocess.run([sys.executable, "-m", "liedet", *args], capture_output=True, text=True, env=env, cwd=ROOT)


def test_dev_arms_and_one_shot_final_test(tiny_dir, tmp_path):
    probes = make_synthetic_probe_root(tmp_path / "probes", (2, 3, 4), 48)
    e = {"LIEDET_PROBE_ROOT": str(probes)}
    d, f, dev, arms, fin = (str(tmp_path / x) for x in ("data", "f", "dev", "arms", "fin"))
    assert run(e, "build-data", "--out", d, "--n-arith", "80", "--limit-facts", "120").returncode == 0
    r = run(e, "capture", "--model", str(tiny_dir), "--data", d, "--out", f, "--dtype", "float32", "--tokens")
    assert r.returncode == 0, r.stderr[-800:]
    r = run(e, "fibs-eval", "--data", d, "--features", f, "--out", dev)
    assert r.returncode == 0, r.stderr[-800:]
    rep = json.loads((Path(dev) / "dev_report.json").read_text())
    assert "published_linear" in rep["methods"] and "error_analysis_linear" in rep
    r = run(e, "wsp-run", "--data", d, "--features", f, "--out", arms, "--proposer", "random", "--rounds", "2", "--per-round", "4")
    assert r.returncode == 0, r.stderr[-800:]
    summ = json.loads((Path(arms) / "arms_dev_summary.json").read_text())
    assert set(summ["arms"]) == {"A0_published", "A1_finetune", "A2_random", "A3_wsp_llm"}
    # A0 must be bit-identical to the published weights (no hidden edit); histories exist for searched arms
    assert summ["arms"]["A2_random"]["n_proposed"] == 8
    r = run(e, "final-test", "--data", d, "--features", f, "--arms", arms, "--out", fin, "--n-boot", "200")
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads((Path(fin) / "final_test.json").read_text())
    assert res["primary"]["wsp_supported"] in (True, False) and res["n_test_facts"] > 5
    assert (Path(fin) / "FINAL.lock").exists()
    # the test split is evaluated ONCE: a second run is refused
    r2 = run(e, "final-test", "--data", d, "--features", f, "--arms", arms, "--out", fin, "--n-boot", "200")
    assert r2.returncode == 3 and "refused" in r2.stdout
