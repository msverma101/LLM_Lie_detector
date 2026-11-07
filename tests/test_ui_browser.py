"""Real browser (Chromium via Playwright) against a real server process. Skipped if Playwright/Chromium is unavailable.
Regression: a CSP that blocks inline style attributes made every bar zero-width while all HTTP tests stayed green."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import make_synthetic_probe_root
from test_serve_e2e import ROOT, TOKEN, call, free_port

pw = pytest.importorskip("playwright.sync_api")

ARM = {"recall": 0.2, "recall_ci95": [0.1, 0.3], "fpr": 0.03, "fpr_ci95": [0.0, 0.08], "auroc": 0.6, "threshold": 0.5, "certified": True}


@pytest.fixture(scope="module")
def ui(tiny_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uib")
    probes = make_synthetic_probe_root(tmp / "probes", (2, 3, 4), 48)
    res = tmp / "results"; res.mkdir()
    (res / "final_test.json").write_text(json.dumps({"n_test_facts": 5, "n_test_rows": 20,
        "arms": {"A0_published": ARM, "A3_wsp_llm": dict(ARM, recall=0.5)},
        "primary": {"wsp_supported": False, "recall_diff_A3_minus_best_ordinary": -0.01, "ci95": [-0.1, 0.1], "a3_fpr_not_above_a0_plus_0.01": False}}))
    (res / "dev_report_followup.json").write_text(json.dumps({"error_analysis_linear": {"cells": {
        "prompt=truth,label=0": {"n": 4, "mean_score": 0.2, "flagged": 0.0}, "prompt=lie,label=1": {"n": 4, "mean_score": 0.5, "flagged": 0.3}}}}))
    port = free_port()
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "LIEDET_ARCHS": "linear", "LIEDET_PROBE_ROOT": str(probes)}
    p = subprocess.Popen([sys.executable, "-m", "liedet", "serve", "--results", str(res), "--model", str(tiny_dir), "--dtype", "float32",
                          "--port", str(port), "--token", TOKEN], env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if call(port, "/api/health")[0] == 200:
                break
        except Exception:
            time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    p.terminate(); p.wait(timeout=10)


def test_ui_renders_without_csp_errors_bars_have_width_and_scoring_works(ui):
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        pg = b.new_page(viewport={"width": 1100, "height": 900})
        errors = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"{ui}/?t={TOKEN}")
        pg.wait_for_selector(".verdict")
        widths = pg.evaluate("[...document.querySelectorAll('#tab-result .bar>i')].map(e => e.getBoundingClientRect().width)")
        assert len(widths) == 2 and all(w > 5 for w in widths) and widths[1] > widths[0]   # 0.5 > 0.2
        pg.click("#tabs button:nth-child(4)")
        pg.click("button.preset >> nth=1"); pg.click("button.primary")
        pg.wait_for_selector("#try-out table", timeout=30000)
        assert pg.locator("#try-out table tr").count() == 3                                 # header + linear + baseline
        pg.fill("#f-a", "<img src=x onerror=window.__pwned=1>"); pg.click("button.primary")
        pg.wait_for_selector("#try-out table", timeout=30000)
        assert pg.evaluate("window.__pwned === 1") is False and pg.locator("#tab-try img").count() == 0
        pg.click("#lang"); assert pg.inner_text("#t-title") in ("Lie detector lab", "Lügen-Detektor-Labor")
        b.close()
        assert errors == [], errors[:2]


def test_without_token_the_page_shows_no_results_and_an_error(ui):
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        pg = b.new_page(); pg.goto(f"{ui}/")
        assert "access token required" in pg.inner_text("body")
        assert pg.locator(".verdict").count() == 0
        b.close()
