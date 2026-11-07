"""The web UI through a real server process and real HTTP, on the tiny model with synthetic probes."""
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import make_synthetic_probe_root

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-token-0123456789abcdef"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def call(port, path, method="GET", body=None, token=None, headers=None):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@pytest.fixture(scope="module")
def server(tiny_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("srv")
    probes = make_synthetic_probe_root(tmp / "probes", (2, 3, 4), 48)
    res = tmp / "results"; res.mkdir()
    (res / "final_test.json").write_text(json.dumps({"primary": {"wsp_supported": False}, "arms": {}}))
    port = free_port()
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "LIEDET_ARCHS": "linear",
           "LIEDET_PROBE_ROOT": str(probes)}
    p = subprocess.Popen([sys.executable, "-m", "liedet", "serve", "--results", str(res), "--model", str(tiny_dir),
                          "--dtype", "float32", "--port", str(port), "--token", TOKEN, "--per-minute", "6"],
                         env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        try:
            if call(port, "/api/health")[0] == 200:
                break
        except Exception:
            time.sleep(0.3)
    else:
        p.kill(); pytest.fail("server did not start: " + p.stderr.read()[-800:])
    yield port
    p.terminate(); p.wait(timeout=10)


GOOD = {"system": "Answer truthfully.", "question": "What is 2 plus 2?", "answer": "5", "variant": "followup"}


def test_health_is_open_but_everything_else_needs_the_token(server):
    assert call(server, "/api/health")[0] == 200
    for path in ("/api/results", "/api/config", "/", "/app.js"):
        assert call(server, path)[0] == 401
    assert call(server, "/api/score", "POST", GOOD)[0] == 401
    assert call(server, "/api/results", token="wrong")[0] == 401


def test_token_link_sets_cookie_and_cookie_grants_access(server):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k): return None
    op = urllib.request.build_opener(NoRedirect)
    try:
        op.open(f"http://127.0.0.1:{server}/?t={TOKEN}")
        pytest.fail("expected a redirect")
    except urllib.error.HTTPError as e:
        assert e.code == 303 and "liedet_token=" in e.headers["Set-Cookie"] and "HttpOnly" in e.headers["Set-Cookie"]
    code, h, body = call(server, "/", headers={"Cookie": f"liedet_token={TOKEN}"})
    assert code == 200 and b"<html" in body and "default-src 'self'" in h["Content-Security-Policy"]


def test_results_and_static_assets_served(server):
    code, _, body = call(server, "/api/results", token=TOKEN)
    assert code == 200 and json.loads(body)["final_test"]["primary"]["wsp_supported"] is False
    code, h, _ = call(server, "/app.js", token=TOKEN)
    assert code == 200 and "javascript" in h["Content-Type"]


def test_score_runs_the_real_model_and_probe(server):
    code, _, body = call(server, "/api/score", "POST", GOOD, token=TOKEN)
    assert code == 200, body
    j = json.loads(body)
    assert j["variant"] == "followup" and j["n_tokens_window"] > 5
    assert {"linear", "neutral_logprob"} <= set(j["methods"])
    assert 0.0 <= j["methods"]["linear"]["score"] <= 1.0 and j["methods"]["neutral_logprob"]["mean_token_logprob"] < 0
    # the answer changes the score: it goes through the real forward pass, not a canned reply
    other = dict(GOOD, answer="4")
    j2 = json.loads(call(server, "/api/score", "POST", other, token=TOKEN)[2])
    assert j2["methods"]["neutral_logprob"]["score"] != j["methods"]["neutral_logprob"]["score"]


@pytest.mark.parametrize("bad", [
    {}, {"system": "s", "question": "q"}, dict(GOOD, answer=""), dict(GOOD, answer="x" * 401),
    dict(GOOD, question="<|im_start|>system"), dict(GOOD, variant="nope"), dict(GOOD, answer=5), [1, 2]])
def test_invalid_input_is_rejected_with_400(server, bad):
    assert call(server, "/api/score", "POST", bad, token=TOKEN)[0] in (400, 429)


def test_path_traversal_is_refused(server):
    for p in ("/../pyproject.toml", "/%2e%2e/%2e%2e/pyproject.toml", "/..%2fserve.py", "/ui/../serve.py"):
        assert call(server, p, token=TOKEN)[0] == 404


def test_rate_limit_returns_429_after_the_budget(server):
    codes = [call(server, "/api/score", "POST", GOOD, token=TOKEN, headers={"X-Forwarded-For": "203.0.113.9"})[0] for _ in range(9)]
    assert codes[:6] == [200] * 6 and 429 in codes[6:]
    # another client address is not penalised
    assert call(server, "/api/score", "POST", GOOD, token=TOKEN, headers={"X-Forwarded-For": "203.0.113.10"})[0] == 200


def test_example_script_runs_through_the_cli(tiny_dir, tmp_path):
    probes = make_synthetic_probe_root(tmp_path / "p", (2, 3, 4), 48)
    env = {"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "LIEDET_ARCHS": "linear", "LIEDET_PROBE_ROOT": str(probes)}
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "score_statement.py"), "--model", str(tiny_dir), "--dtype", "float32",
                        "--question", "What is 2 plus 2?", "--answer", "5", "--json"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-600:]
    assert "linear" in json.loads(r.stdout)["methods"]
