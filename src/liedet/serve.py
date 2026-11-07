"""Small token-protected web UI + JSON API on top of the same code the experiments use.

  GET  /                 the single-page UI (needs the access token once: /?t=<token> sets a cookie)
  GET  /api/health       liveness (no token, no model call)
  GET  /api/results      stored experiment results (preregistered run), read from the results directory
  POST /api/score        score ONE statement with the 4 published FIBS probes + the model's own plausibility

Scoring goes through liedet.capture.Capturer + liedet.fibs.score_published (the call path of the experiments), so what the
UI shows is what the pipeline computes. Access: bearer token or cookie, per-IP rate limit, input length limits, CSP.
"""
from __future__ import annotations

import hmac
import json
import mimetypes
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import evalkit as ev

UI_DIR = Path(__file__).parent / "ui"
MAX_FIELD = 400
RESULT_FILES = ("fibs_parity.json", "dev_report_followup.json", "dev_report_answer.json", "arms_dev_summary.json",
                "final_test.json", "secondary_test_analysis.json", "exploratory_a1b_dev.json", "split.json", "meta.json")


class Scorer:
    """Holds the model, the published probes and the calibration thresholds."""

    def __init__(self, model_dir: str, run_dir: str | None, dtype: str = "bfloat16", alpha: float = 0.05):
        from . import fibs
        from .capture import Capturer
        self.cap = Capturer(model_dir, dtype=dtype)
        self.probes = {a: fibs.load_published(a) for a in fibs.ARCHS}
        for a, p in self.probes.items():
            if tuple(p.layers) != tuple(self.cap.layers):
                raise RuntimeError(f"probe {a} layers {p.layers} != captured layers {self.cap.layers}")
        self.lock = threading.Lock()
        self.thresholds = self._thresholds(run_dir, alpha) if run_dir else {}

    def _thresholds(self, run_dir: str, alpha: float) -> dict:
        """Calibration-negative thresholds per method and variant, from the stored dev scores (never the test split)."""
        from . import pipeline
        run = Path(run_dir)
        out: dict = {}
        sc_file = run / "dev_followup" / "published_scores.npz"
        if not sc_file.exists():
            return out
        rows, feats, margins, split = pipeline._load(run / "features", run / "data")
        y = np.array([r.label for r in rows])
        sc = dict(np.load(sc_file))
        sc["neutral_logprob"] = -feats["lp_neutral"].astype(float)
        for variant in ("followup", "answer"):
            ca = pipeline._mask(rows, margins, split, "calib", variant)
            for name, s in sc.items():
                neg = s[ca & (y == 0)]
                try:
                    thr, cert = ev.certified_threshold(neg, alpha, 0.05), True
                except ev.CannotCertify:
                    thr, cert = float(np.quantile(neg, 1 - alpha)), False
                out.setdefault(variant, {})[name] = {"threshold": float(thr), "certified": cert, "n_calib_negatives": int(len(neg))}
        return out

    def score(self, system: str, question: str, answer: str, variant: str) -> dict:
        from . import fibs
        from .capture import encode_cell, encode_neutral
        with self.lock:
            enc = encode_cell(self.cap.tokenizer, system, question, answer)
            feats = self.cap.cell_features(enc, keep_tokens=True)[variant]
            ids, s, e = encode_neutral(self.cap.tokenizer, question, answer)
            lp = self.cap.span_logprob(ids, s, e)
        res = {}
        for arch, probe in self.probes.items():
            val = fibs.score_published(probe, feats["tokens"], self.cap.layers)
            t = self.thresholds.get(variant, {}).get(arch)
            res[arch] = {"score": val, "threshold": t["threshold"] if t else None,
                         "flagged": (val > t["threshold"]) if t else None}
        t = self.thresholds.get(variant, {}).get("neutral_logprob")
        res["neutral_logprob"] = {"score": -lp, "mean_token_logprob": lp, "threshold": t["threshold"] if t else None,
                                  "flagged": (-lp > t["threshold"]) if t else None}
        return {"variant": variant, "n_tokens_window": int(feats["tokens"].shape[0]), "methods": res}


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic(); q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now); return True


def load_results(results_dir: Path) -> dict:
    out = {}
    for name in RESULT_FILES:
        p = results_dir / name
        if p.exists():
            out[name.removesuffix(".json")] = json.loads(p.read_text())
    return out


def validate_score_request(body: dict) -> tuple[str, str, str, str]:
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    vals = []
    for k in ("system", "question", "answer"):
        v = body.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"'{k}' must be a non-empty string")
        if len(v) > MAX_FIELD:
            raise ValueError(f"'{k}' is longer than {MAX_FIELD} characters")
        if "<|" in v:
            raise ValueError(f"'{k}' must not contain chat control tokens")
        vals.append(v.strip())
    variant = body.get("variant", "followup")
    if variant not in ("followup", "answer"):
        raise ValueError("'variant' must be 'followup' or 'answer'")
    return vals[0], vals[1], vals[2], variant


def make_handler(scorer: Scorer | None, results_dir: Path, token: str, per_minute: int):
    limiter = RateLimiter(per_minute)

    class H(BaseHTTPRequestHandler):
        server_version = "liedet-ui"

        def log_message(self, fmt, *args):   # no client data in logs beyond method/path/status
            pass

        def _client(self) -> str:
            return self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip() \
                or self.client_address[0]

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def _json(self, code, obj, extra=None):
            self._send(code, json.dumps(obj, default=float).encode(), extra=extra)

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            cookie = {c.split("=", 1)[0].strip(): c.split("=", 1)[-1].strip() for c in self.headers.get("Cookie", "").split(";") if "=" in c}
            cand = got[7:] if got.startswith("Bearer ") else cookie.get("liedet_token", "")
            return bool(cand) and hmac.compare_digest(cand.encode(), token.encode())

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/health":
                return self._json(200, {"ok": True, "model_loaded": scorer is not None})
            q = parse_qs(u.query)
            if "t" in q and hmac.compare_digest(q["t"][0].encode(), token.encode()):
                secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
                return self._send(303, b"", extra={"Location": "/", "Set-Cookie":
                                  f"liedet_token={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400{secure}"})
            if not self._authed():
                return self._json(401, {"error": "access token required: open the link you were given (/?t=...)"})
            if u.path == "/api/results":
                return self._json(200, load_results(results_dir))
            if u.path == "/api/config":
                return self._json(200, {"model_loaded": scorer is not None, "max_field": MAX_FIELD, "per_minute": per_minute,
                                        "thresholds": scorer.thresholds if scorer else {}})
            name = "index.html" if u.path in ("/", "/index.html") else u.path.lstrip("/")
            f = (UI_DIR / name).resolve()
            if UI_DIR.resolve() not in f.parents or not f.is_file():
                return self._json(404, {"error": "not found"})
            return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")

        def do_POST(self):
            if self.path != "/api/score":
                return self._json(404, {"error": "not found"})
            if not self._authed():
                return self._json(401, {"error": "access token required"})
            if scorer is None:
                return self._json(503, {"error": "model not loaded on this server"})
            if not limiter.allow(self._client()):
                return self._json(429, {"error": f"rate limit: {per_minute} requests per minute"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > 8192:
                    raise ValueError("request body must be 1..8192 bytes")
                body = json.loads(self.rfile.read(n))
                system, question, answer, variant = validate_score_request(body)
            except (ValueError, json.JSONDecodeError) as ex:
                return self._json(400, {"error": str(ex)})
            try:
                return self._json(200, scorer.score(system, question, answer, variant))
            except Exception as ex:   # surface a generic message, keep details out of the client
                return self._json(500, {"error": f"scoring failed ({type(ex).__name__})"})

    return H


def serve(host: str, port: int, results_dir: str, model_dir: str | None, run_dir: str | None, dtype: str,
          token: str | None, per_minute: int = 12, token_file: str | None = None) -> None:
    token = token or secrets.token_urlsafe(24)
    if token_file:
        p = Path(token_file); p.write_text(token + "\n"); p.chmod(0o600)
    scorer = Scorer(model_dir, run_dir, dtype) if model_dir else None
    srv = ThreadingHTTPServer((host, port), make_handler(scorer, Path(results_dir), token, per_minute))
    print(f"liedet-ui listening on http://{host}:{port}/  (model_loaded={scorer is not None})", flush=True)
    srv.serve_forever()
