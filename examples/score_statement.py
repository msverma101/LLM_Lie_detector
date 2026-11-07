#!/usr/bin/env python3
"""Score ONE statement with the model's own plausibility and the 4 published FIBS probes.

This is the same call path the web UI and the experiments use (liedet.serve.Scorer -> Capturer -> fibs.score_published).

  python examples/score_statement.py --model <Qwen3.5-2B dir> \
      --system "You are a helpful assistant. Answer truthfully and concisely." \
      --question "What is the capital of France?" --answer "Dublin" [--run runs/20261107-main] [--json]

--run (optional) points at the experiment run so each score is compared with the calibrated threshold (flagged yes/no).
Higher score = more suspicious. One statement proves nothing; see the experiment report for how reliable that is on average.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liedet.serve import Scorer, validate_score_request  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", default="You are a helpful assistant. Answer truthfully and concisely.")
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--variant", default="followup", choices=["followup", "answer"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    system, question, answer, variant = validate_score_request(
        {"system": a.system, "question": a.question, "answer": a.answer, "variant": a.variant})
    res = Scorer(a.model, a.run, a.dtype).score(system, question, answer, variant)
    if a.json:
        print(json.dumps(res, indent=2, default=float))
        return 0
    print(f"variant={res['variant']}  window tokens={res['n_tokens_window']}")
    for name, m in res["methods"].items():
        thr = "n/a" if m["threshold"] is None else f"{m['threshold']:.3f}"
        flag = "-" if m["flagged"] is None else ("SUSPICIOUS" if m["flagged"] else "ok")
        print(f"  {name:16s} score={m['score']:+.3f}  threshold={thr:>7s}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
