"""The research loop: Claude proposes a change to probe.py -> train + eval -> keep if better -> repeat.

  python loop.py --acts acts --iters 15

State lives in runs/ (resumable: rerun the same command to continue):
  runs/results.tsv      one row per iteration (val + test scores, kept?, hypothesis)
  runs/iter_XXX.py      every candidate probe
  runs/iter_XXX.json    its full eval result
  runs/iter_XXX.md      Claude's hypothesis + reasoning summary
probe.py always holds the best probe so far; runs/iter_000.py is the baseline.

The agent only ever sees VAL scores. TEST scores are logged for the final report.
Needs ANTHROPIC_API_KEY (or another credential the Anthropic SDK can find).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
TSV = os.path.join(RUNS, "results.tsv")
COLUMNS = ["iter", "val_auroc", "test_auroc", "train_auroc", "kept", "status", "seconds", "hypothesis"]


def run_eval(probe_path, acts, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), "--probe", probe_path, "--acts", acts],
                           capture_output=True, text=True, timeout=timeout, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout: eval took longer than {timeout}s"}, time.time() - t0
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):]), time.time() - t0
    tail = (p.stderr or p.stdout)[-2500:]
    return {"error": f"crash (exit {p.returncode}):\n{tail}"}, time.time() - t0


def history_table(df):
    if df.empty:
        return "(no iterations yet)"
    rows = ["iter | val_auroc | status | hypothesis", "---|---|---|---"]
    for r in df.itertuples():
        val = "nan" if pd.isna(r.val_auroc) else f"{r.val_auroc:.4f}"
        rows.append(f"{r.iter} | {val} | {r.status} | {str(r.hypothesis)[:300]}")
    return "\n".join(rows)


def build_prompt(best_code, best_res, df, last, data_desc):
    parts = [
        "## Data\n" + data_desc,
        "## Current best probe.py (val_auroc = {:.4f})\n```python\n{}\n```".format(best_res["val_auroc"], best_code),
        "Per-family val AUROC of the current best:\n" + json.dumps(best_res.get("val_per_family", {}), indent=1),
        f"Train AUROC of the current best: {best_res.get('train_auroc', float('nan')):.4f}",
        "## All experiments so far (val scores only)\n" + history_table(df),
    ]
    if last is not None:
        it, res = last
        if "error" in res:
            parts.append(f"## Last attempt (iter {it}) FAILED\n{res['error']}")
        else:
            parts.append(f"## Last attempt (iter {it}) per-family val AUROC\n" + json.dumps(res.get("val_per_family", {}), indent=1))
    parts.append("Propose the next experiment. Reply with <hypothesis>...</hypothesis> and the complete new probe.py in one ```python block.")
    return "\n\n".join(parts)


def ask_claude(client, system, prompt, model, effort):
    kwargs = dict(model=model, max_tokens=32000, system=system,
                  thinking={"type": "adaptive", "display": "summarized"},
                  output_config={"effort": effort},
                  messages=[{"role": "user", "content": prompt}])
    try:  # server-side fallback: if a safety classifier declines, the API retries on a fallback model
        with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs) as s:
            msg = s.get_final_message()
    except TypeError:  # older SDK without the `fallbacks` parameter
        with client.messages.stream(**kwargs) as s:
            msg = s.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request (stop_reason=refusal)")
    text = "".join(b.text for b in msg.content if b.type == "text")
    thinking = "".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking")
    return text, thinking, msg.usage


def mock_claude(best_code, it):
    """Offline stand-in for testing the loop mechanics: just varies C."""
    new_c = [0.001, 0.003, 0.03, 0.1][it % 4]
    code = re.sub(r"^C = .*$", f"C = {new_c}", best_code, flags=re.M)
    return f"<hypothesis>mock: set C={new_c}</hypothesis>\n```python\n{code}\n```", "", None


def parse(text):
    hyp = re.search(r"<hypothesis>(.*?)</hypothesis>", text, re.S)
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no ```python block in the reply")
    code = blocks[-1]
    compile(code, "probe.py", "exec")
    if "def fit" not in code or "def predict" not in code:
        raise ValueError("probe must define fit() and predict()")
    return (hyp.group(1).strip() if hyp else "(no hypothesis given)").replace("\n", " "), code


def append_row(row):
    df = pd.read_csv(TSV, sep="\t") if os.path.exists(TSV) else pd.DataFrame(columns=COLUMNS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TSV, sep="\t", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--min_gain", type=float, default=0.002, help="val improvement needed to keep a change")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--mock", action="store_true", help="no API calls; test the loop mechanics")
    args = ap.parse_args()
    acts = os.path.abspath(args.acts)
    os.makedirs(RUNS, exist_ok=True)
    probe_path = os.path.join(HERE, "probe.py")
    system = open(os.path.join(HERE, "program.md")).read()
    info = json.load(open(os.path.join(acts, "info.json")))
    meta = pd.read_csv(os.path.join(acts, "meta.csv"))
    data_desc = (f"model {info['model']}, layers {info['layers']}, d_model {info['d_model']}, "
                 f"{info['n']} examples from {meta.family.nunique()} families "
                 f"(~{int(meta.groupby('family').size().median())} per family, balanced honest/deceptive).")

    client = None
    if not args.mock:
        import anthropic
        client = anthropic.Anthropic()

    # iteration 0 = baseline
    if not os.path.exists(TSV):
        shutil.copy(probe_path, os.path.join(RUNS, "iter_000.py"))
        res, secs = run_eval(probe_path, acts, args.timeout)
        if "error" in res:
            sys.exit("baseline failed:\n" + res["error"])
        json.dump(res, open(os.path.join(RUNS, "iter_000.json"), "w"), indent=2)
        append_row({"iter": 0, "val_auroc": res["val_auroc"], "test_auroc": res["test_auroc"],
                    "train_auroc": res["train_auroc"], "kept": True, "status": "baseline",
                    "seconds": round(secs, 1), "hypothesis": "baseline: FIBS LinearEnsemble on last token"})
        print(f"[iter 0] baseline val={res['val_auroc']:.4f} test={res['test_auroc']:.4f}", flush=True)

    df = pd.read_csv(TSV, sep="\t")
    kept = df[df.kept.astype(bool)]
    best_it = int(kept.loc[kept.val_auroc.idxmax(), "iter"])
    best_res = json.load(open(os.path.join(RUNS, f"iter_{best_it:03d}.json")))
    best_code = open(probe_path).read()
    last = None
    start = int(df.iter.max()) + 1

    for it in range(start, start + args.iters):
        print(f"\n[iter {it}] asking {'mock' if args.mock else args.model} for an experiment...", flush=True)
        t0 = time.time()
        try:
            prompt = build_prompt(best_code, best_res, df, last, data_desc)
            text, thinking, usage = mock_claude(best_code, it) if args.mock else ask_claude(
                client, system, prompt, args.model, args.effort)
            hyp, code = parse(text)
        except Exception as e:
            print(f"[iter {it}] proposal failed: {e}", flush=True)
            res = {"error": f"proposal failed: {e}"}
            df = append_row({"iter": it, "val_auroc": float("nan"), "test_auroc": float("nan"),
                             "train_auroc": float("nan"), "kept": False, "status": "bad_proposal",
                             "seconds": round(time.time() - t0, 1), "hypothesis": str(e)[:200]})
            last = (it, res)
            continue
        cand = os.path.join(RUNS, f"iter_{it:03d}.py")
        open(cand, "w").write(code)
        open(os.path.join(RUNS, f"iter_{it:03d}.md"), "w").write(
            f"# iter {it}\n\n**Hypothesis:** {hyp}\n\n## Reasoning summary\n\n{thinking}\n")
        print(f"[iter {it}] hypothesis: {hyp}", flush=True)

        res, secs = run_eval(cand, acts, args.timeout)
        json.dump(res, open(os.path.join(RUNS, f"iter_{it:03d}.json"), "w"), indent=2)
        val = res.get("val_auroc", float("nan"))
        if "error" in res or val != val:
            status, keep = "crash", False
            print(f"[iter {it}] FAILED: {res.get('error', 'nan score')[:300]}", flush=True)
        else:
            keep = val > best_res["val_auroc"] + args.min_gain
            status = "keep" if keep else "discard"
            print(f"[iter {it}] val={val:.4f} (best {best_res['val_auroc']:.4f}) -> {status.upper()}", flush=True)
        df = append_row({"iter": it, "val_auroc": val, "test_auroc": res.get("test_auroc", float("nan")),
                         "train_auroc": res.get("train_auroc", float("nan")), "kept": keep, "status": status,
                         "seconds": round(secs, 1), "hypothesis": hyp})
        if keep:
            shutil.copy(cand, probe_path)
            best_code, best_res = code, res
        last = (it, res)

    print(f"\nDone. Best val AUROC {best_res['val_auroc']:.4f} (baseline "
          f"{df.loc[df.iter == 0, 'val_auroc'].iloc[0]:.4f}). Run: python report.py", flush=True)


if __name__ == "__main__":
    main()
