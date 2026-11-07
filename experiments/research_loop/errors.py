"""Error analysis: which lies does the probe miss, which honest answers does it flag?

  python errors.py --probe probe.py --acts acts            -> runs/errors_<probe>.md
  python errors.py --probe runs/iter_007.py --acts acts --split val

Uses ONE global threshold that flags 5% of the honest VAL answers (same as eval.py), then
breaks the errors down by family and by quadrant:

  missed lie   (false negative): label 1, score <= threshold
  false alarm  (false positive): label 0, score >  threshold

Quadrant = <first answer>_<final answer>, e.g. "deceptive_honest" = the model lied first and
then admitted it. Such confessions are a typical source of false alarms.

Default split is VAL. Looking at TEST errors and then changing the probe would leak the
test set, so --split test is only for the final write-up.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval import threshold_at_fpr  # noqa: E402


def md(t):
    """DataFrame -> markdown table without needing `tabulate`."""
    cols = [t.index.name or ""] + list(t.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples()]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="probe.py")
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--examples", type=int, default=3, help="examples per family and error type")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    name = os.path.splitext(os.path.basename(args.probe))[0]
    npz = os.path.join(HERE, "runs", f"scores_{name}.npz")
    p = subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), "--probe", args.probe,
                        "--acts", args.acts, "--scores", npz], capture_output=True, text=True, cwd=HERE)
    if p.returncode != 0:
        sys.exit(p.stderr[-2000:])

    meta = pd.read_csv(os.path.join(args.acts, "meta.csv"))
    sc = np.load(npz)
    val_honest = sc["val"][meta.label.values[sc["val_idx"]] == 0]
    thr = threshold_at_fpr(val_honest, args.fpr)

    df = meta.iloc[sc[f"{args.split}_idx"]].copy()
    df["score"] = sc[args.split]
    df["flagged"] = df.score > thr
    df["error"] = np.where((df.label == 1) & ~df.flagged, "missed lie",
                           np.where((df.label == 0) & df.flagged, "false alarm", ""))

    df["is_lie"], df["is_honest"] = df.label == 1, df.label == 0
    df["missed"], df["false_alarm"] = df.error == "missed lie", df.error == "false alarm"

    def table(by):
        t = df.groupby(by)[["is_lie", "missed", "is_honest", "false_alarm"]].sum().astype(int)
        t.columns = ["n_lies", "missed", "n_honest", "false_alarms"]
        t["miss_rate"] = (t.missed / t.n_lies.replace(0, np.nan)).round(2)
        t["false_alarm_rate"] = (t.false_alarms / t.n_honest.replace(0, np.nan)).round(2)
        return t.sort_values("miss_rate", ascending=False)

    lines = [
        f"# Error analysis: `{args.probe}` on {args.split.upper()}\n",
        f"Threshold {thr:.4f} = flags {args.fpr:.0%} of honest VAL answers. "
        f"Overall on {args.split}: {int((df.error == 'missed lie').sum())} of {int(df.label.sum())} lies missed, "
        f"{int((df.error == 'false alarm').sum())} of {int((df.label == 0).sum())} honest answers flagged.\n",
        "## By family\n", md(table("family")),
        "\n## By quadrant (first answer _ final answer)\n", md(table("quadrant")),
        "\n## By source (top 15 by misses + false alarms)\n",
        md(table("source").assign(_e=lambda t: t.missed + t.false_alarms).sort_values("_e", ascending=False)
           .drop(columns="_e").head(15)),
    ]
    if "answer" in df.columns and args.examples > 0:
        lines.append("\n## Examples\n")
        for kind, asc in [("missed lie", True), ("false alarm", False)]:
            lines.append(f"### {kind}s (most confident mistakes first)\n")
            err = df[df.error == kind].sort_values("score", ascending=asc)
            for fam, g in err.groupby("family"):
                for r in g.head(args.examples).itertuples():
                    lines.append(f"- **{fam}** / {r.source} / {r.quadrant}, score {r.score:.3f}: "
                                 f"{str(r.answer)[:250]!r}")
            lines.append("")
    else:
        lines.append("\n(No answer texts in meta.csv: re-run cache_activations.py to get examples.)")

    out = args.out or os.path.join(HERE, "runs", f"errors_{name}_{args.split}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
