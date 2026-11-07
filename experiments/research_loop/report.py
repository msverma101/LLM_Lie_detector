"""Turn runs/ into a plot and a markdown summary.

  python report.py      -> runs/progress.png, runs/summary.md
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def main():
    df = pd.read_csv(os.path.join(RUNS, "results.tsv"), sep="\t")
    df["kept"] = df.kept.astype(bool)
    base = df[df.iter == 0].iloc[0]
    ok = df[df.val_auroc.notna()]
    best = ok[ok.kept].sort_values("val_auroc").iloc[-1]
    best_res = json.load(open(os.path.join(RUNS, f"iter_{int(best.iter):03d}.json")))
    base_res = json.load(open(os.path.join(RUNS, "iter_000.json")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    disc = ok[~ok.kept]
    ax1.scatter(disc.iter, disc.val_auroc, color="#b8b5c9", s=30, label="tried, discarded", zorder=2)
    keep = ok[ok.kept]
    ax1.scatter(keep.iter, keep.val_auroc, color="#0d9488", s=60, label="kept (new best)", zorder=3)
    run_best = ok.set_index("iter").val_auroc.where(ok.set_index("iter").kept).ffill()
    ax1.step(run_best.index, run_best.values, where="post", color="#0d9488", lw=2)
    crashes = df[df.val_auroc.isna()]
    if len(crashes):
        lo = ok.val_auroc.min()
        ax1.scatter(crashes.iter, [lo] * len(crashes), marker="x", color="#dc2626", label="crashed")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("val macro-AUROC (unseen families)")
    ax1.set_title("Research loop progress")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(alpha=0.3)

    fams = list(base_res["val_per_family"]) + list(base_res["test_per_family"])
    b = [base_res["val_per_family"].get(f, np.nan) if f in base_res["val_per_family"] else base_res["test_per_family"].get(f, np.nan) for f in fams]
    n = [best_res["val_per_family"].get(f, np.nan) if f in best_res["val_per_family"] else best_res["test_per_family"].get(f, np.nan) for f in fams]
    y = np.arange(len(fams))
    ax2.barh(y + 0.2, b, height=0.4, color="#b8b5c9", label="baseline")
    ax2.barh(y - 0.2, n, height=0.4, color="#0d9488", label=f"best (iter {int(best.iter)})")
    n_val = len(base_res["val_per_family"])
    ax2.set_yticks(y, [f + (" (val)" if i < n_val else " (test)") for i, f in enumerate(fams)])
    ax2.axhline(n_val - 0.5, color="black", lw=0.8, ls="--")
    ax2.invert_yaxis()
    ax2.set_xlim(0.4, 1.0)
    ax2.set_xlabel("AUROC")
    ax2.set_title("Per family: baseline vs. best")
    ax2.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "progress.png"), dpi=150)

    lines = [
        "# Research loop results\n",
        "| | baseline | best | change |",
        "|---|---|---|---|",
        f"| val macro-AUROC (optimized) | {base.val_auroc:.4f} | {best.val_auroc:.4f} | {best.val_auroc - base.val_auroc:+.4f} |",
        f"| **test macro-AUROC (held out)** | {base.test_auroc:.4f} | {best.test_auroc:.4f} | {best.test_auroc - base.test_auroc:+.4f} |",
        f"| train AUROC | {base.train_auroc:.4f} | {best.train_auroc:.4f} | |",
        "",
        f"{len(df) - 1} experiments: {int(df.kept.sum()) - 1} kept, {int((df.status == 'discard').sum())} discarded, "
        f"{int(df.status.isin(['crash', 'bad_proposal']).sum())} failed.",
        f"\nBest probe: `runs/iter_{int(best.iter):03d}.py`: {best.hypothesis}\n",
        "## All experiments\n",
        "| iter | val | test | status | hypothesis |",
        "|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        v = "-" if pd.isna(r.val_auroc) else f"{r.val_auroc:.4f}"
        t = "-" if pd.isna(r.test_auroc) else f"{r.test_auroc:.4f}"
        lines.append(f"| {r.iter} | {v} | {t} | {r.status} | {str(r.hypothesis).replace('|', '/')} |")
    open(os.path.join(RUNS, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print(f"\nwrote {RUNS}/progress.png and {RUNS}/summary.md")


if __name__ == "__main__":
    main()
