"""Robustness check for the research-loop result (no GPU, no API calls; uses cached activations).

Question: is the gain of the best probe (iteration 9) over the baseline real, or noise from one
split and small samples? Three checks, each comparing probes PAIRED on the same examples:

  1. Original split (the one in RESULTS.md): test macro-AUROC and TPR at 1% / 5% FPR
     (threshold from honest VAL answers, as in eval.py), with two bootstrap intervals for the
     difference best - baseline:
       - rows:     resample conversations within each test family (families fixed)
       - families: resample test families too (two-level; wide, since there are only ~6-7)
  2. Family re-splits ("seeds"): N random new assignments of the families to
     train 14 / val 7 / test 7. Every probe is refit on each split. This asks whether the
     winning *idea* helps on other unseen families, not only on the ones it was picked on.
  3. Optional: the best probe of the non-LLM search (runs_search/), compared the same way.

Caveats (also printed in the report):
  - Check 1 reuses the test families whose scores were already printed during the loop.
    Humans have seen them, Claude has not.
  - Re-splits put the original val/test families into training, and the iteration-9 idea was
    selected on the original val families. They test transfer of the idea, not a sealed test.
  - Re-sampling the conversations themselves would need a new GPU caching run.

Usage (in research_loop/, after cache_activations.py):
  python robustness.py --acts acts                 # writes runs/robustness.md + runs/robustness.json
  python robustness.py --acts acts --n_splits 3    # quicker
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval import (TEST_FAMILIES, VAL_FAMILIES, load, load_probe, macro_auroc, make_split,  # noqa: E402
                  rates, threshold_at_fpr)

FPRS = (0.01, 0.05)


def search_best_probe():
    """Last kept trial of search.py (that is its best by val), or None."""
    tsv = os.path.join(HERE, "runs_search", "results.tsv")
    if not os.path.exists(tsv):
        return None
    df = pd.read_csv(tsv, sep="\t")
    kept = df[df.kept.astype(str) == "True"]
    if len(kept) < 2:  # only the baseline was kept: search found nothing better
        return None
    path = os.path.join(HERE, "runs_search", f"iter_{int(kept.iloc[-1]['iter']):03d}.py")
    return path if os.path.exists(path) else None


def fit_score(probe_path, train, val, test):
    probe = load_probe(probe_path)
    np.random.seed(0)
    model = probe.fit(train)
    s_val = np.asarray(probe.predict(model, val), dtype=np.float64).reshape(-1)
    s_test = np.asarray(probe.predict(model, test), dtype=np.float64).reshape(-1)
    return s_val, s_test


def metrics(s_val, y_val, s_test, y_test, f_test):
    out = {"auroc": macro_auroc(s_test, y_test, f_test)[0]}
    for t in FPRS:
        thr = threshold_at_fpr(s_val[y_val == 0], t)
        out[f"tpr@{int(t * 100)}"], out[f"fpr@{int(t * 100)}"] = rates(s_test, y_test, thr)
    return out


def splits_for(meta, val_fams, test_fams):
    fam = meta.family.values
    is_val, is_test = np.isin(fam, val_fams), np.isin(fam, test_fams)
    return ~(is_val | is_test), is_val, is_test


def run_split(meta, info, arrays, probes, val_fams, test_fams):
    is_train, is_val, is_test = splits_for(meta, val_fams, test_fams)
    train, _, _ = make_split(meta, info, arrays, is_train, with_labels=True)
    val, y_val, _ = make_split(meta, info, arrays, is_val, with_labels=False)
    test, y_test, f_test = make_split(meta, info, arrays, is_test, with_labels=False)
    scores = {name: fit_score(p, train, val, test) for name, p in probes.items()}
    return scores, y_val, y_test, f_test


def bootstrap(scores, y_val, y_test, f_test, ref, other, n_boot, rng, by_family):
    """Paired bootstrap of metric(other) - metric(ref) on test. Threshold stays fixed from val."""
    thr = {(name, t): threshold_at_fpr(scores[name][0][y_val == 0], t) for name in (ref, other) for t in FPRS}
    fams = np.array(sorted(set(f_test)))
    idx_by = {f: (np.where((f_test == f) & (y_test == 0))[0], np.where((f_test == f) & (y_test == 1))[0])
              for f in fams}
    diffs = {k: [] for k in ["auroc"] + [f"tpr@{int(t * 100)}" for t in FPRS]}
    for _ in range(n_boot):
        chosen = rng.choice(fams, size=len(fams), replace=True) if by_family else fams
        parts, groups = [], []
        for g, f in enumerate(chosen):
            neg, pos = idx_by[f]
            parts.append(np.concatenate([rng.choice(neg, len(neg)), rng.choice(pos, len(pos))]))
            groups.append(np.full(len(neg) + len(pos), g))
        ix, grp = np.concatenate(parts), np.concatenate(groups)
        y = y_test[ix]
        for k in diffs:
            vals = []
            for name in (ref, other):
                s = scores[name][1][ix]
                if k == "auroc":
                    vals.append(macro_auroc(s, y, grp)[0])
                else:
                    vals.append(rates(s, y, thr[(name, int(k[4:]) / 100)])[0])
            diffs[k].append(vals[1] - vals[0])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in diffs.items()}


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--baseline", default=os.path.join(HERE, "results", "baseline_probe.py"))
    ap.add_argument("--best", default=os.path.join(HERE, "results", "best_probe_iter009.py"))
    ap.add_argument("--n_splits", type=int, default=5, help="random family re-splits")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "robustness"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    probes = {"baseline": args.baseline, "best": args.best}
    sb = search_best_probe()
    if sb:
        probes["search"] = sb
    print("probes:", probes, flush=True)

    meta, info, arrays = load(args.acts)
    present = sorted(set(meta.family.values))
    report = {"probes": probes, "n_rows": int(len(meta)), "families": present}
    lines = ["# Robustness check: best probe (iter 9) vs. baseline", "",
             f"Probes: {', '.join(f'`{k}` = `{os.path.relpath(v, HERE)}`' for k, v in probes.items())}. "
             f"{len(meta)} conversations, {len(present)} families. Differences are paired (same examples). "
             f"Bootstrap: {args.n_boot} resamples, 95% percentile intervals.", ""]

    # 1. original split
    print("[1] original split ...", flush=True)
    scores, y_val, y_test, f_test = run_split(meta, info, arrays, probes, VAL_FAMILIES, TEST_FAMILIES)
    orig = {name: metrics(s[0], y_val, s[1], y_test, f_test) for name, s in scores.items()}
    report["original"] = {"metrics": orig, "bootstrap": {}}
    lines += ["## 1. Original split (test families of RESULTS.md)", "",
              "| probe | test macro-AUROC | TPR @1% FPR (test FPR) | TPR @5% FPR (test FPR) |", "|---|---|---|---|"]
    for name, m in orig.items():
        lines.append(f"| {name} | {m['auroc']:.4f} | {m['tpr@1']:.3f} ({m['fpr@1']:.3f}) | "
                     f"{m['tpr@5']:.3f} ({m['fpr@5']:.3f}) |")
    lines += ["", "| difference vs. baseline | point | 95% CI, rows resampled | 95% CI, families + rows |",
              "|---|---|---|---|"]
    for other in [n for n in probes if n != "baseline"]:
        ci_rows = bootstrap(scores, y_val, y_test, f_test, "baseline", other, args.n_boot, rng, by_family=False)
        ci_fam = bootstrap(scores, y_val, y_test, f_test, "baseline", other, args.n_boot, rng, by_family=True)
        report["original"]["bootstrap"][other] = {"rows": ci_rows, "families": ci_fam}
        for k in ci_rows:
            d = orig[other][k] - orig["baseline"][k]
            lines.append(f"| {other} − baseline, {k} | {d:+.4f} | {fmt_ci(ci_rows[k])} | {fmt_ci(ci_fam[k])} |")
    if "search" in probes:
        ci = bootstrap(scores, y_val, y_test, f_test, "search", "best", args.n_boot, rng, by_family=False)
        report["original"]["bootstrap"]["best_vs_search"] = {"rows": ci}
        d = orig["best"]["auroc"] - orig["search"]["auroc"]
        lines.append(f"| best − search, auroc | {d:+.4f} | {fmt_ci(ci['auroc'])} | |")
    lines.append("")
    print(f"    done ({time.time() - t0:.0f}s)", flush=True)

    # 2. family re-splits
    rows = []
    for i in range(args.n_splits):
        perm = rng.permutation(present)
        val_f, test_f = sorted(map(str, perm[:7])), sorted(map(str, perm[7:14]))
        print(f"[2] re-split {i + 1}/{args.n_splits}: test={test_f}", flush=True)
        sc, yv, yt, ft = run_split(meta, info, arrays, probes, val_f, test_f)
        m = {name: metrics(s[0], yv, s[1], yt, ft) for name, s in sc.items()}
        row = {"split": i + 1, "test_families": ", ".join(test_f)}
        for name in probes:
            row[f"{name}_auroc"] = m[name]["auroc"]
            row[f"{name}_tpr@5"] = m[name]["tpr@5"]
        rows.append(row)
        print(f"    " + "  ".join(f"{n}={m[n]['auroc']:.4f}" for n in probes) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
    df = pd.DataFrame(rows)
    report["resplits"] = rows
    if len(df):
        lines += [f"## 2. {len(df)} random family re-splits (all probes refit on each)", "",
                  "| split | test families | " + " | ".join(f"{n} AUROC" for n in probes) + " | best − baseline |",
                  "|---|---|" + "---|" * len(probes) + "---|"]
        for r in rows:
            lines.append(f"| {r['split']} | {r['test_families']} | "
                         + " | ".join(f"{r[f'{n}_auroc']:.4f}" for n in probes)
                         + f" | {r['best_auroc'] - r['baseline_auroc']:+.4f} |")
        d = (df.best_auroc - df.baseline_auroc).values
        d5 = (df["best_tpr@5"] - df["baseline_tpr@5"]).values
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        summary = {"mean_diff_auroc": float(d.mean()), "se": float(se), "n_better": int((d > 0).sum()),
                   "n": int(len(d)), "mean_diff_tpr@5": float(d5.mean())}
        report["resplit_summary"] = summary
        lines += ["", f"Mean best − baseline: **{d.mean():+.4f}** macro-AUROC (± {1.96 * se:.4f}, ~95%), "
                      f"best better on **{(d > 0).sum()} of {len(d)}** splits; "
                      f"mean TPR @5% FPR change {d5.mean():+.3f}."]
        if "search" in probes:
            ds = (df.best_auroc - df.search_auroc).values
            lines.append(f"Mean best − search: {ds.mean():+.4f}, best better on {(ds > 0).sum()} of {len(ds)} splits.")
        lines.append("")

    lines += ["## Caveats", "",
              "- Check 1 reuses the original test families. Their scores were printed during the loop, so humans "
              "have seen them (Claude has not).",
              "- Re-splits move original val/test families into training; the iteration-9 idea was chosen on the "
              "original val families. This tests whether the idea transfers, not a sealed test.",
              f"- The conversations are the same {len(meta)} everywhere. New samples need a new GPU caching run.",
              "- The family-level interval has only ~6-7 clusters, so read it as rough.",
              "", f"_Runtime: {time.time() - t0:.0f} s._"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    json.dump(report, open(args.out + ".json", "w"), indent=2, default=float)
    print("\n".join(lines))
    print(f"\nWrote {args.out}.md and {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
