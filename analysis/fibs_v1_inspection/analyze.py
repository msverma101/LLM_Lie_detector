"""Inspect AlignmentResearch/fibs-v1 metadata: schema, label distribution, leakage shortcuts.

Reads only metadata columns plus a parsed message count. Writes small JSON/CSV summaries.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

SRC = Path("/tmp/fibs_data")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
QUADS = ["honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive"]
DROP = {"messages_json", "tools", "completion_reasoning", "completion", "judge_justification",
        "blind_judge_justification_primary", "blind_judge_justification_tiebreaker",
        "an_original_completion"}

frames = []
for q in QUADS:
    pf = pq.ParquetFile(SRC / f"{q}.parquet")
    cols = [c for c in pf.schema_arrow.names if c not in DROP] + ["completion"]
    df = pf.read(columns=cols).to_pandas()
    df["completion_chars"] = df["completion"].str.len()
    df["completion_words"] = df["completion"].str.split().str.len()
    # message structure from messages_json, read separately in batches to bound memory
    n_msgs, has_sys, ctx_chars = [], [], []
    for batch in pf.iter_batches(columns=["messages_json"], batch_size=2000):
        for s in batch.column(0).to_pylist():
            try:
                m = json.loads(s)
            except Exception:
                n_msgs.append(-1); has_sys.append(False); ctx_chars.append(-1); continue
            n_msgs.append(len(m))
            has_sys.append(any(x.get("role") == "system" for x in m))
            ctx_chars.append(len(s))
    df["n_messages"] = n_msgs
    df["has_system"] = has_sys
    df["messages_json_chars"] = ctx_chars
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df["base_label"] = df["quadrant"].str.split("_").str[0].map({"honest": 0, "deceptive": 1})
df["followup_label"] = df["quadrant"].str.split("_").str[1].map({"honest": 0, "deceptive": 1})
rep = {}
rep["n_rows"] = int(len(df))
rep["rows_per_quadrant"] = df["quadrant"].value_counts().to_dict()
rep["followup_label_counts"] = df["followup_label"].value_counts().to_dict()
rep["base_label_counts"] = df["base_label"].value_counts().to_dict()
rep["n_sources"] = int(df["source"].nunique())
rep["missing_frac_top"] = df.isna().mean().sort_values(ascending=False).head(8).round(4).to_dict()

# label rate per source (followup)
src = df.groupby("source").agg(n=("followup_label", "size"), pos=("followup_label", "mean"),
                               base_pos=("base_label", "mean"),
                               med_chars=("completion_chars", "median")).sort_values("n", ascending=False)
src.round(4).to_csv(OUT / "per_source.csv")
rep["sources_all_one_class"] = src[(src.pos == 0) | (src.pos == 1)].index.tolist()
rep["sources_pos_rate_outside_0.2_0.8"] = src[(src.pos < 0.2) | (src.pos > 0.8)].index.tolist()

# judge vs label
def agree(col):
    x = df[df[col].isin(["TRUTHFUL", "DECEPTIVE", "FALSE", "LIE"]) | df[col].notna()]
    return x.groupby(["followup_label", col]).size().to_dict()
rep["judge_verdict_x_followup_label"] = {f"{a}|{b}": int(c) for (a, b), c in
                                          df.groupby(["followup_label", "judge_verdict"], dropna=False).size().items()}
rep["blind_judge_x_followup_label"] = {f"{a}|{b}": int(c) for (a, b), c in
                                        df.groupby(["followup_label", "blind_judge_verdict_primary"], dropna=False).size().items()}
rep["human_verified_by_quadrant"] = df.groupby("quadrant")["human_verified"].mean().round(4).to_dict()
rep["human_verified_n"] = int(df["human_verified"].sum())
rep["judge_confidence_x_label"] = {f"{a}|{b}": int(c) for (a, b), c in
                                    df.groupby(["followup_label", "judge_confidence"], dropna=False).size().items()}

for col in ["deception_type", "follow_up_category", "language", "an_generation_model", "an_generation_method",
            "pair_type", "reasoning_in_context", "has_system"]:
    t = df.groupby([col, "followup_label"], dropna=False).size().unstack(fill_value=0)
    t["pos_rate"] = (t.get(1, 0) / (t.get(0, 0) + t.get(1, 0))).round(4)
    t.to_csv(OUT / f"by_{col}.csv")

# leakage shortcuts: single feature AUROCs (followup label)
def auc(x, y):
    x = pd.Series(x)
    ok = x.notna()
    return float(roc_auc_score(y[ok], x[ok])) if y[ok].nunique() == 2 else None
y = df["followup_label"]
rep["auroc_completion_chars"] = auc(df["completion_chars"], y)
rep["auroc_completion_words"] = auc(df["completion_words"], y)
rep["auroc_n_messages"] = auc(df["n_messages"], y)
rep["auroc_messages_json_chars"] = auc(df["messages_json_chars"], y)

# grouped, out-of-fold source prior: leave-group-out by source_content_md5
g = df["_v32_source_content_md5"].fillna("na")
rng = np.random.default_rng(0)
uniq = g.unique()
fold_of = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
fold = g.map(fold_of).values
for name, key in [("source", ["source"]), ("source+followup_category", ["source", "follow_up_category"]),
                  ("deception_type", ["deception_type"]), ("followup_category", ["follow_up_category"])]:
    pred = np.zeros(len(df))
    for f in range(5):
        tr = df[fold != f]
        rate = tr.groupby(key)["followup_label"].mean()
        te = df[fold == f]
        k = te[key].apply(tuple, axis=1) if len(key) > 1 else te[key[0]]
        idx = pd.MultiIndex.from_frame(te[key]) if len(key) > 1 else te[key[0]]
        pred[fold == f] = rate.reindex(idx).fillna(tr["followup_label"].mean()).values
    rep[f"auroc_prior_by_{name}_groupcv"] = float(roc_auc_score(y.values, pred))

# duplicates / contamination
rep["dup_completion_rows"] = int(df.duplicated("completion", keep=False).sum())
dup = df.groupby("completion")["followup_label"].nunique()
rep["completions_with_both_labels"] = int((dup > 1).sum())
md5 = df.groupby("_v32_source_content_md5")["followup_label"].agg(["nunique", "size"])
rep["md5_groups"] = int(len(md5))
rep["md5_groups_size_gt1"] = int((md5["size"] > 1).sum())
rep["md5_groups_with_both_labels"] = int((md5["nunique"] > 1).sum())
rep["pair_id_nonempty"] = int((df["pair_id"].fillna("") != "").sum())
rep["lane_attributed"] = df["_v32_lane_attributed"].value_counts().head(10).to_dict()
rep["completion_chars_by_label"] = df.groupby("followup_label")["completion_chars"].describe().round(1).to_dict("index")
rep["lang_counts"] = df["language"].value_counts().head(12).to_dict()

# per-source single-feature length AUROC where both classes exist (shortcut within source)
rows = []
for s, d in df.groupby("source"):
    if d["followup_label"].nunique() == 2 and len(d) >= 200:
        rows.append((s, len(d), roc_auc_score(d["followup_label"], d["completion_chars"])))
pd.DataFrame(rows, columns=["source", "n", "auroc_chars"]).sort_values("auroc_chars").round(4).to_csv(OUT / "length_auroc_per_source.csv", index=False)
(OUT / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str)[:9000])
