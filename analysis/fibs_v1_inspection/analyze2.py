"""Second pass on fibs-v1: family-level view, label-noise proxy, real group-split leakage baselines."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SRC = Path("/tmp/fibs_data")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
QUADS = ["honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive"]
COLS = ["source", "quadrant", "deception_type", "follow_up_category", "language", "pair_id", "pair_type",
        "judge_verdict", "judge_confidence", "blind_judge_verdict_primary", "blind_judge_verdict_tiebreaker",
        "blind_judge_confidence_primary", "an_generation_model", "completion", "_v32_source_content_md5"]
df = pd.concat([pq.read_table(SRC / f"{q}.parquet", columns=COLS).to_pandas() for q in QUADS], ignore_index=True)
df["y"] = df["quadrant"].str.split("_").str[1].map({"honest": 0, "deceptive": 1})
df["base"] = df["quadrant"].str.split("_").str[0].map({"honest": 0, "deceptive": 1})
df["chars"] = df["completion"].str.len()


def family(s: str) -> str:
    s = s.lower()
    for fam in ["persona_prefill", "doluschat", "longcontext", "negotiation", "werewolf", "sycophancy", "sandbagging",
                "liarsbench", "truthfulqa", "wikihownest", "wikihow", "ultrachat", "synthetic", "fabricated",
                "instructed", "b1_followup", "b1_bare", "b1-followup", "b1-bare", "caving", "medqa", "mask",
                "fever", "harm", "anti", "among", "lies", "model", "multi", "original"]:
        if s.startswith(fam):
            return fam.replace("_", "-")
    return re.split(r"[_:\-]", s)[0]


df["family"] = df["source"].map(family)
TRUE_SET = {"TRUTHFUL", "HONEST", "honest", "MAINTAIN", "RETRACT", "HOLD"}
LIE_SET = {"LIE", "DECEPTIVE", "UNTRUTHFUL", "deceptive", "CAVE"}
bv = df["blind_judge_verdict_primary"].fillna("")
df["blind_says_truthful"] = bv.isin(TRUE_SET)
df["blind_says_lie"] = bv.isin(LIE_SET)
df["blind_present"] = bv.isin(TRUE_SET | LIE_SET)
df["blind_disagrees"] = df["blind_present"] & (((df["y"] == 1) & df["blind_says_truthful"]) | ((df["y"] == 0) & df["blind_says_lie"]))

fam = df.groupby("family").agg(n=("y", "size"), pos=("y", "mean"), n_sources=("source", "nunique"),
                               blind_cov=("blind_present", "mean"),
                               med_chars=("chars", "median"))
dis = df[df.blind_present].groupby("family")["blind_disagrees"].mean().rename("blind_disagree_rate")
dis_pos = df[df.blind_present & (df.y == 1)].groupby("family")["blind_disagrees"].mean().rename("blind_disagree_on_pos")
dis_neg = df[df.blind_present & (df.y == 0)].groupby("family")["blind_disagrees"].mean().rename("blind_disagree_on_neg")
fam = fam.join([dis, dis_pos, dis_neg])
auc_len = {}
for f, d in df.groupby("family"):
    if d.y.nunique() == 2 and len(d) > 100:
        auc_len[f] = roc_auc_score(d.y, d.chars)
fam["auroc_len_within_family"] = pd.Series(auc_len)
fam = fam.sort_values("n", ascending=False).round(4)
fam.to_csv(OUT / "family_table.csv")
print(fam.to_string())

rep = {}
bp = df[df.blind_present]
rep["blind_coverage"] = float(df.blind_present.mean())
rep["blind_disagree_overall"] = float(bp.blind_disagrees.mean())
rep["blind_disagree_on_pos"] = float(bp[bp.y == 1].blind_disagrees.mean())
rep["blind_disagree_on_neg"] = float(bp[bp.y == 0].blind_disagrees.mean())
rep["n_families"] = int(df.family.nunique())

# metadata-only baselines, TRUE group split (GroupKFold by source = scenario; and by family)
y = df.y.values
def oof_prior(keys, groups, k=5):
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=k).split(df, y, groups):
        rate = df.iloc[tr].groupby(keys)["y"].mean()
        idx = pd.MultiIndex.from_frame(df.iloc[te][keys]) if len(keys) > 1 else df.iloc[te][keys[0]]
        pred[te] = rate.reindex(idx).fillna(y[tr].mean()).values
    return float(roc_auc_score(y, pred))
for gname, g in [("by_source", df.source.values), ("by_family", df.family.values)]:
    for name, keys in [("deception_type", ["deception_type"]), ("follow_up_category", ["follow_up_category"]),
                       ("family", ["family"]), ("family+followup", ["family", "follow_up_category"]),
                       ("generation_model", ["an_generation_model"])]:
        rep[f"auroc_metadata_prior|{name}|{gname}"] = oof_prior(keys, g)
rep["auroc_chars_overall"] = float(roc_auc_score(y, df.chars))

# length-matched check: AUROC of chars within quantile bins of chars
# pairs
pt = df.pair_type.fillna("").value_counts().head(10).to_dict()
rep["pair_types"] = pt
pairs = df[df.pair_id.fillna("") != ""].groupby("pair_id")["y"].agg(["size", "nunique", "mean"])
rep["pair_groups"] = int(len(pairs))
rep["pair_groups_size_dist"] = pairs["size"].value_counts().head(6).to_dict()
rep["pair_groups_with_both_labels"] = int((pairs["nunique"] == 2).sum())

# identical completion with different labels
g = df.groupby("completion")["y"].agg(["nunique", "size"])
both = g[g["nunique"] > 1]
rep["identical_completion_both_labels"] = int(len(both))
rep["rows_in_those"] = int(both["size"].sum())
bad = df[df.completion.isin(both.index)]
rep["identical_completion_both_labels_by_family"] = bad.family.value_counts().head(8).to_dict()

# base/followup interplay: how often does the follow-up label differ from base label
rep["followup_vs_base_agree_rate"] = float((df.y == df.base).mean())
rep["followup_label_by_base"] = df.groupby("base")["y"].mean().to_dict()
(OUT / "summary2.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
