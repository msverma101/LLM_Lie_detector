"""Capture new recorded groups for the preregistered, frozen-LLM skill study."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_wsp"))
from capture import Union, aliases
from capture_base import MODEL, MODEL_REV, DATA_REV, LAYERS, PARTS, QUADS, family, chat_render, digest

SEED = 202611082
ROLES = ["diagnostic", "val_calib", "validation", "calib", "test"]
CAPS = {"diagnostic": 50, "val_calib": 50, "validation": 50, "calib": 100, "test": 100}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous", required=True, nargs="+")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "meta.csv").exists(), "Do not overwrite an existing capture"
    previous = pd.concat([pd.read_csv(p).fillna("") for p in args.previous])
    old_rows, old_groups = set(previous.row_id), set(previous.group_id)
    frames, rejected = [], {"unmapped_family": 0, "char_prefilter": 0,
                            "examined_over_length": 0, "examined_render_error": 0}
    for quad in QUADS:
        path = hf_hub_download("AlignmentResearch/fibs-v1", f"data/{quad}-00000-of-00001.parquet",
                               repo_type="dataset", revision=DATA_REV)
        d = pd.read_parquet(path, columns=["source", "quadrant", "pair_id", "messages_json",
                                          "completion", "language", "follow_up_category", "human_verified"])
        d["dataset_split"], d["dataset_row"] = quad, np.arange(len(d))
        d["family"] = d.source.map(family)
        rejected["unmapped_family"] += int((d.family == "excluded_family").sum())
        d = d[d.family != "excluded_family"].copy()
        long = (d.messages_json.str.len() + d.completion.fillna("").str.len()) > 2048 * 8
        rejected["char_prefilter"] += int(long.sum())
        d = d[~long].copy()
        d["row_id"] = [digest(DATA_REV + ":" + quad + ":" + str(i)) for i in d.dataset_row]
        frames.append(d)
        print("LOADED", quad, len(d), flush=True)
    d = pd.concat(frames, ignore_index=True)
    uf, keys_per_row, banned = Union(), [], set()
    for r in d.itertuples():
        keys, old = aliases(r)
        for key in keys[1:]:
            uf.join(keys[0], key)
        keys_per_row.append(keys)
        if r.row_id in old_rows or any(key in old_groups for key in old):
            banned.update(keys)
    banned_roots = {uf.find(key) for key in banned}
    d["group_id"] = [digest("component:" + uf.find(keys[0])) for keys in keys_per_row]
    banned_ids = {digest("component:" + key) for key in banned_roots}
    excluded = d.group_id.isin(banned_ids)
    rejected["previous_connected_rows"] = int(excluded.sum())
    d = d[~excluded].copy()
    source_parts = {f: p for p, fs in PARTS.items() for f in fs}
    d["source_partition"] = d.family.map(source_parts)
    clash = d.groupby("group_id").source_partition.nunique()
    bad = set(clash[clash > 1].index)
    rejected["cross_source_partition_groups"] = len(bad)
    d = d[~d.group_id.isin(bad) & (d.source_partition == "target")].copy()
    d["label"] = d.quadrant.str.endswith("_deceptive").astype(int)
    d["partition"] = d.group_id.map(lambda g: ROLES[int(digest(str(SEED) + g)[:12], 16) % 5])
    unused = d.partition.isin(["val_calib", "calib"]) & (d.label == 1)
    rejected["unused_calibration_positives"] = int(unused.sum())
    d = d[~unused].copy()
    d["draw_order"] = [digest(str(SEED) + g + rid) for g, rid in zip(d.group_id, d.row_id)]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    chosen, rendered = [], []
    for (part, fam, label), block in d.groupby(["partition", "family", "label"], sort=True):
        queues = [list(b.sort_values("draw_order").index) for _, b in block.groupby("quadrant", sort=True)]
        kept = 0
        while any(queues) and kept < CAPS[part]:
            for queue in queues:
                if not queue or kept >= CAPS[part]:
                    continue
                row = d.loc[queue.pop(0)]
                try:
                    ids = chat_render(tok, row)
                except Exception:
                    rejected["examined_render_error"] += 1
                    continue
                if len(ids[0]) > 2048:
                    rejected["examined_over_length"] += 1
                    continue
                chosen.append(row.to_dict())
                rendered.append(ids)
                kept += 1
        print("CELL", part, fam, label, kept, flush=True)
    meta = pd.DataFrame(chosen).reset_index(drop=True)
    assert meta.groupby("group_id").partition.nunique().max() == 1
    assert not set(meta.row_id) & old_rows
    assert not set(meta.group_id) & set(previous.group_id)
    assert not meta.duplicated(["messages_json", "completion"]).any()
    meta["n_tokens"] = [len(x[0]) for x in rendered]
    meta["answer_tokens"] = [x[3] - x[1] + 1 for x in rendered]
    for part, block in meta.groupby("partition"):
        block.to_json(root / f"cases_{part}.jsonl", orient="records", lines=True, force_ascii=False)
    meta.drop(columns=["messages_json", "completion"]).to_csv(root / "meta.csv", index=False)
    manifest = {"date": "2026-11-08", "seed": SEED, "model": MODEL, "model_revision": MODEL_REV,
                "dataset_revision": DATA_REV, "layers": LAYERS, "pool": "last_answer_token",
                "max_tokens": 2048, "previous_files": [{"path": p, "sha256": sha(p)} for p in args.previous],
                "rejections": rejected, "counts": meta.groupby(["partition", "family", "label"]).size()
                .rename("n").reset_index().to_dict("records"), "previous_row_overlap": 0,
                "previous_group_overlap": 0, "cross_role_group_overlap": 0,
                "scope": "Familiar families, recorded-group independence only; fixed text-only adaptation"}
    (root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAMPLE_READY", len(meta), json.dumps(rejected), flush=True)
    assert torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.float16).cuda().eval()
    hidden = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    out = np.lib.format.open_memmap(root / "acts_last.npy", mode="w+", dtype=np.float16,
                                    shape=(len(meta), len(LAYERS), hidden))
    start = time.monotonic()
    for i, (ids, _, _, last) in enumerate(rendered):
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            result = model(input_ids=x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        h = torch.stack([result.hidden_states[k + 1][0, last] for k in LAYERS]).float().cpu().numpy().astype(np.float16)
        assert np.isfinite(h).all()
        out[i] = h
        del result, x, h
        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - start
            print("CAPTURE", i + 1, len(meta), "minutes", round(elapsed / 60, 2),
                  "eta_minutes", round(elapsed / (i + 1) * (len(meta) - i - 1) / 60, 2), flush=True)
            out.flush()
    out.flush()
    del out, model
    torch.cuda.empty_cache()
    hashes = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
    (root / "CAPTURE_DONE.json").write_text(json.dumps({"rows": len(meta), "minutes":
        (time.monotonic() - start) / 60, "sha256": hashes}, indent=2))
    print("CAPTURE_DONE", len(meta), flush=True)


if __name__ == "__main__":
    main()
