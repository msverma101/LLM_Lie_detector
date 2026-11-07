"""Fresh, pinned FIBS capture. Never truncates a transcript to fit the token limit."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"
MODEL_REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA_REV = "4ad7414fc2e1cc8d694b160e360722a13bee718e"
SEED = 20261107
LAYERS = [7, 10, 13, 16, 19, 22]
PARTS = {
    "fit": ["anti", "doluschat", "fever", "instructed", "lies", "multi", "persona_prefill", "synthetic", "wikihow", "wikihownest"],
    "inspect": ["liarsbench", "mask", "original", "ultrachat"],
    "selection": ["b1-bare", "b1-followup", "caving", "negotiation"],
    "target": ["among", "harm", "medqa", "model", "sandbagging", "werewolf"],
}
QUADS = ["honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive"]


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def family(source):
    matches = [f for fs in PARTS.values() for f in fs if source == f or source.startswith(f + "_") or source.startswith(f + ":")]
    return max(matches, key=len) if matches else "excluded_family"


def chat_render(tok, row):
    messages = json.loads(row.messages_json)
    chat = []
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if m.get("tool_calls"):
            content = (content + "\n" + json.dumps(m["tool_calls"])).strip()
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unsupported role")
        chat.append({"role": role, "content": content})
    completion = str(row.completion or "").strip()
    if not chat or chat[-1]["role"] == "assistant" or not completion:
        raise ValueError("invalid final answer boundary")
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = prompt + completion + "<|im_end|>"
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    header = prompt.rfind("<|im_start|>assistant")
    turn_char = prompt.rfind("<|im_start|>", 0, header)
    first = lambda c: next(i for i, (_, end) in enumerate(offsets) if end > c)
    answer_start = first(len(prompt))
    turn_start = first(max(turn_char, 0))
    last = max(i for i, (start, _) in enumerate(offsets) if start < len(prompt) + len(completion))
    assert last >= answer_start
    return enc["input_ids"], answer_start, turn_start, last


def sample(tok, n, max_tokens):
    frames = []
    for quad in QUADS:
        path = hf_hub_download("AlignmentResearch/fibs-v1", f"data/{quad}-00000-of-00001.parquet", repo_type="dataset", revision=DATA_REV)
        df = pd.read_parquet(path, columns=["source", "quadrant", "pair_id", "messages_json", "completion", "language", "follow_up_category", "human_verified"])
        df["dataset_split"], df["dataset_row"] = quad, np.arange(len(df))
        df["family"] = df.source.map(family)
        # This filter is a computational eligibility restriction, not truncation.
        df = df[(df.family != "excluded_family") & (df.messages_json.str.len() + df.completion.fillna("").str.len() <= max_tokens * 6)].copy()
        df["row_id"] = [digest(DATA_REV + ":" + quad + ":" + str(i)) for i in df.dataset_row]
        groups = []
        for r in df.itertuples():
            if pd.notna(r.pair_id) and str(r.pair_id).strip():
                groups.append(digest("pair:" + str(r.pair_id)))
            else:
                msgs = json.loads(r.messages_json)
                first_user = next((m.get("content") for m in msgs if m.get("role") == "user"), r.messages_json)
                groups.append(digest("fallback:" + json.dumps(first_user, sort_keys=True)))
        df["group_id"] = groups
        df["draw_order"] = [digest(str(SEED) + g + rid) for g, rid in zip(df.group_id, df.row_id)]
        frames.append(df)
        print("DATA_LOADED", quad, len(df), flush=True)
    data = pd.concat(frames, ignore_index=True)
    data["label"] = data.quadrant.str.endswith("_deceptive").astype(int)
    mapping = {f: p for p, fs in PARTS.items() for f in fs}
    data["partition"] = data.family.map(mapping)
    target = data.partition == "target"
    calibration_group = data.group_id.map(lambda g: int(g[:12], 16) % 5 == 0)
    data.loc[target & calibration_group, "partition"] = "calib"
    data.loc[target & ~calibration_group, "partition"] = "test"
    data = data[~((data.partition == "calib") & (data.label != 0))].copy()
    # Drop ambiguous group membership rather than allowing a source-family shortcut to leak pairs.
    clashes = data.groupby("group_id").partition.nunique()
    bad = set(clashes[clashes > 1].index)
    data = data[~data.group_id.isin(bad)]
    selected, tokens, rejected = [], [], {"over_length": 0, "render_error": 0, "conflicting_groups": len(bad)}
    per_quad = n // 4
    for (part, fam, quad), group in data.groupby(["partition", "family", "quadrant"], sort=True):
        kept = 0
        for r in group.sort_values("draw_order").itertuples():
            if kept >= per_quad:
                break
            try:
                rendered = chat_render(tok, r)
            except Exception:
                rejected["render_error"] += 1
                continue
            if len(rendered[0]) > max_tokens:
                rejected["over_length"] += 1
                continue
            selected.append(data.loc[r.Index].to_dict())
            tokens.append(rendered)
            kept += 1
    result = pd.DataFrame(selected).reset_index(drop=True)
    assert not (result.groupby("group_id").partition.nunique() > 1).any()
    assert not result.duplicated(["messages_json", "completion"]).any(), "duplicate transcripts require explicit grouping"
    return result, tokens, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-family", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "GPU required; no silent slow CPU fallback"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    meta, rendered, rejected = sample(tok, args.n_per_family, args.max_tokens)
    meta["n_tokens"] = [len(x[0]) for x in rendered]
    meta["answer_tokens"] = [x[3] - x[1] + 1 for x in rendered]
    manifest = {"model": MODEL, "model_revision": MODEL_REV, "dataset_revision": DATA_REV, "seed": SEED, "layers": LAYERS,
                "max_tokens": args.max_tokens, "dtype": "float16", "sampling": "stable hash, balanced quadrant caps; no truncation",
                "calibration": "honest target-domain groups with group hash modulo 5 = 0; disjoint from test",
                "historical_independence": "not established: prior exact row IDs unavailable; family identities previously discussed",
                "label_scope": "FIBS labels on supplied completions; no assertion about Qwen2B subjective belief or intent",
                "render_contract": "text-only ChatML adaptation, tools serialized; reasoning sidecars omitted; not official EFC replay",
                "rejected": rejected, "counts": meta.groupby(["partition", "family", "quadrant"]).size().to_dict().__str__()}
    for part in meta.partition.unique():
        block = meta[meta.partition == part]
        block.to_json(root / f"cases_{part}.jsonl", orient="records", lines=True, force_ascii=False)
    meta.drop(columns=["messages_json", "completion"]).to_csv(root / "meta.csv", index=False)
    (root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAMPLE", meta.groupby(["partition", "family"]).size().to_json(), "REJECTED", rejected, flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.float16).cuda().eval()
    start = time.monotonic()
    pools = {k: [] for k in ("last", "mean_answer", "mean_turn")}
    for i, (ids, a0, t0, last) in enumerate(rendered):
        checkpoint = root / f"row_{i:05d}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                assert saved["row_id"].item() == meta.iloc[i].row_id
                for p in pools:
                    pools[p].append(saved[p])
            continue
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            output = model(input_ids=x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        h = torch.stack([output.hidden_states[k + 1][0] for k in LAYERS]).float()
        row = {"last": h[:, last], "mean_answer": h[:, a0:last + 1].mean(1), "mean_turn": h[:, t0:last + 1].mean(1)}
        stored = {p: z.cpu().numpy().astype(np.float16) for p, z in row.items()}
        assert all(np.isfinite(z).all() for z in stored.values()), "non-finite activations; stop rather than silently repair"
        np.savez(checkpoint, row_id=meta.iloc[i].row_id, **stored)
        for p in pools:
            pools[p].append(stored[p])
        del output, h, x
        if (i + 1) % 20 == 0:
            seconds = time.monotonic() - start
            print("CAPTURE", i + 1, "/", len(meta), "seconds", round(seconds, 1), "eta_min", round(seconds / (i + 1) * (len(meta) - i - 1) / 60, 1), flush=True)
    for p, values in pools.items():
        np.save(root / f"acts_{p}.npy", np.stack(values))
    hashes = {}
    for p in sorted(root.glob("*")):
        if p.is_file() and not p.name.startswith("row_"):
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "CAPTURE_SHA256.json").write_text(json.dumps(hashes, indent=2))
    print("CAPTURE_DONE", len(meta), "rows", round((time.monotonic() - start) / 60, 2), "minutes", flush=True)


if __name__ == "__main__":
    main()
