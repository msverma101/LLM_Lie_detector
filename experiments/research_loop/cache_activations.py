"""Cache residual-stream activations of an open-weight model on a balanced FIBS sample.

This is the only expensive step (needs a GPU). Run it once; the research loop then
trains probes on the cached activations in seconds.

Output directory (--out):
  meta.csv               one row per example: family, source, quadrant, label, answer (first 300 chars), n_tokens
  acts_last.npy          float16 [N, L, d]  residual stream at the last token of the final answer
  acts_mean_answer.npy   float16 [N, L, d]  mean over the final assistant answer tokens
  acts_mean_turn.npy     float16 [N, L, d]  mean over the final user turn + assistant answer
  info.json              model, layers, dtype and sampling settings

label = 1 when the FINAL assistant answer (the reply to the follow-up question) is a
knowing falsehood, i.e. the quadrant ends in "_deceptive".

Layer k is the output of decoder block k (Hugging Face hidden_states[k + 1]), the same
convention as the published FIBS probes.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

SPLITS = ["honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive"]
COLS = ["source", "quadrant", "messages_json", "completion"]
FAMILIES = [
    "among", "anti", "b1-bare", "b1-followup", "caving", "doluschat", "fabricated", "fever",
    "harm", "instructed", "liarsbench", "lies", "longcontext", "mask", "medqa", "model",
    "multi", "negotiation", "original", "persona_prefill", "sandbagging", "sycophancy",
    "synthetic", "truthfulqa", "ultrachat", "werewolf", "wikihow", "wikihownest",
]
LONGCONTEXT_SUBSOURCES = ("v2", "v3_bookqa", "lc3flip", "lc5flip", "lcflip", "v5a_onpolicy",
                          "v5b_onpolicy", "v5c3_intrace", "v5c5_novels", "v4_offpolicy")
POOLS = ["last", "mean_answer", "mean_turn"]


def family_of(source: str) -> str:
    """Map a fine-grained FIBS source (e.g. 'harm_pressure_knowledge') to its family ('harm')."""
    matches = [f for f in FAMILIES if source == f or source.startswith(f + "_") or source.startswith(f + ":")]
    if matches:
        return max(matches, key=len)
    if source.startswith(LONGCONTEXT_SUBSOURCES):
        return "longcontext"
    return source  # unknown: kept as its own family (eval.py puts unknown families into train)


def load_candidates(n_per_family: int, seed: int, data_dir: str | None, max_chars: int) -> pd.DataFrame:
    """Sample up to 8 * n_per_family / 4 candidates per (family, quadrant); we keep n/4 after length filtering."""
    per_cell = max(1, n_per_family // 4)
    parts = []
    for split in SPLITS:
        if data_dir:
            path = os.path.join(data_dir, f"{split}.parquet")
        else:
            path = hf_hub_download("AlignmentResearch/fibs-v1", f"data/{split}-00000-of-00001.parquet",
                                   repo_type="dataset")
        df = pd.read_parquet(path, columns=COLS)
        df["family"] = df["source"].map(family_of)
        # cheap prefilter: rows far too long to fit max_tokens are never tokenized
        df = df[df["messages_json"].str.len() + df["completion"].fillna("").str.len() < max_chars]
        for _, g in df.groupby("family"):
            parts.append(g.sample(n=min(len(g), 8 * per_cell), random_state=seed))
        del df
    out = pd.concat(parts, ignore_index=True)
    out["label"] = out["quadrant"].str.endswith("_deceptive").astype(int)
    return out


def to_chat(messages):
    """Keep role + text content; flatten list contents; drop sidecar fields."""
    chat = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if content is None:
            content = ""
        if m.get("tool_calls"):
            content = (content + "\n" + json.dumps(m["tool_calls"])).strip()
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        chat.append({"role": role, "content": content})
    return chat


def render(tok, row, end_token):
    """Return (token ids, answer_start, turn_start, last_idx) or None if the row can't be rendered."""
    chat = to_chat(json.loads(row.messages_json))
    if not chat or chat[-1]["role"] == "assistant":  # must end with a user turn or a tool result
        return None
    completion = (row.completion or "").strip()
    if not completion:
        return None
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        prompt = tok.apply_chat_template(chat, enable_thinking=False, **kwargs)
    except TypeError:
        prompt = tok.apply_chat_template(chat, **kwargs)
    full = prompt + completion + end_token
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    # final turn = the last message before the assistant header (a user question or a tool result)
    header = prompt.rfind("<|im_start|>assistant")
    turn_char = prompt.rfind("<|im_start|>", 0, header) if header > 0 else -1
    answer_char, answer_end_char = len(prompt), len(prompt) + len(completion)

    def first_token_at(c):
        for i, (s, e) in enumerate(offs):
            if e > c:
                return i
        return len(ids) - 1

    answer_start = first_token_at(answer_char)
    turn_start = first_token_at(max(turn_char, 0)) if turn_char >= 0 else answer_start
    last_idx = max(i for i, (s, e) in enumerate(offs) if s < answer_end_char)
    if last_idx < answer_start:
        return None
    return ids, answer_start, turn_start, last_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--layers", default="7,10,13,16,19,22", help="decoder blocks to read (0-indexed)")
    ap.add_argument("--n_per_family", type=int, default=120, help="examples per family, balanced over the 4 quadrants")
    ap.add_argument("--max_tokens", type=int, default=4096, help="skip rows longer than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--data_dir", default=None, help="local parquet files <split>.parquet (for testing)")
    ap.add_argument("--out", default="acts")
    ap.add_argument("--chunk", type=int, default=200, help="rows per checkpoint file (resumable)")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    end_token = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else (tok.eos_token or "")

    # 1) pick rows: balanced per family x quadrant, short enough
    cand = load_candidates(args.n_per_family, args.seed, args.data_dir, max_chars=args.max_tokens * 5)
    per_cell = max(1, args.n_per_family // 4)
    rendered, keep_rows = [], []
    for (fam, quad), g in cand.groupby(["family", "quadrant"], sort=True):
        kept = 0
        for row in g.itertuples():
            if kept >= per_cell:
                break
            try:
                r = render(tok, row, end_token)
            except Exception as e:  # malformed row or template error
                r = None
            if r is None or len(r[0]) > args.max_tokens:
                continue
            rendered.append(r)
            keep_rows.append(row.Index)
            kept += 1
    meta = cand.loc[keep_rows, ["family", "source", "quadrant", "label"]].reset_index(drop=True)
    # short snippet of the final answer, so errors.py can show concrete misses / false alarms
    meta["answer"] = cand.loc[keep_rows, "completion"].fillna("").str.slice(0, 300).str.replace(r"\s+", " ", regex=True).values
    meta["n_tokens"] = [len(r[0]) for r in rendered]
    print(f"selected {len(meta)} rows from {meta.family.nunique()} families; "
          f"median {int(meta.n_tokens.median())} tokens", flush=True)
    print(meta.groupby("family").size().to_string(), flush=True)

    # 2) forward passes, checkpointed per chunk
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    assert max(layers) < n_layers, f"model has {n_layers} blocks"

    t0, done = time.time(), 0
    n_chunks = (len(rendered) + args.chunk - 1) // args.chunk
    for c in range(n_chunks):
        path = os.path.join(args.out, f"part_{c:04d}.npz")
        lo, hi = c * args.chunk, min(len(rendered), (c + 1) * args.chunk)
        if os.path.exists(path):
            done += hi - lo
            continue
        feats = {p: [] for p in POOLS}
        for i in range(lo, hi):
            ids, a0, t0_, last = rendered[i]
            x = torch.tensor([ids], device=device)
            with torch.inference_mode():
                try:
                    out = model(input_ids=x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                except TypeError:
                    out = model(input_ids=x, output_hidden_states=True, use_cache=False)
            hs = torch.stack([out.hidden_states[k + 1][0] for k in layers]).float()  # [L, seq, d]
            feats["last"].append(hs[:, last].cpu().numpy())
            feats["mean_answer"].append(hs[:, a0:last + 1].mean(1).cpu().numpy())
            feats["mean_turn"].append(hs[:, t0_:last + 1].mean(1).cpu().numpy())
            done += 1
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(rendered)} rows  {rate:.2f} rows/s  "
                      f"ETA {(len(rendered) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        np.savez(path, **{p: np.clip(np.stack(v), -65000, 65000).astype(np.float16) for p, v in feats.items()})

    # 3) merge
    for p in POOLS:
        arr = np.concatenate([np.load(os.path.join(args.out, f"part_{c:04d}.npz"))[p] for c in range(n_chunks)])
        bad = ~np.isfinite(arr.astype(np.float32)).all(axis=(1, 2))
        if bad.any():
            print(f"WARNING: {bad.sum()} rows with non-finite '{p}' activations - consider --dtype float32")
        np.save(os.path.join(args.out, f"acts_{p}.npy"), arr)
    meta.to_csv(os.path.join(args.out, "meta.csv"), index=False)
    json.dump({"model": args.model, "layers": layers, "dtype": str(dtype), "n_per_family": args.n_per_family,
               "max_tokens": args.max_tokens, "seed": args.seed, "pools": POOLS, "n": len(meta),
               "d_model": int(arr.shape[-1])}, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print(f"done: {len(meta)} rows in {(time.time() - t0) / 60:.1f} min -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
