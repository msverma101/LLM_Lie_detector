"""Adapter to the published fibs-probes (probe_inference) + a parity check against FIBSv1 rows.

Parity check: run FIBS rows through OUR capture (HF, chat template, thinking off) and score them with the
published probe. FIBS rows are in-distribution for these probes, so a working activation contract must
reproduce a high AUROC (published validation AUROC: linear 0.926 [authors-reported]). A value near 0.5 means
our activations do not match what the probes were trained on.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import os as _os
ARCHS = tuple(_os.environ.get("LIEDET_ARCHS", "linear,mlp,efc,axial").split(","))


def load_published(arch: str, model: str = "qwen3.5-2b"):
    """Published probe from the pinned Hub revision. LIEDET_PROBE_ROOT points at a local directory with the same
    layout (used by the tests to run the real pipeline against synthetic probes of a tiny model)."""
    import os
    root = os.environ.get("LIEDET_PROBE_ROOT")
    if root:
        from probe_inference import load_probe
        return load_probe(f"{root}/{arch}")
    from probe_inference import load_probe_from_hub
    return load_probe_from_hub(f"{model}/{arch}")


def format_fibs_row(tokenizer, messages: list[dict], completion: str, max_tokens: int = 1500):
    """ids, window_start, window_end(excl), last_idx for a FIBS row: history + final user turn + completion.

    Window = from the start of the final user turn to the assistant's end-of-turn token (the contract of the
    efc/axial probes); last_idx = the token before end-of-turn (linear / mlp)."""
    msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])}
            for m in messages]
    text = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": completion}],
                                         tokenize=False, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = max(i for i, t in enumerate(ids) if t == im_end) + 1
    starts = [i for i, t in enumerate(ids[:end]) if t == im_start]
    start = starts[-2]              # [-1] = final assistant turn, [-2] = final user turn
    if end > max_tokens:
        raise ValueError("row too long")
    return ids[:end], start, end, end - 2


def score_published(probe, tok_acts: np.ndarray, layers: tuple[int, ...]) -> float:
    """Score ONE row. tok_acts: [W, L, d] (window tokens, our layer order == probe.layers)."""
    assert tuple(layers) == tuple(probe.layers), "layer mismatch between capture and probe"
    x = torch.as_tensor(tok_acts.astype(np.float32))            # [W, L, d]
    W = x.shape[0]
    if probe.read_window == "second-last-token-generation":
        acts = {l: x[W - 2, i][None, :] for i, l in enumerate(probe.layers)}
        mask = torch.ones(1, dtype=torch.bool)
    else:
        acts = x.permute(1, 0, 2).contiguous()                   # [L, W, d]
        mask = torch.ones(W, dtype=torch.bool)
    return float(probe.score(acts, mask))


def parity_sample(parquet_dir: str, n_per_class: int, seed: int, max_chars: int = 3000):
    """English FIBS rows, balanced on the follow-up label, spread over families. Returns a list of dicts."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    out = []
    for quad in ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive"):
        t = pq.read_table(f"{parquet_dir}/{quad}.parquet",
                          columns=["source", "language", "completion", "messages_json"]).to_pandas()
        t = t[(t.language == "en") & (t.completion.str.len() < max_chars) & (t.messages_json.str.len() < max_chars * 2)]
        fam = t.source.str.split(":").str[0]
        # one pass per family round-robin so the sample is not dominated by the big families
        picks = []
        groups = {f: g.index.to_numpy().copy() for f, g in t.groupby(fam)}
        for f in groups:
            rng.shuffle(groups[f])
        while len(picks) < n_per_class // 2 and any(len(v) for v in groups.values()):
            for f in list(groups):
                if groups[f].size:
                    picks.append(groups[f][0]); groups[f] = groups[f][1:]
                if len(picks) >= n_per_class // 2:
                    break
        label = 1 if quad.endswith("deceptive") else 0
        for i in picks:
            r = t.loc[i]
            out.append({"label": label, "quadrant": quad, "family": str(r.source).split(":")[0],
                        "messages": json.loads(r.messages_json), "completion": r.completion})
    return out


@torch.no_grad()
def window_acts(cap, ids, start, end):
    """[W, L, d] float16 window activations for one tokenised row (block outputs, no intervention)."""
    out = cap._forward(ids)
    hs = torch.stack([out.hidden_states[l + 1][0, start:end] for l in cap.layers]).float()   # [L, W, d]
    return hs.permute(1, 0, 2).to(torch.float16).numpy()


def parity_check(cap, sample: list[dict], archs=ARCHS, progress=None) -> dict:
    from . import evalkit as ev
    probes = {a: load_published(a) for a in archs}
    scores = {a: [] for a in archs}; fams, labels, kept = [], [], 0
    for n, r in enumerate(sample):
        try:
            ids, s, e, last = format_fibs_row(cap.tokenizer, r["messages"], r["completion"])
        except ValueError:
            continue
        acts = window_acts(cap, ids, s, e)
        for a, p in probes.items():
            scores[a].append(score_published(p, acts, cap.layers))
        fams.append(r["family"]); labels.append(r["label"]); kept += 1
        if progress and n % 20 == 0:
            progress(n, len(sample))
    y = np.array(labels); fam = np.array(fams)
    res = {"n_rows": kept, "n_pos": int(y.sum()), "archs": {}}
    for a in archs:
        s = np.array(scores[a])
        res["archs"][a] = {"auroc": ev.auroc(s[y == 1], s[y == 0]),
                           "by_family": {f: ev.auroc(s[(fam == f) & (y == 1)], s[(fam == f) & (y == 0)])
                                         for f in sorted(set(fam)) if ((fam == f) & (y == 1)).sum() >= 5 and ((fam == f) & (y == 0)).sum() >= 5},
                           "score_mean_pos": float(s[y == 1].mean()), "score_mean_neg": float(s[y == 0].mean())}
    res["published_validation_auroc_qwen3.5-2b"] = {"linear": 0.926, "mlp": 0.968, "efc": 0.987, "axial": 0.991,
                                                    "source": "fibs-probes README, authors-reported, in-distribution"}
    return res
