"""Tokenisation with exact read windows, and residual-stream capture.

Activation contract (mirrors fibs-probes where we know it): the output of decoder block ``l``
(``hidden_states[l+1]``), thinking disabled (empty think block in the assistant turn), read window =
every token from the start of the final user turn through the final assistant end-of-turn token.
The last block is refused because HF applies the final norm to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import FOLLOWUP_A_TRUE, FOLLOWUP_Q, Row

DEFAULT_FRACTIONS = (0.30, 0.42, 0.55, 0.67, 0.80, 0.90)
THINK_OFF = "<think>\n\n</think>\n\n"


def default_layers(n_layers: int) -> tuple[int, ...]:
    # the last block is never usable (HF applies the final norm to its hidden state), so clip to n-2
    return tuple(sorted({min(int(round(f * n_layers)), n_layers - 2) for f in DEFAULT_FRACTIONS}))


@dataclass
class Encoded:
    ids: list[int]
    # variant -> (window_start, window_end_exclusive, last_idx) ; last_idx = token before the final <|im_end|>
    spans: dict[str, tuple[int, int, int]]


def _tok(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_cell(tokenizer, system: str, user: str, assistant: str,
                fq: str = FOLLOWUP_Q, fa: str = FOLLOWUP_A_TRUE) -> Encoded:
    """Encode system/user/assistant plus the follow-up turn; return spans for both variants.

    Because the model is causal, the activations inside the 'answer' span of this longer sequence equal
    those of the truncated sequence (asserted in tests), so one forward pass serves both variants.
    """
    ids: list[int] = []
    marks: dict[str, int] = {}

    def add(text: str, mark: str | None = None):
        if mark:
            marks[mark] = len(ids)
        ids.extend(_tok(tokenizer, text))

    add(f"<|im_start|>system\n{system}<|im_end|>\n")
    add(f"<|im_start|>user\n{user}<|im_end|>\n", "u1")
    add(f"<|im_start|>assistant\n{THINK_OFF}{assistant}", "a1")
    add("<|im_end|>")
    marks["a1_end"] = len(ids)
    add("\n")
    add(f"<|im_start|>user\n{fq}<|im_end|>\n", "u2")
    add(f"<|im_start|>assistant\n{THINK_OFF}{fa}", "a2")
    add("<|im_end|>")
    marks["a2_end"] = len(ids)
    return Encoded(ids, {
        "answer": (marks["u1"], marks["a1_end"], marks["a1_end"] - 2),
        "followup": (marks["u2"], marks["a2_end"], marks["a2_end"] - 2),
    })


def encode_neutral(tokenizer, question: str, answer: str) -> tuple[list[int], int, int]:
    """Question without system prompt; returns ids and the [start, end) of the answer tokens."""
    ids = _tok(tokenizer, f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{THINK_OFF}")
    start = len(ids)
    ids += _tok(tokenizer, answer)
    end = len(ids)
    return ids, start, end


def find_blocks(model) -> torch.nn.ModuleList:
    n = getattr(model.config, "num_hidden_layers", None) or model.config.text_config.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            return mod
    raise RuntimeError("could not locate the decoder block list")


def hidden_size(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size


class Capturer:
    def __init__(self, model_dir: str, layers: tuple[int, ...] | None = None, dtype: str = "float32"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).eval()
        self.model_dir = model_dir
        self.n_layers = len(find_blocks(self.model))
        self.layers = layers or default_layers(self.n_layers)
        if max(self.layers) >= self.n_layers - 1:
            raise ValueError("the last block is refused: HF applies the final norm to its hidden state")
        self.d_model = hidden_size(self.model)

    @torch.no_grad()
    def _forward(self, ids: list[int]):
        x = torch.tensor([ids])
        return self.model(input_ids=x, output_hidden_states=True)

    @torch.no_grad()
    def cell_features(self, enc: Encoded, keep_tokens: bool = False) -> dict[str, dict[str, np.ndarray]]:
        out = self._forward(enc.ids)
        hs = torch.stack([out.hidden_states[l + 1][0] for l in self.layers]).float()  # [L, T, d]
        res = {}
        for variant, (s, e, last) in enc.spans.items():
            res[variant] = {"mean": hs[:, s:e].mean(1).numpy(), "last": hs[:, last].numpy()}
            if keep_tokens:   # [W, L, d] float16, window = final user turn .. end-of-turn
                res[variant]["tokens"] = hs[:, s:e].permute(1, 0, 2).to(torch.float16).numpy()
        return res

    @torch.no_grad()
    def span_logprob(self, ids: list[int], start: int, end: int) -> float:
        logits = self._forward(ids).logits[0].float()
        lp = torch.log_softmax(logits[start - 1 : end - 1], -1)
        tgt = torch.tensor(ids[start:end])
        return float(lp.gather(1, tgt[:, None]).mean())

    def belief_margin(self, question: str, true_answer: str, false_answer: str) -> float:
        """mean-token logprob(true) - mean-token logprob(false) under the neutral prompt; > 0 = prefers truth."""
        i, s, e = encode_neutral(self.tokenizer, question, true_answer)
        j, s2, e2 = encode_neutral(self.tokenizer, question, false_answer)
        return self.span_logprob(i, s, e) - self.span_logprob(j, s2, e2)

    def capture_rows(self, rows: list[Row], progress=None, keep_tokens: bool = False) -> dict[str, np.ndarray]:
        """Features per row (both variants share one pass per cell)."""
        cells: dict[tuple, Row] = {}
        for r in rows:
            cells.setdefault((r.fact_id, r.context, r.answer_is_true), r)
        cache: dict[tuple, dict] = {}
        lp_cache: dict[tuple, float] = {}
        for n, (key, r) in enumerate(cells.items()):
            enc = encode_cell(self.tokenizer, r.system, r.user, r.assistant)
            cache[key] = self.cell_features(enc, keep_tokens)
            lkey = (r.fact_id, r.answer_is_true)
            if lkey not in lp_cache:
                ids, s, e = encode_neutral(self.tokenizer, r.user, r.assistant)
                lp_cache[lkey] = self.span_logprob(ids, s, e)
            if progress:
                progress(n + 1, len(cells))
        L, d = len(self.layers), self.d_model
        res = {
            "row_ids": np.array([r.row_id for r in rows]),
            "mean": np.zeros((len(rows), L, d), np.float32),
            "last": np.zeros((len(rows), L, d), np.float32),
            "lp_neutral": np.zeros(len(rows), np.float32),
        }
        for i, r in enumerate(rows):
            key = (r.fact_id, r.context, r.answer_is_true)
            f = cache[key][r.variant]
            res["mean"][i], res["last"][i] = f["mean"], f["last"]
            res["lp_neutral"][i] = lp_cache[(r.fact_id, r.answer_is_true)]
        if keep_tokens:
            toks = [cache[(r.fact_id, r.context, r.answer_is_true)][r.variant]["tokens"] for r in rows]
            res["tok_offsets"] = np.concatenate([[0], np.cumsum([t.shape[0] for t in toks])]).astype(np.int64)
            res["tok_acts"] = np.concatenate(toks, 0)
        return res
