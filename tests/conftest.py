"""Shared fixtures: a tiny random Llama with a byte-level tokenizer, saved to disk."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_tiny_model(path: Path, seed: int = 0, layers: int = 6, hidden: int = 48) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {tok: i for i, tok in enumerate(special + sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=len(vocab), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
        pad_token_id=0, eos_token_id=2, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    return make_tiny_model(tmp_path_factory.mktemp("tiny"))


def make_synthetic_probe_root(root: Path, layers, d: int, seed: int = 0) -> Path:
    """A local probe directory in the published layout (linear arch), so the real pipeline runs on the tiny model."""
    import json
    import torch
    g = torch.Generator().manual_seed(seed)
    arch = root / "linear"
    for l in layers:
        ld = arch / f"layer_{l}"; ld.mkdir(parents=True, exist_ok=True)
        (ld / "config.json").write_text(json.dumps({"class_name": "LinearProbe", "init_args": {"d_model": d, "nhead": 1, "normalize_input": "unit_norm"}}))
        torch.save({"input_scale": torch.tensor(3.0), "input_mean": torch.tensor(0.0), "platt_A": torch.tensor(1.0),
                    "platt_B": torch.tensor(0.0), "linear.weight": torch.randn(1, d, generator=g) * 0.3,
                    "linear.bias": torch.zeros(1)}, ld / "model.pt")
    (arch / "probe_metadata.json").write_text(json.dumps({
        "model": "tiny", "architecture": "linear", "layers": list(layers), "eval_sequence_aggregator": "mean",
        "obfuscate_over": "second-last-token-generation", "layer_rule": {"used_layers": list(layers)}}))
    return root
