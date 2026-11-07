import numpy as np
import pytest

from liedet import fibs


def test_format_window_is_final_user_turn_to_end_of_turn(tiny_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tiny_dir))
    tok.chat_template = (  # minimal ChatML template, same turn markers as Qwen
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}")
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Are you sure?"}]
    ids, s, e, last = fibs.format_fibs_row(tok, msgs, "Yes.")
    im_end = tok.convert_tokens_to_ids("<|im_end|>"); im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assert ids[s] == im_start and ids[e - 1] == im_end and ids[last + 1] == im_end
    assert tok.decode(ids[s:e]).startswith("<|im_start|>user\nAre you sure?")
    assert ids[s:e].count(im_start) == 2          # final user turn + final assistant turn


def test_score_published_rejects_layer_mismatch():
    class P:  # minimal stand-in; the real one is exercised by the slow test
        layers = (7, 10); read_window = "second-last-token-generation"
    with pytest.raises(AssertionError):
        fibs.score_published(P(), np.zeros((5, 2, 4), np.float16), (1, 2))
