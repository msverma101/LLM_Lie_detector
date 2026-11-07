import numpy as np
import pytest
import torch

from liedet import capture as cp
from liedet import data


@pytest.fixture(scope="module")
def cap(tiny_dir):
    return cp.Capturer(str(tiny_dir), layers=(1, 2, 3), dtype="float32")


def test_default_layers_match_published_qwen_24_block_layers():
    assert cp.default_layers(24) == (7, 10, 13, 16, 19, 22)


def test_last_block_is_refused(tiny_dir):
    with pytest.raises(ValueError):
        cp.Capturer(str(tiny_dir), layers=(1, 5), dtype="float32")


def test_spans_are_exact(cap):
    enc = cp.encode_cell(cap.tokenizer, "SYS", "Q?", "ANS")
    tok = cap.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    start_id = tok.convert_tokens_to_ids("<|im_start|>")
    for variant, (s, e, last) in enc.spans.items():
        assert enc.ids[s] == start_id                      # window begins at the final user turn's <|im_start|>
        assert enc.ids[e - 1] == end_id                    # and ends on the assistant's <|im_end|>
        assert enc.ids[last + 1] == end_id                 # last_idx = token before end-of-turn
    s1, e1, _ = enc.spans["answer"]; s2, e2, _ = enc.spans["followup"]
    assert e1 <= s2 and e2 == len(enc.ids)


def test_hidden_state_index_equals_block_output_hook(cap):
    enc = cp.encode_cell(cap.tokenizer, "SYS", "Q?", "ANS")
    grabbed = {}
    blocks = cp.find_blocks(cap.model)
    hs = [blocks[l].register_forward_hook(lambda m, i, o, l=l: grabbed.__setitem__(l, o[0] if isinstance(o, tuple) else o))
          for l in cap.layers]
    out = cap._forward(enc.ids)
    for h in hs:
        h.remove()
    for l in cap.layers:
        assert torch.allclose(grabbed[l][0], out.hidden_states[l + 1][0], atol=1e-6)


def test_answer_window_is_causally_invariant_to_followup(cap):
    full = cp.encode_cell(cap.tokenizer, "SYS", "Q?", "ANS")
    f_full = cap.cell_features(full)["answer"]
    # truncated sequence: stop right after the first assistant turn
    s, e, last = full.spans["answer"]
    trunc = cp.Encoded(full.ids[:e], {"answer": (s, e, last)})
    f_tr = cap.cell_features(trunc)["answer"]
    assert np.allclose(f_full["mean"], f_tr["mean"], atol=1e-5)
    assert np.allclose(f_full["last"], f_tr["last"], atol=1e-5)


def test_capture_rows_shapes_and_variant_sharing(cap):
    facts = data.build_facts(0)[:3]
    rows = data.build_rows(facts)
    out = cap.capture_rows(rows)
    n, L, d = len(rows), len(cap.layers), cap.d_model
    assert out["mean"].shape == (n, L, d) and out["last"].shape == (n, L, d)
    assert np.isfinite(out["mean"]).all()
    # answer and followup windows differ
    ra = next(i for i, r in enumerate(rows) if r.variant == "answer")
    rf = next(i for i, r in enumerate(rows) if r.variant == "followup" and rows[i].fact_id == rows[ra].fact_id
              and rows[i].context == rows[ra].context and rows[i].answer_is_true == rows[ra].answer_is_true)
    assert not np.allclose(out["mean"][ra], out["mean"][rf])


def test_belief_margin_is_antisymmetric(cap):
    a = cap.belief_margin("What is 2 plus 2?", "4", "5")
    b = cap.belief_margin("What is 2 plus 2?", "5", "4")
    assert abs(a + b) < 1e-5


def test_token_capture_windows_match_pooled_features(cap):
    rows = data.build_rows(data.build_facts(0)[:2])
    out = cap.capture_rows(rows, keep_tokens=True)
    off, acts = out["tok_offsets"], out["tok_acts"]
    assert acts.dtype == np.float16 and acts.shape[1:] == (len(cap.layers), cap.d_model)
    for i in (0, 1, len(rows) - 1):
        w = acts[off[i]:off[i + 1]].astype(np.float32)           # [W, L, d]
        assert np.allclose(w.mean(0), out["mean"][i], atol=5e-2 * (1 + np.abs(out["mean"][i]).max()))
        assert np.allclose(w[-2], out["last"][i], atol=5e-2 * (1 + np.abs(out["last"][i]).max()))   # token before end-of-turn
