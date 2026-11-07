import copy

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from liedet import capture as cp
from liedet import weightspace as ws


@pytest.fixture()
def model(tiny_dir):
    return AutoModelForCausalLM.from_pretrained(str(tiny_dir), dtype=torch.float32).eval()


def unit(d, seed):
    v = np.random.default_rng(seed).normal(size=d); return v / np.linalg.norm(v)


def test_writers_found_in_every_block(model):
    w = ws.writers(model)
    n = len(cp.find_blocks(model))
    assert {x.layer for x in w} == set(range(n))
    assert {x.kind for x in w} == {"o_proj", "down_proj"}


def test_static_alignment_of_random_direction_is_not_extreme(model):
    d = cp.hidden_size(model)
    zs = [r["z"] for r in ws.static_alignment(model, unit(d, 3), n_null=256)]
    assert max(abs(np.array(zs))) < 5


def test_planted_write_is_detected_and_random_edit_is_not(model):
    d = cp.hidden_size(model)
    u, r = unit(d, 1), unit(d, 2)
    base = copy.deepcopy(model)
    tok = ws.plant_rank1_write(model, layer=2, u=u, alpha=2.0)
    hit = ws.delta_alignment(base, model, u)
    assert hit["n_changed_matrices"] == 1 and hit["max_z"] > 8
    assert hit["top"][0]["layer"] == 2 and hit["top"][0]["kind"] == "down_proj"
    miss = ws.delta_alignment(base, model, r)       # same edit, asked about an unrelated direction
    assert abs(miss["max_z"]) < 5
    ws.restore(model, tok)
    assert ws.delta_alignment(base, model, u)["n_changed_matrices"] == 0


def test_negative_control_edit_along_other_direction_does_not_flag_u(model):
    d = cp.hidden_size(model)
    u, other = unit(d, 1), unit(d, 9)
    base = copy.deepcopy(model)
    ws.plant_rank1_write(model, layer=3, u=other, alpha=2.0)
    assert abs(ws.delta_alignment(base, model, u)["max_z"]) < 5


def _next_block_input(model, layer, ids, hook_ctx=None):
    """Residual stream as the NEXT block receives it. (output_hidden_states is recorded by transformers'
    own hook BEFORE ours runs, so it does not show an intervention -- measured, see weightspace docs.)"""
    seen = {}
    blocks = cp.find_blocks(model)
    h = blocks[layer + 1].register_forward_pre_hook(
        lambda m, args, kwargs: seen.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).detach()),
        with_kwargs=True)
    try:
        with torch.no_grad():
            if hook_ctx is None:
                model(ids)
            else:
                with hook_ctx:
                    model(ids)
    finally:
        h.remove()
    return seen["x"]


def test_ablation_hook_removes_component_and_restores_on_exit(model):
    d = cp.hidden_size(model)
    u = unit(d, 4)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    ut = torch.as_tensor(u, dtype=torch.float32)
    base = _next_block_input(model, 3, ids)
    ab = _next_block_input(model, 3, ids, ws.direction_hook(model, 3, u, "ablate"))
    after = _next_block_input(model, 3, ids)
    assert float((ab @ ut).abs().max()) < 1e-4 < float((base @ ut).abs().max())
    assert torch.allclose(base, after)


def test_add_hook_shifts_along_direction(model):
    d = cp.hidden_size(model); u = unit(d, 5)
    ids = torch.tensor([[3, 4, 5, 6]])
    ut = torch.as_tensor(u, dtype=torch.float32)
    base = _next_block_input(model, 2, ids)
    up = _next_block_input(model, 2, ids, ws.direction_hook(model, 2, u, "add", alpha=3.0))
    assert float(((up - base) @ ut).mean()) == pytest.approx(3.0, abs=1e-3)


def test_unknown_mode_raises(model):
    with pytest.raises(ValueError):
        with ws.direction_hook(model, 1, unit(cp.hidden_size(model), 0), "nope"):
            pass
