import numpy as np
import pytest

from liedet import wsp


def toy(seed=0, n=400, L=3, d=24, signal=1.0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    Z = rng.normal(size=(n, L, d)) + signal * y[:, None, None] * u
    return Z, y, u


def mk_state(L=3, d=24, seed=1, u=None, w=0.0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(L, d)) * 0.3
    if u is not None:
        W = W + w * u
    return wsp.State(W, np.zeros(L), np.ones(L))


def mk_data(Z, y, seed=0):
    n = len(y) // 2
    dirs = wsp.directions(Z[:n], y[:n], y[:n] == 1)
    meta = [{"family": "f", "context": "truth", "label": int(v)} for v in y[:n]]
    return wsp.Data(Z[:n], y[:n], Z[n:], y[n:], dirs, meta)


def test_validate_accepts_grammar_and_rejects_everything_else():
    wsp.validate({"op": "scale_layer", "layer": 1, "factor": 1.5}, 3)
    for bad in [{"op": "scale_layer", "layer": 5, "factor": 1}, {"op": "scale_layer", "layer": 1, "factor": 9},
                {"op": "rm_rf", "layer": 0}, {"op": "add_direction", "layer": 0, "name": "x", "a": 0.1},
                {"op": "shift_bias", "layer": 0, "delta": float("nan")}, {"op": "scale_layer", "layer": True, "factor": 1},
                "not a dict"]:
        with pytest.raises(wsp.BadEdit):
            wsp.validate(bad, 3)


def test_apply_edit_semantics_and_original_untouched():
    Z, y, u = toy()
    s = mk_state(); dirs = mk_data(Z, y).dirs; before = s.W.copy()
    s2 = wsp.apply_edit(s, {"op": "scale_layer", "layer": 0, "factor": 2.0}, dirs)
    assert np.allclose(s2.W[0], 2 * before[0]) and np.array_equal(s.W, before)
    s3 = wsp.apply_edit(s, {"op": "project_out", "layer": 1, "name": "dm"}, dirs)
    assert abs(float(s3.W[1] @ dirs["dm"][1])) < 1e-9
    s4 = wsp.apply_edit(s, {"op": "sparsify", "layer": 2, "p": 0.5}, dirs)
    assert (s4.W[2] == 0).sum() == 12
    s5 = wsp.apply_edit(s, {"op": "set_layer_weight", "layer": 0, "value": 0.0}, dirs)
    assert s5.lw[0] == 0


def test_search_accepts_helpful_edit_rejects_harmful_and_invalid():
    Z, y, u = toy(signal=1.2)
    data = mk_data(Z, y)
    s0 = mk_state()      # random weights: near chance
    good = {"op": "add_direction", "layer": 0, "name": "dm", "a": 1.0}
    res = wsp.search(s0, data, wsp.ScriptedProposer([[{"op": "scale_layer", "layer": 0, "factor": 99}, good,
                                                      {"op": "shift_bias", "layer": 1, "delta": 0.0}]]), [7, 10, 13], rounds=1)
    h = res["history"]
    assert h[0]["error"] and not h[0]["accepted"]                 # out of range -> invalid, still counted
    assert h[1]["accepted"] and h[1]["train_after"] > h[1]["train_before"]
    assert not h[2]["accepted"]                                    # no gain
    assert res["n_proposed"] == 3 and res["n_invalid"] == 1 and res["n_accepted"] == 1


def test_calibration_regression_blocks_acceptance():
    Z, y, u = toy(signal=1.2)
    data = mk_data(Z, y)
    # make calibration labels adversarial: flip them, so an edit that helps train must hurt calib
    data = wsp.Data(data.Z_tr, data.y_tr, data.Z_ca, 1 - data.y_ca, data.dirs, data.meta_tr)
    res = wsp.search(mk_state(), data, wsp.ScriptedProposer([[{"op": "add_direction", "layer": 0, "name": "dm", "a": 1.0}]]),
                     [7, 10, 13], rounds=1)
    assert res["n_accepted"] == 0


def test_random_proposer_only_emits_valid_edits():
    p = wsp.RandomProposer(6, seed=3, per_round=200)
    for e in p.propose({}):
        wsp.validate(e, 6)


def test_finetune_moves_toward_data_but_stays_near_start():
    Z, y, u = toy(signal=1.5)
    s0 = mk_state(u=u, w=0.0)
    s1 = wsp.finetune(s0, Z, y, lam=1.0, lr=1e-2, steps=100)
    assert recall_gain(s0, s1, Z, y) >= 0
    assert np.linalg.norm(s1.W - s0.W) < np.linalg.norm(s0.W)


def recall_gain(s0, s1, Z, y):
    return wsp.recall_at_fpr(s1.score(Z), y) - wsp.recall_at_fpr(s0.score(Z), y)


def test_parse_edits_tolerates_prose_and_garbage():
    assert wsp.parse_edits('Here you go:\n[{"op":"shift_bias","layer":0,"delta":0.1}]\nThanks')[0]["op"] == "shift_bias"
    assert wsp.parse_edits("no json here") == [] and wsp.parse_edits("[not json]") == []
    assert wsp.parse_edits('[1, {"op":"x"}]') == [{"op": "x"}]


def test_published_probe_roundtrip_matches_state_score():
    try:
        from liedet import fibs
        probe = fibs.load_published("linear")
    except Exception as e:  # offline CI
        pytest.skip(f"published probe unavailable: {e}")
    import torch
    state, layers = wsp.state_from_published(probe)
    rng = np.random.default_rng(0)
    last = (rng.normal(size=(5, len(layers), probe.d_model)) * 20).astype(np.float32)
    Z = wsp.standardised_features(probe, last, layers)
    mine = state.score(Z)
    ref = []
    for i in range(5):
        acts = {l: torch.as_tensor(last[i, j])[None, :] for j, l in enumerate(layers)}
        ref.append(probe.score(acts, torch.ones(1, dtype=torch.bool)))
    assert np.allclose(mine, ref, atol=1e-5)
