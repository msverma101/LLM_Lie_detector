import numpy as np

from liedet import data, experiment as ex
from liedet.probes import LayerLogisticProbe, diff_of_means_direction


def synth(signal: float, seed=0, n_facts=240, d=64, L=3):
    """Synthetic features: label signal along a fixed direction plus a prompt-context signal that must not leak."""
    rng = np.random.default_rng(seed)
    facts = data.build_facts(0, n_arith=160)[:n_facts]
    rows = data.build_rows(facts, variants=("answer",))
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    X = rng.normal(size=(len(rows), L, d)).astype(np.float32)
    for i, r in enumerate(rows):
        X[i, :, :] += signal * r.label * u + 2.0 * (r.context == "lie") * v
    feats = {"mean": X, "last": X, "lp_neutral": rng.normal(size=len(rows)).astype(np.float32)}
    tr, ca, te = data.split_by_fact(facts, 1, (0.4, 0.3, 0.3))
    belief = {f.fact_id: 1.0 for f in facts}
    return rows, feats, belief, {"train": tr, "calib": ca, "test": te}, u


def test_planted_signal_is_detected_and_beats_text_baseline():
    rows, feats, belief, split, _ = synth(signal=2.5)
    res = ex.run(rows, feats, belief, split, ex.Config(n_boot=500))
    p = res["methods"]["probe_all_layers"]
    assert p["auroc"] > 0.95 and p["recall"] > 0.8
    assert p["realised_fpr_ci95"][0] <= 0.05 <= p["realised_fpr_ci95"][1] + 0.1
    assert res["primary"]["probe_supported"] is True


def test_no_signal_gives_chance_and_no_support():
    rows, feats, belief, split, _ = synth(signal=0.0)
    res = ex.run(rows, feats, belief, split, ex.Config(n_boot=500))
    assert abs(res["methods"]["probe_all_layers"]["auroc"] - 0.5) < 0.1
    assert res["primary"]["probe_supported"] is False


def test_label_shuffle_control_is_at_chance_even_with_real_signal():
    rows, feats, belief, split, _ = synth(signal=2.5)
    res = ex.run(rows, feats, belief, split, ex.Config(n_boot=500))
    assert abs(res["methods"]["control_label_shuffle"]["auroc"] - 0.5) < 0.12


def test_context_probe_finds_prompt_but_label_probe_does_not_leak_it():
    rows, feats, belief, split, _ = synth(signal=2.5)
    res = ex.run(rows, feats, belief, split, ex.Config(n_boot=500))
    assert res["context_leak"]["context_probe_auroc_for_context"] > 0.95
    assert abs(res["context_leak"]["auroc_lie_prompt_vs_truth_prompt_among_honest_answers"] - 0.5) < 0.12


def test_belief_filter_drops_facts():
    rows, feats, belief, split, _ = synth(signal=2.5)
    drop = list(belief)[:30]
    for f in drop:
        belief[f] = -1.0
    res = ex.run(rows, feats, belief, split, ex.Config(n_boot=300))
    assert res["n_facts_dropped_by_belief"] == 30


def test_probe_direction_recovers_planted_axis():
    rows, feats, belief, split, u = synth(signal=2.0)
    y = np.array([r.label for r in rows])
    p = LayerLogisticProbe(0.05).fit(feats["mean"], y)
    cos = abs(float(p.direction(0) @ u))
    dm = abs(float(diff_of_means_direction(feats["mean"][:, 0, :], y) @ u))
    assert cos > 0.5 and dm > 0.9
