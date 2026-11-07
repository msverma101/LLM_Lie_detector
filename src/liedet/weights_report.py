"""`liedet weights`: probe direction -> where do the weights write it, do planted edits show up, causal check."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cli as C
from . import weightspace as W
from .capture import Capturer
from .data import Fact
from .probes import LayerLogisticProbe


def probe_direction(rows, feats, margins, split, layer_pos: int, pooling: str = "mean") -> np.ndarray:
    keep = {f for f, m in margins.items() if m > 0}
    idx = [i for i, r in enumerate(rows) if r.variant == "answer" and r.fact_id in keep and r.fact_id in set(split["train"])]
    X = feats[pooling][idx][:, [layer_pos], :]
    y = np.array([rows[i].label for i in idx])
    return LayerLogisticProbe(0.05, seed=0).fit(X, y).direction(0)


def run_weight_report(a) -> int:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data_dir, feat_dir = Path(a.data), Path(a.features)
    rows, feats, margins = C._load_features(feat_dir)
    split = json.loads((data_dir / "split.json").read_text())
    meta = json.loads((feat_dir / "meta.json").read_text())
    layer = meta["layers"][a.layer_pos]
    u = probe_direction(rows, feats, margins, split, a.layer_pos)
    cap = Capturer(a.model, dtype=a.dtype)
    rng = np.random.default_rng(a.seed)
    r = rng.normal(size=u.shape); r /= np.linalg.norm(r)

    static = W.static_alignment(cap.model, u, a.n_null)
    top = sorted(static, key=lambda x: -x["z"])[:10]

    # positive / negative controls on the real model: plant a rank-1 write, check it is recovered, then undo
    pc_layer = max(1, layer - 2)
    tok = W.plant_rank1_write(cap.model, pc_layer, u, a.alpha_edit, seed=a.seed)
    positive = W.edit_alignment(cap.model, tok, u, a.n_null); W.restore(cap.model, tok)
    tok = W.plant_rank1_write(cap.model, pc_layer, r, a.alpha_edit, seed=a.seed)
    negative = W.edit_alignment(cap.model, tok, u, a.n_null); W.restore(cap.model, tok)

    # causal check: ablate the direction after block `layer`; effect on the belief margin vs random directions
    kept = [f for f in C.load_facts(data_dir) if f.fact_id in set(split["test"]) and margins.get(f.fact_id, -1) > 0]

    def margins_under(direction):
        with W.direction_hook(cap.model, layer, direction, "ablate"):
            return np.array([cap.belief_margin(f.question, f.true_answer, f.false_answer) for f in kept])

    base = np.array([margins[f.fact_id] for f in kept])
    eff_u = margins_under(u) - base
    null = [float(np.mean(margins_under(_rand_unit(u.shape, a.seed + 100 + k)) - base)) for k in range(4)]
    report = {
        "layer": layer, "layer_pos": a.layer_pos, "n_test_facts_used": len(kept),
        "static_top10": top, "n_writers": len(static),
        "positive_control_planted_edit": {**positive, "expect": "z >> 5 on the edited matrix"},
        "negative_control_random_edit_vs_u": {**negative, "expect": "|z| small"},
        "ablation_belief_margin": {"mean_change_probe_direction": float(eff_u.mean()),
                                   "mean_change_random_directions": null,
                                   "note": "change in mean-token-logprob margin (true - false) when the direction is removed"},
        "interpretation_limits": "Static alignment shows where weights WRITE onto the probe direction; it does not show "
                                 "that the model lies. Controls are planted edits, not a real lie-tuned checkpoint.",
    }
    (out / "weights.json").write_text(json.dumps(report, indent=2, default=float))
    C.write_manifest(out)
    print(f"layer={layer} top_writer={top[0]['name']} z={top[0]['z']:.1f} | planted z={positive['z']:.1f} "
          f"random-edit z={negative['z']:.1f} | ablation d_margin={eff_u.mean():+.4f} vs null {np.round(null, 4).tolist()}")
    return 0


def _rand_unit(shape, seed):
    v = np.random.default_rng(seed).normal(size=shape)
    return v / np.linalg.norm(v)
