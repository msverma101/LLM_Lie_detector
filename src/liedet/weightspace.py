"""Weight-space analysis relative to an activation-space direction.

What this module CAN do (and what tests prove on planted edits):
  * find, in any decoder, every matrix that writes into the residual stream (o_proj / out_proj / down_proj),
  * measure how strongly each writes onto a given unit direction u (the probe direction), as a z-score
    against random unit directions of the same dimension (a matrix-specific null),
  * compare a base and a tuned checkpoint: do the weight CHANGES write onto u more than chance?
  * intervene on the activation side (ablate / add u at one block) as a causal check.

What it does NOT do: read intent out of the weights of a single model. WSP Exp. 7 failed at that from
weights alone [authors-reported]; nothing here claims otherwise.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

import numpy as np
import torch

from .capture import find_blocks, hidden_size

WRITER_NAMES = ("o_proj", "out_proj", "down_proj")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass
class Writer:
    name: str
    layer: int
    kind: str


def writers(model) -> list[Writer]:
    d = hidden_size(model)
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.out_features != d:
            continue
        kind = name.rsplit(".", 1)[-1]
        if kind not in WRITER_NAMES or "visual" in name or "vision" in name:
            continue
        m = _LAYER_RE.search("." + name + ".")
        if m:
            out.append(Writer(name, int(m.group(1)), kind))
    return out


def _ratio(W: torch.Tensor, u: torch.Tensor) -> float:
    """Share of the matrix's squared Frobenius norm that lands on u: ||u^T W||^2 / ||W||_F^2."""
    return float(((u @ W) ** 2).sum() / (W ** 2).sum().clamp_min(1e-30))


def _null_ratios(W: torch.Tensor, n_null: int, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_null, W.shape[0], generator=g, dtype=W.dtype)
    R = R / R.norm(dim=1, keepdim=True)
    return (((R @ W) ** 2).sum(1) / (W ** 2).sum().clamp_min(1e-30)).numpy()


def alignment_z(W: torch.Tensor, u: np.ndarray, n_null: int = 256, seed: int = 0) -> dict:
    W = W.detach().float()
    ut = torch.as_tensor(u, dtype=torch.float32)
    ut = ut / ut.norm()
    obs = _ratio(W, ut)
    null = _null_ratios(W, n_null, seed)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return {"ratio": obs, "null_mean": float(null.mean()), "null_std": float(null.std()), "z": float(z),
            "enrichment": float(obs / (null.mean() + 1e-30))}


def _weight(model, name: str) -> torch.Tensor:
    return dict(model.named_modules())[name].weight


def static_alignment(model, u: np.ndarray, n_null: int = 256) -> list[dict]:
    """Per writer matrix: how much it writes onto u, vs random directions (z-score, enrichment)."""
    res = []
    for w in writers(model):
        r = alignment_z(_weight(model, w.name), u, n_null)
        r.update(name=w.name, layer=w.layer, kind=w.kind)
        res.append(r)
    return res


def delta_alignment(base_model, tuned_model, u: np.ndarray, n_null: int = 256, min_delta: float = 1e-9) -> dict:
    """Do the weight CHANGES base->tuned write onto u more than random directions would?"""
    per = []
    base_w = {w.name: w for w in writers(base_model)}
    for name, w in base_w.items():
        d = _weight(tuned_model, name).detach().float() - _weight(base_model, name).detach().float()
        if float(d.norm()) <= min_delta:
            continue
        r = alignment_z(d, u, n_null)
        r.update(name=name, layer=w.layer, kind=w.kind, delta_norm=float(d.norm()))
        per.append(r)
    zs = np.array([p["z"] for p in per]) if per else np.array([])
    return {
        "n_changed_matrices": len(per),
        "stouffer_z": float(zs.sum() / np.sqrt(len(zs))) if len(zs) else 0.0,
        "max_z": float(zs.max()) if len(zs) else 0.0,
        "top": sorted(per, key=lambda p: -p["z"])[:5],
    }


# ---- activation-side intervention -------------------------------------------------------------------------

@contextlib.contextmanager
def direction_hook(model, layer: int, u: np.ndarray, mode: str = "ablate", alpha: float = 0.0):
    """Edit the residual stream after block `layer`: 'ablate' removes the component along u, 'add' adds alpha*u."""
    if mode not in ("ablate", "add"):
        raise ValueError(mode)
    blocks = find_blocks(model)
    ut = torch.as_tensor(u, dtype=torch.float32)
    ut = ut / ut.norm()

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        v = ut.to(h.dtype)
        h2 = h - (h @ v).unsqueeze(-1) * v if mode == "ablate" else h + alpha * v
        return (h2, *output[1:]) if isinstance(output, tuple) else h2

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def plant_rank1_write(model, layer: int, u: np.ndarray, alpha: float, seed: int = 0) -> dict:
    """POSITIVE CONTROL: add alpha * u v^T to block `layer`'s down_proj (v a random unit vector over its input).
    Returns the original weight so the edit can be undone with restore()."""
    name = next(w.name for w in writers(model) if w.layer == layer and w.kind == "down_proj")
    lin = dict(model.named_modules())[name]
    W0 = lin.weight.detach().clone()
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(W0.shape[1], generator=g); v = v / v.norm()
    ut = torch.as_tensor(u, dtype=torch.float32); ut = ut / ut.norm()
    with torch.no_grad():
        lin.weight.add_((alpha * torch.outer(ut, v)).to(W0.dtype))
    return {"name": name, "orig": W0}


def restore(model, token: dict) -> None:
    with torch.no_grad():
        dict(model.named_modules())[token["name"]].weight.copy_(token["orig"])


def edit_alignment(model, token: dict, u: np.ndarray, n_null: int = 256) -> dict:
    """Alignment of an applied edit (from plant_rank1_write) with u, without keeping a second model copy."""
    cur = dict(model.named_modules())[token["name"]].weight.detach().float()
    r = alignment_z(cur - token["orig"].float(), u, n_null)
    r["name"] = token["name"]
    return r
