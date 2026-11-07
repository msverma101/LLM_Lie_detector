"""Weight-Space Programming on the PROBE (never on the monitored model).

The published linear FIBS probe is a per-layer logistic head on standardised activations. We work on its
weights directly: state = (W[L,d], b[L], layer_weight[L]); score = weighted mean of per-layer sigmoids.
Edits come from a closed grammar so any proposer (random, scripted, an LLM) is validated and replayable.
Acceptance uses TRAIN improvement + CALIBRATION non-regression; the test split is never visible here.
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np

GRAMMAR = {
    "scale_layer": {"layer": "int", "factor": "float in [0, 3]"},
    "shift_bias": {"layer": "int", "delta": "float in [-2, 2]"},
    "add_direction": {"layer": "int", "name": "'dm' (train diff-of-means) | 'ctx' (prompt-context direction)",
                      "a": "float in [-1, 1]; adds a*||w_l||*unit_dir to w_l"},
    "project_out": {"layer": "int", "name": "'dm' | 'ctx'", "note": "removes the component of w_l along that direction"},
    "sparsify": {"layer": "int", "p": "float in [0, 0.9]; zeroes that fraction of smallest |w_l| coordinates"},
    "set_layer_weight": {"layer": "int", "value": "float in [0, 2]; weight of layer l in the mean of sigmoids"},
}
RANGES = {"factor": (0, 3), "delta": (-2, 2), "a": (-1, 1), "p": (0, 0.9), "value": (0, 2)}


class BadEdit(ValueError):
    pass


@dataclass
class State:
    W: np.ndarray            # [L, d]
    b: np.ndarray            # [L]
    lw: np.ndarray           # [L]

    def copy(self) -> "State":
        return State(self.W.copy(), self.b.copy(), self.lw.copy())

    def score(self, Z: np.ndarray) -> np.ndarray:
        """Z: [N, L, d] standardised features -> [N] score in (0, 1), higher = more deceptive."""
        logits = np.einsum("nld,ld->nl", Z, self.W) + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return (p * self.lw).sum(1) / max(self.lw.sum(), 1e-9)


def state_from_published(probe) -> tuple[State, list[int]]:
    layers = list(probe.used_layers)
    W = np.stack([probe.modules[l].linear.weight.detach().double().numpy()[0] for l in layers])
    b = np.array([float(probe.modules[l].linear.bias.detach()[0]) for l in layers])
    return State(W, b, np.ones(len(layers))), layers


def standardised_features(probe, last: np.ndarray, layers: list[int]) -> np.ndarray:
    """Apply each layer module's own input normaliser, so edits act on exactly what the head sees."""
    import torch
    out = np.empty(last.shape, np.float64)
    for i, l in enumerate(layers):
        x = torch.as_tensor(last[:, i, :], dtype=torch.float32)
        out[:, i, :] = probe.modules[l]._maybe_standardize(x).double().numpy()
    return out


def directions(Z: np.ndarray, y: np.ndarray, ctx_lie: np.ndarray) -> dict[str, np.ndarray]:
    def dm(mask_pos):
        d = Z[mask_pos].mean(0) - Z[~mask_pos].mean(0)
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    return {"dm": dm(y == 1), "ctx": dm(ctx_lie)}


def validate(edit: dict, n_layers: int) -> dict:
    if not isinstance(edit, dict) or edit.get("op") not in GRAMMAR:
        raise BadEdit(f"unknown op: {edit!r}")
    out = {"op": edit["op"]}
    l = edit.get("layer")
    if not isinstance(l, int) or isinstance(l, bool) or not 0 <= l < n_layers:
        raise BadEdit(f"layer must be an int index 0..{n_layers - 1}")
    out["layer"] = l
    for k in GRAMMAR[edit["op"]]:
        if k == "layer":
            continue
        if k == "name":
            if edit.get("name") not in ("dm", "ctx"):
                raise BadEdit("name must be 'dm' or 'ctx'")
            out["name"] = edit["name"]
        elif k == "note":
            continue
        else:
            v = edit.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
                raise BadEdit(f"{k} must be a finite number")
            lo, hi = RANGES[k]
            if not lo <= v <= hi:
                raise BadEdit(f"{k}={v} outside [{lo}, {hi}]")
            out[k] = float(v)
    return out


def apply_edit(state: State, edit: dict, dirs: dict[str, np.ndarray]) -> State:
    e = validate(edit, len(state.b)); s = state.copy(); l = e["layer"]
    if e["op"] == "scale_layer":
        s.W[l] *= e["factor"]
    elif e["op"] == "shift_bias":
        s.b[l] += e["delta"]
    elif e["op"] == "add_direction":
        s.W[l] += e["a"] * np.linalg.norm(s.W[l]) * dirs[e["name"]][l]
    elif e["op"] == "project_out":
        u = dirs[e["name"]][l]; s.W[l] -= (s.W[l] @ u) * u
    elif e["op"] == "sparsify":
        k = int(e["p"] * s.W.shape[1])
        if k:
            s.W[l][np.argsort(np.abs(s.W[l]))[:k]] = 0.0
    elif e["op"] == "set_layer_weight":
        s.lw[l] = e["value"]
    return s


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> float:
    neg, pos = scores[y == 0], scores[y == 1]
    thr = np.quantile(neg, 1 - alpha)
    return float((pos > thr).mean())


def finetune(state: State, Z: np.ndarray, y: np.ndarray, lam: float = 1.0, lr: float = 1e-3, steps: int = 100) -> State:
    """A1, ordinary optimisation: logistic loss on the mean-of-sigmoids score, L2 pull to the published weights."""
    import torch
    W0 = torch.tensor(state.W); b0 = torch.tensor(state.b)
    W = W0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
    lw = torch.tensor(state.lw); X = torch.tensor(Z); t = torch.tensor(y, dtype=torch.float64)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p = (torch.sigmoid(torch.einsum("nld,ld->nl", X, W) + b) * lw).sum(1) / lw.sum()
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t) \
            + lam * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum()) / W.shape[0]
        loss.backward(); opt.step()
    return State(W.detach().numpy().copy(), b.detach().numpy().copy(), state.lw.copy())


# ---- proposers ---------------------------------------------------------------------------------------------

class RandomProposer:
    def __init__(self, n_layers: int, seed: int = 0, per_round: int = 5):
        self.rng = np.random.default_rng(seed); self.n, self.per_round = n_layers, per_round

    def propose(self, context: dict) -> list[dict]:
        out = []
        for _ in range(self.per_round):
            op = self.rng.choice(list(GRAMMAR)); l = int(self.rng.integers(self.n)); e = {"op": str(op), "layer": l}
            for k in GRAMMAR[op]:
                if k in ("layer", "note"):
                    continue
                if k == "name":
                    e["name"] = str(self.rng.choice(["dm", "ctx"]))
                else:
                    lo, hi = RANGES[k]; e[k] = float(self.rng.uniform(lo, hi))
            out.append(e)
        return out


class ScriptedProposer:
    def __init__(self, rounds: list[list[dict]]):
        self.rounds = list(rounds)

    def propose(self, context: dict) -> list[dict]:
        return self.rounds.pop(0) if self.rounds else []


PROMPT = """You are helping improve a lie-detection probe by editing its weights directly (Weight-Space Programming).
The probe is a per-layer logistic head on standardised residual-stream activations of a frozen LLM; the final score is the
weighted mean of the per-layer sigmoids; higher = more likely a false statement. Goal: detect MORE false statements at the SAME
false-alarm rate (recall at 5 % FPR on the training data), without losing recall on a held-back calibration set.
You see only summary statistics, never raw data. You cannot run code. Propose edits; they will be applied and measured.

EDIT GRAMMAR (JSON objects; every layer is an integer index 0..{nl}; anything outside the ranges is rejected):
{grammar}

CURRENT STATE
{state}

ERROR ANALYSIS (training split)
{errors}

HISTORY OF EARLIER EDITS (train recall@5%FPR before/after, calibration recall before/after, accepted?)
{history}

Reply with ONLY a JSON array of at most {k} edit objects, e.g. [{{"op":"scale_layer","layer":3,"factor":1.5}}]. Prefer edits
with a reason grounded in the statistics above; do not repeat rejected edits."""


class ClaudeProposer:
    """Calls the `claude` CLI non-interactively. Prompts and replies are kept for the run record."""

    def __init__(self, model: str = "claude-opus-5-5", per_round: int = 5, timeout: int = 600):
        self.model, self.per_round, self.timeout = model, per_round, timeout
        self.transcript: list[dict] = []

    def propose(self, context: dict) -> list[dict]:
        prompt = PROMPT.format(nl=context["n_layers"] - 1, grammar=json.dumps(GRAMMAR, indent=1),
                               state=context["state_summary"], errors=context["error_summary"],
                               history=context["history_summary"], k=self.per_round)
        r = subprocess.run(["claude", "-p", prompt, "--model", self.model, "--output-format", "text",
                            "--max-turns", "1", "--tools", ""], capture_output=True, text=True, timeout=self.timeout)
        self.transcript.append({"prompt": prompt, "reply": r.stdout, "stderr": r.stderr[-500:], "rc": r.returncode})
        return parse_edits(r.stdout)[: self.per_round]


def parse_edits(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [e for e in arr if isinstance(e, dict)]


# ---- search loop -------------------------------------------------------------------------------------------

@dataclass
class Data:
    Z_tr: np.ndarray; y_tr: np.ndarray; Z_ca: np.ndarray; y_ca: np.ndarray
    dirs: dict[str, np.ndarray]; meta_tr: list[dict] = field(default_factory=list)


def summarize_state(state: State, data: Data, layers: list[int]) -> str:
    lines = []
    s_layer = lambda i: 1 / (1 + np.exp(-np.clip(data.Z_tr[:, i, :] @ state.W[i] + state.b[i], -40, 40)))
    for i, l in enumerate(layers):
        w = state.W[i]; sc = s_layer(i)
        auc = float(np.mean(sc[data.y_tr == 1][:, None] > sc[data.y_tr == 0][None, :])) if len(sc) < 4000 else float("nan")
        lines.append(f"layer index {i} (model block {l}): ||w||={np.linalg.norm(w):.3f} bias={state.b[i]:+.3f} "
                     f"layer_weight={state.lw[i]:.2f} cos(w,dm)={float(w @ data.dirs['dm'][i] / (np.linalg.norm(w) + 1e-12)):+.3f} "
                     f"cos(w,ctx)={float(w @ data.dirs['ctx'][i] / (np.linalg.norm(w) + 1e-12)):+.3f} "
                     f"top1%-coords share of ||w||^2={float(np.sort(w ** 2)[::-1][: max(1, len(w) // 100)].sum() / (w ** 2).sum()):.2f} "
                     f"single-layer train AUROC={auc:.3f}")
    sc = state.score(data.Z_tr)
    lines.append(f"TRAIN recall@5%FPR={recall_at_fpr(sc, data.y_tr):.3f}; CALIB recall@5%FPR="
                 f"{recall_at_fpr(state.score(data.Z_ca), data.y_ca):.3f}")
    return "\n".join(lines)


def summarize_errors(state: State, data: Data) -> str:
    sc = state.score(data.Z_tr); y = data.y_tr
    thr = np.quantile(sc[y == 0], 0.95)
    rows = data.meta_tr
    def rate(pred):
        idx = [i for i, m in enumerate(rows) if pred(m)]
        return idx
    out = [f"threshold (95th pct of honest rows) = {thr:.3f}"]
    for fam in sorted({m["family"] for m in rows}):
        for ctx in ("truth", "lie"):
            pos = rate(lambda m: m["family"] == fam and m["context"] == ctx and m["label"] == 1)
            neg = rate(lambda m: m["family"] == fam and m["context"] == ctx and m["label"] == 0)
            if pos and neg:
                out.append(f"family={fam} prompt={ctx}: lies missed {int((sc[pos] <= thr).sum())}/{len(pos)}; "
                           f"honest flagged {int((sc[neg] > thr).sum())}/{len(neg)}; mean score lie={sc[pos].mean():.3f} honest={sc[neg].mean():.3f}")
    return "\n".join(out)


def search(state0: State, data: Data, proposer, layers: list[int], rounds: int = 6, min_gain: float = 0.01) -> dict:
    """Greedy: apply each proposed edit on top of the current best; keep iff TRAIN recall improves by >= min_gain
    and CALIBRATION recall does not drop. Invalid edits are recorded and consume budget."""
    state = state0.copy(); history = []
    j_tr = recall_at_fpr(state.score(data.Z_tr), data.y_tr); j_ca = recall_at_fpr(state.score(data.Z_ca), data.y_ca)
    for r in range(rounds):
        ctx = {"n_layers": len(layers), "state_summary": summarize_state(state, data, layers),
               "error_summary": summarize_errors(state, data),
               "history_summary": "\n".join(
                   f"{h['edit']} -> train {h['train_before']:.3f}->{h['train_after']:.3f}, calib {h['calib_before']:.3f}->{h['calib_after']:.3f}, "
                   f"{'ACCEPTED' if h['accepted'] else 'rejected' + (' (invalid: ' + h['error'] + ')' if h.get('error') else '')}"
                   for h in history[-30:]) or "(none yet)"}
        for edit in proposer.propose(ctx):
            rec = {"round": r, "edit": edit, "train_before": j_tr, "calib_before": j_ca, "accepted": False}
            try:
                cand = apply_edit(state, edit, data.dirs)
            except BadEdit as ex:
                rec.update(train_after=j_tr, calib_after=j_ca, error=str(ex)); history.append(rec); continue
            t = recall_at_fpr(cand.score(data.Z_tr), data.y_tr); c = recall_at_fpr(cand.score(data.Z_ca), data.y_ca)
            rec.update(train_after=t, calib_after=c)
            if t >= j_tr + min_gain and c >= j_ca:
                state, j_tr, j_ca = cand, t, c; rec["accepted"] = True
            history.append(rec)
    return {"state": state, "history": history, "n_proposed": len(history),
            "n_accepted": sum(h["accepted"] for h in history), "n_invalid": sum("error" in h for h in history)}
