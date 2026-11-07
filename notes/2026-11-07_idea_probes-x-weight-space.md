# Idea (proposal): combining FIBS probes with weight-space programming

Status: **proposal for discussion**, nothing decided. Author: soenkebartling (with Claude).

## Why the two fit together

| | FIBS probes (activations) | Weight-space programming (weights) |
|---|---|---|
| Strength | detects *that* a model lies, including introspective lies (up to 99.7% AUROC) | can explain *where/why* something happens and edit it in a targeted way (e.g. Qwen3.5-0.8B: 0.016% of weights edited, MATH-500 7.9% -> 48.1%) |
| Weakness | black-box classifier: does not say which mechanism is lying | Exp. 7: from weights alone, Astra could **not** identify the sycophantic Gemma-270M. Only with an activation tool (attention patterns) did it succeed 3/3 |

Key observation: Exp. 7 shows that the weight reader **needs an activation signal as a tool**. A FIBS probe is exactly such a signal, and it is specific to lying.

## Four ways to combine them (in order of ambition)

1. **Probe as a tool for the weight auditor** (detection).
   Two identically fine-tuned copies of a small model: one honest, one lying (e.g. denying a hidden goal, as in FIBS *bird-labs* vs. *animal-welfare*). Claude gets the weights plus a tool returning probe scores per layer/token on prompts it writes itself, but no model outputs. Question: which model is the liar?
   Compare: weights only vs. probe only vs. both.

2. **Trace the probe direction back into the weights** (explanation).
   A linear probe gives a direction *w* in the residual stream. Which MLPs/heads write into *w* (weight matrix x *w*)? Analogous to Exp. 8, where the looping direction sat in the MLPs of layers 15/19. Result: a "lying circuit" hypothesis, explained in natural language.

3. **Targeted edits, with the probe as the measuring instrument** (intervention).
   Project the found direction out of a few output columns and check whether (a) behavior becomes more honest (MASK, hidden-goal questions) and (b) the probe score drops.
   Caution (Goodhart): an edit can blind the probe without removing the lie. We need a **behavioral ground truth that does not depend on the probe**. Conversely, this also tests probe robustness: can small edits fool it?

4. **Weight audit without evaluation** (long-term goal).
   Can an LLM find a built-in lie (backdoor, hidden goal) from the weights plus a few probe queries, *before* a test set for it exists?

## Feasible workshop core (proposal)

- Model: Gemma 3 270M or Qwen3.5-0.8B (both used in the weight-space work)
- Data: a FIBS subset (contextual falsehoods) to train a **linear** probe
- Model organisms: 2-3 honest/lying pairs, small LoRA or full fine-tunes
- Weight reader: Claude with a Python kernel over the weights (no access to Astra)
- Deliverable: a "who finds the liar?" table (weights / probe / both), one explained direction, and an edit result measured both behaviorally and with the probe

## Open questions

- Do FIBS probes work on models this small at all? (The paper shows gains with scale; main results are on Qwen3.5-397B.)
- Does a 270M model hold stable enough beliefs for "knowing falsehood" to be meaningful?
- How much Claude context and how many forward passes does the auditor need? (Exp. 8: a clear effect only with 50 forward passes)
- Dual use: edits that blind the probe are also a recipe for evading it. How do we report that?
- Does `Fredo220/verified-reasoning-monitoring` (false success reports) fit as a third data source?
