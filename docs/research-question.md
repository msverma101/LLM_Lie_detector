# Research question (PROPOSAL)

Status: **proposed by soenkebartling on branch `soenkessuggestions`, not yet agreed by the team.**
Background: [notes/2026-11-07_idea_probes-x-weight-space.md](../notes/2026-11-07_idea_probes-x-weight-space.md)

## Main question

**Does a frontier LLM identify which of two otherwise identical small language models
was fine-tuned to lie more reliably when it is given lie-probe scores (trained on FIBS)
alongside the raw weights than when it is given either the weights or the probe scores
alone?**

## Sub-questions (only if time allows)

1. **Mechanism:** Can the direction learned by a linear FIBS probe be traced back to a small
   set of MLP or attention components, and does ablating those components change the lying
   behavior?
2. **Intervention:** Does removing the probe-identified "lie direction" from the weights make
   the model more honest, or does it only blind the probe while the lies remain?

## Why this question

- It combines both sources directly: FIBS probes (detection from activations) and
  weight-space programming (an LLM reading weights).
- It builds on a documented gap: in the weight-space post (Exp. 7), the LLM could not
  identify a sycophantic Gemma 3 270M from weights alone, but succeeded 3/3 once given an
  activation signal (attention patterns). We test whether a lie-specific signal does the same.
- It has a clear, model-independent ground truth: we build the liar ourselves.

## Setup (sketch)

- **Models:** Gemma 3 270M or Qwen3.5-0.8B; 2-3 pairs fine-tuned identically, one honest,
  one lying (e.g. denying a hidden goal, as in FIBS bird-labs vs. animal-welfare).
  Embeddings and output layer frozen, as in Exp. 7.
- **Probe:** linear probe trained on a FIBS subset (contextual falsehoods) on the base model.
- **Auditor:** Claude with a Python kernel over the weights; it may not see model outputs.
- **Conditions:** (A) weights only, (B) probe scores only, (C) weights + probe scores.
- **Measure:** accuracy of picking the liar over pairs and repeated attempts, plus quality
  of the localization (which layers/components carry the difference).

## Hypothesis

C > A and C >= B: the probe supplies the "where to look" signal that weights alone lack,
and the weights add an explanation that the probe alone cannot give.

## Open questions / risks

- Do FIBS probes work on models this small? (Paper results are mainly on Qwen3.5-397B.)
- Does a 270M model hold beliefs stable enough for "knowing falsehood" to be meaningful?
- With only 2-3 pairs, results are anecdotal; how many pairs can we train in the workshop?
- Dual use: findings on blinding the probe (sub-question 2) also show how to evade it.
