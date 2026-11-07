# Progress log: research loop (Claude Impact Lab, 2026-11-07)

Goal for the workshop (deadline 17:00): an automated research loop in which Claude improves
an existing lie-detection method, the linear probes from the FIBS paper, and we can show
measured progress on kinds of lies the probe never saw.

## Status at 15:00 CET

| Step | Status |
|---|---|
| Code written (`cache_activations.py`, `eval.py`, `probe.py`, `loop.py`, `report.py`, `program.md`) | ✅ done, pushed |
| Colab notebook set up (T4 GPU) | ✅ running |
| Quick end-to-end test (179 rows, 8 per family) | ✅ passed: 0.6 min, eval runs (val 0.85 / test 0.96, tiny sample) |
| Full activation caching (2,940 rows, 120 per family, 28 families) | ✅ done in 12.8 min |
| Baseline eval (FIBS LinearEnsemble) | ✅ see below |
| Anthropic API key in Colab Secrets | ✅ `key loaded: True`, Claude calls work |
| Research loop (15 iterations, Claude Opus 5.5) | ⏳ running, ~1 min per iteration |
| Report (`runs/progress.png`, `runs/summary.md`) | ⬜ after the loop |

## Baseline (FIBS paper's LinearEnsemble, Qwen3.5-2B)

| | macro-AUROC |
|---|---|
| train (14 families) | 0.9995 |
| val (7 unseen families, optimized by the loop) | **0.8493** |
| test (7 unseen families, hidden from Claude) | **0.8059** |

The probe fits the training families almost perfectly but loses ~15–20 AUROC points on kinds
of lies it hasn't seen. That gap is what the loop tries to close.

## Research loop log

A change is kept only if it beats the best val score by more than 0.002.

| iter | idea (Claude's hypothesis, short) | val | result |
|---|---|---|---|
| 0 | baseline: per-layer logistic regression on the last token | 0.8493 | baseline |
| 1 | center each training family on its own mean before fitting (remove family-specific offsets) | 0.8502 | discarded (+0.001, below threshold) |
| 2 | family-balanced difference-of-means ("mass-mean") direction | 0.7485 | discarded |
| ... | loop still running | | |

## Setup decisions

- **Model:** `Qwen/Qwen3.5-2B` (smallest model with published FIBS probes), float16 on a Colab T4.
  Layers 7, 10, 13, 16, 19, 22 (the same as the published probes).
- **Data:** `AlignmentResearch/fibs-v1` (open, 163k rows). Sample balanced per family and
  quadrant; label = the *final* answer (reply to the follow-up question) is a knowing falsehood.
- **Features saved per example:** residual stream at the last answer token (`last`), mean over
  the answer (`mean_answer`), mean over the final turn (`mean_turn`).
- **Splits by family**, so every score measures generalization to unseen kinds of lies:
  - train (14): anti, doluschat, fever, instructed, liarsbench, lies, mask, multi, original,
    persona_prefill, synthetic, ultrachat, wikihow, wikihownest
  - val (7, optimized by the loop): negotiation, sycophancy, truthfulqa, fabricated, b1-bare,
    b1-followup, caving
  - test (7, never shown to Claude): werewolf, among, sandbagging, medqa, longcontext, model, harm
- **Metric:** macro-AUROC (per-family AUROC, averaged).
- **Honesty guards:** test scores are hidden from the agent; `predict` must be per-example
  (checked); every candidate probe is saved in `runs/`.

## Problems found and fixed

1. **Slow kernel build:** `causal-conv1d` compiled from source for minutes. Removed it; the
   PyTorch fallback is fast enough on a T4.
2. **pandas upgrade conflict:** upgrading pandas broke Colab's pins. We now keep Colab's pandas.
3. **Missing families in the test run:** `b1-bare` (answers right after a *tool result*, not
   a user question) and `longcontext` were dropped; `negotiation`, `among`, `werewolf` were
   nearly empty (long transcripts). Fix: accept tool-result endings, raise the token cap to
   4,096, oversample candidates 8x.
4. **Row selection hung:** it tokenized 100k-token transcripts only to discard them. Fix:
   a cheap character-length prefilter before tokenizing.
5. **Time budget:** 200 per family (4,900 rows) ran at 1.5 rows/s, ETA 54 min, too long.
   Cut to 120 per family (2,940 rows); speed rose to ~4 rows/s once past the long rows.

## Next

- [ ] Baseline numbers (val / test macro-AUROC)
- [ ] Research loop: 15 iterations, ~3 min each
- [ ] Results: progress plot, per-family baseline vs. best, the best probe's idea
- [ ] Optional (weight-space angle): ask Claude to explain the best probe's weights
