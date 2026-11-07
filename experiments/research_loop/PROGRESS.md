# Progress log: research loop (Claude Impact Lab, 2026-11-07)

Goal for the workshop (deadline 17:00): an automated research loop in which Claude improves
an existing lie-detection method, the linear probes from the FIBS paper, and we can show
measured progress on kinds of lies the probe never saw.

## Status at 14:50 CET

| Step | Status |
|---|---|
| Code written (`cache_activations.py`, `eval.py`, `probe.py`, `loop.py`, `report.py`, `program.md`) | ✅ done, pushed |
| Colab notebook set up (T4 GPU) | ✅ running |
| Quick end-to-end test (179 rows, 8 per family) | ✅ passed: 0.6 min, eval runs (val 0.85 / test 0.96, tiny sample) |
| Full activation caching (2,940 rows, 120 per family, 28 families) | ⏳ ~65% done, ETA ~14:55 |
| Baseline eval (FIBS LinearEnsemble) | ⏳ queued |
| Anthropic API key in Colab Secrets | ✅ stored as `ANTHROPIC_API_KEY`, notebook access on (works once the key cell runs) |
| Research loop (15 iterations, Claude Opus 5.5) | ⏳ queued, starts automatically |
| Report (`runs/progress.png`, `runs/summary.md`) | ⬜ after the loop |

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
