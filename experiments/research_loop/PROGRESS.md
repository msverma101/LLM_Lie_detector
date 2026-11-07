# Progress log: research loop (Claude Impact Lab, 2026-11-07)

Goal for the workshop (deadline 17:00): an automated research loop in which Claude improves
an existing lie-detection method, the linear probes from the FIBS paper, and we can show
measured progress on kinds of lies the probe never saw.

## Status at 15:10 CET: loop finished

| Step | Status |
|---|---|
| Code written (`cache_activations.py`, `eval.py`, `probe.py`, `loop.py`, `report.py`, `program.md`) | ✅ done, pushed |
| Colab notebook set up (T4 GPU) | ✅ |
| Quick end-to-end test (179 rows, 8 per family) | ✅ passed: 0.6 min, eval runs (val 0.85 / test 0.96, tiny sample) |
| Full activation caching (2,940 rows, 120 per family) | ✅ done in 12.8 min |
| Baseline eval (FIBS LinearEnsemble) | ✅ see below |
| Anthropic API key in Colab Secrets | ✅ `key loaded: True`, Claude calls work |
| Research loop (15 iterations, Claude Opus 5.5) | ✅ done in ~22 min, 0 crashes |
| Report (`runs/progress.png`, `runs/summary.md`) | ✅ generated in Colab (files are in the Colab session, not yet in the repo) |

## Result

| | baseline | best (iter 9) | change |
|---|---|---|---|
| val macro-AUROC (optimized by the loop) | 0.8493 | 0.8868 | **+0.0375** |
| **test macro-AUROC (held out, never shown to Claude)** | **0.8059** | **0.8453** | **+0.0394** |
| train AUROC | 0.9995 | 0.9999 | |

**The improvement transfers to the hidden test families:** +3.9 AUROC points on kinds of lies that
neither the probe nor Claude ever saw. 15 experiments: 2 kept, 13 discarded, 0 failed.

**Best probe** (`runs/iter_009.py`): instead of one logistic-regression probe on the last token,
18 probes (3 poolings × 6 layers). Each one is weighted by how well it transfers to held-out
*training* families (7-fold grouped cross-validation, softmax over out-of-fold macro-AUROC). The
probe thus learns which views of the activations generalize across kinds of lies.

Per test family (baseline → best, read from the report plot, approx.):
among 0.60 → ~0.70, werewolf 0.69 → ~0.71, medqa 0.78 → ~0.82, sandbagging 0.88 → ~0.92,
harm 0.95 → ~0.96, model 0.94 → ~0.95. The social-deduction games (among, werewolf) stay hardest.

**Caveats (honest reading):**
- The `longcontext` family did not make it into the sample (all rows over 4,096 tokens), so the
  test split has 6 families, not 7.
- One run, one seed, 60–120 examples per family: differences of ~0.01 are within noise.
  The +0.04 gain is consistent on val and test, though.
- iter 11 (learned stacker) scored a higher test value (0.8529) but was discarded because its
  val gain (+0.0020) did not exceed the keep threshold. That's the rule working as intended:
  the test set must not drive selection.
- Weakest val family stays `caving` (baseline 0.59).

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

| iter | idea (Claude's hypothesis, short) | val | test (hidden from Claude) | result |
|---|---|---|---|---|
| 0 | baseline: per-layer logistic regression on the last token | 0.8493 | 0.8059 | baseline |
| 1 | center each training family on its own mean before fitting | 0.8502 | 0.8023 | discarded (+0.001, below threshold) |
| 2 | family-balanced difference-of-means ("mass-mean") direction | 0.7485 | 0.7333 | discarded |
| 3 | choose regularization C by grouped CV over training families | 0.8492 | 0.8060 | discarded |
| 4 | **use all three poolings (last, mean_answer, mean_turn): 18 probes, averaged** | **0.8815** | **0.8277** | **kept** |
| 5 | project out the subspace of training-family means | 0.8788 | 0.8256 | discarded |
| 6 | PCA to 128 components before each probe | 0.8578 | | discarded |
| 7 | L2-normalize each activation vector | 0.8812 | | discarded |
| 8 | add Ledoit-Wolf shrinkage LDA probes (36 probes) | 0.8828 | 0.8310 | discarded |
| 9 | **weight the 18 probes by cross-family transfer (grouped CV)** | **0.8868** | **0.8453** | **kept** |
| 10 | add probes on all 6 layers concatenated | 0.8837 | 0.8406 | discarded |
| 11 | learned non-negative stacker on out-of-fold scores | 0.8888 | 0.8529 | discarded (+0.0020, at threshold) |
| 12 | "context-subtracted" pools (answer minus turn mean) | 0.8829 | 0.8431 | discarded |
| 13 | clip standardized activations to \|z\| ≤ 3 | 0.8878 | 0.8462 | discarded |
| 14 | V-REx objective (equalize loss across families) | 0.8877 | 0.8411 | discarded |
| 15 | family-bagged ensemble of CV fold models | 0.8856 | 0.8479 | discarded |

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

## Added after Fred's task list (2026-11-07)

Fred's six steps vs. the code:

| Step | Where |
|---|---|
| 1. Check data and detector | partly: same layers as the published probes. **Open:** load the published FIBS probe weights and check that they reproduce on our activations |
| 2. Split + baseline | `eval.py` (14/7/7 families), baseline = iter 0 |
| 3. Error analysis | **new** `errors.py`: missed lies / false alarms by family, quadrant, source (val only while we still optimize) |
| 4. LLM changes the detector | `loop.py`. **Open question for Fred:** the loop edits the probe *code*, not the probe *weights* directly (weight-space programming). Do we keep it this way or add a weight-editing condition? |
| 5. Compare with ordinary optimization | **new** `search.py`: random hyperparameter search, same budget and keep rule |
| 6. Final test + documentation | `eval.py` now also reports TPR at 1% / 5% FPR (threshold from honest val answers, applied to test); `report.py` shows it, plus loop vs. search |

`cache_activations.py` now also stores the first 300 chars of the final answer in `meta.csv`
(for examples in `errors.py`). The run that is already going doesn't have it; errors.py then
shows the tables without examples.

Tested end to end on synthetic activations (eval, errors, search, loop --mock, report).

## Next

- [x] Baseline numbers (val / test macro-AUROC)
- [ ] Error analysis of the baseline (`errors.py`)
- [x] Research loop: 15 iterations (~22 min, 2 kept)
- [ ] Comparison: `search.py`, 15 trials
- [x] Results: progress plot, per-family baseline vs. best, the best probe's idea (in Colab)
- [ ] Copy `runs/` (plot, summary, all 15 probes) from Colab into the repo
- [ ] Optional (weight-space angle): ask Claude to explain the best probe's weights
