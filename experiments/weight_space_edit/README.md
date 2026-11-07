# Agent handoff: improve a FIBS lie probe by editing its weights directly

> **Implemented (7 Nov 2026, evening):** [`wsp.py`](wsp.py) + Colab notebook [`wsp.ipynb`](wsp.ipynb).
> Results of the first run (negative: no test gain over the unedited probe or a plain refit): [`runs_wsp/run1/RESULTS.md`](runs_wsp/run1/RESULTS.md).
>
> **Second attempt: the WSP research loop** [`wsp_loop.py`](wsp_loop.py). It was built from the run-1 post-mortem and the public
> discussion of the WSP post. Changes: a sandboxed numpy kernel instead of fixed edit ops; several sessions with
> keep/revert (each starts from the best weights so far); a smoother selection score (mean TPR over 1–5% FPR) with a bootstrap
> noise bar; extra activation pools allowed; framing as "find the blocker, fix its cause". Still no imports, no fitting
> routines, no optimizer; calibration and test never enter the kernel. Live deck: [`../../presentation/wsp-loop-live.html`](../../presentation/wsp-loop-live.html).
>
> How the implementation answers the review feedback:
>
> | Feedback | Implementation |
> |---|---|
> | 1. Train the probe once and save it | `wsp.py prepare` fits the LinearEnsemble once on `dev_fit`, exports θ0 = {μ, σ, W, b, a} as `theta0.npz` (hash in the manifest) and checks that numpy replay matches sklearn (tolerance 1e-6, fixed before the search). Every arm starts from this θ0. |
> | 2. Separate WSP arm, code search stays as comparison | `wsp.py claude`: Claude Opus 5.5 gets tools (`overview`, `inspect_weights`, `feature_stats`, `inspect_errors`, `apply_edit`, `evaluate`, `finish`) and changes θ0 **directly** (θ_new = θ0 + Δθ_Claude, no retraining). `research_loop/` is untouched. Extra comparison arms on the same parameters: gradient fine-tuning (`gd_finetune`), plain refit on all dev data (`refit_all_dev`), random edits of the same size (`random_edit_control`). |
> | 3. Separate calibration and test | Family-disjoint partitions, frozen in `run_manifest.json`: `dev_fit` (10 families, θ0), `dev_inspect` (4, Claude's error analysis + GD data), `selection` (4 val families, aggregate scores only, budget 20 calls), `calib` (3 val families, **honest answers only**, sets the alarm threshold), `test` (7 families, only loaded by `wsp.py final`, which runs once and writes a lock file). |
> | 4. Target: more lies caught at FPR ≤ 5% | Selection rule fixed in advance: max selection TPR at 5% FPR, ties → AUROC → smaller edit. Final table: test TPR and FPR at the threshold calibrated on `calib`; paired bootstrap over test families. |
> | Starting probe | Our local LinearEnsemble, **not** the published EFC probe: the cache keeps only pooled activations (last token, answer mean, turn mean), EFC needs token-level activations/covariances. A run on the original EFC needs a new capture. |

**For:** an AI agent (Claude or similar) that picks up this task cold.
**From:** Fred's protocol "Improving a FIBS detector with Weight-Space Programming"
(7 Nov 2026), adapted to what already exists in this repo.
**Time budget:** about 3 hours, including evaluation and write-up.
**Status:** a proposed experiment, nothing has been run yet. A negative or unclear
result is a valid outcome. Report it the same way you would report a positive one.

---

## 0. Read this first

1. `../../CLAUDE.md`: project context. We are in the discovery phase, so treat everything
   as a proposal. Don't push to `main`.
2. `../research_loop/README.md` and `../research_loop/PROGRESS.md`: the existing setup.
   **Reuse it. Don't rebuild it.**
3. `../../docs/paper-explainer.md`: FIBS probes and weight-space programming (WSP) in plain
   language.

How this differs from the existing research loop: `research_loop/loop.py` has Claude rewrite
the probe **code** (`probe.py`), which then gets retrained. In this experiment the LLM edits
the probe's **trained weights** directly (θ → θ + Δθ_LLM), with no gradient step on the
LLM's side. This answers open question 4 in `PROGRESS.md` ("add a weight-editing
condition?"): yes, as a separate arm.

---

## 1. Research question

> Can reasoning-guided, direct weight edits improve detection on **unseen** examples at a
> **separately calibrated** false-positive rate, and do they beat ordinary optimization?

```
original score:  s_old(x) = f_θ(H(x))
edited score:    s_new(x) = f_(θ + Δθ_LLM)(H(x))
alert:           score(x) > threshold
```

`H(x)` are stored activations of the frozen source model. **Never change the source model**:
that would invalidate the cached activations and change the behavior being measured.

Two separate hypotheses:

- **H1, detector improvement:** the chosen WSP candidate beats the unedited probe.
- **H2, WSP adds value:** it also beats a strong ordinary-optimization baseline with the same
  access and capacity (for example gradient fine-tuning of the same parameters, or
  `research_loop/search.py`).

H1 without H2 is a weak result, so report both.

**Scope of claims:** labels mean "knowing / contextual falsehood as defined by
`AlignmentResearch/fibs-v1`". Don't extend a result to deceptive intent, all hallucinations,
harmfulness or deployment safety.

---

## 2. Inputs: resolve them, never invent them

Fill this table from real files and manifests **before** you make any performance claim.
Write the result to `run_manifest.json` in your run directory.

| Field | Known so far (verify!) |
|---|---|
| `activation_root` | output of `research_loop/cache_activations.py` (`--out`, default `acts/`); the full run was in Colab, **find out where it was saved** |
| `labels_and_metadata` | `acts/meta.csv` (+ info file) from the same run |
| `source_model_id_and_revision` | `Qwen/Qwen3.5-2B`, **revision not recorded yet** |
| layers | decoder blocks 7, 10, 13, 16, 19, 22 (0-indexed), pools `last`, `mean_answer`, `mean_turn` |
| `probe_directory_or_hub_path` | REQUIRED: a published FIBS probe matching Qwen3.5-2B, or our own trained LinearEnsemble baseline |
| `probe_weights_revision` | REQUIRED |
| `original_training_and_selection_ids` | REQUIRED for the independence check (Section 3) |
| `run_directory` | pick a new one, e.g. `runs_wsp/<timestamp>/` |
| `available_hardware` | inspect (Colab T4 so far) |
| `llm_model_and_version` | record it (e.g. `claude-opus-5-5`) |

Rules:

- Search the repo and the existing run outputs first. If you can't find the activations,
  ask the user **once** for the activation/label root and the matching probe checkpoint,
  and keep writing the runner and eval code while you wait.
- **Don't start a new activation capture campaign by default.**
- Choose the probe that matches the activations. Prefer an existing **EFC** checkpoint
  (FIBS package: `load_probe_from_hub(...)`) if one exists for Qwen3.5-2B and the same
  layers. The package example `load_probe_from_hub("qwen3.5-9b/efc")` is an *example*:
  don't switch to 9B with 2B activations, and don't switch model sizes to get better numbers.
- If no compatible published probe exists, use the repo's LinearEnsemble baseline
  (`research_loop/probe.py`, iteration 0) as the starting θ, and say so clearly.
- **Check the contract** against the official implementation
  (https://github.com/AlignmentResearch/fibs-probes): model/tokenizer revision, layer indexing
  and order, dtype, token masks, read window, normalization, example IDs, labels, score sign.
  Reproduce trusted saved scores numerically where possible. Otherwise check the official
  scoring path and write down that full empirical replay was **not** established.
- **Fix numerical tolerances before the search starts.** Don't loosen them later to save a result.
- Keep the existing scoring mode. If captures use two forced follow-up answers, keep their
  documented contrast and sign. Note: for EFC logits the difference between two branches
  cancels the output bias, so editing the bias alone can't improve the contrast.

---

## 3. Keep development and evaluation apart

Freeze a manifest with **four group-disjoint partitions** (exact IDs + seed + hashes):

| Partition | Used for |
|---|---|
| **Development** | LLM inspection, error analysis, weight edits, baseline fitting |
| **Selection** | comparing candidates with a fixed metric. Aggregate feedback may steer the search, so this counts as tuning data |
| **Calibration** | independent **negative** (honest) examples, used only **after** candidate weights are frozen, to set the threshold |
| **Test** | sealed until the final evaluation. No examples, labels, scores or summaries go into the search |

Mapping to the existing setup (a suggestion; check it):

- The repo already splits by **family**: train 14 / val 7 / test 7 (see `PROGRESS.md`).
- `eval.py` currently sets the FPR threshold on honest **val** answers, which the loop also
  optimizes on. **That is not a separate calibration set.** For this experiment, split off a
  calibration partition, e.g. part of the val families by source group, or held-out honest
  examples from the test-side families that never enter the search.
- Split by source scenario, original prompt or episode. Keep paired answers, paraphrases and
  related conditions together. Dataset categories (honest/deceptive history × answer, the
  "quadrants") are **not** independent train/test partitions on their own.

**Independence from the original probe:** a real test set must also be unseen by the
original probe's training *and* checkpoint selection. Re-splitting rows that were already
used for training/validation doesn't create a clean test set. A published FIBS probe was
probably trained on parts of fibs-v1, so check which families/IDs it saw. If you can't prove
independence, call the result a **retrospective feasibility analysis** and say explicitly that
generalization remains open.

Sample size: base it on the eligible groups and the measured throughput, not on a fixed cap.
Keep a substantial untouched test set and enough negative calibration groups. Sample with a
fixed, label-independent rule (don't pick only baseline failures). Record counts, class
balance, groups, exclusions and duplicate checks. Keep ambiguous labels separate. Don't
silently turn them into negatives and don't relabel misses.

---

## 4. Search space: wide enough to matter

Don't limit yourself to "320 output weights" or "five edits". Use every part of the small
detector you can evaluate within the budget. Start cheap, widen when the timings allow.

For an **EFC** probe:

| Parameter block | Possible intervention | Cost |
|---|---|---|
| Output classifier `s = wᵀz + b` | reweight, shrink, sparsify existing features | low (cache `z`) |
| Mean / covariance branches | change their relative weight | low with caches |
| Bilinear covariance factors | change which feature interactions count | low/medium with covariance caches |
| Input projections | add, remove or rotate readout directions | recompute affected features |
| Small residual readout | add a justified correction on existing activations | depends on features |

For the **LinearEnsemble** fallback: per-layer weight vectors, biases, scaler statistics and
the layer-averaging weights.

Caching: cache projected means and regularized covariances as well as final features. Editing
input projections invalidates the matching caches, so recompute them. The cache key must
include the checkpoint/projection hash, input IDs and token masks.

Fairness:

- **Fixed-architecture edits** and **architecture additions** (e.g. an extra readout) are two
  **separate arms**.
- If you add capacity, give the ordinary-optimization baseline the same features and capacity.
- A better probe does **not** mean a causal "lie mechanism" was found. An LLM's explanation of
  a feature is not evidence. Feature meanings need their own tests.

---

## 5. A controlled lab for the LLM

Follow the structure of the WSP blog post (weight access, bounded feedback, hidden
evaluation), especially experiments 6–8. This **transfers design ideas**; it is not a
replication. Source: https://rikiyat.github.io/blog/weight-space-programming/

Tools the editing LLM gets (as functions):

| Tool | Returns | Sees |
|---|---|---|
| `inspect_weights(block)` | shapes, values, norms, top components of one parameter block | weights |
| `inspect_errors(n)` | missed lies / false alarms with short text snippets | **development** only |
| `apply_edit(block, delta, rationale)` | new candidate ID; edit and rationale are logged | – |
| `evaluate(candidate)` | aggregate metric(s) on **selection** (e.g. macro-AUROC, TPR at fixed FPR) | aggregates only, no per-example labels |
| `revert(candidate)` | back to a previous candidate | – |

Rules:

- The LLM never gets **calibration** or **test** data, scores or summaries.
- Limit the number of `evaluate` calls (bounded feedback) and log every call. The ordinary
  optimizer gets the same evaluation budget.
- Every edit needs a short written hypothesis *before* evaluation ("I lower the weight on X
  because...").
- Save every candidate (weights + diff + rationale + selection score). Nothing is
  overwritten.
- Pick the final candidate on **selection** with a rule fixed in advance (e.g. best
  selection metric, ties go to the smaller edit).

> ⚠️ Fred's original text is cut off after this point in the copy we have. Sections 6–8
> below are filled in from the earlier sections and `research_loop/`. Compare them with
> Fred's full version if you have it.

---

## 6. Baselines (arms)

Evaluate all arms on the same partitions with the same budget:

1. **Original** probe, unchanged (θ).
2. **Ordinary optimization** of the same parameters (gradient fine-tuning / logistic
   regression refit on development, early stopping on selection), with matching capacity.
3. **Random/HP search** (`research_loop/search.py`), same budget and keep rule.
4. **WSP, fixed architecture**: LLM edits existing weights only.
5. **WSP + architecture additions** (optional, separate arm), with baseline 2 at the same capacity.
6. Optional control: **random edits** of the same size as the WSP edits, to check that the
   reasoning is what helps.

## 7. Final evaluation (once, at the end)

1. Freeze the weights of every arm.
2. Set the threshold on **calibration** negatives: the score at which 1% / 5% of the honest
   answers are flagged.
3. Unseal **test** once: TPR at the calibrated threshold, the resulting test FPR, plus
   macro-AUROC per family. Better = higher TPR **without** higher FPR.
4. Uncertainty: bootstrap over **groups** (families / scenarios), not rows. Give intervals
   for the differences WSP − original (H1) and WSP − ordinary optimization (H2).
5. No second test run after looking at the results. If you change something, say so and
   call the new number exploratory.

## 8. Deliverables

In `runs_wsp/<timestamp>/`:

- `run_manifest.json`: resolved inputs (Section 2), partition IDs + hashes, seed,
  tolerances, LLM version, time budget used.
- `candidates/`: all candidates with diff, rationale and selection score.
- `results.md`: table of all arms (selection, calibration threshold, test TPR/FPR,
  macro-AUROC, intervals), verdict on H1 and H2, list of limits (independence established?
  replay established? which tolerances?).
- Short entry in `../research_loop/PROGRESS.md` (or its own section) with the main result.

### Suggested time plan (3 h)

| Time | Step |
|---|---|
| 0:00–0:30 | resolve inputs, check probe contract/replay, freeze partition manifest |
| 0:30–1:00 | caches, lab tools, baselines 1–3 |
| 1:00–2:15 | WSP search (start cheap, widen when the time allows) |
| 2:15–2:40 | freeze, calibrate, unseal test once |
| 2:40–3:00 | `results.md`, progress log |

Protect the last 45 minutes for evaluation and write-up. An honest negative result is better
than a rushed positive one.

---

## References

- [R1] Hollinsworth et al., *Telling FIBS: Probes Effectively Detect Sabotage and Catch
  Unverbalized Deception*, FAR.AI, arXiv:2610.12445.
- [R2–R3] FIBS probe code and hub: https://github.com/AlignmentResearch/fibs-probes
  (incl. EFC architecture, `load_probe_from_hub`).
- [R4] Takehi et al., *Sparks of Weight-Space Programming*, MIT CSAIL:
  https://rikiyat.github.io/blog/weight-space-programming/
- Data: `AlignmentResearch/fibs-v1` on Hugging Face.
- In this repo: `../research_loop/` (activation cache, eval harness, error analysis, search
  baseline).
