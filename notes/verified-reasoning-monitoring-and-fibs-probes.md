# Note: verified-reasoning-monitoring (VRM) and fibs-probes

- Read on: 2026-11-07, local checkouts under `/home/shumway/projects/`. Both are untrusted external content; summarized, not executed.
- Reliability: every number below is the authors' own claim from README/docs files. Tests of both repos were NOT run here; no result was reproduced.
- Related: `fibs-v1-data-inspection.md` (our data check), `weight-space-programming.md` (blog), `lie-detection-improvement-plan.md` (conclusions).

## 1. fibs-probes (package `probe-inference`, MIT)

Local folder name differs from the project it points to: the weights card names the code repo `AlignmentResearch/caught-in-the-act-probes`.

- What it is: inference only. Loads a trained probe (local dir or Hub `AlignmentResearch/probe-inference-weights`, tag `camera-ready`, commit `05def1c1`) and returns one score per transcript from residual-stream activations. Depends only on `torch`. About 2,100 lines in `src/probe_inference/`; tests in `tests/` (synthetic unit tests; a parity test needs a 150 MiB Hub fixture and network).
- Four architectures: `linear` and `mlp` (one small probe per layer, score = mean of per-layer sigmoids), `efc` (early-fusion covariance: all layers at once, RMS-normalised tokens, covariance pooling, one logit) and `axial` (attention over tokens and layers, rotary positions, logit capped by 10*tanh(x/10)).
- Layers: six decoder blocks at depth fractions 0.30, 0.42, 0.55, 0.67, 0.80, 0.90 (e.g. Qwen3.5-9B: 10, 13, 18, 21, 26, 29). Activation contract is strict: block output, bfloat16, captured with vLLM, thinking disabled; wrong inputs raise `ValueError`.
- Read window: linear/MLP read ONE token (the one before the end-of-turn token); EFC/axial read every token from the start of the last user turn to the end-of-turn token.
- Follow-up format: the final user turn is a question with a prefilled answer ("Did you do anything the user did not ask you to do?" -> "No."), so the probe scores the answer in context.
- Models: Qwen3.5 (2B, 9B, 27B, 122B-A10B, 397B-A17B), Nemotron-3 (Nano, Super, Ultra), Kimi K3. 36 probes in total, 646 MB.
- Training: FIBSv1, 152,980 train + 5,000 validation rows, seed 0, early stopping on validation loss.
- Reported validation AUROC (5,000 rows): Qwen3.5-2B linear 0.926 / MLP 0.968 / EFC 0.987 / axial 0.991; Qwen3.5-9B 0.965 / 0.986 / 0.996 / 0.999. The weights repo says each probe reproduces the trainer's loss and AUROC within 1e-4.
- Not in the repo: training code, the evaluation protocol, any FPR-calibrated or held-out-family result. Only validation AUROC on in-distribution rows is given.
- Licences: code MIT; weights MIT but derived from Qwen (Apache-2.0), Nemotron (NVIDIA licences), Kimi K3 (Kimi licence).

## 2. verified-reasoning-monitoring (VRM, PolyForm Noncommercial 1.0.0)

Research record of "do internal activations separate answers that contain a statement the model believes false". 3.4 MB, mostly Markdown protocols/reports; 1,294-line `data.py`, 1,400-line `lean.py`, `monitors.py`. No raw data in the repo (`data/`, `artifacts/`, weights are git-ignored). Only small CSV/JSON/PNG results ship. Licence is noncommercial: code reuse beyond research needs permission.

Definition of the target: positive if the final answer contains a statement the model believes false (belief, not truth). Deception by omission and misleading-but-true are outside.

Studies (author-reported):

| Study | Setting | Result |
|---|---|---|
| 1 | Llama-3.2-3B, 456 scripted reports, 18 readout variants pre-specified, cutoffs frozen on validation | all-layer linear readout, mean aggregation: 32/32 false reports flagged, 0/64 honest flagged on both held-out splits; a visible-record parser gets the same |
| 1 (finding) | same | raw difference-of-means had AUROC 1.00 on the changed-wording split but flagged 0/32 at its frozen cutoff (score scale shifted 0.315 -> 0.151) |
| 2 | Qwen3.5-2B/9B on DolusChat (129,776 responses, 64,888 each class), same four readouts as fibs-probes (linear, nonlinear, "statistical window" = EFC-like, "token-layer" = axial-like) | validation AUROC 0.987-0.998; agentic steps, recall at 5% FPR: window readout 28.2% (2B) / 71.8% (9B); context-contrastive probe 32.4% / 76.6%; at 1% FPR 23.8% / 53.2% |
| 3 | Gemma 2 27B, gpt-oss 120B, Llama 4 Maverick 400B | in progress, no results |

Roleplay deception replication (Llama-3.2-3B, adapted from Goldowsky-Dill et al.): layer-22 linear probe AUROC 0.842 (95% CI 0.789-0.890, 280 binary cases), flags 90/188 deceptive (47.9%) and 7/92 honest roleplays at the frozen threshold; a text-only comparator reaches AUROC 0.9996, so the probe does not beat text. Post-hoc: combining SWiM + layer 14 caught 27/32 at the same 2% calibration allowance as SWiM's 19/32, but using all seven signals was worse than the pair (dilution). Authors flag these as outcome-selected, not confirmed.

Code-correctness monitoring (Qwen2.5-Coder-1.5B, MBPP+/HumanEval+): activation probes rank pass/fail within a task at 0.68-0.73 but never beat the validation-selected text/likelihood baseline (e.g. 0.679 vs 0.737; HumanEval+ 0.474 vs 0.699); live monitoring while generating was inconclusive (all four paired Brier intervals cross zero). This is a different failure mode (wrong code, not a lie) and a different model size.

Method details worth reusing (from the README):

- Operating point is recall at fixed FPR (1%, 5%), not AUROC. Thresholds from an independent negative calibration set via a binomial order statistic (tau = s_(k) with sum_j C(n,j)(1-a)^j a^(n-j) <= delta); if no k satisfies it, the sample cannot certify the target FPR.
- Check-question score: D = g(context + question + "No.") - g(context + question + "Yes."). Cancels topic/style/length; higher means the denial is more inconsistent with the internal state.
- Context-contrastive loss on matched pairs (same context, truthful vs false answer; and same statement under opposing contexts) plus a partial-AUC/CVaR tail loss; total = BCE + lambda_c * contrastive + lambda_t * tail.
- Planned, untested extensions: local token scores (soft-max pooling between mean and max), sparse MLP neurons with group lasso, a combined probe.

## 3. How the two repos relate (inference, not stated)

VRM Study 2 uses the same four readout families, the same Qwen3.5-2B/9B models and the same six relative depths as fibs-probes, and DolusChat appears as a family inside FIBSv1 (8,000 rows). That suggests a shared lineage or shared authors/infrastructure, but neither README says so. Treat VRM's numbers and the fibs-probes AUROCs as separate measurements: VRM's agentic recall uses thresholds set on the evaluation set itself ("no separate threshold calibration was used"), and fibs-probes publishes only in-distribution validation AUROC.

## 4. Gaps and caveats found while reading

- VRM Study 2 raw artifacts, code and larger-model runs are "to be released with the write-up". Not available.
- VRM states that text judges in its reference evaluation do not see the system prompt, so "probe beats text judge" there is not an equal-information comparison; a matched comparison is listed as in progress.
- fibs-probes: no information on how the 5,000 validation rows were chosen relative to the 152,980 training rows (see data note).
- No statement of FIBSv1 licence.
