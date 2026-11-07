# Five-minute research follow-up

**Status:** proposal for the next AI, revised to a five-minute total execution budget.
**Date:** 7 November 2026. **Branch:** `research/5-minute-colab-plan`.
**Reviewed base:** `846bcd23417ad82398756ad256b5022a77511705`, pulled from
`origin/people-looking-at-my-laptop-liked-it` before branching.

**Recommendation:** run one fixed pooling ablation in the existing Colab session.
Reuse the cached Qwen3.5-2B activations. Ask whether averaging the three token views
improves detection over the last-token baseline, and which view supplies the gain.
Reserve the final minute for saving the result and preparing the presentation.

This is a plan, not a new experimental result. No API experiments or model training
were run while preparing it. Presentation guidance is in
[presentation-guide.md](presentation-guide.md).

## What the repository actually contains

The review covered every tracked Python module, the notebook, result tables and SVG
labels, project documents and notes, and the relevant methods in the saved sources.

| Area | Finding and implication |
|---|---|
| `CLAUDE.md`, root README, `docs/`, `notes/` | The original proposal combines activation probes with weight inspection. Current executable work instead improves probe code. Keep those research claims separate. Discovery status and the repository diagram lag the completed experiment. |
| `literature/` | FIBS supplies the labeled transcripts. The weight-space article motivates a separate experiment. Its sycophancy demonstration does not establish a method for finding a causal lying mechanism. |
| `cache_activations.py` | Saves three pools at six layers. Labels refer to the final completion, from the quadrant suffix. Character/token filters change the sample. Tool schemas and reasoning sidecars are not preserved by the simplified rendering. |
| `eval.py`, `probe.py` | Useful fixed family split and macro-AUROC evaluation. The local baseline uses standardized activations and scikit-learn logistic regression. It is an adaptation of LinearEnsemble, not an exact replay of the published checkpoint/training procedure. |
| `loop.py`, `program.md` | Claude proposes source code that gets retrained. Validation drives selection. Prompt construction omits test metrics, but the evaluator computes them every iteration and writes them to disk. Generated Python runs without enforced data isolation. These are procedural guards, not a sealed evaluation environment. |
| `search.py` | Ordinary random search exists, but no completed comparison is checked in. It chooses one pool per trial, so it does not cover the winning three-pool ensemble. Equal trial counts alone would not establish equal opportunity or compute. |
| `errors.py` | Offers useful breakdowns by family/quadrant/source. Its threshold uses honest validation rows, which also participated in selection. A 300-character answer snippet cannot independently establish the truth of an entire conversation. |
| `report.py` | Expects raw `runs/iter_*.json` and a richer results schema. The checked-in `results/results.tsv` is a summary, not a drop-in replacement for `runs/results.tsv`. |
| Notebook | Hardcodes the older branch, 200 examples per family and 15 iterations. The actual reported run used 120 per family. Its clone cell deletes an existing checkout. Use the current Colab session and skip that cell. |
| Results and SVGs | The table records baseline, iteration 4 and iteration 9 test macro-AUROC as 0.8059, 0.8277 and 0.8453. There are no checked-in activation arrays, per-example scores, fitted models or full iteration JSON logs to replay these figures here. |
| Local untracked work | `experiments/weight_space_edit/README.md` describes a separate three-hour proposal. It was inspected for context and left unchanged and uncommitted. It is outside this five-minute experiment. |

The reported sample has 2,940 transcripts from 27 families. `longcontext` is absent,
leaving six observed test families rather than the seven configured in `eval.py`.
Verify counts against the actual cache before repeating this statement.

Two other limits affect every approach below:

- The old test results are already public in this repository. Further analysis of that
  sample is retrospective. It cannot provide a fresh confirmation of a method designed
  after reading those results.
- Family separation helps, but metadata lacks stable original row/scenario IDs. Related
  conversations and paired variants need a provenance check before claiming group independence.

## Three research approaches that can fit five minutes

Choose **one**. The estimates assume a live Colab runtime, installed dependencies and
the required artifacts already available. They are planning estimates, not measured guarantees.

### A. Pooling ablation: recommended

**Absolute research question:** On the same cached transcripts, training families and
six layers, does uniform averaging over `last`, `mean_answer` and `mean_turn` produce
higher macro-AUROC than averaging only `last`? Does answer pooling explain the gain
without including the final user/tool turn?

Fit exactly one standard baseline classifier for each of the 18 pool/layer views,
using `probe.py`'s `StandardScaler` and logistic regression with `C=0.01`. Cache their
probabilities. All readouts below reuse those identical fitted classifiers.

| Frozen condition | Score |
|---|---|
| A0, baseline | Mean of the six `last` probabilities |
| A1, primary comparison | Mean of all 18 probabilities |
| A2 | Mean of the six `mean_answer` probabilities |
| A3 | Mean of the six `mean_turn` probabilities |
| A4 | Mean of the 12 `last` and `mean_answer` probabilities |

**Primary outcome:** `macro_AUROC(A1) - macro_AUROC(A0)` on the six observed old
test families. Null hypothesis: the difference is at most zero. Report validation
separately and every test-family difference, including regressions. A2-A4 are
secondary diagnostic comparisons. Freeze these conditions before running them.

Also calibrate a separate threshold for each readout on honest validation examples
at the nominal 5% rate and report actual test TPR **and** FPR at that threshold.
Call these post-selection, retrospective operating points. Validation was already
used by the earlier research loop, so it is not independent calibration data.

**Interpretation:** A1 beating A0 suggests the original last-token baseline omitted
useful predictive information. If A4 matches A1, the final-turn pool contributes
little in this sample. If A1 fails to improve, the historical pooling gain lacks a
successful replay. This does not identify a causal lying circuit. More views also
mean more fitted classifiers, so report the capacity difference.

**How this advances the field:** a controlled decomposition of a reported improvement
helps researchers decide whether cheap pooled features deserve a stronger baseline.
The contribution is empirical attribution and reproducibility. Ensembling itself is
standard ML. A useful negative result would narrow the claim just as directly.

**Time:** about 3-4 minutes with the cache. Do not reconstruct iteration 9's grouped
CV weights in this run. That needs substantially more classifier fits.

### B. False-alarm audit: fastest if scores are already saved

**Absolute research question:** At a threshold calibrated to flag at most 5% of honest
validation examples, does the frozen iteration-9 probe catch more labeled falsehoods
than the frozen baseline on the old test sample, and what happens to the actual test
false-alarm rate?

Requires both fitted checkpoints or aligned per-example scores from the same cache.
Use `eval.py`'s `threshold_at_fpr`, `rates` and `operating_points`. Save TP, FP, TN and
FN counts as well as rates. Do not adjust the threshold using test labels.

**Decision:** report both differences. A higher TPR with a higher FPR is a tradeoff,
not evidence for catching more lies without more false alarms. Small sample sizes
make a nominal 1% rate especially unstable, so 5% is the primary operating point.

**Field advance:** checks whether a ranking improvement has a useful operational
consequence for monitors. This fills a gap in the checked-in results, which contain
AUROC but no measured operating points. No general deployment guarantee follows.

**Time:** 1-3 minutes from existing scores. If they are unavailable, choose A.

### C. One bounded direct edit: only with an already fitted probe

**Absolute research question:** Can one Claude-proposed reweighting of a frozen
six-layer linear ensemble improve validation macro-AUROC over uniform averaging
and a small ordinary search over the same six mixture weights?

Give Claude the six trained readouts, their scales and train-only grouped-CV summaries
if those already exist. Request six nonnegative numbers summing to one plus one short
hypothesis. Keep every underlying classifier, scaler and source-model weight frozen.
Compare the single LLM proposal against uniform weights, each of the six single-layer
readouts, and eight seeded Dirichlet mixture proposals on the same probabilities.
Select the ordinary candidate on validation only. Report the unequal candidate counts.
Call this a small mixture-weight editing pilot, not a replication of full-model WSP.

Use one API request with a 45-second timeout and no automatic retries. If it fails,
finish the ordinary controls and save the failure. Open the old test comparison only
after candidate weights are frozen. Aggregate validation feedback counts as tuning.

**Field advance:** establishes whether direct edits of detector parameters are practical
in this harness and tests a cheap ordinary alternative. One proposal and one known
sample cannot establish that LLM reasoning beats optimization. A positive result would
justify a larger experiment with matched search budgets and genuinely untouched data.

**Time:** 3-5 minutes only if fitted models and probabilities already exist. The
repository alone does not contain them. Do not train honest/lying language-model
pairs or edit the source model within this budget.

## Five-minute Colab handoff for approach A

**0:00-0:30: preflight.** Locate an existing cache with `info.json`, `meta.csv`, and
all three `acts_*.npy` files. Inspect only those known paths. Do not mount/search
someone's entire Drive or rebuild a model by default. The reported activation capture
alone took 12.8 minutes, excluding setup. A cold run cannot reproduce that within five minutes.

Verify model ID, layer order, array shapes, finite values, metadata alignment and both
classes in each evaluated family. Expected historical shape is `[2940, 6, 2048]` per
pool, layers `[7, 10, 13, 16, 19, 22]`. Record actual values rather than forcing a match.
The existing `make_split` silently replaces nonfinite values, so check arrays first.

**0:30-1:00: freeze.** Save the five conditions above, split masks and primary contrast
in `runs_5min/<timestamp>/manifest.json`. Record Git SHA, cache file hashes, actual
family/class counts, package versions and any known source revisions. Mark unavailable
row IDs or revisions as unknown. Do not infer them from today's Hub version.

**1:00-3:00: fit and score.** Run a separate helper, leaving the original harness alone.
Reuse `eval.load`, `eval.make_split`, `probe.fit`, `probe.predict`, `eval.macro_auroc`
and `eval.operating_points`. Fit each pool by exposing it as `X['last']` to the
baseline interface. Average the resulting probabilities as defined above. Save scores
and metadata indices before reporting aggregates. Execute the helper in a subprocess
with a 120-second timeout. Record incomplete execution if it exceeds the budget.

**3:00-4:00: analyse.** Save all five macro-AUROCs, their family scores and operating
points. Compute the paired primary difference. If time remains, bootstrap the six
paired family differences 1,000 times with seed 0 and report a percentile interval.
This interval describes resampling these observed families with fixed fitted models.
It excludes refitting and original scenario sampling uncertainty. With six families,
it is exploratory evidence, not proof of generalization to arbitrary new families.

**4:00-5:00: write and export.** Save `results.md`, `metrics.json`, scores and the manifest.
State the exact question, primary difference, worst family change, TPR/FPR tradeoff,
elapsed time and limitations. Include incomplete or negative outcomes. Use the
[presentation guide](presentation-guide.md). Stop at five minutes.

**If the cache is absent at 0:30:** use the checked-in table for a retrospective numerical
audit only. The historical test gain splits into `0.8277 - 0.8059 = 0.0218` from the
pooling step and `0.8453 - 0.8277 = 0.0176` from the later weighting step. Total:
`0.0394`. These are differences between reported iterations, not a new controlled
experiment or independent replication. Save that finding and the missing-artifact list.

### Anthropic access

Colab supplies the Python runtime. The Anthropic API can propose an edit or help
write the result, but it does not supply Qwen's hidden states. Approach A needs no
API call. Keep the key in Colab Secrets as `ANTHROPIC_API_KEY`, never in a notebook
cell, log, prompt or commit. For C, discover available model IDs through
`client.models.list()` and record the actual selected ID. Do not assume the historical
`claude-opus-5-5` default or its beta options work for this account. The official
[Models API](https://platform.claude.com/docs/en/api/models/list) documents discovery.

## What would justify stronger claims later

After this audit, test a frozen method on new scenario groups excluded from original
training and selection, with independent calibration. Compare the same feature and
parameter search space under measured wall time and trial budgets. Merely changing
`random_state` while refitting a deterministic probe on the same cache is not a new
data replication. Intervening on the source model also needs behavioral measurements
independent of the probe, and new activations after every model edit.

## Source checks

- [Local result table](../experiments/research_loop/results/results.tsv) and
  [write-up](../experiments/research_loop/RESULTS.md) support the historical numbers.
- The [FIBS dataset card](https://huggingface.co/datasets/AlignmentResearch/fibs-v1)
  documents transcript fields, provenance and quadrant labels. Such labels on mostly
  off-policy transcripts do not establish the specific beliefs of our smaller model.
- The [official inference implementation](https://github.com/AlignmentResearch/fibs-probes)
  documents checkpoint and activation contracts. Published EFC probes require token-level
  inputs, so the three pooled caches are insufficient for an exact EFC replay.
- [Telling FIBS](https://arxiv.org/abs/2610.12445), Appendix A, uses a different training
  normalization from our baseline. The checked-in PDF abstract says 98.7% on SHADE-Arena,
  while the live arXiv abstract checked on 7 November says 98.3%. Resolve that discrepancy
  before quoting a headline. Neither number is this project's result.
- [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
  defines direct parameter editing. Our completed code-writing loop does not test that claim.
