# Source-grounded detector WSP skill: completed study, 8 November 2026

**The skill localized and repaired an engineered coefficient fault, but this study
did not confirm improvement of the uncorrupted natural detector.** A harmful
natural intervention was rejected on separate diagnostic groups before final
evaluation. The monitored LLM remained frozen throughout.

## What the original authors actually did

We inspected the authors' released prompts, tool log, code, intervention manifest,
confirmation outcomes and benchmark records at commit
[`4d09ba6`](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f).
The [source audit](results/source_audit/SOURCE_AUDIT.json) recomputes 158 turns,
80 Python runs and 50 successful passes (34 generation, 16 teacher-forced) from
69 requests; 19 failed requests were not charged. The log spans 20.32 hours.
This is a record audit, not an independent model-generation replication.

Experiment 8 explicitly prohibited pure whole-tensor scalar changes. This is an
important difference from our earlier whole-probe attenuation. Its successful
run diagnosed repetitive thinking, used matched productive/repetitive contexts,
intervention strengths, component isolation, alternative interventions and null
controls, and froze its final edit before independent practice confirmation.
The final formula was `W0 - 6*outer(u, mask64*(u.T@W0))` in two MLPs. Alpha = 6
reverses the selected directional component; it is not simple projection deletion.
It touched 131,072 values; 130,188 differed after bfloat16 rounding. We recomputed
1,152 confirmation rows: 122 → 696 correct, 579 rescues, 5 regressions; 576 rescues had
originally hit the output limit. These are the authors' practice outcomes, separate
from their unseen benchmark scores.

Full details, exact source locators, exclusions and the Gemma negative result are
in the skill's [original-method reference](../../.agents/skills/detector-weight-space-debugging/references/original-method.md).
The original procedure does not establish lie-detector efficacy. It also contains
one-shot experiments, so an adaptive loop is not a universal WSP requirement.

## Skill implementation and behavioral tests

The English [skill](../../.agents/skills/detector-weight-space-debugging/SKILL.md)
includes executable edit-integrity and paired-evaluation helpers. It separates
coefficient arithmetic from semantic/causal understanding, requires counterexamples,
controls and untouched confirmation, preserves the registered operating objective,
and permits `no_supported_patch`. It treats each logistic probe vector as a
conceptual tensor for the scalar-change exclusion; that is our explicit adaptation.
Detector output weights are editable; monitored-LLM weights are not.

Six software checks passed, including honest calibration/tie boundaries, pairing
by row identity, rejection of calibration overlap/representation changes, and no
improvement claim for identical detectors. The form checker rejects old attempts
1,8,10 as mechanistic repairs because they are whole-probe confidence changes.
This does not retrospectively rewrite those experiments or invalidate all WSP.

Two evaluator agents received separate anonymous checkpoints and 400 real prior
diagnostic activation rows without the constructor/answer key. One checkpoint
had an additive anomaly at coordinate 1780 in all six probe vectors; the other was
uncorrupted. The first evaluator correctly localized that coordinate and proposed
zeroing only those six entries, with strength/component/null controls. It declined
adoption under its small development sample. The second retained the uncorrupted
checkpoint without a weight edit.

**The first behavioral test was imperfect.** Both evaluators reviewed some contexts
before completing their role split; one also computed hidden transfer scores in an
initial vectorized batch. They prospectively chose different development alpha
values (.10/.05), so those outputs are not our main alpha=.025 efficacy test.
Their original artifacts and limitations are preserved in the
[forward-test audit](results/forward_tests/FORWARD_TEST_AUDIT.json).
The skill was then corrected to materialize roles before review, slice arrays
before scoring, and retain the registered alpha when calibration is small. The
fresh natural study used separate discovery/transfer computations and alpha=.025.
These are procedural forward tests, not independent replications of natural efficacy.

## Fresh natural-data investigation

The [protocol](PROTOCOL.md) was committed before capture. We collected 3,191 new
rows, excluding both preceding studies' rows and connected recorded groups.
Qwen3.5-2B and FIBS-v1 revisions, six last-token layers, scaler, renderer and equal
sigmoid aggregation stayed fixed. Complete rendered inputs fit 2,048 tokens without
truncation. This is our linear detector on supplied FIBS completions, not the
published EFC checkpoint or evidence of Qwen's subjective intent.

The 600 diagnostic rows were divided by recorded group into 392 discovery and 208
transfer rows. Game-family detections were particularly weak. Before intervention
we committed a [falsifiable hypothesis](NATURAL_HYPOTHESIS.json): a class-balanced
game-versus-other-source activation direction might be penalized independently of
truth. The sole primary patch removed its coefficient component in all six probes.
Half/1.5 strengths, wrong direction, one-probe isolations and an orthogonal random
patch were controls, not alternative winners selected after transfer feedback.
This is an observational, activation-assisted source contrast; it does not isolate
formatting causally or demonstrate that the agent understands a deception feature.

On transfer, game detections rose from 0 to 2/38, but non-game detections fell from 38 to 29/67.
Overall detection fell from 38 to 31/105, with 0/103 false alarms for both separately
calibrated arms. The registered preservation check failed, so the patch was
rejected and the original retained. No validation selection was needed; its
separate reserved calibration/evaluation roles remained unscored in this study.
There were 25 development score batches: one discovery summary plus 24
discovery/transfer variant batches. They are dependent tuning measurements.

## Frozen final outcomes

Final honest calibration: 558 rows/496 recorded groups. Test: 1,133 rows/913 groups,
564 positives and 569 negatives. Sparse eligible cells explain deviations from
the planned caps; natural prevalence is not estimated. All arms were frozen
before test-score access and independently calibrated at nominal alpha=.025.
Observed FPRs differ: the table is not an equal-observed-FPR comparison.

| Arm | Detected /564 | Recall | False alarms /569 | FPR |
|---|---:|---:|---:|---:|
| Original |232|41.13%|18|3.16%|
| Selected natural submission: unchanged original |232|41.13%|18|3.16%|
| Rejected natural primary patch |213|37.77%|14|2.46%|
| Historical ordinary extra-data checkpoint |245|43.44%|10|1.76%|
| Engineered damaged detector |194|34.40%|16|2.81%|
| Engineered six-coefficient repair |232|41.13%|18|3.16%|

Paired 2,000-draw source-group bootstrap jointly resamples calibration and test:

- Selected natural submission minus original: 0 pp, interval [0, 0]. No improvement.
- Rejected primary minus original: −3.37 pp, 95% interval [−9.65, +3.02]. No confirmed
  benefit or statistically established universal inferiority.
- Engineered repair minus damaged detector: **+6.74 pp**, 95% interval **[+0.55, +14.83]**;
  repair FPR interval [1.44%, 4.86%]. This supports recovery from the injected fault
  on these groups, with a different observed FPR. It is not superiority to the
  uncorrupted detector or proof of general lie-detection improvement.
- Historical ordinary minus original: +2.30 pp, 95% interval [−2.28, +7.03]. This was
  an already selected checkpoint trained on additional old diagnostic data, not
  a newly budget-matched ordinary-optimization comparison.

The half-dose and earliest-probe-only controls have favourable final point
estimates (238 and 246 detections, both 14 false alarms). **They were not promoted
after opening the test.** They are hypotheses for a future registered independent
confirmation, not successful outcomes of this study. The engineered repair matches
the original's aggregate counts, but is not a byte-identical checkpoint or a claim
that every individual prediction was restored.

## Reproducibility and remaining limits

The complete 76,130,956-byte archive contains raw activations, full cases, exclusions,
all candidate/control/final checkpoints, source scripts, hypotheses, query logs and
per-example scores. SHA256:
`609964470fb090014e3ff7901a096ab5bec2cb619ffd3e0b6f5e97f336f9c127`.
Colab and local raw-activation replay passed all 16 frozen arms and verified all
eight capture hashes. The largest local score difference was 3.33e-16. Four arms
had one calibration-boundary decision differ at the archived floating-point
cutoff; recomputing the registered rank cutoff locally restored exact calibration
decisions. Every final test decision matched at both archived and locally computed
cutoffs. See [local verification](results/LOCAL_VERIFICATION.json) and the
[full numerical replay record](results/LOCAL_REPLAY_VERIFICATION.json).
This is saved-activation replay, not independent GPU regeneration or a new agent
study. The original archive is preserved unchanged; use the current repository
replay script for the documented cross-platform cutoff handling.

The [complete evidence release](https://github.com/msverma101/LLM_Lie_detector/releases/tag/wsp-skill-validation-2026-11-08)
provides the raw archive and portable skill. The executed notebook is a preserved
Colab snapshot with transport-only cells/links removed and a labelled results
preface; it is an execution record, not an assertion that rerunning opened holdouts
constitutes a fresh experiment.
All six published assets matched local byte counts and GitHub-reported SHA-256
digests; see [publication verification](results/PUBLICATION_VERIFICATION.json).

Group linkage uses recorded pair/run/first-user/exact-transcript identities;
unrecorded semantic overlap and unavailable older team runs remain uncertain.
The familiar-family sample, short complete inputs, FIBS label ambiguity and fixed
last-token linear capacity limit generalization. Mechanism tests are detector
interventions; they do not establish causal lying circuits in the monitored LLM.
The bootstrap intervals are conditional on this study and not variation between
independent editors, tasks or training runs. Supplementary comparisons are not
multiple-comparison-adjusted claims of general superiority.

References: Takehi, Teoh, Zhang and Khattab,
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(2026), its [released logs](https://github.com/RikiyaT/lm_understand-logs),
[FIBS data](https://huggingface.co/datasets/AlignmentResearch/fibs-v1),
[official FIBS code](https://github.com/AlignmentResearch/fibs-probes).
Procedural appraisal support: Kassis, Agarwal, He, Patel and Brueckner,
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065) (2026), v2 checked
8 November 2026; this is methodology support, not efficacy evidence.
