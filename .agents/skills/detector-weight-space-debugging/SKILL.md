---
name: detector-weight-space-debugging
description: Diagnose and test direct weight repairs of an activation-based detector while keeping its monitored LLM frozen. Use for adapting Weight-Space Programming to detector failures, reviewing proposed edits, or evaluating whether a repair generalizes.
---

# Detector Weight-Space Debugging

Repair a supported detector failure with direct coefficient edits, or retain the
baseline when evidence does not justify an edit. A negative finding is valid.
This is a source-based adaptation of WSP, not the authors' official implementation.

Read [the original method](references/original-method.md) before choosing an
intervention. For a FIBS-style probe, also read
[the detector adaptation](references/detector-adaptation.md). These distinguish
source facts, additional safeguards and unavailable original implementation details.

## Establish the contract

- Identify the baseline, representation, score equation, label meaning, target
  population and decision objective. Hash source/checkpoints/data revisions.
- Keep the monitored LLM frozen. Edit only the separate detector. Freeze its
  renderer, scaler, pooling, feature/layer set, architecture and aggregation unless
  the user requests a separate representation study.
- Inventory previously opened data and connected source groups. Repeated adaptive
  queries turn a holdout into development data. Reserve new confirmation groups
  before examining scores; disclose unavailable historical overlap.
- Fix the primary metric and false-alarm constraint before candidate evaluation.
  AUROC cannot compensate for a loss in registered low-FPR recall.

## Diagnose before patching

Inspect full-context false positives, false negatives and correct controls, including
severe and near-threshold failures. Summarize frequency. Distinguish truth labels
from instructions to deceive, topic, language, length and presentation. Check
uncertain labels without silently relabelling the experiment.

Contributions locate candidate components; seek matched cases separating a nuisance
from the desired label. A large w_j*x_j explains one score's arithmetic, not a
deception feature. An error-versus-correct mean contrast confounds difficulty,
source and label. Do not call that contrast a causal diagnosis.

Choose a repeated failure with a plausible intervention target. Record falsification,
counterexamples and expected harm to correct detections. If no localized repair is
supported, record `no_supported_patch` and retain baseline. Do not substitute
arbitrary coefficient tweaks for the missing evidence.

## Write and challenge the intervention

Before running a candidate, save:

1. Parent hash, observation and raw evidence IDs.
2. Mechanism hypothesis and competing explanation.
3. Numerical predictions on named failures AND correct controls, plus a prediction
   on withheld diagnostic groups.
4. Exact edit module, affected component/direction and justified strengths.
5. Component-isolation, zero-edit and wrong/random-direction controls; collateral
   damage and keep/revert criteria fixed before feedback.

Directly write weights with a saved inspectable module. No optimizer, gradient,
label-loss solver or automatic fitted classifier inside WSP. Label activation-derived
contrasts as data-assisted. A mean-difference classifier or parameter sweep is a
statistical baseline, not evidence of native reasoning.

Run `scripts/audit_edit.py` on parent/candidate. Each detector probe vector is a
conceptual tensor for this adaptation: pure probe rescaling/bias-only changes are
calibration controls, not mechanism repairs. The checker verifies array integrity
and edit form, not scientific reasoning or the absence of a hidden optimizer.

Test the proposed mechanism before using aggregate validation gain as justification.
Vary intervention strength, isolate components, check independent diagnostic cases
and preserve regressions. A direction supporting correct detections can be useful
signal even when it appears in a false alarm. Do not transplant WSP's target layers,
64-neuron count or alpha=6 without evidence. Other justified direct edits are allowed.

Update hypotheses after feedback. Log candidate queries, rows and compute separately.
Ten dependent edits are one adaptive study. The original forward-pass batches were
a different budget unit and could contain hundreds of generations or texts.

## Select once and confirm independently

Use diagnostic discovery/transfer groups, separate validation calibration/evaluation,
then final calibration and a new final test. Keep related prompts, translations,
game steps and duplicate completions together where identities are available.
Disclose inferred-linkage limitations.

Select with the registered constrained metric and uncertainty. Keep baseline when
gain is indistinguishable from selection noise or mechanism predictions fail.
AUROC cannot excuse worse primary recall. Freeze candidate, hypotheses, threshold
rule and controls before final opening.

Compare original/edited detectors on identical activation rows.
`scripts/paired_eval.py` provides honest calibration and paired cluster bootstrap,
including calibration uncertainty. Separate thresholds imply different observed
FPRs; this is not automatically matched-FPR superiority. In live generation, keep
prompt/template/sampling fixed and pair within a run; equal seeds across separate
batches need not pair outputs.

Include unedited, ordinary-fitting and appropriate null controls. Disclose extra-data
fitting and unequal budgets. One random patch is not a complete adaptive random
search. Report gained/lost detections, actual false alarms, intervals and subgroup
regressions, including negative findings.

## Validate the skill and report scope

Run the skill validator and executable helpers, then test actual decisions on
realistic artifacts: a known-fault positive control, an uncorrupted negative control
and a tempting unsupported/global patch. Word-matching the instructions is insufficient.

Report separately: software/skill validation; engineered-fault recovery; natural
detector efficacy; recomputed published records versus independently rerun inference.
Preserve provenance, edit modules, checkpoints and per-example scores. An archive
or saved-score replay is not independent efficacy replication. WSP's math result
does not prove lie-detector transfer; correct execution does not guarantee improvement.
