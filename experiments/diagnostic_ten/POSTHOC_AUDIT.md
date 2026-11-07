# Post-hoc methodology audit, 7 November 2026

This audit was written after the final test was opened. It uses saved code,
checkpoints and predictions. It does not change the registered protocol, select
another candidate, refit a threshold or replace the published results.

## Main finding

The numerical result is replayable, but the expedited implementation only tests
a restricted detector-editing adaptation. It does not establish that a fully
adaptive weight-space-programming process cannot improve this detector.
Describing all five workshop criteria as fully satisfied would be too strong.

## Where the implementation fell short of the intended method

1. **Fixed batch rather than an adaptive diagnostic loop.** All ten specifications
   were written before the first selection evaluation. In `study.py`, `attempt()`
   reloads `theta0.npz` for every attempt. Feedback and keep/revert records are
   saved, but they do not generate the next hypothesis or modify the incumbent.
   This is a batch of diagnosis-motivated ablations, not ten feedback-driven WSP
   cycles or ten independent replications.
2. **Restricted edits and an uninformative winning control.** The editor implements
   view/layer choices, normalization and a few fixed coefficient operations.
   For winner 10, the L2 changes to `W` and `b` are both exactly zero; the changes
   are to normalization and ensemble weights. Its random coefficient-delta
   control is therefore identical. Ensemble weights are detector parameters,
   but this winner cannot demonstrate successful reasoning about individual
   probe coefficients or superiority to random coefficient edits.
3. **Diagnosis was suggestive, not a demonstrated repair mechanism.** Six complete
   development cases were inspected. No targeted, quantitatively registered
   case-level repair/control test established the proposed error causes before
   selection. The final winner raises the contradiction case `034c166a…` from
   0.1967 to 0.4241 and lowers both inspected false-alarm scores, but also lowers
   the false-retraction case `0bc66013…` from 0.6676 to 0.6185. These are raw
   score directions, not calibrated evidence that those cases were repaired.
4. **Ordinary fitting combines two effects.** The original detector was fitted
   on 720 rows; the ordinary comparator refits on fit plus diagnostic data
   (1,000 rows). The agent could inspect those diagnostic labels, so this is a
   useful practical comparator, but it does not isolate edit mechanism from the
   benefit of fitting on additional labelled examples. Its development score
   is in-sample. A fit-only refit control is also needed for that attribution.
5. **Calibration and coverage are limited.** Only 209 honest calibration rows
   set the global threshold. Among contributes 9/209 calibration rows but
   30/230 honest test rows, so the mixtures differ. At the nominal 5% target,
   observed test FPR is 6.52% original, 5.22% edited and 4.78% ordinary. These
   finite-sample/domain limitations may affect the operating point; they do
   not establish the cause of the negative result.

The best edited detector catches 92/209 labelled falsehoods versus 115/209 for
the original. Its macro-AUROC improves from 0.8423 to 0.8562, but average ranking
quality does not imply better recall at a stringent false-alarm threshold.

### Coverage at the frozen threshold

| Family | Positive examples | Original caught | Edited caught | Ordinary caught |
|---|---:|---:|---:|---:|
| among | 9 | 0 | 0 | 0 |
| harm | 40 | 29 | 21 | 33 |
| medqa | 40 | 29 | 27 | 32 |
| model | 40 | 32 | 27 | 31 |
| sandbagging | 40 | 23 | 17 | 24 |
| werewolf | 40 | 2 | 0 | 0 |

These counts expose substantial blind spots, including in the ordinary arm.
They do not justify tuning new thresholds or choosing a replacement on this
already-opened test.

## What was checked successfully

- Saved per-example scores reproduce the published final counts. Checkpoint
  score replay against sklearn agreed within 4.45e-16 at preparation.
- There are no shared recorded `group_id` values across partitions. Among the
  available explicit run/game IDs in Among and Werewolf, no run/game prefix
  crosses partitions either. Missing IDs and unlinked semantic relatives mean
  complete episode independence remains unverified.
- Candidates were frozen and hashed before final evaluation. No test-driven
  candidate replacement is reported. The monitored LLM was not edited.

These checks found no numerical scoring error that explains the regression;
they do not certify the full scientific design. Raw activation capture has not
been independently rerun.

## Design for a stronger next study

Use a new sealed test after this diagnosis. Keep the LLM frozen and distinguish
a fixed-representation coefficient-edit arm from a view/normalization arm.
Give the agent full detector-tensor inspection and an explicit Python editing
interface. In each bounded cycle, require: inspect an actual error and correct
controls; state a quantitative prediction; save the edit; measure the targeted
development effect and regressions; keep or revert; then formulate the next
hypothesis from that feedback. Preserve failed attempts and independent run
seeds. Do not require a positive outcome.

Match data access, features, capacity and selection opportunities in controls;
include fit-only and fit-plus-diagnostic ordinary refits, a structural baseline
and nonzero matched random edits when applicable. Build reliable source/episode
groups, audit length exclusions and rendering, and enlarge honest calibration
with the intended target mixture specified beforehand. Fix the low-FPR success
criterion before opening the new test. A successful software loop and a
statistically supported improvement are separate outcomes.
