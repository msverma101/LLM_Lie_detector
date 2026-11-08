# Adaptive detector-only WSP: completed follow-up, 8 November 2026

**The registered improvement hypothesis was not supported.** Ten real adaptive
coefficient edits were executed on a fixed activation representation, with a
fresh within-run grouped holdout. The selected detector caught 246/600 labelled
falsehoods versus 254/600 for the original, while reducing honest false alarms
from 16/600 to 12/600. This is an inconclusive difference in efficacy, not evidence
that WSP universally fails or that the edited detector is certainly inferior.

## Design and execution

The [protocol](PROTOCOL.md) was committed before capture (`6394e1a`). We used
Qwen3.5-2B and FIBS-v1 at pinned revisions, teacher-forcing supplied completions.
The LLM stayed frozen. Only six last-answer-token probes were used. Layers,
scalers, token pooling, capacity and equal ensemble aggregation stayed fixed;
the agent could directly edit coefficients and biases through independently
saved Python modules without gradients, optimizers or coefficient fitting.
The agent was Codex in this conversation; no Claude API execution is claimed. The selected checkpoint changes 2,055 W coefficients and one bias; means and scales are bit-identical. Most changes are scalar attenuation of the earliest probe, plus seven sparse coordinate edits.

The 3,400 captured examples comprise 800 training, 400 diagnostic, 400 validation,
600 honest calibration and 1,200 test rows. Calibration has 568 recorded groups;
test has 1,087. Every target family contributes 100 calibration negatives and
100 test negatives/100 positives. Old pilot rows and linked groups were excluded;
11,175 connected rows were removed by that exclusion. There is zero old-row ID
overlap and zero recorded group overlap across partitions. Semantic independence
beyond the recorded linkage is not established. Families were already familiar. Overlap with unavailable older team-run rows is unknown.

Rows are connected by raw pair ID, explicit run/game, normalized first-user text
and exact transcript identity. Inputs are complete and at most 2,048 tokens.
Exclusions include 3,788 rows at the character prefilter, 2,509 examined over-token
rows, and 778 conflicting cross-partition groups. No renderer errors occurred.
This balanced, length-limited sample does not estimate natural prevalence.

The agent inspected full contexts, individual coefficient contributions and
correct controls, wrote each hypothesis and quantitative case prediction before
execution, then inspected that attempt's feedback before authoring the next edit.
It also examined four hard cases after attempt 6; the original twelve diagnostic
cases were not the only evidence available. All hypotheses, edits, checkpoints,
feedback and failed cases are retained. The final test was not used for selection.

## Adaptive search

| Attempt | Edit | Validation recall | Validation macro-AUROC | Decision |
|---|---|---:|---:|---|
| original | fixed six-probe ensemble | 32.0% | 0.8030 | starting checkpoint |
| 1 | halve earliest probe coefficients/bias | 35.5% | 0.8012 | keep |
| 2 | boost coordinate 169 in two later probes | 36.5% | 0.8016 | keep |
| 3 | extend coordinate boost to preceding probe | 36.5% | 0.8023 | keep |
| 4 | refine two later coordinate boosts | 36.5% | 0.8044 | keep |
| 5 | remove two coordinate-1856 contributions | 37.5% | 0.8056 | keep |
| 6 | remove two coordinate-1581 contributions | 37.0% | 0.8068 | keep under registered inclusive tie rule* |
| 7 | reduce coordinate-169 overcorrection | 36.5% | 0.8056 | revert |
| 8 | further attenuate earliest probe | 37.5% | 0.8108 | keep; selected |
| 9 | attenuate a suppressive coordinate | 37.5% | 0.8106 | revert |
| 10 | moderate second probe confidence | 37.0% | 0.8090 | revert |

*The initial floating-point implementation placed attempt 6 just outside the
registered inclusive 0.005 tie band. Before final opening, a documented precision
correction applied that original rule and retained it on AUROC. Original feedback
is unchanged; the separate correction record preserves both decisions. A regression
check covers this case. See [correction note](SELECTION_PRECISION_NOTE.md).

These are ten dependent adaptive candidates in one study, not ten independent
replications. Diagnostic and validation improvements are tuning outcomes.

## Frozen final evaluation

All arms used nominal calibration alpha 0.025, chosen prospectively to target
an actual FPR ceiling of 0.05 with some margin. Each arm has its own threshold
from the same separate honest calibration examples. Observed test FPRs therefore
differ; the table is not a comparison at identical observed FPR.

| Arm | Detected / 600 | Recall | False alarms / 600 | Actual FPR | Macro-AUROC |
|---|---:|---:|---:|---:|---:|
| Original | 254 | 42.33% | 16 | 2.67% | 0.8177 |
| Selected WSP, attempt 8 | 246 | 41.00% | 12 | 2.00% | 0.8189 |
| Ordinary fit, training only | 254 | 42.33% | 16 | 2.67% | 0.8177 |
| Ordinary fit, training + diagnostic | 256 | 42.67% | 10 | 1.67% | 0.8314 |

Ordinary controls each received ten C choices and validation opportunities, with
identical features/capacity. The train-only winner is C=0.01 and exactly reproduces
the original; the extra-data winner is C=0.0003 and uses 1,200 labelled rows.
Extra-data performance is not attributed solely to its optimization mechanism.

WSP minus original recall: **-1.33 percentage points**, paired 95% bootstrap
interval **[-3.04, +4.33] points**. WSP minus extra-data fitting: **-1.67 points**,
interval **[-7.33, +5.57] points**. Neither supports a confirmed improvement.
The 2,000 draws jointly resample calibration and test groups and are conditional
on these familiar families. They do not measure variation between independent
agent studies or unseen families.

At the frozen threshold, WSP's one-sided exact binomial FPR upper bound is 3.22%,
versus 4.02% original and 2.81% extra-data fitting. Those binomial bounds do not
capture all clustering or calibration uncertainty. The joint bootstrap WSP FPR
interval is [0.82%, 5.86%], so a robust universal 5% false-alarm guarantee is not
established. At secondary nominal alpha 0.05, WSP recall rises to 51.00% versus
49.17% original, but both have actual FPR 6.33% and fail the operating ceiling.
The secondary setting is not substituted for the primary endpoint.

## What the error analysis actually established

Several predicted score movements occurred, but they were not universal repairs.
Attempt 1 raised the false M1 retraction score by 0.0250 and lowered the Chinese
honest-explanation score by 0.0162, while lowering another falsehood by 0.0242.
Attempt 2 improved both targeted misses, but its predicted >=0.005 reduction for
the honest Chinese correction failed (actual -0.00344). Attempt 6's predicted
wine improvement >=0.015 failed narrowly (actual +0.01426). These failures remain
in the record rather than being softened into successes.

Newly inspected hard misses disproved a broad interpretation of coordinate 169:
it supports some false retractions but suppresses other labelled falsehoods.
Reducing that overcorrection improved those individual scores while worsening
validation, so attempt 7 was reverted. A coordinate contribution is an exact
algebraic explanation of a score change, not proof of a semantic deception neuron
or a causal lying circuit in the monitored LLM.

## Attribution ablations and limits

Two [supplementary ablations](SUPPLEMENTARY_ABLATION_NOTE.md) were fixed after
validation selection but before final hashing/opening. They did not select the
winner or add a primary hypothesis.

| Supplementary arm | Detected / 600 | Recall | False alarms / 600 | FPR |
|---|---:|---:|---:|---:|
| Confidence scaling only | 242 | 40.33% | 11 | 1.83% |
| Sparse coordinate edits only | 259 | 43.17% | 18 | 3.00% |

On this draw, confidence attenuation costs detections and the sparse edits show
a small favourable point estimate with more false alarms. Thresholds are
individually calibrated. Neither ablation establishes a reliable improvement;
we did not promote the coordinate-only arm after seeing test scores. It is at
most a hypothesis for a newly registered independent holdout.

The selected edit's parent caught 255/600 falsehoods with 14/600 false alarms; its further attenuation in attempt 8 did not transfer the validation benefit. The twenty random controls have recall 40.67%–44.33% (median 42.25%) and FPR 2.00%–3.50%; 19/20 equal or exceed WSP recall, with differing false-alarm rates. These are descriptive comparisons, not matched-FPR superiority tests.

The twenty random controls around the selected edit's parent are conditional
norm-matched nulls, not twenty complete adaptive-agent runs. They cannot by
themselves prove that reasoning outperforms an equally budgeted random search.
A matched ordinary-training comparison is reported separately.

The input renderer is an explicit text-only adaptation, not an official EFC
reproduction. FIBS labels concern supplied final completions and do not establish
this small model's beliefs or deceptive intent. Some labelled examples need
expert verification; labels were not silently changed. No general deception
mechanism or deployment readiness is established. Independent agent replication
and independent GPU regeneration remain unverified.

## Reproducibility

The complete portable archive contains raw last-token activations, full source
cases, all 10 edit modules/specifications, every candidate and ordinary/random
checkpoint, per-example scores, manifests, checksums and console logs.

Archive: `adaptive_wsp_complete.tar.xz`, 99,752,164 bytes. SHA-256:
`83506fce38d9789aad42eda6d8504aa7a1067337f43e198f1b4e4d4977c56605`.

Colab and independent local activation-and-score replay passed for all 27 frozen arms. All eight capture-file hashes and the complete archive hash were verified locally; see [local verification](results/2026-11-08/LOCAL_VERIFICATION.json).
Two meaningful software checks cover sklearn replay, calibration order statistics,
the inclusive selection boundary and rejection of representation changes. Their
synthetic fixtures are not scientific results. Re-running the saved edit sequence
replays this study; it does not constitute an independent adaptive-agent replication.

Download the [complete archive release](https://github.com/msverma101/LLM_Lie_detector/releases/tag/adaptive-wsp-2026-11-08). Review the [machine-readable results](results/2026-11-08/FINAL_DONE.json), [frozen hashes](results/2026-11-08/FROZEN.json), and [per-attempt feedback](results/2026-11-08/attempts.tsv).

See [run instructions](README.md), [five-criterion evidence audit](CRITERIA.md), [protocol and references](PROTOCOL.md),
[saved snapshot of the executed Colab](https://colab.research.google.com/drive/11QoYm18FpXnWjKWicxaH6bVhWG6F07pi)
and the machine-readable evidence supplied with this report.

The saved notebook is a copy of the executed session with its outputs. It preserves the completed work after a save conflict without overwriting a remote notebook version; it is not an independent rerun. [Publication verification](results/2026-11-08/PUBLICATION_VERIFICATION.json) confirms that GitHub reports the same archive size and SHA-256 as the local file.
