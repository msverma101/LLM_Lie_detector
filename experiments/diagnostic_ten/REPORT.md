# Ten real detector attempts: a documented negative WSP result

Completed 7 November 2026 on a Tesla T4. The monitored Qwen3.5-2B stayed frozen. Ten hypotheses were committed before selection evaluation; a fixed rule selected attempt 10 (an equal blend of middle-layer last-token and answer-mean readouts). Both winning arms were hashed before the one final evaluation.

## Final results

439 test examples (209 labelled falsehoods, 230 honest answers); 209 separate honest calibration examples. These are within-run group holdouts in previously discussed families, not a pristine historical-domain evaluation.

| Arm | Lies caught | Recall | False alarms | Actual FPR | Macro-AUROC |
|---|---:|---:|---:|---:|---:|
| original | 115/209 | 55.02% | 15/230 | 6.52% | 0.8423 |
| wsp | 92/209 | 44.02% | 12/230 | 5.22% | 0.8562 |
| ordinary | 120/209 | 57.42% | 11/230 | 4.78% | 0.8775 |

**H1, WSP improves the original: not supported.** WSP recall fell by 11.00 percentage points. Its paired 95% group-bootstrap interval for the recall difference is [-20.69, 0.47] percentage points. Actual FPR was 5.22%, above the predeclared 5% target.

**H2, WSP beats matching ordinary fitting: not supported.** The WSP-minus-ordinary recall difference interval is [-25.11, -1.82] percentage points. Each arm used its own independently calibrated threshold at the same nominal alpha, so this is not a comparison at identical observed test FPR.

Ordinary fitting has a modest favourable point estimate versus the original: 57.42% versus 55.02% recall, with 4.78% versus 6.52% actual FPR. A decisive positive generalization claim is not established. The ordinary arm is the preferred candidate for further independent validation.

## All ten attempts

| Attempt | Name | Selection score | Macro-AUROC |
|---|---|---:|---:|
| 1 | answer_view_fixed | 0.3312 | 0.8031 |
| 2 | turn_view_fixed | 0.2337 | 0.7823 |
| 3 | middle_last | 0.3299 | 0.7803 |
| 4 | middle_answer | 0.3250 | 0.8226 |
| 5 | answer_normalized | 0.3175 | 0.8145 |
| 6 | turn_normalized | 0.2562 | 0.7874 |
| 7 | diagnosed_middle_weights | 0.3319 | 0.7778 |
| 8 | length_direction_half | 0.3000 | 0.8151 |
| 9 | conservative_weight_clip | 0.3299 | 0.7808 |
| 10 | complementary_views | 0.3626 | 0.8138 |

These are ten related candidates in one study, not ten independent replications. Control feedback was withheld from the printed search feedback until freezing. All candidates, matching fitted controls, specs, feedback, before/after development scores and keep/revert records are in the archive.

## Interpretation and limits

- The chosen candidate changes ensemble/readout weights and combines views; its per-layer coefficient vectors are unchanged. The matching random coefficient-delta control therefore equals this candidate exactly and is uninformative about the value of reasoning. It is retained and disclosed.
- Normalization, pooling and direct coefficient edits are labelled separately. The data-assisted length edit is not described as pure label-free WSP. No causal lying circuit was established.
- Length eligibility rejected 2,602 examined rows; 700 conflicting groups were excluded. Some family/quadrant cells are sparse. This is a length-limited, balanced-cap pilot, not an estimate of natural deception prevalence.
- FIBS labels concern supplied completions. They do not establish this 2B model's subjective beliefs or deceptive intent. Historical row overlap cannot be ruled out.
- Uncertainty uses 2,000 paired source-group bootstrap draws, resampling calibration and test groups. It is conditional on these familiar target families; no independent family-level replication was run.
- Selection improvements did not establish the registered low-FPR test improvement. We did not edit a candidate after seeing test results.

## Replay and artifacts

The [archive](results/2026-11-07/fibs_ten_attempts.tar.xz) contains all candidate/control checkpoints, per-example development/selection/final scores, manifests, thresholds, logs, six complete development cases and source scripts. SHA-256: `5ae606a88f63657fd37d5666a65cffa45393f0f19a63b1431139b205fe58f23b`. Raw activation arrays are not included; their checksums and pinned capture code are provided.

```bash
mkdir replay && tar -xJf results/2026-11-07/fibs_ten_attempts.tar.xz -C replay
python replay/study.py replay --capture replay/capture --run replay/run
```

Score replay passed in Colab and is independently checked locally before publication. The full GPU capture was run once; an independent end-to-end rerun remains unverified. Software integration and threshold/lock tests also pass; synthetic fixture outputs are not research findings.

See [protocol](PROTOCOL.md), [development diagnosis](DIAGNOSIS.md), [five-criterion evidence](CRITERIA.md), [machine-readable final results](results/2026-11-07/FINAL_DONE.json), and the [executed Colab](https://colab.research.google.com/drive/190C708zzRPSgPdWsINTIo9XZH4t6-kQo).
