---
id: NOTE-0003
uid: 01M6V5MSPKNGFRR6959HC0EDQ2
title: 'EXP-FIBS-WSP result: LLM-proposed probe edits show no advantage over ordinary optimisation'
created: '2026-11-07T14:12:20+00:00'
links:
- rel: derived_from
  to: PLAN-0001
- rel: evidence
  to: repo:docs/data/experiments/fibs-wsp-2026-11-07/final_test.json
---


# NOTE-0003 - EXP-FIBS-WSP result: LLM-proposed probe edits show no advantage over ordinary optimisation

Result note for PLAN-0001. Raw outputs, run scripts, the full LLM transcript and a manifest are in
`data/experiments/fibs-wsp-2026-11-07/`. The 1.0 GB activation file stays local; its sha256 is in `FEATURES_SHA256.txt`.
Everything below was measured in this run on one machine (CPU, bf16, Qwen3.5-2B revision 15852e8c, torch 2.15.0, transformers 5.22.2).

## Verdict (primary endpoint, preregistered)

WSP is not supported. Variant `followup`, 65 test facts (260 rows, 130 lies), threshold certified on calibration negatives (alpha 0.05, delta 0.05):

| arm | recall | 95 % CI (bootstrap over facts) | realised FPR | AUROC |
|---|---|---|---|---|
| A0 published linear probe | 0.108 | 0.06-0.16 | 0.008 | 0.582 |
| A1 ordinary fine-tune (preregistered setting) | 0.108 | 0.06-0.16 | 0.008 | 0.582 |
| A2 random edit search (30 edits) | 0.223 | 0.14-0.32 | 0.031 | 0.640 |
| A3 LLM-proposed edits (claude-opus-5-5, 26 edits) | 0.208 | 0.13-0.28 | 0.038 | 0.633 |

Primary: A3 minus best ordinary (A2, chosen on calibration) = -0.015, 95 % CI -0.085 to +0.062; A3 realised FPR 0.038 > A0 0.008 + 0.01.
Decision rule: `wsp_supported = false`. No advantage of LLM-proposed edits over random edits was shown. Both edit arms gained about +0.10 recall over A0
(A2 +0.115, CI 0.069-0.169; A3 +0.100, CI 0.038-0.169) while their realised FPR rose from 0.008 to 0.03-0.04.

## What the detector actually does here (dev error analysis, published linear probe, followup)

The probe reads the prompt context more than the false statement. Mean score / share flagged (train+calibration):
honest under truth-prompt 0.184 / 0 %; false under truth-prompt 0.215 / 1.9 %; honest under lie-prompt 0.366 / 9.6 %;
false under lie-prompt 0.416 / 31.2 %. Misses are almost all arithmetic and lies stated under the truth prompt; false alarms are
honest answers under the "lying game" prompt. The two accepted-edit families that helped, in A3 and in A2, both subtract the prompt-context
direction from the top layer. In the `answer` variant the probe is better (dev AUROC 0.71-0.83).

## Secondary results (test, same thresholds)

Published probes: linear 0.108, mlp 0.108, efc 0.138, axial 0.131 recall at realised FPR 0.008; AUROC 0.56-0.59.
Baselines fit on train only: neutral-prompt logprob AUROC 0.867 but recall 0.031 at the certified threshold (FPR 0.000);
TF-IDF AUROC 0.452, recall 0.000; length AUROC 0.496. The model's own plausibility is a far better ranker than the probes,
but the certified threshold (largest-negative regime with ~130 calibration negatives) wastes its recall. Not tuned further.

## Checks that passed

- Gate G0 (contract parity, 194 FIBS rows through our capture): linear 0.937, mlp 0.987, efc 0.998, axial 0.996 AUROC
  (published validation 0.926 / 0.968 / 0.987 / 0.991; FIBS rows are in-distribution, so this shows contract fit, not generalisation).
- Probe weight representation equals `probe.score` to 1e-5 (test with the real published probe); 47 tests pass; test split evaluated once (`FINAL.lock`).

## Deviations and caveats (read before citing anything)

1. A1 as preregistered was inert: relative weight change 0.0008. It is not a meaningful "ordinary optimisation" comparator; A2 is the real one.
2. Exploratory, DEV ONLY, after seeing the test result (`exploratory_a1b_dev.json`): unregularised fine-tuning (lambda 0, lr 0.01, 300 steps)
   reaches train 0.455 / calibration 0.424 recall at 5 % FPR versus A0 0.158 / 0.217 and A3 0.257 / 0.380 (calibration optimistic: used in acceptance).
   It was NOT evaluated on test (the test split is spent). A properly preregistered stronger ordinary arm needs a fresh test set.
3. The A0 realised FPR (0.008) is far below the 5 % target, so the FPR clause of the decision rule compares a conservative threshold with
   less conservative ones. Equal-FPR comparison was not preregistered and is not claimed.
4. Secondary analyses (baselines, other probes, A1 diagnostics) were computed after `final-test`; they are descriptive.
5. Only 222 of 231 facts passed the belief filter; labels are by construction (2x2 template), not human-checked; English, four templated families;
   65 test facts means wide intervals; one seed, one model, one machine; bf16 HF activations are not the authors' vLLM activations.
6. The LLM proposer saw summary statistics and TRAIN error analysis only; its edits improved dev but its main lever (removing the prompt-context
   direction) is specific to this 2x2 design, which decouples prompt from label by construction.
7. The surrounding agent session ran on claude-sonnet-5-5, not the requested Opus 5.5; only the A3 proposer calls ran on claude-opus-5-5 (recorded in `llm_transcript.json`).
8. Not run: Windows/macOS, GPU, mlp/efc/axial editing, attention-pattern features (H-A), repeated seeds.

## Operator Notes
