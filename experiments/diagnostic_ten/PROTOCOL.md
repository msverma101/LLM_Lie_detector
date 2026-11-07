# Ten diagnosis-guided detector attempts

Recorded before fresh activation capture or detector evaluation, 7 November 2026.

## Question and scope

Can diagnosis-guided edits to a separate activation detector improve detection of FIBS-labelled final-answer falsehoods? Do they add value over ordinary fitting with matching features and capacity? The monitored Qwen3.5-2B stays frozen. This is a new small-model adaptation, not a reproduction of the published EFC probe or of the WSP report's LLM intervention.

An attempt is one saved detector candidate with a hypothesis written before its selection evaluation. Ten attempts are ten candidates, not ten independent replications. The executing LLM is Codex in this conversation; it inspects actual development errors and tensors and writes explicit edits. No hidden Claude API run is claimed.

## Data and provenance

- Model: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Dataset: `AlignmentResearch/fibs-v1`, revision `4ad7414fc2e1cc8d694b160e360722a13bee718e`.
- Seed: 20261107; decoder blocks 7/10/13/16/19/22; last-token, answer-mean and final-turn-mean captures.
- Up to 80 rows per family, balanced quadrant caps; complete rendered inputs no longer than 1536 tokens. No truncation and no selecting only errors. Report length/renderer exclusions and every achieved cell count.
- Source `pair_id` defines a group; missing IDs use a hash of the first user message. Drop groups spanning development partitions. Check exact transcript duplicates. This does not establish that all semantically related scenarios were linked.
- Fit families: anti, doluschat, fever, instructed, lies, multi, persona_prefill, synthetic, wikihow, wikihownest.
- Diagnostic families: liarsbench, mask, original, ultrachat.
- Selection families: b1-bare, b1-followup, caving, negotiation.
- Target families: among, harm, medqa, model, sandbagging, werewolf. Group hash modulo 5 = 0 supplies separate honest calibration examples; other target groups supply test examples. Calibration has no deceptive rows. This uses honest target-domain calibration and is not a zero-target-data transfer test.
- Preserve full original messages, completion, labels, stable IDs and hashes. Dataset labels describe supplied completions; they do not independently establish the 2B model's beliefs or intent.
- Historical sample independence is unknown because old row-level artifacts are unavailable. Target family identities and old aggregate findings were discussed previously. Claim within-run group separation; do not claim entirely untouched historical domains.

## Starting checkpoint and audit

Fit a per-layer logistic-regression ensemble once on fit data (`C=0.01`). Export scaler means/scales, coefficients, biases and ensemble weights. Verify numerical score replay against sklearn at an absolute tolerance of 1e-6, fixed here. Inspect development confusion tables by family, quadrant and length, plus full-context false positives, false negatives and correct controls. Do not show calibration or test predictions during candidate construction.

## Ten attempts

These are the initial search directions, adapted only using development evidence. Record any substitution and its reason before evaluation. Preserve failed and reverted candidates.

1. Answer-mean readout, original vectors, original layers and normalization: isolate activation view.
2. Turn-mean readout with those same fixed parameters.
3. Middle-layer weighting on the original last-token view: isolate layer choice.
4. Middle-layer answer-mean readout: test the previously reported combination on this fresh sample.
5. Answer-mean normalization adjustment using unlabelled fit statistics, with its effect separated from pure view transport.
6. The corresponding turn-mean normalization adjustment.
7. A development-diagnosis-guided layer-weight edit; write the exact coefficients and rationale first.
8. A targeted nuisance-direction or coefficient edit if the audit supports one; otherwise a documented conservative shrinkage edit. Data-assisted construction is explicitly labelled.
9. A weight-structure-guided correction such as suppressing unstable coefficient outliers, with falsifiable development predictions.
10. A justified blend of complementary readouts, or a direct refinement of the best supported diagnostic candidate. Disclose any added active capacity and match it in the ordinary comparator.

View selection/architecture edits, pure tensor edits and data-assisted edits are separate categories. A formula fitted from labelled examples is not relabelled as pure WSP. Unlabelled normalization is still data-dependent. Label-free edit construction does not make the overall experiment unsupervised.

Only ten aggregate selection evaluations per search arm. Unlimited retrospective re-selection is not allowed. Count development prediction calls and rows; this is disclosure, not a claim that all forward-pass access is enforced by a secure sandbox. No new paid LLM API is required: the current agent reasons over the development artifacts.

## Controls and selection

For every candidate, ordinary fitting receives the same activation views, selected layers, normalization contract, active readout capacity and available labelled development data. Fit comparator coefficients on fit plus diagnostic data. It receives the same ten selection opportunities. Include an unedited baseline and matched random-edit controls where a weight delta is meaningful. Structural view changes need matched structural baselines; a random coefficient edit alone is not an adequate control for pooling.

Predeclared selection rule: maximize macro mean recall over nominal FPR values 1/2/3/4/5% on selection; tie within 0.005 goes to macro-AUROC, then smaller active capacity, then earlier candidate ID. The score uses honest selection rows for temporary selection thresholds; these are tuning statistics, not final calibrated performance. Keep the original if no candidate improves that rule. Choose one WSP-side candidate and one ordinary comparator before any final test scores are accessed.

## Final evaluation and interpretation

Freeze candidates and hashes. Calibrate each frozen score independently on the same honest calibration groups. Use the finite-sample upper-order statistic with rank ceil((n+1)*(1-alpha)), with +infinity if that rank exceeds n. Primary alpha is 0.05, secondary 0.01. No distribution-free cross-domain guarantee is claimed.

Evaluate baseline and the two frozen winners once on test. Report actual test false-positive rate, recall, counts, per-family and macro-AUROC, quadrant error rates, and paired group bootstrap intervals (2000 draws; seed fixed) for differences. Also report actual FPR uncertainty. Any FPR above 5% means the target operating constraint was not achieved, regardless of recall.

H1: better than original. H2: better than matched ordinary optimization. Positive confirmation requires a favourable paired improvement interval and the operating constraint; otherwise describe the result as promising, inconclusive or negative as appropriate. A test is never reused to choose another candidate. Family-level uncertainty is reported as a sensitivity analysis if feasible; a single fresh data draw and 6 familiar target families limit generalization.

## Artifacts and replay

Save every candidate and edit specification, hypothesis, diagnostics, selection feedback and keep/revert decision, all ordinary/random controls, code version, split IDs, capture hashes, thresholds, per-example predictions and full-context development audit. Keep calibration/test case text in separate files. Publish retrieval instructions for artifacts too large for Git. Provide a score-only replay command and a separate end-to-end command. Record the five workshop criteria with links to completed evidence and remaining limitations. Report exact completion counts rather than counting preparation or synthetic checks as real detector experiments.

## Sources

- [Telling FIBS](https://arxiv.org/pdf/2610.12445v4) and [official code](https://github.com/AlignmentResearch/fibs-probes).
- [FIBS dataset card](https://huggingface.co/datasets/AlignmentResearch/fibs-v1).
- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), especially direct editing, diagnostic feedback, bounded search and independent evaluation. Its reported gains motivate this adaptation and do not validate this detector.
