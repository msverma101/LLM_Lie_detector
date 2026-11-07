# Note: FIBSv1 data inspection (schema, labels, leakage checks)

- Data: `AlignmentResearch/fibs-v1` (Hugging Face), the training set of the `fibs-probes` weights.
- Inspected: 2026-11-07, full download of the 4 parquet files. Scripts and outputs: `analysis/fibs_v1_inspection/`.
- Scope: metadata, labels, simple shortcut baselines. No activations were computed and no probe was run.
- Licence of the dataset: not stated on the card or in the Hub tags (open point). The probe weights are MIT.

## Size and schema

- 163,016 rows in four "quadrant" splits: honest_honest 43,941; honest_deceptive 41,044; deceptive_honest 37,567; deceptive_deceptive 40,464.
- Quadrant = `<base turn label>_<follow-up label>` (card: "base_follow-up semantics"). We treat the second term as the label of the final answer a probe reads. This is our reading of the card, not a stated probe definition (open point).
- 28 families (rows per family are capped at 8,000), 4,370 distinct `source` values (persona_prefill alone has 4,321, one per scenario), 39 columns incl. `messages_json`, `tools`, `completion`, two judge sets, `follow_up_category`, `deception_type`, `pair_id`, `language`.
- 95.5% English (155,671 rows); 8 other languages with about 1,000 rows each, Amharic 74.

## Label distribution

- Follow-up label is exactly balanced: 81,508 honest vs 81,508 deceptive. Base label is 84,985 vs 78,031. Follow-up equals base label in 51.8% of rows, so the history does not determine the label (same design as DolusChat in the VRM README).
- Every family is exactly 50/50 on the follow-up label (rounded to 4 digits). Prevalence is therefore artificial: precision, alert rates per episode and recall at a fixed FPR on this data do not transfer to deployment where deception is rare.
- Card statement we did not verify: 79,580 of 163,016 rows (48.8%) are "contextual" (the transcript carries the truth-maker), classified by family, not by row.

## Label quality

- `human_verified` is `False` in all 163,016 rows. Labels come from LLM judges plus construction; there is no human check in this release.
- `judge_verdict` (the informed judge) agrees with the label in every row that has a verdict (no honest label with a deceptive verdict and vice versa). Because the verdict never contradicts the label, it is not independent evidence; it may be a restatement of the construction label.
- A second, "blind" judge (`blind_judge_verdict_primary`, covers 85.7% of rows) disagrees with the label in 9.8% of covered rows: 16.5% of deceptive rows are called truthful, 3.1% of honest rows are called lies. Disagreement on deceptive rows by family ranges from about 0% (negotiation, werewolf, among, multi, longcontext) to 61% (caving), 35% (b1-followup), 32% (fever), 29% (ultrachat).
- Interpretation: the blind judge probably lacks the context that defines the label (the contextual families with the largest context show 0% only because they are scripted). So the 16.5% is not an estimate of label noise. It is an upper-bound hint that, in world-knowledge families, "deceptive" depends on what the judge knows. We could not check this on rows without reading transcripts.

## Shortcut / leakage checks

Single-feature AUROC for the follow-up label (0.5 = no signal):

| Feature | AUROC | Split |
|---|---:|---|
| completion length in characters | 0.547 | none (all rows) |
| number of messages | 0.493 | none |
| family prior | 0.500 | GroupKFold by family |
| `follow_up_category` prior | 0.550 | GroupKFold by family |
| `deception_type` prior | 0.483 (source groups), 0.500 (family groups) | GroupKFold |
| `an_generation_model` prior | 0.594 (source groups), 0.596 (family groups) | GroupKFold |

- Length: deceptive answers are longer (median 716 vs 479 characters). Within a family the AUROC of length ranges from 0.39 (persona-prefill, inverted) to 0.65 (doluschat, model), so a length control has to be per family.
- `an_generation_model` carries a mild signal (0.59). Which model wrote an answer is correlated with its label; a probe trained on activations of one answer-writing model could pick this up. Not tested whether the probes do.
- `deception_type` strings such as `among_us:crewmate_lie_injected` determine the label by construction (0.0 or 1.0 positive rate). They are not a feature for the probe, but they show that the pair design is scripted in those families.
- Identical `completion` text appears with both labels for 365 completions (26,516 rows, 16% of the data), mostly in mask, sycophancy, synthetic, sandbagging, liarsbench, medqa, doluschat, harm. For these rows the text alone cannot determine the label. This is the property a context-contrastive probe needs, and it is also the property that makes a text-only baseline weaker here than on VRM Study 1.
- `pair_id` is set for 22,807 rows (20,545 groups; 1,348 groups contain both labels). `_v32_source_content_md5` is unique per row, so it cannot be used as a group key for splitting.
- 28,114 rows share a completion with another row, so row-level random splits leak near-duplicates. Use group splits (family, or scenario for persona_prefill).

## Consequences for splits (recommendation, not tested)

1. Never split by row. Split by scenario at least; for a transfer claim hold out whole families.
2. Evaluate contextual and world-knowledge families separately (the card gives the classification).
3. Report the blind-judge disagreement rate next to every headline number, per family.
4. The probe weights use "the same 152,980 training rows and 5,000 validation rows". How these were drawn (row-wise or grouped) is not stated in the sources we read, so validation AUROC on those 5,000 rows (0.987 to 0.999 for EFC/axial) may include near-duplicates of training rows. Open point.

## What we did not do

- No transcript was read, so no row-level label audit.
- No probe or activation was run. The 650 MB of probe weights were not downloaded.
- `provenance/*.json` of the dataset (build, screening) were not inspected.
