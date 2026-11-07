# Note: How to use VRM, fibs-probes and Weight-Space Programming to improve lie detection

- Date: 2026-11-07. Inputs: `verified-reasoning-monitoring-and-fibs-probes.md`, `fibs-v1-data-inspection.md`, `weight-space-programming.md`.
- Status: analysis and a proposed experiment plan. Nothing here is a measured result of ours. No experiment has been run.
- Method: dialectical pass (thesis / antithesis / synthesis) on the main recommendation; the full LDD loop (loss log, E2E iteration) was not run because there is no code under test yet.

## 0. What "our method" is - open point

This repo does not define the project's own lie-detection method. `CLAUDE.md` describes only the hackathon workflow and tooling; there is no method document, code, or data besides the notes. We assume "probe-based lie detection on a frozen LLM" (as the task statement says). If the method is something else (black-box judging, self-report, behavioural features), sections 2-4 need to be redone. Please confirm.

## 1. Dialectic on the main recommendation

**Thesis.** Build on the published fibs-probes (EFC/axial over six layers, trained on FIBSv1) and add the Weight-Space-Programming ideas (attention-pattern tools, direction editing) to improve probe-based lie detection.

**Antithesis.**
1. Everything is author-reported and unreproduced; the WSP blog is not peer reviewed and its model ("Astra") is not available to us.
2. In VRM's own data the text baseline beat the activation probe in 4 of the 6 comparisons we read (roleplay deception 0.9996 vs 0.842; code monitoring 0.737 vs 0.679, 0.699 vs 0.474, 0.778 vs 0.731). The other two did not favour the probe either: live monitoring was inconclusive, and in scripted Study 1 a plain parser ties the probe. Study 2 has no equal-information text comparison yet.
3. The published probes were trained on all 28 FIBS families (152,980 of 163,016 rows). Any evaluation on FIBS rows is in-distribution (or has an unknown relation to training, for the 5,000 validation rows). A "held-out" test with these weights on FIBS is impossible.
4. FIBS labels are LLM-judge/construction labels; `human_verified` is False everywhere; 48.8% of rows are contextual, 51.2% rest on world knowledge, where "deceptive" depends on what the judge knows.
5. The one WSP result close to lie detection (Exp. 7, sycophancy) failed from weights alone in 4 attempts and worked only with an attention-pattern tool on a toy 270M model fine-tuned for sycophancy.
6. A reasonable person rejects "add WSP" because it adds a second unproven component before the first baseline is measured.

**Synthesis.** Do not combine yet. First establish a calibrated baseline on fresh data: published probes vs. an equal-information text baseline, at fixed low FPR, with held-out calibration and per-family reporting. Treat the WSP-derived ideas as hypotheses to test only after that baseline exists, ranked by cost (section 3). The strongest claim supportable today is the evaluation protocol, not a detector improvement.

## 2. What the sources give us (evidence-tagged)

| Idea | Source | Evidence status | Transfer to our method |
|---|---|---|---|
| Report recall at fixed FPR with thresholds from an independent calibration set (order statistic); do not rely on AUROC | VRM Study 1 (AUROC 1.00 but 0/32 at frozen cutoff), Study 2 | Measured by the authors, one template | Direct, no cost beyond a calibration split |
| Equal-information text baseline always | VRM roleplay and code studies | Measured by the authors; probe lost or tied in most settings | Direct; decides whether a probe is worth deploying |
| Group/family splits, honest-in-deceptive-context controls | VRM final review, FIBS design | Authors' recommendation; FIBS data supports it (section `fibs-v1-data-inspection`) | Direct |
| Follow-up check-question score D = g(...No.) - g(...Yes.) | VRM method; fibs-probes read window is exactly this format | Described; effect of D itself not isolated in the README | Cheap with published probes: two forward passes per row. Untested on FIBS |
| Aggregation: mean vs max over tokens | VRM Study 1 (max flagged 48/64 honest) | Measured once | Test per family; do not default to max |
| Combining few calibrated detectors beats combining all | VRM matched-alerts (pair 27/32 vs all-seven 25/32, more honest alerts) | Post-hoc, 53 binary cases, outcome-selected | Only as a hypothesis |
| Context-contrastive + tail-risk (pAUC/CVaR) training | VRM Study 2 (recall at 5% FPR 32.4% vs 28.2% on 2B, 76.6% vs 71.8% on 9B) | Authors' numbers; thresholds set on the evaluation set for the 4-readout table | Needs our own training code |
| Attention-pattern evidence for sycophancy (head 10.2 attends to the user's answer: 0.75% honest vs 85.7% sycophant) | WSP Exp. 7 | Verified on the blog page; one toy model pair | Hypothesis H-A below |
| Subtract one direction from the weights to remove a behaviour | WSP Exp. 8 (looping, not deception) | Authors' result, 131,072 weights | Only as a causal check of a probe direction (H-B); not a detector |
| Read weights alone to find hidden behaviour | WSP Exp. 7 negative result | Failed 4/4 | Deprioritise |
| Larger models detect better in agentic steps (recall 28% -> 72% at 5% FPR, 2B -> 9B) | VRM Study 2 | Confounded: depth, width and heads differ | Use at most as a reason to start with 9B, not as a law |

## 3. Proposed improvements, ranked

1. **Evaluation protocol (cost: low, value: high).** Fixed FPR (1% and 5%), thresholds from an independent honest calibration set drawn from different scenarios than the test set, binomial-certified; per-family tables; equal-information text baseline (same transcript, same prefix) and a length-only control; label-shuffle and random-weight controls; bootstrap by scenario, never by row. Grounded in measured VRM failures.
2. **Fresh evaluation data (cost: medium).** Because of the training overlap, new scenarios are required (section 5). Source options: our own ACS-generated scenarios (open point: generation prompts and judge not designed yet), or a public set not in FIBS (not searched).
3. **Follow-up contrast score D with the published EFC/axial probes (cost: low once activations exist).** Pre-registered comparison of single-answer score vs D on the same rows.
4. **Aggregation and fusion sweep (cost: low).** Mean vs max vs soft-max pooling; best calibrated pair vs all heads. Pre-register the grid in advance; VRM's own post-hoc selections were optimistic by their own account.
5. **Label audit (cost: medium).** Re-judge the rows where the blind judge disagrees with a judge that sees the context; report the disagreement per family. Needs a judge budget and a hand-checked sample (human check does not exist in FIBS).
6. **WSP-derived hypotheses (cost: medium to high, value: unknown).**
   - H-A: an attention-mass feature (share of attention from the answer position to the user's suggested answer vs the question) improves detection in the sycophancy family (7,976 rows, contextual) over the probe alone. Testable on Qwen3.5-2B with `output_attentions`.
   - H-B: ablating the probe direction in the residual stream (as in Exp. 8, but as an activation edit, not a weight edit) changes the lying behaviour, which would make the probe a causal rather than only a decodable signal. VRM explicitly says its signals do not establish causality.
   - Not recommended: weight-only auditing, LLM-agent-as-weight-reader (needs a frontier model with no tools; only the blog's authors have it).

## 4. What we should NOT claim

- That any of these methods improves lie detection. None has been run by us.
- That FIBS validation AUROC (0.987-0.999) measures generalisation. It is in-distribution and the split is unspecified.
- That probes beat text. In the sources we read they mostly did not.
- That a probe detects a model's intent. Labels are behavioural and judge-based.

## 5. Experiment plan (draft for preregistration; commit as a plan via `kb new plan` BEFORE any data)

**EXP-1: calibrated probe vs text baseline on fresh data.**

- Question: on scenarios the published probes never saw, does a fixed-FPR probe detect more deceptive answers than an equal-information text baseline?
- Model/probes: Qwen3.5-2B (24 blocks, layers 7, 10, 13, 16, 19, 22) for the first pass; published `linear`, `mlp`, `efc`, `axial` probes at `WEIGHTS_REVISION`. Go to 9B only if the 2B run is feasible and informative.
- Data: N fresh scenarios, 50/50 labels per family, at least 6 families spanning contextual and world-knowledge; each scenario appears in exactly one of calibration / test. Minimum sizes: calibration at least 500 honest answers per family pool (needed to certify 5% FPR at delta=0.05 with the order-statistic rule; with 100 negatives the rule cannot certify 1%); test at least 200 per class per family for stable recall estimates. Open point: where the data comes from.
- Primary endpoint: recall at 5% FPR on the test set, threshold fixed on calibration, scenario-level bootstrap, 10,000 draws.
- Comparators: TF-IDF + length logistic regression on the same transcript (fit on calibration-disjoint training scenarios), mean token log-likelihood, flag-all control.
- Decision rule: "probe supported" only if the lower 95% bound of (probe recall - best text recall) is above 0 in the pooled test and in at least half of the families; otherwise report "no advantage shown".
- Secondary: D vs single score; mean vs max pooling; family-wise table; length-matched subset.
- Stop conditions: any FIBS row used for test makes the run invalid; changing the grid after seeing test labels voids the primary endpoint.
- Compute: activation capture for a 2B model on a few thousand transcripts. Hardware (GPU?) is an open point; CPU capture is plausible but slow and not measured. ACS/compute workflows can host the sweep, not the activation capture (needs a model server we have not set up).

**EXP-2 (only after EXP-1): retrain small probes with leave-one-family-out on FIBS** to estimate cross-family transfer. Requires activations for a subset of FIBS rows and our own training code (fibs-probes has none; VRM describes the recipe). Compare with the published probes on the held-out family only where the family was not in training (impossible for the published ones, hence the retrain).

**EXP-3 (hypotheses H-A, H-B):** run only if EXP-1 shows a usable baseline.

## 6. Open points that the sources cannot settle

- What "our" detection method is (section 0).
- Meaning of LDD in this setup. We applied only the dialectical-reasoning step; no `.ldd` config, loss log or E2E target exists in this repo.
- FIBSv1 licence; whether `main` equals the training revision `65dccf12`; how the 5,000 validation rows were selected.
- Whether VRM Study 2 and fibs-probes share data and code (suggested by matching architecture, not stated).
- WSP: the repo's claim of 1,125 audited tool-free turns is not on the blog page and was not verified.
- Whether the probes' training uses the final-token score or a pooled score for validation AUROC when comparing across architectures (the card gives one number per probe).
- Hardware for activation capture.

## 7. Next actions (need a decision from the operator)

1. Confirm the method definition (section 0) and the data source for EXP-1.
2. Approve registering EXP-1 as a preregistration plan in the knowledge base (it must exist before any data point).
3. Decide on hardware for activation capture.
