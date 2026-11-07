---
id: CONCEPT-0003
uid: 01M6V1YAEXAEGV9G4YX53QJ12T
title: 'LDD-governed calibrated probe lie detection: loss-driven method selection with a vault firewall'
status: proposed
created: '2026-11-07T13:07:38+00:00'
links:
- rel: derived_from
  to: IDEA-0001
- rel: related
  to: CONCEPT-0002
- rel: related
  to: CONCEPT-0001
---

# CONCEPT-0003 — LDD-governed calibrated probe lie detection: loss-driven method selection with a vault firewall

Status of this document: a method discussion and a proposal. Nothing in it is a measured result of ours.
No experiment has been run. Every number is tagged: [ours-computed], [authors-reported, unreproduced],
[from-memory, unverified] or [subjective prior].

## Context

### The two ideas to unify

1. **Repo idea (IDEA-0001, CONCEPT-0002, `notes/`).** Probe-based lie detection on a frozen LLM: a small
   classifier reads residual-stream activations and scores whether an answer contains a statement the model
   itself believes false. Published probes (fibs-probes: linear, MLP, EFC, axial) exist; the evaluation
   protocol of VRM (recall at a fixed false-positive rate, thresholds from independent calibration, an
   equal-information text baseline, group splits) is the measurement standard. CONCEPT-0002 adds
   Weight-Space-Programming hypotheses (H-A attention feature, H-B causal direction ablation) and a hard
   guardrail: the probe score is never a training or selection signal for the monitored model.
2. **LDD idea (the operator's).** Loss-Driven Development, "gradient descent for agents": an explicit loss
   function, measurement before every step, a hostile counter-argument before every step (dialectic),
   iteration budgets, plateau stopping, and an outer loop that changes the method itself with rollback on
   regression. Four loops: inner (code), refinement (deliverable), outer (method), CoT (reasoning chain);
   a rigor ladder L0 to L4; `define-metric` (a new metric is advisory until calibrated).

### ASSUMPTION A1 - what "LDD" means here (not stated in the task)

`LDD` is not defined in this repo (`notes/lie-detection-improvement-plan.md` section 6 lists it as open).
Most probable reading, checked against the operator's global instructions and the local plugin
(`/home/shumway/projects/loss-driven-development`, README and `docs/theory.md` read on 2026-11-07):
**Loss-Driven Development**. The operator's specific idea of how to apply it to lie detection is not written
down anywhere I could read. This concept therefore reads it as: *use the LDD discipline as the control
system that develops and selects the detector, not as a new detection signal.* A second reading, LDD's
CoT loop (dialectical chain-of-thought) as a signal, is kept as a low-prior hypothesis H3. If the operator
meant something else, sections "The Method" and "Hypotheses" need redoing; the dialectic and the validity
analysis stay.

### ASSUMPTION A2 - the detected phenomenon

"Lie" = the final answer contains a statement the model believes false (VRM definition). Omission and
misleading-but-true statements are out of scope. Labels in FIBSv1 are judge/construction labels, not human
verified [ours-computed, `notes/fibs-v1-data-inspection.md`].

## Dialectic

### Thesis

Combine the probe detector with LDD: let an LDD-style loop (measure, diagnose, step, remeasure; outer loop
over detector design choices) iterate the detector until a loss defined on calibrated recall is minimal.
The loop gives rigor and speed; the probe gives access to internal state a text judge lacks.

### Antithesis (strongest counter-case)

1. **Goodhart / adaptive overfitting.** Loss-driven iteration on a finite dev set is exactly adaptive data
   analysis. The more steps the loop takes, the more the winning configuration reflects dev noise. An LDD
   loop that reports its own final loss as success is an overfitting machine with good manners. The notes
   already document the failure: VRM's post-hoc selections were optimistic by the authors' own account, and
   VRM Study 1 had AUROC 1.00 but 0 of 32 detected at the frozen cutoff [authors-reported].
2. **The loss is not the construct.** Judge labels, construction labels and family priors are proxies for
   "model believes it false". Minimising loss on proxies can minimise detection of the proxy
   (length, author-model, template) rather than of deception. FIBSv1 shows weak shortcuts: length AUROC
   0.547, answer-writer model prior 0.594 [ours-computed].
3. **LDD's own evidence is thin for this use.** The LDD bundle's headline numbers (a `Δloss_bundle`) come
   from its authors' skill benchmarks on coding-agent tasks, not from detection problems. No evidence in
   hand that LDD improves a statistical detector. The rigor ladder has a token cost of 3 to 5 times for
   dialectical CoT by its own documentation, and CoT needs verifiable tasks; deception labels are not.
4. **Reflexivity.** The agent that runs the loop (an LLM) also wrote scenarios, ran judges and reads the
   results. Shared blind spots between developer, judge and detectee model families are a threat that no
   loop on the same agent can see.
5. **Wrong order.** Two unproven components (probe, LDD loop) combined before the first baseline is
   measured makes every outcome uninterpretable. The probe lost to text in most VRM comparisons we read
   (4 of 6) [authors-reported].
6. **Danger of a self-defeating use.** If the loop ever feeds probe scores back into training, prompt
   optimisation or weight edits of the monitored model, the probe stops being a measuring instrument. This
   is forbidden by CONCEPT-0002 and stays forbidden here.

### Synthesis

LDD is valuable only where it is a **discipline about measurement and selection**, not an optimiser pointed
at the detector's headline metric. Concretely:

- The loss is defined on **calibrated operating points** (recall at fixed FPR with an honest realised-FPR
  check), not on AUROC.
- The loop is **firewalled**: development data may be used by the loop without limit; a *vault* split
  (scenarios and one full family never seen) is opened once per preregistered claim. Without the vault, no
  claim is made.
- LDD's contribution is itself **tested as a hypothesis** (H1): does a gated selection procedure generalise
  better than a naive one on the same candidates? If not, the LDD part is dropped, the evaluation protocol
  stays.
- The probe is compared against **equal-information text baselines** from day one, and the study targets
  the regime where a probe has a *structural* information advantage (see "Where a probe can win").
- Realistic claim today: a rigorous evaluation harness and an honest answer to "does the probe beat text
  where it matters". Not: a better lie detector.

## The Method

### M0. Where a probe can win (scoping, before any build)

A black-box judge that sees the full transcript and context can detect a lie whenever the context contains
the truth-maker. In such "contextual" cases (48.8% of FIBSv1 rows by the card, unverified [authors-reported])
a strong text judge is the expected winner. A probe has a structural edge only where **truth lives in the
model's belief and not in the transcript**: world-knowledge statements the judge does not know, private
belief that differs from stated belief, or cases where a judge lacks the knowledge to decide. 365 identical
completions carry both labels in FIBSv1 [ours-computed], so text alone cannot decide those rows; they are the
natural test bed for context-contrastive evidence and the place where "text baseline" must receive the
*context* too (otherwise the comparison is rigged for the probe). Primary analysis is therefore stratified:
contextual vs world-knowledge/belief-gap families, reported separately and never only pooled.

### M1. Measurement layer (carries the scientific value; independent of LDD)

- **Operating point.** Recall at 1% and 5% FPR. Threshold from an independent honest calibration split by
  the binomial order-statistic rule (VRM). Minimum honest calibration negatives to certify the target at
  delta = 0.05 even with the extreme threshold: 59 for 5% FPR, 299 for 1% FPR [ours-computed:
  ln(0.05)/ln(1-alpha)]. The notes use 500 per pool for a less conservative threshold; plan on at least 500.
- **Realised-FPR check on the vault**, with a Clopper-Pearson interval. A threshold that does not hold its
  nominal FPR out of distribution is reported as a calibration failure, whatever the recall.
- **Secondary:** AUROC (comparability with sources), pAUC at low FPR, ECE of the probability scale,
  per-family tables, length-matched subset, cross-family drop (leave-one-family-out).
- **Uncertainty.** Cluster bootstrap by scenario (never by row), 10,000 draws; paired differences against
  baselines on the same scenarios.
- **Controls (must be at chance or the harness is broken):** label shuffle; random-direction probe; random
  weights; length-only classifier; flag-all.
- **Baselines (equal information):** (a) informed LLM judge that sees system prompt and context; (b) blind judge
  for reference only, not as the decision baseline; (c) TF-IDF + length logistic regression on the same
  transcript, fit on scenarios disjoint from calibration and test; (d) mean token log-likelihood;
  (e) a retrained linear probe on the same layers, to separate "published weights" from "probe class".
- **Data.** FIBSv1 cannot be the test: published probes trained on 152,980 of its 163,016 rows. Test data
  must be fresh scenarios (own generation, e.g. via ACS, with a family never in training) or a public set
  not in FIBS (not searched). Labels need a human-checked subset (open: budget); judge-only labels are
  reported as such.

### M2. LDD mapped onto the detector pipeline

| LDD loop | Parameter | Concrete instance here | Loss (all on dev, never on vault) |
|---|---|---|---|
| Inner, theta | pipeline code | activation capture contract (block output, bf16, no thinking), pooling, D-score implementation, thresholding | harness E2E: controls at chance, parity with published probe scores within 1e-4 on a fixture, no leakage test red |
| Refinement, y | the evaluation report | per-family table, intervals, controls, deviations section | reporting rubric violations (missing control, row-level bootstrap, pooled-only claim) |
| Outer, m | detector design choices | layer set, aggregation (mean / max / soft-max), D-score vs single score, fusion of few calibrated heads, probe class | dev loss `L = 1 - recall@5%FPR` averaged over families, plus penalty for realised-FPR miss and for cross-family drop; leave-one-family-out as the "N-task suite" |
| CoT, t | reasoning chain (research agent, or optionally the monitored model) | agent plans reviewed by dialectical antithesis; or H3 below | not used as a loss in v1 |

Rules that make the LDD part defensible instead of decorative:

1. **Loss declared before the loop starts** and written into the preregistration; changing it afterwards voids
   the claim (not the exploration).
2. **`define-metric` discipline.** Any new metric (for example "attention-mass share", H-A) is advisory until
   calibrated against an already trusted measurement on dev; only then may it gate a decision.
3. **Iteration and look budget.** Fixed number of outer-loop steps and a fixed candidate grid, committed
   before the first data point (CONCEPT-0001). Each extra look at dev costs statistical credit; the budget is
   part of the report.
4. **Rollback on regression** is implemented as: a change is kept only if the leave-one-family-out mean loss
   improves with a paired interval below zero, not on point estimates.
5. **Vault opened once** per preregistered claim, by a step that does not return per-example results to the
   loop. Nothing from the vault flows back into candidates.
6. **Thinking level.** L2 for harness code, L3 for the preregistration and the selection design (dialectic is
   cheap here; token cost is irrelevant next to a wrong claim); no dialectical CoT inside the loop (unverifiable
   steps, cost).
7. **Separation of roles.** Scenario generator, labeller and judge use different model families than the
   monitored model where possible; the developer agent sees dev results only.
8. **Never** use probe scores as training, selection or reward signal for the monitored model (CONCEPT-0002).

### M3. Experiments (each needs a preregistration plan BEFORE data, per CONCEPT-0001)

- **EXP-1 (the necessary baseline).** Published probes vs equal-information baselines on fresh scenarios,
  stratified by contextual vs belief-gap families. Primary endpoint: recall at 5% FPR, threshold from
  calibration, cluster bootstrap. Decision rule as in `notes/lie-detection-improvement-plan.md`: "probe
  supported" only if the lower 95% bound of (probe minus best text) is above 0 pooled and in at least half the
  families; else "no advantage shown".
- **EXP-2 (the LDD meta-experiment, H1).** Method-selection study: take one fixed candidate pool (layers x
  aggregation x probe class x D-score on/off). Procedure A: pick the winner by dev AUROC. Procedure B: the
  gated LDD procedure (calibrated loss, paired-interval keep rule, leave-one-family-out, budget). Repeat on
  many random dev/vault re-draws of scenarios (cheap, compute only) and compare the **generalisation gap**
  (dev score minus vault score) and the vault score. This measures what LDD adds to *selection*, which is
  the one thing LDD claims to improve. Needs activations for the candidate pool once; selection reruns are
  cheap and can run as an ACS/compute sweep.
- **EXP-3 (only after EXP-1 shows a usable baseline).** H2 (D-score), H3, then CONCEPT-0002's H-A, H-B.

## Hypotheses (falsifiable)

- **H0.** On fresh scenarios, recall at 5% FPR of the best published probe exceeds the best equal-information
  text baseline, pooled and in at least half of the families. Falsified if the lower 95% bound of the paired
  difference is at or below 0. [subjective prior that H0 holds in the belief-gap stratum: about 35%;
  in the contextual stratum: about 10%. Basis: VRM text baselines won 4 of 6 comparisons
  [authors-reported]; no evidence of ours.]
- **H1 (LDD's contribution).** Procedure B has a smaller generalisation gap than Procedure A, with a
  paired-bootstrap interval excluding 0 over at least 200 dev/vault re-draws, and B's vault score is not
  lower. Falsified otherwise; then LDD is dropped from the selection part. [subjective prior about 50% for
  a smaller gap, because any budgeted, calibrated procedure beats AUROC-picking on a small dev set; the
  interesting question is the size, not the sign.]
- **H2.** The follow-up contrast score D improves recall at 5% FPR over the single-answer score on the same
  rows. Cost: two forward passes per row. [prior about 40%.]
- **H3 (low prior, optional).** A black-box signal from dialectical self-consistency (thesis and antithesis
  of the model about its own claim) adds incremental information to the probe score in a logistic stacker
  evaluated on the vault. [prior about 15%; known risk: a model can produce consistent text while lying.]
- **H-A / H-B** as in CONCEPT-0002.

## Expectations: what is realistic

- **Achievable in a hackathon day:** the harness (capture, calibration, bootstrap, controls, baselines),
  the preregistration, and EXP-1 on a small scale (a few thousand transcripts, Qwen3.5-2B) if hardware
  permits; otherwise the harness plus a feasibility measurement on 100 transcripts. EXP-2 is feasible once
  activations exist.
- **Power.** To detect a recall difference of 0.40 vs 0.30 with 80% power, unpaired, one needs roughly 356 positives per
  group by the normal approximation [ours-computed]; clustering by scenario raises this, pairing lowers it.
  Detecting 0.30 vs 0.25 needs about 1,251 per group [ours-computed]. Small effects will not be resolved
  on a single day. A null result is the likely outcome for small effects, and it is informative only if the
  interval is reported.
- **Not achievable / not to be claimed:** a deployable detector; detection of intent; any improvement by
  WSP; any claim from FIBS validation AUROC (0.987 to 0.999, in-distribution) as generalisation; transfer
  to larger models from 2B and 9B; prevalence-transferable precision (FIBS is exactly 50/50).
- **Most probable outcome (subjective):** the probe matches or beats text in the belief-gap stratum with a
  wide interval, loses in the contextual stratum, and degrades across families. The publishable
  contribution is then the protocol plus the stratified comparison, and a measured statement on selection
  (H1).

## Trade-offs (Zielkonflikte)

| Conflict | Side A | Side B | Position taken |
|---|---|---|---|
| Rigor vs speed | LDD loops, preregistration, vault | one day, 2B model, small data | rigor on claims, speed on exploration; scratch runs are labelled exploratory |
| Peak accuracy vs calibration | maximise AUROC / recall | hold nominal FPR out of distribution | calibration wins; recall is reported at the calibrated point |
| Generality vs accuracy | leave-one-family-out loss | in-distribution gains | generality in the loss |
| Interpretability vs performance | few heads, sparse features | EFC/axial, all layers | start with few calibrated heads; larger only if they win on the vault |
| Information fairness | text baseline sees context | probe sees activations | text baseline must see everything the probe's input contained |
| Optimisation vs measurement | iterate on dev | never touch vault | vault firewall |
| Scientific claim vs LDD narrative | "LDD improves detection" | "LDD is a process" | only H1 supports the first; otherwise the second |

## Validity threats

- **Construct.** Behavioural/judge labels vs belief; Qwen-belief vs ground truth; a model that "believes"
  falsehoods from training; probes may track "statement is false in the world" rather than "I say what I
  believe false". Mitigation: belief elicited separately from the deceptive context where possible; report
  both label sources.
- **Internal.** Near-duplicate leakage (28,114 rows share a completion with another row [ours-computed]),
  answer-writer-model shortcut (0.59), length, template; probes calibrated on one prompt format breaking on
  another (VRM changed-wording split: score scale shifted 0.315 to 0.151 [authors-reported]).
- **Statistical.** Garden of forking paths in the outer loop; multiple comparisons across families and
  readouts; bootstrap by row inflates confidence; dev reuse. Mitigation: grid and budget committed first,
  scenario-level clustering, vault once.
- **External.** Two Qwen sizes; scripted/synthetic deception; English; 50/50 prevalence; capture contract
  (vLLM, bf16, no thinking) is strict, so real agent traffic differs.
- **Reflexivity / AI-as-researcher.** Developer, labeller and judge share biases; mitigated by role separation
  and a human-checked label subset, not removed.
- **Ethical / dual use.** A better detector can be used to train models against it. The guardrail of CONCEPT-0002
  stays: detector output is a measurement only.

## Operationalisation checklist for the first preregistration

1. Fix: model (Qwen3.5-2B), layers (7, 10, 13, 16, 19, 22), capture contract, `WEIGHTS_REVISION`.
2. Fix: data sources, families, split by scenario, vault family, human-checked subset size.
3. Fix: loss definition, candidate grid, outer-loop budget, keep rule, stopping rule, analysis code hash.
4. Fix: baselines and their exact inputs, controls and their expected outcomes (chance).
5. Commit as `kb new plan` BEFORE any activation is captured. Data to `data/experiments/<slug>/` with a manifest.

## When to use / When NOT to use

Use as the design frame for EXP-1 and EXP-2 and as the vocabulary for the discussion with the operator.

Do not use:

- as a claim that LDD improves lie detection (that is H1 and untested);
- before the EXP-1 baseline exists as the basis for combining anything with WSP;
- to optimise the monitored model against the detector, in any form;
- as a substitute for human-checked labels.

## Evidence

Verified by me in this session (read from disk): repo notes and KB entities; FIBSv1 inspection outputs
(ours, from the previous session, not re-run); LDD README, `docs/theory.md`, skill texts for dialectical-cot
and define-metric in `/home/shumway/projects/loss-driven-development`; the arithmetic for 59/299 negatives and
the power figures (computed here with the Python standard library).

Authors-reported and not reproduced: all fibs-probes, VRM and WSP numbers, all quoted from the repo notes
(secondary source). Not independently re-read in this session.

Literature on lie / truth detection from internal states, **from memory, not checked, do not cite without
verification**: Azaria and Mitchell (2023, "The Internal State of an LLM Knows When It's Lying"); Burns et al.
(2022, contrast-consistent search); Marks and Tegmark (2023, "The Geometry of Truth"); Pacchiardi et al.
(2023, black-box lie detector with unrelated follow-up questions); Goldowsky-Dill et al. (2025, probes for
strategic deception; referenced by the VRM README); critiques that probes generalise poorly across
distributions (for example Levinstein and Herrmann, 2023). My recollection is that a common finding is
in-distribution success with weaker out-of-distribution transfer, which is consistent with the notes here but
is not verified.

Update 2026-11-07 (after the first run): PLAN-0001 / NOTE-0003 measured one slice of this concept. Published FIBS probes
read the prompt context more than the false statement in our 2x2 design (AUROC 0.56-0.59 on 65 held-out facts, followup variant);
LLM-proposed probe edits did not beat random edits (difference -0.015, 95 % CI -0.085 to +0.062). The own-model plausibility
baseline ranks far better (AUROC 0.867). H0 (probe beats equal-information baseline) was therefore NOT supported in this setting;
H1 (LDD-gated selection) was not tested. Not evidence: any measured improvement from combining probes with LDD.

## Open questions for the operator

1. Is assumption A1 right, and what exactly is your LDD idea for lie detection (selection discipline, CoT
   signal, something else)?
2. Data source for fresh scenarios (own ACS generation vs a public set), and budget for a human-checked label
   subset.
3. Hardware for activation capture (GPU?); not measured.
4. Is EXP-2 (the meta-experiment on selection) the right scientific claim for LDD, or do you want LDD judged
   by the detector's end performance only (weaker design, more confounded)?
5. Language of repo artifacts: this file follows the repo rule (English); CONCEPT-0002 is German. Translate?

## Operator Notes
