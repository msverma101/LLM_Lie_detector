# Adaptive detector-only WSP: prospective protocol, 8 November 2026

Recorded before fresh data capture and detector evaluation. This is a new pilot,
not a reproduction of the published EFC probe. The monitored Qwen model is frozen.

## Question and success

Does an LLM's explicit, diagnosis-guided coefficient editing improve a fixed
activation detector over its initial checkpoint, and over ordinary fitting?
Primary measure: recall on FIBS-labelled deceptive final answers at a calibrated
low false-alarm operating point. All arms use nominal calibration alpha=0.025;
the intended test FPR ceiling is 0.05. Report actual FPR, one-sided 95% exact
binomial FPR upper bounds (descriptive if clusters remain), and paired source-group
bootstrap intervals. A positive result requires a positive 95% interval for the
recall difference against the original and an FPR upper bound <=0.05. Superiority
to ordinary fitting is a separate, stronger claim. Also report nominal-alpha 0.05
as secondary, without selecting a winner on the final test. No positive result is
required. No test-driven follow-up is counted as confirmation.

## Fresh data and fixed representation

Use model Qwen/Qwen3.5-2B revision
15852e8c16360a2fea060d615a32b45270f8a8fc and dataset AlignmentResearch/fibs-v1
revision 4ad7414fc2e1cc8d694b160e360722a13bee718e. Seed 20261108.
Text-only renderer is copied from the previous pilot and explicitly remains a
paper-based adaptation: tools are serialized, reasoning sidecars are omitted,
supplied completions are teacher-forced, and labels do not establish Qwen's intent.
Only last-answer-token activations at decoder blocks 7/10/13/16/19/22 are used.
Complete rendered inputs must fit 2048 tokens; no truncation. Record all examined
length/renderer exclusions and the prefilter restriction separately.

Training families: anti, doluschat, fever, instructed, lies, multi, persona_prefill,
synthetic, wikihow, wikihownest. Diagnostic: liarsbench, mask, original, ultrachat.
Validation: b1-bare, b1-followup, caving, negotiation. Target: among, harm, medqa,
model, sandbagging, werewolf. Family identities are already known from the pilot;
new rows are not an untouched-domain claim.

Connect rows by exact source pair ID, explicit run/game prefix, normalized first
user content and exact transcript/completion identity. Drop connected groups
spanning train/diagnostic/validation/target families. Exclude previous pilot rows
and groups, including recognizable episode prefixes and first-user hash aliases.
Report uncertain semantic linkage; do not assert universal scenario independence.
Stable group hashing splits target groups 50/50 into honest-only calibration and
test. Calibration positives are unused. Aim for 80 train rows/family, 100 rows per
diagnostic/validation family, and 100 honest calibration plus 100 honest and 100
deceptive test rows per target family. Balanced quadrant caps are computational
sampling, not natural prevalence. If cells are sparse retain them and report counts.
Do not inspect final prediction scores until the lock; publish actual group counts.

## Baseline and fair controls

Fit six L2 logistic probes (C=0.01) on training only, with fixed training means and
scales and fixed equal ensemble weights. Freeze representation/scaler/ensemble
weights throughout this study. WSP may change W and b, preserving capacity; no
pooling, layer selection or architecture changes. Verify sklearn score replay.
Use an identical refit on train as a replay control, and ten conventional C choices
on both train and train+diagnostic labels: 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,
0.3,1,3. Same representation and ten validation selection opportunities per arm.
Extra-data fitting is labelled explicitly; diagnostic performance there is in-sample.
Controls stay hidden from the agent until freezing.

## Ten genuine adaptive edits

The executing agent is Codex in this conversation, not an unreported Claude API.
It may inspect tensors with Python and view diagnostic errors, correct controls,
per-neuron contributions, lengths and language/source summaries. It authors a free
Python edit function for each attempt; no fixed menu of ten candidate operations.
No gradient, optimizer, solver, search over fitted coefficients or automatic
label-loss minimization is allowed in the WSP arm. Data-assisted direction edits
must be labelled as such, not presented as pure weight-only reasoning. All ten
attempts count even if they fail or are reverted.

Each attempt must durably record before execution: current parent, observed failure
and evidence IDs, mechanism hypothesis, quantitative prediction about named
failure/control cases, precise tensor change, and possible side effects. Execute
the edit in a separate saved module, validate frozen components and finite arrays,
then evaluate named diagnostic cases and aggregate diagnostic/validation outcomes.
Explicitly decide keep/revert. The next hypothesis is written only after that
feedback is inspected. Candidates normally edit the incumbent; a return to the
original must have a recorded reason. The original is eligible throughout.

Selection: macro recall at nominal 0.025 on validation with a temporary honest
validation threshold. Require +0.005 to replace the incumbent; ties within 0.005
use macro-AUROC, then earlier checkpoint. If none qualifies keep the original.
This validation feedback is tuning, not final calibrated evidence. A mechanism
prediction and measured score change are reported separately from aggregate gain.
For each real edit use 20 norm-matched random W/b direction controls around the
same parent (component/block-wise norms); no random-control feedback drives the
next edit. Freeze the selected ordinary and WSP candidates before final opening.

## Final evaluation and artifacts

Before final evaluation save checkpoint hashes, immutable input contract, all ten
specifications, free edit modules, diagnostic case records, contribution summaries,
feedback, keep/revert decisions and counts of prediction queries. Use finite-sample
order-statistic thresholds on the separate honest calibration groups. Evaluate
original, selected WSP, the ordinary train-only and extra-data winners, and the
selected edit's random controls on the same final test. Preserve an unedited parent
if the selected edit is not applied to the original. Report confusion counts,
per-family outcomes, AUROC, recall/FPR, 2000 paired group-bootstrap draws jointly
resampling calibration/test (conditional on familiar families), and intervals for
WSP minus each comparator. Record all exclusions and limitations. Repeated rows,
missingness and selection opportunities are disclosed. Freeze prevents further
candidate edits after test access, not cryptographic isolation from the agent.

Store raw last-token activations, cases, metadata, weights and all scripts in a
durable downloadable archive with checksums. Verify score replay locally; if raw
activation or full GPU reproduction is unavailable, disclose it. Synthetic fixtures
only verify software. After opening, publish the negative or positive report and a
five-criterion evidence audit. Independent study replication remains a separate task.

## References

- Telling FIBS: https://arxiv.org/pdf/2610.12445v4
- Official FIBS code: https://github.com/AlignmentResearch/fibs-probes
- Pinned dataset: https://huggingface.co/datasets/AlignmentResearch/fibs-v1
- Takehi, Teoh, Zhang and Khattab, Sparks of Weight-Space Programming (2026):
  https://rikiyat.github.io/blog/weight-space-programming/
  This motivates diagnosis and direct editing; it does not demonstrate lie-detector efficacy.
- Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills, arXiv
  2609.00065, version 2 checked 8 November 2026:
  https://doi.org/10.48550/arXiv.2609.00065
  Procedural appraisal support, not empirical evidence for this hypothesis.
