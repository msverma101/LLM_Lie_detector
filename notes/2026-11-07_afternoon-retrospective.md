# Analysis of the workshop afternoon, 7 November 2026

**The afternoon produced a working research system and a promising empirical result.**
The clearest reported finding is that a richer ensemble of activation probes ranks
FIBS-labeled falsehoods better than the local last-token baseline on six held-out
conversation families. The project also acquired the infrastructure to investigate
that result further. The evidence does not yet establish an advantage over ordinary
optimization, a causal mechanism for lying, or successful direct weight editing.

This account covers the documented afternoon **so far**, from the first repository
commit at 13:19 through the five-minute plan at 15:34 CET. The local snapshot was
inspected around 15:36 CET, before the workshop's recorded 17:00 deadline. It cannot
account for later work or unrecorded conversations. All times below use Europe/Berlin.

## Evidence and attribution

The committed sources are the Git history through `73c44dc`, the
[kickoff note](2026-11-07_kickoff-discord.md),
[original idea](2026-11-07_idea_probes-x-weight-space.md),
[research question](../docs/research-question.md),
[progress log](../experiments/research_loop/PROGRESS.md),
[results write-up](../experiments/research_loop/RESULTS.md),
[iteration table](../experiments/research_loop/results/results.tsv), and executable
probe, loop and evaluation code. The accompanying
[evidence snapshot](2026-11-07_afternoon-evidence.json) records source hashes,
commit times and arithmetic derived from the iteration table.

Local untracked drafts were also inspected: a closing-talk script, a robustness
runner and frozen baseline source, and a separate direct-weight-editing protocol.
These show work in progress. Their presence does not demonstrate that an experiment
ran or that a result was obtained. They remain outside this commit. No activation
cache, full run logs, fitted checkpoints, ordinary-search results or robustness
outputs were present at their expected local paths during inspection.

Commit timestamps establish when material entered this repository, not the exact
time an experiment started. Progress notes report events and timings that cannot
all be reconciled. Git author identities are sometimes placeholders, so this review
attributes conclusions to artifacts rather than assigning personal credit or blame.

## How the afternoon developed

| Time, CET | Evidence | What changed and why it matters |
|---|---|---|
| 13:19-13:36 | Initial commits, scaffold, literature and kickoff notes | The team assembled two motivating sources and a place to collaborate. The initial scope was deliberately open. The 13:00 Discord start appears in later prose, not a contemporaneous repository record. |
| 13:44 | `723ebbe`, research-question proposal | The idea became testable: distinguish an honest/lying model pair using weights alone, probe scores alone, or both. This proposed a genuine combination of the two sources, but required model pairs and a trustworthy probe. |
| 13:52-13:54 | `5148fe2`, `ae45086` | A plain-language explainer and illustrated README made the project accessible. Their ambitious framing later needed updating to match the narrower experiment. |
| 14:18 | `6e9cdc4` | A different executable question emerged: can Claude improve a probe's generalization by proposing code changes? Activation caching, a fixed harness, a baseline and an agent loop arrived together. |
| 14:27-14:31 | `178ffa1`, `6c5c564` | An end-to-end Colab test was reported. Tool-result endings, token limits and sampling were repaired, followed by a character-length prefilter. These fixes affected which families could enter the study. |
| By 14:51 | `f16a482` | A progress snapshot recorded the baseline and first two proposals. It reported a 2,940-row cache taking 12.8 minutes. The note's internal heading says 15:00, showing that headings are approximate. |
| 15:11 | `3253ca9` | The completed 15-proposal loop and selected iteration-9 result entered the record. Two proposals were kept. The log reports approximately 22 minutes and no crashes. |
| 15:23 | `8dc5539` | A substantial write-up, summarized table, winning probe code and SVG figures made the result reviewable. Raw Colab artifacts were still absent from the checkout. |
| 15:26 | `846bcd2` | Error analysis, ordinary random search and fixed-threshold TPR/FPR reporting were added. These address important gaps, but their addition happened after the first reported result. |
| 15:34 | `73c44dc` | A five-minute follow-up plan and presentation guide were committed after the requested budget narrowed from 20 minutes. This is a proposal, not another completed experiment. |
| At inspection | Untracked local drafts | Robustness checks, a closing talk and a direct-weight-editing protocol were being prepared. No completed outputs establish their outcomes in this snapshot. |

The write-up places the loop start at 14:56, gives a 22-minute duration, and places
the result at 15:10. Those entries conflict: 14:56 plus 22 minutes is 15:18.
The earlier 14:51 commit already records initial iterations. Preserve the approximate
sequence and reported durations, but do not invent a precise minute-by-minute history.

## The scientific pivot

The original question concerned **auditing language-model weights with help from a
lie probe**. Its main comparison required trained honest/lying pairs, controlled
auditor access and independent behavioral ground truth. None of those arms has a
completed result in the record.

The afternoon's working experiment instead asked whether **an LLM can propose useful
changes to the code of a small activation classifier**. The source language model
remained frozen. Conventional optimization trained each proposed classifier.

This was a practical reduction in dependencies: one expensive feature-capture stage
supported many cheap classifier experiments. It made a measured workshop result
possible. Its cost was a change in the research question. The original weight-space
question remains open, and there is no recorded team decision establishing that the
new experiment answered it. Future documents should introduce the actual experiment
first and present the earlier idea as motivation or a separate proposal.

## What the result shows

The recorded setup is Qwen3.5-2B, three token pools at six decoder layers, 2,940
transcripts from 27 FIBS families, and a 14/7/6 observed train/validation/test family
split. The configured `longcontext` test family dropped out under the length filters.
Labels describe the final transcript completion according to FIBS. Mostly off-policy
transcripts do not independently establish what this particular smaller model believes.

| Recorded condition | Validation macro-AUROC | Test macro-AUROC | Selection status |
|---|---:|---:|---|
| Iteration 0: six last-token probes, uniform mean | 0.8493 | 0.8059 | Baseline |
| Iteration 4: 18 probes across three pools, uniform mean | 0.8815 | 0.8277 | Kept |
| Iteration 9: those views weighted by training-family grouped CV | 0.8868 | 0.8453 | Kept, final selected method |
| Iteration 11: learned nonnegative stacker | 0.8888 | 0.8529 | Discarded under the validation keep rule |

The selected method's gain is **0.0375 on validation and 0.0394 on test**. These are
AUROC differences, not accuracy improvements or fractions of lies caught. The iteration
table contains 15 proposals, two accepted updates and 13 discarded proposals. The
progress log, rather than that summary table, supplies the claim of zero crashes.

Two historical steps explain the numerical change:

- Adding pools coincides with a test gain of `0.8277 - 0.8059 = 0.0218`.
- Adding transfer-based weights coincides with another `0.8453 - 0.8277 = 0.0176`.

The weighting step contributes nearly as much of the recorded test gain as pooling,
despite its smaller validation increment, 0.0053 versus 0.0322. This makes both steps
worth checking. It is a sequential comparison, not a factorial ablation that separates
all interactions. The winning method also uses more fitted classifiers and CV fits
than the baseline, so the gain cannot be assigned to reasoning alone.

Iteration 11 has a test score 0.0076 above the selected method. Selecting it now for
that reason would use the test set for tuning. The rounded validation figures place
its gain at the strict 0.002 keep boundary. Full-precision logs are needed to explain
the exact boundary decision. The threshold is an engineering heuristic, not a test
of statistical significance.

The write-up reports gains across all six observed test families, with the largest
reported gain in `among`, about 0.098. Those are encouraging descriptive findings.
Per-example predictions are needed to quantify paired uncertainty or verify them
independently. No confidence interval can be recovered from the macro-AUROC table alone.

## What worked well

**Caching made iteration affordable.** The reported 12.8-minute capture separated
the GPU workload from downstream probe research. The loop then explored 15 ideas
in approximately 22 minutes. These are reported component timings, not the complete
afternoon's cost. Setup, debugging and reporting also consumed time.

**The baseline exposed a meaningful weakness.** Near-perfect training AUROC alongside
0.8059 test macro-AUROC suggested a generalization problem worth studying. Family
splits and macro averaging made the evaluation more relevant to unfamiliar settings
than a simple random row split. They do not by themselves prove the cause of the gap.

**The harness constrained the search.** Candidate selection used validation scores,
and prompt construction omitted test results. Prediction received neither labels nor
family names, and a subset check tested batch dependence. The winning source is short
enough to inspect. These are useful design choices and a good foundation for a stronger
study, although they are not a complete data-isolation audit.

**Debugging addressed data coverage.** Accepting tool-result endings and increasing
the token cap recovered important examples. The length prefilter avoided spending
the budget tokenizing unusable transcripts. The remaining family loss was disclosed.

**The result was made understandable.** The explainer, tables and winning probe source
give a new collaborator a useful entry point. The later addition of controls and
operating points shows that the investigation began addressing its own limitations.

## What limited the afternoon's scientific yield

| Limitation | Consequence | Next action |
|---|---|---|
| The ordinary-search control has code but no recorded result | The afternoon shows a candidate improvement, not that Claude improves probes better than conventional alternatives | Compare the same pools, ensemble capacity and tunable parameters under measured time and evaluation budgets. The existing single-pool random search has a narrower space. |
| The same seven validation families guided 15 proposals | The selected validation score is optimistic after adaptive search | Freeze the method and test new scenario groups excluded from the original training and selection. |
| Test scores were computed/logged each iteration, omitted from prompts | Procedural separation exists, but the repository cannot prove nobody inspected scores. Published scores are now known | Separate development from final evaluation in future runs. Describe new analysis of this sample as retrospective. There is no evidence here that the original selection used test scores. |
| No measured uncertainty in the committed result | The write-up's “about two standard errors” cannot be treated as an established statistical result | Use paired predictions, retain scenario groups and report what each interval resamples. Six test-family clusters provide limited evidence. |
| Validation also supplies calibration thresholds | Operating-point estimates reuse selection data | Reserve independent calibration groups for future confirmation. Report actual test FPR alongside TPR. |
| The archive contains summaries and source, not the full experiment state | Independent replay and audit are incomplete | Preserve the cache, IDs, revisions, fitted weights, every candidate, score arrays, API metadata and raw evaluation logs outside the transient Colab runtime. |
| The baseline adapts the paper's architecture and preprocessing | This is not an exact published-probe reproduction | Name it a local LinearEnsemble-style baseline and document the rendering, normalization and checkpoint differences. |
| Cache metadata lacks stable scenario IDs and pinned source revisions | Exact provenance and related-example overlap remain unresolved | Recover identifiers from the original data/run before claiming group independence. Avoid assuming today's Hub revision generated the cache. |

Several implementation details deserve follow-up without being diagnoses of the
completed run. `make_split` replaces nonfinite values rather than failing. Unknown
families default into training. Cached chunks resume by filename without checking
that model and sampling settings match. Generated probe code has no enforced
filesystem isolation. The current record does not show these mechanisms corrupted
the result, but each weakens the guarantees of an unattended repeat.

## How the ongoing work should be interpreted

The untracked robustness runner proposes paired bootstraps and family re-splits.
That can make the original result easier to assess. Re-splitting the same sample
after selecting the method is still retrospective. Its overlapping splits also
do not justify treating split scores as independent new studies. Changing only a
deterministic classifier seed on the same data is not independent data replication.

The direct-weight-editing protocol is a more substantial separate study. It
distinguishes detector improvement from an advantage over ordinary optimization
and proposes separate calibration. It needs compatible saved artifacts and more
time than the current five-minute budget. Its existence is progress in experimental
design, not evidence for direct editing efficacy.

The closing-talk draft communicates the afternoon's breadth but needs three
scientific corrections before presenting:

1. Replace “84.5% of the time” as a detection claim. AUROC 0.845 does not specify
   recall or accuracy at an operating threshold.
2. Replace “where the lie lives” with a description of the ensemble's predictive
   layer preferences. The reported concentration at layers 10-16 does not establish
   a causal circuit or a successful intervention.
3. Qualify “fully reproducible” and statistical certainty. The local checkout lacks
   the necessary run state, cost records and measured intervals.

These observations concern an unfinished local draft. This retrospective does not
edit or publish that draft.

## Contribution and next-session priorities

The defensible contribution so far is a small-model empirical result, a readable
candidate method and an accessible research workflow. The potentially useful feature
is choosing pooled activation views according to transfer between training families.
That could improve inexpensive monitoring baselines if it survives new data and
stronger controls. Ensembling and grouped cross-validation are established techniques.
The afternoon does not establish literature-wide novelty or frontier deployment value.

The remaining questions are distinct and should receive separate experiments:

| Question | Evidence that would answer it |
|---|---|
| Which pooled views produce the gain? | Fixed readout ablations using identical component fits and examples, as in the [five-minute plan](../docs/research-plan-5min.md) |
| Does Claude add value beyond ordinary search? | A matched feature/parameter search space, comparable resources and repeated runs, with final candidates selected without test feedback |
| Does the gain transfer to new data? | New independent scenario groups, frozen method and calibration, paired evaluation |
| Does a better AUROC improve useful monitoring? | More true positives at a documented threshold, with measured false positives and uncertainty |
| Do weights and probe signals jointly help an auditor? | The original controlled weights-only/probe-only/combined comparison on independently evaluated model pairs |
| Can direct parameter edits improve the detector? | Saved before/after weights, ordinary controls with the same capacity, and independent final evaluation |

Prioritize preserving the Colab state first, because losing it makes every other
question slower to answer. Then run one bounded ablation and report its outcome,
including an inconclusive result. Complete matched controls before expanding claims
about automated research. Keep direct weight editing and causal intervention as
explicit later studies, each with its own success criterion.

A fair closing statement for the afternoon is:

> We built and ran a constrained loop that proposed changes to activation-probe code.
> The selected method improved reported test macro-AUROC from 0.8059 to 0.8453 on six
> held-out families. The next work is to preserve the experiment state, attribute the
> gain and test it against matched controls on independent data.
