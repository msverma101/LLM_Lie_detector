# Presenting the five-minute follow-up

This guide assumes a **five-minute talk**, in addition to the five-minute research
budget. If the workshop allows only two minutes, use slides 1, 4 and 5.
The experiment plan is [research-plan-5min.md](research-plan-5min.md).

## The central message

Explain one measurable question: which cached token views help a small detector
generalize between the conversation families in this sample?

Use this opening, with the final sentence adjusted to the actual outcome:

> A lie detector may perform well on familiar conversations and miss other kinds of
> falsehood. Our earlier run improved its ranking score by combining several views
> of the model's activations. We used five more minutes to check which views account
> for that improvement.

If the cache is missing, say: "We audited the reported numbers and identified what
is required for a replay." Do not present historical arithmetic as a new experiment.

## Five slides, five minutes

| Time | Slide | Evidence to show and point to say |
|---|---|---|
| 0:00-0:40 | **Falsehood detection on unfamiliar conversations** | One synthetic tool-report example: the tool says tests failed, the answer claims they passed. Label it illustrative. Define the task as predicting FIBS labels. Avoid treating all mistakes or hallucinations as knowing lies. |
| 0:40-1:25 | **The earlier experiment** | Qwen3.5-2B, reported 2,940 transcripts, six observed test families. Show the historical baseline 0.8059 and selected iteration 9 at 0.8453. Cite the result table and label this a single earlier run. Explain that AUROC measures ranking, not percentage of lies caught. |
| 1:25-2:15 | **The pooling ablation** | Show the five fixed readouts from the plan. Explain that every condition reuses the same 18 trained sub-probes, layers, transcripts and split. State A1 minus A0 as the primary comparison. Mention that using additional pools increases the number of classifiers in the readout. |
| 2:15-3:50 | **Results on the observed families** | Fill a compact table with baseline, all pools and last-plus-answer AUROC. Add their actual test TPR and FPR at the validation-calibrated 5% threshold. Show the six paired family changes if space permits. Say whether the primary gain is positive, negative or unresolved. Use measured numbers only. |
| 3:50-5:00 | **What this result supports** | State the practical implication of the measured result. Explain that this is a retrospective audit of an already inspected test sample. Name the next decisive test: new scenario groups, independent calibration and ordinary controls with the same features. |

With no new run, slide 4 can show the historical decomposition:

| Historical step | Test macro-AUROC | Increment |
|---|---:|---:|
| Last-token baseline | 0.8059 | |
| Three pools, uniform average, iteration 4 | 0.8277 | +0.0218 |
| Transfer-weighted pools, iteration 9 | 0.8453 | +0.0176 |

Source: [checked-in result table](../experiments/research_loop/results/results.tsv).
Label the table **historical results, no independent replay**. The full increase is
0.0394 AUROC units, or 3.94 AUROC points. It is not a 3.94% increase in accuracy.

## How to make the evidence readable

- Use a plain 16:9 layout, approximately 32-point slide titles and 24-point body
  text. Keep one claim or setup step on each slide. Put detail in speaker notes.
- Use the same colors for the same conditions throughout. Print values and split
  names directly beside the marks. Keep validation and test visibly distinct.
- Label every result as historical, newly measured, or proposed. Show the number
  of families and examples. Name the sampling unit behind every interval.
- If using bars, start the axis at zero. For small AUROC differences, a dot plot
  with a clearly labeled restricted axis is easier to read. Include a 0.5 reference
  when useful and avoid decorative precision beyond the evidence.
- Use the existing progress SVG only for historical context. The old layer-weight
  SVG is a predictive readout summary. It does not localize a causal lying mechanism.
- Put a short source citation on the relevant slide and the exact Git SHA and run
  artifact path in the appendix or speaker notes. Keep keys and sensitive Colab UI
  out of screenshots. Present saved artifacts rather than depending on a live API demo.

## What to say for each outcome

| Measured outcome | Defensible interpretation |
|---|---|
| Pooling gain reappears and last-plus-answer retains it | The answer's pooled representation is a useful candidate feature in this sample. A fresh-data test should examine whether the final-turn pool is necessary. |
| AUROC rises, test FPR also rises at the calibrated threshold | Ranking improves, with an unresolved false-alarm tradeoff. |
| Differences are small or inconsistent across families | This quick audit does not resolve which view generalizes better. |
| The gain does not reappear | Report the replay failure and check cache identity and evaluation differences before attributing a cause. |
| No cache or incomplete fit | Show the historical audit and the specific missing input. State that no new quantitative experiment completed. |

## Likely audience questions

**What is new here?** The proposed contribution is a controlled attribution of an
existing empirical gain and a stronger cheap baseline. Pooling and ensembling are
established methods. Novelty across the literature has not been established.

**Does Claude beat normal optimization?** The existing experiment lacks a completed,
matched ordinary-search comparison. The pooling ablation cannot answer that.

**Did you make the language model more honest?** We changed the readout of a frozen
model. We did not measure a behavioral change in the language model.

**Did you find where lying happens?** Predictive probe weights support a feature
hypothesis. Causal localization requires an intervention and an independent behavioral
test. Those are later experiments.

**Was the test untouched?** The earlier loop omitted test scores from Claude's prompts.
It computed and logged them each iteration, and the published results are now known.
This follow-up is retrospective.

Rehearse once with a timer. Finish by stating the actual result and the next experiment
it justifies. Keep the full experiment log and limitations available for questions.
