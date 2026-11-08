# Original WSP method: source audit

Checked 8 November 2026. Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar
Khattab (MIT CSAIL). The source is a research blog publication, not a peer-reviewed
lie-detection result: https://rikiyat.github.io/blog/weight-space-programming/.
Released evidence is pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`
of https://github.com/RikiyaT/lm_understand-logs. Consult raw sources for details;
the accompanying audit recomputes record counts and preserves file hashes.

## Different experiments have different rules

The common element is a general-purpose LLM reasoning over weights and directly
writing them without an optimizer/training loop. An adaptive loop is not universal.

| Experiment | Editor access/action | Result type |
|---|---|---|
| 1, games | Same architecture; raw weights and input/action description | Identify trained task |
| 2, diffusion | Raw weights; no tools or forward passes | Identify generated shape |
| 3, language | Middle blocks only, Python, no executing layers | Language/copying-head identification |
| 4, car | Weights, scene, one failed trajectory; one reply without tools | Controller repair, evaluated later |
| 5, MinAtar | Rules/source, zero weights; no rollouts/tools | Write policies from scratch |
| 6, Go | Python and 0–50 batches of practice feedback | Program a policy; held-out opponents |
| 7, sycophancy | Weight-only attempt failed; attention access relaxed task | Identify a trained behavior |
| 8, Qwen | Weights, practice answers, Python, generation/teacher forcing | Adaptive repetition repair |

Sources: [overview](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/README.md),
[car](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/README.md),
[Go](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp6_go_from_scratch/README.md),
and the original post's numbered sections. Do not merge their permissions into one
fictitious mandatory protocol.

## Experiment 8: actual environment

Read the [initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt)
and [README](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md).

Editor: `gpt-6-astra`, maximum reasoning, Codex CLI with normal built-in tools
disabled. A separate harness exposed the experiment tools. Target: Qwen3.5-0.8B's
text component, 24 layers, width1024, SwiGLU width3584; base checkpoint revision
`2fc06364715b967f1860aea9cf38778875588b17`.

The persistent CPU Python kernel had no network/GPU. Limits were300 code runs,
900seconds/96GB per execution. Load-original/save/drop helpers exposed tensors.
The kernel could analyze weights and recorded activations, but only the forward
tool could execute the model. Handoffs preserved task, notes and current edit.

Practice material:7469 GSM8K train,7275 MATH train,714 AIME problems (answers only
for AIME). Practice labels were allowed; final benchmark sets were unavailable.
Calling this label-free or pure weight-only inference would be inaccurate.

Forbidden: training, autograd/backward/optimizers, iterative fitting to label loss;
changing tensor names/shapes, tokenizer/template/decoding; editing final norm or
embedding/output head; **a patch that only scales an entire tensor**. Indiscriminate
early thinking closure was outside the task.

The release says the complete math agent harness/server is not included. Its
`harness/sandbox.py` is the Go sandbox. Prompts document the math rules, but are
not independent verification of unavailable enforcement internals.

## Feedback and budget

Each successful forward request used one pass; failed requests did not count.

- Generation: up to512 problem-samples, or256 per model with `both`; same practice
  items for original/edited. Returns accuracy, paired difference/SE, rescues/losses,
  length/cutoff summaries and raw answers.
- Teacher forcing: up to1024 texts,32768 tokens/text,524288 tokens total. Returns
  token log probabilities, hidden states, MLP activations and bounded attention.
  This supports dense matched-continuation comparisons.
- p50c used69 requests,50 successful passes,80 Python runs,158 turns. Reported wall
  time20.3hours included a4.7hour service interruption.

The10/25/50-pass rows are separate runs with different final edits:30-matrix broad
directional changes; one attention-head ablation; sparse two-MLP repair. They are
not successive evaluations of one unchanged candidate or a universal monotonic law.

## Diagnosis and intervention

Initial practice:96 generations,11 correct,85 cut at32768 tokens. Repeated tails
suggested an obstacle to finishing. Correct-versus-error activation differences
were exploratory; four paired productive/repetitive continuations of the same
problems then isolated the proposed failure more closely. An initial chat template
inserted an empty thinking block; raw-text recapture corrected the mismatch.

The editor analyzed residual contrasts and MLP writes, tried one writer, a layer19
direction, then layers15+19. Early32-item gains were treated as weak. Controls
included several strengths, one-layer isolation, attention alternatives, orthogonal
norm-matched random changes, new matched continuations, productive/numeric branches,
and premature-closure checks. Several alternatives were removed.

For each l in{15,19}, W0 is the original down-projection (1024x3584); u is a unit
repetition direction from the four matched pairs at residual outputs16/20; m masks
64 writers selected using original diagnostic activation-weight contributions:

    W' = W0 - 6 * outer(u, m * (u.T @ W0))

**Alpha6 does not merely delete a projection.** On selected columns,
u.T@W'=-5*(u.T@W0): it reverses/amplifies that component. Other columns stay fixed.
This strength, layer choice and sparsity are target-specific, not a detector recipe.

The edit touches131072 values; after bfloat16 rounding130188 actually differ.
Local Frobenius changes are about10.64% and8.11%. Small total-model sparsity does
not imply a tiny local perturbation. Reconstruction checks base/edited tensor
hashes; reconstruction alone is not benchmark regeneration.

Sources: [tool log](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl),
[report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md),
[frozen audit](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/frozen_edit_audit.json),
[edit manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json),
[reconstruction](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py).

## Confirmation and limits

The frozen edit received paired confirmation on1152 distinct held-out practice
problems:122->696 correct,579 rescues/5losses. Of the rescues576 had originally
hit the output limit; aggregate accuracy on originally finished answers was
unchanged. Correct edited answers still used long reasoning. Wrong answers,
missing boxes and whitespace degeneration remained. These are practice results,
separate from final unseen benchmarks.

Evaluation fixed thinking/template/decoding:temperature.6,top-p.95,top-k20,
presence penalty1.5,max32768, last boxed answer grading. Sampling repetitions per
problem varied by benchmark. The log warns that batching can change RNG: pair
models within requests, not nominally equal seeds across separate calls.

Released baseline/50-pass records: MATH-5007.9%->48.1%,GSM8K13.8%->67.1%,
IFEval21.0%->62.3%; MATH cutoff93.0%->33.1%. The completed baseline GPQA result
differs from the post's interim figure: preserve versions and denominators.
Gemma270M lacked the same dominant looping failure and had no established gain
(MATH6.8%->7.0% at50passes). WSP did not generally add missing reasoning ability.

Sources: [final answer](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/final_answer.md),
[baseline benchmarks](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/unedited/merged.json),
[edited benchmarks](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/p50c/merged.json),
[Gemma](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/README.md).

## Transfer boundary and attribution

Transfer diagnosis, targeted intervention, controls and independent confirmation.
Frozen monitored LLM, linear probe, low-FPR objective, grouped splits/inference and
abstention are our adaptation. They are not rules attributed to every WSP experiment.
Original logs/data/networks:CC BY4.0; authors' scripts:MIT; benchmark terms separate.
This document paraphrases and credits those sources.
