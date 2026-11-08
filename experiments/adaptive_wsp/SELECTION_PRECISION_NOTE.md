# Numerical implementation correction before final opening

Detected after adaptive attempt 6, before any final prediction access, 8 November 2026.
The protocol specifies an inclusive 0.005 recall tie band followed by macro-AUROC.
Binary floating point represents 0.370 - 0.375 as -0.005000000000000004, so the initial
implementation incorrectly placed attempt 6 just outside that band. The code adds
an absolute 1e-12 comparison tolerance and a regression check. This does not change
the registered threshold or endpoints. No metrics, checkpoints or original feedback
are overwritten. Save an explicit correction record and replay keep/revert decisions
and ordinary winner selection from their saved validation summaries before continuing.
This is a prospective software correction, not a change motivated by final test scores.
