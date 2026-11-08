# Attribution controls fixed before final opening

Added after adaptive candidate construction and before frozen-checkpoint hashing or
any final score access, 8 November 2026. These are supplementary descriptive arms;
they cannot select the winner or alter the primary hypotheses.

The observed edits combine scalar probe-confidence changes with sparse coordinate
changes. To avoid attributing a confidence effect to semantic coordinate repair,
freeze two ablations: (1) original plus only the selected winner's row-wise median
coefficient scaling and corresponding bias scaling; (2) selected winner with those
scalings undone. The exact factors, checkpoints and hashes are saved. All receive
independent thresholds from the same honest calibration set. Compare their final
counts descriptively. No extra validation selection opportunity is used. These are
not independent replications and no new causal-circuit claim follows from them.
