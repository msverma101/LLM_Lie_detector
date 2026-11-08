# Detector Weight-Space Debugging skill and tests

Start with the [completed report](REPORT.md) and
[installable skill](../../.agents/skills/detector-weight-space-debugging/SKILL.md).
The skill is a detector-only adaptation of the original WSP procedure. The monitored
LLM is frozen. An engineered fault was successfully repaired; improvement over the
uncorrupted natural detector was not confirmed.

The source audit includes the original prompts, 69 forward requests, 80 Python
blocks, exact masked directional formula and failure controls at a pinned revision.
See [the detailed method reference](../../.agents/skills/detector-weight-space-debugging/references/original-method.md),
[prospective protocol](PROTOCOL.md),[natural hypothesis](NATURAL_HYPOTHESIS.json),
and [source-record audit](results/source_audit/SOURCE_AUDIT.json).

## Use the skill

The directory `.agents/skills/detector-weight-space-debugging` is discoverable by
compatible agents. For a user-level Codex installation, copy that entire folder
to `~/.codex/skills/detector-weight-space-debugging`. Invoke:

```text
Use $detector-weight-space-debugging to investigate a justified repair of our
activation detector while keeping the monitored LLM frozen. Preserve the baseline
if diagnostic intervention and independent confirmation do not support a patch.
```

Claude or another agent can read the same SKILL.md and its linked resources.
No message was sent to another external service or collaborator as part of installation.

```bash
python experiments/wsp_skill_validation/test_skill.py -v
python .agents/skills/detector-weight-space-debugging/scripts/audit_edit.py parent.npz candidate.npz
```

Requirements: Python, NumPy, pandas, SciPy, scikit-learn. The original skill-creator
validator additionally needs PyYAML. No GPU is needed for saved-activation replay.

## Replay the completed study

Download/extract the complete evidence archive into an empty directory and run:

```bash
python experiments/wsp_skill_validation/replay_and_pack.py --root /absolute/path/to/extracted-study
```

This checks all capture hashes and recomputes all 16 frozen detectors from raw
activations. It does not repeat the agent's discovery or regenerate Qwen activations.
Run this command from the current repository: the archived original script passed
in Colab, while the current script explicitly handles machine-precision differences
at a calibration tie and still requires exact final test decisions.
For a fresh study, follow PROTOCOL.md, new grouped capture and new confirmation
data; never reuse these opened final scores for another confirmatory claim.

`capture_fresh.py` shares grouping/renderer helpers with the preceding adaptive
experiment. It accepts both earlier metadata files through `--previous`. Source
capture and hypothesis commits precede their respective measurements. Engineered
control construction and independent-agent outputs are documented separately.
Some skill v1 forward-test limitations were found and corrected; those records
are retained rather than described as a fully clean independent evaluation.
