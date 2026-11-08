# Adaptive weight-space edits to a frozen-LLM detector

This follow-up holds last-token activations, six layers, scaling and ensemble
aggregation fixed. The agent inspects real development failures and authors ten
sequential coefficient-edit modules, using each outcome to decide the next edit.
The monitored Qwen model is never edited. This is a FIBS-labelled pilot using our
own linear ensemble; it is not the published EFC checkpoint.

The [prospective protocol](PROTOCOL.md) was committed before capture. Read the
[numerical correction](SELECTION_PRECISION_NOTE.md) and the
[supplementary attribution controls](SUPPLEMENTARY_ABLATION_NOTE.md) when evaluating
the final evidence. Both notes predate final-test opening. Original feedback is
retained alongside the correction record.

## Reproduce

Python packages used: numpy, pandas, scipy, scikit-learn, pyarrow, torch,
transformers and huggingface-hub. Capture needs a compatible GPU; editing, fitting,
scoring and replay run on CPU. Model/data revisions and input eligibility are pinned
in the protocol and capture source. `capture_base.py` preserves the exact renderer.

```bash
python experiments/adaptive_wsp/capture.py --out /tmp/fibs_adaptive/capture \
  --previous experiments/diagnostic_ten/results/2026-11-07/meta.csv
python experiments/adaptive_wsp/study.py prepare \
  --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run
python experiments/adaptive_wsp/study.py controls \
  --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run
for n in $(seq -w 1 10); do
  python experiments/adaptive_wsp/study.py attempt \
    --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run \
    --spec experiments/adaptive_wsp/edits/attempt_${n}.json \
    --edit experiments/adaptive_wsp/edits/attempt_${n}.py
done
python experiments/adaptive_wsp/study.py freeze \
  --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run
python experiments/adaptive_wsp/study.py final \
  --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run
python experiments/adaptive_wsp/study.py replay \
  --capture /tmp/fibs_adaptive/capture --run /tmp/fibs_adaptive/run
```

Those commands replay the already-authored sequence, not a new adaptive-agent
replication. A new agent study requires new development diagnoses, recorded edits
and a newly sealed test. The corrected harness implements the inclusive tie band
from the outset; do not apply `fix_selection_precision.py` to such a new run. That
script exists to audit the original execution's one pre-test implementation correction.
Fresh capture refuses to overwrite an already-completed capture. The final lock
refuses a second opening, while replay recomputes previously reported outcomes.

```bash
python -m unittest discover -s experiments/adaptive_wsp -p 'test_*.py' -v
```

Generated fixtures are software checks only. Raw activations, complete cases,
weights, edits, diagnostics, outcomes and source checksums belong in the complete
replay archive; score-only reproduction and independent GPU regeneration are
reported separately. See the protocol for FIBS, WSP and procedural-skill references.
