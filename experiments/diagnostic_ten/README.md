# Ten diagnosis-guided attempts to improve the detector

This continuation keeps Qwen3.5-2B frozen and edits its separate FIBS-labelled activation detector. It captures a fresh, pinned sample because the earlier runtime artifacts were unavailable. [Protocol](PROTOCOL.md) was committed before capture. The ten attempts have not been claimed complete by this README; execution status belongs in the run manifest and final report.

The study distinguishes three questions: can we explain a repeatable error, can we improve the detector, and does direct reasoning-guided editing outperform ordinary fitting? A negative result is retained.

## Running the study

Requirements in the verified Colab environment: torch 2.14.1+cu130, transformers 5.22.1, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, pyarrow 23.0.1, huggingface-hub 1.33.0. Use the pinned model/data revisions in `capture.py`. GPU capture is the expensive stage; subsequent detector fitting is CPU work.

```bash
python capture.py --out /content/fibs_diagnostic_ten/capture --n-per-family 80 --max-tokens 1536
python study.py prepare --capture /content/fibs_diagnostic_ten/capture --run /content/fibs_diagnostic_ten/run
python study.py attempt --capture /content/fibs_diagnostic_ten/capture --run /content/fibs_diagnostic_ten/run --spec attempt_01.json
# Repeat with the next reviewed, development-grounded specification, through attempt 10.
python study.py freeze --capture /content/fibs_diagnostic_ten/capture --run /content/fibs_diagnostic_ten/run
python study.py final --capture /content/fibs_diagnostic_ten/capture --run /content/fibs_diagnostic_ten/run
python study.py replay --capture /content/fibs_diagnostic_ten/capture --run /content/fibs_diagnostic_ten/run
```

Write each hypothesis and falsifying prediction before its selection evaluation. `attempt` saves those first, then constructs the candidate and matching ordinary/random controls. Control feedback is saved and withheld from the agent's printed feedback until candidate selection is frozen. Candidate weights are explicit arrays; the edit implementation uses no gradient optimizer. The ordinary arm uses logistic-regression fitting. Data-assisted edits are labelled separately.

Selection thresholds are tuning statistics. Final thresholds come from separate honest target-domain groups. Test results are opened only after ten attempts and immutable winner hashes. Prior target-family findings were public, and exact old row IDs are unavailable: historical independence is not established.

## Artifacts and interpretation

The run saves checkpoints, specifications, development and selection predictions, six full-context development cases, error tables, raw-weight diagnostics, prediction-call accounting, ordinary-fit timing, thresholds, final predictions, joint calibration/test group-bootstrap intervals, and a metrics replay result. Counts of nested node computations are not added to parent score-call counts. Internal sklearn solver evaluations are not enumerated.

Readout-view changes, normalization changes, direct coefficient edits and added readout capacity remain distinct. A readout transplant gain alone does not prove that reasoning over weights adds value. Report actual false-alarm rate alongside recall and preserve all failed candidates.

Software checks:

```bash
python -m unittest discover -s experiments/diagnostic_ten -p 'test_*.py' -v
```

These tests use generated fixtures to exercise score replay, finite-sample thresholds, ten-attempt limits and the final-evaluation lock. Fixture numbers are not detector performance results.

Sources: [FIBS paper](https://arxiv.org/pdf/2610.12445v4), [FIBS dataset](https://huggingface.co/datasets/AlignmentResearch/fibs-v1), [official probe implementation](https://github.com/AlignmentResearch/fibs-probes), [original WSP report](https://rikiyat.github.io/blog/weight-space-programming/).
