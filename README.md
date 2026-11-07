# LLM_Lie_detector

Harness to test whether lies of a frozen LLM (Qwen3.5-2B) can be detected from **activations** (published FIBS probes)
and whether **weight-space edits of the probe** proposed by an LLM (Weight-Space Programming, WSP) add anything over
ordinary optimisation. Every claim is measured with a preregistered plan; negative results are written up like positive ones.

Status (2026-11-07, one machine, CPU): the pipeline runs end to end and is tested (47 tests). First result, preregistered
endpoint: **WSP not supported** (see `docs/notes/NOTE-0003-*.md`). Read the caveats there before citing numbers.

## Architecture

```
liedet build-data   facts (4 families) x 2x2 {system asks truth|lie} x {assistant states true|false}   -> rows.jsonl, split.json (by FACT)
liedet capture      Qwen3.5-2B (HF, bf16, thinking off) -> block outputs at layers 7,10,13,16,19,22; window = final user turn..end-of-turn
                    + belief margins (keep facts the model prefers to answer correctly) + neutral-prompt logprob   -> features.npz
liedet fibs-check   gate G0: published probes on FIBS rows through OUR capture (contract parity)
liedet fibs-eval    DEV ONLY: published probes + baselines, error analysis (what is missed / wrongly flagged)
liedet wsp-run      DEV ONLY: arms A0 published | A1 ordinary fine-tune | A2 random edits | A3 LLM-proposed edits (claude -p)
liedet final-test   the single look at the test facts; writes FINAL.lock and refuses a second run
liedet weights      weight-space view: where do the model's matrices write onto the probe direction (planted-edit controls)
liedet evaluate     activation-space probe vs baselines, cross-family splits (own probes, not the published ones)
```

| module | role |
|---|---|
| `data.py` | fact bank, 2x2 design, fact-level split |
| `capture.py` | exact token windows, hidden-state capture, belief margin, per-token windows |
| `evalkit.py` | certified low-FPR thresholds (order statistic), recall@FPR, Clopper-Pearson, cluster bootstrap |
| `fibs.py` | adapter to `probe_inference`, FIBS row formatting, parity check |
| `wsp.py` | edit grammar (closed, validated), acceptance rule, proposers (random / scripted / Claude), ordinary fine-tune |
| `pipeline.py` | stage commands fibs-eval, wsp-run, final-test |
| `weightspace.py`, `weights_report.py` | residual-writer matrices, alignment z-scores vs random directions, ablate/add hooks, planted-edit controls |

Guardrails: the probe score is never a training or selection signal for the monitored model; edits touch the probe only.

## Run it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e . -e ../fibs-probes pandas pyarrow pytest
.venv/bin/python -m pytest tests -q                       # 47 tests, ~35 s, no network needed except the optional real-probe round trip
# real run (downloads Qwen/Qwen3.5-2B ~4.5 GB and the qwen3.5-2b probes ~ MBs; ~10 min capture on 16 CPU cores)
bash runs/20261107-main/run_capture.sh                    # copies of the scripts: docs/data/experiments/fibs-wsp-2026-11-07/
bash runs/20261107-main/run_arms.sh                       # A3 needs the `claude` CLI and model access
.venv/bin/python -m liedet final-test --data runs/20261107-main/data --features runs/20261107-main/features \
    --arms runs/20261107-main/arms_followup --out runs/20261107-main/final_followup
```

Parity gate: `liedet fibs-check --model <qwen dir> --parquet-dir <FIBSv1 parquet dir> --out <dir>` (FIBSv1 from Hugging Face `AlignmentResearch/fibs-v1`).

## What is verified and what is not

Verified here: unit and CLI end-to-end tests on a tiny random model with synthetic probes; real run on Qwen3.5-2B (CPU, bf16); parity gate G0;
the published linear probe's weights reproduce `probe.score` to 1e-5.
Not verified: Windows/macOS, GPU, equality with the authors' vLLM activations, mlp/efc/axial editing, repeated seeds, any human-checked labels.
