# Research loop: improving FIBS lie-detection probes

Claude runs an automated research loop: **propose a probe change → train → evaluate on lie
families it never saw → keep if better, revert if not → repeat.**

**Open in Colab:** [research_loop.ipynb](https://colab.research.google.com/github/msverma101/LLM_lie_detector/blob/people-looking-at-my-laptop-liked-it/experiments/research_loop/research_loop.ipynb)

```mermaid
flowchart LR
    A["cache_activations.py<br/>Qwen3.5-2B on FIBS<br/>(GPU, once)"] --> B["eval.py (fixed)<br/>train 14 / val 7 / test 7 families"]
    B --> C{"loop.py"}
    C -->|"history + val scores"| D["Claude Opus 5.5<br/>proposes new probe.py"]
    D --> E["train + eval"]
    E -->|"val better?"| F["keep"]
    E -->|"no"| G["revert"]
    F --> C
    G --> C
    C --> H["report.py<br/>progress.png + summary.md"]
```

| File | Role |
|---|---|
| `cache_activations.py` | Runs the model once over a family- and quadrant-balanced FIBS sample; saves activations from 6 layers (last token, answer mean, final-turn mean) |
| `eval.py` | **Fixed** harness. Splits by family; metric = macro-AUROC; checks scores are per-example |
| `probe.py` | The only file the loop edits. Starts as the FIBS paper's LinearEnsemble baseline |
| `program.md` | Instructions and rules for the researcher agent |
| `loop.py` | The loop. Claude only sees **val** scores; **test** families are reserved for the final result |
| `errors.py` | Error analysis at a fixed threshold (5% false alarms on honest val answers): missed lies and false alarms by family, quadrant, source, with examples. VAL by default |
| `search.py` | Comparison without an LLM: random hyperparameter search, same budget and keep rule, results in `runs_search/` |
| `robustness.py` | Is the best probe's gain real? Paired bootstrap intervals on the test families + random family re-splits (all probes refit), optionally vs. the `search.py` winner. No GPU, no API. Output `runs/robustness.md` |
| `results/baseline_probe.py` | Frozen copy of the baseline, because `loop.py` overwrites `probe.py` with the best candidate |
| `report.py` | Plot + summary table, incl. TPR at fixed FPR and Claude loop vs. search |

Final criterion ("more lies caught without more false alarms"): besides macro-AUROC,
`eval.py` reports **TPR at a fixed FPR**. One threshold is set so that 1% / 5% of honest
*val* answers are flagged and applied unchanged to *test*. A better probe has higher test
TPR at a test FPR that does not rise. Negative or unclear results go into the summary too.

Honesty rules built in: test families are never shown to the agent, `predict` may not use
batch statistics, and every candidate probe is saved in `runs/`.
