# 🕵️ Can We Catch a Model Lying?
### One afternoon · one Colab T4 · 27 kinds of lies
*Claude Impact Lab 2026 · team `llm-lying-detection` · 2026-11-07*

> 5-minute closing talk. Animated version: [`afternoon-wrap-up.html`](afternoon-wrap-up.html)
> (open in a browser, `→` / `space` to advance, `N` for notes, `F` for fullscreen).
> Timings in *italics* add up to ~5:00.

---

## 1 · The question *(0:00 – 0:40, incl. title)*

> **Can we tell when a language model says something it *believes* is false,
> and find *where* in the network that lie lives?**

A *knowing falsehood* (FIBS definition): the model contradicts what it demonstrably believes,
e.g. a fact it read two turns earlier.

Three sub-questions we picked up during the afternoon:

1. **Detect** — does a probe on the activations spot the lie?
2. **Generalize** — does it still work on *kinds* of lies it never trained on?
3. **Locate** — which layers / weights carry the signal?

---

## 2 · Two papers, two halves *(0:40 – 1:00)*

| | 🔬 **Telling FIBS** (FAR.AI) | 🧠 **Weight-Space Programming** (MIT) |
|---|---|---|
| Reads | activations at runtime | raw weights |
| Knows | **THAT** the model lies (98.7% AUROC) | **WHERE** something happens, can edit it |
| Gap | black box, no explanation | could *not* find a sycophant from weights alone; 3/3 once given an activation signal |

**Our bet:** a lie probe *is* the activation signal the weight reader was missing. 🤝

---

## 3 · Five approaches on the table *(1:00 – 1:30)*

| # | Approach | Who / where | Status |
|---|---|---|---|
| A | **Probe → weight auditor**: Claude gets two identical small models (one fine-tuned to lie) + probe scores; picks the liar. Weights only vs. probe only vs. both | `docs/research-question.md` | 📝 proposed |
| B | **Ladder of combinations**: detect → trace probe direction into MLPs/heads → edit it out → audit without evals | `notes/…probes-x-weight-space.md` | 📝 proposed |
| C | **Claude research loop**: Claude rewrites the probe *code*, fixed harness scores it on unseen lie families, keep-or-revert, repeat | `experiments/research_loop/` | ✅ **ran** |
| D | **Brute-force control**: random hyperparameter search, same budget, same keep rule | `research_loop/search.py` | 🔧 built, not run |
| E | **WSP on the probe itself**: LLM edits the trained probe *weights* directly (θ → θ + Δθ), vs. gradient fine-tuning | `experiments/weight_space_edit/` (Fred) | ❌ **ran: negative** |

We ran **C** first because it fit in an afternoon on one GPU and gives a hard number; **E** followed after 15:40.

---

## 4 · The afternoon, as a lab log *(1:30 – 1:50)*

| Time | What happened |
|---|---|
| 13:00 | Team forms on Discord, sources collected |
| 13:30 | Repo, literature notes, research question |
| 14:00 | Laptop has 8 GB RAM → 🫠 → Colab T4 |
| 14:20 | End-to-end test: 179 conversations in 0.6 min ✅ |
| 14:25 | 🐛 families missing (tool-result endings), row selection hung on **100k-token** transcripts → length prefilter |
| 14:35 | 🐢 4,900 conversations = 54 min ETA → cut to 2,940 (12.8 min) |
| 14:55 | Baseline: val 0.849 · test 0.806 |
| 14:56 | 🔁 Loop: 15 iterations in 22 min, 0 crashes |
| 15:10 | 🎉 Test 0.806 → **0.845** |
| 15:54 | 🧠 Approach E: Claude edits the probe *weights* directly |
| 16:06 | ❌ E on sealed test: no gain — an honest negative |

---

## 5 · The loop *(1:50 – 2:10)*

```
probe.py (best) → Claude Opus 5.5 → new probe.py + hypothesis
      ↑                                     ↓
   keep / revert  ←  val > best + 0.002?  ←  eval.py (fixed): train 14 families, score 7 val
                                            └→ test score: logged, NEVER shown to Claude
```

**Honesty guards:** test hidden · `predict()` must score each example alone (re-checked) ·
no labels or family names at predict time · every candidate saved.

Setup: Qwen3.5-2B · FIBS-v1 · 2,940 conversations · split **by family** (14 train / 7 val / 6 test).

---

## 6 · The headline number *(2:10 – 2:35)*

| | Baseline (FIBS LinearEnsemble) | Best (iter 9) | Δ |
|---|---|---|---|
| val macro-AUROC *(Claude optimized this)* | 0.849 | 0.887 | +0.037 |
| **test macro-AUROC *(hidden)*** | **0.806** | **0.845** | **+0.039** |
| train | 0.9995 | 0.9999 | |

Every unseen family improved or held. Biggest wins on the hardest:
**among +0.098** · sandbagging +0.046 · medqa +0.044 · werewolf +0.026.

---

## 7 · What Claude tried (15 experiments) *(2:35 – 3:00)*

| Idea family | Iters | Verdict |
|---|---|---|
| **More views** of activations (3 poolings × 6 layers = 18 probes) | 4 | ✅ +0.032 — biggest step |
| **Trust views that transfer** (grouped-CV softmax weights) | 9 | ✅ +0.005 — the winner |
| Cleverer combining (LDA, concat, stacker) | 8, 10, 11 | 〰️ neutral |
| Remove "family identity" | 1, 5, 12 | ❌ |
| Simpler / regularized directions (mass-mean, C, PCA) | 2, 3, 6 | ❌ |
| OOD robustness tricks (L2, clip, V-REx, bagging) | 7, 13, 14, 15 | 〰️ neutral |

**Lesson:** the problem is less *nuisance offsets*, more *which layer you read*.

---

## 8 · Where the lie lives *(3:00 – 3:15)*

Weight the winning probe puts on each (pool × layer), Qwen3.5-2B, 24 layers:

| pool \ layer | 7 | 10 | **13** | 16 | 19 | 22 |
|---|---|---|---|---|---|---|
| last | 0% | 5% | **32%** | 7% | 1% | 1% |
| mean_answer | 0% | 16% | **26%** | 2% | 0% | 0% |
| mean_turn | 0% | 2% | **7%** | 0% | 0% | 0% |

**Layer 13 ≈ 65 %, layers 10–16 ≈ 97 %.** The transferable lie signal sits mid-network,
in line with "truth peaks at intermediate depth".

---

## 9 · Approach E: Claude edits the weights directly *(3:15 – 3:50)*

θ = θ₀ + Δθ_Claude on the saved LinearEnsemble probe. Five family-disjoint partitions
(dev / selection / **separate calibration** / test), test unsealed **once**, 20 evaluations per arm.

| arm | selection TPR @ 5% FPR | **test TPR @ calibrated 5%** | test macro-AUROC |
|---|---|---|---|
| original θ₀ | 0.317 | 0.503 | 0.791 |
| **Claude's weight edits** | **0.378** 📈 | **0.500** 💨 | 0.790 |
| random edits, same size | 0.367 | 0.486 | 0.787 |
| gradient fine-tune | 0.394 | 0.519 | 0.809 |
| plain refit on dev | 0.406 | **0.533** | 0.806 |

- **Negative result:** the selection gain did not transfer. Claude − original = −0.003 [−0.039, +0.025];
  Claude − refit = −0.034 [−0.069, −0.003] (the only interval that excludes 0).
- Claude's edits ≈ random edits of the same size. Claude itself called its gain "probably within noise".
- Bright spot (not significant): best AUROC on `among`, the hardest family (0.677 vs. 0.635).
- 🕳️ Calibration leaks: a 5% FPR threshold flags **7–10%** of honest answers on new families.
- **Caveat:** fixed 8-op edit vocabulary + a data-derived edit, no free Python kernel → this tests a
  constrained search over a convex probe, **not** weight-space programming as the MIT post does it.

---

## 10 · Reviewer #2 reviews our approaches *(3:50 – 4:30)*

| Approach | 👍 Strength | 👎 Concern | Score |
|---|---|---|---|
| **C · Research loop** | Real number on *unseen* lie types; hidden test; fully reproducible for a few $ | 1 seed, +0.039 ≈ 2 SE; winner's curse on 7 val families; winning idea is classic ML (ensembling + CV selection); TPR@FPR threshold set on val the loop optimized | ⭐⭐⭐⭐ promising, not proven |
| **D · Search control** | Answers "does the LLM beat brute force?" | Not run yet → we **cannot** claim the LLM added value | ⭐⭐ (blocks C's claim) |
| **E · WSP on probe weights** | Best-designed test of the day: separate calibration set, test unsealed once, 5 arms incl. random-edit control; ≈ $2, 8 min | Negative (0.500 vs 0.503, below plain refit 0.533); fixed edit vocabulary ≠ real WSP; LinearEnsemble, not the published EFC probe; test families already seen by humans in C | ⭐⭐⭐ rigorous, negative |
| **A · Liar auditor** | Clean ground truth (we build the liar); directly tests the WSP gap | Need to fine-tune liar models; FIBS probes on 270M models untested; 2–3 pairs = anecdote | ⭐⭐⭐ best science, most work |
| **B · Ladder** | Good roadmap; flags Goodhart (edit blinds probe ≠ fixes lie) | Steps 3–4 are dual-use; step 4 is a research program, not a workshop | ⭐⭐ vision |

Small bugs Reviewer #2 found in our own docs:
- `PROGRESS.md` lists `longcontext` as a test family, but it dropped out → **6** test families, not 7.
- Iter 11 had the best test (0.853) but was correctly discarded on val → ranking of near-ties is noisy.
- E showed calibration doesn't transfer: 5% FPR on calibration families → 7–10% on test families.
- Absolute numbers aren't comparable to the paper (own baseline, HF hooks instead of vLLM, 2B vs 397B).

---

## 11 · Next experiments *(4:30 – 4:50)*

1. 🎲 3–5 seeds + bootstrap CIs over **families** (cheap, features cached)
2. 🤖 vs 🎰 run `search.py` — does Claude beat random search?
3. 🧪 calibrate per domain: a 5 % threshold became 7–10 % on new families
4. 🔍 load published FIBS probe weights, check they reproduce
5. 🧠 E v2, real WSP: a free Python kernel instead of 8 fixed edit ops

---

## 12 · Thank you *(4:50 – 5:00)*

> **Questions?**
> We promise to answer truthfully.
> *(And if not: our probe scores AUROC 0.845 on lies it has never seen.)*

`github.com/msverma101/LLM_lie_detector` · results: `experiments/research_loop/RESULTS.md` · `experiments/weight_space_edit/runs_wsp/run1/RESULTS.md`
