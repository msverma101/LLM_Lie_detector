# SOTA candidates (generated)

Generated 2026-11-07 by `literature/tools/find_sota.py --days 365 --aspect weights`.
Sources: arXiv API, OpenAlex citations of the seed papers, OpenReview. No API keys.

Yardsticks: **FIBS** probes 0.987 AUROC (SHADE-Arena) / 0.980 (BashArena); **Weight-Space Programming** reads and edits raw weights (0.016% edit, MATH-500 7.9% -> 48.1%).
A ⚑ flag means the abstract reports a score of 0.95 or higher, or claims to beat prior work. Benchmarks differ between papers, so read the paper before calling it better than FIBS.

**7 new in-field papers** (not yet in the literature files), out of 15 in-field, 44 total.

| Rank | Paper | Date | Scores in abstract | Flag | Found via |
|---|---|---|---|---|---|
| 1 | [Data Attribution of Emergent Misalignment with Persona Features](https://arxiv.org/abs/2608.11025) | 2026-08-11 | - | ⚑ | arXiv search |
| 2 | [A General Model Diffing Framework for Understanding Post-Training: Attributing Behavioral Differences to Parameter Updates](https://openreview.net/forum?id=8ikN42M4Sy) | 2026-09-15 | - |  | OpenReview (ICLR.cc/2027/Conference) |
| 3 | [Targeted Recovery of Weight-Space Mechanisms From Neural Networks](https://arxiv.org/abs/2607.13047) | 2026-06-19 | - |  | arXiv search |
| 4 | [When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability](https://arxiv.org/abs/2605.15183) | 2026-05-14 | - |  | arXiv search |
| 5 | [Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes](https://arxiv.org/abs/2603.04426) | 2026-02-16 | - |  | arXiv search; OpenReview (aclweb.org/ACL/ARR/2026/May) |
| 6 | [Decomposition of Small Transformer Models](https://arxiv.org/abs/2511.08854) | 2025-11-12 | - |  | arXiv search |
| 7 | [Synchronization Reveals Market Structure That Classical Randomness Tests Cannot Detect](https://arxiv.org/abs/2610.11028) | 2026-10-08 | - | ⚑ | arXiv search |

## Abstracts of the top 25

### Data Attribution of Emergent Misalignment with Persona Features
arXiv 2608.11025 · 2026-08-11 · arXiv search

Emergent misalignment (EM) is the phenomenon where fine-tuning a language model on a narrow task leads to harmful behavior in unrelated domains. A leading mechanistic account attributes EM to persona features: latent directions acquired during pre-training that misaligned fine-tuning amplifies. We ask where these features come from: which pre-training documents activate them, and whether naturally occurring human-written text suffices to induce EM. Using Sparse Autoencoder (SAE) based model diffing across four open-weight models, we find that features related to jailbreak personas, sarcasm, deception, and manipulation are amplified by misalignment fine-tuning, while safety-relevant and assistant-identity features are suppressed. Steering individual features controls EM in both directions: it induces misalignment rates of up to 62% in aligned models -- exceeding the 35% reached by misalignment fine-tuning itself -- and re-aligns misaligned models to near-baseline misalignment rates. Attributing the causal features to a corpus of one million pre-training web documents retrieves semantically relevant narratives about villainous characters, domination, and harmful agency. However, fine

### A General Model Diffing Framework for Understanding Post-Training: Attributing Behavioral Differences to Parameter Updates
OpenReview 8ikN42M4Sy · 2026-09-15 · OpenReview (ICLR.cc/2027/Conference)

Post-training dramatically changes the behavior of large language models, yet the mechanism behind these changes is hard to interpret because it originates from parameter updates of enormous scale and dimensionality. Prior model diffing works in the representation space and cannot trace behavioral differences back to the parameter updates that cause them. We propose a circuit-level model diffing framework that attributes the difference between two models within a single attribution graph. Running the base model and the post-trained model on the same input, we isolate the direct linear effect of the parameter difference at each block and convert it into an activation difference. We model these activation differences as nodes in an attribution graph built with a cross-layer transcoder, so that every path runs causally from a parameter-induced activation difference through interpretable features to the logits on which the two models disagree. Clustering these circuits across inputs yields a shared circuit that characterizes the behavioral difference. Applying the framework to a base model and the reasoning model obtained by reinforcement learning, we find that shallow-layer parameter 

### Targeted Recovery of Weight-Space Mechanisms From Neural Networks
arXiv 2607.13047 · 2026-06-19 · arXiv search

Parameter decomposition (PD) decomposes neural networks into interpretable computational components that faithfully reflect the original network's operations. However, scaling PD to large models requires vast compute, making it a costly and risky endeavor. Here we propose targeted PD (tPD), which identifies only the components that process specific inputs of interest -- from isolated prompts to large subtasks -- by introducing a high-rank catch-all component that handles all non-target data. We validate tPD on toy models and on transformer language models trained on The Pile, where it recovers reproducible, mechanistically faithful circuits. We extract a CSS-only submodel of a 4-block transformer using 7% of the FLOPs of its published decomposition, and in a 12-block transformer we surgically ablate and rewire memorized sequences, with negligible side effects on other inputs.

### When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability
arXiv 2605.15183 · 2026-05-14 · arXiv search

Mechanistic interpretability aims to break models into meaningful parts; verifying that two such parts implement the same computation is a prerequisite. Existing similarity measures evaluate either empirical behaviour, leaving them blind to out-of-distribution mechanisms, or basis-dependent parameters, meaning they disregard weight-space symmetries. To address these issues for the class of tensor-based models, we introduce a weight-based metric, tensor similarity, that is invariant to such symmetries. This metric captures global functional equivalence and accounts for cross-layer mechanisms using an efficient recursive algorithm. Empirically, tensor similarity tracks functional training dynamics, such as grokking and backdoor insertion, with higher fidelity than existing metrics. This reduces measuring similarity and verifying faithfulness into a solved algebraic problem rather than one of empirical approximation.

### Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes
arXiv 2603.04426 · 2026-02-16 · arXiv search; OpenReview (aclweb.org/ACL/ARR/2026/May)

Model diffing aims to identify how fine-tuning changes a model's internal representations. Crosscoders are attractive for this task because they learn a sparse, indexable dictionary of latent directions, but existing formulations fail under narrow fine-tuning, where behaviorally important changes are often sparse, low-magnitude, and dominated by shared structure. We introduce $\textbf{Delta-Crosscoder}$, which combines Dual-K BatchTopK sparsity, a $\textit{delta-based}$ loss that reconstructs activation differences through non-shared latents, and contrastive signals from paired activations. Across 10 model organisms spanning synthetic false facts, emergent misalignment, subliminal learning, and taboo word guessing, Delta-Crosscoder recovers causally validated latents for all organisms, compared to 4--6 of 10 for prior crosscoder variants. Importantly, the recovered dictionary is structured: in emergent misalignment, it separates a misalignment-inducing latent from a refusal-gating latent, two causal directions that pooled activation-difference methods cannot disentangle by construction. Delta-Crosscoder also preserves reconstruction quality and improves dictionary utilization in a 

### Decomposition of Small Transformer Models
arXiv 2511.08854 · 2025-11-12 · arXiv search

Recent work in mechanistic interpretability has shown that decomposing models in parameter space may yield clean handles for analysis and intervention. Previous methods have demonstrated successful applications on a wide range of toy models, but the gap to "real models" has not yet been bridged. In this work, we extend Stochastic Parameter Decomposition (SPD) to Transformer models, proposing an updated causal importance function suited for sequential data and a new loss function. We demonstrate that SPD can successfully decompose a toy induction-head model and recover the expected 2-step circuit. We also show that applying SPD to GPT-2-small can successfully locate subcomponents corresponding to interpretable concepts like "golf" and "basketball". These results take the first step in the direction of extending SPD to modern models, and show that we can use the method to surface interpretable parameter-space mechanisms.

### Synchronization Reveals Market Structure That Classical Randomness Tests Cannot Detect
arXiv 2610.11028 · 2026-10-08 · arXiv search

Synchronization among coupled oscillators -- from neurons to power grids -- is a universal signature of collective behavior, classically measured by the Kuramoto order parameter. Applied identically across five asset classes and 61 financial instruments, a Kuramoto-style cross-timeframe coherence measure detects structure that a calibrated, heteroskedasticity-robust variance-ratio test of the random-walk hypothesis does not: the two diagnostics are statistically independent even on the same price windows. Here we explain why, and extract the physics behind it. The four-state regime alphabet underlying the measure is a Z4 clock model whose exact factorization into two Ising sectors is precisely the order parameter's decomposition into a direction channel and a volatility channel, and the sign-flip symmetry of returns acts on the regime circle as a reflection, yielding a parity selection rule: the direction channel is provably independent of the variance ratio and of every statistic built from squared returns -- confirmed by controlled simulation on processes with and without genuine persistence. Boltzmann inversion of the regime statistics then yields effective couplings between adj

