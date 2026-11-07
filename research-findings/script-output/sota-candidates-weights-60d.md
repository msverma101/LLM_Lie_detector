# SOTA candidates (generated)

Generated 2026-11-07 by `research-findings/tools/find_sota.py --days 60 --aspect weights`.
Sources: arXiv API, OpenAlex citations of the seed papers, OpenReview. No API keys.

Yardsticks: **FIBS** probes 0.987 AUROC (SHADE-Arena) / 0.980 (BashArena); **Weight-Space Programming** reads and edits raw weights (0.016% edit, MATH-500 7.9% -> 48.1%).
A ⚑ flag means the abstract reports a score of 0.95 or higher, or claims to beat prior work. Benchmarks differ between papers, so read the paper before calling it better than FIBS.

**2 new in-field papers** (not yet in the literature files), out of 8 in-field, 15 total.

| Rank | Paper | Date | Scores in abstract | Flag | Found via |
|---|---|---|---|---|---|
| 1 | [A General Model Diffing Framework for Understanding Post-Training: Attributing Behavioral Differences to Parameter Updates](https://openreview.net/forum?id=8ikN42M4Sy) | 2026-09-15 | - |  | OpenReview (ICLR.cc/2027/Conference) |
| 2 | [Synchronization Reveals Market Structure That Classical Randomness Tests Cannot Detect](https://arxiv.org/abs/2610.11028) | 2026-10-08 | - | ⚑ | arXiv search |

## Abstracts of the top 25

### A General Model Diffing Framework for Understanding Post-Training: Attributing Behavioral Differences to Parameter Updates
OpenReview 8ikN42M4Sy · 2026-09-15 · OpenReview (ICLR.cc/2027/Conference)

Post-training dramatically changes the behavior of large language models, yet the mechanism behind these changes is hard to interpret because it originates from parameter updates of enormous scale and dimensionality. Prior model diffing works in the representation space and cannot trace behavioral differences back to the parameter updates that cause them. We propose a circuit-level model diffing framework that attributes the difference between two models within a single attribution graph. Running the base model and the post-trained model on the same input, we isolate the direct linear effect of the parameter difference at each block and convert it into an activation difference. We model these activation differences as nodes in an attribution graph built with a cross-layer transcoder, so that every path runs causally from a parameter-induced activation difference through interpretable features to the logits on which the two models disagree. Clustering these circuits across inputs yields a shared circuit that characterizes the behavioral difference. Applying the framework to a base model and the reasoning model obtained by reinforcement learning, we find that shallow-layer parameter 

### Synchronization Reveals Market Structure That Classical Randomness Tests Cannot Detect
arXiv 2610.11028 · 2026-10-08 · arXiv search

Synchronization among coupled oscillators -- from neurons to power grids -- is a universal signature of collective behavior, classically measured by the Kuramoto order parameter. Applied identically across five asset classes and 61 financial instruments, a Kuramoto-style cross-timeframe coherence measure detects structure that a calibrated, heteroskedasticity-robust variance-ratio test of the random-walk hypothesis does not: the two diagnostics are statistically independent even on the same price windows. Here we explain why, and extract the physics behind it. The four-state regime alphabet underlying the measure is a Z4 clock model whose exact factorization into two Ising sectors is precisely the order parameter's decomposition into a direction channel and a volatility channel, and the sign-flip symmetry of returns acts on the regime circle as a reflection, yielding a parity selection rule: the direction channel is provably independent of the variance ratio and of every statistic built from squared returns -- confirmed by controlled simulation on processes with and without genuine persistence. Boltzmann inversion of the regime statistics then yields effective couplings between adj

