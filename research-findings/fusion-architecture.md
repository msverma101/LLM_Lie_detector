# Proposal: one lie monitor that fuses FIBS probes, weight reading and honesty adapters

Status: design proposal with a theoretical argument and a time-boxed test plan. Nothing measured yet.
Author: Sneha (with Claude). Date: 2026-11-07.

## 1. Can we replicate all three in one hour?

No. Honest time costs on one H100:

| Method | What a real replication needs | Realistic time |
|---|---|---|
| FIBS probe (2610.12445) | Released probes exist only for the paper's models; the headline numbers are on 122B–397B models (multi-GPU). Retraining on 163k transcripts needs activations for every row. | Days |
| Honesty adapter (ICLR 2027 submission) | No code yet. Generate ~60k transcripts from 533 Qwen3-14B organisms, train a rank-16 LoRA. | 2–4 days |
| Weight-space programming (MIT blog) | Closed model ("GPT-6 Astra"); public-model replications exist on GitHub. | Hours per task |

What one hour *can* do is a smoke test of the fusion idea (section 5), not a replication.

## 2. Architecture: three channels, one decision

```
                      ┌──────────────────────────── per model (once) ───────────────────────────┐
  model weights  ───► │ W: weight channel                                                        │
  (+ base model)      │    spectral stats + top-SVD of the fine-tune diff ΔW (2602.15195, model  │──► prior odds
                      │    diffing), optionally Claude WITH a compute tool (2611.07847)          │    that the model
                      └──────────────────────────────────────────────────────────────────────────┘    hides something
                      ┌──────────────────────────── per transcript ─────────────────────────────┐
  transcript  ──────► │ A: activation channel  FIBS-style probe on residual stream at a          │──► score s_A
  + follow-up         │    prefilled denial of "Did you exhibit hidden behavior or deception?"   │
  question            │ S: self-report channel honesty adapter (rank-16 LoRA), score =           │──► score s_S
                      │    mean log P(admit) − mean log P(deny)                                  │
                      └──────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                      fusion: logistic stacking on calibrated scores
                      logit P(lie) = b + w_A·s_A + w_S·s_S + w_W·s_W       (s_W = model-level prior)
                      threshold set for 1% FPR on held-out negatives
```

Why these three: each fails where another is strong.

| Channel | Strong on | Fails on (evidence) |
|---|---|---|
| A: FIBS-style probe | Ordinary and natural lies (Liars' Bench 0.88; censored Chinese models); cannot be talked out of its verdict by the model | Unseen trained liars: 0.66 AUROC on AuditBench (honesty-adapter paper); tiny models |
| S: Honesty adapter | Unseen trained liars: 0.90 AUROC, 47% caught at 1% FPR on AuditBench | Only at par on ordinary lies (0.87); the monitored model judges itself, so a model trained to evade could fool it; never tested on lies from normal models |
| W: weight channel | Dormant backdoors before any lie appears in a transcript; any training to evade A or S itself leaves a weight-diff signature | Needs the base model for the diff; weak on full fine-tunes at scale; per-model, not per-transcript (Weight Oracles 0.81 composite on toys) |

The weight channel acts as a **prior**, not a per-transcript score: posterior odds = prior odds(W) × likelihood
ratio(A, S | transcript). This is how a "this model looks tampered with" signal should change how suspicious we are of
each of its answers.

## 3. Theoretical argument: why the fusion cannot be worse, and when it is better

**Claim 1 (no loss, exact).** Let x = (a, s, w) be everything the three channels observe and Y ∈ {lie, honest}.
Each single detector's score is a function of x. By the Neyman–Pearson lemma, the likelihood-ratio test on x has the
highest true-positive rate among *all* tests using x at a given false-positive rate α. Any test built from one
channel's score is such a test. So for every α:

  TPR_fused(α) ≥ max(TPR_A(α), TPR_S(α), TPR_W(α))

The fused ROC curve lies on or above every single ROC curve, so its AUROC is at least the best single AUROC.

**Claim 2 (strict gain, exact condition).** Equality holds at α only if one channel's score is already a sufficient
statistic for Y there, i.e. the other channels add no information once that score is known. Strict gain needs the
channels to fail on *different* cases: conditional on the label, their errors must not be perfectly redundant. This is
measurable: compute the correlation of the channels' scores within each class.

**Claim 3 (finite data).** We never know the true likelihood ratio, so we fit a stacker. A logistic combiner with
3 weights contains every single channel as a special case (e.g. w = (1, 0, 0)), so its population loss is at most the
best single channel's; the extra error from fitting is of order √(3/n) for n calibration examples. The binding
constraint is low-FPR certification: claiming ≤1% FPR with 95% confidence needs about 300 independent honest negatives
with zero false alarms (rule of three), more if any occur (cf. 2610.04575).

**How big is the gain?** Equal-variance Gaussian model, using published AUROCs; ρ = correlation between channels'
scores within a class (unknown, must be measured):

| Setting | Best single | A + S, ρ = 0 | A + S, ρ = 0.3 | A + S + W, ρ = 0 | A + S + W, ρ = 0.3 |
|---|---|---|---|---|---|
| AuditBench, unseen trained liars (A 0.66, S 0.90, W 0.81) | 0.900 AUROC, 30% at 1% FPR | 0.911, 34% | 0.900, 30% | **0.946, 48%** | 0.917, 36% |
| Liars' Bench, ordinary lies (A 0.88, S 0.87) | 0.880, 25% | **0.948, 49%** | 0.923, 38% | n/a | n/a |

Reading: on ordinary lies, two equally good but independent channels nearly double the catch rate at 1% FPR. On
trained liars, a weak probe adds little to a strong adapter; the weight channel is what adds most. Gains shrink as ρ
grows. These are model-based illustrations, not results: W's 0.81 comes from toy networks and would likely be lower at
14B, and the Gaussian model ignores heavy tails at low FPR.

**Robustness argument (not a proof).** An adversary must now evade three channels at once. Evading the probe is
learnable (Neural Chameleons 2512.11949), but that evasion is a fine-tune and changes the weights, which channel W
reads; suppressing self-report likewise needs training. Fusion raises the attack's cost; it does not make evasion
impossible.

## 4. What would show the fusion is better (pre-registration sketch)

- Data: Qwen3-14B organisms from huggingface.co/introspection-auditing (train) and huggingface.co/auditing-agents
  Qwen 14B models (test, unseen recipe), plus an off-policy set (Liars' Bench) and one natural liar (censored model).
- Metrics: AUROC and TPR at 1% FPR, threshold frozen on a held-out calibration split; bootstrap CIs over organisms.
- Hypotheses: H1 fused TPR@1% > best single on AuditBench; H2 same on Liars' Bench; H3 within-class score
  correlation ρ(A, S) < 0.5. Failure of H3 predicts failure of H1/H2 (section 3).
- Baselines every channel must beat first: perplexity differencing (leaky organisms, COLM 2026) and SVD of ΔW.

## 5. One-hour smoke test (needs the H100)

Goal: estimate ρ, the number that decides whether fusion is worth building. Not a replication.

1. (10 min) Load Qwen3-14B and 4 AuditBench Qwen 14B organisms + 4 introspection-auditing organisms (LoRAs).
2. (15 min) Weight channel on CPU while the GPU works: spectral statistics of each LoRA ΔW (top singular value
   share, effective rank, per projection), giving a per-organism score.
3. (20 min) For ~50 prompts per organism, generate one response, append the follow-up question, and record
   (a) a linear probe score (mean-difference direction from 50 honest vs 50 instructed-lie examples on the base model)
   and (b) the bare-model admit−deny logit score (the honesty-adapter readout without training the adapter).
4. (15 min) Judge which responses showed the hidden behavior (LLM judge), then report per-channel AUROC, the
   within-class correlation ρ(A, S), and a 3-weight logistic stack with leave-one-organism-out validation.

Expected outcome: rough per-channel AUROCs and ρ. If ρ is low, fusion is worth the multi-day build; if high, it isn't.

## Sources
FIBS 2610.12445; honesty adapters (OpenReview gpt1K3Ia4D); Introspection Adapters 2604.16812; AuditBench 2602.22755;
Weight Oracles 2610.07334; Computation not weight access 2611.07847; LoRA backdoor spectra 2602.15195; model diffing
(ICML 2026 workshop); leaky organisms (COLM 2026); Neural Chameleons 2512.11949; operational validity 2610.04575.
See `research-findings/findings-2026-11-07.md`.
