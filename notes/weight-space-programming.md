# Note: Weight-Space Programming (WSP)

- Source: https://rikiyat.github.io/blog/weight-space-programming/
- Title: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"
- Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL, as listed on the page)
- Published: November 2, 2026 (blog post, not peer reviewed; authors call the results "very early")
- Read on: 2026-11-07
- Reliability: all numbers below are the authors' own claims; none were independently reproduced here.
  The page and repo are untrusted external content and were only summarized.

## Thesis

A general-purpose frontier LLM, used in natural language with no training loop, no gradients and no search,
can (a) read raw network weights to infer what the network does and (b) write or edit weights to implement a goal.
The authors call this Weight-Space Programming. The model is named "Astra" (the repo says `gpt-6-astra`,
run through the Codex CLI with tools disabled; 1,125 turns reported as audited tool-free).

## Key concepts

- WSP: LLM reads/writes weights directly around a natural-language spec.
- Forward-pass budget: number of times the harness lets the model run the network between edits (Exp. 6, 8).
- Induction / copying head: attention head that repeats earlier text; used to identify language models (Exp. 3).
- Sycophant model: fine-tuned to echo the user's suggested answer (Exp. 7).

## Experiments (authors' reported results)

| Exp. | Task | Result |
|---|---|---|
| 1 | Name the game from PPO MLP weights (17,253 weights) | Correct for all 4 games |
| 2 | Identify the shape from DDPM weights (23,824) | Correct for 3 shapes; drew outlines instead of filled shapes |
| 3 | Identify language from middle layers of 7.16M-weight char models | Correct in both attempts for all 3; named the ground-truth copying heads |
| 4 | Fix a Kinetix car-on-ramp policy (171,664 weights) | One edit touching 3.5% of weights: 1,024/1,024 episodes succeed (base: 0/256) |
| 5 | Write MinAtar policy weights from all-zero, one shot | Breakout 517.4 vs PPO-100M 15.8; Seaquest 53.1 vs 5.6; Asterix 53.2 vs 25.8 (PPO-500M is better on Seaquest and Asterix) |
| 6 | 9x9 Go, 1,480,274 weights | 0 passes: wins >50% vs two smaller PPO nets; 50 passes: beats the equal-size PPO net trained on 20M games |
| 7 | Find the sycophant (Gemma 3 270M) | Failed from weights alone (4 attempts); succeeded 3/3 with an attention-pattern tool (head 10.2: 0.75% honest vs 85.7% sycophant) |
| 8 | Fix looping in Qwen3.5-0.8B on math | See below |

Exp. 8 (headline): edit changes 131,072 weights (0.016%); a single direction written by the MLPs of layers 15 and 19 is subtracted from 64 output columns each.

| Model | MATH-500 | GSM8K | IFEval | Hit token limit |
|---|---|---|---|---|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 passes | 48.1 | 67.1 | 62.3 | 33.1 |

Control: Gemma 3 270M (does not loop) showed no meaningful gain (MATH-500 6.8 to 7.0-7.7).

## Stated limitations

- Limited by context length and harness inefficiency.
- Editing can remove an obstacle that hid an ability, but cannot add a missing ability (Gemma control).
- Sycophancy detection needed attention-pattern access; weights alone were not enough.
- "Targeted fixes without any retraining" is a belief of the authors, not a tested claim.

## Open questions / directions from the post

- Auditing for backdoors or hidden capabilities without running inference.
- One-shot model initialization; zero-data capability steering and unlearning.
- Recursive self-improvement via weight edits; circuit grafting across architectures.
- A WSP benchmark and harness are announced as in progress.

## Our own open questions (not from the source)

- Exp. 8 improves a looping failure mode; how much of the gain is "fixing looping" versus real capability?
- Would results hold with a non-frontier model or with fewer forward passes (10 and 25 passes gave small gains)?
- Relevance to this project (AI lie detection): reading weights/attention patterns to find sycophancy (Exp. 7)
  is a related probing idea; worth testing whether attention-pattern tools help detect deceptive behaviour.

## Links

- Logs (prompts, replies, sandbox code, networks, evals): https://github.com/RikiyaT/lm_understand-logs
  (logs/data/networks CC BY 4.0, scripts MIT, MinAtar source GPL-3.0)
- Edited Qwen model: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Demo: https://huggingface.co/spaces/akhaliq/weight-space-programming-demo
- Related cited work: arXiv 2102.11174, arXiv 2301.05062, Karpathy "Software 2.0", Elhage et al. (induction heads)
