# Which work is at par with or beats FIBS and Weight-Space Programming, and what we could do instead

Written 2026-11-07. Numbers come from each paper's arXiv abstract unless marked "(FIBS text)", which means
read in the FIBS paper itself. Benchmarks differ between papers, so "better" is only claimed where two methods
were measured on the same data, or where a paper shows FIBS's own weak spot fixed.

## 0. The two yardsticks

| | What it achieves | Where it is weak |
|---|---|---|
| **FIBS** (2610.12445) | 163k-transcript falsehood dataset + axial/EFC probes. 98.7% AUROC on SHADE-Arena, 98.0% on BashArena, beating action-only text monitors (97.7/97.6). Hidden-goal denials up to 99.7%. More data keeps helping up to 153k rows. | Needs very large models: "every probe scores far lower on the smallest model of each family" (FIBS text). Off-policy, artificial data. Weak on math-sandbagger (0.63). False positives in long episodes. |
| **Weight-Space Programming** (MIT blog, Nov 2026) | Frontier model reads raw weights (names the game, the language) and edits them (0.016% of Qwen3.5-0.8B: MATH-500 7.9% → 48.1%). | Failed to find a sycophant from weights alone (needed an attention tool). Blog post, no peer review, single runs, closed model ("Astra"). |

## 1. At par with or better than FIBS

| Paper | Head-to-head result | Verdict |
|---|---|---|
| **Activation Oracles** (Karvonen et al. 2512.15674; Transluce scaling, Aug 2026) | FIBS ran an oracle on its own 8 hidden-goal organisms (FIBS text, Table 20): oracle 0.80–1.00 AUROC; significantly **above** the FIBS axial probe on 3 organisms (math-sandbagger +0.17, two gender organisms +0.15/+0.19), below on 1, level on 4. | **Better on the hardest organisms**, at much higher cost. FIBS's own weakest case (math-sandbagger) is the oracle's biggest win. |
| **Why Static Probes Fail** (Huang et al. 2610.27577, 20 Oct 2026) | Multi-direction deception subspace + few-shot domain gating: average AUROC 0.97 and 94% recall at 5% FPR across all datasets, **with only 10 examples** per new domain; "substantially outperforming existing methods". | **At par on accuracy, far better on data cost** (10 examples vs 153k rows). Not tested on SHADE-Arena, so not a direct replacement. |
| **Rift: A Conflict Signature for Deception** (2606.17229) | Label-free "residual rank" signal: picks the lie between two responses with 100% accuracy; AUC 1.0 on Phi-3; zero-shot transfer to other model families (mean AUC 0.933) and across 5 languages. Works on **GPT-2 small/medium and Qwen2.5-1.5B/7B**. | **Better where FIBS is weakest**: tiny models, no training labels. Single-author, small models, simpler settings than agentic sabotage: treat as promising, needs replication. |
| **Monitoring Reward Hacking with Internal Representations** (Bergen et al. / Goodfire 2609.19101) | Difference-of-means vectors on Kimi K3 / GLM 5.2 match an LLM monitor at equal FPR (+3.1% hacks caught on Kimi K3, −7.9% on GLM 5.2), "virtually free", and fire on the CoT *before* the hack happens. | **At par** with LLM monitors, like FIBS, using a far simpler probe; adds prediction ahead of the action. |
| **Before Agent Tells the Lie** (2610.06576) | Deception predicted from hidden states several model calls before it appears; steering along the direction reduces later deception. | **Goes beyond FIBS** (FIBS detects after the fact); no AUROC on a shared benchmark. |
| **MCAIR** (2610.19679) | Multi-step representation probe beats LLM-as-a-judge on agent "false completion" reports, and adds a correction loop. | **At par in spirit** on a different benchmark (ACUD). |
| **Probe of Internal Recognition / Lie Detector Test** (2609.21996) | Concealed-information test inside the model: 0.85–0.93 recognition under prompted deception, trained sandbagging, password-locked and circuit-broken models, across 8 models in 5 families; no labelled truth corpus needed. | **Better than FIBS on sandbagging-style hiding**, which is FIBS's weakest organism. Only for multiple-choice-like questions. |
| **Linear Probe Accuracy Scales with Model Size + Multi-Layer Ensembling** (2604.13386) | Multi-layer ensembles: +29% AUROC on Insider Trading, +78% on Harm-Pressure; 12 models 0.5B–176B, ~5% AUROC per 10x parameters. | **At par in method** (FIBS's LinearEnsemble is the same idea); quantifies the small-model penalty. |
| **Pressure-Testing Deception Probes** (2605.27958) | Style-augmented probes recover 0.979–0.983 AUROC on unseen styles (Gemma 3 1B–27B). | **Fixes a robustness gap** FIBS does not test (style shift). |
| **Building Production-Ready Probes for Gemini** (DeepMind 2601.11516) | Long-context probe architectures, deployed in Gemini; probe + prompted classifier is the best accuracy for the cost. | **Ahead of FIBS on deployment** (actually in production), but for cyber misuse, not lying. |
| **Stress-Testing LLM Lie Detectors** (2609.39807) | Eight existing probes fail under anti-factual personas; their new simple linear probe is best on both stress tests. | **Better on robustness** to role-play and spurious cues; not run on agentic sabotage. |
| **Did you lie?** (Cooney et al. 2606.12618) | On *trained* organisms every activation and logprob detector drops sharply; a CoT judge stays at 0.82 balanced accuracy. | **Not better, but a stronger test** than FIBS's: its organisms are what FIBS borrowed. Read as a warning. |
| **Fine-Tuned Lie Detectors Failed to Generalize** (Anthropic, Aug 2026) | Fine-tuned LLM detectors: 0.95 on seen lie types, ~0.70–0.75 on held-out types. | **Worse**: supports FIBS's bet on probes + broad data. |

**From OpenReview (ICLR 2027 submissions and 2026 workshops, not on arXiv):**

| Paper | Result | Verdict |
|---|---|---|
| ★ *Training Honesty Adapters To Detect Deception In LLMs* (ICLR 2027 submission, OpenReview gpt1K3Ia4D) | A LoRA "honesty adapter" trained on on-policy transcripts from 533 model organisms reports whether the model lied. Near-perfect AUROC on held-out organisms; **0.90 AUROC on AuditBench vs 0.74 for a probe** trained on the same data (Qwen3-14B). | **Better than probes on trained organisms**, the case where FIBS-style probes and "Did you lie?" detectors drop. Under review; read in full 7 Nov: the baseline 0.66 / 0.74 probe is an **axial (FIBS-architecture) probe on 155k off-policy rows**, so this is the closest thing to a direct FIBS comparison. At par on off-policy lies (Liars' Bench 0.87 vs 0.88). Organisms from Shenoy et al., *Introspection Adapters* (ICML 2026, Anthropic). |
| ★ Wang, Marks, Treutlein, Hubinger, Roger. *Evaluating Honesty and Lie Detection Techniques on a Diverse Suite of Dishonest Models* (Anthropic Alignment blog / LessWrong, 25 Nov 2025) | Five lying testbeds (MASK, secret side constraint, hidden-goal organism, harm pressure, password-locked). Asking the model to classify its own statements: 0.82 AUROC; **an honesty-fine-tuned model as the classifier: 0.88**. Truth probes, honesty steering, fuzzing and persona fine-tuning did **worse than simple prompting**. | **Independent, published support** for the honesty-adapter direction: fine-tuned self-classifiers beat probes on trained liars. Stylized settings, limited tuning. |
| *What Do Linear Deception Probes Actually Detect?* (ICLR 2027 submission, zprNXrgKhb) | The common contrast-pair probe mainly detects an instruction-induced *persona* state; benign role-play explains much of the score. | **Critique**, matching 2609.39807. Applies to simple probes more than to FIBS's data-scaled ones. |
| *Linear Probes for Strategic Deception Generalize Across Three LLM Families in Negotiation* (AAMAS 2026 workshop) | AUC > 0.93 on 11 of 12 model × scenario cells, Gemma-7B / Llama-3.1-8B / Mistral-7B, including deception by omission. | **At par** at 7–8B on multi-turn deception. |
| *Activation Oracles Detect Prompt-Induced but Not Fine-Tuned Sandbagging* (CAISc 2026) | Oracle P(sandbagging) 0.77 on prompted sandbagging (Llama-3.3-70B) but far weaker on the trained Cedar organism. | **Limit** on the oracle result above: trained concealment is still hard. |
| *Eliciting Sandbagged Capability via Attention-Head Functional Networks* (ICLR 2027 submission) | Graphs of attention-head activity differ between honest and sandbagging runs of the same weights. | **Parallel** white-box method for sandbagging. |
| *Dissociating Evaluation Awareness from Sandbagging, Qwen2.5 0.5B–32B* (NeurIPS 2026 workshop) | "Evaluation awareness" direction is perfectly decodable but mostly captures formatting and does not cause sandbagging. | **Caution** for probe interpretation in small models. |

Also checked and not above FIBS: Goodfire *Shaping and Monitoring Models Through Their Internal Representations* (5 Nov 2026: sycophancy probe 0.98 → 0.93 AUROC after training, 1.00 when retrained; no deception benchmark), Anthropic *Training Alignment Auditors via RL* (Oct 2026: RL-trained Haiku 4.5 auditor scores 72.7 vs Opus 4.6's 68.0 on audit quality; black-box auditing, no lie-detection numbers), and DeepMind production probes (cyber misuse, not lying).

**Bottom line (detection):** nobody beats FIBS on its own agentic benchmarks yet. Methods that beat it on its
weak points are activation oracles (hard organisms), Rift (tiny models, no labels), PIR (sandbagging),
and few-shot subspace gating (data cost).

## 2. At par with or better than Weight-Space Programming

| Paper | Result | Verdict |
|---|---|---|
| **Weight Oracles** (Kabra, Venhoff, Schroeder de Witt 2610.07334) | Fine-tuned LLM simulates small transformers' forward pass from weights (99% holdout accuracy), then finds **unseen backdoors from weights alone**: AUROC 0.93 on attention-routed backdoors, 0.81 on a broad threat set; hand-made detectors collapse when the threat changes. | **Better on the auditing question** that WSP failed (finding a hidden behavior from weights alone), with reproducible numbers. Only small models so far. |
| **WSP for Auditing Clinical Decision Networks** (2611.05341, 3 Nov) | Claude Opus 5.5, blind and without tools, named the planted flaw in 16 of 18 networks and fixed each by changing a median of **3 parameters**; fine-tuning also fixed them but changed nearly every weight. But it found only 1 of 3 hidden triggers and cleared only 2 of 5 healthy networks; black-box tests found confounds faster. | **At par, independently replicated with a public model**, on tiny networks (433–4,801 weights). Shows the limit: triggers / backdoors are the hard part, the exact case we care about. |
| **WeightLoop** (2611.07319, 5 Nov) | Same "Astra" model writes reusable weight-editing programs updated by experimental feedback; better ranking utility in 7 of 8 settings. | **Extends WSP** (edit → measure → edit loop). Same closed model. |
| **Can an LLM Actually Reason about Weights...?** (2611.08070, 5 Nov) | Claude Opus 5.5 reading 125k-weight CNNs: accurate local arithmetic, but class judgments come from memorized dataset priors (permuted labels fool it). | **Counter-evidence**: zero-shot "reading" may be shallower than WSP claims. Comes with an audit protocol worth reusing. |
| **Noise Injection Reveals Hidden Capabilities of Sandbagging LMs** (NeurIPS 2025) | Adding noise to the weights makes sandbagging models *improve*; honest models only get worse. | **Simpler and proven** weight-space detector for one lie type (sandbagging). Good baseline. |
| **Interpreting LM Parameters / Stochastic Parameter Decomposition** (Goodfire, Bushnaq et al. 2025–26) | Decomposes weight matrices into interpretable components, without an LLM reader. | **More principled** route to "what does this weight do"; heavier to run. |
| **Persona Cartography** (2607.07916) | Personality traits as additive LoRA directions in weight space across 4B–32B models; agreeableness axis moves sycophancy. | **Parallel**: shows sycophancy (a WSP Exp. 7 target) has a weight-space direction you can write. |
| **Shared Sycophancy-Lying Circuit** (2604.19117) | In 12 open models (small to frontier) the same few attention heads carry "this is wrong"; silencing them flips sycophancy without hurting accuracy; same head-to-head paths drive factual and instructed lying. | **Better targeted than WSP Exp. 7**: tells you *which heads* to read or edit, the exact place WSP needed an attention tool. |

**Weight-difference methods (found on OpenReview):**

| Paper | Result | Verdict |
|---|---|---|
| ★ Naphade, Ramanathan, Ryd, Shenoy. *Auditing LLMs for Hidden Behaviors using Model Diffing* (ICML 2026 workshop) | **SVD rank truncation**: project the fine-tuned-minus-base weight difference onto its top singular direction; this isolates the implanted behavior, which is "geometrically concentrated". Evaluated on AuditBench, plus a black-box adversarial-decoding variant. | **Parallel to WSP and much cheaper**: no LLM reader, just linear algebra on the weight diff. The natural baseline for any "Claude reads the weights" experiment. |
| ★ Abu Baker, Baroni, Wilhelm. *Most Current Model Organisms Are Leaky: Perplexity Differencing Often Reveals Finetuning Objectives* (COLM 2026) | Rank random-prefix completions by perplexity difference between fine-tuned and base model; this surfaces the fine-tuning objective for the **vast majority of 76 organisms (0.5B–70B)**, including backdoored ones. | **Warning for every idea here**: if this trivial method finds our liar, a fancier detector proves little. Run it first as a sanity baseline. |
| Shenoy, Lindsey, Marks et al. *Introspection Adapters* (ICML 2026; Anthropic) | A LoRA trained across many fine-tunes with implanted behaviors makes models verbalize their own implanted behavior, generalizing to differently-trained fine-tunes. | **Source** of the 533 organisms used by the honesty-adapter submission above. |

**Learned weight-diff readers and spectral detectors (found via Weight Oracles' references):**

| Paper | Result | Verdict |
|---|---|---|
| ★ Goel, Kim, Shavit, Wang. *Learning to Interpret Weight Differences in Language Models* (DIT, 2510.05092) | Trains an adapter that makes a fine-tuned model describe in words how its weights changed, including hidden behaviors. | **Closest real-LLM-scale peer to WSP's "reading" side**: works on actual LLM weight diffs, not toys. |
| Puertolas Merenciano et al. *Weight Space Detection of Backdoors in LoRA Adapters* (2602.15195) | 20 spectral statistics of the LoRA update ΔW + logistic regression detect poisoned adapters without running the model, on Llama-3.2-3B, Qwen2.5 and others. | **Simple, real-scale weight-space baseline.** |
| Pitzalis et al. *Z-PEFT: Zero-shot Backdoor Detection via Canonical Spectral Signatures* (2608.02271) | Spectral meta-classifier evaluated on unseen attacks and datasets. | **Shows the zero-shot gap** Weight Oracles also reports. |

**Found by `tools/find_sota.py` (run 7 Nov 2026, last 120 days):**

| Paper | Result | Verdict |
|---|---|---|
| ★ *Computation, not weight access: LLMs as auditors of clinical prediction networks* (2611.07847, 5 Nov) | Separates "weights in the prompt, no tools" from "weights plus a calculator tool" and "predictions only", using GPT-6 Astra itself. Without tools Astra recovered planted relationships in 353-weight networks but became unreliable at 3,457 weights, and declined 11 of 12 audit/repair tasks under its default instruction. | **The strongest critique of WSP so far**, using the same model: the wins come from *running computation*, not from reading weights. For our project, give Claude a calculator / forward-pass tool, not just raw numbers. |
| ★ *Weights Read and Write Features: Activation-Supported Parameter Decomposition* (2609.37731) | Decomposes weights of **Qwen-3-8B** into components grounded in the activation features they read and write; components are causally editable and compose into circuits. | **Better than WSP on scale and rigour** for "what does this weight do": works on an 8B model with causal checks, no LLM reader needed. |
| *Verbalizing Activation Differences with Natural Language Autoencoders* (2610.30934) | NLAs reused on fine-tunes without retraining can recover fine-tuning objectives in model diffing, across 4 models from 3 families. | **Parallel** to Goel et al.'s weight-diff reader, through activations. |
| *Many Biases, One Circuit* (2610.22937) | Sycophancy and three other cue biases share a few attention heads already present in pretrained models; swapping six layers' attention patterns moves the effect between checkpoints. | **Supports Idea A**: the "liar heads" are few, shared and present before alignment. |
| *Unlearning Deceptive Behaviors with Contrastive Forget Sets* (2609.38909) | Unlearns *when* a model deceives rather than what it knows. | **Alternative to weight editing** for step 4 of Idea A. |
| *Train the Model, Not the Reader* (2607.20379) | Activation verbalizers' reconstruction scores track gist, not specific claims (~2% of claims matter). | **Caution** for activation-oracle results. |
| *A False Average: Pooled CoT-Monitor Accuracy* (2608.00583) | Rewriting only the reasoning drops a CoT monitor's catch rate on hard hacks from ~95% to 4–11%. | **Supports FIBS's case** for white-box over CoT monitors. |

**Independent replications of the MIT post (GitHub, checked directly):**
- **MaceTenth/weight-space-programming-replication:** Breakout networks reproduce (Astra 506, Claude Fable 5.1 500, Claude Opus 5.5 55), but the author shows the high scores come from a hand-coded "never lose the ball" controller written as weights, so "RL fails at this game, not that the LLM out-learned it". On a fair agent-repair task (Qwen2.5-1.5B), **LoRA fine-tuning beat the Claude Opus weight edit: 48/48 vs 46/48 tasks, 8 min vs 110 min**; "Against a fair baseline, ordinary fine-tuning won." Opus named induction heads with precision 0.67–0.73 on SmolLM2-135M / Qwen2.5-0.5B, 0.20 on Qwen3-0.6B. Single runs.
- **Hammad-Anjum/weight-space-replication:** the Qwen3.5-0.8B math edit works, but turning thinking off scores 43% on MATH-500 with no edit. The edited score is not on the page, so the two can't be compared yet, but this is a cheap baseline the MIT post did not report.
- Also: HomoHabilis (8 GB GPU replication kit), ctrimm (exact arithmetic rules written as weights for $0.65), HarleyCooper (1 of 3 Breakout repairs helped).

**Bottom line (weights):** Weight Oracles is the strongest peer-style result for finding hidden behavior from
weights; the clinical replication shows Claude Opus 5.5 can do WSP-style edits too, but also that hidden triggers
are where it fails. The sycophancy-lying circuit paper narrows the search to a handful of heads.

## 2b. Numbers checked in the full text
- Weight Oracles (read in full): 0.93 ± 0.04 AUROC over five runs = **94% true positives at a 20% false-positive rate**. The targets are **toy 1-layer transformers of about 2K parameters** (d = 8), all triggers single-token and single-class; the reader is a fine-tuned Qwen2.5-3B. Hand-made baselines: total L2 norm 0.445, best V/O-pathway statistic 0.752; per attack type the oracle stays at 0.74–0.89 while the statistics collapse. The authors say scaling beyond a few million parameters is unverified and the oracle detects statistical correlates, not the mechanism. So it beats WSP on *rigour* (controlled, zero-shot, repeated runs), not on *scale* (WSP edited a 0.8B model).
- Honesty adapter (OpenReview gpt1K3Ia4D): OpenReview blocks automated PDF download (browser check), so only the abstract could be read. The 0.90 vs 0.74 AuditBench result is confirmed in the full text (PDF downloaded by hand); the direction (fine-tuned self-classifier > probe on trained liars) is independently supported by Anthropic's Nov 2025 study (0.88 vs probes below prompting).
- Rift: the paper's own limitations say that in unpaired comparisons on the smallest model, residual rank **can confuse deception with uncertainty**, and the signal can't be used to steer (0/8). Use it only with paired responses.
- Shared sycophancy-lying circuit: the 12 models include **Qwen2.5-1.5B and Gemma-2-2B**, so its heads exist at hackathon scale.
- Why Static Probes Fail: the full HTML has no machine-readable baseline table; "substantially outperforming existing methods" is the authors' claim, not checked.

## 3. Ideas better than, or parallel to, the team's FIBS × WSP plan

The team's plan: small model (Gemma 3 270M / Qwen3.5-0.8B), honest vs lying pair, FIBS-style probe as a tool
for a Claude weight-reader, then trace and edit. The literature says three things about that plan:

- FIBS-style probes are weak on the smallest models (FIBS text; 2604.13386: ~5% AUROC per 10x params).
- Simply fine-tuned liars are too easy (Model Organism Lottery 2607.01033; Did-you-lie 2606.12618).
- WSP finds silenced inputs and shifted thresholds, not triggers (2611.05341).

### Idea A (recommended): "Find the liar heads": circuit-guided weight audit
1. Use Qwen2.5-1.5B or Gemma-2-2B (both are among the 12 models in the sycophancy-circuit paper; Rift also ran Qwen2.5-1.5B).
2. Build the pair the Rift way: a **sleeper liar** (knows the truth, lies on a trigger) vs a **naive liar** (trained to give the same wrong answers with no honest training). Outputs are identical, so only the internals can tell them apart. This fixes the "too easy organism" problem.
3. Three detectors, same data: (a) multi-layer linear probe (2604.13386), (b) Rift residual-rank signal (label-free; paired responses only, see 2b), (c) Claude reading the **weight difference of the candidate heads** from 2604.19117 between the two models, as a WSP-style reader with a narrow search space, (d) optionally an honesty adapter (LoRA that reports lies; ICLR 2027 submission) as the strongest current organism detector.
4. Edit: silence or project out those heads; measure behavior with a probe-independent check (MASK-style questions) *and* the probe score, to catch the Goodhart case.

Baselines every detector must beat first: perplexity differencing (COLM 2026) and SVD of the weight difference (model-diffing paper). If they already find the liar, the organism is too easy.

Give the weight-reader a calculator / forward-pass tool: 2611.07847 shows raw weights alone stop working at a few thousand weights.

Baseline to beat for any edit: a small LoRA fine-tune, which won the only fair edit-vs-fine-tune comparison so far (MaceTenth replication).

Why it beats the original plan: it fits the model size, uses an organism that can't be caught by surface cues,
and gives Claude a small, concrete weight region instead of the whole network.

### Idea B: Weight-space sandbagging detector, cheap and quick
Noise injection (NeurIPS 2025) + PIR concealed-information test (2609.21996) + Claude reading the weight
diff. Sandbagging is FIBS's weakest case (0.63) and the one activation oracles win on. Smallest compute of
all ideas; a good fallback if training organisms takes too long.

### Idea C: Few-shot detector on a new domain
Reproduce the 10-example subspace-gating detector (2610.27577) and compare it with a FIBS-trained probe on a
domain FIBS never saw (e.g. agent false-completion reports from MCAIR's ACUD set, or Fredo's setting).
Directly answers "do we need 153k rows?". Pairs with 2610.39766 (probes work with 1–10 positives).

### Idea D: Weight Oracle vs zero-shot Claude, head to head
The literature has two conflicting claims: a *fine-tuned* reader finds backdoors (Weight Oracles, 0.93) and a
*zero-shot* frontier model mostly uses priors (2611.08070) and misses triggers (2611.05341). Run Claude Opus 5.5
zero-shot on Weight Oracles' public backdoor benchmark, using the audit protocol from 2611.08070. One clean
table settles a live question.

### Idea E: Predict the lie before it happens (agent setting)
Following 2610.06576 and Goodfire's CoT-time DoM vectors: probe an agent's hidden states before the final
report and steer away from the lie. Most "useful product" of the ideas, but needs a model big enough to run
agent tasks (≥7B).

### Not recommended for a one-day hackathon
- Training a FIBS-scale axial probe (data and model size too large).
- Building a new activation oracle (training cost); use a released one if needed.

## 4. Things every idea should include (from the evaluation papers)
- Report recall at a fixed low false-positive rate, not only AUROC (2610.04575, Fredo's README).
- Use a ground truth that does not depend on the probe (FIBS idea note, 2609.18080).
- Check for spurious cues: compliance, likelihood, persona (2609.39807).
- Preregister before the first data point (repo CLAUDE.md, CONCEPT-0001).

## 5. Where we searched
Rerun anytime without API keys: `python3 research-findings/tools/find_sota.py --days 60` writes `research-findings/script-output/sota-candidates.md`.

arXiv (API keyword, date-window and reference-list scans), OpenAlex (citations and keywords, covers journals and ACL/EMNLP), OpenReview (ICLR 2027 submissions and 2026 workshops), the research pages of Anthropic, Apollo, Goodfire and Transluce, GitHub replications of the MIT post, and web/LessWrong searches. Full list and method in `state-of-the-art.md`. Semantic Scholar refused requests without an API key.
