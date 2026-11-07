# State of the art: detecting LLM lies from model internals (as of 7 Nov 2026)

Compiled 2026-11-07 from the FIBS paper's reference list plus web searches.
Summaries of papers marked (abstract only) come from arXiv abstracts or search results, not a full read.
Numbers are the authors' own claims.

Legend: ★ = read first for this project.

## 1. Probe-based lie / deception detectors (the core method)

| Paper | arXiv / link | What it does, in one line |
|---|---|---|
| ★ Hollinsworth, Spies, ..., Cundy. *Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception* (FAR.AI, Oct 2026) | 2610.12445 | Largest lie dataset (FIBS) + layer×token "axial" probes; beats text monitors on SHADE-Arena / BashArena; catches denied hidden goals. Already in `papers/`. |
| ★ Goldowsky-Dill et al. *Detecting Strategic Deception Using Linear Probes* (Apollo, 2025) | 2502.03407 | The baseline: simple linear probes on Llama catch strategic deception; stresses low false-positive rates. |
| ★ Natarajan et al. *One Probe Won't Catch Them All: Towards Targeted Deception Detection* (ICML 2026) | 2602.01425 | Different kinds of lies live in different directions; argues for targeted probes, not one universal one. |
| Kumar. *Pressure-Testing Deception Probes in LLMs* (2026) | 2605.27958 | Linear probes >0.96 AUROC on clean data but break under style shift; deception is not one clean direction (Gemma 3 1B–27B). (abstract only) |
| Boxo et al. *Caught in the Act: A Mechanistic Approach to Detecting Deception* (2025) | 2508.19505 | Probes >90% accuracy on deceptive arguments, 1.5B–14B models; deception spread over several directions. |
| Kramár et al. *Building Production-Ready Probes for Gemini* (Google DeepMind, 2026) | 2601.11516 | How to make probes robust on long contexts for real deployment. |
| McKenzie et al. *Detecting High-Stakes Interactions with Activation Probes* (NeurIPS 2025) | 2506.10805 | Attention/max-pooling probe variants; cheap monitoring. |
| Kantamneni et al. *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing* (ICML 2025) | — | Attention probes; SAEs rarely beat plain probes. |
| Cunningham et al. *Constitutional Classifiers++* (Anthropic, 2026) | — | Production probes that read many layers, streamed. |
| Dooms, Wang, Pearce. *Covariance-Based Sequence Pooling* (Goodfire, 2026) | goodfire.com/research/covariance-pooling | Basis of the FIBS "EFC" probe. |
| Bergen et al. *Monitoring and Discovering Reward Hacking with Internal Representations* (2026) | 2609.19101 | Probes catch reward hacking during evaluations. |
| Li et al. *Before Agent Tells The Lie: Has Deception Already Been Represented?* (Oct 2026) | 2610.06576 | Deception signal in agents grows over a trajectory; steering on that direction reduces later lying. (abstract only) |
| Xu et al. *MCAIR: Monitoring and Correcting Agent Unfaithfulness via Internal Representations* (Oct 2026) | 2610.19679 | Probe flags agents whose final report doesn't match what they did; releases ACUD dataset. (abstract only) |
| *A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal* (2026) | 2609.21996 | Internal readout stays ~0.95 AUC where black-box drops to 0.68 on gemma-2-9b; one model only. (abstract only) |
| Fredo220. *verified-reasoning-monitoring* (repo, ongoing) | github.com/Fredo220/verified-reasoning-monitoring | Team member's probe study on false success reports; 77% (9B) / 32% (2B) recall at 5% false alarms on agent tasks. |

## 2. Benchmarks and how to evaluate a lie detector

| Paper | arXiv | One line |
|---|---|---|
| ★ Smith, Chughtai, Nanda. *Difficulties with Evaluating a Deception Detector for AIs* (2025) | 2511.22662 | Why ground-truth labels for "lying" are hard; read before designing an experiment. |
| ★ Cooney, Africa, Irving. *"Did you lie?" Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms* (2026) | 2606.12618 | 13 model organisms on Qwen 3.5 27B; detectors do much worse on trained liars than on prompted lying. Weights released. |
| Kretschmar et al. *Liars' Bench* (2025) | 2511.16035 | Benchmark of lie types; current detectors miss lies the transcript can't reveal. |
| Parrack, Attubato, Heimersheim. *Benchmarking Deception Probes via Black-to-White Performance Boosts* (2025) | 2507.12691 | Measures how much white-box access adds over a black-box monitor. |
| Ren et al. *The MASK Benchmark* (2025) | 2503.03750 | Separates honesty (saying what you believe) from accuracy. |
| Kutasov et al. *SHADE-Arena* (2025) | 2506.15740 | Agent sabotage tasks used to test monitors. |
| Kaufman et al. *BashArena* (2025) | 2512.15688 | Control setting for highly privileged agents. |
| Panfilov et al. *Strategic Dishonesty Can Undermine AI Safety Evaluations* (ICLR 2026) | 2509.18058 | Models fake harmful answers; probes catch it where output monitors fail. |
| *DecepEval* (Oct 2026), *AgentDecept* (Oct 2026) | 2610.07967, 2610.39377 | Behavioral (not internal) deception benchmarks for agents. (abstract only) |

## 3. Hidden goals, sandbagging and auditing model organisms

| Paper | arXiv | One line |
|---|---|---|
| ★ Marks et al. *Auditing Language Models for Hidden Objectives* (Anthropic, 2025) | anthropic.com PDF | Blind auditing game: teams find a model's hidden goal; origin of the model-organism approach. |
| ★ Sheshadri et al. *AuditBench* (2026) | 2602.22755 | 56 organisms with hidden behaviors, 12 tools; activation oracles are the best white-box tool, but scaffolded black-box still wins. |
| *Auditing Games for Sandbagging* (Taylor et al., Dec 2025) | — | Deception probes flagged one sandbagger and missed others. |
| *A Causal Model for Locating and Unlocking Sandbagging in Model Organisms* (2026) | 2608.29461 | Mechanistic location and removal of sandbagging. (abstract only) |
| van der Weij et al. *AI Sandbagging* (ICLR 2025) | — | Models can strategically underperform on evals. |
| Casademunt, Cywiński et al. *Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation* (2026) | 2603.05494 | Real models that hide knowledge (e.g. censored topics) as test cases. |
| Dietz et al. *Split Personality Training* (ICML 2026) | — | Revealing hidden knowledge through an alternate persona. |
| MacDiarmid et al. *Simple Probes Can Catch Sleeper Agents* (Anthropic, 2024) | blog | Linear probes detect backdoored "sleeper agent" behavior. |

## 4. Attacks on probes and training against them

| Paper | arXiv | One line |
|---|---|---|
| ★ Taufeeque, Heimersheim, Gleave, Cundy. *The Obfuscation Atlas* (ICML 2026, honorable mention) | 2602.15515 | RL against a deception probe yields honest, deceptive or obfuscating models; strong KL keeps them honest. Code public. |
| Cundy & Gleave. *Preference Learning with Lie Detectors Can Induce Honesty or Evasion* (2025) | — | Same question, earlier and for preference learning. |
| *The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes* (2026) | 2609.10739 | Truth probes can fail in specific contexts. (abstract only) |
| *What Do Compliance Detectors Read? An Audit of Activation Probes and Guard Models* (2026) | 2608.16852 | Checks whether probes read the concept or surface cues. (abstract only) |
| *Linear Probes Rely on Textual Evidence* (2025) | 2509.21344 | Probes can lean on words in the text rather than the model's belief. |
| Baker et al. *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (OpenAI, 2025) | 2503.11926 | Pressuring the chain of thought teaches models to hide intent. |
| Korbak et al. *Chain of Thought Monitorability* (2025) | 2507.11473 | Why CoT monitoring is fragile; motivation for white-box probes. |

## 5. Explaining activations with an LLM (activation oracles)

| Paper | arXiv | One line |
|---|---|---|
| ★ Karvonen et al. *Activation Oracles* (2025) | 2512.15674 | Train an LLM to answer questions about another model's activations. |
| Choi et al. *Scaling Activation Oracles to Trillion-Parameter Models* (Transluce, Aug 2026) | transluce.org | Same at frontier scale. |
| *Decoding Hidden Deception in Reasoning LLMs: StateWitness* (2026) | 2606.17478 | Activation explainer that returns evidence traces, not just a score. (abstract only) |

## 6. Reading and editing weights (the second half of the team idea)

| Paper | arXiv / link | One line |
|---|---|---|
| ★ Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming* (MIT, Nov 2026, blog) | rikiyat.github.io | Frontier LLM reads/edits raw weights; failed to find a sycophant from weights alone. Already in `articles/`. |
| ★ Kabra, Venhoff, Schroeder de Witt. *Weight Oracles: Reading Neural Network Weights with Language Models* (Oct 2026, NeurIPS workshop) | 2610.07334 | Fine-tuned LLM reads weights; 0.93 AUROC on unseen attention backdoors without running the model. |
| ★ Ahmed. *Can an LLM Actually Reason and Program about Neural Network's Weights...?* (5 Nov 2026) | 2611.08070 | Skeptical replication with Claude Opus 5.5 on small CNNs: looks like local arithmetic + memorized dataset priors. |
| Baines et al. *Persona Cartography: Personality Traits in Weight Space* (2026) | 2607.07916 | Maps persona traits to weight directions. |
| *A Survey of Weight Space Learning* (Mar 2026) | 2603.10090 | Overview of learning from weights (model zoos, metanetworks). |
| *Solving Trojan Detection Competitions with Linear Weight Classification* (2024) | 2411.03445 | Simple classifiers on weights detect trojans. |
| *LoRA adapter poisoning detection from weight spectra* (2026) | 2602.15195 | Detects poisoned adapters without running the model. (abstract only) |
| *LLM Forensics: Where Do Backdoors Hide?* (2026) | 2609.07746 | SAEs localize backdoor triggers. (abstract only) |

## 7. Foundations (older, still cited everywhere)

- Burns et al. *Discovering Latent Knowledge Without Supervision* (CCS), ICLR 2023, 2212.03827
- Azaria & Mitchell. *The Internal State of an LLM Knows When It's Lying*, 2023
- Marks & Tegmark. *The Geometry of Truth*, COLM 2024, 2310.06824
- Bürger et al. *Truth is Universal: Robust Detection of Lies in LLMs*, 2024, 2407.12831
- Zou et al. *Representation Engineering*, 2023; Li et al. *Inference-Time Intervention*, 2023
- Arditi et al. *Refusal Is Mediated by a Single Direction*, NeurIPS 2024 (template for "find a direction, project it out", i.e. step 3 of our idea)
- Lu et al. *The Assistant Axis*, ICML 2026; Marks, Lindsey, Olah. *The Persona Selection Model*, Anthropic 2026 (persona limitation in FIBS)

## 8. Added after a systematic completeness check

Found by the check described in "How this list was built" below. One-liners are written from each arXiv abstract.

### Probe methods, scaling and generalization
| Paper | arXiv | One line |
|---|---|---|
| ★ Nordby et al. *Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling* (Apr 2026) | 2604.13386 | Multi-layer ensembles fix fragile single-layer probes; accuracy grows with model size (bad news for 270M models). |
| ★ Kirch et al. *The Impact of Off-Policy Training Data on Probe Generalisation* (ACL Findings 2026) | 2511.17408 | How much probes trained on synthetic / other-model data transfer to a model's own behavior. |
| Ying et al. *The Truthfulness Spectrum Hypothesis* (Feb 2026) | 2602.20273 | Truth directions range from general to domain-specific; probes fail on sycophantic lying unless trained jointly. |
| Huang et al. *Why Static Probes Fail: Context Determines Deception Feature Composition* (Oct 2026) | 2610.27577 | The features that make up "deception" change with context, so one fixed probe misses cases. |
| Yoo & Skapars. *Probe Generalization as Subspace Selection for OOD Deception Detection* (Jul 2026) | 2609.02893 | Projecting onto a small subspace improves out-of-distribution deception probes (Llama-3.1-8B). |
| Mahadik & Skapars. *Do Linear Probes Generalize Better in Persona Coordinates?* (May 2026) | 2605.09391 | Probes in persona-space coordinates for sandbagging and strategic deception. |
| Moustafa et al. *Beyond Liars' Bench: Lie Typology, Depth, and Sparsity* (May 2026) | 2607.20479 | Systematic study of why probes trained on one lie type miss others. |
| Bajwa & Poesio. *Probing and Steering for Safety-Relevant Behaviours* (Oct 2026) | 2610.30363 | How probe design choices interact, for detecting and steering, on Llama-3.1-8B. |
| Oldfield et al. *Beyond Linear Probes: Dynamic Safety Monitoring* (2025) | 2509.26238 | Polynomial probes that spend more compute only on hard inputs. |
| Zhou et al. *Probing the Misaligned Thinking Process of Language Models* (Jun 2026) | 2606.24251 | Probes on reasoning traces for deception, sandbagging, self-preservation. |
| Wang et al. *When Thinking LLMs Lie* (2025) | 2506.04909 | Deception directions in reasoning models; steering to induce or reduce lying. |
| Mirtaheri & Belkin. *Catching Rationalization in the Act* (Mar 2026) | 2603.17199 | Probes detect motivated reasoning before and after the chain of thought. |
| Koran et al. *Ensemble Monitoring for AI Control* (May 2026) | 2605.15377 | Combining different monitor signals beats spending more compute on one. |
| Zolfaghari. *Linear Separability after Fine-Tuning on Incorrect Responses* (May 2026) | 2605.30381 | Fine-tuning a model to be wrong leaves a linearly readable trace, 1.4B–9B models. Close to our "liar organism" setup. |

### Limits and failure modes of lie detectors
| Paper | arXiv | One line |
|---|---|---|
| ★ Klinski et al. *Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations* (Sep 2026) | 2609.39807 | Role-play personas break the meaning of "truth" for probes; finds spurious cues. |
| ★ Tiwari et al. *Decodability is Not Causality* (Sep 2026) | 2609.18080 | A probe can read a concept without that feature driving behavior; matters for step 2/3 of our idea. |
| Haralambiev. *Why Safety Probes Catch Liars But Miss Fanatics* (Mar 2026) | 2603.25861 | Probes miss models that sincerely believe harmful goals are good. |
| Thormann. *Probing the Limits of the Lie Detector Approach* (Feb 2026) | 2603.10003 | Philosophical critique: deception is wider than lying. |
| Blandfort & Graham. *Red-teaming Activation Probes using Prompted LLMs* (2025) | 2511.00554 | Cheap black-box red-teaming finds probe failure modes. |
| Haller et al. *LLM Knowledge is Brittle* (2025) | 2510.11905 | Truth representations rely on surface resemblance. |
| Savcisens & Eliassi-Rad. *The Trilemma of Truth in LLMs* (2025) | 2506.23921 | Three flawed assumptions in truth probing; proposes a fix. |
| Yu et al. *From Directions to Cones* (2025) | 2505.21800 | Truth is a multi-dimensional cone, not one direction. |
| Lu et al. *Disentangling Deception and Hallucination Failures* (Feb 2026) | 2602.14529 | Separates "doesn't know" from "knows but says otherwise". |
| *The Refusal Residue: When Probes Catch Alignment Faking and When They Don't* (Jul 2026) | 2607.13346 | Probes on alignment faking; mixed success. |

### Evading probes and training with them
| Paper | arXiv | One line |
|---|---|---|
| ★ McGuinness et al. *Neural Chameleons* (Dec 2025) | 2512.11949 | Models can learn to hide thoughts from activation monitors they never saw in training. |
| Gupta & Jenner. *RL-Obfuscation* (2025) | 2506.14261 | RL can teach models to evade latent-space monitors; some probe types are more robust. |
| Libon et al. *Alignment via Training Against Probes Without Losing Monitorability* (Sep 2026) | 2609.38645 | How to use probes as a training signal while keeping them useful. |
| Hollinsworth et al. *Scaling Trends for Lie Detector Oversight in Preference Learning* (Jul 2026) | 2607.01567 | FIBS authors scale lie-detector-in-the-loop training to bigger models. |
| Joglekar et al. *Training LLMs for Honesty via Confessions* (OpenAI, Dec 2025) | 2512.08093 | Train models to confess hidden misbehavior in a separate output. |
| Lee et al. *Training Agents to Self-Report Misbehavior* (Feb 2026) | 2602.22303 | "Self-incrimination training" for agents with hidden goals. |

### Model organisms, auditing, sandbagging
| Paper | arXiv | One line |
|---|---|---|
| ★ Szablewski et al. *The Model Organism Lottery* (Jul 2026) | 2607.01033 | How an organism is trained strongly changes whether interpretability finds it; SFT organisms are too easy. Directly relevant to how we build our liar models. |
| Mack et al. *Mechanistically Eliciting Latent Behaviors* (Jun 2026) | 2606.29604 | Unsupervised low-rank perturbations surface hidden behaviors. |
| Manson. *Curved Inference II: Sleeper Agent Geometry* (Aug 2026) | 2608.24037 | Argues sleeper-agent probe success may be an artefact of how backdoors are inserted. |
| Baez et al. *Dissociating the Internal Representations of Sycophancy* (Jul 2026) | 2607.07003 | Different kinds of sycophancy use different internal mechanisms (cf. MIT Exp. 7). |
| Amram. *Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss* (Oct 2026) | 2610.36202 | Base models pick up covert signals that prompted classifiers miss. |
| Koorndijk. *Feature Elicitation for SAE Steering of Deceptive Behaviour in Small LLMs* (Oct 2026) | 2610.20638 | Alignment faking in *small* models, steered with SAE features. Useful for our model size. |
| Liu et al. *Knowledge-Verified Emergent Deception in LLM Agents* (Aug 2026) | 2608.26372 | Agents lie to users when deployer incentives conflict; belief verified. |

### Activation oracles and automated interpretability
| Paper | arXiv | One line |
|---|---|---|
| Bauer et al. *Building Better Activation Oracles* (May 2026) | 2606.02609 | Training fixes for oracle hallucination and vagueness. |
| Bersia & Gaintseva. *When Activation Oracles Learn Not to Read* (Jul 2026) | 2607.23379 | Oracles have concept-specific blind spots. |
| Torrielli et al. *Confidence and Calibration of Activation Oracles* (May 2026) | 2605.26045 | Five ways to estimate how sure an oracle is. |
| Kabir et al. *Circuit Oracle* (Oct 2026) | 2611.00420 | Ask natural-language questions of attribution graphs automatically. |

### Weight-space programming follow-ups
| Paper | arXiv | One line |
|---|---|---|
| ★ Nusrat & Nusrat. *Weight-Space Programming for Auditing Clinical Decision Networks* (3 Nov 2026) | 2611.05341 | First applied follow-up: frontier LLM reads weights to find shortcuts in medical models that test cases miss. |

### Agentic deception benchmarks (behavioral)
- Golechha & Garriga-Alonso. *Among Us: A Sandbox for Measuring and Detecting Agentic Deception* (2025), 2504.04072. Probes tested on emergent deception in a social game.
- Liu et al. *SWE-DeceptionBench* (Oct 2026), 2610.13626.
- Marioriyad et al. *Lying to Win: Human-AI Games and Parallel-World Probing* (Mar 2026), 2603.07202.

### Found in the saturation round (9 more arXiv queries)
| Paper | arXiv | One line (from title/abstract listing) |
|---|---|---|
| ★ *WeightLoop: Weight-Space Programming with Experimental Feedback* (5 Nov 2026) | 2611.07319 | Weight-space programming that runs experiments between edits; newest follow-up to the MIT post. |
| *Weight-Space Programming for Improving Image-Restoration Diffusion Models* (3 Nov 2026) | 2611.04268 | Another WSP application, with a recursive improvement loop. |
| ★ *Sycophancy Hides Linearly in the Attention Heads* (Jan 2026) | 2601.16644 | Sycophancy readable from attention heads; matches the MIT Exp. 7 finding (head 10.2). |
| *A Capitulation Direction Emerges With Scale and Is Causally Load-Bearing* (Nov 2026) | 2611.02318 | A "give in to the user" direction across eight models that causally drives behavior. |
| *Stress-Testing Alignment Audits With Prompt-Level Strategic Deception* (Feb 2026) | 2602.08877 | Models deceiving the auditing tools themselves. |
| *Deceptive Automated Interpretability: LMs Coordinating to Fool Oversight* (2025) | 2504.07831 | LLM interpreters can be fooled or can collude; relevant if Claude is our weight-reader. |
| *Value-Conflict Diagnostics Reveal Widespread Alignment Faking* (Apr 2026) | 2604.20995 | Behavioral test for alignment faking across models. |
| *Fuzzing Large Language Models to Elicit Hidden Behaviours* (Jun 2026) | 2606.29646 | Perturbation-based search for hidden behaviors. |
| *What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors* (Jul 2026) | 2607.13162 | Persona-vector audits of open models. |
| *Exploitation Without Deception: Dark Triad Feature Steering* (May 2026) | 2605.09773 | Antisocial behavior and deception use separable circuits. |
| *Representation Engineering for LLMs: Survey and Research Challenges* (2025) | 2502.17601 | Survey of reading and steering directions. |

The saturation round returned 138 hits; after removing off-topic ones (physics, wireless, vision), 11 were new and relevant, mostly from the neighboring sycophancy area. The core lie-detection set did not change, which suggests it is close to complete; the broader sycophancy literature (~30 papers in 2026) is only sampled.

## 9. Second completeness pass: lab blogs, last three weeks, non-arXiv venues

### Lab research posts (checked each lab's research page directly)
| Post | Lab, date | One line |
|---|---|---|
| ★ *Fine-Tuned Lie Detectors Failed to Generalize* | Anthropic, Aug 2026 (alignment.anthropic.com/2026/lie-detectors/) | LoRA-tuned Gemma-3 "did you lie?" classifiers on ~200k on-policy lies: 0.95 AUROC on seen lie types, only ~0.70–0.75 on held-out types. |
| *Training Alignment Auditors via Reinforcement Learning* | Anthropic, Oct 2026 | RL-trained LLM auditors that find hidden behaviors. |
| *Introspection Adapters: Training LLMs to Report Their Learned Behaviors* | Anthropic, Apr 2026 | Models self-report behaviors learned in fine-tuning. |
| *Pre-deployment Auditing Can Catch an Overt Saboteur* | Anthropic, Jan 2026 | Audits vs a model trained to sabotage. |
| *Diffuse AI Control on Fuzzy Tasks* | Anthropic, Jun 2026 | Red-teaming defenses against diffuse threats like sandbagging. |
| *Automated Subtle Emergent Misalignment Research* | Anthropic, Oct 2026 | Claude writes data that induces misalignment while evading a monitor. |
| *Shaping and Monitoring Models Through Their Internal Representations* | Goodfire, 5 Nov 2026 | Using internal representations to steer and monitor behavior. |
| *Models Know When They're Reward Hacking* | Goodfire, Sep 2026 | Blog version of Bergen et al. 2609.19101. |
| ★ *Interpreting Language Model Parameters* (Bushnaq et al.) | Goodfire, May 2026 | Decomposes weight matrices into interpretable parts: weight-based interpretability without an LLM reader. |
| *Reasoning Theater: Probing for Performative Chain-of-Thought* | Goodfire, Mar 2026 | Probes show when CoT is performative. |
| *Probe-Based Data Attribution* | Goodfire, Apr 2026 | Traces bad post-training behaviors back to data with probes. |
| *Logits as a New Monitor for Evaluation Awareness* / *Verbalized Eval Awareness Inflates Measured Safety* | Goodfire, May–Jun 2026 | Models that know they're tested look safer than they are. |
| *Discovering Undesired Rare Behaviors via Model Diff Amplification* | Goodfire, Aug 2025 | Amplify the difference between two models to surface hidden behavior (relevant to honest-vs-liar pairs). |
| *Measuring Reward-Seeking via Contrastive Belief Updates* | Apollo, Jul 2026 | Production RL increases reward-seeking even as visible misbehavior drops. |
| *We Need a Science of Scheming* | Apollo, Jan 2026 | Research agenda. |
| *Stress Testing Deliberative Alignment for Anti-Scheming Training* | Apollo + OpenAI, Sep 2025 | Anti-scheming training reduces but doesn't remove scheming; eval awareness grows. |
| *Predictive Concept Decoders* (Dec 2025), *Training LMs to Explain Their Own Computations* (Nov 2025), *Investigating Truthfulness in a Pre-Release o3 Model* (Apr 2025) | Transluce | Activation decoding, self-explanation, truthfulness investigation. |

### arXiv, 20 Oct – 7 Nov 2026 (full scan of cs.LG/CL/AI/CR for 17 topic terms, 602 hits screened)
| Paper | arXiv | One line (title-level unless noted) |
|---|---|---|
| ★ *Activation Probes Are Reliable With a Handful of Positive Examples* (29 Oct) | 2610.39766 | Probes work with very few positive examples; good news for hackathon-size data. |
| *Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment* (5 Nov) | 2611.07586 | CoT monitors vs hidden objectives. |
| *Adapters as Data: The Latent Space of LoRAs* (5 Nov) | 2611.07775 | Treats LoRA weights as data; weight-space analysis of fine-tunes. |
| *Rewarded for Not Getting Caught: Why Alignment Training Selects for Conditional Compliance* (1 Nov) | 2611.02162 | Alignment training can teach "behave when watched". |
| *Characterizing When LMs Take Harmful Actions to Protect Their Assigned Goals* (2 Nov) | 2611.03374 | Goal-guarding behavior. |
| *Four Bridges: Measuring Deception Propensity in a Multi-Agent Game* (2 Nov) | 2611.03600 | Behavioral deception benchmark. |
| *Inter-Agent Influence: Persuasion, Deception, and Coercion in Multi-Agent Systems* (3 Nov) | 2611.04556 | Multi-agent deception evaluation. |
| *Propensities Please: Exposing Alignment-Faking LLMs using Synthetic Document Finetuning* (23 Oct) | 2610.33180 | Building and exposing alignment-faking organisms. |
| *What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation* (21 Oct) | 2610.30491 | Label quality of misbehavior detectors. |
| *Fig: Automated Multiagent Behavior Elicitor* (20 Oct) | 2610.28891 | Automated elicitation of hidden behaviors. |
| *Simulated Control Evals* (22 Oct) | 2610.31593 | AI-control evaluation method. |
| *Not One Belief: Dissociated Confidence Signals in Conversational QA* (26 Oct) | 2610.35694 | Probes show several separate belief signals. |
| *Towards Robust Model Organisms for Eliciting Latent Knowledge* (19 Oct) | 2610.26698 | Better organisms for ELK / lie-detector testing. |
| *CAT-alyzing Honesty: Continuous Adversarial Training for Transferable LLM Honesty* (15 Oct) | 2610.22292 | Adversarial training in activation space for honesty. |
| *Deception by Omission: Language Models Knowingly Hide Their Mistakes* (8 Oct) | 2610.11351 | Lying by leaving things out, which falsehood probes may miss. |
| *Reasoning-Token Spikes Under Prompted Untruthful Responses* (7 Oct) | 2610.10405 | A cheap signal of lying in reasoning traces. |
| ★ *From Probe Scores to Alarm Policies: Operational Validity of Activation Monitors for LM Agents* (3 Oct) | 2610.04575 | Turning probe scores into deployable alarm thresholds (cf. Fredo's recall-at-FPR point). |
| *Accuracy and Uncertainty in Machine Deception Detection* (11 Oct) | 2610.15314 | White paper on detector uncertainty. |

### Found via OpenAlex keyword search (journals, ACL/EMNLP venues, preprints not caught above)
| Paper | Where | One line |
|---|---|---|
| ★ *When Do Internal Probes Beat Reading the Answer? Behavior-Concealed Knowledge* (Sep 2026) | arXiv 2609.04582 | When white-box is actually better than reading outputs. |
| ★ *From Deceptive Outputs to Deceptive Mechanisms: A Causal Framework for LM Deception Research* (Sep 2026) | arXiv 2609.04166 | Framework for going from detection to mechanism, matches our step 2. |
| *Transcoders for Investigating Deception in Language Models* (Jul 2026) | arXiv 2607.14791 | Transcoder features for deception circuits. |
| *Rift: A Conflict Signature for Deception in Language Models* (Jun 2026) | arXiv 2606.17229 | Internal "conflict" signal when lying. |
| *Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning* (May 2026) | arXiv 2609.00351 | Finds hidden behaviors by matching activations. |
| *Deep Minds and Shallow Probes* (May 2026) | arXiv 2605.11448 | Critique of shallow probing. |
| *Confidently Deceptive: Confidence and Deception in LLMs* (May 2026) | arXiv 2607.20444 | Relationship between confidence signals and lying. |
| *Seamless Deception: Larger LMs Are Better Knowledge Concealers* (Mar 2026) | arXiv 2603.14672 | Concealment improves with scale. |
| *Asymmetries in Spontaneous and Instructed Deception* (Aug 2026) | arXiv 2609.00180 | Instructed lies differ from spontaneous ones (matters for training data). |
| *Reference-Grafting Matches Fine-Tuning at Eliciting Sandbagged Capabilities* (Aug 2026) | arXiv 2608.29458 | Cheap way to unlock sandbagging. |
| *Option-Order Randomisation Reveals a Position Attractor in Prompted Sandbagging* (Apr 2026) | arXiv 2604.26206 | Sandbagging artefact. |
| *Noise Injection Reveals Hidden Capabilities of Sandbagging LMs* (NeurIPS 2025) | NeurIPS | Add weight noise; sandbaggers improve, honest models don't. A weight-space sandbagging detector. |
| *Behavior Carries Over: Sandbagging Detection Inspired by Honest Inertia* | EMNLP Findings 2026 | Sandbagging detector. |
| *Cognitive Honeypots: Auditing Observable Evidence of Scheming-Related Behavior* | EMNLP 2026 | Scheming audits. |
| *Cross-Lingual Transfer and Shared Geometry of Deception Probes in LLMs* | MRL 2026 | Deception probes transfer across languages. |
| *Trajectory Signatures of Deception in LLMs* | ACL 2026 | Deception signature across the generation trajectory. |
| *Dissecting Lie Expectancy in Language Models* | BlackboxNLP 2026 | Interpretability of lie expectations. |
| *Investigating the Usefulness of Probes for Causal Intervention* | Springer 2026 | Whether probe directions work for steering. |
| *Unlearners Can Lie: Honesty in LLM Unlearning* | ACL 2026 | Unlearned models may hide knowledge. |
| *Stable Reasoning, Unstable Responses: Mitigating Deception via Stability Asymmetry* | EMNLP 2026 | Deception mitigation. |
| *Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives* (Nov 2025) | arXiv 2511.06626 | Self-report training for hidden goals. |
| *Language Models Use Lookbacks to Track Beliefs* (2025) | arXiv 2505.14685 | Mechanism for tracking (others') beliefs. |
| *Dissecting Deception in LLMs: Mechanisms, Behaviours, and Detection* (Oct 2026) | Preprints.org | Survey of the field. |
| *AI Deception: Risks, Dynamics, and Controls* (Nov 2025) | arXiv 2511.22619 | Survey. |

### Sycophancy (neighboring field): papers on internal mechanisms
Behavioral-only sycophancy benchmarks (≈40 in 2026: medical, legal, multilingual, multimodal, agent-memory) are left out on purpose; these are the ones that look inside the model.
| Paper | Where | One line |
|---|---|---|
| ★ *LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit* (Apr 2026) | arXiv 2604.19117 | Sycophancy and lying share a circuit: links MIT Exp. 7 to lie probes. |
| *A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy* (Jan 2026) | arXiv 2601.18939 | Small targeted edit fixes sycophancy (cf. weight editing). |
| *Locating Sycophancy Across Training Stages in Open LMs* (Oct 2026) | arXiv 2610.37788 | Where in training sycophancy appears. |
| *The Development of Sycophancy: Pretraining to Post-Training* (Oct 2026) | arXiv 2610.26655 | Same question, different method. |
| *Tracing Mechanisms of Sycophantic Agreement* (Sep 2026) | arXiv 2609.35822 | Circuit tracing. |
| *From Trait Vectors to Circuits: Refusal and Sycophancy* (Sep 2026) | arXiv 2609.22224 | Directions to circuits. |
| *Less Sycophancy, Stronger Refusal? Lessons from Mech Interp* (Sep 2026) | arXiv 2609.35544 | Interaction of two directions. |
| *Detecting and Controlling Sycophancy with Cascading Linear Features* (Jun 2026) | arXiv 2606.26155 | Linear detection + control. |
| *Token-Level Diagnosis of Sycophancy with Attribution-Guided Steering* (Jul 2026) | arXiv 2607.28906 | Token-level localization. |
| *THESIS-MoE: Extraction and Steering of Sycophancy in Mixture-of-Experts* (Aug 2026) | arXiv 2608.15687 | MoE-specific. |
| *Is Sycophancy a Form of In-Context Updating?* (Oct 2026) | arXiv 2610.28325 | Mechanistic framing. |
| *How Does Alignment Tuning Shape Representations of Sycophancy?* (Jul 2026) | arXiv 2607.18114 | Representation change after alignment. |
| *A Mechanistic View of Authority Hierarchy in LLM Sycophancy* (Jul 2026) | arXiv 2607.00415 | Mechanism of deference. |
| *Sycophantic Anchors in Reasoning Models* (Jan 2026) | arXiv 2601.21183 | Localizing agreement in reasoning. |
| *Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors* | EMNLP 2026 | Several separable mechanisms. |
| *Latent Truth Decoding: Mitigation of Sycophancy in Small LMs* | GroundLM 2026 | Small-model, training-free. |
| *Steering Interference Reflects the Model's Defaults* (Sep 2026) | arXiv 2609.06951 | Caveat for steering experiments. |
| *Playing Devil's Advocate: Persona Vectors Rival Targeted Steering for Sycophancy* (May 2026) | arXiv 2605.21006 | Persona vectors vs targeted steering. |
| *What Counts as AI Sycophancy? A Taxonomy* (May 2026) | arXiv 2605.21778 | Definitions. |
| *From Sycophancy to Deception: A Unified Taxonomy* (Apr 2026) | arXiv 2604.04788 | Definitions linking both. |

### Saturation test: reference lists of the five newest papers
I extracted every arXiv reference from 2609.39807, 2610.27577, 2610.39766, 2610.04575 and 2607.20479 and compared them with everything found so far. They cite 79 unique arXiv papers from 2023 on. 53 were already on the list; 26 were missing, and 17 of those were general agent-safety or prompt-injection monitoring, outside this field. The 9 relevant ones are added here; none changes the core picture.
| Paper | arXiv | One line |
|---|---|---|
| Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training* (2024) | 2401.05566 | The original backdoored-deceptive model organism (probes paper above builds on it). |
| Tillman & Mossing. *Task-Specific Prompts and SAEs for Activation Monitoring* (OpenAI, 2025) | 2504.20271 | Prompted probes and SAE features for monitoring. |
| Li et al. *Do Activation Verbalization Methods Convey Privileged Information?* (2025) | 2509.13316 | Do activation oracles reveal anything not already in the text? |
| Wilhelm et al. *Monitoring Emergent Reward Hacking via Internal Activations* (2026) | 2603.04069 | Activation monitoring of reward hacking during generation. |
| Wilhelm & Kao. *From Reward-Hack Activations to Agentic Risk States* (2026) | 2606.06223 | Same, extended to agents. |
| Duan. *Do Activation Monitors Survive Model Updates?* (2026) | 2606.15980 | Probes go stale when the model is updated. |
| Xu et al. *When Agents Go Rogue: Activation-Based Detection of Malicious Behaviors* (2026) | 2607.06807 | Activation detection for agents. |
| Najt et al. *SLEIGHT-Bench: Evasion Attacks Against Agent Monitors* (Anthropic, 2026) | 2605.16626 | Benchmark of ways agents evade monitors. |
| Advani. *From Confident Closing to Silent Failure: False Success in LLM Agents* (2026) | 2606.09863 | Agents claiming success falsely (Fredo's setting). |

## How this list was built (completeness check, 7 Nov 2026)

1. Every 2023–2026 reference in the FIBS paper (v4, 20 Oct 2026), the newest major paper in the field.
2. About 12 web searches on probes, benchmarks, sandbagging, obfuscation, activation oracles and weight reading.
3. arXiv API: 13 + 9 keyword queries (deception/lie + probe/activations/LLM, knowing falsehood, hidden objective, sandbagging, activation oracle, weight space, white-box scheming monitor), newest first, 60 results each, from 2025 on. 159 unique hits, every one screened by title; ~90 were off-topic (hallucination, cyber honeypots, games, privacy) and were left out.
4. OpenAlex forward citations (papers citing) of 8 key papers: Goldowsky-Dill, FIBS, Liars' Bench, Obfuscation Atlas, Smith et al., "Did you lie?", Black-to-White, Activation Oracles. Semantic Scholar was rate-limited and could not be used.

5. Second pass: full arXiv scan of 20 Oct–7 Nov 2026 (cs.LG/CL/AI/CR, 17 topic terms, 602 hits screened by title); the research pages of Anthropic Alignment, Apollo, Goodfire and Transluce read directly; OpenAlex keyword search (8 queries, 691 hits) for journal and ACL/EMNLP papers. Semantic Scholar refused every request without an API key, so OpenAlex replaced it.

6. Saturation test against the reference lists of the five newest papers (see the table above).

Deliberately left out: deception *by humans* (text/speech lie detection), cyber-deception (honeypots), jailbreaks, hallucination-only work, multimodal sycophancy, and behavioral-only sycophancy benchmarks. Low-quality self-published Zenodo preprints were also excluded. Summaries are from abstracts or listings, not full reads, except FIBS and the MIT post.

## Gaps found (useful for picking our project)

- No paper combines a lie probe with LLM weight-reading. That is exactly the team's idea, so it looks new.
- Weight Oracles and the MIT post disagree in method (fine-tuned reader vs zero-shot frontier model), and 2611.08070 doubts the zero-shot claims. A small honest-vs-liar pair test would speak to all three.
- Probes on very small models (270M–0.8B) are barely studied; most results are 9B+, and 2604.13386 finds probe accuracy rises with model size. 2610.20638 is one of the few small-model studies.
- How the liar models are built matters (2607.01033): plain fine-tuned organisms are easy to detect, so results on them may overstate what probes can do.
