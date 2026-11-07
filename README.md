<p align="center">
  <img src="assets/banner.svg" alt="LLM Lie Detector: catching knowing falsehoods in activations and weights" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-discovery%20phase-f59e0b" alt="status: discovery phase">
  <img src="https://img.shields.io/badge/Claude%20Impact%20Lab-2026-8b5cf6" alt="Claude Impact Lab 2026">
  <img src="https://img.shields.io/badge/topic-AI%20safety%20%C2%B7%20interpretability-14b8a6" alt="topic: AI safety and interpretability">
  <img src="https://img.shields.io/badge/discussion-open-22c55e" alt="discussion: open">
</p>

> **Can we tell when a language model says something it believes is false, and find *where* in the network that lie lives?**

We are a team at the **Claude Impact Lab workshop** working on **LLM lying detection**.
We are in the **discovery phase**: collecting sources, comparing ideas and shaping a project
definition together. Everything here is a proposal, open for discussion.

---

## 💡 The idea in one picture

Two recent results each solve half of the problem:

```mermaid
flowchart LR
    subgraph A["🔬 FIBS probes (FAR.AI)"]
        A1["Read activations<br/>at runtime"] --> A2["Lie score<br/>98.7% AUROC"]
    end
    subgraph B["🧠 Weight-space programming (MIT)"]
        B1["LLM reads raw<br/>weights"] --> B2["Explains & edits<br/>the network"]
    end
    A2 -- "knows THAT it lies<br/>but not WHERE" --> C{{"🤝 Combine"}}
    B2 -- "knows WHERE<br/>but needs a signal" --> C
    C --> D["🎯 Find, explain and maybe<br/>fix the lying mechanism"]

    style A fill:#0f3b37,stroke:#5eead4,color:#f5f3ff
    style B fill:#3d2c0c,stroke:#fbbf24,color:#f5f3ff
    style C fill:#2e2552,stroke:#c4b5fd,color:#f5f3ff
    style D fill:#14532d,stroke:#4ade80,color:#f5f3ff
```

**The gap we want to close:** in the weight-space experiments, a frontier LLM could **not** tell
an honest model from a sycophantic one by reading weights alone. Once it got an activation
signal (attention patterns), it was right **3 out of 3** times. A FIBS lie probe is exactly that
kind of signal, and it is specific to lying.

---

## 📚 The two sources at a glance

| | 🔬 **Telling FIBS** | 🧠 **Sparks of Weight-Space Programming** |
|---|---|---|
| **Who** | FAR.AI · [arXiv:2610.12445](https://arxiv.org/abs/2610.12445) | MIT CSAIL · [blog post](https://rikiyat.github.io/blog/weight-space-programming/) |
| **Looks at** | Activations at runtime | Weights (plus attention patterns) |
| **Gives you** | A lie score per answer | An explanation, and possibly an edit |
| **Headline** | 98.7% AUROC on SHADE-Arena; up to 99.7% on hidden-goal lies | Edits 0.016% of Qwen3.5-0.8B → MATH-500 7.9% → 48.1% |
| **Strength** | Very accurate, cheap | Can say *why* and *where* |
| **Weakness** | Black box: no explanation | Fails on behavior from weights alone (Exp. 7) |

👉 Full plain-language explainer: **[docs/paper-explainer.md](docs/paper-explainer.md)**

### What counts as a "lie"?

FIBS defines a **knowing falsehood**: the model says something *it believes* is false.

```mermaid
flowchart TD
    L["Model says <i>s</i>,<br/>but believes <i>not s</i>"] --> Q{"How do we know<br/>what it believes?"}
    Q -->|"Truth is in the context<br/>(e.g. a tool output)"| C["📄 Contextual lie<br/>visible in the transcript"]
    Q -->|"It says the opposite<br/>once pressure is removed"| N["🔍 Introspective lie<br/>(neutral elicitation)"]
    Q -->|"We trained the belief in<br/>(e.g. a hidden goal)"| T["🔍 Introspective lie<br/>(model organism)"]
    C --> BB["Black-box monitors<br/>can catch it"]
    N --> WB["Only white-box<br/>probes can catch it"]
    T --> WB

    style C fill:#1e3a5f,stroke:#60a5fa,color:#f5f3ff
    style N fill:#3d2c0c,stroke:#fbbf24,color:#f5f3ff
    style T fill:#3d2c0c,stroke:#fbbf24,color:#f5f3ff
    style WB fill:#0f3b37,stroke:#5eead4,color:#f5f3ff
```

---

## 🎯 Proposed research question

> Does a frontier LLM pick out which of two otherwise identical small models was fine-tuned
> to lie **more reliably** when it gets **lie-probe scores alongside the raw weights** than
> with either one alone?

Details, sub-questions and risks: **[docs/research-question.md](docs/research-question.md)**
*(proposal by soenkebartling, not yet agreed)*

### Proposed experiment

```mermaid
flowchart LR
    M["Small model<br/>Gemma 3 270M /<br/>Qwen3.5-0.8B"] --> H["😇 Honest copy"]
    M --> X["🤥 Lying copy<br/>(denies a hidden goal)"]
    F[("FIBS dataset<br/>163k transcripts")] --> P["Linear lie probe"]

    H --> AUD
    X --> AUD
    P --> AUD

    subgraph AUD["🕵️ Claude as auditor (no access to outputs)"]
        direction TB
        CA["A · weights only"]
        CB["B · probe scores only"]
        CC["C · weights + probe"]
    end

    AUD --> R["📊 Who finds the liar?<br/>accuracy + localization"]

    style X fill:#4c1d1d,stroke:#f87171,color:#f5f3ff
    style H fill:#14532d,stroke:#4ade80,color:#f5f3ff
    style P fill:#0f3b37,stroke:#5eead4,color:#f5f3ff
    style CC fill:#2e2552,stroke:#c4b5fd,color:#f5f3ff
```

**Hypothesis:** C > A and C ≥ B. The probe tells the auditor *where to look*, and the weights
add an explanation the probe alone can't give.

### Four levels of ambition

```mermaid
flowchart LR
    L1["1️⃣ Detect<br/>Probe as a tool<br/>for the weight auditor"] --> L2["2️⃣ Explain<br/>Trace the probe direction<br/>back into heads / MLPs"]
    L2 --> L3["3️⃣ Intervene<br/>Edit the weights,<br/>measure with probe + behavior"]
    L3 --> L4["4️⃣ Audit<br/>Find a hidden lie<br/>before any eval exists"]

    style L1 fill:#14532d,stroke:#4ade80,color:#f5f3ff
    style L2 fill:#0f3b37,stroke:#5eead4,color:#f5f3ff
    style L3 fill:#3d2c0c,stroke:#fbbf24,color:#f5f3ff
    style L4 fill:#4c1d1d,stroke:#f87171,color:#f5f3ff
```

Level 1 is a realistic workshop core; levels 3–4 are stretch goals.
See [notes/2026-11-07_idea_probes-x-weight-space.md](notes/2026-11-07_idea_probes-x-weight-space.md).

---

## 🗺️ Where we are

```mermaid
flowchart LR
    S1["✅ Collect sources"] --> S2["🟡 Discuss ideas<br/><b>← we are here</b>"]
    S2 --> S3["⬜ Agree on<br/>project definition"]
    S3 --> S4["⬜ Pick model,<br/>data, compute"]
    S4 --> S5["⬜ Run experiments"]
    S5 --> S6["⬜ Present results"]

    style S1 fill:#14532d,stroke:#4ade80,color:#f5f3ff
    style S2 fill:#3d2c0c,stroke:#fbbf24,color:#f5f3ff
```

### ❓ Open questions

- Do FIBS probes work on models as small as 270M–0.8B? The paper's main results are on Qwen3.5-397B.
- Does a tiny model hold beliefs stable enough for "knowing falsehood" to mean anything?
- How many honest/lying pairs can we fine-tune in the workshop time?
- **Goodhart risk:** an edit might blind the probe without removing the lie. We need a
  behavioral ground truth that doesn't depend on the probe.
- **Dual use:** ways to blind a probe are also ways to evade it. How do we report that responsibly?

💬 Have an opinion? Open an issue, comment on a branch, or bring it up in `#llm-lying-detection` on Discord.

---

## 📁 Repository layout

```
.
├── README.md                 ← you are here
├── CLAUDE.md                 project context for Claude (and humans)
├── assets/                   graphics for this README
├── docs/
│   ├── paper-explainer.md    both papers in plain language
│   └── research-question.md  proposed research question
├── literature/
│   ├── README.md             annotated list of all sources
│   ├── papers/               PDFs (FIBS paper)
│   └── articles/             saved blog posts (weight-space programming)
├── notes/                    kickoff summary, idea notes
└── experiments/              code (coming once we agree on a plan)
```

## 🔗 Sources & related work

- 📄 **Telling FIBS:** Hollinsworth, Spies, Diriba, Gleave, Cundy (FAR.AI, 2026). [arXiv](https://arxiv.org/abs/2610.12445) · [code & data](https://github.com/AlignmentResearch/fibs-probes)
- 📝 **Sparks of Weight-Space Programming:** Takehi, Teoh, Zhang, Khattab (MIT CSAIL, 2026). [blog](https://rikiyat.github.io/blog/weight-space-programming/)
- 🧪 [Fredo220/verified-reasoning-monitoring](https://github.com/Fredo220/verified-reasoning-monitoring): internal signals for false success reports
- 🤖 [CorvinLabs/CorvinOS](https://github.com/CorvinLabs/CorvinOS): agentic OS connecting Claude Code to Discord & more

Full annotated list: [literature/README.md](literature/README.md)

---

<p align="center"><sub>Built during the Claude Impact Lab workshop · team <code>llm-lying-detection</code> · 2026</sub></p>
