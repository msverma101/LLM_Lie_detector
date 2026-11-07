# CLAUDE.md — LLM Lying Detection (Claude Impact Lab workshop)

## Project status: DISCOVERY PHASE

We are a team at the **Claude Impact Lab workshop**, working on **LLM lying detection**
(Discord channel: `llm-lying-detection`).

We have **not** fixed the project scope yet. Right now we are collecting material and
looking for a concrete project definition. **Everything here is open for discussion.**
Nothing is a decision until the group agrees on it.

When you help with this repo (as Claude or a human):
- Treat ideas as proposals, not plans. Mark open questions explicitly.
- Prefer small, reversible steps: literature notes, quick experiments, sketches.
- Put new ideas in `notes/` and new sources in `literature/` + `literature/README.md`.
- Don't push to `main` directly. Use a personal suggestion branch (e.g. `soenkessuggestions`)
  and discuss before merging.

## Working idea (to be discussed)

One question from the chat: **combine two methods** for detecting when an LLM lies:

1. **White-box probes on activations** (FIBS paper): linear/MLP/axial probes trained on a
   large deception dataset detect "knowing falsehoods", including *introspective* lies
   that can't be seen from the text alone.
2. **Weight-space programming** (MIT blog post): frontier LLMs can read and edit raw
   neural-network weights directly. Possible angle: can an LLM inspect weights or
   probe weights to find, explain or remove a "lying" mechanism?

Open questions:
- Which is the "second method" we should combine with the first? (asked in chat, not answered yet)
- Detection only, or detection + intervention (editing weights)?
- Which open-weight model is small enough for a workshop (e.g. Qwen3.5-0.8B)?
- What is our ground truth for "lying"? Contextual vs. introspective falsehoods (see FIBS §2.1).
- What can we show in the workshop's time and compute budget?

## Repository layout

```
CLAUDE.md               this file: project context and status
README.md               short project description
literature/
  README.md             annotated list of all sources (papers, posts, repos)
  papers/               downloaded PDFs
  articles/             saved blog posts / web articles
notes/                  meeting notes, ideas, discussion summaries
experiments/            code for quick experiments (empty for now)
docs/                   project definition, once we agree on one
```

## Key sources (details in `literature/README.md`)

- **Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception**
  (Hollinsworth, Spies, Diriba, Gleave, Cundy, FAR.AI, arXiv:2610.12445).
  Code: https://github.com/AlignmentResearch/fibs-probes
- **Sparks of Weight-Space Programming: LLMs can natively read & write neural network
  weights** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026).
- Related team repos: `Fredo220/verified-reasoning-monitoring`, `CorvinLabs/CorvinOS`, `veegee82`.
