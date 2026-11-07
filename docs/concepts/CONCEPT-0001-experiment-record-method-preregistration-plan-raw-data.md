---
id: CONCEPT-0001
uid: 01M6S6SWHB99X5HGMY1XBJQW3C
title: 'Experiment record method: preregistration plan, raw data, result note'
status: proposed
created: '2026-11-06T19:54:06+00:00'
---

# CONCEPT-0001 — Experiment record method: preregistration plan, raw data, result note

## Context

The Open Science Hackathon (2026-11-07) produces experiments run with Claude Code. Their
measurements must be reproducible and citable, and an experiment's hypothesis must be fixed
before its data exists, otherwise the result can be fitted to the data after the fact.
This knowledge base knows the types decision, concept, idea, plan, review, note,
initiative, epic and task. It has no experiment type, and the engine is vendored and pinned,
so this method maps experiments onto the existing types instead of extending the engine.

## The Method

1. **Idea**: `kb.py new idea --title "..."`: a question worth testing.
2. **Preregistration**: `kb.py new plan --title "EXP-<slug>: ..." --link derived_from=IDEA-NNNN`.
   Before any run, write down the hypothesis, the metric, the sample size, the stopping rule
   and the analysis. Commit it. The commit timestamp is the preregistration proof.
3. **Raw data**: lands under `data/experiments/<slug>/` (plain files: CSV, JSONL, logs, plus
   a `MANIFEST.sha256`). This directory is not an entity type; the plan and the result note
   point at it.
4. **Result**: `kb.py new note --title "EXP-<slug> result" --link evidence=repo:claude-impact-lab/docs/data/experiments/<slug> --link derived_from=PLAN-NNNN`.
   Report the measured values, the deviations from the plan, and failed runs too.
5. **Decision** (only if a result changes how the project is built): `kb.py new decision`
   with `--link formalized_as=CONCEPT-...`.

Never hand-number a file. Never edit a preregistration after its first data point; amend it
with a dated note instead.

## When to use / When NOT to use

Use it for every measurement that a claim in a paper, slide or README will rest on.
Do not use it for exploratory runs nobody will cite; those stay in `data/scratch/`
(git-ignored by the project root).

## Evidence

## Operator Notes
