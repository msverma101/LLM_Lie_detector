---
id: NOTE-0001
uid: 01M6SA0VMNFFE5V5HJ8F9DNAJN
title: 'Agentic Compute verification 2026-11-06: licensed member, all engines except workflow_run'
created: '2026-11-06T20:50:20+00:00'
links:
- rel: related
  to: T-0001
---

# NOTE-0001 — Agentic Compute verification 2026-11-06: licensed member, all engines except workflow_run


Measured on the maintainer's install with the repaired licence path (CorvinOS `f3351f9b9`).
Tool under test: `assistant.impactlab_quadratic_probe`, loss = (x - 3)^2, optimum x = 3.

| Engine / strategy | Handle | Result |
|---|---|---|
| flat, grid (7 points) | `compute_M3k-j-I96cpvBuswiSsV1h` | best x=3, loss 0.0, `grid-exhausted` |
| flat, bayesian (8 it, seed 7) | `compute_SxXTBvInkS2tpyLyuiguas` | best x=3, loss 0.0 |
| flat, random (8 it, seed 7) | `compute_OUhQvhMTkWyjqAdiHH6QZt` | best x=2, loss 1.0 (8 of 11 points sampled) |
| pipeline, 2 stages | `pipeline_kllSzGv3FE194onvSgfFp_` | `converged`, coarse 1.0, fine 0.0 |
| hac, 2 sub-managers | `hac_CUvKbUsfz5vnJD4yOpfehY` | composite 0.5 = 0.5*1.0 + 0.5*0.0, both `budget_exhausted` |
| acs_delegate (real run) | `acs-1793994909-7df12858` | `{"mean": 5}` in 17.8 s |

Licence evidence: `compute.license.checked` on the tenant chain reads `mode: licensed, tier:
member` for these runs; the trial counter (`trial_compute.json`) stayed at 2.

Failure modes seen (both look like success because the run state is `budget_exhausted`, not
`failed`; only `best_loss: null` and `iterations/0001.json` show it):
- wrong `param_grid` keys: `SchemaError: 'row' is a required property`;
- tool not in the user-scope registry: `ToolError: unknown tool` — promote with
  `forge_promote(name, to="user")` first.

Not verified: `workflow_run` (registered workflows may spend LLM budget), gates, fabric.
Raw run artifacts live in the CorvinOS tenant `compute/runs/<handle>/`, not copied here.
