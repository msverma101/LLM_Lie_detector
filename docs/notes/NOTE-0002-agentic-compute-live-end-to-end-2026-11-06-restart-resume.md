---
id: NOTE-0002
uid: 01M6SGHF1JTW0VY31FN3A2ZXDT
title: 'Agentic Compute live end-to-end 2026-11-06: restart resume, refunds, flat submit, adapter licence,
  ACS skill re-use'
created: '2026-11-06T22:44:16+00:00'
links:
- rel: related
  to: NOTE-0001
- rel: related
  to: T-0001
---

# NOTE-0002 — Agentic Compute live end-to-end 2026-11-06: restart resume, refunds, flat submit, adapter licence, ACS skill re-use


Live run on the maintainer's install (member licence) after CorvinOS e06f45810, b5dbdde61, 6f24dc9d5.

| Point | Live evidence |
|---|---|
| Restart continues a run | a real `systemctl restart corvin-compute@_default` in the middle of run `compute_CwfKzX8OtkE1Tdn1MzoMdj` (120 points): on disk right after the restart `state=running` (not `aborted`); final 120 iterations, the 5 files written before the restart byte-identical, every x once in grid order, best x=3 loss 0.0 |
| Refused runs are free | daily units 16 -> 16 -> 16 across a `ToolNotAvailable` and a `ParamGridMismatch` refusal; an accepted run costs exactly 1 (17) |
| `compute_submit(engine=flat)` | `budget_exhausted`, best x=3 loss 0.0, charged once |
| Adapter licence | member + `spotify-charts-postgres` binding accepted by the worker (it loaded the live licence), run finished best_loss 0.0; the Free refusal is pinned in the sandbox tests (the install's own licence was not touched) |
| ACS skill grading and re-use | run `acs-1794004778-d748d95f` forged `acs.holdout_choice_guide` (meta.persona=assistant); `assistant` graded it (1 grade, mean 0.3 = clamped), `coder` was refused (`namespace-gate`); run `acs-1794004955-1fb9963a` injected exactly that skill (`acs.worker_skills_injected skill_count=1`) and its worker used the guide (28-day holdout, multiple of 7, last-week baseline) |

Found by the live run, not by the tests: with `dynamic_skills.enabled` a worker received three unrelated graded
skills of the persona; `acs_delegate` now sets `only_prefix: "acs."` (6f24dc9d5).
Scripts: `claude-playground/agentic-compute-2026-11-06/e2e_live_*.py`.

Review of the fixes and the final live re-check (CorvinOS 9b38917eb): a second real `systemctl restart` in the middle
of run `compute_xIX94foo_RRxlKvscnr0yB` (120 points) ended with 120 iterations, the pre-restart files byte-identical,
every x once in order. Not shown live: that the socket answers while a recovered run is still going (the service
needs ~9 s to start and the run was done by then) — `test_worker_restart_semantics.py` proves it (red with the
previous worker: socket after 13.6 s instead of < 6 s). Live skill scope on the real registry: ordinary turn gets no
ACS skill; ACS worker of owner `assistant` gets `acs.holdout_choice_guide`; owner `coder` gets none.
