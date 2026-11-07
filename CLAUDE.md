# claude-impact-lab — Conventions for Claude Code

## Purpose

Working project for the **Open Science Hackathon for Claude Code on 2026-11-07**.
Goal: run reproducible scientific experiments with Claude Code and CorvinOS Agentic Compute
(ACS, compute runs, workflows), and record every idea, method, decision, preregistration
and measurement in a knowledge graph that is created only through gates.

## Layout

| Path | What |
|---|---|
| `docs/` | Knowledge graph (own git repo, created by the CorvinOS Knowledge Graph plugin's `kb_init.py`) |
| `docs/concepts/`, `docs/decisions/`, `docs/ideas/`, `docs/implementation-plans/`, `docs/notes/`, `docs/adversarial-reviews/` | Entities, created by `kb new` only |
| `docs/kb/initiatives`, `docs/kb/epics`, `docs/kb/tasks` | Work items (I-01 = the hackathon) |
| `docs/data/experiments/<slug>/` | Raw measurement data + `MANIFEST.sha256` (not an entity type) |
| `docs/data/scratch/` | Exploratory output nobody will cite (git-ignored) |
| `.mcp.json` | CorvinOS MCP servers (forge/compute, orchestration/ACS, skill_forge) for plain `claude` sessions here |

## The knowledge graph — gates only

Run kb through the pinned engine with `--repo` (the L10 path gate pins this exact path):

```bash
KB="python3 /home/shumway/projects/Corvin-Knowledge/scripts/kb.py --repo /home/shumway/projects/claude-impact-lab/docs"
$KB find <words>                       # always search first, amend instead of duplicating
$KB new idea --title "..."
$KB new plan --title "EXP-<slug>: ..." --link derived_from=IDEA-0001   # preregistration
$KB new note --title "EXP-<slug> result" --link derived_from=PLAN-0001 --link evidence=repo:claude-impact-lab/docs/data/experiments/<slug>
$KB new concept --title "..."
$KB new decision --title "..." --link formalized_as=CONCEPT-0001
$KB new task --title "..." --epic E-001 --dod "..."
$KB task T-0001 in_progress|blocked|done|cancelled [--reason ...] [--dod ...]
$KB check                              # must report blocking: 0 before any push
```

Rules:
- Never hand-number or hand-create an entity file; `kb new` numbers, links and commits.
- Task status changes only through `kb task`.
- The experiment method is CONCEPT-0001: idea, preregistration plan (committed BEFORE the
  first data point), raw data under `data/experiments/`, result note with evidence link.
  A preregistration is never edited after data exists; amend with a dated note.
- `docs/scripts/kb.py` is a vendored copy of the same engine (identical sha256); prefer the
  Corvin-Knowledge path above, since the path gate pins that one.

## Where ADRs go (scope boundary)

- Decisions **about this hackathon project** (experiment design, tooling choices for the lab)
  go into `docs/decisions/` via `kb new decision`.
- Decisions **about CorvinOS itself** (core, layers, compliance, plugins) never go here: they
  belong in `/home/shumway/projects/Corvin-Knowledge/decisions/` (ADR-0516), created with
  `python3 /home/shumway/projects/Corvin-Knowledge/scripts/kb.py new decision`.
- Never copy an ADR from one knowledge base into the other; link it as a URI instead
  (`--link related=repo:Corvin-Knowledge/decisions/ADR-NNNN-...md`).
- Ids are local to each base: this project's `ADR-0001` is not Corvin-Knowledge's `ADR-0001`.
  Always qualify cross-base references.

## Agentic Compute (CorvinOS)

Available through `.mcp.json` (approve the three servers on the first `claude` start here):
- `mcp__corvin_orchestration__acs_delegate`: open-ended manager/worker loop; use `dry_run: true` to validate.
- `mcp__corvin_orchestration__workflow_run` / `workflow_resume` / `workflow_list_paused`: AWP DAG workflows.
- `mcp__forge__compute_run` / `compute_submit` / `compute_status` / `compute_result` / `compute_abort`: parameter sweeps (grid/random/bayesian), pipeline and HAC engines over an existing Forge tool.
- `mcp__forge__data_register` / `data_snapshot`: register local data files for compute runs.

Verified 2026-11-06 on this install, after the licence fix in CorvinOS `f3351f9b9` (member
licence is now loaded in every MCP subprocess; before it, `forge_tool` was refused and compute
ran on the 500-iteration trial):
- `forge_tool` creates tools (member), `compute_run` runs `mode: licensed, tier: member`.
- Strategies grid, random and bayesian all ran; `compute_submit` engines `pipeline` (2 stages,
  `converged`) and `hac` (2 sub-managers, composite loss) ran. `acs_delegate` ran a real task.
- NOT verified: `workflow_run` (registered workflows may spend LLM budget), steering/backprop
  gates, compute fabric (`fabric_allowed` is false for the operator licence).

Traps:
- **The compute worker reads ONLY the user-scope tool registry.** A tool forged in task or
  session scope makes every iteration fail with `ToolError: unknown tool` and the run ends
  `budget_exhausted` with `best_loss: null` — it looks like success. Always
  `forge_promote(name, to="user")` before `compute_run`/`compute_submit`.
- A persona may only forge tools named `assistant.<name>` (namespace gate).
- `param_grid` keys must be the tool's input fields; a wrong key fails every iteration with a
  `SchemaError` and the same `best_loss: null`. Check `<run>/iterations/0001.json` first.
- Set `steering_gate: false` (pipeline) / `backprop_gate: false` (HAC) for unattended runs, else
  they wait for a human action.

## Language and style

- Repo content (code, file names, commit messages, entity files): English.
- Chat replies to the operator: German with correct umlauts.
- Status icons only in chat replies, never in files, commits or code.
- No success claim without captured output (exit code, run id, measured value).
