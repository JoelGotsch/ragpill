# Plan: Architecture Decision Records (ADRs)

**Status:** Proposed
**Date:** 2026-05-13

## Motivation

Decisions made in Claude Code chat sessions disappear once the session ends.
The reasoning behind a choice — what alternatives were weighed, what
constraint forced the call, what we tried and rejected — only lives in
PR descriptions or scattered "Decisions resolved" sections in
`plans/*.md`. When a future change touches the same area, neither the
human nor the next agent can recover the *why*, only the *what*.

Recent concrete examples that illustrate the gap:

- **2026-05-13** — Disabling `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING` in unit
  tests. MLflow 3.12 flipped the default to async; our unit tests
  immediately read back what they just wrote, so they raced. Decision:
  force sync in `tests/conftest.py`; keep async coverage in the
  `mlflow-integration` job. *Currently this rationale lives only in the
  conftest docstring and this chat — it should be ADR-0001.*
- **2026-05-05** — Using `pd.to_json(orient="table")` over `orient="split"`
  for `EvaluationOutput` roundtrip, because `split` collapses int-valued
  floats. Lives in `plans/llm-readable-outputs-and-mcp.md` under
  "Deviations from the original plan."
- Several "Decisions resolved during planning" entries scattered across
  `plans/llm-readable-outputs-and-mcp.md`,
  `designs/llm-readable-outputs-and-mcp.md`, etc.

## Goals

1. A lightweight ADR format that is fast to write and easy to find.
2. Code links to ADRs by number — `See ADR-0007` — so a reader of the
   code can pull up the rationale without spelunking through git history.
3. An impact taxonomy so we know which decisions warrant the full ADR
   treatment vs. a one-paragraph note.
4. Backfill: capture the decisions that have already been made but never
   written down.

## Non-goals

- Replacing `designs/` or `plans/`. Designs describe a *system*; plans
  describe a *change*; ADRs capture a *single decision* and its
  alternatives. They cross-reference each other.
- A heavyweight RFC process. ADRs are a record, not a gate.

## Format

### Location and naming

`docs/adr/NNNN-<tier>-kebab-title.md`, where `NNNN` is a pure
sequential ID (zero-padded, four digits) and `<tier>` is one of
`large` / `medium` / `small`. Examples:

- `0001-large-mlflow-canonical-backend.md`
- `0007-small-async-trace-logging-disabled-in-tests.md`

Why pure-sequential IDs (not impact-banded):

- The number is an **immutable handle**. Code references like
  `# See ADR-0007` must keep working forever; banding impact into the
  number forces renumbering whenever a decision's impact reclassifies,
  which breaks every reference.
- It matches every common ADR tool and external reference
  (`adr-tools`, MADR, log4brains, `adr.github.io`).
- Reclassification is real: "small" decisions become load-bearing more
  often than the reverse. The ID has to survive that.

Glanceability comes from the **filename suffix and the index page**,
not the number. From a file listing or an `ls docs/adr/`, the
`-large-` / `-medium-` / `-small-` segment is immediately visible;
from an index page (Task 7), ADRs are grouped/badged by impact.

Numbers are never reused. Superseded ADRs link forward to the one
that replaces them but keep their original number and filename.

### Template

```markdown
# ADR-NNNN: <Short imperative title>

**Status:** Proposed | Accepted | Superseded by ADR-NNNN | Deprecated
**Date:** YYYY-MM-DD (when the decision was made; not when it was written down)
**Impact:** Large | Medium | Small
**Related:** designs/<...>.md, plans/<...>.md, ADR-NNNN

## Context
What forced this decision? Constraints, prior state, what broke.

## Decision
The choice, stated as a single sentence first, then any qualifiers.

## Alternatives considered
Bullet list. For each: what it was, why it was rejected.

## Consequences
- What gets easier
- What gets harder
- What we now have to remember (e.g. "tests run sync; integration job is the only async coverage")

## References
Code paths, PR links, chat session IDs if relevant.
```

### Impact tiers

Tier dictates rigor, not file structure — every ADR uses the same template.

| Tier | When | Expected effort |
|---|---|---|
| **Large** | Public API shape, cross-layer architecture, dependency choices, anything that other code now has to integrate around | Full template; review with stakeholder; link from `designs/` |
| **Medium** | Internal module convention, non-obvious data shape (e.g. dtype-preserving JSON), opt-in defaults | Full template; reviewer optional |
| **Small** | Test-suite convention, narrow workaround, single-file invariant | Template fields can be one line each; written by the author of the change |

If you're unsure, write it as Medium. Better an over-documented small
decision than an under-documented big one.

### Code references

- In code that embodies an ADR-worthy decision, add a short comment:
  `# See ADR-0001 — sync trace logging in tests.`
- Don't restate the decision in code; just point at the ADR. Code rots,
  decisions rot slower, the link tells the reader where to look for fresh
  context.
- Tests, fixtures, and conftest files count — anywhere a future
  contributor might ask "why is this here?"

## Tasks

1. Decide directory: `docs/adr/` (under mkdocs nav) vs. top-level `adrs/`.
   Recommend `docs/adr/` so they ship with the public docs site.
2. Add `docs/adr/template.md`.
3. **Backfill ADRs in strict date order across all tiers.** The first
   pass assigns sequential numbers based on when each decision was
   made (oldest = ADR-0001), so the early IDs read like a history of
   the project. After the backfill is in, new ADRs simply take the
   next number. `tests/conftest.py` will be updated to reference
   ADR-0010 (the async-logging decision) once written.
4. **Search all prior Claude Code chat transcripts** for decisions that
   should be backfilled as ADRs. Transcripts are JSONL files at
   `~/.claude/projects/-Users-joelgotsch-Desktop-joel-backup-ragpill/*.jsonl`
   (currently 39 sessions). Triage approach:
   - Grep for decision-signaling phrases: "we decided", "going with",
     "decided to", "instead of", "rejected", "deviation", "trade-off",
     "compromise", "fall back to", "why not".
   - For each hit, capture: the decision (one sentence), the
     alternatives, the rationale, and a suggested impact tier.
   - Output a backlog table at the bottom of this plan.
   - Do NOT write 39 ADRs in one batch — prioritize Large and Medium
     impact, queue Small ones.
5. Backfill from existing docs that already contain decisions in
   free-form prose:
   - `plans/llm-readable-outputs-and-mcp.md` — "Decisions resolved during
     planning" and "Deviations from the original plan" sections each
     contain ~5 decisions.
   - `designs/llm-readable-outputs-and-mcp.md` — design constraints in
     §1, §7.6.
   - `designs/concurrent-scheduling.md`, `designs/langfuse-integration.md`,
     `designs/evaluator-logical-operations.md`, `designs/otel-trace-ingestion.md`
     — extract decisions that are already settled (vs. open design
     questions).
6. Add a short note to `CONTRIBUTING.md` (or create one) describing when
   to write an ADR and how to reference it from code.
7. Add ADR index page (`docs/adr/index.md`) that lists ADRs by number
   with status + impact badges. Auto-generate or maintain manually —
   start manual, automate if it becomes painful.

## Open questions

- Do we need a "Superseded" state, or just point-in-time records? ADR
  community convention is to keep superseded ADRs and link forward.
  Adopt that.
- This plan itself is not an ADR. It is the plan to introduce ADRs;
  the decisions it captures live in the individual ADR files it
  produces.

## Initial backlog (to be expanded from chat search)

Ordered strictly by decision date (oldest first). Numbers assigned in
that order, regardless of impact. Impact lives in the filename suffix
and the `**Impact:**` field, not the number.

| Number | Date | Impact | Title | Source / timeline anchor |
|---|---|---|---|---|
| ADR-0001 | ~project inception | Large | MLflow as the canonical tracing/eval backend (Langfuse added later as alternative, not replacement) | Reaffirmed in `designs/langfuse-integration.md` §1 |
| ADR-0002 | 2026-04-23 | Large | Three-layer architecture: execute / evaluate / upload (replaced monolithic `evaluate_testset`) | Commit `c66ca6b`, PR #6 |
| ADR-0003 | 2026-04-24 | Medium | Phase B (MCP server): directory-only, no live MLflow lookup | `designs/llm-readable-outputs-and-mcp.md` (design dated 2026-04-24) |
| ADR-0004 | 2026-04-27 | Medium | LLM-judge trace suppression at OTel exporter layer, not via `mlflow.tracing.disable()` (global singleton + concurrency unsafe) | `plans/suppress-llm-judge-traces.md` |
| ADR-0005 | 2026-05-05 | Medium | Pluggable OTel trace ingestion vs. hard-coded MLflow exporter | `designs/otel-trace-ingestion.md`, commit `ac06809` |
| ADR-0006 | 2026-05-05 | Medium | `EvaluationOutput.to_json` uses `pd.to_json(orient="table")` not `orient="split"` to preserve dtypes | `plans/llm-readable-outputs-and-mcp.md` "Deviations" (Phase A) |
| ADR-0007 | 2026-05-05 | Small | `RunResult.error` is coerced to `RuntimeError` on JSON roundtrip (exceptions don't survive JSON) | `plans/llm-readable-outputs-and-mcp.md` (Phase A) |
| ADR-0008 | 2026-05-05 | Small | Trace subtree filter keeps `RETRIEVER, TOOL, LLM, RERANKER, CHAT_MODEL, AGENT` (expanded from design's four) | `plans/llm-readable-outputs-and-mcp.md` (Phase A) |
| ADR-0009 | 2026-05-05 | Small | `model_params` not surfaced in triage header (lives on upload layer, not stored on `EvaluationOutput`) | `plans/llm-readable-outputs-and-mcp.md` (Phase A) |
| ADR-0010 | 2026-05-13 | Small | Disable async MLflow trace logging in unit tests; cover async path in `mlflow-integration` job only | This chat, `tests/conftest.py` |

The chat transcript search (Task 4) will likely 2–4x this list. For the
**initial backfill only**, you can interleave newly discovered entries
into this date-ordered table before any ADR has been written. Once an
ADR file exists on disk, its number is fixed and later additions just
take the next sequential ID (their date may sit anywhere in the
timeline; that's fine — the `**Date:**` field is the authoritative
record).

## Success criteria

- ADR-0001 through ADR-0004 written (the two Large foundational decisions
  and the first two Medium ones); this plan moves to "Accepted".
- ADR-0010 written and `tests/conftest.py` updated to reference it.
- Next non-trivial change (e.g. Phase B MCP server, Langfuse integration)
  ships with at least one ADR.
- A new contributor can read a `# See ADR-NNNN` comment in code and find
  the rationale in under 30 seconds.
