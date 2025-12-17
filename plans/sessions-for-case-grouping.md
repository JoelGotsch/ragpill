# Plan: Sessions for case-level grouping (Issue: repeats not visible as turns)

**Status:** Implemented
**Date:** 2026-05-21

## Context

User-reported bug: with `repeat > 1` on a case, the MLflow Sessions UI
showed one session for the *whole evaluation invocation* and each case as
a single turn. Repeats of the same case were buried as sibling spans
inside one trace, not as turns of the same session.

Root cause: ragpill opened one parent span per case and let repeats nest
as children. No `mlflow.trace.session` metadata was ever set, so the
Sessions UI fell back to per-run grouping.

This collides head-on with the multi-backend abstraction (`feature/
multi-backend-step1-protocols` at 0.5.0). The MLflow-specific call
`mlflow.update_current_trace(metadata={"mlflow.trace.session": …})` is
exactly the kind of vendor leak we're trying to push behind the
adapter boundary. Langfuse uses `session_id`; Phoenix uses
OpenInference's `session.id`. Each backend has a session concept; the
spelling differs.

## Resolution

Implemented option **B + E** with adapter-side fallback:

- New `Backend.start_case_grouping(case_id, name, inputs, attributes)`
  on `TraceCaptureBackend`. Returns a `CaseGroupingHandle` whose
  ``.mode`` is either ``"session"`` or ``"span"``.
- MLflow adapter: **session mode**. Records ``case_id`` as the active
  session id; each child trace opened via ``start_span`` inside the
  context gets the ``mlflow.trace.session`` metadata via
  ``mlflow.update_current_trace``. No parent span is opened — each
  repeat is its own top-level trace, which is what the Sessions UI
  needs to show repeats as turns.
- Fallback `"span"` mode (used by any future adapter without a
  sessions concept) opens a parent span and lets repeats nest as
  children — the pre-0.5 behaviour. The execution layer detects this
  via ``handle.mode`` and runs the legacy fetch-then-filter path.

## Execution-layer changes

- `_execute_case_runs` calls `backend.start_case_grouping(...)` instead
  of opening a parent span directly. Branches on `handle.mode`:
  - session mode: per-repeat traces fetched individually via
    `backend.get_trace(task_run.trace_id)`.
  - span mode: case-level trace fetched once and filtered per repeat
    (existing logic).
- `_execute_single_run` now captures `trace_id` (mlflow `request_id`)
  alongside `run_span_id` so session-mode post-loop can resolve each
  repeat's own trace.
- `TaskRunOutput.trace_id` added (JSON schema additive; default `""`).

## API impact

- `CaseRunOutput.trace` is **None** in session mode (no case-level
  trace exists). Each `TaskRunOutput.trace` carries the per-repeat
  trace.
- `CaseRunOutput.trace_id` is **""** in session mode.
- JSON round-trip continues to work; new `trace_id` field on the
  per-run dict is additive.
- Existing custom `SpanBaseEvaluator` subclasses that read `ctx.trace`
  are unaffected — `ctx.trace` already prefers `task_run.trace`.

## Tests

- New `tests/test_backends.py` cases:
  - `start_case_grouping` yields a session-mode handle with the
    requested `case_id`.
  - `start_span` inside the grouping calls
    `mlflow.update_current_trace(metadata={"mlflow.trace.session": …})`.
  - `start_span` outside the grouping does not touch session metadata.
- New `tests/test_execution_integration.py` case
  `test_repeats_share_mlflow_session_id`: 3-repeat run against a
  temp SQLite "server"; asserts each repeat is a distinct top-level
  trace and all three carry the same `mlflow.trace.session` metadata
  equal to the case `base_input_key`.
- Two existing integration tests pivoted from `case_run.trace` to
  `task_run.trace` because session mode removes the case-level trace.

## Out of scope

- Phoenix and Langfuse adapter implementations of session mode (Phase
  2 of `plans/multi-backend-tracking.md`). Their session mappings:
  - Langfuse: `langfuse.update_current_trace(session_id=…)`.
  - Phoenix: set OpenInference `session.id` attribute on the root span.
- Per-turn metadata richer than `session_id` (e.g. turn-level inputs,
  turn-level user-id). Easy to add later via the same `CaseGroupingHandle`
  surface.
