# Plan: Fix silently-broken trace-based evaluators (fetch race + failure visibility)

**Status:** In progress
**Date:** 2026-06-19
**Branch:** `feature/multi-backend-step1-protocols` (this work lands here — it is **not** orthogonal)
**Related:**
- `plans/multi-backend-tracking.md` — the refactor that moved trace fetching behind `get_backend()`.
- `plans/sessions-for-case-grouping.md` — introduced session-mode fetching (`backend.get_trace`), which shares the same race.
- Upstream bug report against `llm_eval` 0.4.5 (pre-refactor naming of this package).

## Why this is on-branch, not orthogonal

The bug report's proposed fix patches `_fetch_trace` to call `mlflow.search_traces(...)`
directly. The multi-backend refactor (step 3) already rewrote that function to route through
`get_backend().search_traces(...)`. Applying the report's patch verbatim on a separate branch would
**revert** the abstraction and conflict with this branch.

Two further couplings make this a multi-backend concern:

1. **The async-export race is per-backend.** Flush/export timing differs across MLflow / Langfuse /
   Phoenix. "Wait until the trace is queryable" is a backend capability, so the retry/readiness
   logic belongs in the backend protocol — not hard-coded in the shared execution layer.
2. **Session mode (commit `038da30`) has the same race, uncovered by the report.** It fetches each
   repeat via `backend.get_trace(tr.trace_id)` immediately after the grouping context exits. The
   report predates session mode and only addresses span mode.

## Decisions (confirmed with maintainer)

1. **Retry/poll lives in a backend protocol method**, not a generic loop in `execution.py`.
2. **Fix both paths** — span-mode `_fetch_trace` *and* session-mode `get_trace`.
3. **Visibility only for `all_passed`** — surface `evaluator_failures` in the triage report but do
   **not** change `RunResult.all_passed` pass/fail semantics. (Report flagged this as a deliberate
   non-change because it shifts reported pass rates and could move existing thresholds.)
4. **Configurable poll budget, ~10s default**, exposed via settings.

## Bugs mapped to current code (ragpill 0.5.0)

| Bug | Current location | State |
|---|---|---|
| 1 — async export race | `execution.py:395` `_fetch_trace` — single `get_backend().search_traces()`, no retry | present |
| 2 — wrong-trace fallback | `execution.py:402` `return traces[0] if len(traces) == 1 else None` | present, verbatim |
| 3a — rollup hides failures | `report/triage.py:233` `_per_evaluator_rollup` iterates only `rr.assertions` | present |
| 3b — failing-run hides failures | `report/triage.py:297` `_render_failing_run` iterates only `rr.assertions` | present |
| 3c — `all_passed` ignores failures | `types.py:52` (NOT changed per decision 3) | left as-is by choice |

## Design

### 1. New readiness primitive on `TraceQueryBackend`

Add to `src/ragpill/backends/_base.py`:

```python
def await_trace(
    self,
    trace_id: str,
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.5,
) -> Trace | None:
    """Fetch the trace `trace_id`, polling until it is fully exported.

    Returns the trace once available, or `None` on timeout. MUST NOT fall back
    to a different trace — a miss returns `None`, never the wrong trace.
    """
    ...
```

MLflow adapter (`mlflow_backend.py`) implements it by polling its own by-id `get_trace(trace_id)`
(the authoritative readiness check — returns the trace with its full span tree only once exported).
This unifies both call paths on a single fetch-by-id and **drops the `search_traces`-plus-fallback
logic entirely**, killing Bug 2 at the root.

### 2. `execution.py`

- `_fetch_trace(...)` → thin wrapper that calls `get_backend().await_trace(parent_trace_id, run_id=..., experiment_id=..., timeout_s=..., poll_interval_s=...)`. The `case_trace_id` opened in span mode *is* a trace id, so by-id polling is correct. Remove the `traces[0] if len == 1` fallback.
- Session-mode loop (`execution.py:466`) → replace `backend.get_trace(tr.trace_id)` with `backend.await_trace(tr.trace_id, run_id=..., experiment_id=..., timeout_s=..., poll_interval_s=...)`.
- Thread the configured timeout/interval through `_TracingContext` (new fields), populated in `_setup_local_tracing` / `_setup_server_tracing` from settings.

### 3. `settings.py`

Add to `MLFlowSettings`:

```python
ragpill_trace_fetch_timeout_s: float = Field(10.0, ge=0.0, description="...")
ragpill_trace_fetch_poll_interval_s: float = Field(0.5, gt=0.0, description="...")
```

Env: `MLFLOW_RAGPILL_TRACE_FETCH_TIMEOUT_S`, `MLFLOW_RAGPILL_TRACE_FETCH_POLL_INTERVAL_S`.

### 4. `report/triage.py` (visibility)

- `_per_evaluator_rollup`: after the assertions loop, count each `ef` in `rr.evaluator_failures` as
  `(0 passes, 1 attempt)` so errored evaluators appear in the header rollup.
- `_render_failing_run`: after the assertions loop, append a line per `ef`:
  `` - `{ef.name}`: **ERROR** — {render_value(ef.error_message)}``. Also include
  `evaluator_failures` count in the run header so a run with only errors (no failing assertions)
  still renders as a problem.
- `_render_failing_case` / `failing_runs` selection: ensure a run with `evaluator_failures` but all
  assertions passing is still surfaced. `all_passed` stays unchanged (decision 3), so the
  triage-side filter must independently treat `evaluator_failures` as "show this run".

### 5. `types.py`

No change to `all_passed` (decision 3). A short comment notes that `evaluator_failures` is
intentionally surfaced in triage but not folded into pass/fail.

## Test plan

- **Unit — readiness poll:** `await_trace` returns the trace once a fake backend's `get_trace`
  starts returning non-None after N polls; returns `None` on timeout; never returns a different
  trace when the id is absent (Bug 2 regression).
- **Unit — `_fetch_trace`:** patches `get_backend()` to a mock; asserts it calls `await_trace` with
  the configured timeout/interval and no longer has any single-trace fallback path.
- **Unit — session mode:** asserts the session-mode loop calls `await_trace(tr.trace_id, ...)`.
- **Unit — triage rollup:** a `RunResult` with an `EvaluatorFailureInfo` and empty assertions makes
  the evaluator appear in `_per_evaluator_rollup` (0/1) and renders an `ERROR` line.
- **Settings:** env vars parse into the two new fields with documented defaults.
- **No-regression:** full `[mlflow]` suite stays green.

## Files touched

| File | Change |
|---|---|
| `src/ragpill/backends/_base.py` | add `await_trace` to `TraceQueryBackend` |
| `src/ragpill/backends/mlflow_backend.py` | implement `await_trace` via polling `get_trace` |
| `src/ragpill/execution.py` | `_fetch_trace` → `await_trace`; session loop → `await_trace`; remove fallback; thread timeout via `_TracingContext` |
| `src/ragpill/settings.py` | two new `MLFlowSettings` fields |
| `src/ragpill/report/triage.py` | surface `evaluator_failures` in rollup + failing-run + case selection |
| `src/ragpill/types.py` | comment only (no semantic change) |
| `tests/...` | new unit tests per the test plan |

## Out of scope

- Changing `all_passed` semantics (deferred by decision; revisit if maintainer later wants errored
  evaluators to fail a run).
- Per-backend readiness for Langfuse/Phoenix — those adapters implement `await_trace` when they
  land in Phase 2; this plan only ships the protocol method + MLflow implementation.
