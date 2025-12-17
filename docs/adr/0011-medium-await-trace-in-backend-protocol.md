# ADR-0011: Trace-readiness polling lives in the backend protocol (`await_trace`)

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Medium
**Related:** designs/otel-trace-ingestion.md, designs/langfuse-integration.md, ADR-0001

## Context
`SpanBaseEvaluator` subclasses were silently failing for roughly 16 of 17
cases. `execution._fetch_trace` fetched a trace immediately after the case
span context closed, racing MLflow's asynchronous span export, so the search
returned nothing. Worse, a `traces[0] if len == 1 else None` fallback meant
that when exactly one stale trace happened to be present, the evaluator was
handed the WRONG trace — the previous case's — rather than no trace.

The multi-backend refactor had already routed trace fetching through
`get_backend().search_traces()`. A point fix that reverted to raw
`mlflow.search_traces` would have regressed that abstraction, and the
flush/export timing that causes the race is per-backend (Langfuse and Phoenix
flush differently from MLflow), so a generic retry in `execution.py` would
have to know backend-specific readiness semantics it has no business knowing.

## Decision
Add an `await_trace(trace_id, *, run_id, experiment_id, timeout_s,
poll_interval_s)` method to the `TraceQueryBackend` protocol; it polls the
backend by trace id until the trace is exported, returns `None` on timeout,
and MUST NOT fall back to a different trace.

The MLflow adapter implements `await_trace` by polling its by-id `get_trace`.
Both the span-mode `_fetch_trace` and the session-mode `get_trace` loop route
through it. The previous single-trace (`traces[0]`) fallback is removed
entirely.

The poll budget is configurable rather than hard-coded: `TrackingSettings`
exposes `ragpill_trace_fetch_timeout_s` (default `10.0`) and
`ragpill_trace_fetch_poll_interval_s` (default `0.5`), bound to env vars
`RAGPILL_TRACE_FETCH_TIMEOUT_S` and
`RAGPILL_TRACE_FETCH_POLL_INTERVAL_S`, threaded through
`_TracingContext` to the fetch sites.

## Alternatives considered
- **Generic retry loop in `execution.py` calling `search_traces`.** Rejected:
  flush/export timing is per-backend, so trace readiness belongs in the
  backend abstraction. Langfuse and Phoenix flush differently from MLflow; a
  retry in the orchestration layer would leak backend-specific timing
  knowledge upward.
- **Keep the single-trace `traces[0] if len == 1 else None` fallback.**
  Rejected: this was the direct cause of the wrong-trace bug — it could hand
  an evaluator a stale trace belonging to a different case.
- **Hard-coded ~10s poll budget / shorter 5s budget.** Rejected in favour of
  configurable-with-10s-default: 10s is safe for slow remote MLflow servers
  while the env vars allow tuning down for fast local setups.

## Consequences
- A silently-dropped evaluator from the export race no longer happens; either
  the trace is found within the budget or `await_trace` returns `None`
  explicitly.
- Per-case latency can rise to the full poll budget (default 10s) when a trace
  never arrives.
- Every future `TraceQueryBackend` implementation must provide its own
  `await_trace` honouring its own flush/export semantics.
- Operators on slow remote servers can raise the timeout; fast local runs can
  lower it, both without code changes.

## References
- `TraceQueryBackend` protocol; MLflow adapter `await_trace`
- `execution._fetch_trace` (span mode) and session-mode `get_trace` loop
- `TrackingSettings.ragpill_trace_fetch_timeout_s` /
  `ragpill_trace_fetch_poll_interval_s`; `_TracingContext`
- Chat session 2026-06-19
