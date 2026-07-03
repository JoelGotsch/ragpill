# ADR-0017: `get_trace` / `await_trace` return the neutral trace, not native

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Medium
**Related:** ADR-0011, ADR-0013, ADR-0014, plans/multi-backend-tracking.md, plans/phoenix-backend-findings.md

## Context
After the trace-model flip (ADR-0014), the execution layer still converted
captured traces MLflow-specifically: `execution._fetch_trace` did
`from_mlflow_trace(get_backend().await_trace(...))`, and the session-mode loop
likewise. So `TraceQueryBackend.await_trace` / `get_trace` returned each
backend's *native* trace and the executor hard-coded the MLflow converter. A
second backend (Phoenix, see plans/phoenix-backend-findings.md) returning
native Phoenix spans would be fed to `from_mlflow_trace` and break — this
blocked any non-MLflow backend.

## Decision
`get_trace` and `await_trace` return a vendor-neutral `ragpill.trace.Trace`.
Each backend converts its own native trace internally (`MLflowBackend.get_trace`
calls `from_mlflow_trace`; a Phoenix backend would build span dicts and run them
through the OpenInference adapter / `parse_otel`). The execution layer drops its
`from_mlflow_trace` call and consumes the neutral trace directly.

`search_traces` (and `delete_traces`) stay **native** — they back
backend-internal work like judge-trace cleanup that introspects native span
attributes (`upload._delete_llm_judge_traces` reads `trace.data._get_root_span()`).

## Alternatives considered
- **Keep native returns; add a shim in the Phoenix adapter** so its `await_trace`
  output is `from_mlflow_trace`-compatible. Rejected: an awkward shim that defers
  the real fix and keeps the executor MLflow-coupled.
- **A separate `backend.to_neutral_trace(native)` method.** Rejected as redundant
  — the conversion has exactly one caller per backend (its own read methods), so
  folding it into `get_trace`/`await_trace` is simpler.

## Consequences
- The backend protocol is genuinely backend-agnostic for the read path; adding a
  backend no longer requires touching `execution.py`.
- One conversion point per backend (`MLflowBackend.get_trace`), polled by
  `await_trace`.
- A small split in the protocol: `get_trace`/`await_trace` are neutral while
  `search_traces`/`delete_traces` are native. Documented on `_base.py`.
- Internal-only change (all within 0.5.0); no user-visible API shift, since the
  execution layer already exposed neutral traces on `CaseRunOutput`/`TaskRunOutput`.

## References
- `ragpill.backends._base.TraceQueryBackend`, `ragpill.backends.mlflow_backend.MLflowBackend.get_trace`
- `ragpill.execution._fetch_trace` (session + span modes)
- ADR-0014 (neutral trace model), ADR-0013 (adapter interface)
- Chat session 2026-06-19
