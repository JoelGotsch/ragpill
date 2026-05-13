# Plan: Suppress LLM-judge traces via OpenTelemetry exporter filter

**Status:** Proposed — not started. `_delete_llm_judge_traces` ([upload.py:56](../src/ragpill/upload.py#L56)) is still the active mechanism.
**Date:** 2026-04-27

## Problem statement

`LLMJudge.run()` creates an MLflow span (plus child spans from pydantic-ai + openai autolog) that currently gets exported to MLflow and then post-hoc deleted via `_delete_llm_judge_traces`. That round-trip is wasteful; separately, judge traces surface as independent root traces in the MLflow UI rather than nested under the task trace. We want to keep creating those spans in-process (so the autolog integrations don't race to insert a root trace and hit the SQLite UNIQUE constraint), but filter them at the OpenTelemetry exporter layer so they never hit the tracking server.

`mlflow.tracing.disable()` is ruled out: it's a global singleton and would be unsafe during concurrent async evaluation, and would also suppress task traces if evaluation overlaps execution in any path.

## Architecture — how the filter plugs into MLflow's OTel pipeline

Pinned MLflow: **3.11.1** (`pyproject.toml` line 15: `mlflow>=3.8.1`).

Relevant classes under `.venv/lib/python3.14/site-packages/mlflow/tracing/`:

- `provider.py` — `provider: _TracerProviderWrapper` is the module-level singleton; `provider.get()` returns the active `opentelemetry.sdk.trace.TracerProvider`.
- `processor/mlflow_v3.py::MlflowV3SpanProcessor(BaseMlflowSpanProcessor)` — a `SimpleSpanProcessor` subclass attached to the provider when the tracking URI is an MLflow backend (our case).
- `export/mlflow_v3.py::MlflowV3SpanExporter(SpanExporter)` — the exporter we intercept. Key method: `export(self, spans: Sequence[ReadableSpan]) -> None`.
- Each processor stores its exporter as `processor.span_exporter`.

**Chosen injection point: wrap `MlflowV3SpanExporter` in a delegating `FilteringSpanExporter`.** Rationale: the processor populates MLflow's internal `InMemoryTraceManager` in `on_start`/`on_end` — filtering at the processor level would leave half-registered state. Filtering at the exporter boundary keeps all in-process bookkeeping intact (autolog behaves normally, no UNIQUE-constraint race) but suppresses the network calls.

## File-level changes

### 1. New module `src/ragpill/_tracing_filter.py`

```python
"""OpenTelemetry exporter filter that drops LLM-judge spans before upload."""
from __future__ import annotations

import threading
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

_JUDGE_ATTR = "ragpill_is_judge_trace"
_JUDGE_TRACE_IDS: set[int] = set()     # OTel trace_ids marked as judge-owned
_LOCK = threading.Lock()


def register_judge_trace_id(otel_trace_id: int) -> None:
    with _LOCK:
        _JUDGE_TRACE_IDS.add(otel_trace_id)


def _is_judge_span(span: ReadableSpan) -> bool:
    if span.attributes and span.attributes.get(_JUDGE_ATTR) is True:
        return True
    with _LOCK:
        return span.context.trace_id in _JUDGE_TRACE_IDS


class FilteringSpanExporter(SpanExporter):
    """Delegating exporter that drops judge spans before export."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = [s for s in spans if not _is_judge_span(s)]
        if not kept:
            return SpanExportResult.SUCCESS
        return self._inner.export(kept)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def install_judge_filter() -> bool:
    """Wrap the currently-installed MLflow exporter. Idempotent."""
    from mlflow.tracing.provider import provider
    tp = provider.get()
    if tp is None:
        return False
    processors = getattr(tp._active_span_processor, "_span_processors", ())
    installed = False
    for p in processors:
        exp = getattr(p, "span_exporter", None)
        if exp is None or isinstance(exp, FilteringSpanExporter):
            continue
        p.span_exporter = FilteringSpanExporter(exp)
        installed = True
    return installed
```

Notes:
- Key is OTel integer `span.context.trace_id` — every descendant of the judge root inherits it without touching contextvars.
- `threading.Lock` is required because `MlflowV3SpanExporter` can dispatch to `AsyncTraceExportQueue` worker threads.
- Trace-ids are never removed from the set — spans can export after the judge context manager exits; the set lives for the process lifetime (add `clear_all()` for tests).

### 2. Edits to `src/ragpill/evaluators.py` (L27–L41, L114)

Add field and register the trace-id *before* any pydantic-ai call so autolog child spans inherit:

```python
@dataclass(kw_only=True, repr=False)
class LLMJudge(BaseEvaluator):
    rubric: str
    model: models.Model = field(repr=False, default_factory=_get_default_judge_llm)
    include_input: bool = field(default=False)
    upload_traces: bool = field(default=False)   # NEW

    async def run(self, ctx):
        with mlflow.start_span(name="llm-judge-evaluation", span_type=SpanType.LLM) as span:
            span.set_attribute("ragpill_is_judge_trace", True)
            if not self.upload_traces:
                from ragpill._tracing_filter import register_judge_trace_id
                register_judge_trace_id(span._span.context.trace_id)
            ...
```

Verify the OTel id attribute chain at implementation time — `mlflow.entities.Span` wraps an OTel `Span` but exposes string ids; the integer lives at `span._span.context.trace_id` (private). If the private path shifts between MLflow versions, fall back to parsing `span.trace_id` (hex string) to int.

### 3. Install the filter at MLflow setup sites

In `src/ragpill/execution.py`, at the end of `_setup_local_tracing` (~L286) and `_setup_server_tracing` (~L304), after `mlflow.start_run()`:

```python
from ragpill._tracing_filter import install_judge_filter
install_judge_filter()
```

The tracer provider is lazily initialized on first span creation, so by post–`start_run()` the processor exists. Idempotent — safe to call again in `evaluate_results` (`src/ragpill/evaluation.py` ~L343) for offline paths that skip execute.

### 4. Edits to `src/ragpill/upload.py`

Gate `_delete_llm_judge_traces` behind a `fallback_delete_judge_traces: bool = False` kwarg on `upload_to_mlflow`. Default OFF once the filter is verified; keep as belt-and-suspenders for one release, then remove. Current call at line 227 becomes `if fallback_delete_judge_traces: _delete_llm_judge_traces(...)`.

## Investigation finding: broken parent-child relationship

**Diagnosis (not a bug in `mlflow.start_span`):**

In the split architecture, `execute_dataset` creates the task span, closes it, and returns. The task trace is already exported to the backend by the time `evaluate_results` runs. `LLMJudge.run()` calls `mlflow.start_span(...)` with no active parent in the OTel context (the task span closed when `_execute_case_runs` exited its `with` block around `execution.py:381`), so MLflow creates a **new root** — exactly what the user observes.

Even if the task trace object (`ctx.trace`) is carried into the evaluator context, it's a read-only `mlflow.entities.Trace`, not a live OTel span. You cannot re-open a closed span or append to an exported trace.

**Asyncio is not the cause** — pydantic-ai's autolog uses standard OTel `contextvars` and `mlflow.start_span` receives context correctly; there's simply no parent to inherit from in the evaluation phase.

**Recommended fix (separate plan, not bundled here):** either (a) run evaluators inline *inside* the task span's `with` block (blurs the execute/evaluate split that was the point of the refactor), or (b) use `provider.start_detached_span(name, parent=...)` (see `provider.py:252`) and plumb an OTel span handle into `EvaluatorContext`. Option (b) is least invasive but requires keeping the underlying tracer provider/span alive across phases.

**For this plan:** once judges are filtered out entirely, the "separate root trace" UX problem disappears as a side effect — they're just gone from the UI. Parent-child fix is moot for the upload-suppression goal.

## Async / concurrency safety

- `_JUDGE_TRACE_IDS: set[int]` guarded by `threading.Lock`; O(1) add/contains.
- Concurrent judges in one event loop: OTel `IdGenerator` guarantees unique trace_ids.
- Cross-thread: `MlflowV3SpanExporter._should_log_async()` can enqueue exports onto `AsyncTraceExportQueue` worker threads — the lock covers this.
- MLflow's `InMemoryTraceManager` is untouched; the wrapped exporter lets the processor do its usual bookkeeping and only drops the network call.
- `install_judge_filter()` checks `isinstance(exp, FilteringSpanExporter)` so repeat calls are no-ops.

## Test plan

Unit tests (new `tests/test_tracing_filter.py`):

1. `test_filter_drops_judge_root` — mock inner exporter; feed a `ReadableSpan` with `ragpill_is_judge_trace=True`; assert inner not called.
2. `test_filter_drops_descendant_by_trace_id` — register a fake trace_id, feed a child without the attribute; assert dropped.
3. `test_filter_passes_unrelated_spans` — different trace_id → delegated.
4. `test_register_thread_safety` — 20 threads adding concurrently; no exceptions.
5. `test_install_judge_filter_idempotent` — called twice, wrap only once.

Integration tests:

6. `test_llmjudge_does_not_upload_traces` — local sqlite backend (same pattern as `_setup_local_tracing`), run dataset with `LLMJudge(upload_traces=False)`; assert `mlflow.search_traces(...)` returns zero traces carrying the judge attribute, task trace still present.
7. `test_llmjudge_upload_traces_true_keeps_trace` — same with `upload_traces=True`; regression lock for the opt-in path.
8. `test_concurrent_judges_isolated` — two judges via `asyncio.gather`; both complete, no UNIQUE-constraint crash, neither leaks into the task trace.
9. `test_fallback_delete_path_still_works` — call `upload_to_mlflow(..., fallback_delete_judge_traces=True)`; verify old behavior.

## Open questions / risks

1. **Private API reliance.** `tp._active_span_processor._span_processors` and `span._span.context.trace_id` are not public MLflow APIs. `pyproject.toml` pins `mlflow>=3.8.1` — a wide range. Mitigation: wrap `install_judge_filter()` in `try/except AttributeError`, log a warning, fall back to post-hoc delete.
2. **Eval-harness branch.** `MlflowV3SpanExporter._should_log_async` checks `maybe_get_request_id(is_evaluate=True)`. We don't use `mlflow.evaluate`, but verify if that codepath appears later.
3. **pydantic-ai autolog ordering.** If a child span reaches the exporter before the judge root registers its trace_id, descendants won't match. Mitigation: register trace_id *before* `await judge_output(...)` (already in the sketch above).
4. **Non-MlflowV3 processors.** Databricks UC and InferenceTable processors also expose `span_exporter` — same wrapper works but hasn't been tested.
5. **Parent-child fix is explicitly out of scope** but the user should know the UX change from this plan (judges vanish from UI) may be what they actually wanted.

## Critical files

- `src/ragpill/evaluators.py` (edit)
- `src/ragpill/execution.py` (edit — install site)
- `src/ragpill/evaluation.py` (edit — install site for offline path)
- `src/ragpill/upload.py` (edit — gate fallback)
- `src/ragpill/_tracing_filter.py` (new)
- `.venv/lib/python3.14/site-packages/mlflow/tracing/provider.py` (reference — private API surface)
- `.venv/lib/python3.14/site-packages/mlflow/tracing/export/mlflow_v3.py` (reference — the exporter being wrapped)
