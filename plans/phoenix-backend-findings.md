# Findings: Phoenix backend (multi-backend Phase 2.C)

**Status:** Research done; blocked on two decisions (below)
**Date:** 2026-06-19
**Related:** `plans/multi-backend-tracking.md` Phase 2.C, `designs/otel-trace-ingestion.md`

Researched by installing the real SDKs into the venv, introspecting, then
restoring the env (`uv sync`). Suite still green (540).

## Verified API (current versions)

- `arize-phoenix-otel` **0.16.1**, `arize-phoenix-client` **2.9.0**,
  `openinference-instrumentation-pydantic-ai` **0.1.16**.
- **Tracing setup:** `from phoenix.otel import register` —
  `register(*, endpoint=None, project_name=None, batch=False,
  set_global_tracer_provider=True, headers=None, protocol=None,
  auto_instrument=False, api_key=None, **kwargs) -> TracerProvider`. Reads
  `PHOENIX_COLLECTOR_ENDPOINT` / `PHOENIX_PROJECT_NAME` / `PHOENIX_API_KEY` /
  `PHOENIX_CLIENT_HEADERS` from env when args omitted. `auto_instrument=True`
  instruments all installed OpenInference libraries.
- **pydantic-ai capture:** `openinference.instrumentation.pydantic_ai.OpenInferenceSpanProcessor`
  added to the tracer provider; pydantic-ai must emit OTel (native
  `Agent(instrument=True)` / global instrumentation). Differs from MLflow's
  monkeypatch `autolog()`.
- **Query:** `phoenix.client.Client(base_url=, api_key=, headers=)`;
  `client.spans.get_spans_dataframe(*, query=None, limit=1000,
  project_identifier=None, root_spans_only=None, ...) -> pd.DataFrame`.
- **Assessments:** `client.spans.add_span_annotation(*, span_id, annotation_name,
  annotator_kind="HUMAN"|"LLM", label=None, score=None, explanation=None,
  metadata=None)` and bulk `client.spans.log_span_annotations(span_annotations=[SpanAnnotationData(...)])`.
- **No native concepts:** runs, metrics, params, tables, trace deletion. Phoenix
  "project" ≈ MLflow "experiment". Deletion is unsupported via SDK.

## Blocker 1 — OpenTelemetry version conflict

`phoenix.otel` pulls `opentelemetry-*==1.42` (otlp-grpc exporter), which is
incompatible with the `opentelemetry-sdk` the existing stack pins (via
mlflow-skinny): importing `phoenix.otel` raises
`ImportError: cannot import name '_OTEL_PYTHON_EXPORTER_OTLP_GRPC_RETRYABLE_ERROR_CODES'`.
The `ragpill[phoenix]` extra is not cleanly installable alongside `ragpill[mlflow]`
as-is. Options: pin a compatible otel range; use http/protobuf only and avoid the
grpc exporter; or document that phoenix and mlflow extras are mutually exclusive
in one env. **Does not block mocked unit tests** (the adapter uses lazy imports
that tests patch), only live integration.

## Blocker 2 — the trace-return contract (architectural)

The execution layer converts captured traces MLflow-specifically:
`execution._fetch_trace` does `from_mlflow_trace(get_backend().await_trace(...))`,
and the session-mode loop calls `from_mlflow_trace(backend.await_trace(...))`. So
`await_trace` / `get_trace` / `search_traces` return **backend-native** traces and
the execution layer hard-codes the MLflow converter. A Phoenix backend returning
Phoenix-native spans would be fed to `from_mlflow_trace` and break.

**Proposed fix (clean, but touches the working MLflow path):** make
`await_trace` / `get_trace` return a vendor-neutral `ragpill.trace.Trace`
directly — each backend converts its own native trace internally
(`MLflowBackend` calls `from_mlflow_trace`; `PhoenixBackend` builds span dicts
and runs them through `OpenInferenceAdapter` / `parse_otel`). The execution layer
then drops its `from_mlflow_trace` call and just uses the returned neutral trace.
This is the natural endpoint of the neutral-trace effort and makes the protocol
genuinely backend-agnostic, but it is an ADR-worthy change that edits
`backends/_base.py`, `mlflow_backend.py`, and `execution.py` (the live path).

## Mapping sketch (once unblocked)

| Backend method | Phoenix |
|---|---|
| set_destination | `register(endpoint, project_name)`; store tracer provider |
| autolog_pydantic_ai | add `OpenInferenceSpanProcessor` + enable pydantic-ai OTel |
| start_run / end_run | no run concept; synthesize handle (run_id=project), `force_flush` on end |
| start_span | OTel span via tracer; set `openinference.span.kind`, `input.value`/`output.value` |
| start_case_grouping | span mode (parent span) initially; sessions via `session.id` later |
| await_trace / get_trace | `get_spans_dataframe(project_identifier=...)` → openinference adapter → neutral Trace |
| search_traces | same, grouped by trace_id |
| delete_traces | unsupported → exporter-filter for judge traces (separate plan) |
| log_assessment | `add_span_annotation(span_id, annotation_name, score/label, explanation)` |
| set_trace_tag | root-span attribute |
| log_metric / log_params / log_table / log_artifact | no native concept → no-op or local artifact, per multi-backend plan risk table |
| resolve_experiment_id | project name/id |

## Recommended next step

Resolve Blocker 2 first (it's a prerequisite for *any* second backend and is the
right design regardless of Phoenix), as its own small refactor + ADR; then build
`PhoenixBackend` with lazy imports + mocked unit tests + an env-gated integration
test, and pin the extra to resolve Blocker 1.
