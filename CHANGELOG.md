# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). This project is
pre-1.0, so minor versions may carry breaking changes.

## [0.5.0] - 2026-06-19

The 0.4.x → 0.5.0 release: MLflow becomes an optional backend behind a small
set of protocols, and the captured trace is decoupled from MLflow into a
vendor-neutral model.

### Changed

- **MLflow is now an optional dependency.** `pip install ragpill` no longer
  pulls MLflow; opt in with `pip install ragpill[mlflow]`. Tracking is driven
  through a small backend protocol (`ragpill.backends`) so other backends can
  be added without touching the execute/evaluate/upload layers.
- **Backend read methods return the neutral trace.** `Backend.get_trace` /
  `await_trace` now return a `ragpill.trace.Trace` (each backend converts its
  own native trace internally), so the execution layer is backend-agnostic.
  See ADR-0017.
- **Judge-trace cleanup moved behind the protocol.** `search_traces` (which
  leaked backend-native trace objects into the upload layer) is replaced by
  `Backend.delete_judge_traces(experiment_id, run_id)`; each backend owns how
  its store surfaces the `ragpill_is_judge_trace` marker.
- **Span handles have an explicit contract.** `start_span` yields a
  `backends.SpanHandle` (`span_id`, `trace_id`, `set_attribute/inputs/outputs`);
  the MLflow-flavoured `request_id` alias is gone.
- **`DatasetRunOutput` fields renamed** `mlflow_run_id`/`mlflow_experiment_id`
  → `run_id`/`experiment_id` (dataclass and v2 run-JSON keys) — the ids are
  backend-neutral now.
- **Metric-name sanitisation moved into `MLflowBackend.log_metric`** — the
  upload layer passes raw names; each backend applies its own naming rules.
- **Evaluator-level tags/attributes now reach the runs DataFrame.** The
  documented case↔evaluator metadata union-merge previously silently dropped
  the evaluator side; per-tag accuracy now sees evaluator tags too.
- **`RegexInOutputEvaluator` normalises the pattern on every construction
  path** (previously only `from_csv_line`), matching its documented contract.

### Added (backends)

- **Experimental Arize Phoenix backend** (`ragpill.backends.phoenix_backend.PhoenixBackend`,
  extra `ragpill[phoenix]`): OpenInference-native capture + spans-dataframe
  reads converted via the OpenInference adapter. Select it with
  `configure_backend(PhoenixBackend)`. Note: `ragpill[phoenix]` and
  `ragpill[mlflow]` are not co-installable in one environment (OpenTelemetry
  version conflict). Metrics/params/tables/artifacts/trace-deletion no-op
  (Phoenix has no native equivalent); the live path is covered by an env-gated
  integration test.
- **Experimental Langfuse backend** (`ragpill.backends.langfuse_backend.LangfuseBackend`,
  extra `ragpill[langfuse]`): v4 OTel SDK; assessments map to Langfuse **scores**
  (BOOLEAN/NUMERIC/CATEGORICAL), trace deletion is supported, and observations
  convert to the neutral trace by direct field mapping. Select with
  `configure_backend(LangfuseBackend)`. Metrics/params/tables/artifacts no-op;
  live path covered by an env-gated integration test.

### Breaking

- **Span-based evaluators now receive a vendor-neutral trace.**
  `EvaluatorContext.trace` and `SpanBaseEvaluator.get_trace()` are now
  `ragpill.trace.Trace` (a plain dataclass) instead of `mlflow.entities.Trace`.
  Custom `SpanBaseEvaluator` subclasses that read the MLflow trace/span surface
  directly (e.g. `trace.data.spans`, `trace.search_spans(...)`,
  `span.span_type`) must migrate to `ragpill.trace.Span` (`trace.spans`,
  `span.kind`, `span.inputs/outputs/attributes`). See ADR-0014.
- **Retrieved documents are `ragpill.trace.Document`.** `SourcesBaseEvaluator`
  (and `RegexInSourcesEvaluator`, `RegexInDocumentMetadataEvaluator`,
  `LiteralQuoteEvaluator`) now build `ragpill.trace.Document` with a `content`
  field (was `mlflow.entities.Document` with `page_content`). Custom
  `evaluation_function`s passed to these evaluators must read `doc.content`.
- **Run-JSON schema bumped to v2.** `DatasetRunOutput.to_json()` /
  `EvaluationOutput.to_json()` serialise the neutral trace model and stamp
  `schema_version: 2`. Old (v1) run files store MLflow-shaped traces and are
  **not** migrated — `from_json` raises a clear error pointing here. Re-run the
  evaluation to produce a v2 file. See ADR-0014.

### Added

- `ragpill.trace` — vendor-neutral trace model (`Trace`, `Span`, `Message`,
  `Document`, `Usage`, `SpanKind`) plus:
  - `from_mlflow_trace()` — convert a captured `mlflow.entities.Trace`.
  - `trace_to_dict` / `trace_from_dict` — JSON-safe (de)serialisation.
  - `filter_to_subtree()` — subtree filter over the neutral model.
  - `adapters.SpanAdapter` (Option C interface: `signature_attributes` +
    `from_otel`) with `MLflowAdapter`, `GenAIAdapter` (OTel GenAI convention),
    and `OpenInferenceAdapter` (Arize/Phoenix). See ADR-0013.
  - `parse_otel(source, dialect="auto")` + `detect_dialect()` — ingest
    OTLP-JSON (or a list of span dicts) from any supported dialect, with a
    priority registry, per-span auto-detect, and a best-effort universal
    fallback. `RagpillTraceSettings` (env `RAGPILL_TRACE_*`) configures the
    default dialect/fallback. The `gen_ai` and `openinference` adapters ship
    always-on (no extra) since they only read attribute keys.

### Fixed

- **Session-mode assessments and trace tags are uploaded again.** With the
  default MLflow backend's session grouping there is no case-level trace, so
  every `log_assessment`/`set_trace_tag` was silently skipped; per-run
  assessments now target each repeat's own trace (`RunResult.trace_id`), with
  aggregates and tags fanned out to the per-run traces.
- **Remote backends no longer receive the temp SQLite URI.** With no tracking
  URI, Langfuse/Phoenix used the MLflow-specific `sqlite:///` temp path as
  their endpoint; a `supports_local_file_store` capability now gates the temp
  store, and remote backends fall back to their env-derived destination.
- **Source evaluators read the neutral `Span.documents` field** (with the
  legacy `page_content` outputs parse as fallback), so Phoenix/OpenInference
  retriever documents are found.
- **Evaluator isolation restored.** When a run's span is missing from the
  trace, `SpanBaseEvaluator.get_trace` returns an empty span set instead of
  the full case trace (which silently scored other repeats' spans).
- **Langfuse/Phoenix `await_trace` no longer returns partial traces** — they
  poll until the span set is stable across two consecutive polls.
- **Phoenix backend lazily configures its tracer from env defaults** instead
  of crashing with `AttributeError` when spans open before `set_destination`.
- **Langfuse root spans keep `parent_id=None`** (a `str(None)` bug rendered
  whole traces empty).
- **GenAI adapter routes `gen_ai.assistant.message` prompt-history events to
  `messages_in`** (only `gen_ai.choice` is the completion).
- **MLflow adapter maps `TASK`/`GUARDRAIL` span types** instead of degrading
  ragpill's own run spans to `UNKNOWN`.
- **Trace fetches no longer block the event loop** — synchronous polling runs
  in a worker thread and session-mode repeats are fetched concurrently.
- **Span-based evaluators no longer silently dropped by a trace-export race.**
  Trace fetching now polls the backend until the trace is exported
  (`Backend.await_trace`), instead of reading immediately after the span
  closes and missing it (or returning the wrong trace via a one-result
  fallback). Poll budget is configurable via
  `MLFLOW_RAGPILL_TRACE_FETCH_TIMEOUT_S` / `_POLL_INTERVAL_S` (default
  10s / 0.5s). See ADR-0011.
- **Evaluator failures are visible in the triage report.** Evaluators that
  raise now appear in the per-evaluator rollup and render an `ERROR` line; a
  run whose assertions pass but whose evaluator errored is no longer reported
  as clean. `RunResult.all_passed` semantics are unchanged. See ADR-0012.
