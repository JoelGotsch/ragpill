# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). This project is
pre-1.0, so minor versions may carry breaking changes.

## [Unreleased]

Phases 1–8 of the review follow-up: correctness blockers, honest failure
attribution, a backend-neutral API clean break, concurrency foundations, upload
robustness, backend data fidelity, and a documentation golden-path sprint.
Backwards compatibility is a non-goal pre-1.0, so the renames below have no
deprecated aliases.

### Documentation

- Rewrote the getting-started surface around the current API: correct `RAGPILL_`
  env vars (the docs previously showed a nonexistent `EVAL_MLFLOW_*` prefix),
  the backend extras and the Phoenix↔MLflow conflict, a finished `index.md`
  homepage, a zero-server quickstart with `asyncio.run(...)`, and the renamed
  functions/settings throughout (guides, how-tos, tutorial notebook).
- Custom-evaluator examples now decorate field-bearing subclasses with
  `@dataclass(kw_only=True)` (they previously raised `TypeError`) and drop the
  nonexistent `BaseTestInput`/`input=` API.
- README gained an install section with the extras; Langfuse/Phoenix are shown
  as shipped (not "planned"); the ableist LLMJudge analogy in the test-sets
  guide was replaced.
- ADRs are now in the mkdocs nav; the stale pydantic-evals `site_description`
  was corrected. A `tests/test_docs_quickstart.py` exercises the documented
  zero-server flow so it can't silently rot.

### Fixed

- **The traced capture path is now portable across asyncio and trio.** The
  trace-fetch offloading used asyncio primitives directly; it now uses anyio,
  so traced runs work under either backend (matching the Phase 4 evaluation path).

- **Langfuse and Phoenix traces now carry real span timestamps and status.**
  Both adapters previously hard-coded `start_time`/`end_time` to `0`, so every
  span rendered as `0ms` and time-ordering was arbitrary. They now lift the real
  timestamps (and Langfuse maps its level to a span status).
- **Phoenix trace reads are filtered with a vectorized mask** instead of
  iterating and dict-converting every span in the project on each poll.

### Added

- **Idempotent, resumable upload.** `upload_results` records an upload-state tag
  on the run: a run that already completed refuses to re-upload (raising unless
  `overwrite=True`), and a retry after a partial failure replaces the append-only
  results-table artifact instead of duplicating rows. Adds `set_run_tag` /
  `get_run_tag` / `delete_run_artifact` to the backend protocol (no-ops on
  backends without a native run concept).
- **`upload_results(tracking_uri=…)`** and destination provenance: the upload
  destination resolves as explicit arg > the run's recorded
  `DatasetRunOutput.tracking_uri` > `settings.tracking_uri`, so an upload can no
  longer silently reattach a run against a different server than it was captured
  on.
- **Opt-in, caller-specified timeouts.** `execute_dataset(task_timeout_s=…)`
  bounds each task and `LLMJudge(timeout_s=…)` bounds each judge call; both
  default to `None` (no timeout) — ragpill never imposes a latency budget on
  your code. A timed-out task/judge is recorded as a `TimeoutError` and the run
  continues.
- **Bounded-concurrent evaluation.** `evaluate_results(max_concurrency=…)`
  overlaps `(case, run)` evaluations (default `1`, i.e. unchanged sequential
  behaviour); results are identical regardless of the value. Built on anyio so
  it works under both asyncio and trio.

### Changed

- **Judge-trace cleanup is no longer silently capped at 1000.** `delete_judge_traces`
  now searches with a large limit (MLflow paginates internally) and logs how
  many judge traces it deleted, so large runs don't leave judge clutter behind.
- **Synchronous tasks run off the event loop** (via `anyio.to_thread`), so a
  blocking client (e.g. a `requests`-based RAG call) no longer stalls the loop
  and the tracking exporters running on it.
- **MLflow case-grouping session state moved to `ContextVar`s**, so two
  concurrent `execute_dataset` calls (each an asyncio task with its own context)
  can't cross-tag each other's traces through the shared backend singleton.
  Capture itself remains sequential per call (a deliberate constraint of the
  process-global tracking state, now documented on `execute_dataset`).

### Breaking

- **`MLFlowSettings` → `TrackingSettings`**, with the env prefix changed from
  the double-prefixed `MLFLOW_RAGPILL_*` to `RAGPILL_*` and the redundant
  `ragpill_` field-name prefix dropped: `tracking_uri`, `experiment_name`,
  `run_description`, `repeat`, `threshold`, `trace_fetch_timeout_s`,
  `trace_fetch_poll_interval_s` (env `RAGPILL_TRACKING_URI`, `RAGPILL_REPEAT`, …).
- **`tracking_uri` now defaults to `None`** (a private temp SQLite store /
  zero-server capture) instead of a silent `http://localhost:5000`. The
  dead `tracking_username` / `tracking_password` fields are removed — set
  `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in the environment,
  which mlflow reads directly.
- **`upload_to_mlflow` → `upload_results`**; **`evaluate_testset_with_mlflow` →
  `evaluate_testset`**; **`execute_dataset(mlflow_tracking_uri=…)` →
  `tracking_uri=`**; the `upload_results` / `evaluate_testset` settings param is
  now `settings`. `evaluate_testset` requires a `tracking_uri` (it uploads to a
  server) and raises a clear error when none is set — use
  `execute_dataset` + `evaluate_results` for a zero-server run.
- **Write-side `ragpill.backends.SpanKind` → `CaptureSpanKind`** (the
  ingest-side `ragpill.trace.SpanKind` keeps its name), removing the same-name
  collision between the two enums.
- **Keyword-only booleans**: `execute_dataset`'s `settings` / `tracking_uri` /
  `capture_traces` and `upload_results`'s `model_params` / `upload_traces` are
  now keyword-only.
- **Curated public exports**: `load_testset`, `default_evaluator_classes`,
  `configure_backend`, `get_backend`, `Trace`, `TrackingSettings`, and
  `CaptureSpanKind` are now exported from `ragpill`; the backend adapter classes
  are exported lazily from `ragpill.backends`. `merge_settings` was removed from
  the top-level export (still importable from `ragpill.utils`).

### Changed

- **`LLMJudgeSettings` honors its documented defaults.** `base_url` and
  `api_key` are genuinely optional now: when unset, the OpenAI client resolves
  them from `OPENAI_BASE_URL` / `OPENAI_API_KEY`, and only a fully-missing API
  key fails — matching the field descriptions instead of pre-raising when all
  three of api_key/base_url/model_name weren't set.
- **`.env` files are loaded.** All settings classes set `env_file=".env"`, so
  the documented dotenv workflow works.
- **Trace-adapter entry-point discovery is cached** (`functools.cache`), so
  `parse_otel` no longer re-scans installed distributions once per span.

### Fixed

- **A trace that could not be read is now an evaluator error, not a false
  `False`.** When a span-based evaluator's trace was unavailable (fetch timed
  out, backend error, or the run's spans were still in flight), `get_trace`
  returned an empty span set, so `RegexInSourcesEvaluator` reported "pattern not
  found in any document" — a flaky tracking server was indistinguishable from a
  real regression. It now raises `TraceUnavailableError` (exported from
  `ragpill`), which the evaluation layer records as an evaluator failure (error
  state), distinct from a fail verdict.
- **Evaluator-failure rows no longer depress accuracy.** Rows for evaluators
  that could not run are written as `NaN` (not `False`) in the runs DataFrame,
  so `overall_accuracy` / `per_tag_accuracy` exclude them — matching
  `RunResult.all_passed` and the documented behavior, so the case-pass and
  accuracy dashboards no longer disagree.

### Fixed (Phase 1)

- **`evaluate_results` now verifies case identity, not just count.** Runs were
  paired to testset cases by index with only a length check, so a reordered or
  edited testset silently judged every output against the wrong case's
  evaluators/expected/rubric. It now compares each case's input hash
  (`base_input_key`) and raises with the mismatched indices.
- **Tasks with an async `__call__` are awaited.** `execute_dataset` used
  `inspect.iscoroutinefunction`, which is `False` for a callable instance whose
  `__call__` is async (the documented `task_factory` "stateful task" shape) — the
  output was an un-awaited coroutine graded by its repr. It now awaits any
  awaitable result.
- **CSV encoding fallback fixed.** `latin-1` (which decodes any byte sequence)
  was tried before `cp1252`/`utf-8`, so cp1252 files silently mojibaked and the
  later encodings were dead code; the fallback also retried IO errors and
  mislabeled a missing file as an encoding failure. `latin-1` is now the final
  catch-all, only `UnicodeDecodeError` is retried, and IO errors propagate.
- **Backend read paths distinguish "not found" from real errors.** MLflow /
  Langfuse / Phoenix `get_trace` swallowed *all* exceptions as `None`, so an
  auth/connection/5xx fault looked like an in-flight trace and burned the whole
  poll budget before surfacing a misleading "trace not populated". Not-found now
  returns `None`; transport/auth/server errors are logged and raised so polling
  aborts. A failed export flush on `end_run` is warned instead of silently
  dropping spans.
- **LLM judge is hardened against prompt injection.** Untrusted task output /
  inputs are escaped so they can't forge their own `</Output><Rubric>…` sections
  to steer the verdict, and each judge system prompt now states that tagged
  content is untrusted data. A `JUDGE_PROMPT_VERSION` + prompt hash are logged as
  run params (when a judge ran) so a prompt edit that shifts scores is auditable.
- **Stable evaluator assertion names.** Duplicate-name suffixes (`LLMJudge_2`)
  are assigned once from the full evaluator list before any evaluator runs, so a
  judge that raises on one run no longer shifts another judge's identity across
  runs (which blended distinct rubrics in cross-run aggregation). Suffix counting
  is by exact class name, fixing an over-count when one name prefixed another.
- **Failed task runs record their real duration** instead of `0.0`.
- **`DatasetRunOutput.to_json` no longer crashes on non-JSON-serializable task
  outputs** — they are coerced with `str()` and a warning rather than raising
  after the expensive run completed.
- **Unknown span kinds deserialize to `UNKNOWN`** instead of raising, honoring
  the additive-schema promise of the run-JSON format.
- `default_input_to_key` uses `usedforsecurity=False` (works under FIPS Python).

### Changed

- **Evaluator provenance is declared, not guessed.** `BaseEvaluator.source_type`
  (`"CODE"` / `"LLM_JUDGE"`, overridden by `LLMJudge`) replaces string-matching
  `"LLMJudge"` in the class name to classify assessments; carried on
  `EvaluatorSource` and typed on `Assessment.source_type`.
- `Case.evaluators` / `Dataset.evaluators` are typed `list[BaseEvaluator]`
  (was `list[Any]`), removing the runtime isinstance asserts.
- `load_testset`'s `evaluator_classes` default is `None` (copied internally);
  `default_evaluator_classes` is now read-only (`MappingProxyType`) so in-place
  mutation can't leak across calls.
- User-facing validation (`BaseEvaluator.evaluate`, CSV `from_csv_line`) raises
  `TypeError`/`ValueError` instead of `assert` (which vanishes under `python -O`).
- Default secret-redaction patterns extended (bearer tokens, secret, password,
  set-cookie, access/refresh tokens).

### Removed

- `csv/questions_answers.py` (an empty vaporware stub).

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
