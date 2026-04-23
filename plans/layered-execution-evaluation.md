# Plan: Layered Execution / Evaluation Architecture

**Status:** Ready for implementation
**Date:** 2026-04-21
**Design:** [designs/layered-execution-evaluation.md](../designs/layered-execution-evaluation.md)
**Dependency baked in:** [designs/remove-pydantic-evals.md](../designs/remove-pydantic-evals.md) — executed in Phase 0
**Approach:** Test-Driven Development (red → green → refactor) per phase

---

## 1. Goals

Split the monolithic `evaluate_testset_with_mlflow()` into three composable layers:

1. **Execute** — run tasks, capture traces, return `DatasetRunOutput`
2. **Evaluate** — run evaluators against captured outputs, return `EvaluationOutput`
3. **Upload** — persist results to MLflow server

Along the way:
- Drop the `pydantic_evals` dependency and own our evaluation primitives.
- Make the library **async-only** (remove `_sync` wrappers).
- Make `DatasetRunOutput` **JSON-serializable** for cross-machine sharing.
- Support **dual-backend tracing** from the start (local temp SQLite when no URI, direct server tracing when URI is provided).

## 2. Principles

- **TDD per phase.** Write failing tests first, then implement to green, then clean up.
- **One logical concern per commit** (Phase 0 → Phase 1 → …).
- **Public API surface is explicit.** Every new public function/class listed in this plan gets a Google-style docstring and an `::: module.path` entry in `docs/api/` (Phase 4).
- **Tests follow [.claude/skills/test-writing-guidelines](../.claude/skills/test-writing-guidelines/SKILL.md):** flat functions, `@pytest.mark.anyio` for async, `TestModel()` for LLM calls, integration tests in `*_integration.py` files, no class grouping unless shared state demands it.
- **Docs follow [.claude/skills/documentation-guidelines](../.claude/skills/documentation-guidelines/SKILL.md):** Google-style docstrings are the source of truth, API pages auto-generate.

## 3. Phase Map

| Phase | Focus | Output |
|---|---|---|
| 0 | Remove `pydantic_evals`, make library async-only | `eval_types.py`, `llm_judge.py`, no `_sync` wrappers |
| 1 | Output types + Execute layer (with JSON serialization + dual-backend tracing) | `execution.py`, `DatasetRunOutput` |
| 2 | Evaluate layer + evaluator refactor | `evaluation.py`, `evaluate_results()`, refactored `SpanBaseEvaluator` |
| 3 | Upload layer + full pipeline rewire | `upload.py`, `upload_to_mlflow()`, thin `evaluate_testset_with_mlflow()` |
| 4 | Documentation | Docstrings, API pages, guide, notebooks, README |

---

## Phase 0 — Remove `pydantic_evals` + Async-Only

### 0.1 Goals

- Replace every `pydantic_evals` import with local types in `src/ragpill/eval_types.py`.
- Reimplement the LLM judge using `pydantic_ai.Agent` directly in `src/ragpill/llm_judge.py`.
- Remove `WrappedPydanticEvaluator`.
- Remove `evaluate_testset_with_mlflow_sync()` (breaking change — library becomes async-only).
- `EvaluatorContext` is defined locally **without** `trace`/`run_span_id` fields yet — those are added in Phase 1 where they belong with the data flow.

### 0.2 TDD — Red

Write these tests first (all expected to fail until implementation lands):

**`tests/test_eval_types.py`** (new, unit)
- `EvaluationReason` holds `value` and optional `reason`.
- `EvaluatorSource` holds `name` and `arguments` dict.
- `EvaluationResult` composes `EvaluatorSource`.
- `Case` and `Dataset` are plain dataclasses with the expected fields.
- `EvaluatorContext` constructs with `kw_only=True` and the design's fields (no `_span_tree`).
- `BaseEvaluator.build_serialization_arguments()` returns only non-default fields.
- `BaseEvaluator.get_serialization_name()` returns the class name.

**`tests/test_llm_judge.py`** (update existing)
- Replace `pydantic_evals` imports with `ragpill.eval_types` / `ragpill.llm_judge`.
- Use `TestModel()` from `pydantic_ai.models.test` for fast deterministic tests.
- Assert `judge_output()` returns a `GradingOutput` with `reason`, `pass_`, `score`.
- Assert `judge_input_output()` includes inputs in the prompt body.

**Update existing tests to import from `ragpill.eval_types`:**
`test_types.py`, `test_dataframe.py`, `test_aggregation.py`, `test_mlflow_integration.py`,
`test_literal_quotes.py`, `test_regex_in_sources.py`, `test_regex_in_doc_metadata.py`,
`test_regex_in_output.py`, `test_has_quotes.py`. Also replace `EvaluatorSpec(...)` →
`EvaluatorSource(...)` in those files.

**Sync removal check:** Remove any test that calls `evaluate_testset_with_mlflow_sync`. Add a
`@pytest.mark.xfail(strict=True)` test asserting that attribute is gone (then delete when green).

### 0.3 TDD — Green

1. Create `src/ragpill/eval_types.py` with `EvaluationReason`, `EvaluatorSource`,
   `EvaluationResult`, `EvaluatorContext` (no `_span_tree`, no trace fields yet), `Case`,
   `Dataset`.
2. Create `src/ragpill/llm_judge.py` with `GradingOutput`, `JUDGE_SYSTEM_PROMPT`,
   `judge_output()`, `judge_input_output()`. Extract the exact system prompt from the
   installed `pydantic_evals` source for behavioral parity.
3. Rewrite `BaseEvaluator` in `base.py` as a standalone `@dataclass`:
   - Add `get_serialization_name()` classmethod.
   - Add `build_serialization_arguments()` method (iterate `fields`, compare to defaults).
   - Switch imports to `ragpill.eval_types`.
4. Update `evaluators.py`:
   - Switch imports.
   - Remove `WrappedPydanticEvaluator`.
   - Update `LLMJudge` to call `ragpill.llm_judge.judge_*`. Fix the `super(BaseEvaluator,
     self).build_serialization_arguments()` call → `super().build_serialization_arguments()`.
5. Update `mlflow_helper.py`:
   - Switch imports.
   - `EvaluatorSpec(...)` → `EvaluatorSource(...)`.
   - Remove `_span_tree=None` from the `EvaluatorContext` construction.
   - **Delete `evaluate_testset_with_mlflow_sync()`**.
6. Update `csv/testset.py`, `types.py`, `utils.py`, `__init__.py` imports; drop
   `WrappedPydanticEvaluator` from `default_evaluator_classes` and `__all__`.

### 0.4 TDD — Refactor

- Run `grep -r "pydantic_evals" src/ tests/` — must return zero hits.
- Run `uv run pytest tests/ -v` — all tests pass.
- Run `uv run basedpyright src/` — no new type errors.

### 0.5 Files touched (Phase 0)

| File | Change |
|---|---|
| `src/ragpill/eval_types.py` | **NEW** |
| `src/ragpill/llm_judge.py` | **NEW** |
| `src/ragpill/base.py` | rewrite `BaseEvaluator`, swap imports |
| `src/ragpill/evaluators.py` | swap imports, remove `WrappedPydanticEvaluator`, fix LLMJudge super call |
| `src/ragpill/mlflow_helper.py` | swap imports, `EvaluatorSpec→EvaluatorSource`, **delete `_sync`** |
| `src/ragpill/csv/testset.py` | swap imports, drop `WrappedPydanticEvaluator` |
| `src/ragpill/types.py` | swap import |
| `src/ragpill/utils.py` | swap import |
| `src/ragpill/__init__.py` | drop `WrappedPydanticEvaluator`, add new exports |
| `tests/test_eval_types.py` | **NEW** |
| `tests/test_llm_judge.py` | rewrite imports, use `TestModel()` |
| `tests/test_*.py` (9 files) | swap imports |

### 0.6 Success criteria

- Zero `pydantic_evals` imports remain in `src/` or `tests/`.
- No `*_sync` entry point in the public API.
- Full test suite green.

---

## Phase 1 — Output Types + Execute Layer

### 1.1 Goals

- Introduce `TaskRunOutput`, `CaseRunOutput`, `DatasetRunOutput` dataclasses.
- Introduce JSON serialization (`to_json()` / `from_json()` on `DatasetRunOutput`).
- Implement `execute_dataset()` in a new `src/ragpill/execution.py` module.
- Support **dual-backend tracing** from the start:
  - `mlflow_tracking_uri=None` → local temp SQLite backend
  - `mlflow_tracking_uri=<uri>` → trace directly to the provided server
  - Both paths extract `Trace` objects into the output dataclasses.
- Add `trace` and `run_span_id` fields to `EvaluatorContext` (they belong to the data flow
  from Phase 1 onward even though they are consumed in Phase 2).

### 1.2 TDD — Red

**`tests/test_execution.py`** (new, unit — no MLflow server required)
- `TaskRunOutput` dataclass shape matches design (run_index, input_key, output, duration,
  trace, run_span_id, error).
- `CaseRunOutput` and `DatasetRunOutput` shape matches design.
- `DatasetRunOutput.to_json()` produces a JSON string; `from_json()` round-trips it.
- Trace serialization strips `RLock` and OpenTelemetry refs; span data (span_id, parent_id,
  name, span_type, inputs, outputs, attributes) survives the round trip.
- `execute_dataset(..., capture_traces=False)` returns a `DatasetRunOutput` with
  `TaskRunOutput.trace is None` and `run_span_id == ""`.
- `execute_dataset()` propagates task exceptions into `TaskRunOutput.error` (does not raise).
- `resolve_repeat()` still applies (per-case `repeat` / `threshold` overrides).

**`tests/test_execution_integration.py`** (new, integration — real local SQLite)
- `execute_dataset()` with `capture_traces=True` and no URI creates and cleans up a temp
  SQLite DB; the DB file does not survive after the call.
- Traces captured from a minimal pydantic-ai agent contain the expected child spans
  (Agent, LLM call). Use a `TestModel()` so no real LLM API call is made.
- A case with `repeat=3` yields 3 `TaskRunOutput` entries each with a distinct
  `run_span_id`.
- When `mlflow_tracking_uri` is passed, execution uses that URI (dual-backend). Use a
  second temp SQLite URI as the "server" and assert traces appear there.

Mark anything not-yet-implemented as `@pytest.mark.xfail(strict=False, reason="Phase 1 TDD")`.

### 1.3 TDD — Green

1. Add `trace: Trace | None = None` and `run_span_id: str | None = None` to
   `EvaluatorContext` in `eval_types.py`. These are unused until Phase 2 but the type is
   settled now.
2. Define `TaskRunOutput`, `CaseRunOutput`, `DatasetRunOutput` in `src/ragpill/execution.py`
   (or a new `src/ragpill/run_types.py` — see 1.5).
3. Implement `_trace_to_dict()` / `_trace_from_dict()` helpers that convert
   `mlflow.entities.Trace` ↔ plain dicts. Walk `trace.data.spans` and serialize each
   `Span` into a dict of its safe fields.
4. Implement `DatasetRunOutput.to_json()` / `from_json()` using the helpers above.
5. Implement `_setup_local_tracing()` / `_teardown_local_tracing()` (temp SQLite path).
6. Implement `_setup_server_tracing(uri)` that sets the tracking URI to the provided server.
7. Implement `execute_dataset(...)` that:
   - Chooses backend based on `mlflow_tracking_uri`.
   - Wraps each case in a parent span; each run in a `run-{i}` span (as existing code does).
   - Captures each run's `run_span_id` and wall-clock duration.
   - Extracts `Trace` objects after each case's spans are committed.
   - Cleans up temp DB on success and failure.
8. Extract the current `_execute_case_runs` logic from `mlflow_helper.py` into
   `execution.py`. `evaluate_testset_with_mlflow()` keeps working by calling
   `execute_dataset()` internally (for now — full rewire happens in Phase 3).

### 1.4 TDD — Refactor

- Remove duplication between `mlflow_helper.py` task-execution code and
  `execution.py`.
- Make sure temp DB cleanup is in a `try/finally` — tests should verify cleanup even on
  task errors.

### 1.5 Design decision: module boundary

Keep all new types and the execute function in a single `src/ragpill/execution.py` for
now. If file size exceeds ~500 lines, split types into `run_types.py`. Decision is made
when file is written, not upfront.

### 1.6 Files touched (Phase 1)

| File | Change |
|---|---|
| `src/ragpill/execution.py` | **NEW** — types + `execute_dataset()` + serialization |
| `src/ragpill/eval_types.py` | add `trace`/`run_span_id` to `EvaluatorContext` |
| `src/ragpill/mlflow_helper.py` | delegate task execution to `execute_dataset()` (internal call) |
| `src/ragpill/__init__.py` | export `execute_dataset`, `DatasetRunOutput`, `CaseRunOutput`, `TaskRunOutput` |
| `tests/test_execution.py` | **NEW** |
| `tests/test_execution_integration.py` | **NEW** |

### 1.7 Success criteria

- `execute_dataset()` works standalone without a running MLflow server.
- Dual-backend: both `mlflow_tracking_uri=None` and `mlflow_tracking_uri=<uri>` produce a
  valid `DatasetRunOutput` with trace data.
- `DatasetRunOutput.to_json()` followed by `from_json()` round-trips identically
  (assertion on dict equality of the serialized form).
- Temp DB files are cleaned up in both success and exception paths.
- Existing `evaluate_testset_with_mlflow()` still green — Phase 1 is additive.

---

## Phase 2 — Evaluate Layer

### 2.1 Goals

- Refactor `SpanBaseEvaluator` and `SourcesBaseEvaluator` to read trace data from
  `ctx.trace` (added in Phase 1) instead of calling `mlflow.search_traces()`.
- Implement `evaluate_results(dataset_run, dataset) -> EvaluationOutput` in a new
  `src/ragpill/evaluation.py` module.
- Remove obsolete plumbing: `_current_run_span_id` ContextVar, `inputs_to_key_function`,
  and the MLflow-querying fields on `SpanBaseEvaluator`.

### 2.2 TDD — Red

**`tests/test_evaluation.py`** (new, unit)
- `evaluate_results()` on a minimal `DatasetRunOutput` + `Dataset` returns an
  `EvaluationOutput` with the expected `runs` / `cases` DataFrames.
- Each `EvaluatorContext` passed to evaluators carries the `Trace` from
  `CaseRunOutput.trace` and the `run_span_id` from `TaskRunOutput.run_span_id`.
- An evaluator raising an exception is captured into `RunResult.evaluator_failures` (no
  crash of the outer loop).
- A task with `error != None` short-circuits all evaluators to failure (current
  behavior preserved).
- `_aggregate_runs()` behavior matches the existing pre-refactor tests.

**`tests/test_span_based_evaluators.py`** (existing files, update)
- `RegexInSourcesEvaluator`, `RegexInDocumentMetadataEvaluator`, `LiteralQuoteEvaluator`
  still produce identical results when given a `ctx.trace` directly. Tests feed a
  hand-crafted `Trace` rather than going through MLflow.
- `SpanBaseEvaluator.get_trace(ctx)` raises `ValueError` when `ctx.trace is None`.
- `SourcesBaseEvaluator.get_documents(ctx)` extracts from `ctx.trace` (not from
  `ctx.inputs`).

**Regression sweep:** every existing evaluator test passes with the new ctx signature.

### 2.3 TDD — Green

1. Refactor `SpanBaseEvaluator` in `evaluators.py`:
   - Remove `_mlflow_settings`, `_mlflow_experiment_id`, `_mlflow_run_id` fields.
   - Remove the `mlflow_settings`, `mlflow_experiment_id`, `mlflow_run_id` properties.
   - Remove `inputs_to_key_function` field.
   - Rewrite `get_trace(ctx)` to return `ctx.trace`, raising `ValueError` if absent.
     Apply `_filter_trace_to_subtree(ctx.trace, ctx.run_span_id)` when `run_span_id` is
     set.
2. Refactor `SourcesBaseEvaluator.get_documents(ctx)` to call `self.get_trace(ctx)`.
   Its `run(ctx)` already had `ctx` in scope — simple update.
3. Update `RegexInSourcesEvaluator`, `RegexInDocumentMetadataEvaluator`,
   `LiteralQuoteEvaluator` to pass `ctx` to `get_documents` rather than `ctx.inputs`.
4. Delete `_current_run_span_id` ContextVar from `base.py`. Grep for any remaining use;
   remove.
5. Delete `default_input_to_key()` from `base.py`.
6. Implement `evaluate_results(dataset_run, dataset)` in new `src/ragpill/evaluation.py`:
   - Iterate `zip(dataset_run.cases, dataset.cases)`.
   - For each run, build an `EvaluatorContext` with `trace=case_run.trace`,
     `run_span_id=task_run.run_span_id`.
   - Run each evaluator; collect `RunResult` / `CaseResult`.
   - Aggregate via `_aggregate_runs()`.
   - Build `runs` / `cases` DataFrames.
   - Return `EvaluationOutput(runs=..., cases=..., case_results=..., dataset_run=dataset_run)`.
7. Add `dataset_run: DatasetRunOutput | None = None` to `EvaluationOutput` in `types.py`.
8. In `mlflow_helper.py`, replace the Phase-1 bridge code that still calls
   `_evaluate_run` inline with a call to `evaluate_results()`.

### 2.4 TDD — Refactor

- Remove `_evaluate_run()` from `mlflow_helper.py` (moved to `evaluation.py`).
- Remove unused imports in `mlflow_helper.py`.
- Confirm `grep -r "_current_run_span_id" src/ tests/` returns zero hits.
- Confirm `grep -r "inputs_to_key_function" src/ tests/` returns zero hits.

### 2.5 Files touched (Phase 2)

| File | Change |
|---|---|
| `src/ragpill/evaluation.py` | **NEW** — `evaluate_results()` |
| `src/ragpill/evaluators.py` | refactor `SpanBaseEvaluator`, `SourcesBaseEvaluator`, regex/literal subclasses |
| `src/ragpill/base.py` | delete `_current_run_span_id`, `default_input_to_key()` |
| `src/ragpill/types.py` | add `dataset_run` to `EvaluationOutput` |
| `src/ragpill/mlflow_helper.py` | delegate evaluation to `evaluate_results()` |
| `src/ragpill/__init__.py` | export `evaluate_results` |
| `tests/test_evaluation.py` | **NEW** |
| `tests/test_regex_in_sources.py` | feed `ctx.trace` directly |
| `tests/test_regex_in_doc_metadata.py` | feed `ctx.trace` directly |
| `tests/test_literal_quotes.py` | feed `ctx.trace` directly |

### 2.6 Success criteria

- `evaluate_results()` runs with zero MLflow configuration (no tracking URI, no active
  run).
- Span-based evaluators pass unit tests with hand-built `Trace` objects.
- The legacy ContextVar and key-function plumbing are gone from the codebase.

---

## Phase 3 — Upload Layer + Full Pipeline Rewire

### 3.1 Goals

- Implement `upload_to_mlflow(evaluation, mlflow_settings, model_params, upload_traces)`
  in `src/ragpill/upload.py`.
- Rewire `evaluate_testset_with_mlflow()` into a thin orchestrator calling the three
  layers.
- Use the dual-backend model end-to-end:
  - When `evaluate_testset_with_mlflow()` is called with settings that include a server
    URI, `execute_dataset()` traces directly to the server; `upload_to_mlflow()` runs
    with `upload_traces=False` because traces are already there.
  - When `upload_to_mlflow()` is called standalone on a `DatasetRunOutput` that was
    captured locally, it uploads the trace data as MLflow artifacts (serialized JSON),
    plus assessments/metrics/tags.

### 3.2 TDD — Red

**`tests/test_upload.py`** (new, unit where possible — mock MLflow client)
- `upload_to_mlflow()` with `upload_traces=False` calls `mlflow.log_table`,
  `mlflow.log_metric`, assessment log calls with the expected arguments.
- `upload_to_mlflow()` with `upload_traces=True` additionally serializes traces and
  uploads them as artifacts (assert `mlflow.log_artifact` called with a JSON blob).
- `upload_to_mlflow()` returns `None` and closes the run cleanly on success and on
  exception.

**`tests/test_mlflow_integration.py`** (update — this is the real end-to-end test)
- Full pipeline: `execute_dataset` → `evaluate_results` → `upload_to_mlflow` against a
  running MLflow server (same gating as today: skip unless the server is reachable).
- `evaluate_testset_with_mlflow()` still produces the same DataFrames it did before the
  refactor. Use a snapshot-style assertion on the `cases` DataFrame shape.
- New test: `execute_dataset` offline → JSON round-trip → `upload_to_mlflow` later.
  Proves the "disconnected execution" use case from the design.
- New test: one `DatasetRunOutput`, two different evaluator sets, two separate
  `evaluate_results()` + `upload_to_mlflow()` calls. Proves "run once, evaluate many."

### 3.3 TDD — Green

1. Implement `upload_to_mlflow()` in `src/ragpill/upload.py`:
   - Connect to server, start run.
   - Upload assessments (per-run, aggregate).
   - Upload metrics (overall accuracy, per-tag accuracy).
   - Upload `runs` DataFrame via `mlflow.log_table`.
   - When `upload_traces=True`, serialize each `CaseRunOutput.trace` as JSON and
     `mlflow.log_artifact` it. Defer server-side trace recreation — it's a UI-only
     concern.
   - End the run in a `finally` block.
2. Rewire `evaluate_testset_with_mlflow()` in `mlflow_helper.py`:

   ```python
   async def evaluate_testset_with_mlflow(testset, task, task_factory, mlflow_settings, model_params):
       settings = mlflow_settings or MLFlowSettings()
       run_output = await execute_dataset(
           testset, task=task, task_factory=task_factory,
           settings=settings,
           mlflow_tracking_uri=settings.ragpill_tracking_uri,
       )
       eval_output = await evaluate_results(run_output, testset)
       upload_to_mlflow(eval_output, settings, model_params, upload_traces=False)
       return eval_output
   ```
3. Move tag/metadata upload helpers into `upload.py` (keep them as private `_` helpers).
4. Leave `mlflow_helper.py` as a compatibility shim re-exporting
   `evaluate_testset_with_mlflow` from the new modules — or delete it entirely if no
   existing imports point there. Decide when touching the file.

### 3.4 TDD — Refactor

- After rewiring, `mlflow_helper.py` should either be ~40 lines (thin orchestrator) or
  gone.
- No duplicated upload logic across files.
- `grep -r "evaluate_testset_with_mlflow_sync" .` should still return zero.

### 3.5 Files touched (Phase 3)

| File | Change |
|---|---|
| `src/ragpill/upload.py` | **NEW** — `upload_to_mlflow()` + helpers |
| `src/ragpill/mlflow_helper.py` | shrink to thin orchestrator or delete |
| `src/ragpill/__init__.py` | export `upload_to_mlflow` |
| `tests/test_upload.py` | **NEW** |
| `tests/test_mlflow_integration.py` | update to exercise full pipeline + new use cases |

### 3.6 Success criteria

- All three layers callable independently; combined orchestrator reproduces old behavior.
- "Run once, evaluate many times" verified by integration test.
- "Disconnected execution + upload later" verified by integration test.
- No regression in existing integration test assertions.

---

## Phase 4 — Documentation

### 4.1 Goals

- Bring docstrings, API reference, guides, notebooks, and README in line with the new
  public API.
- Follow [documentation-guidelines](../.claude/skills/documentation-guidelines/SKILL.md):
  Google-style docstrings as source of truth, `::: module.path` only in `docs/api/`,
  guides use markdown links.

### 4.2 Work items

**Docstrings (required sections: Summary, Args, Returns, Raises, Example, See Also):**
- `execute_dataset()`
- `evaluate_results()`
- `upload_to_mlflow()`
- `evaluate_testset_with_mlflow()` — rewritten as "chains the three layers"
- `TaskRunOutput`, `CaseRunOutput`, `DatasetRunOutput` — include invariants and
  serialization round-trip example
- `DatasetRunOutput.to_json()` / `from_json()`
- New fields on `EvaluatorContext` (`trace`, `run_span_id`)
- New field on `EvaluationOutput` (`dataset_run`)
- `EvaluationReason`, `EvaluationResult`, `EvaluatorSource`, `Case`, `Dataset`,
  `EvaluatorContext` (from Phase 0)
- `judge_output()`, `judge_input_output()`, `GradingOutput` (from Phase 0)
- Revised `BaseEvaluator` class docstring
- `SpanBaseEvaluator.get_trace(ctx)` — document the new `ctx.trace` contract

**API reference (`docs/api/`):**
- New file `docs/api/execution.md` with `::: ragpill.execution.*` entries.
- New file `docs/api/evaluation.md` with `::: ragpill.evaluation.*` entries.
- New file `docs/api/upload.md` with `::: ragpill.upload.*` entries.
- New file `docs/api/eval_types.md` with `::: ragpill.eval_types.*` entries.
- New file `docs/api/llm_judge.md` with `::: ragpill.llm_judge.*` entries.
- Update `docs/api/mlflow.md` — only `evaluate_testset_with_mlflow` remains here (or redirect if file deleted).
- Update `docs/api/evaluators.md` — drop `WrappedPydanticEvaluator`.
- Add every new `.md` file to `mkdocs.yml` `nav:`.

**Guides (`docs/guide/`):**
- Add `docs/guide/layered-architecture.md` — explains the three-layer model, when to use
  each layer, shows the four use cases from design section 7 (run-once-evaluate-many,
  CI without server, serialize-and-share, disconnected execution).
- Update `docs/guide/evaluators.md` — `WrappedPydanticEvaluator` removed, LLMJudge reimplemented.
- Update `docs/guide/csv-adapter.md` — drop `WrappedPydanticEvaluator` reference.

**How-to / tutorials / notebooks:**
- Update every notebook that constructs/uses `evaluate_testset_with_mlflow` to work with
  the new three-layer API. Re-run the notebooks and commit updated outputs.
- Add a new how-to notebook: "Evaluate historical outputs" exercising JSON
  serialization + `evaluate_results()` on a persisted `DatasetRunOutput`.

**Getting started / README:**
- Update `docs/getting-started/quickstart.md` — simplest path still goes through
  `evaluate_testset_with_mlflow()`, but mention the three-layer API.
- Update `README.md` — drop "built on pydantic-evals" framing; mention async-only.

### 4.3 Validation

- `uv run mkdocs build --strict` passes.
- Manual site walkthrough: every new public symbol appears in the rendered API
  reference.
- `grep -r "WrappedPydanticEvaluator" docs/` returns zero hits.
- `grep -r "_sync" docs/` returns zero hits (except in historic release notes, if any).

### 4.4 Files touched (Phase 4)

| File | Change |
|---|---|
| `src/**/*.py` | docstring updates for every new/changed public symbol |
| `docs/api/{execution,evaluation,upload,eval_types,llm_judge}.md` | **NEW** |
| `docs/api/{mlflow,evaluators}.md` | update |
| `docs/guide/layered-architecture.md` | **NEW** |
| `docs/guide/{evaluators,csv-adapter}.md` | update |
| `docs/how-to/*` | update notebooks |
| `docs/tutorials/*` | update notebooks |
| `docs/getting-started/quickstart.md` | update |
| `mkdocs.yml` | add new `nav:` entries |
| `README.md` | update |

### 4.5 Success criteria

- `uv run mkdocs build --strict` passes.
- Every public symbol has a Google-style docstring with an Example block.
- Every new `.md` is in `mkdocs.yml` nav.
- All notebooks re-run without errors.

---

## 4. Out of Scope

- **Server-side trace recreation via MLflow client API** (design §5.2 Option B). Artifact
  upload (Option D) is the chosen mechanism for the locally-captured → upload-later path.
  A future plan can revisit this if users want native trace UI for that case.
- **Parallel case execution.** Current behavior is sequential; not changed here.
- **Trace truncation / `keep_traces=False`** — design §12.1 mitigation. Defer until
  memory becomes a reported problem.

## 5. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| JSON serialization may lose OTel-specific span attributes | Serialize only the fields listed in design §3.2; add a regression test asserting round-trip equality of the serialized form |
| LLM judge prompt drift vs pydantic_evals version | Copy the exact system prompt at Phase 0 implementation time |
| `mlflow.pydantic_ai.autolog()` globally mutates state | Save and restore tracking URI around `execute_dataset()`; test asserts URI is restored |
| Tests flake due to MLflow global state | Integration tests already gated on server availability; unit tests avoid global state by passing `Trace` objects directly |
| Breaking change on sync removal | Release notes + migration section in `docs/guide/layered-architecture.md` |

## 6. Exit Criteria (whole plan)

- All phases landed and tested.
- No `pydantic_evals` imports, no `_sync` entry points, no `_current_run_span_id`
  ContextVar, no `inputs_to_key_function`.
- Four design use cases (§7.1–§7.4) each exercised by an integration test.
- Docs build cleanly.
- `uv run pytest tests/ -v` green; `uv run basedpyright src/` clean.
