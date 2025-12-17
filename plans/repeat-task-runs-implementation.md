---
Status: Ready to implement
---

# Implementation Plan: Repeatable Task Runs (Approach C)

**Decisions locked in from design plan:**
- Option 1: Mutually exclusive `task` / `task_factory` parameters
- Approach C: Own orchestration loop (replace `testset.evaluate()`)
- Option 2: Return `EvaluationOutput` with `.runs`, `.cases`, `.summary` DataFrames
- Factory invocation policy: per-run (fresh task instance per run)
- Backward compatibility: not required (breaking change to return type is acceptable)
- Global defaults on `MLFlowSettings`, per-case overrides on `TestCaseMetadata`
- Two-phase execution: spans committed before evaluators run
- ContextVar + trace subtree filtering for span-based evaluator isolation
- MLflow assessment naming: `run-{i}_{evaluator_name}`, `agg_{evaluator_name}`

---

## Phase 1: Tests (TDD)

Write all tests first. Tests define the contract; implementation follows.

### Step 1.1 — Unit tests for new data types (`tests/test_types.py`)

**New file.** Tests for `RunResult`, `AggregatedResult`, `CaseResult`, `EvaluationOutput`.

```
Tests to write:

# RunResult
- test_run_result_all_passed_true: all assertions pass, error is None -> all_passed is True
- test_run_result_all_passed_false_assertion: one assertion fails -> all_passed is False
- test_run_result_all_passed_false_error: error is set -> all_passed is False regardless of assertions
- test_run_result_all_passed_empty_assertions: no assertions, no error -> all_passed is True

# AggregatedResult
- test_aggregated_result_passed: pass_rate >= threshold -> passed is True
- test_aggregated_result_failed: pass_rate < threshold -> passed is False
- test_aggregated_result_boundary: pass_rate == threshold exactly -> passed is True

# EvaluationOutput
- test_evaluation_output_summary_property: summary DataFrame has one row per case with case_id, passed, pass_rate, threshold
- test_evaluation_output_runs_and_cases_shapes: runs has more rows than cases when repeat > 1
```

These are pure data-class tests. No mocking, no I/O.

### Step 1.2 — Unit tests for aggregation logic (`tests/test_aggregation.py`)

**New file.** Tests for `_aggregate_runs()`.

```
Tests to write:

# Basic aggregation
- test_aggregate_all_pass: 3/3 runs pass, threshold=0.6 -> passed=True, pass_rate=1.0
- test_aggregate_all_fail: 0/3 runs pass, threshold=0.6 -> passed=False, pass_rate=0.0
- test_aggregate_partial_above_threshold: 2/3 runs pass, threshold=0.6 -> passed=True, pass_rate≈0.667
- test_aggregate_partial_below_threshold: 1/3 runs pass, threshold=0.6 -> passed=False, pass_rate≈0.333
- test_aggregate_exact_threshold: 3/5 runs pass, threshold=0.6 -> passed=True (0.6 >= 0.6)
- test_aggregate_just_below_threshold: 2/5 runs pass, threshold=0.5 -> passed=False (0.4 < 0.5)

# Threshold edge cases
- test_aggregate_threshold_zero: always passes regardless of results
- test_aggregate_threshold_one: requires all runs to pass
- test_aggregate_single_run_pass: repeat=1, passes -> passed=True
- test_aggregate_single_run_fail: repeat=1, fails -> passed=False

# Per-evaluator pass rates
- test_aggregate_per_evaluator_rates: 2 evaluators, different pass rates across 3 runs
- test_aggregate_per_evaluator_one_always_fails: one evaluator fails every run -> its rate is 0.0

# Task errors
- test_aggregate_with_task_error: one run has task error -> that run counts as failed
- test_aggregate_all_task_errors: all runs error -> pass_rate=0.0

# Summary text
- test_aggregate_summary_passed_includes_ratio: summary contains "2/3 runs passed"
- test_aggregate_summary_failed_includes_details: summary includes failed run details with reasons
```

### Step 1.3 — Unit tests for `task` / `task_factory` validation (`tests/test_task_factory.py`)

**New file.** Tests for parameter validation and factory invocation behavior.

```
Tests to write:

# Parameter validation (test the validation logic directly, no MLflow)
- test_both_task_and_factory_raises: providing both -> ValueError
- test_neither_task_nor_factory_raises: providing neither -> ValueError
- test_task_only_accepted: providing just task -> no error
- test_factory_only_accepted: providing just task_factory -> no error

# Factory invocation counting (mock/spy on factory)
- test_factory_called_per_run: repeat=3, 2 cases -> factory called 6 times (3 * 2)
- test_task_reused_across_runs: task= mode -> same callable used for all runs (not wrapped in factory)
- test_factory_returns_fresh_instance: factory returns distinct objects each call

# Stateful factory isolation
- test_stateful_task_without_factory_leaks: a stateful task (appends to list) with task= 
    -> second run sees first run's state (demonstrates the problem)
- test_stateful_task_with_factory_isolated: same stateful task via task_factory= 
    -> each run starts clean (demonstrates the fix)
```

### Step 1.4 — Unit tests for `TestCaseMetadata` and `MLFlowSettings` changes (`tests/test_base.py`, `tests/test_settings.py`)

**Extend `test_base.py`.** Add tests for `repeat` and `threshold` fields on `TestCaseMetadata`.

```
Tests to add to existing test_base.py:

# TestCaseMetadata repeat/threshold (per-case overrides, nullable)
- test_metadata_repeat_default: default repeat=None (defers to global)
- test_metadata_threshold_default: default threshold=None (defers to global)
- test_metadata_repeat_valid: repeat=5 accepted
- test_metadata_repeat_zero_rejected: repeat=0 -> ValidationError (ge=1)
- test_metadata_repeat_negative_rejected: repeat=-1 -> ValidationError
- test_metadata_threshold_valid_range: threshold=0.5 accepted
- test_metadata_threshold_zero_accepted: threshold=0.0 accepted (always pass)
- test_metadata_threshold_one_accepted: threshold=1.0 accepted
- test_metadata_threshold_above_one_rejected: threshold=1.1 -> ValidationError (le=1.0)
- test_metadata_threshold_negative_rejected: threshold=-0.1 -> ValidationError (ge=0.0)
```

**New file `test_settings.py`.** Tests for global defaults on `MLFlowSettings` and `resolve_repeat()`.

```
Tests to write:

# MLFlowSettings global defaults
- test_settings_repeat_default: default repeat=1
- test_settings_threshold_default: default threshold=1.0
- test_settings_repeat_from_env: MLFLOW_RAGPILL_REPEAT=5 -> repeat=5
- test_settings_threshold_from_env: MLFLOW_RAGPILL_THRESHOLD=0.7 -> threshold=0.7

# resolve_repeat() — merges per-case overrides with global defaults
- test_resolve_both_none_uses_global: case has repeat=None, threshold=None -> uses settings values
- test_resolve_case_overrides_repeat: case has repeat=5, settings has repeat=1 -> returns 5
- test_resolve_case_overrides_threshold: case has threshold=0.5, settings has threshold=1.0 -> returns 0.5
- test_resolve_case_overrides_both: case has both set -> both override
- test_resolve_no_metadata_uses_global: metadata is None -> uses settings values
```

### Step 1.5 — Unit tests for CSV parsing of repeat/threshold (`tests/test_testset_csv.py` — extend existing)

**Extend existing file.** Add tests for new CSV columns.

```
Tests to add:

# CSV repeat/threshold parsing
- test_csv_with_repeat_and_threshold: CSV has repeat=3, threshold=0.7 -> TestCaseMetadata has those values
- test_csv_without_repeat_threshold: CSV lacks columns -> defaults (1, 1.0)
- test_csv_empty_repeat_threshold: columns exist but values are empty -> defaults
- test_csv_inconsistent_repeat_across_rows: same question, different repeat values -> ValueError
- test_csv_inconsistent_threshold_across_rows: same question, different threshold values -> ValueError
- test_csv_repeat_not_in_evaluator_attributes: repeat column not leaked into evaluator.attributes
- test_csv_threshold_not_in_evaluator_attributes: threshold column not leaked into evaluator.attributes
```

**Test data:** Create `tests/data/testset_repeat.csv`:
```csv
Question,test_type,expected,tags,check,repeat,threshold
What is X?,RegexInOutputEvaluator,true,factual,x,3,0.6
What is X?,RegexInOutputEvaluator,true,factual,capital,3,0.6
What is Y?,RegexInOutputEvaluator,true,geography,y,,
```

### Step 1.6 — Unit tests for evaluator span isolation (`tests/test_evaluator_isolation.py`)

**New file.** Tests for `_current_run_span_id` ContextVar and `_filter_trace_to_subtree()`.

```
Tests to write:

# _filter_trace_to_subtree
- test_filter_returns_only_subtree: trace with root -> [run-1 -> [llm, retriever], run-2 -> [llm]]
    -> filter to run-1 span_id returns only run-1, llm, retriever spans
- test_filter_single_span: filter to a leaf span -> returns just that span
- test_filter_preserves_span_data: filtered spans have same attributes/inputs/outputs as originals
- test_filter_unknown_span_id: span_id not in trace -> returns empty span list

# ContextVar integration with SpanBaseEvaluator.get_trace()
- test_get_trace_without_contextvar: _current_run_span_id is None -> returns full trace (existing behavior)
- test_get_trace_with_contextvar: _current_run_span_id set to run-2 span_id
    -> returned trace only contains run-2's subtree
- test_contextvar_reset_after_run: after resetting token, get_trace returns full trace again
```

### Step 1.7 — Unit tests for execution loop (`tests/test_execution.py`)

**New file.** Tests for `_execute_run` and `_execute_case` without MLflow.

These tests mock out MLflow (no actual tracing) but exercise the two-phase orchestration logic.

```
Tests to write:

# _execute_run (Phase 2 — evaluators only, spans already committed)
- test_execute_run_success: task output + evaluators pass -> RunResult with output, assertions, no error
- test_execute_run_task_failure: task raised in Phase 1 -> RunResult with error set, empty assertions
- test_execute_run_evaluator_failure: evaluator raises -> RunResult with eval_failures populated
- test_execute_run_mixed: some evaluators pass, some fail -> correct assertion values
- test_execute_run_async_task: async task callable works correctly
- test_execute_run_sync_task: sync task callable works correctly

# _execute_case (two-phase orchestration)
- test_execute_case_repeat_1: single run, collects one RunResult
- test_execute_case_repeat_3: three runs, collects three RunResults, aggregation applied
- test_execute_case_uses_factory_per_run: factory called once per run (verify via spy)
- test_execute_case_sequential_runs: runs execute in order (run_index 0, 1, 2)
- test_execute_case_partial_failures: 1 of 3 runs fails, others succeed -> correct aggregation
- test_execute_case_phase1_then_phase2: verify all task spans are closed before evaluators run
    (mock mlflow.start_span to track open/close ordering)
- test_execute_case_contextvar_set_during_evaluation: verify _current_run_span_id is set
    to the correct run span ID when each run's evaluators execute
```

### Step 1.8 — Unit tests for DataFrame construction (`tests/test_dataframe.py`)

**New file.** Tests for the new DataFrame building functions.

```
Tests to write:

# Runs DataFrame
- test_runs_df_columns: has all expected columns (case_id, run_index, repeat_total, threshold, evaluator_name, etc.)
- test_runs_df_row_count: 2 cases, 3 runs each, 2 evaluators -> 12 rows
- test_runs_df_single_repeat: repeat=1 -> run_index always 0
- test_runs_df_input_key_format: input_key = "{hash}_{run_index}"

# Cases DataFrame  
- test_cases_df_columns: has aggregated columns (pass_rate, passed, aggregated_reason, etc.)
- test_cases_df_row_count: 2 cases, 2 evaluators -> 4 rows (one per case*evaluator)
- test_cases_df_pass_rate_correct: 2/3 runs pass for an evaluator -> pass_rate ≈ 0.667
- test_cases_df_single_repeat_matches_runs: when repeat=1, cases.passed matches runs.evaluator_result

# Summary DataFrame
- test_summary_df_one_row_per_case: 3 cases -> 3 rows
- test_summary_df_overall_passed: overall case pass/fail reflects threshold
```

### Step 1.9 — Integration tests (`tests/test_mlflow_integration.py` — extend existing)

**Extend existing file.** These require MLflow and are gated behind `RUN_MLFLOW_INTEGRATION_TESTS=1`.

```
Tests to add:

# --- Repeat integration (MLflow required) ---

# Basic repeat with MLflow
- test_repeat_with_mlflow_async: repeat=3, threshold=0.6, stateless task
    -> returns EvaluationOutput
    -> .runs has 3 * num_evaluators rows
    -> .cases has num_evaluators rows with pass_rate
    -> .summary has 1 row with overall pass/fail

- test_repeat_with_mlflow_sync: same as above but via evaluate_testset_with_mlflow_sync

# Task factory with MLflow  
- test_task_factory_with_mlflow: repeat=2, task_factory that returns a fresh dummy task
    -> factory called 2 times per case
    -> EvaluationOutput has correct structure

# Single repeat backward compatibility
- test_single_repeat_with_mlflow: repeat=1 (default), task= mode
    -> EvaluationOutput with .runs having 1 row per evaluator
    -> .cases matches .runs (no aggregation difference)
    -> verifies the new return type works for the non-repeat case

# Mixed repeat values across cases
- test_mixed_repeat_with_mlflow: case A has repeat=1, case B has repeat=3
    -> .runs has 1 + 3 = 4 sets of evaluator rows
    -> .cases has 2 sets of evaluator rows

# --- Challenging / realistic integration tests ---

# Deterministic failure pattern
- test_repeat_deterministic_failures: task that fails on specific run indices (via counter)
    -> verify exact pass_rate matches expected fraction
    -> verify failed run details appear in summary

# Evaluator disagreement across runs
- test_repeat_evaluator_disagreement: 2 evaluators, one always passes, one alternates
    -> per-evaluator pass rates differ
    -> case pass/fail based on ALL evaluators per run

# Task error on some runs
- test_repeat_with_task_errors: task raises on run index 1
    -> that run marked as failed
    -> other runs evaluated normally
    -> overall pass_rate accounts for error run

# CSV-loaded testset with repeat
- test_csv_repeat_with_mlflow: load testset_repeat.csv, run with MLflow
    -> cases with repeat=3 produce 3 runs each
    -> cases without repeat produce 1 run

# --- MLflow assessment naming ---

# Assessment naming convention
- test_assessment_naming_multi_run: repeat=3, evaluator "RegexInOutput"
    -> trace has assessments: run-0_RegexInOutput, run-1_RegexInOutput, run-2_RegexInOutput, agg_RegexInOutput
- test_assessment_naming_single_run: repeat=1 -> only run-0_RegexInOutput, no agg_ prefix
- test_aggregate_assessment_value: agg_ assessment value = (pass_rate >= threshold)

# --- Global defaults from MLFlowSettings ---

# Global repeat/threshold
- test_global_repeat_from_settings: MLFlowSettings(ragpill_repeat=3), case has repeat=None
    -> case runs 3 times (resolved from global)
- test_case_overrides_global: MLFlowSettings(ragpill_repeat=3), case has repeat=5
    -> case runs 5 times (per-case wins)
```

---

## Phase 2: Implementation

Implementation follows the test contracts. Each step makes a batch of tests pass.

### Step 2.1a — `src/ragpill/settings.py`: Add global `repeat` and `threshold` to `MLFlowSettings`

```python
# Add to MLFlowSettings:
ragpill_repeat: int = Field(default=1, ge=1, description="Default number of times to run each test case.")
ragpill_threshold: float = Field(default=1.0, ge=0.0, le=1.0, description="Default minimum fraction of runs that must pass. Env: MLFLOW_RAGPILL_REPEAT, MLFLOW_RAGPILL_THRESHOLD.")
```

### Step 2.1b — `src/ragpill/base.py`: Add `repeat` and `threshold` to `TestCaseMetadata` + `resolve_repeat()`

```python
# Add to TestCaseMetadata (nullable — None means "defer to global"):
repeat: int | None = Field(default=None, ge=1, description="Per-case override: number of times to run. None defers to MLFlowSettings.ragpill_repeat.")
threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Per-case override: minimum pass fraction. None defers to MLFlowSettings.ragpill_threshold.")
```

```python
# New helper at bottom of base.py:
def resolve_repeat(case_metadata: TestCaseMetadata | None, settings: "MLFlowSettings") -> tuple[int, float]:
    """Resolve effective repeat/threshold from per-case override or global default."""
    repeat = (case_metadata.repeat if (case_metadata and case_metadata.repeat is not None) else settings.ragpill_repeat)
    threshold = (case_metadata.threshold if (case_metadata and case_metadata.threshold is not None) else settings.ragpill_threshold)
    return repeat, threshold
```

**Makes pass:** Step 1.4 tests (TestCaseMetadata) + Step 1.4 tests (MLFlowSettings + resolve_repeat).

### Step 2.2 — `src/ragpill/types.py`: New result types

**New file.** Define:

```python
@dataclass
class RunResult:
    run_index: int
    input_key: str
    run_span_id: str          # captured during Phase 1, used to set ContextVar in Phase 2
    output: Any
    duration: float
    assertions: dict[str, EvaluationResult]
    evaluator_failures: list[EvaluatorFailure]
    error: Exception | None = None
    
    @property
    def all_passed(self) -> bool: ...

@dataclass
class AggregatedResult:
    passed: bool
    pass_rate: float
    threshold: float
    summary: str
    per_evaluator_pass_rates: dict[str, float]

@dataclass
class CaseResult:
    case_name: str
    inputs: Any
    metadata: TestCaseMetadata
    base_input_key: str
    trace_id: str
    run_results: list[RunResult]
    aggregated: AggregatedResult

@dataclass
class EvaluationOutput:
    runs: pd.DataFrame
    cases: pd.DataFrame
    case_results: list[CaseResult]
    
    @property
    def summary(self) -> pd.DataFrame: ...
```

Import `EvaluatorFailure` from pydantic_evals — verify available type:
```bash
uv run python -c "from pydantic_evals.reporting import ReportCase; print([f.name for f in ReportCase.__dataclass_fields__.values()])"
```
If `EvaluatorFailure` isn't importable, define a minimal dataclass locally.

**Makes pass:** Step 1.1 tests.

### Step 2.3 — `src/ragpill/mlflow_helper.py`: Aggregation function

Add `_aggregate_runs()` as a standalone, pure function:

```python
def _aggregate_runs(run_results: list[RunResult], threshold: float) -> AggregatedResult:
    # Logic from design doc: count all_passed runs, compute rates, build summary
    ...
```

**Makes pass:** Step 1.2 tests.

### Step 2.4 — `src/ragpill/mlflow_helper.py`: Task/factory validation

Add validation at the top of `evaluate_testset_with_mlflow`:

```python
async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    if task is not None and task_factory is not None:
        raise ValueError("Provide either 'task' or 'task_factory', not both.")
    if task is None and task_factory is None:
        raise ValueError("Provide either 'task' or 'task_factory'.")
    
    _factory: Callable[[], TaskType]
    if task is not None:
        _factory = lambda: task
    else:
        assert task_factory is not None
        _factory = task_factory
    ...
```

Update `evaluate_testset_with_mlflow_sync` signature to match.

**Makes pass:** Step 1.3 tests (validation subset).

### Step 2.5a — `src/ragpill/base.py`: Add ContextVar for run span isolation

```python
from contextvars import ContextVar

# Module-level, after imports:
_current_run_span_id: ContextVar[str | None] = ContextVar("_current_run_span_id", default=None)
```

This ContextVar is set during Phase 2 (evaluation) so that `SpanBaseEvaluator.get_trace()` can filter the returned trace to only the current run's subtree.

### Step 2.5b — `src/ragpill/mlflow_helper.py`: Two-phase execution loop

Implement `_evaluate_run` and `_execute_case` using a **two-phase design**:
- **Phase 1** (inside span context): Execute task for all N runs, capture outputs + span IDs. All spans close at end of outer `with` block, so MLflow traces are committed.
- **Phase 2** (after spans committed): For each run, set `_current_run_span_id` ContextVar, run evaluators, reset ContextVar. This ensures span-based evaluators can query committed traces and see only their run's subtree.

```python
from ragpill.base import _current_run_span_id

async def _evaluate_run(
    case: Case,
    output: Any,
    duration: float,
    evaluators: list[BaseEvaluator],
    run_index: int,
    input_key: str,
    run_span_id: str,
) -> RunResult:
    """Phase 2: Run evaluators for a single run (spans already committed)."""
    ctx = EvaluatorContext(
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=output,
        duration=duration,
        _span_tree=SpanTreeRecordingError("MLflow tracing used instead of OpenTelemetry SpanTree"),
        attributes={},
        metrics={},
    )
    
    assertions: dict[str, EvaluationResult] = {}
    evaluator_failures: list = []
    for evaluator in evaluators:
        try:
            result = await evaluator.evaluate(ctx)
            eval_name = evaluator.get_serialization_name()
            # Handle duplicate names
            if eval_name in assertions:
                n = sum(1 for k in assertions if k.startswith(eval_name))
                eval_name = f"{eval_name}_{n + 1}"
            assertions[eval_name] = EvaluationResult(
                name=eval_name,
                value=result.value,
                reason=result.reason,
                source=EvaluatorSpec(
                    name=evaluator.get_serialization_name(),
                    arguments={"evaluation_name": str(evaluator.evaluation_name)},
                ),
            )
        except Exception as e:
            evaluator_failures.append(...)
    
    return RunResult(
        run_index=run_index, input_key=input_key, run_span_id=run_span_id,
        output=output, duration=duration,
        assertions=assertions, evaluator_failures=evaluator_failures,
    )


async def _execute_case(
    case: Case,
    task_factory: Callable[[], TaskType],
    dataset_evaluators: list[BaseEvaluator],
    input_to_key: Callable[[Any], str],
    repeat: int,
    threshold: float,
) -> CaseResult:
    """Execute all runs for a case (two-phase) and aggregate.
    
    repeat/threshold are the resolved values (from per-case override or global default).
    """
    metadata = case.metadata
    assert isinstance(metadata, TestCaseMetadata)
    base_key = input_to_key(case.inputs)
    all_evaluators = list(case.evaluators) + list(dataset_evaluators)
    
    # ── Phase 1: Task execution (inside span context) ──────────────────────
    # All spans close at end of `with` block → traces committed to MLflow.
    run_outputs: list[tuple[Any, Exception | None]] = []
    run_span_ids: list[str] = []
    run_durations: list[float] = []
    
    with mlflow.start_span(name=(case.name or str(case.inputs))[:60], span_type=SpanType.TASK) as parent_span:
        parent_span.set_inputs(case.inputs)
        parent_span.set_attribute("input_key", base_key)
        parent_span.set_attribute("n_runs", repeat)
        parent_trace_id = parent_span.request_id
        
        for i in range(repeat):
            fresh_task = task_factory()
            input_key = f"{base_key}_{i}"
            try:
                with mlflow.start_span(name=f"run-{i}", span_type=SpanType.TASK) as run_span:
                    _run_span_id = run_span.span_id
                    run_span.set_attribute("run_index", i)
                    run_span.set_attribute("input_key", input_key)
                    run_span.set_inputs(case.inputs)
                    t0 = time.perf_counter()
                    if inspect.iscoroutinefunction(fresh_task):
                        output = await fresh_task(case.inputs)
                    else:
                        output = fresh_task(case.inputs)
                    duration = time.perf_counter() - t0
                    run_span.set_outputs(output)
                run_outputs.append((output, None))
            except Exception as e:
                run_outputs.append((None, e))
                duration = 0.0
            run_span_ids.append(_run_span_id)
            run_durations.append(duration)
    
    # ── Phase 2: Evaluation (all spans committed) ──────────────────────────
    # Set _current_run_span_id so SpanBaseEvaluator.get_trace() filters to this run's subtree.
    run_results: list[RunResult] = []
    for i, ((output, exc), run_span_id, duration) in enumerate(
        zip(run_outputs, run_span_ids, run_durations)
    ):
        input_key = f"{base_key}_{i}"
        
        if exc is not None:
            # Task failed in Phase 1 — mark all evaluators as failed
            assertions = {
                ev.get_serialization_name(): EvaluationResult(
                    name=ev.get_serialization_name(), value=False,
                    reason=f"Task execution failed: {exc}",
                    source=EvaluatorSpec(name="CODE", arguments={"evaluation_name": ev.evaluation_name}),
                )
                for ev in all_evaluators
            }
            run_results.append(RunResult(
                run_index=i, input_key=input_key, run_span_id=run_span_id,
                output=None, duration=0.0, assertions=assertions,
                evaluator_failures=[], error=exc,
            ))
            continue
        
        token = _current_run_span_id.set(run_span_id)
        try:
            run_result = await _evaluate_run(
                case, output, duration, all_evaluators, i, input_key, run_span_id,
            )
            run_results.append(run_result)
        finally:
            _current_run_span_id.reset(token)
    
    aggregated = _aggregate_runs(run_results, threshold)
    return CaseResult(
        case_name=case.name or str(case.inputs),
        inputs=case.inputs,
        metadata=metadata,
        base_input_key=base_key,
        trace_id=parent_trace_id,
        run_results=run_results,
        aggregated=aggregated,
    )
```

**Makes pass:** Step 1.5, Step 1.6, and Step 1.7 tests.

### Step 2.6 — `src/ragpill/mlflow_helper.py`: DataFrame construction

Replace `_create_evaluation_dataframe` with two new functions:

```python
def _create_runs_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """One row per (run x evaluator)."""
    rows = []
    for cr in case_results:
        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                rows.append({
                    "case_id": cr.base_input_key,
                    "case_name": cr.case_name,
                    "run_index": rr.run_index,
                    "repeat_total": cr.metadata.repeat,
                    "threshold": cr.metadata.threshold,
                    "inputs": str(cr.inputs),
                    "output": str(rr.output),
                    "evaluator_name": eval_name,
                    "evaluator_result": eval_result.value,
                    "evaluator_reason": eval_result.reason,
                    "evaluator_data": ...,  # from eval metadata
                    "expected": ...,
                    "attributes": ...,
                    "tags": ...,
                    "task_duration": rr.duration,
                    "source_type": ...,
                    "source_id": eval_result.source.name,
                    "input_key": rr.input_key,
                    "trace_id": cr.trace_id,
                })
            # Also handle evaluator_failures
    return pd.DataFrame(rows)


def _create_cases_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """One row per (case x evaluator), aggregated."""
    rows = []
    for cr in case_results:
        for eval_name, pass_rate in cr.aggregated.per_evaluator_pass_rates.items():
            rows.append({
                "case_id": cr.base_input_key,
                "case_name": cr.case_name,
                "repeat_total": cr.metadata.repeat,
                "threshold": cr.metadata.threshold,
                "inputs": str(cr.inputs),
                "evaluator_name": eval_name,
                "pass_rate": pass_rate,
                "passed": pass_rate >= cr.metadata.threshold,
                "aggregated_reason": ...,
                "expected": ...,
                "attributes": ...,
                "tags": ...,
                "avg_task_duration": ...,
                "trace_id": cr.trace_id,
            })
    return pd.DataFrame(rows)
```

**Makes pass:** Step 1.7 tests.

### Step 2.7 — `src/ragpill/mlflow_helper.py`: Main function rewrite

Rewrite `evaluate_testset_with_mlflow` to use the new orchestration, `resolve_repeat()` for global/per-case resolution, and structured assessment naming:

```python
from ragpill.base import resolve_repeat

async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    mlflow_settings = mlflow_settings or MLFlowSettings()
    # 1. Validate task/task_factory (from Step 2.4)
    # 2. Setup MLflow
    # 3. Fix evaluator global flags
    # 4. For each case:
    #      a. Resolve repeat/threshold via resolve_repeat(case.metadata, mlflow_settings)
    #      b. Call _execute_case (two-phase: spans committed, then evaluators run)
    #      c. Collect CaseResult (trace_id captured during Phase 1)
    # 5. Build runs and cases DataFrames
    # 6. Upload to MLflow (metrics, assessments, tags)
    # 7. End run
    # 8. Return EvaluationOutput
    
    case_results: list[CaseResult] = []
    for case in testset.cases:
        assert isinstance(case.metadata, TestCaseMetadata)
        repeat, threshold = resolve_repeat(case.metadata, mlflow_settings)
        all_evals = list(case.evaluators) + list(testset.evaluators)
        case_result = await _execute_case(case, _factory, all_evals, default_input_to_key, repeat, threshold)
        case_results.append(case_result)
    ...
```

**MLflow assessment naming convention:**

When uploading assessments to MLflow traces, use this naming scheme:
- Per-run: `run-{run_index}_{evaluator_name}` (e.g. `run-0_RegexInOutput`, `run-1_RegexInOutput`)
- Aggregate (only when repeat > 1): `agg_{evaluator_name}` (e.g. `agg_RegexInOutput`)
  - Value = `pass_rate >= threshold`
  - Reason = `"Aggregate: {passed}/{total} runs passed (threshold={threshold})"`

For repeat=1, only emit `run-0_{evaluator_name}` — no `agg_` prefix (redundant).

Functions to **remove** (no longer needed):
- `_get_input_key_trace_id_map` — we capture trace_id during Phase 1
- `_get_input_key_report_case_map` — we build CaseResult directly
- `_get_evaluation_id_eval_metadata_map` — metadata comes from CaseResult
- `_handle_task_failures` — handled in Phase 2 of `_execute_case`
- `_create_evaluation_dataframe` — replaced by `_create_runs_dataframe` + `_create_cases_dataframe`

Functions to **keep** (adapted):
- `_setup_mlflow_experiment` — unchanged
- `_delete_llm_judge_traces` — still needed for LLMJudge spans (they create their own traces)
- `_upload_mlflow` — rewritten to accept `EvaluationOutput` and `list[CaseResult]`, uses new assessment naming

**Makes pass:** Step 1.9 integration tests.

### Step 2.8 — `src/ragpill/csv/testset.py`: Parse repeat/threshold

Update `_parse_row_data` and `_create_case_from_rows`:

1. Add `repeat` and `threshold` to `standard_columns` set so they aren't treated as evaluator attributes.

2. In `_create_case_from_rows`, extract `repeat`/`threshold` from rows:
   ```python
   # Extract repeat/threshold (must be consistent across rows for same question)
   # Use None when absent — resolve_repeat() will apply global defaults from MLFlowSettings
   repeat_values = {int(r["repeat"]) if r.get("repeat") else None for r in rows}
   threshold_values = {float(r["threshold"]) if r.get("threshold") else None for r in rows}
   if len(repeat_values) > 1:
       raise ValueError(f"Inconsistent 'repeat' values for question '{question}': {repeat_values}")
   if len(threshold_values) > 1:
       raise ValueError(f"Inconsistent 'threshold' values for question '{question}': {threshold_values}")
   repeat = repeat_values.pop()
   threshold = threshold_values.pop()
   ```

3. Pass to `TestCaseMetadata`:
   ```python
   metadata=TestCaseMetadata(attributes=case_attributes, tags=case_tags, repeat=repeat, threshold=threshold)
   ```

4. In `load_testset`, add `repeat` and `threshold` to `standard_columns`.

**Makes pass:** Step 1.5 CSV tests.

### Step 2.9 — `src/ragpill/evaluators.py`: SpanBaseEvaluator adaptation + trace subtree filtering

Two changes: (a) add `_filter_trace_to_subtree()`, and (b) integrate `_current_run_span_id` ContextVar into `get_trace()`.

**a) Add `_filter_trace_to_subtree()`:**

```python
from copy import copy

def _filter_trace_to_subtree(trace: Trace, root_span_id: str) -> Trace:
    """Return a copy of the trace containing only the subtree rooted at root_span_id.
    
    This ensures span-based evaluators only see spans from their specific run,
    not spans from other runs in the same case trace.
    """
    all_spans = trace.data.spans
    included: set[str] = set()
    queue = [root_span_id]
    while queue:
        current = queue.pop()
        included.add(current)
        for span in all_spans:
            if span.parent_id == current:
                queue.append(span.span_id)
    filtered_data = copy(trace.data)
    filtered_data.spans = [s for s in all_spans if s.span_id in included]
    return Trace(info=trace.info, data=filtered_data)
```

**b) Update `get_trace()` — search any span for `input_key`, then filter to run subtree:**

```python
from ragpill.base import _current_run_span_id

def get_trace(self, inputs: Any) -> Trace:
    target_key = self.inputs_to_key_function(inputs)
    traces = mlflow.search_traces(...)
    matching = []
    for t in traces:
        for span in (t.data.spans or []):
            if span.attributes.get("input_key", "") == target_key:
                matching.append(t)
                break
    if not matching:
        raise ValueError(f"No trace found for input {inputs}.")
    
    trace = matching[0]
    
    # If we're inside a multi-run evaluation, filter to only the current run's subtree.
    # This prevents evaluators from accidentally inspecting spans from other runs.
    run_span_id = _current_run_span_id.get()
    if run_span_id is not None:
        trace = _filter_trace_to_subtree(trace, run_span_id)
    
    return trace
```

**Why this matters:** Without filtering, a span-based evaluator (e.g. one that checks retriever spans) could see spans from run-0 when evaluating run-2. The ContextVar is set per-run during Phase 2 of `_execute_case`, ensuring each evaluator invocation only sees its own run's spans.

**Makes pass:** Step 1.6 tests (evaluator isolation).

### Step 2.10 — `src/ragpill/__init__.py`: Update exports

Add new public types to `__all__`:

```python
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult

__all__ = [
    # ... existing ...
    "AggregatedResult",
    "CaseResult", 
    "EvaluationOutput",
    "RunResult",
]
```

### Step 2.11 — Type checking and linting

```bash
uv run basedpyright src/ragpill/
uv run ruff check src/ragpill/ tests/
uv run pytest tests/ -x
```

Fix any type errors, lint issues, or test failures.

---

## Phase 3: Documentation

### Step 3.1 — Docstrings (all modified/new public API)

Update docstrings in:

| Symbol | File | What to update |
|--------|------|----------------|
| `evaluate_testset_with_mlflow()` | `mlflow_helper.py` | New signature (task/task_factory), new return type, examples |
| `evaluate_testset_with_mlflow_sync()` | `mlflow_helper.py` | Mirror async docstring updates |
| `MLFlowSettings` | `settings.py` | Document `ragpill_repeat` and `ragpill_threshold` global defaults |
| `TestCaseMetadata` | `base.py` | Document `repeat` and `threshold` as nullable per-case overrides |
| `resolve_repeat()` | `base.py` | Document merging logic |
| `EvaluationOutput` | `types.py` | Document `.runs`, `.cases`, `.summary`, `.case_results` |
| `RunResult` | `types.py` | All fields, `.all_passed` property |
| `AggregatedResult` | `types.py` | All fields, threshold semantics |
| `CaseResult` | `types.py` | All fields |
| `_aggregate_runs()` | `mlflow_helper.py` | Internal but document algorithm |
| `load_testset()` | `csv/testset.py` | Mention repeat/threshold columns |

### Step 3.2 — New guide: `docs/guide/repeated-runs.md`

**Content outline** (from design doc):
1. **Why Repeat?** — LLM stochasticity, statistical confidence
2. **Quick Start** — minimal example with `task=` and `repeat=3, threshold=0.8`
3. **Stateful Task (task_factory)** — example with agent that has memory
4. **When do I need a factory?** — decision table
5. **Threshold Semantics** — how pass/fail is decided, edge cases
6. **Reading the Results** — `.runs`, `.cases`, `.summary` examples with sample output
7. **MLflow Trace Structure** — ASCII diagram of hierarchical spans
8. **CSV Integration** — repeat/threshold columns
9. **Failure Explanations** — example of aggregated failure output

### Step 3.3 — New how-to: `docs/how-to/task-factory.md`

**Content outline** (from design doc):
1. Problem statement (stateful agent + repeat = contamination)
2. Before/After code examples
3. pydantic-ai Agent + message_history example
4. Resource cleanup patterns

### Step 3.4 — Update existing docs

| File | Changes |
|------|---------|
| `docs/getting-started/quickstart.md` | Add "Repeated Runs" section with 5-line example |
| `docs/guide/overview.md` | Mention repeat/threshold in the evaluation flow description |
| `docs/guide/testsets.md` | Document `repeat`/`threshold` as TestCaseMetadata fields |
| `docs/guide/csv-adapter.md` | Add `repeat`/`threshold` columns to CSV format docs, add example rows |
| `docs/tutorials/full.md` | Add section showing repeat with the existing tutorial agent |
| `docs/api/mlflow.md` | Auto-generated from docstrings — verify renders correctly |
| `docs/api/base.md` | Auto-generated from docstrings — verify renders correctly |

### Step 3.5 — Update `mkdocs.yml`

Add new pages to the navigation:

```yaml
nav:
  - Guide:
    - ...
    - Repeated Runs: guide/repeated-runs.md
  - How-To:
    - ...
    - Task Factory: how-to/task-factory.md
```

### Step 3.6 — Update notebooks (if any exist as `.ipynb`)

Check `docs/tutorials/` for notebook files. If `full.md` references a notebook, add a repeat section to it.

---

## Execution Order Summary

```
Phase 1: Tests
  1.1  test_types.py               (new) — RunResult, AggregatedResult, EvaluationOutput
  1.2  test_aggregation.py         (new) — _aggregate_runs()
  1.3  test_task_factory.py        (new) — validation, factory invocation, isolation
  1.4  test_base.py, test_settings.py (extend/new) — TestCaseMetadata, MLFlowSettings, resolve_repeat()
  1.5  test_testset_csv.py         (extend) — CSV parsing, testset_repeat.csv
  1.6  test_evaluator_isolation.py (new) — _filter_trace_to_subtree, ContextVar integration
  1.7  test_execution.py           (new) — two-phase _execute_case, _evaluate_run
  1.8  test_dataframe.py           (new) — runs/cases DataFrame construction
  1.9  test_mlflow_integration.py  (extend) — end-to-end with MLflow, assessment naming, global defaults

Phase 2: Implementation
  2.1a settings.py             — ragpill_repeat/ragpill_threshold on MLFlowSettings (global defaults)
  2.1b base.py                 — repeat/threshold on TestCaseMetadata (nullable) + resolve_repeat()
  2.2  types.py                (new) — RunResult (with run_span_id), AggregatedResult, CaseResult, EvaluationOutput
  2.3  mlflow_helper.py        — _aggregate_runs()
  2.4  mlflow_helper.py        — task/task_factory validation
  2.5a base.py                 — _current_run_span_id ContextVar
  2.5b mlflow_helper.py        — two-phase _execute_case + _evaluate_run
  2.6  mlflow_helper.py        — _create_runs_dataframe, _create_cases_dataframe
  2.7  mlflow_helper.py        — main function rewrite (resolve_repeat, assessment naming)
  2.8  csv/testset.py          — parse repeat/threshold (None when absent)
  2.9  evaluators.py           — _filter_trace_to_subtree + ContextVar in get_trace()
  2.10 __init__.py             — export new types
  2.11 type check + lint + test

Phase 3: Documentation
  3.1  Docstrings              — all modified/new API (incl. MLFlowSettings, resolve_repeat)
  3.2  guide/repeated-runs.md  (new)
  3.3  how-to/task-factory.md  (new)
  3.4  Update existing docs    — quickstart, overview, testsets, csv-adapter, full tutorial
  3.5  mkdocs.yml              — add new pages to nav
  3.6  Notebooks               — if applicable
```

---

## Risk Notes

1. **`EvaluatorContext._span_tree`**: We pass a `SpanTreeRecordingError` sentinel since we don't capture OpenTelemetry spans. Evaluators that access `ctx.span_tree` will get an error. Currently, no ragpill evaluators use this (they use MLflow traces directly via `SpanBaseEvaluator`). If a user's custom evaluator does, they'll get a clear error message.

2. **`EvaluatorContext` is a pydantic_evals internal**: Its constructor signature could change between versions. Pin `pydantic-ai>=1.39.1` and add a comment noting the coupling.

3. **LLMJudge trace cleanup**: With our own loop, LLMJudge still creates separate traces (it starts its own MLflow span with `ragpill_is_judge_trace=True`). `_delete_llm_judge_traces` is still needed.

4. **Concurrency**: Runs within a case are sequential. Cases are sequential in the initial implementation (can add `asyncio.Semaphore`-based concurrency later). This is simpler and sufficient.

5. **`evaluate_testset_with_mlflow_sync`**: Must be updated to match the new async signature and return `EvaluationOutput`.

6. **ContextVar thread safety**: `_current_run_span_id` uses `contextvars.ContextVar`, which is async-safe (each task gets its own context). Since runs within a case are sequential (not concurrent), there's no risk of one run's ContextVar leaking into another. If we later add concurrency (asyncio.Semaphore), each `asyncio.Task` gets its own copy of the ContextVar automatically.

7. **Two-phase timing**: Phase 2 evaluators query MLflow for committed traces. There may be a small delay between span closure and trace availability in MLflow. If this becomes an issue, add a brief `await asyncio.sleep(0.1)` between phases, but this is unlikely with the local MLflow server.
