# Multi-Run Evaluation Plan

## Context

Currently each test case runs exactly once via `testset.evaluate()`. The goal is to run
each case N times to test LLM pipeline consistency, with configurable pass-rate thresholds
per question. The existing `testset.evaluate()` + `_mlflow_runnable_wrapper` approach is
**replaced entirely** by a new unified execution loop.

---

## MLflow Trace Structure (always)

```
Trace: "<question text[:60]>" (root span, SpanType.TASK)
  Attributes: input_key, n_runs
  ├── run-1  (child span, SpanType.TASK)  ← attribute: run_index=1
  │     └── [task's sub-spans: llm, retriever, reranker, …]
  ├── run-N  (child span, SpanType.TASK)  ← attribute: run_index=N
  │     └── [task's sub-spans]
  Assessments (all at trace level):
    run-1_<evaluator_name>  →  per-run bool
    run-N_<evaluator_name>  →  per-run bool
    agg_<evaluator_name>    →  pass_rate >= threshold  (only when n_runs > 1)
```

For n_runs=1: one "run-1" child span; no aggregate assessment (redundant).

---

## DataFrame Schema (always uniform)

| Column | n_runs=1 | n_runs>1 per-run | n_runs>1 aggregate |
|--------|----------|-------------------|--------------------|
| `run_index` | 1 | 1..N | None |
| `pass_rate` | None | None | 0.0..1.0 |
| `n_runs` | 1 | N | N |
| `evaluator_result` | bool | bool | bool (pass_rate>=threshold) |
| *(all existing columns retained)* | | | |

Aggregate rows (run_index=None) only emitted when n_runs > 1.

---

## Files to Modify

### 1. `src/llm_eval/settings.py`
Add two fields to `MLFlowSettings`:
```python
n_runs: int = Field(1, description="Number of task runs per test case.")
pass_rate_threshold: float = Field(1.0, description="Fraction of runs that must pass (0–1).")
```
Env vars: `EVAL_MLFLOW_N_RUNS`, `EVAL_MLFLOW_PASS_RATE_THRESHOLD`.

---

### 2. `src/llm_eval/base.py`

**a) ContextVar** (module level, after imports):
```python
from contextvars import ContextVar
_current_run_span_id: ContextVar[str | None] = ContextVar("_current_run_span_id", default=None)
```

**b) New fields on `TestCaseMetadata`**:
```python
n_runs: int | None = Field(None, description="Per-case override for n_runs.")
pass_rate_threshold: float | None = Field(None, description="Per-case threshold (0–1).")
```

**c) New helper at bottom of file**:
```python
def resolve_n_runs(case_metadata: TestCaseMetadata | None, settings: "MLFlowSettings") -> tuple[int, float]:
    n = case_metadata.n_runs if (case_metadata and case_metadata.n_runs is not None) else settings.n_runs
    t = case_metadata.pass_rate_threshold if (case_metadata and case_metadata.pass_rate_threshold is not None) else settings.pass_rate_threshold
    return n, t
```

---

### 3. `src/llm_eval/evaluators.py`

**a) Import** `_current_run_span_id` from `llm_eval.base`.

**b) Add `_filter_trace_to_subtree()`**:
```python
def _filter_trace_to_subtree(trace: Trace, root_span_id: str) -> Trace:
    from copy import copy
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

**c) Modify `SpanBaseEvaluator.get_trace()`** — after finding `traces[0]`, add:
```python
run_span_id = _current_run_span_id.get()
if run_span_id is not None:
    return _filter_trace_to_subtree(traces[0], run_span_id)
return traces[0]
```

---

### 4. `src/llm_eval/csv/testset.py`

In `_create_case_from_rows()`, after `case_tags, case_attributes = _find_common_tags_and_attributes(...)`,
extract special fields before passing to `TestCaseMetadata`:
```python
n_runs_val = int(case_attributes.pop("n_runs")) if "n_runs" in case_attributes else None
threshold_val = float(case_attributes.pop("pass_rate_threshold")) if "pass_rate_threshold" in case_attributes else None
return Case(
    inputs=question, evaluators=evaluators,
    metadata=TestCaseMetadata(
        attributes=case_attributes, tags=case_tags,
        n_runs=n_runs_val, pass_rate_threshold=threshold_val,
    )
)
```

---

### 5. `src/llm_eval/mlflow_helper.py` — full replacement of evaluation path

**Remove** (no longer needed):
- `_mlflow_runnable_wrapper()`
- `_get_input_key_report_case_map()` (replaced by direct case tracking)
- `_handle_task_failures()` (folded into `_execute_multirun_case`)
- `_create_evaluation_dataframe()` (replaced by `_build_multirun_rows()`)
- `_upload_mlflow()` (replaced by `_upload_mlflow_multirun()`)

**Add `RunResult` dataclass**:
```python
@dataclass
class RunResult:
    run_index: int           # 1-based
    run_span_id: str
    output: Any
    task_duration: float
    assertions: dict[str, EvaluationResult]   # key = evaluator display name
    evaluator_failures: list                   # EvaluatorFailure-like objects
    success: bool = True
    error_message: str = ""
```

**Add `_execute_multirun_case()`** — two-phase design:

*Phase 1* (inside span context): create "question" root span → N "run-i" child spans →
execute task in each → collect outputs + span IDs. All spans close at end of outer `with`.
MLflow traces are committed only after all spans close.

*Phase 2* (after all spans closed): for each run, set `_current_run_span_id` ContextVar,
construct `EvaluatorContext` manually, call `evaluator.evaluate(ctx)`, reset ContextVar.

```python
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.otel._errors import SpanTreeRecordingError
import time, inspect

async def _execute_multirun_case(
    case, task: TaskType, n_runs: int,
    all_evaluators: list[BaseEvaluator],
    input_to_key=default_input_to_key,
) -> tuple[list[RunResult], str]:
    input_key = input_to_key(case.inputs)
    span_name = (case.name or str(case.inputs))[:60]

    # ── Phase 1: task execution (inside span context) ──────────────────────
    run_outputs, run_span_ids, run_durations = [], [], []
    with mlflow.start_span(span_name, span_type=SpanType.TASK) as parent_span:
        parent_span.set_inputs(case.inputs)
        parent_span.set_attribute("input_key", input_key)
        parent_span.set_attribute("n_runs", n_runs)
        parent_trace_id = parent_span.request_id   # capture trace_id before exit

        for i in range(1, n_runs + 1):
            try:
                with mlflow.start_span(f"run-{i}", span_type=SpanType.TASK) as run_span:
                    _run_span_id = run_span.span_id    # capture inside context
                    run_span.set_attribute("run_index", i)
                    t0 = time.perf_counter()
                    if inspect.iscoroutinefunction(task):
                        output = await task(case.inputs)
                    else:
                        output = task(case.inputs)
                    duration = time.perf_counter() - t0
                    run_span.set_outputs(output)
                run_outputs.append((output, None))
            except Exception as e:
                run_outputs.append((None, e))
                duration = 0.0
            run_span_ids.append(_run_span_id)
            run_durations.append(duration)

    # ── Phase 2: evaluation (all spans committed) ──────────────────────────
    _err = SpanTreeRecordingError("not available in multi-run path")
    run_results = []
    for i, ((output, exc), run_span_id, duration) in enumerate(
        zip(run_outputs, run_span_ids, run_durations), 1
    ):
        if exc is not None:
            # Task failed — mark all evaluators as failed
            assertions = {
                ev.get_serialization_name(): EvaluationResult(
                    name=ev.get_serialization_name(), value=False,
                    reason=f"Task execution failed: {exc}",
                    source=EvaluatorSpec(name="CODE", arguments={"evaluation_name": ev.evaluation_name}),
                )
                for ev in all_evaluators
            }
            run_results.append(RunResult(i, run_span_id, str(exc), 0.0, assertions, [], success=False, error_message=str(exc)))
            continue

        token = _current_run_span_id.set(run_span_id)
        try:
            ctx = EvaluatorContext(
                name=case.name, inputs=case.inputs, metadata=case.metadata,
                expected_output=case.expected_output, output=output,
                duration=duration, _span_tree=_err, attributes={}, metrics={},
            )
            assertions, failures = {}, []
            for ev in all_evaluators:
                try:
                    result = await ev.evaluate(ctx)
                    assertions[ev.get_serialization_name()] = EvaluationResult(
                        name=ev.get_serialization_name(), value=result.value,
                        reason=result.reason,
                        source=EvaluatorSpec(name=type(ev).__name__,
                                             arguments={"evaluation_name": ev.evaluation_name}),
                    )
                except Exception as e:
                    failures.append(e)
            run_results.append(RunResult(i, run_span_id, output, duration, assertions, failures))
        finally:
            _current_run_span_id.reset(token)

    return run_results, parent_trace_id
```

> **Verify before implementing**: `EvaluationReason` returned by `ev.evaluate()` has `.value`
> and `.reason` attributes. Check whether `EvaluationResult` vs `EvaluationReason` is the right
> type to use in assertions dict — look at how `_handle_task_failures` in the current code builds it.
> Also confirm `ev.get_serialization_name()` is the correct method (used in existing `_handle_task_failures`).

**Add `_build_multirun_rows()`**:
```python
def _build_multirun_rows(
    case, run_results: list[RunResult], threshold: float,
    trace_id: str, eval_metadata_map: dict,
) -> list[dict]:
    rows = []
    input_key = default_input_to_key(case.inputs)
    all_eval_names = {name for rr in run_results for name in rr.assertions}

    for eval_name in all_eval_names:
        eval_key = f"{input_key}_{...}"   # match how eval_metadata_map is keyed
        evaluator_metadata = eval_metadata_map.get(eval_key)
        merged = merge_metadata(case.metadata, evaluator_metadata) if evaluator_metadata else None

        per_run_values = []
        for rr in run_results:
            er = rr.assertions.get(eval_name)
            val = bool(er.value) if er else False
            per_run_values.append(val)
            rows.append({
                # all existing columns:
                "inputs": str(case.inputs), "output": str(rr.output),
                "evaluator_result": val,
                "evaluator_reason": er.reason if er else "not executed",
                "expected": merged.expected if merged else True,
                "mandatory": merged.mandatory if merged else True,
                "evaluator_data": merged.other_evaluator_data if merged else "",
                "attributes": ta.dump_json(merged.attributes) if merged else b"{}",
                "tags": merged.tags if merged else set(),
                "task_duration": rr.task_duration,
                "evaluator_name": eval_name,
                "case_name": case.name, "case_id": input_key,
                "source_type": "LLM_JUDGE" if "LLMJudge" in (er.source.name if er else "") else "CODE",
                "source_id": er.source.name if er else "CODE",
                "input_key": input_key, "trace_id": trace_id,
                # new columns:
                "run_index": rr.run_index, "pass_rate": None, "n_runs": len(run_results),
            })

        if len(run_results) > 1:
            pass_rate = sum(per_run_values) / len(per_run_values)
            rows.append({
                **rows[-1],   # copy last row as template
                "evaluator_result": pass_rate >= threshold,
                "evaluator_reason": f"Aggregate: {sum(per_run_values)}/{len(per_run_values)} runs passed (threshold={threshold})",
                "run_index": None, "pass_rate": pass_rate,
            })
    return rows
```

**Add `_upload_mlflow_multirun()`** (replaces `_upload_mlflow`):
- `mlflow.log_table(df, "evaluation_results.json")`
- Log model params
- Compute metrics on aggregate rows only (run_index is None), or all rows for n_runs=1:
  `df_agg = df[df["run_index"].isna()]` — then same `overall_accuracy` etc. logic as today
- Assessments:
  - Per-run rows: `assessment.name = f"run-{row.run_index}_{row.evaluator_name}"`
  - Aggregate rows: `assessment.name = f"agg_{row.evaluator_name}"`
  - All logged to `trace_id` (parent question trace)
- Trace tags: set from aggregate rows' metadata (same logic as current `_upload_mlflow`)

**Modify `evaluate_testset_with_mlflow()`**:
```python
async def evaluate_testset_with_mlflow(
    testset, task, mlflow_settings=None, model_params=None,
) -> pd.DataFrame:
    mlflow_settings = mlflow_settings or MLFlowSettings()
    _setup_mlflow_experiment(mlflow_settings)
    _fix_evaluator_global_flag(testset)

    case_run_results: dict[str, tuple] = {}  # input_key -> (case, run_results, threshold)
    for case in testset.cases:
        assert isinstance(case.metadata, TestCaseMetadata)
        n_runs, threshold = resolve_n_runs(case.metadata, mlflow_settings)
        all_evals = list(case.evaluators) + list(testset.evaluators)
        input_key = default_input_to_key(case.inputs)
        run_results, _ = await _execute_multirun_case(case, task, n_runs, all_evals)
        case_run_results[input_key] = (case, run_results, threshold)

    experiment, latest_run_id = _delete_llm_judge_traces(mlflow_settings)
    input_key_trace_map = _get_input_key_trace_id_map(experiment, latest_run_id)
    eval_metadata_map = _get_evaluation_id_eval_metadata_map(testset)

    all_rows = []
    for input_key, (case, run_results, threshold) in case_run_results.items():
        trace_id = input_key_trace_map[input_key]
        all_rows.extend(_build_multirun_rows(case, run_results, threshold, trace_id, eval_metadata_map))

    df = pd.DataFrame(all_rows)
    _upload_mlflow_multirun(df, model_params)
    mlflow.end_run()
    return df
```

`evaluate_testset_with_mlflow_sync()` — no signature change needed (passes through to async version).

---

## Removed / Deleted Code in `mlflow_helper.py`

| Function | Action |
|----------|--------|
| `_mlflow_runnable_wrapper()` | Delete |
| `_get_input_key_report_case_map()` | Delete |
| `_handle_task_failures()` | Delete (folded into `_execute_multirun_case`) |
| `_create_evaluation_dataframe()` | Delete (replaced by `_build_multirun_rows`) |
| `_upload_mlflow()` | Delete (replaced by `_upload_mlflow_multirun`) |

`_get_input_key_trace_id_map()`, `_get_evaluation_id_eval_metadata_map()`,
`_delete_llm_judge_traces()`, `_setup_mlflow_experiment()` — **kept unchanged**.

---

## Verification

1. **Unit** — `_filter_trace_to_subtree()`: construct synthetic Trace with known spans;
   verify only the target subtree is returned.
2. **Unit** — `resolve_n_runs()`: all None combinations resolve correctly.
3. **Integration** — run existing `tests/mlflow_integration.py` with n_runs=1 (default);
   confirm existing behaviour preserved (DataFrame shape, MLflow metrics, trace structure).
4. **Integration (new test)** — create a testset, call with `n_runs=3`, `pass_rate_threshold=0.67`:
   - MLflow shows one trace per question with 3 run child spans
   - DataFrame has 3 per-run rows + 1 aggregate row per evaluator
   - 2/3 runs passing → `evaluator_result=True` on aggregate row
   - `run-1_eval_name`, `run-2_eval_name`, `run-3_eval_name`, `agg_eval_name` assessments on trace
5. **CSV** — add `n_runs` column to test CSV; verify `TestCaseMetadata.n_runs` is populated
   and overrides the global setting.
