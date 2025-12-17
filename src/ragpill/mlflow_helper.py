import inspect
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Experiment, Feedback, SpanType, Trace
from pydantic import TypeAdapter

from ragpill.base import (
    BaseEvaluator,
    CaseMetadataT,
    EvaluatorMetadata,
    TestCaseMetadata,
    _current_run_span_id,  # pyright: ignore[reportPrivateUsage]
    default_input_to_key,
    merge_metadata,
    resolve_repeat,
)
from ragpill.eval_types import Case, Dataset, EvaluationResult, EvaluatorContext, EvaluatorSource
from ragpill.settings import MLFlowSettings
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    EvaluatorFailureInfo,
    RunResult,
)
from ragpill.utils import _fix_evaluator_global_flag  # pyright: ignore[reportPrivateUsage]

TaskType = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]


# ---------------------------------------------------------------------------
# MLflow setup & cleanup
# ---------------------------------------------------------------------------


def _setup_mlflow_experiment(mlflow_settings: MLFlowSettings) -> None:
    """Setup mlflow experiment with given settings."""
    mlflow.set_tracking_uri(mlflow_settings.ragpill_tracking_uri)
    mlflow.set_experiment(mlflow_settings.ragpill_experiment_name)  # pyright: ignore[reportUnknownMemberType]
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.start_run(description=mlflow_settings.ragpill_run_description)


def _delete_llm_judge_traces(mlflow_settings: MLFlowSettings) -> tuple[Experiment, str]:
    """Remove LLMJudge evaluation-only traces that clutter the MLflow tracing UI.

    Returns:
        Tuple of (experiment, latest_run_id).
    """
    experiment = mlflow.get_experiment_by_name(mlflow_settings.ragpill_experiment_name)
    assert experiment is not None, f"Experiment '{mlflow_settings.ragpill_experiment_name}' not found."
    experiment_id: str = str(experiment.experiment_id)  # pyright: ignore[reportUnknownArgumentType]
    df: Any = mlflow.search_runs([experiment_id], order_by=["start_time DESC"])
    latest_run_id: str = str(df.iloc[0]["run_id"])

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_settings.ragpill_tracking_uri)
    traces: list[Trace] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
        locations=[experiment_id], run_id=latest_run_id, return_type="list"
    )
    delete_traces: list[str] = []
    for trace in traces:
        root = trace.data._get_root_span()  # pyright: ignore[reportPrivateUsage]
        if root and root.attributes.get("ragpill_is_judge_trace"):
            delete_traces.append(trace.info.trace_id)
    if delete_traces:
        client.delete_traces(experiment_id=experiment_id, trace_ids=delete_traces)
    return experiment, latest_run_id


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_runs(run_results: list[RunResult], threshold: float) -> AggregatedResult:
    """Aggregate multiple RunResults into a single verdict.

    A run counts as "passed" when ``run_result.all_passed`` is True.
    The case passes when ``pass_rate >= threshold``.

    Args:
        run_results: List of RunResult objects for a single case.
        threshold: Minimum fraction of runs that must pass (0.0 to 1.0).

    Returns:
        AggregatedResult with pass/fail verdict, pass_rate, and per-evaluator rates.
    """
    total = len(run_results)
    passed_count = sum(1 for r in run_results if r.all_passed)
    pass_rate = passed_count / total if total > 0 else 0.0
    passed = pass_rate >= threshold

    # Per-evaluator pass rates
    evaluator_names: set[str] = set()
    for r in run_results:
        evaluator_names.update(r.assertions.keys())

    per_evaluator_pass_rates: dict[str, float] = {}
    for eval_name in sorted(evaluator_names):
        eval_passed = sum(1 for r in run_results if eval_name in r.assertions and r.assertions[eval_name].value is True)
        per_evaluator_pass_rates[eval_name] = eval_passed / total if total > 0 else 0.0

    # Build summary text
    if passed:
        summary = f"{passed_count}/{total} runs passed (threshold={threshold})"
    else:
        failed_details: list[str] = []
        for r in run_results:
            if not r.all_passed:
                if r.error:
                    failed_details.append(f"run-{r.run_index}: task error: {r.error}")
                else:
                    failed_evals = [
                        f"{name}: {res.reason}" for name, res in r.assertions.items() if res.value is not True
                    ]
                    failed_details.append(f"run-{r.run_index}: {'; '.join(failed_evals)}")
        summary = f"{passed_count}/{total} runs passed (threshold={threshold}). Failed: {'; '.join(failed_details)}"

    return AggregatedResult(
        passed=passed,
        pass_rate=pass_rate,
        threshold=threshold,
        summary=summary,
        per_evaluator_pass_rates=per_evaluator_pass_rates,
    )


# ---------------------------------------------------------------------------
# Two-phase execution
# ---------------------------------------------------------------------------


async def _evaluate_run(
    case: Case[Any, Any, Any],
    output: Any,
    duration: float,
    evaluators: list[BaseEvaluator],
    run_index: int,
    input_key: str,
    run_span_id: str,
) -> RunResult:
    """Phase 2: Run evaluators for a single run (spans already committed).

    Args:
        case: The test case being evaluated.
        output: The task output from Phase 1.
        duration: Wall-clock seconds the task took.
        evaluators: All evaluators to run (case-level + dataset-level).
        run_index: Zero-based index of this run.
        input_key: Unique key for this run.
        run_span_id: MLflow span ID from Phase 1.

    Returns:
        RunResult with assertions and any evaluator failures.
    """
    ctx = EvaluatorContext(
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=output,
        duration=duration,
        attributes={},
        metrics={},
    )

    assertions: dict[str, EvaluationResult] = {}
    evaluator_failures: list[EvaluatorFailureInfo] = []

    for evaluator in evaluators:
        eval_name = evaluator.get_serialization_name()
        try:
            result = await evaluator.evaluate(ctx)
            # Handle duplicate evaluator names
            if eval_name in assertions:
                n = sum(1 for k in assertions if k.startswith(eval_name))
                eval_name = f"{eval_name}_{n + 1}"
            assertions[eval_name] = EvaluationResult(
                name=eval_name,
                value=result.value,
                reason=result.reason,
                source=EvaluatorSource(
                    name=evaluator.get_serialization_name(),
                    arguments={"evaluation_name": str(evaluator.evaluation_name)},
                ),
            )
        except Exception as e:
            evaluator_failures.append(
                EvaluatorFailureInfo(
                    name=eval_name,
                    error_message=str(e),
                    error_stacktrace=traceback.format_exc(),
                )
            )

    return RunResult(
        run_index=run_index,
        input_key=input_key,
        run_span_id=run_span_id,
        output=output,
        duration=duration,
        assertions=assertions,
        evaluator_failures=evaluator_failures,
    )


async def _execute_case(
    case: Case[Any, Any, Any],
    task_factory: Callable[[], TaskType],
    all_evaluators: list[BaseEvaluator],
    input_to_key: Callable[[Any], str],
    repeat: int,
    threshold: float,
) -> CaseResult:
    """Execute all runs for a case using two-phase design, then aggregate.

    Phase 1 (inside span context): Execute the task N times, capturing outputs + span IDs.
    All spans close at end of outer ``with`` block, so MLflow traces are committed.

    Phase 2 (after spans committed): For each run, set ``_current_run_span_id`` ContextVar,
    run evaluators, reset ContextVar. Span-based evaluators can query committed traces
    and see only their run's subtree.

    Args:
        case: The test case.
        task_factory: Callable that returns a fresh task callable per invocation.
        all_evaluators: All evaluators (case-level + dataset-level).
        input_to_key: Function to hash inputs into a string key.
        repeat: Number of times to run this case.
        threshold: Minimum pass fraction for this case.

    Returns:
        CaseResult with all run results and aggregated verdict.
    """
    metadata = case.metadata
    assert isinstance(metadata, TestCaseMetadata)
    base_key = input_to_key(case.inputs)

    # ── Phase 1: Task execution (inside span context) ──────────────────────
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
            _run_span_id = ""
            duration = 0.0
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
    run_results: list[RunResult] = []
    for i, ((output, exc), run_span_id, duration) in enumerate(zip(run_outputs, run_span_ids, run_durations)):
        input_key = f"{base_key}_{i}"

        if exc is not None:
            # Task failed in Phase 1 — mark all evaluators as failed
            assertions: dict[str, EvaluationResult] = {}
            for ev in all_evaluators:
                ev_name = ev.get_serialization_name()
                if ev_name in assertions:
                    n = sum(1 for k in assertions if k.startswith(ev_name))
                    ev_name = f"{ev_name}_{n + 1}"
                assertions[ev_name] = EvaluationResult(
                    name=ev_name,
                    value=False,
                    reason=f"Task execution failed: {exc}",
                    source=EvaluatorSource(
                        name="CODE",
                        arguments={"evaluation_name": str(ev.evaluation_name)},
                    ),
                )
            run_results.append(
                RunResult(
                    run_index=i,
                    input_key=input_key,
                    run_span_id=run_span_id,
                    output=None,
                    duration=0.0,
                    assertions=assertions,
                    evaluator_failures=[],
                    error=exc,
                )
            )
            continue

        token = _current_run_span_id.set(run_span_id)
        try:
            run_result = await _evaluate_run(
                case,
                output,
                duration,
                all_evaluators,
                i,
                input_key,
                run_span_id,
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


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


ta = TypeAdapter(dict[str, Any])


def _create_runs_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """Build a DataFrame with one row per (run x evaluator).

    This is the most granular view of evaluation results.
    """
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                eval_metadata_map = _get_eval_metadata_for_case(cr, eval_name, eval_result)
                merged_metadata = merge_metadata(cr.metadata, eval_metadata_map)
                source_type = "LLM_JUDGE" if "LLMJudge" in eval_result.source.name else "CODE"
                rows.append(
                    {
                        "inputs": str(cr.inputs),
                        "output": str(rr.output),
                        "evaluator_result": eval_result.value,
                        "evaluator_data": merged_metadata.other_evaluator_data,
                        "evaluator_reason": eval_result.reason,
                        "expected": merged_metadata.expected,
                        "attributes": ta.dump_json(merged_metadata.attributes),
                        "tags": merged_metadata.tags,
                        "task_duration": rr.duration,
                        "evaluator_name": eval_name,
                        "case_name": cr.case_name,
                        "case_id": cr.base_input_key,
                        "run_index": rr.run_index,
                        "repeat_total": len(cr.run_results),
                        "threshold": cr.aggregated.threshold,
                        "source_type": source_type,
                        "source_id": eval_result.source.name,
                        "input_key": rr.input_key,
                        "trace_id": cr.trace_id,
                    }
                )
            # Handle evaluator failures
            for ef in rr.evaluator_failures:
                rows.append(
                    {
                        "inputs": str(cr.inputs),
                        "output": str(rr.output),
                        "evaluator_result": False,
                        "evaluator_data": "",
                        "evaluator_reason": f"Evaluator failed: {ef.error_message}\n\n{ef.error_stacktrace}",
                        "expected": True,
                        "attributes": ta.dump_json(cr.metadata.attributes),
                        "tags": cr.metadata.tags,
                        "task_duration": rr.duration,
                        "evaluator_name": ef.name,
                        "case_name": cr.case_name,
                        "case_id": cr.base_input_key,
                        "run_index": rr.run_index,
                        "repeat_total": len(cr.run_results),
                        "threshold": cr.aggregated.threshold,
                        "source_type": "CODE",
                        "source_id": ef.name,
                        "input_key": rr.input_key,
                        "trace_id": cr.trace_id,
                    }
                )
    return pd.DataFrame(rows)


def _create_cases_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """Build a DataFrame with one row per (case x evaluator), aggregated across runs."""
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for eval_name, pass_rate in cr.aggregated.per_evaluator_pass_rates.items():
            # Compute average duration for this evaluator across runs
            durations = [rr.duration for rr in cr.run_results]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            rows.append(
                {
                    "case_id": cr.base_input_key,
                    "case_name": cr.case_name,
                    "repeat_total": len(cr.run_results),
                    "threshold": cr.aggregated.threshold,
                    "inputs": str(cr.inputs),
                    "evaluator_name": eval_name,
                    "pass_rate": pass_rate,
                    "passed": pass_rate >= cr.aggregated.threshold,
                    "aggregated_reason": cr.aggregated.summary,
                    "expected": True,
                    "attributes": ta.dump_json(cr.metadata.attributes),
                    "tags": cr.metadata.tags,
                    "avg_task_duration": avg_duration,
                    "trace_id": cr.trace_id,
                }
            )
    return pd.DataFrame(rows)


def _get_eval_metadata_for_case(
    cr: CaseResult,
    _eval_name: str,
    eval_result: EvaluationResult,
) -> EvaluatorMetadata:
    """Look up the EvaluatorMetadata for a given evaluator result in a CaseResult.

    Falls back to a default metadata if the evaluator can't be found.
    """
    eval_uuid: str = ""
    if eval_result.source:
        eval_uuid = str(eval_result.source.arguments.get("evaluation_name", ""))
    return EvaluatorMetadata(
        expected=True,
        attributes=cr.metadata.attributes,
        tags=cr.metadata.tags,
        is_global_evaluator=False,
        other_evaluator_data=f"eval_uuid={eval_uuid}",
    )


# ---------------------------------------------------------------------------
# MLflow upload
# ---------------------------------------------------------------------------


def _upload_mlflow(
    eval_result_df: pd.DataFrame,
    case_results: list[CaseResult],
    model_params: dict[str, str] | None = None,
) -> None:
    """Upload evaluation results to MLflow as metrics, assessments, and tags.

    Args:
        eval_result_df: The runs DataFrame for logging as a table.
        case_results: Structured case results for assessment naming.
        model_params: Optional model parameters to log.
    """
    mlflow.log_table(eval_result_df, "evaluation_results.json")

    if model_params:
        mlflow.log_params(model_params)

    # Calculate overall accuracy
    eval_df: Any = eval_result_df
    df_valid = eval_df[eval_df["evaluator_result"].notna()]
    if len(df_valid) > 0:
        overall_accuracy: float = float(df_valid["evaluator_result"].mean())
        mlflow.log_metric("overall_accuracy", overall_accuracy)

        # Calculate accuracy per tag
        df_exploded = df_valid.explode("tags")
        accuracy_per_tag = df_exploded.groupby("tags")["evaluator_result"].mean()
        for tag, accuracy in accuracy_per_tag.items():
            if pd.notna(tag):  # pyright: ignore[reportUnknownMemberType]
                mlflow.log_metric(f"accuracy_tag_{tag}", float(accuracy))

    # Log assessments with run-prefixed naming
    for cr in case_results:
        trace_id = cr.trace_id
        repeat = len(cr.run_results)

        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                source_type = "LLM_JUDGE" if "LLMJudge" in eval_result.source.name else "CODE"
                assessment_name = f"run-{rr.run_index}_{eval_name}"
                feedback = Feedback(
                    name=assessment_name,
                    value=eval_result.value,
                    source=AssessmentSource(
                        source_type=source_type,
                        source_id=eval_result.source.name,
                    ),
                    rationale=str(eval_result.reason),
                )
                mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

        # Aggregate assessments (only when repeat > 1)
        if repeat > 1:
            for eval_name, eval_pass_rate in cr.aggregated.per_evaluator_pass_rates.items():
                agg_passed = eval_pass_rate >= cr.aggregated.threshold
                agg_feedback = Feedback(
                    name=f"agg_{eval_name}",
                    value=agg_passed,
                    source=AssessmentSource(source_type="CODE", source_id="ragpill_aggregation"),
                    rationale=f"Aggregate: {sum(1 for r in cr.run_results if eval_name in r.assertions and r.assertions[eval_name].value is True)}/{repeat} runs passed (threshold={cr.aggregated.threshold})",
                )
                mlflow.log_assessment(trace_id=trace_id, assessment=agg_feedback)

        # Set trace tags from case metadata
        for key, value in cr.metadata.attributes.items():
            mlflow.set_trace_tag(trace_id, key, str(value))
        for tag in cr.metadata.tags:
            mlflow.set_trace_tag(trace_id, f"tag_{tag}", "true")


# ---------------------------------------------------------------------------
# Main evaluation functions
# ---------------------------------------------------------------------------


async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    """Evaluate a testset with comprehensive MLflow logging and tracking.

    Orchestrates multi-run evaluation with two-phase execution:
    Phase 1 executes the task and commits spans, Phase 2 runs evaluators
    on committed traces.

    Args:
        testset: The dataset to evaluate.
        task: The task callable (mutually exclusive with task_factory).
            Use for stateless tasks that can be safely reused across runs.
        task_factory: A callable that returns a fresh task instance per run
            (mutually exclusive with task). Use for stateful tasks (e.g. agents
            with message history) to ensure isolation between runs.
        mlflow_settings: MLflow configuration. If None, loads from environment variables.
        model_params: Optional model parameters to log for reproducibility.

    Returns:
        EvaluationOutput with ``.runs``, ``.cases``, ``.summary`` DataFrames
        and ``.case_results`` for programmatic access.

    Raises:
        ValueError: If both or neither of ``task`` and ``task_factory`` are provided.
    """
    # Validate task/task_factory
    if task is not None and task_factory is not None:
        raise ValueError("Provide either 'task' or 'task_factory', not both.")
    if task is None and task_factory is None:
        raise ValueError("Provide either 'task' or 'task_factory'.")

    _factory: Callable[[], TaskType]
    if task is not None:
        _factory = lambda: task  # noqa: E731
    else:
        assert task_factory is not None
        _factory = task_factory

    mlflow_settings = mlflow_settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]
    _setup_mlflow_experiment(mlflow_settings)
    _fix_evaluator_global_flag(testset)

    # Execute each case
    case_results: list[CaseResult] = []
    for case in testset.cases:
        assert isinstance(case.metadata, TestCaseMetadata)
        repeat, threshold = resolve_repeat(case.metadata, mlflow_settings)

        # Collect all evaluators (case-level + dataset-level)
        all_evals: list[BaseEvaluator] = []
        for ev in case.evaluators:
            assert isinstance(ev, BaseEvaluator)
            all_evals.append(ev)
        for ev in testset.evaluators:
            assert isinstance(ev, BaseEvaluator)
            all_evals.append(ev)

        case_result = await _execute_case(
            case,
            _factory,
            all_evals,
            default_input_to_key,
            repeat,
            threshold,
        )
        case_results.append(case_result)

    # Clean up LLMJudge traces
    _delete_llm_judge_traces(mlflow_settings)

    # Build DataFrames
    runs_df = _create_runs_dataframe(case_results)
    cases_df = _create_cases_dataframe(case_results)

    # Upload to MLflow
    _upload_mlflow(runs_df, case_results, model_params)

    mlflow.end_run()

    return EvaluationOutput(
        runs=runs_df,
        cases=cases_df,
        case_results=case_results,
    )
