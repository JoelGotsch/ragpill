"""Evaluate layer: run evaluators against a captured ``DatasetRunOutput``.

This layer has no MLflow server dependency. It consumes:

- A :class:`~ragpill.execution.DatasetRunOutput` produced by Phase 1
  (``execute_dataset``).
- A :class:`~ragpill.eval_types.Dataset` of evaluators (case-level and
  dataset-level).

It returns an :class:`~ragpill.types.EvaluationOutput` with ``runs`` and
``cases`` DataFrames plus the structured ``case_results`` for downstream
Phase 3 upload.
"""

from __future__ import annotations

import traceback
from typing import Any

import pandas as pd
from pydantic import TypeAdapter

from ragpill.base import BaseEvaluator, CaseMetadataT, TestCaseMetadata, merge_metadata, resolve_repeat
from ragpill.eval_types import (
    Case,
    Dataset,
    EvaluationResult,
    EvaluatorContext,
    EvaluatorSource,
)
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.settings import MLFlowSettings
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    EvaluatorFailureInfo,
    RunResult,
)

_ta = TypeAdapter(dict[str, Any])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_runs(run_results: list[RunResult], threshold: float) -> AggregatedResult:
    """Aggregate multiple :class:`RunResult` objects into a pass/fail verdict.

    Args:
        run_results: One :class:`RunResult` per run for a single case.
        threshold: Minimum fraction of passing runs required for the case to pass.

    Returns:
        :class:`AggregatedResult` with pass/fail, pass_rate, and per-evaluator rates.
    """
    total = len(run_results)
    passed_count = sum(1 for r in run_results if r.all_passed)
    pass_rate = passed_count / total if total > 0 else 0.0
    passed = pass_rate >= threshold

    evaluator_names: set[str] = set()
    for r in run_results:
        evaluator_names.update(r.assertions.keys())

    per_evaluator_pass_rates: dict[str, float] = {}
    for eval_name in sorted(evaluator_names):
        eval_passed = sum(1 for r in run_results if eval_name in r.assertions and r.assertions[eval_name].value is True)
        per_evaluator_pass_rates[eval_name] = eval_passed / total if total > 0 else 0.0

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
# Per-run evaluation
# ---------------------------------------------------------------------------


async def _evaluate_single_run(
    case: Case[Any, Any, Any],
    task_run: TaskRunOutput,
    case_run: CaseRunOutput,
    evaluators: list[BaseEvaluator],
) -> RunResult:
    """Run every evaluator against one :class:`TaskRunOutput`."""
    # Task error short-circuits all evaluators to failure.
    if task_run.error is not None:
        assertions: dict[str, EvaluationResult] = {}
        for ev in evaluators:
            ev_name = ev.get_serialization_name()
            if ev_name in assertions:
                n = sum(1 for k in assertions if k.startswith(ev_name))
                ev_name = f"{ev_name}_{n + 1}"
            assertions[ev_name] = EvaluationResult(
                name=ev_name,
                value=False,
                reason=f"Task execution failed: {task_run.error}",
                source=EvaluatorSource(
                    name="CODE",
                    arguments={"evaluation_name": str(ev.evaluation_name)},
                ),
            )
        return RunResult(
            run_index=task_run.run_index,
            input_key=task_run.input_key,
            run_span_id=task_run.run_span_id,
            output=None,
            duration=task_run.duration,
            assertions=assertions,
            evaluator_failures=[],
            error=RuntimeError(task_run.error),
        )

    # Successful run — build a context and run evaluators.
    ctx: EvaluatorContext[Any, Any, Any] = EvaluatorContext(
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=task_run.output,
        duration=task_run.duration,
        attributes={},
        metrics={},
        trace=task_run.trace if task_run.trace is not None else case_run.trace,
        run_span_id=task_run.run_span_id,
    )

    assertions = {}
    evaluator_failures: list[EvaluatorFailureInfo] = []

    for evaluator in evaluators:
        eval_name = evaluator.get_serialization_name()
        try:
            result = await evaluator.evaluate(ctx)
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
        run_index=task_run.run_index,
        input_key=task_run.input_key,
        run_span_id=task_run.run_span_id,
        output=task_run.output,
        duration=task_run.duration,
        assertions=assertions,
        evaluator_failures=evaluator_failures,
    )


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def _get_eval_metadata_for_case(cr: CaseResult, eval_result: EvaluationResult):
    """Look up :class:`~ragpill.base.EvaluatorMetadata` for a given evaluator result.

    Falls back to a default metadata if the evaluator can't be located.
    """
    from ragpill.base import EvaluatorMetadata

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


def _create_runs_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """Build a DataFrame with one row per ``(run, evaluator)``."""
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                eval_metadata_map = _get_eval_metadata_for_case(cr, eval_result)
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
                        "attributes": _ta.dump_json(merged_metadata.attributes),
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
            for ef in rr.evaluator_failures:
                rows.append(
                    {
                        "inputs": str(cr.inputs),
                        "output": str(rr.output),
                        "evaluator_result": False,
                        "evaluator_data": "",
                        "evaluator_reason": f"Evaluator failed: {ef.error_message}\n\n{ef.error_stacktrace}",
                        "expected": True,
                        "attributes": _ta.dump_json(cr.metadata.attributes),
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
    """Build a DataFrame with one row per ``(case, evaluator)``, aggregated across runs."""
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for eval_name, pass_rate in cr.aggregated.per_evaluator_pass_rates.items():
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
                    "attributes": _ta.dump_json(cr.metadata.attributes),
                    "tags": cr.metadata.tags,
                    "avg_task_duration": avg_duration,
                    "trace_id": cr.trace_id,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def evaluate_results(
    dataset_run: DatasetRunOutput,
    testset: Dataset[Any, Any, CaseMetadataT],
    settings: MLFlowSettings | None = None,
) -> EvaluationOutput:
    """Run evaluators against a captured :class:`DatasetRunOutput`.

    For every ``(case, task_run)`` pair, builds an
    :class:`~ragpill.eval_types.EvaluatorContext` with the captured trace and
    runs every case-level + dataset-level evaluator. Results are aggregated
    per case using ``threshold`` from settings (or per-case override).

    Args:
        dataset_run: Output of :func:`ragpill.execution.execute_dataset`.
        testset: The dataset whose cases align one-for-one with
            ``dataset_run.cases``. Evaluators come from ``case.evaluators`` and
            ``testset.evaluators``.
        settings: Global :class:`MLFlowSettings`. Only ``ragpill_repeat`` and
            ``ragpill_threshold`` are consulted — no MLflow connection is made.

    Returns:
        :class:`EvaluationOutput` with ``.runs``, ``.cases``, and
        ``.case_results``.

    Example:
        ```python
        run_output = await execute_dataset(testset, task=my_task)
        eval_output = await evaluate_results(run_output, testset)
        print(eval_output.summary)
        ```

    See Also:
        [`execute_dataset`][ragpill.execution.execute_dataset]: Phase 1.
        [`upload_to_mlflow`][ragpill.upload.upload_to_mlflow]: Phase 3.
    """
    _settings = settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]

    if len(dataset_run.cases) != len(testset.cases):
        raise ValueError(f"dataset_run has {len(dataset_run.cases)} cases but testset has {len(testset.cases)}")

    case_results: list[CaseResult] = []
    for case_run, case in zip(dataset_run.cases, testset.cases):
        # Resolve evaluators: case-level + dataset-level.
        evaluators: list[BaseEvaluator] = []
        for ev in case.evaluators:
            assert isinstance(ev, BaseEvaluator)
            evaluators.append(ev)
        for ev in testset.evaluators:
            assert isinstance(ev, BaseEvaluator)
            evaluators.append(ev)

        case_metadata: TestCaseMetadata | None = case.metadata if isinstance(case.metadata, TestCaseMetadata) else None
        _, threshold = resolve_repeat(case_metadata, _settings)

        run_results: list[RunResult] = []
        for task_run in case_run.task_runs:
            rr = await _evaluate_single_run(case, task_run, case_run, evaluators)
            run_results.append(rr)

        aggregated = _aggregate_runs(run_results, threshold)

        # CaseResult demands a TestCaseMetadata; fall back to an empty one.
        metadata_obj = case_metadata or TestCaseMetadata()

        case_results.append(
            CaseResult(
                case_name=case_run.case_name,
                inputs=case_run.inputs,
                metadata=metadata_obj,
                base_input_key=case_run.base_input_key,
                trace_id=case_run.trace_id,
                run_results=run_results,
                aggregated=aggregated,
            )
        )

    runs_df = _create_runs_dataframe(case_results)
    cases_df = _create_cases_dataframe(case_results)
    return EvaluationOutput(
        runs=runs_df,
        cases=cases_df,
        case_results=case_results,
        dataset_run=dataset_run,
    )


__all__ = ["evaluate_results"]
