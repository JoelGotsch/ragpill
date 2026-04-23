"""Unit tests for ``ragpill.evaluation.evaluate_results``."""

from __future__ import annotations

from typing import Any

import pytest

from ragpill.base import BaseEvaluator, EvaluatorMetadata, TestCaseMetadata
from ragpill.eval_types import Case, Dataset, EvaluationReason, EvaluatorContext
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput


def _make_run(output: str, run_index: int = 0, error: str | None = None) -> TaskRunOutput:
    return TaskRunOutput(
        run_index=run_index,
        input_key=f"k_{run_index}",
        output=output,
        duration=0.01,
        trace=None,
        run_span_id="",
        error=error,
    )


def _make_case_run(task_runs: list[TaskRunOutput], inputs: str = "hello") -> CaseRunOutput:
    return CaseRunOutput(
        case_name=inputs,
        inputs=inputs,
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
        base_input_key="k",
        trace=None,
        trace_id="",
        task_runs=task_runs,
    )


def _make_case(
    inputs: str = "hello", evaluators: list[BaseEvaluator] | None = None
) -> Case[str, str, TestCaseMetadata]:
    return Case(
        inputs=inputs,
        metadata=TestCaseMetadata(),
        evaluators=list(evaluators or []),
    )


# ---------------------------------------------------------------------------
# evaluate_results — basic flow
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_evaluate_results_single_run_passing_evaluator():
    ev = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = _make_case("hello", [ev])
    case_run = _make_case_run([_make_run("hello world")])
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    assert len(output.runs) == 1
    assert bool(output.runs.iloc[0]["evaluator_result"]) is True
    assert output.case_results[0].aggregated.passed is True


@pytest.mark.anyio
async def test_evaluate_results_failing_evaluator():
    ev = RegexInOutputEvaluator(pattern="goodbye", expected=True, tags=set())
    case = _make_case("hi", [ev])
    case_run = _make_case_run([_make_run("hello")])
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    assert bool(output.runs.iloc[0]["evaluator_result"]) is False
    assert output.case_results[0].aggregated.passed is False


@pytest.mark.anyio
async def test_evaluate_results_task_error_fails_all_evaluators():
    ev = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = _make_case("hi", [ev])
    case_run = _make_case_run([_make_run("", error="RuntimeError: boom")])
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    rr = output.case_results[0].run_results[0]
    assert rr.error is not None
    # Every evaluator should be marked False when the task failed
    assert all(a.value is False for a in rr.assertions.values())
    assert len(rr.assertions) == 1


# ---------------------------------------------------------------------------
# Evaluator that raises
# ---------------------------------------------------------------------------


class _BrokenEvaluator(BaseEvaluator):
    async def run(self, ctx: EvaluatorContext[Any, Any, EvaluatorMetadata]) -> EvaluationReason:
        raise RuntimeError("evaluator crashed")


@pytest.mark.anyio
async def test_evaluator_exception_captured_in_failures():
    ev = _BrokenEvaluator(expected=True, tags=set())
    case = _make_case("hi", [ev])
    case_run = _make_case_run([_make_run("hello")])
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    rr = output.case_results[0].run_results[0]
    assert len(rr.evaluator_failures) == 1
    assert "crashed" in rr.evaluator_failures[0].error_message


# ---------------------------------------------------------------------------
# Aggregation / multiple runs
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_evaluate_results_aggregates_multiple_runs():
    ev = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = _make_case("hi", [ev])
    case_run = _make_case_run(
        [
            _make_run("hello", run_index=0),
            _make_run("hello", run_index=1),
            _make_run("goodbye", run_index=2),
        ],
    )
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    agg = output.case_results[0].aggregated
    assert pytest.approx(agg.pass_rate, abs=0.001) == 2 / 3


# ---------------------------------------------------------------------------
# Dataset vs dataset_run length mismatch
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mismatched_case_counts_raises():
    case = _make_case("hi")
    dataset_run = DatasetRunOutput(cases=[])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])
    with pytest.raises(ValueError):
        await evaluate_results(dataset_run, testset)


# ---------------------------------------------------------------------------
# EvaluationOutput.dataset_run round trip
# ---------------------------------------------------------------------------


def test_span_base_evaluator_get_trace_raises_when_trace_missing():
    """SpanBaseEvaluator.get_trace raises ValueError when ctx.trace is None."""
    from ragpill.evaluators import RegexInSourcesEvaluator

    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="x")
    ctx: EvaluatorContext[Any, Any, EvaluatorMetadata] = EvaluatorContext(
        name="t",
        inputs="i",
        metadata=EvaluatorMetadata(expected=True),
        expected_output=None,
        output="o",
        duration=0.0,
    )
    with pytest.raises(ValueError, match=r"ctx\.trace"):
        ev.get_trace(ctx)


@pytest.mark.anyio
async def test_evaluate_results_passes_through_dataset_run():
    ev = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = _make_case("hi", [ev])
    case_run = _make_case_run([_make_run("hello")])
    dataset_run = DatasetRunOutput(cases=[case_run])
    testset = Dataset[str, str, TestCaseMetadata](cases=[case])

    output = await evaluate_results(dataset_run, testset)
    assert output.dataset_run is dataset_run
