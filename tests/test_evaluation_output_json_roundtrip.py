"""Round-trip tests for ``EvaluationOutput.to_json`` / ``from_json``."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import mlflow
import pandas as pd
import pytest
from mlflow.entities import SpanType, Trace

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    EvaluatorFailureInfo,
    RunResult,
)


@pytest.fixture
def _isolated_mlflow_backend() -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    tmp = tempfile.mkdtemp(prefix="ragpill_eo_roundtrip_")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 'mlflow.db')}")
    mlflow.set_experiment(f"exp-{os.path.basename(tmp)}")
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _runs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"case_id": "c1", "evaluator_name": "E", "passed": True, "pass_rate": 1.0},
            {"case_id": "c1", "evaluator_name": "F", "passed": False, "pass_rate": 0.0},
        ]
    )


def _cases_df() -> pd.DataFrame:
    return pd.DataFrame([{"case_id": "c1", "evaluator_name": "E", "pass_rate": 1.0, "passed": True}])


def _build_case_results() -> list[CaseResult]:
    rr = RunResult(
        run_index=0,
        input_key="c1_0",
        run_span_id="span-1",
        output="hello",
        duration=0.42,
        assertions={
            "E": EvaluationResult(
                name="E", value=True, reason="ok", source=EvaluatorSource(name="E", arguments={"k": 1})
            ),
            "F": EvaluationResult(name="F", value=False, reason="bad", source=EvaluatorSource(name="F")),
        },
        evaluator_failures=[EvaluatorFailureInfo(name="X", error_message="boom", error_stacktrace="...")],
        error=ValueError("task crashed"),
    )
    cr = CaseResult(
        case_name="C1",
        inputs={"q": "hi"},
        metadata=TestCaseMetadata(threshold=0.5),
        base_input_key="c1",
        trace_id="t1",
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=False,
            pass_rate=0.5,
            threshold=0.5,
            summary="1/1 runs ok",
            per_evaluator_pass_rates={"E": 1.0, "F": 0.0},
        ),
    )
    return [cr]


def test_roundtrip_preserves_dataframes():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())
    pd.testing.assert_frame_equal(restored.runs.reset_index(drop=True), eo.runs.reset_index(drop=True))
    pd.testing.assert_frame_equal(restored.cases.reset_index(drop=True), eo.cases.reset_index(drop=True))


def test_roundtrip_preserves_case_results_structure():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())

    assert len(restored.case_results) == 1
    cr = restored.case_results[0]
    assert cr.base_input_key == "c1"
    assert cr.aggregated.pass_rate == 0.5
    assert cr.aggregated.per_evaluator_pass_rates == {"E": 1.0, "F": 0.0}
    rr = cr.run_results[0]
    assert rr.assertions["E"].value is True
    assert rr.assertions["F"].reason == "bad"
    assert rr.assertions["E"].source.arguments == {"k": 1}
    assert len(rr.evaluator_failures) == 1
    assert rr.evaluator_failures[0].name == "X"
    # Exceptions become RuntimeError after round trip — content is preserved as a string.
    assert rr.error is not None
    assert "task crashed" in str(rr.error)


def test_roundtrip_with_dataset_run(_isolated_mlflow_backend: None) -> None:
    with mlflow.start_run():
        with mlflow.start_span(name="root", span_type=SpanType.AGENT) as root:
            root.set_inputs("hi")
            root.set_outputs("bye")
            run_span_id = root.span_id
    traces: list[Trace] = mlflow.search_traces(return_type="list", max_results=1)  # pyright: ignore[reportAssignmentType]
    trace = traces[0]
    case = CaseRunOutput(
        case_name="C1",
        inputs="hi",
        expected_output="bye",
        metadata={},
        base_input_key="c1",
        trace=trace,
        trace_id="t1",
        task_runs=[
            TaskRunOutput(
                run_index=0, input_key="c1_0", output="bye", duration=0.0, trace=trace, run_span_id=run_span_id
            )
        ],
    )
    dr = DatasetRunOutput(cases=[case], tracking_uri="x", mlflow_run_id="r", mlflow_experiment_id="e")
    eo = EvaluationOutput(
        runs=_runs_df(),
        cases=_cases_df(),
        case_results=_build_case_results(),
        dataset_run=dr,
    )
    restored = EvaluationOutput.from_json(eo.to_json())
    assert restored.dataset_run is not None
    assert restored.dataset_run.tracking_uri == "x"
    rcase = restored.dataset_run.cases[0]
    assert rcase.trace is not None
    orig_ids = sorted(s.span_id for s in trace.data.spans)
    new_ids = sorted(s.span_id for s in rcase.trace.data.spans)
    assert orig_ids == new_ids


def test_roundtrip_when_dataset_run_is_none():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())
    assert restored.dataset_run is None
