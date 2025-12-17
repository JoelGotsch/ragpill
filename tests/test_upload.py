"""Unit tests for ``ragpill.upload.upload_to_mlflow`` with mocked MLflow."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import DatasetRunOutput
from ragpill.settings import MLFlowSettings
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult
from ragpill.upload import upload_to_mlflow


def _make_assertion(name: str, value: bool) -> EvaluationResult:
    return EvaluationResult(
        name=name,
        value=value,
        reason=f"{name} reason",
        source=EvaluatorSource(name="CODE", arguments={"evaluation_name": "test"}),
    )


def _make_evaluation_output() -> EvaluationOutput:
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="",
        output="out",
        duration=0.1,
        assertions={"e1": _make_assertion("e1", True)},
    )
    cr = CaseResult(
        case_name="c",
        inputs="i",
        metadata=TestCaseMetadata(),
        base_input_key="k",
        trace_id="",  # empty to skip trace-id-dependent calls
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=True,
            pass_rate=1.0,
            threshold=1.0,
            summary="1/1",
            per_evaluator_pass_rates={"e1": 1.0},
        ),
    )
    runs_df = pd.DataFrame(
        [
            {
                "inputs": "i",
                "output": "out",
                "evaluator_result": True,
                "tags": set(),
                "run_index": 0,
                "evaluator_name": "e1",
            }
        ]
    )
    return EvaluationOutput(
        runs=runs_df,
        cases=pd.DataFrame(),
        case_results=[cr],
        dataset_run=DatasetRunOutput(tracking_uri="http://fake", mlflow_run_id="fake-run", mlflow_experiment_id="1"),
    )


def _settings() -> MLFlowSettings:
    return MLFlowSettings(
        ragpill_tracking_uri="http://fake",
        ragpill_experiment_name="fake-exp",
    )


@pytest.fixture
def mlflow_mock():
    """Inject a fully-mocked Backend so upload_to_mlflow never touches a real backend.

    The fixture pretends a run is active until ``end_run`` is called once, so
    the upload's ``finally`` block can call ``end_run`` exactly once.
    """
    from ragpill.backends import RunHandle, configure_backend, reset_backend

    backend = MagicMock()
    backend.get_tracking_uri.return_value = "previous-uri"
    backend.resolve_experiment_id.return_value = "1"
    backend.search_traces.return_value = []
    backend.start_run.return_value = RunHandle(run_id="fake-run", experiment_id="1")

    state = {"active": False}

    def _set_active(*_a, **_kw):
        state["active"] = True
        return backend.start_run.return_value

    def _clear_active(*_a, **_kw):
        state["active"] = False

    backend.start_run.side_effect = _set_active
    backend.end_run.side_effect = _clear_active
    backend.is_run_active.side_effect = lambda: state["active"]

    configure_backend(lambda: backend)
    try:
        yield backend
    finally:
        reset_backend()


def test_upload_calls_log_table(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_table.called
    args, _kwargs = mlflow_mock.log_table.call_args
    assert args[1] == "evaluation_results.json"


def test_upload_reattaches_existing_run(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.start_run.called
    _, kwargs = mlflow_mock.start_run.call_args
    assert kwargs.get("run_id") == "fake-run"


def test_upload_ends_run_in_finally(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.end_run.called


def test_upload_ends_run_even_on_exception(mlflow_mock):
    evaluation = _make_evaluation_output()
    mlflow_mock.log_table.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.end_run.called


def test_upload_logs_assessment_when_trace_id_set(mlflow_mock):
    evaluation = _make_evaluation_output()
    evaluation.case_results[0].trace_id = "trace-1"
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_assessment.called
    args, _kwargs = mlflow_mock.log_assessment.call_args
    # First positional arg is the trace id.
    assert args[0] == "trace-1"


def test_upload_skips_assessments_when_trace_id_empty(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert not mlflow_mock.log_assessment.called


def test_upload_traces_writes_artifact(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=True)
    assert mlflow_mock.log_artifact.called
    _args, kwargs = mlflow_mock.log_artifact.call_args
    assert kwargs.get("artifact_path") == "ragpill_traces"


def test_upload_without_dataset_run_still_works(mlflow_mock):
    evaluation = _make_evaluation_output()
    evaluation.dataset_run = None
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_table.called
    _, kwargs = mlflow_mock.start_run.call_args
    assert "run_id" not in kwargs
