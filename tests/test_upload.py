"""Unit tests for ``ragpill.upload.upload_to_mlflow`` with mocked MLflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    """Patch ``mlflow`` calls at the module-under-test level."""
    with patch("ragpill.upload.mlflow") as m:
        m.get_tracking_uri.return_value = "previous-uri"
        m.active_run.return_value = MagicMock(info=MagicMock(run_id="fake-run"))
        m.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        m.search_traces.return_value = []
        yield m


def test_upload_calls_log_table(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_table.called
    args, kwargs = mlflow_mock.log_table.call_args
    # Second positional is the artifact filename
    positional_filename = args[1] if len(args) > 1 else None
    assert positional_filename == "evaluation_results.json" or kwargs.get("artifact_file") == "evaluation_results.json"


def test_upload_reattaches_existing_run(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.start_run.called
    # Called with run_id=fake-run to reattach
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
    # Give the case a real trace_id so assessment logging fires
    evaluation.case_results[0].trace_id = "trace-1"
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_assessment.called
    _, kwargs = mlflow_mock.log_assessment.call_args
    assert kwargs.get("trace_id") == "trace-1"


def test_upload_skips_assessments_when_trace_id_empty(mlflow_mock):
    evaluation = _make_evaluation_output()
    # trace_id is already "" in the default
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert not mlflow_mock.log_assessment.called


def test_upload_traces_writes_artifact(mlflow_mock):
    evaluation = _make_evaluation_output()
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=True)
    assert mlflow_mock.log_artifact.called
    _, kwargs = mlflow_mock.log_artifact.call_args
    # Artifact path includes "ragpill_traces"
    assert kwargs.get("artifact_path") == "ragpill_traces"


def test_upload_without_dataset_run_still_works(mlflow_mock):
    evaluation = _make_evaluation_output()
    evaluation.dataset_run = None
    upload_to_mlflow(evaluation, mlflow_settings=_settings(), upload_traces=False)
    assert mlflow_mock.log_table.called
    # Without a run_id, a new run is started (no run_id kwarg)
    _, kwargs = mlflow_mock.start_run.call_args
    assert "run_id" not in kwargs
