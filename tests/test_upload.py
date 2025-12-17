"""Unit tests for ``ragpill.upload.upload_results`` with a fake backend.

Uses the shared ``fake_tracking_backend`` fixture from ``conftest.py`` — a
stateful in-memory ``Backend`` whose methods are Mock spies, so tests can
assert on calls and override behavior via ``.side_effect``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import DatasetRunOutput
from ragpill.settings import TrackingSettings
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult
from ragpill.upload import upload_results


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
        dataset_run=DatasetRunOutput(tracking_uri="http://fake", run_id="fake-run", experiment_id="1"),
    )


def _settings() -> TrackingSettings:
    return TrackingSettings(
        tracking_uri="http://fake",
        experiment_name="fake-exp",
    )


def test_upload_calls_log_table(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.log_table.called
    args, _kwargs = fake_tracking_backend.log_table.call_args
    assert args[1] == "evaluation_results.json"


def test_upload_reattaches_existing_run(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.start_run.called
    _, kwargs = fake_tracking_backend.start_run.call_args
    assert kwargs.get("run_id") == "fake-run"


def test_upload_ends_run_in_finally(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.end_run.called


def test_upload_ends_run_even_on_exception(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    fake_tracking_backend.log_table.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.end_run.called


def test_upload_logs_assessment_when_trace_id_set(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    evaluation.case_results[0].trace_id = "trace-1"
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.log_assessment.called
    args, _kwargs = fake_tracking_backend.log_assessment.call_args
    # First positional arg is the trace id.
    assert args[0] == "trace-1"


def test_upload_skips_assessments_when_trace_id_empty(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert not fake_tracking_backend.log_assessment.called


def test_upload_traces_writes_artifact(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    upload_results(evaluation, settings=_settings(), upload_traces=True)
    assert fake_tracking_backend.log_artifact.called
    _args, kwargs = fake_tracking_backend.log_artifact.call_args
    assert kwargs.get("artifact_path") == "ragpill_traces"


def test_upload_without_dataset_run_still_works(fake_tracking_backend):
    evaluation = _make_evaluation_output()
    evaluation.dataset_run = None
    upload_results(evaluation, settings=_settings(), upload_traces=False)
    assert fake_tracking_backend.log_table.called
    _, kwargs = fake_tracking_backend.start_run.call_args
    assert "run_id" not in kwargs


def test_upload_logs_assessments_to_per_run_traces_in_session_mode(fake_tracking_backend):
    # Session-mode grouping: no case-level trace; each repeat carries its own
    # trace id. Assessments and tags must land on the per-run traces.
    evaluation = _make_evaluation_output()
    cr = evaluation.case_results[0]
    cr.trace_id = ""
    cr.run_results[0].trace_id = "run-trace-1"
    cr.metadata = TestCaseMetadata(attributes={"team": "a"}, tags={"t1"})

    upload_results(evaluation, settings=_settings(), upload_traces=False)

    assert fake_tracking_backend.log_assessment.called
    args, _kwargs = fake_tracking_backend.log_assessment.call_args
    assert args[0] == "run-trace-1"
    tagged_ids = {call.args[0] for call in fake_tracking_backend.set_trace_tag.call_args_list}
    assert tagged_ids == {"run-trace-1"}
