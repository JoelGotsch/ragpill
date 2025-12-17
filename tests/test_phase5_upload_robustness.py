"""Phase 5: idempotent upload, destination provenance, judge-trace cap."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ragpill.backends import RunHandle, configure_backend, reset_backend
from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import DatasetRunOutput
from ragpill.settings import TrackingSettings
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult
from ragpill.upload import upload_results


def _evaluation(recorded_uri: str = "http://recorded", run_id: str = "run-1") -> EvaluationOutput:
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="",
        output="out",
        duration=0.1,
        assertions={"e1": EvaluationResult("e1", True, "ok", EvaluatorSource(name="CODE"))},
    )
    cr = CaseResult(
        case_name="c",
        inputs="i",
        metadata=TestCaseMetadata(),
        base_input_key="k",
        trace_id="",
        run_results=[rr],
        aggregated=AggregatedResult(True, 1.0, 1.0, "1/1", {"e1": 1.0}),
    )
    runs_df = pd.DataFrame([{"evaluator_result": True, "tags": set(), "run_index": 0, "evaluator_name": "e1"}])
    return EvaluationOutput(
        runs=runs_df,
        cases=pd.DataFrame(),
        case_results=[cr],
        dataset_run=DatasetRunOutput(tracking_uri=recorded_uri, run_id=run_id, experiment_id="1"),
    )


@pytest.fixture
def fake_backend():
    """A MagicMock backend with a stateful run-tag store and run-active tracking."""
    backend = MagicMock()
    backend.get_tracking_uri.return_value = "previous"
    backend.resolve_experiment_id.return_value = "1"
    backend.start_run.return_value = RunHandle(run_id="run-1", experiment_id="1")

    tags: dict[str, str] = {}
    state = {"active": False}
    backend.set_run_tag.side_effect = lambda _rid, k, v: tags.__setitem__(k, v)
    backend.get_run_tag.side_effect = lambda _rid, k: tags.get(k)

    def _start(*_a, **_kw):
        state["active"] = True
        return backend.start_run.return_value

    backend.start_run.side_effect = _start
    backend.end_run.side_effect = lambda *_a, **_kw: state.__setitem__("active", False)
    backend.is_run_active.side_effect = lambda: state["active"]

    configure_backend(lambda: backend)
    try:
        yield backend
    finally:
        reset_backend()


def _settings() -> TrackingSettings:
    return TrackingSettings(tracking_uri="http://settings", experiment_name="exp")


def test_upload_marks_complete_and_logs_once(fake_backend):
    upload_results(_evaluation(), settings=_settings())
    assert fake_backend.get_run_tag("run-1", "ragpill_upload_state") == "complete"
    assert fake_backend.log_table.call_count == 1


def test_reupload_completed_run_raises(fake_backend):
    upload_results(_evaluation(), settings=_settings())
    with pytest.raises(RuntimeError, match="already uploaded"):
        upload_results(_evaluation(), settings=_settings())
    # The results table was not written a second time.
    assert fake_backend.log_table.call_count == 1


def test_overwrite_replaces_table_artifact(fake_backend):
    upload_results(_evaluation(), settings=_settings())
    upload_results(_evaluation(), settings=_settings(), overwrite=True)
    # The append-only table artifact is deleted before the second write.
    fake_backend.delete_run_artifact.assert_any_call("run-1", "evaluation_results.json")
    assert fake_backend.log_table.call_count == 2


def test_destination_precedence_prefers_recorded_uri(fake_backend):
    # No explicit tracking_uri -> the run's recorded URI wins over settings.
    upload_results(_evaluation(recorded_uri="http://recorded"), settings=_settings())
    fake_backend.set_destination.assert_called_once_with("http://recorded", "exp")


def test_explicit_tracking_uri_overrides_recorded(fake_backend):
    upload_results(_evaluation(recorded_uri="http://recorded"), settings=_settings(), tracking_uri="http://explicit")
    fake_backend.set_destination.assert_called_once_with("http://explicit", "exp")


def test_judge_trace_search_is_not_capped_at_1000():
    from ragpill.backends import mlflow_backend as mb

    assert mb._JUDGE_TRACE_SEARCH_LIMIT > 1000  # pyright: ignore[reportPrivateUsage]


def test_mlflow_backend_run_tag_roundtrip(monkeypatch: pytest.MonkeyPatch):
    from ragpill.backends.mlflow_backend import MLflowBackend

    backend = MLflowBackend()
    fake_client = MagicMock()
    fake_client.get_run.return_value.data.tags = {"ragpill_upload_state": "complete"}
    monkeypatch.setattr(backend, "_client", lambda: fake_client)

    backend.set_run_tag("r1", "k", "v")
    fake_client.set_tag.assert_called_once_with("r1", "k", "v")
    assert backend.get_run_tag("r1", "ragpill_upload_state") == "complete"
    assert backend.get_run_tag("r1", "missing") is None


def test_mlflow_delete_run_artifact_noop_when_absent(monkeypatch: pytest.MonkeyPatch):
    from ragpill.backends.mlflow_backend import MLflowBackend

    backend = MLflowBackend()
    fake_client = MagicMock()
    fake_client.list_artifacts.return_value = []  # nothing to delete
    monkeypatch.setattr(backend, "_client", lambda: fake_client)

    backend.delete_run_artifact("r1", "evaluation_results.json")
    fake_client.get_run.assert_not_called()  # short-circuits without resolving the artifact repo
