"""Phase 5: idempotent upload, destination provenance, judge-trace cap."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

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


def _settings() -> TrackingSettings:
    return TrackingSettings(tracking_uri="http://settings", experiment_name="exp")


def test_upload_marks_complete_and_logs_once(fake_tracking_backend):
    upload_results(_evaluation(), settings=_settings())
    assert fake_tracking_backend.get_run_tag("run-1", "ragpill_upload_state") == "complete"
    assert fake_tracking_backend.log_table.call_count == 1


def test_reupload_completed_run_raises(fake_tracking_backend):
    upload_results(_evaluation(), settings=_settings())
    with pytest.raises(RuntimeError, match="already uploaded"):
        upload_results(_evaluation(), settings=_settings())
    # The results table was not written a second time.
    assert fake_tracking_backend.log_table.call_count == 1


def test_overwrite_replaces_table_artifact(fake_tracking_backend):
    upload_results(_evaluation(), settings=_settings())
    upload_results(_evaluation(), settings=_settings(), overwrite=True)
    # The append-only table artifact is deleted before the second write.
    fake_tracking_backend.delete_run_artifact.assert_any_call("run-1", "evaluation_results.json")
    assert fake_tracking_backend.log_table.call_count == 2


def test_destination_precedence_prefers_recorded_uri(fake_tracking_backend):
    # No explicit tracking_uri -> the run's recorded URI wins over settings.
    upload_results(_evaluation(recorded_uri="http://recorded"), settings=_settings())
    fake_tracking_backend.set_destination.assert_called_once_with("http://recorded", "exp")


def test_explicit_tracking_uri_overrides_recorded(fake_tracking_backend):
    upload_results(_evaluation(recorded_uri="http://recorded"), settings=_settings(), tracking_uri="http://explicit")
    fake_tracking_backend.set_destination.assert_called_once_with("http://explicit", "exp")


# ---------------------------------------------------------------------------
# F6 — an unresolvable destination fails loudly (never silently ./mlruns)
# ---------------------------------------------------------------------------


def test_upload_without_any_tracking_uri_raises(fake_tracking_backend):
    # No explicit arg, run recorded without a URI, settings carry none: upload
    # has nowhere to write and must raise instead of landing in ./mlruns.
    evaluation = _evaluation(recorded_uri="")
    settings = TrackingSettings(tracking_uri=None, experiment_name="exp")
    with pytest.raises(ValueError, match="needs a tracking URI"):
        upload_results(evaluation, settings=settings)
    # Nothing was written before the failure.
    assert not fake_tracking_backend.set_destination.called


def test_explicit_tracking_uri_proceeds_and_logs_winning_source(fake_tracking_backend, caplog):
    # Explicit tracking_uri= beats the (absent) recorded/settings URIs; the log
    # names which source won the precedence race.
    evaluation = _evaluation(recorded_uri="")
    settings = TrackingSettings(tracking_uri=None, experiment_name="exp")
    with caplog.at_level(logging.INFO, logger="ragpill.upload"):
        upload_results(evaluation, settings=settings, tracking_uri="http://explicit")

    fake_tracking_backend.set_destination.assert_called_once_with("http://explicit", "exp")
    messages = [r.getMessage() for r in caplog.records]
    assert any("http://explicit" in m and "tracking_uri arg" in m for m in messages)


# ---------------------------------------------------------------------------
# MLflow adapter specifics (mocked client)
# ---------------------------------------------------------------------------


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
