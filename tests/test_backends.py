"""Tests for ``ragpill.backends`` — Phase 1 Step 1 of multi-backend-tracking.

The behaviour under test is:

- ``MLflowBackend`` satisfies every protocol's ``runtime_checkable`` shape.
- The registry returns an ``MLflowBackend`` by default and lets the caller
  swap it out.
- Each forwarder method actually invokes the expected ``mlflow.*`` symbol
  with the expected arguments (mocked — no live server).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ragpill.backends import (
    Assessment,
    Backend,
    LifecycleBackend,
    ResultsBackend,
    RunHandle,
    SpanKind,
    TraceCaptureBackend,
    TraceQueryBackend,
    configure_backend,
    get_backend,
    reset_backend,
)
from ragpill.backends.mlflow_backend import MLflowBackend


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test gets a fresh registry."""
    reset_backend()
    yield
    reset_backend()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mlflow_backend_satisfies_all_protocols():
    """MLflowBackend implements all four capability protocols."""
    backend = MLflowBackend()
    assert isinstance(backend, TraceCaptureBackend)
    assert isinstance(backend, TraceQueryBackend)
    assert isinstance(backend, ResultsBackend)
    assert isinstance(backend, LifecycleBackend)
    assert isinstance(backend, Backend)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_backend_returns_mlflow_by_default():
    """With no configuration, the registry hands out an MLflowBackend."""
    assert isinstance(get_backend(), MLflowBackend)


def test_get_backend_caches_within_a_process():
    """Repeated calls return the same instance (until reset)."""
    assert get_backend() is get_backend()


def test_configure_backend_swaps_the_factory():
    """A custom factory wins over the default."""

    sentinel = MagicMock(spec=Backend)
    configure_backend(lambda: sentinel)
    assert get_backend() is sentinel


def test_reset_backend_clears_cached_instance():
    """After reset_backend(), the next get_backend() rebuilds."""
    first = get_backend()
    reset_backend()
    second = get_backend()
    assert first is not second


# ---------------------------------------------------------------------------
# Forwarders — assert each MLflowBackend method calls the right mlflow.* symbol
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_mock():
    """Patch the ``mlflow`` module used by the adapter."""
    with patch("ragpill.backends.mlflow_backend.mlflow") as m:
        m.active_run.return_value = MagicMock(info=MagicMock(run_id="run-1", experiment_id="exp-1"))
        m.start_run.return_value = MagicMock(info=MagicMock(run_id="run-1", experiment_id="exp-1"))
        m.get_tracking_uri.return_value = "https://previous"
        m.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
        m.search_traces.return_value = []
        yield m


def test_set_destination_sets_tracking_uri_and_experiment(mlflow_mock):
    MLflowBackend().set_destination("http://server", "my-exp")
    mlflow_mock.set_tracking_uri.assert_called_once_with("http://server")
    mlflow_mock.set_experiment.assert_called_once_with("my-exp")


def test_set_destination_skips_tracking_uri_when_none(mlflow_mock):
    MLflowBackend().set_destination(None, "my-exp")
    mlflow_mock.set_tracking_uri.assert_not_called()
    mlflow_mock.set_experiment.assert_called_once_with("my-exp")


def test_start_run_returns_run_handle(mlflow_mock):
    handle = MLflowBackend().start_run(run_id="r-1", description="d")
    mlflow_mock.start_run.assert_called_once_with(run_id="r-1", description="d")
    assert isinstance(handle, RunHandle)
    assert handle.run_id == "run-1"
    assert handle.experiment_id == "exp-1"


def test_end_run_only_when_active(mlflow_mock):
    mlflow_mock.active_run.return_value = None
    MLflowBackend().end_run()
    mlflow_mock.end_run.assert_not_called()

    mlflow_mock.active_run.return_value = MagicMock()
    MLflowBackend().end_run()
    mlflow_mock.end_run.assert_called_once()


def test_start_span_forwards_kind(mlflow_mock):
    MLflowBackend().start_span("name", SpanKind.LLM)
    _args, kwargs = mlflow_mock.start_span.call_args
    assert kwargs.get("name") == "name"
    # span_type is mapped through; the exact MLflow value comes from the imported SpanType.
    assert "span_type" in kwargs


def test_autolog_pydantic_ai_calls_mlflow(mlflow_mock):
    MLflowBackend().autolog_pydantic_ai()
    mlflow_mock.pydantic_ai.autolog.assert_called_once_with()


def test_search_traces_forwards_run_id(mlflow_mock):
    MLflowBackend().search_traces("run-1", max_results=42)
    _args, kwargs = mlflow_mock.search_traces.call_args
    assert kwargs.get("run_id") == "run-1"
    assert kwargs.get("max_results") == 42


def test_log_metric_forwards(mlflow_mock):
    MLflowBackend().log_metric("accuracy", 0.9)
    mlflow_mock.log_metric.assert_called_once_with("accuracy", 0.9)


def test_log_params_forwards_dict(mlflow_mock):
    MLflowBackend().log_params({"k": "v"})
    mlflow_mock.log_params.assert_called_once_with({"k": "v"})


def test_log_table_forwards(mlflow_mock):
    df = pd.DataFrame([{"a": 1}])
    MLflowBackend().log_table(df, "results.json")
    mlflow_mock.log_table.assert_called_once_with(df, "results.json")


def test_log_artifact_forwards(mlflow_mock):
    MLflowBackend().log_artifact("/tmp/x", artifact_path="dir")
    mlflow_mock.log_artifact.assert_called_once_with("/tmp/x", artifact_path="dir")


def test_log_assessment_builds_feedback(mlflow_mock):
    """The neutral Assessment dataclass converts to mlflow.entities.Feedback."""
    backend = MLflowBackend()
    a = Assessment(
        name="judge",
        value=True,
        source_type="LLM_JUDGE",
        source_id="LLMJudge",
        rationale="ok",
    )
    backend.log_assessment("trace-1", a)
    mlflow_mock.log_assessment.assert_called_once()
    _, kwargs = mlflow_mock.log_assessment.call_args
    assert kwargs.get("trace_id") == "trace-1"
    # The constructed Feedback is the second arg/keyword; assert its shape via repr.
    feedback = kwargs.get("assessment")
    assert feedback.name == "judge"
    assert feedback.value is True
    assert feedback.rationale == "ok"


def test_set_trace_tag_forwards(mlflow_mock):
    MLflowBackend().set_trace_tag("trace-1", "tag_alpha", "true")
    mlflow_mock.set_trace_tag.assert_called_once_with("trace-1", "tag_alpha", "true")


def test_resolve_experiment_id_returns_str(mlflow_mock):
    assert MLflowBackend().resolve_experiment_id("my-exp") == "exp-1"


def test_resolve_experiment_id_raises_on_missing(mlflow_mock):
    mlflow_mock.get_experiment_by_name.return_value = None
    with pytest.raises(RuntimeError, match="not found"):
        MLflowBackend().resolve_experiment_id("missing-exp")


def test_get_and_set_tracking_uri_roundtrip(mlflow_mock):
    backend = MLflowBackend()
    assert backend.get_tracking_uri() == "https://previous"
    backend.set_tracking_uri("https://new")
    mlflow_mock.set_tracking_uri.assert_called_once_with("https://new")


def test_is_run_active_reflects_mlflow_state(mlflow_mock):
    backend = MLflowBackend()
    mlflow_mock.active_run.return_value = None
    assert backend.is_run_active() is False
    mlflow_mock.active_run.return_value = MagicMock()
    assert backend.is_run_active() is True


def test_delete_traces_noop_on_empty(mlflow_mock):
    """Empty request_ids list is a no-op (no client constructed)."""
    MLflowBackend().delete_traces("exp-1", [])
    # MlflowClient is imported lazily; assert it was never instantiated.
    assert "MlflowClient" not in dir(mlflow_mock) or not mlflow_mock.MlflowClient.called


# ---------------------------------------------------------------------------
# Neutral types
# ---------------------------------------------------------------------------


def test_span_kind_enum_has_minimum_values():
    """SpanKind must cover the kinds the existing renderer/evaluators use."""
    expected = {"AGENT", "CHAT_MODEL", "LLM", "RERANKER", "RETRIEVER", "TASK", "TOOL", "UNKNOWN"}
    assert {k.value for k in SpanKind} == expected


def test_run_handle_is_immutable():
    """RunHandle is frozen so it can be safely cached."""
    h = RunHandle(run_id="r", experiment_id="e")
    with pytest.raises(Exception):
        h.run_id = "other"  # type: ignore[misc]


def test_assessment_metadata_defaults_to_empty_dict():
    """Each Assessment gets its own metadata dict (no shared default mutable)."""
    a = Assessment(name="a", value=True, source_type="CODE", source_id="x")
    b = Assessment(name="b", value=False, source_type="CODE", source_id="x")
    a.metadata["k"] = "v"
    assert "k" not in b.metadata


def test_assessment_accepts_numeric_values():
    """Assessment.value handles bool/int/float/str per the type annotation."""
    Any_a: Any = Assessment(name="a", value=0.9, source_type="CODE", source_id="x")
    assert Any_a.value == 0.9
