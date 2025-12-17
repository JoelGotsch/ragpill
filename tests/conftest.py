"""Shared pytest configuration for the ragpill test suite.

Two things live here:

1. The MLflow async-export switch (see below) — must run before any test
   module imports ``mlflow``.
2. The shared fakes: :class:`FakeTrackingBackend` (one stateful in-memory
   implementation of the full ``ragpill.backends.Backend`` protocol, replacing
   the per-file hand-rolled backend mocks) and :func:`make_span` (the span
   builder previously triplicated across test files).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ragpill.backends import configure_backend, reset_backend
from ragpill.backends._types import Assessment, CaptureSpanKind, CaseGroupingHandle, RunHandle
from ragpill.trace import Span, SpanKind, Trace

# Disable MLflow's async trace export by default. The unit tests create
# spans with ``mlflow.start_span`` and then immediately read them back
# with ``mlflow.search_traces``. Async export defers the write to a
# background queue, so the search races and returns ``[]``. Forcing
# synchronous export keeps unit tests deterministic; the
# ``mlflow-integration`` CI job exercises the async path against a real
# tracking server. See ADR-0010 — pending write-up; tracked in
# plans/adr-system.md.
#
# NOTE: nothing imported above pulls in ``mlflow`` (the ragpill imports are
# lazy about it), so this still runs before any mlflow code reads the flag.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")


# ---------------------------------------------------------------------------
# Shared span builder
# ---------------------------------------------------------------------------


def make_span(
    span_id: str,
    parent_id: str | None,
    kind: SpanKind = SpanKind.CHAIN,
    **overrides: Any,
) -> Span:
    """Build a test ``Span`` with sensible defaults; override any field via kwargs."""
    fields: dict[str, Any] = {
        "span_id": span_id,
        "parent_id": parent_id,
        "trace_id": "t",
        "name": span_id,
        "kind": kind,
        "start_time_ns": 0,
        "end_time_ns": 0,
    }
    fields.update(overrides)
    return Span(**fields)


# ---------------------------------------------------------------------------
# Shared fake tracking backend
# ---------------------------------------------------------------------------


class _FakeSpanHandle:
    """Minimal ``SpanHandle``: fixed ids, attribute setters are no-ops."""

    span_id = "s"
    trace_id = "t"

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_inputs(self, value: Any) -> None:
        pass

    def set_outputs(self, value: Any) -> None:
        pass


# Every public Backend-protocol method gets wrapped in a ``MagicMock`` spy
# (real behavior via ``side_effect``), so tests keep the full Mock assertion
# API — ``.call_count``, ``.assert_called_once_with``, ``.call_args_list`` —
# and can override behavior per-test by reassigning ``.side_effect``.
_SPIED_METHODS = (
    # TraceCaptureBackend
    "set_destination",
    "start_run",
    "end_run",
    "start_span",
    "autolog_pydantic_ai",
    "start_case_grouping",
    # TraceQueryBackend
    "get_trace",
    "await_trace",
    "delete_traces",
    "delete_judge_traces",
    # ResultsBackend
    "log_metric",
    "log_params",
    "log_table",
    "log_artifact",
    "log_assessment",
    "set_trace_tag",
    "set_run_tag",
    "get_run_tag",
    "delete_run_artifact",
    # LifecycleBackend
    "resolve_experiment_id",
    "get_tracking_uri",
    "set_tracking_uri",
    "is_run_active",
)


class FakeTrackingBackend:
    """Stateful in-memory implementation of the full ``Backend`` protocol.

    Tracks run-active state, a run-tag store, logged assessments, and trace
    tags. Behavior is overridable via constructor hooks (or ``.side_effect``
    on the per-method spies):

    - ``await_trace``: replaces the default ``(None, True)`` result; may raise.
    - ``on_set_destination`` / ``on_end_run``: callbacks invoked *after* the
      default bookkeeping (e.g. for concurrency-depth tracking).
    """

    supports_local_file_store: ClassVar[bool] = True

    def __init__(
        self,
        *,
        tracking_uri: str | None = "previous",
        experiment_id: str = "1",
        default_run_id: str = "run-1",
        await_trace: Callable[..., tuple[Trace | None, bool]] | None = None,
        on_set_destination: Callable[[str | None, str], None] | None = None,
        on_end_run: Callable[[], None] | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_id = experiment_id
        self._default_run_id = default_run_id
        self._await_trace_hook = await_trace
        self._on_set_destination = on_set_destination
        self._on_end_run = on_end_run
        self._active = False
        self.run_tags: dict[tuple[str, str], str] = {}
        self.trace_tags: dict[tuple[str, str], str] = {}
        self.assessments: list[tuple[str, Assessment]] = []
        # Wrap each protocol method in a spy that delegates to the real one.
        for name in _SPIED_METHODS:
            setattr(self, name, MagicMock(name=name, side_effect=getattr(self, name)))

    # -- TraceCaptureBackend --------------------------------------------------

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        self.destination = (uri, experiment_name)
        if self._on_set_destination is not None:
            self._on_set_destination(uri, experiment_name)

    def start_run(self, run_id: str | None = None, description: str | None = None) -> RunHandle:
        self._active = True
        return RunHandle(run_id=run_id or self._default_run_id, experiment_id=self._experiment_id)

    def end_run(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._on_end_run is not None:
            self._on_end_run()

    @contextmanager
    def start_span(
        self,
        name: str,
        span_type: CaptureSpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[_FakeSpanHandle]:
        yield _FakeSpanHandle()

    def autolog_pydantic_ai(self) -> None:
        pass

    @contextmanager
    def start_case_grouping(
        self,
        case_id: str,
        name: str,
        inputs: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[CaseGroupingHandle]:
        yield CaseGroupingHandle(mode="session", session_id=case_id, case_trace_id=None)

    # -- TraceQueryBackend ----------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace | None:
        return None

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> tuple[Trace | None, bool]:
        if self._await_trace_hook is not None:
            return self._await_trace_hook(
                trace_id,
                run_id=run_id,
                experiment_id=experiment_id,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
        return None, True

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        pass

    def delete_judge_traces(self, experiment_id: str, run_id: str) -> None:
        pass

    # -- ResultsBackend -------------------------------------------------------

    def log_metric(self, name: str, value: float) -> None:
        pass

    def log_params(self, params: Mapping[str, str]) -> None:
        pass

    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        pass

    def log_assessment(self, trace_id: str, assessment: Assessment) -> None:
        self.assessments.append((trace_id, assessment))

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        self.trace_tags[(trace_id, key)] = value

    def set_run_tag(self, run_id: str, key: str, value: str) -> None:
        self.run_tags[(run_id, key)] = value

    def get_run_tag(self, run_id: str, key: str) -> str | None:
        return self.run_tags.get((run_id, key))

    def delete_run_artifact(self, run_id: str, artifact_path: str) -> None:
        pass

    # -- LifecycleBackend -----------------------------------------------------

    def resolve_experiment_id(self, experiment_name: str) -> str:
        return self._experiment_id

    def get_tracking_uri(self) -> str | None:
        return self._tracking_uri

    def set_tracking_uri(self, uri: str) -> None:
        self._tracking_uri = uri

    def is_run_active(self) -> bool:
        return self._active


@pytest.fixture
def make_fake_backend() -> Iterator[Callable[..., FakeTrackingBackend]]:
    """Factory fixture: build a :class:`FakeTrackingBackend` (kwargs forwarded)
    and install it via ``configure_backend``; the registry is reset afterwards."""

    def _make(**kwargs: Any) -> FakeTrackingBackend:
        backend = FakeTrackingBackend(**kwargs)
        configure_backend(lambda: backend)
        return backend

    try:
        yield _make
    finally:
        reset_backend()


@pytest.fixture
def fake_tracking_backend(make_fake_backend: Callable[..., FakeTrackingBackend]) -> FakeTrackingBackend:
    """A default :class:`FakeTrackingBackend`, installed as the active backend."""
    return make_fake_backend()
