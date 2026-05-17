"""Capability protocols ragpill expects from a tracking backend.

The protocols are derived directly from the existing call sites in
``upload.py``, ``execution.py`` and ``evaluators.py``. They are intentionally
small — only what ragpill already uses, nothing speculative. See
``plans/multi-backend-tracking.md`` for the call-site inventory.

The four buckets:

- ``TraceCaptureBackend`` — write-time: configure destination, start runs
  and spans, autolog the agent framework.
- ``TraceQueryBackend`` — read-time: search/get/delete traces.
- ``ResultsBackend`` — persist metrics, params, tables, artifacts,
  assessments, and trace tags.
- ``LifecycleBackend`` — experiment lookup and run reattachment.

A real adapter typically implements all four; the combined
``Backend`` protocol exists for that common case.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ragpill.backends._types import Assessment, RunHandle, SpanKind

# Phase 1 short-cut (see plans/multi-backend-tracking.md Step 1.2(a)):
# The internal trace type is still mlflow's. Non-MLflow adapters convert
# their native trace into this shape. Phase 2 swaps this for a normalised
# dataclass produced by the OTel ingestion plan.
Trace = Any  # alias for type-checker readability


@runtime_checkable
class TraceCaptureBackend(Protocol):
    """Configuration + write side of trace capture during ``execute_dataset``."""

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        """Point future writes at this destination.

        ``uri`` ``None`` means "use the backend's default" (e.g. MLflow's
        environment-derived tracking URI). ``experiment_name`` is the grouping
        bucket (project / experiment / namespace, depending on the backend).
        """
        ...

    def start_run(
        self,
        run_id: str | None = None,
        description: str | None = None,
    ) -> RunHandle:
        """Open a run. When ``run_id`` is given, reattach to that existing run."""
        ...

    def end_run(self) -> None:
        """Close the active run if any. No-op when no run is active."""
        ...

    def start_span(
        self,
        name: str,
        span_type: SpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        """Return a context manager yielding a span handle.

        The yielded object must expose ``set_attribute``, ``set_inputs`` and
        ``set_outputs``. MLflow's ``mlflow.start_span`` already does, so the
        MLflow adapter returns it unchanged.
        """
        ...

    def autolog_pydantic_ai(self) -> None:
        """Install instrumentation so ``pydantic-ai`` calls land as spans."""
        ...


@runtime_checkable
class TraceQueryBackend(Protocol):
    """Read side of the tracing store, used by evaluate/upload layers."""

    def search_traces(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        max_results: int = 1000,
    ) -> list[Trace]:
        """List traces, optionally filtered by run and/or experiment."""
        ...

    def get_trace(self, trace_id: str) -> Trace | None:
        """Fetch a single trace by id, or ``None`` when not found."""
        ...

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        """Delete traces by id. Used today for judge-trace cleanup."""
        ...


@runtime_checkable
class ResultsBackend(Protocol):
    """Persistence side of the upload layer."""

    def log_metric(self, name: str, value: float) -> None:
        """Record a numeric metric for the active run."""
        ...

    def log_params(self, params: Mapping[str, str]) -> None:
        """Record string parameters for the active run."""
        ...

    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None:
        """Persist a DataFrame as a structured artifact. Backends without a
        native table concept may fall back to a JSON/CSV artifact."""
        ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a local file as a run artifact."""
        ...

    def log_assessment(self, trace_id: str, assessment: Assessment) -> None:
        """Attach an evaluator verdict to a trace."""
        ...

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        """Attach a searchable tag to a trace."""
        ...


@runtime_checkable
class LifecycleBackend(Protocol):
    """Experiment + run discovery used outside the trace-capture path."""

    def resolve_experiment_id(self, experiment_name: str) -> str:
        """Look up the backend's id for a named experiment. Raise on miss."""
        ...

    def get_tracking_uri(self) -> str | None:
        """Return the currently active tracking URI, or ``None``."""
        ...

    def set_tracking_uri(self, uri: str) -> None:
        """Set the active tracking URI (used to save/restore around upload)."""
        ...

    def is_run_active(self) -> bool:
        """Whether a run is currently active. Used to decide whether to
        ``end_run`` in finally blocks."""
        ...


@runtime_checkable
class Backend(
    TraceCaptureBackend,
    TraceQueryBackend,
    ResultsBackend,
    LifecycleBackend,
    Protocol,
):
    """Combined surface a full-featured adapter implements.

    Adapters that genuinely cannot implement one bucket (e.g. a backend with
    no trace query API) can opt out by raising ``NotImplementedError`` from
    the unsupported methods; the registry helpers ``get_results_backend``
    etc. (added in a later step) return narrower interfaces when needed.
    """
