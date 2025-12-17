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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from ragpill.backends._types import Assessment, CaptureSpanKind, CaseGroupingHandle, RunHandle

if TYPE_CHECKING:
    from ragpill.trace import Trace as NeutralTrace


@runtime_checkable
class SpanHandle(Protocol):
    """Surface ragpill reads and writes on a span opened via ``start_span``.

    Implementations must expose the ids as strings (empty string when the
    backend cannot provide one) and accept arbitrary JSON-serializable values
    for attributes/inputs/outputs. MLflow's native span object already
    satisfies this; Langfuse/Phoenix wrap their span objects.
    """

    @property
    def span_id(self) -> str: ...

    @property
    def trace_id(self) -> str: ...

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_inputs(self, value: Any) -> None: ...

    def set_outputs(self, value: Any) -> None: ...


@runtime_checkable
class TraceCaptureBackend(Protocol):
    """Configuration + write side of trace capture during ``execute_dataset``.

    Adapters additionally expose a ``supports_local_file_store`` class
    attribute (default ``False`` when absent). ``True`` means the backend can
    write to a local file/SQLite store, so the execution layer may synthesize
    a temp-directory URI for it when no destination is given. Backends that
    talk to a remote service must leave it ``False`` — they receive
    ``uri=None`` instead and fall back to their own environment-derived
    destination (e.g. ``LANGFUSE_HOST`` / ``PHOENIX_COLLECTOR_ENDPOINT``).
    """

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
        span_type: CaptureSpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[SpanHandle]:
        """Return a context manager yielding a :class:`SpanHandle`.

        MLflow's ``mlflow.start_span`` span already satisfies the handle
        protocol, so the MLflow adapter returns it unchanged; other adapters
        wrap their native span object.

        ``attributes``, when given, are set on the span at open time (all
        adapters honour this); callers may also set attributes afterwards via
        the yielded handle's ``set_attribute``.
        """
        ...

    def autolog_pydantic_ai(self) -> None:
        """Install instrumentation so ``pydantic-ai`` calls land as spans."""
        ...

    def start_case_grouping(
        self,
        case_id: str,
        name: str,
        inputs: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[CaseGroupingHandle]:
        """Open a case-level grouping under which the case's repeats relate.

        Two implementation strategies, advertised on the yielded
        :class:`CaseGroupingHandle`:

        - **``"session"`` mode** (preferred when the backend has a native
          sessions concept): the adapter registers ``case_id`` as the
          session id and arranges for each child trace opened via
          :meth:`start_span` inside this context to be tagged with that
          session id. No parent span is opened. The Sessions UI then
          groups one case = one session, one repeat = one turn.

        - **``"span"`` mode** (fallback for backends without sessions): the
          adapter opens a parent span (with ``inputs``/``attributes`` set
          on it) and lets child spans nest under it as today. The execution
          layer reads ``handle.case_trace_id`` to drive its existing
          per-repeat subtree filtering.

        Implementations choose one mode. The execution layer branches on
        ``handle.mode`` to assemble :class:`~ragpill.execution.CaseRunOutput`
        correctly. See ``plans/sessions-for-case-grouping.md`` for the
        backing analysis.
        """
        ...


@runtime_checkable
class TraceQueryBackend(Protocol):
    """Read side of the tracing store, used by evaluate/upload layers."""

    def get_trace(self, trace_id: str) -> NeutralTrace | None:
        """Fetch a single trace by id as a neutral ``ragpill.trace.Trace``.

        The adapter converts its native trace before returning, so callers
        (evaluators, the execution layer) never see backend-specific shapes.

        Returns ``None`` only for a genuine not-found / not-yet-exported trace.
        Transport, auth, and server errors must be raised, not swallowed — the
        polling loop treats a raised error as a hard failure and aborts, rather
        than mistaking an outage for an in-flight trace and burning the whole
        fetch budget.
        """
        ...

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> NeutralTrace | None:
        """Fetch ``trace_id`` as a neutral ``ragpill.trace.Trace``, polling until exported.

        Backends flush spans to their store asynchronously, so a fetch issued
        immediately after a span context closes can miss a trace that is still
        in flight. This polls up to ``timeout_s`` (every ``poll_interval_s``)
        for the trace to become available, converting the native trace to the
        neutral model before returning.

        Returns the trace with its full span tree once available, or ``None``
        on timeout. It MUST NOT fall back to a different trace: a miss returns
        ``None``, never the wrong trace. ``run_id`` / ``experiment_id`` are
        accepted for backends whose readiness query needs them.
        """
        ...

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        """Delete traces by id."""
        ...

    def delete_judge_traces(self, experiment_id: str, run_id: str) -> None:
        """Delete LLM-judge evaluation traces created during ``evaluate_results``.

        Judge traces are marked with the ``ragpill_is_judge_trace`` span
        attribute by :class:`~ragpill.evaluators.LLMJudge`; each backend knows
        how its own store surfaces that attribute. Called at the end of upload
        so the tracing UI only shows task traces. Backends that cannot delete
        traces no-op with a one-time warning.
        """
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

    def set_run_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a key/value tag on the run itself (not a trace).

        Used by the upload layer to record its progress (an upload-state marker)
        so a re-run is idempotent. Backends without a native run concept no-op.
        """
        ...

    def get_run_tag(self, run_id: str, key: str) -> str | None:
        """Read a run tag, or ``None`` when absent. Backends without a native
        run concept return ``None`` (so the idempotency guard simply proceeds)."""
        ...

    def delete_run_artifact(self, run_id: str, artifact_path: str) -> None:
        """Delete a previously-logged run artifact if present.

        Lets the upload layer replace an append-only artifact (e.g. MLflow's
        ``log_table``) on retry instead of duplicating rows. No-op when absent
        or unsupported.
        """
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
