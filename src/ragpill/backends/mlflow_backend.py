"""MLflow adapter for the tracking backend protocols.

Phase 1: pure forwarder. Each method maps to a single ``mlflow.*`` call (or
a tiny piece of glue) so behaviour is identical to the inline calls being
replaced. Call sites still talk to ``mlflow`` directly today; this adapter
is in place so subsequent steps can switch them over without changing
behaviour.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Feedback, SpanType, Trace as MLflowTrace

if TYPE_CHECKING:
    from ragpill.trace import Trace as NeutralTrace

from ragpill.backends._types import Assessment, CaseGroupingHandle, RunHandle, SpanKind

_SPAN_KIND_TO_MLFLOW: dict[SpanKind, str] = {
    SpanKind.AGENT: SpanType.AGENT,
    SpanKind.CHAT_MODEL: SpanType.CHAT_MODEL,
    SpanKind.LLM: SpanType.LLM,
    SpanKind.RERANKER: SpanType.RERANKER,
    SpanKind.RETRIEVER: SpanType.RETRIEVER,
    SpanKind.TASK: SpanType.TASK,
    SpanKind.TOOL: SpanType.TOOL,
    SpanKind.UNKNOWN: SpanType.UNKNOWN,
}


class MLflowBackend:
    """Adapter forwarding to ``mlflow.*``.

    Implements ``TraceCaptureBackend``, ``TraceQueryBackend``,
    ``ResultsBackend``, and ``LifecycleBackend``. The combined ``Backend``
    protocol is the natural shape.
    """

    def __init__(self) -> None:
        # Session id active for the current case-grouping context. When set,
        # each ``start_span`` call inside the context tags its trace with
        # MLflow's ``mlflow.trace.session`` metadata so the Sessions UI
        # groups repeats of a case as turns of one session.
        # Single-threaded by design: ragpill's execute_dataset processes
        # cases sequentially.
        self._active_session_id: str | None = None

    # ------------------------------------------------------------------
    # TraceCaptureBackend
    # ------------------------------------------------------------------

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        if uri is not None:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)  # pyright: ignore[reportUnknownMemberType]

    def start_run(
        self,
        run_id: str | None = None,
        description: str | None = None,
    ) -> RunHandle:
        run = mlflow.start_run(run_id=run_id, description=description)
        info: Any = run.info
        return RunHandle(
            run_id=str(info.run_id),
            experiment_id=str(info.experiment_id),
        )

    def end_run(self) -> None:
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def start_span(
        self,
        name: str,
        span_type: SpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        # ``attributes`` is reserved for future use (Phoenix/Langfuse pass
        # per-span attributes at open time); MLflow callers use the yielded
        # span's ``set_attribute`` method directly today.
        _ = attributes
        inner = mlflow.start_span(name=name, span_type=_SPAN_KIND_TO_MLFLOW[span_type])
        session_id = self._active_session_id

        @contextmanager
        def wrapped() -> Generator[Any, None, None]:
            with inner as span:
                # When inside a case-grouping context, tag this span's trace
                # with the MLflow session id so the Sessions UI groups
                # repeats of the same case as turns. Done lazily on enter so
                # we operate on the actual active trace.
                if session_id is not None:
                    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
                yield span

        return wrapped()

    def autolog_pydantic_ai(self) -> None:
        mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

    def start_case_grouping(
        self,
        case_id: str,
        name: str,
        inputs: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[CaseGroupingHandle]:
        """Session mode: register ``case_id`` as the MLflow session id and
        let each per-repeat ``start_span`` call open as a fresh top-level
        trace tagged with that session id.

        ``inputs`` and ``attributes`` are not surfaced on a parent span
        (there isn't one in session mode); callers can attach them to
        individual per-repeat spans via ``start_span``'s yielded handle.
        """
        # ``inputs`` and ``attributes`` are part of the protocol for
        # symmetry with span-mode backends; the MLflow session mode
        # surfaces those via individual per-repeat span attributes.
        _ = inputs, attributes, name

        @contextmanager
        def cm() -> Generator[CaseGroupingHandle, None, None]:
            previous = self._active_session_id
            self._active_session_id = case_id
            try:
                yield CaseGroupingHandle(mode="session", session_id=case_id, case_trace_id=None)
            finally:
                self._active_session_id = previous

        return cm()

    # ------------------------------------------------------------------
    # TraceQueryBackend
    # ------------------------------------------------------------------

    def search_traces(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        max_results: int = 1000,
    ) -> list[MLflowTrace]:
        kwargs: dict[str, Any] = {"return_type": "list", "max_results": max_results}
        if run_id is not None:
            kwargs["run_id"] = run_id
        if experiment_id is not None:
            kwargs["locations"] = [experiment_id]
        return mlflow.search_traces(**kwargs)  # pyright: ignore[reportReturnType]

    def get_trace(self, trace_id: str) -> NeutralTrace | None:
        # Returns the vendor-neutral ragpill.trace.Trace (converted here), not the
        # raw mlflow.entities.Trace — every backend converts its own native trace
        # so the execution layer stays backend-agnostic. See ADR-0017.
        from mlflow import MlflowClient

        from ragpill.trace import from_mlflow_trace

        try:
            native = MlflowClient().get_trace(trace_id)
        except Exception:
            return None
        # mlflow's stub types this non-Optional, but a not-yet-exported trace can
        # come back falsy at runtime — keep the guard.
        if not native:
            return None
        return from_mlflow_trace(native)

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> NeutralTrace | None:
        # MLflow exports spans asynchronously; the by-id lookup is the
        # authoritative readiness check — it returns the trace with its full
        # span tree only once exported. Poll it rather than search_traces (the
        # latter's "one result" state can transiently belong to a different
        # case, which is how the old fallback returned the wrong trace).
        # run_id / experiment_id are part of the protocol for backends whose
        # readiness query needs them; MLflow's by-id lookup does not.
        del run_id, experiment_id
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            trace = self.get_trace(trace_id)
            if trace is not None:
                return trace
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval_s)

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        from mlflow import MlflowClient

        if not trace_ids:
            return
        MlflowClient().delete_traces(experiment_id=experiment_id, trace_ids=trace_ids)

    # ------------------------------------------------------------------
    # ResultsBackend
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float) -> None:
        mlflow.log_metric(name, value)

    def log_params(self, params: Mapping[str, str]) -> None:
        mlflow.log_params(dict(params))

    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None:
        mlflow.log_table(df, artifact_file)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_assessment(self, trace_id: str, assessment: Assessment) -> None:
        feedback = Feedback(
            name=assessment.name,
            value=assessment.value,
            source=AssessmentSource(
                source_type=assessment.source_type,
                source_id=assessment.source_id,
            ),
            rationale=assessment.rationale,
        )
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        mlflow.set_trace_tag(trace_id, key, value)

    # ------------------------------------------------------------------
    # LifecycleBackend
    # ------------------------------------------------------------------

    def resolve_experiment_id(self, experiment_name: str) -> str:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            raise RuntimeError(f"Experiment '{experiment_name}' not found on server.")
        return str(exp.experiment_id)  # pyright: ignore[reportUnknownArgumentType]

    def get_tracking_uri(self) -> str | None:
        return mlflow.get_tracking_uri()

    def set_tracking_uri(self, uri: str) -> None:
        mlflow.set_tracking_uri(uri)

    def is_run_active(self) -> bool:
        return mlflow.active_run() is not None
