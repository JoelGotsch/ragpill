"""MLflow adapter for the tracking backend protocols.

The default backend. Mostly a thin forwarder to ``mlflow.*`` plus the glue
MLflow needs for ragpill's session-mode case grouping (``mlflow.trace.session``
metadata) and judge-trace cleanup (native root-span introspection). All
MLflow-specific knowledge — trace shapes, metric-name charset, session
metadata keys — is confined to this module.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Feedback, SpanType

if TYPE_CHECKING:
    from mlflow import MlflowClient

    from ragpill.trace import Trace as NeutralTrace

from ragpill.backends._common import JUDGE_TRACE_TAG, RemoteQueryMixin, is_http_not_found, logger
from ragpill.backends._types import Assessment, CaptureSpanKind, CaseGroupingHandle, RunHandle

# MLflow restricts metric names to alphanumerics, `_`, `.`, `/`, space and `-`;
# other backends have their own rules, so the slugging lives here, not in the
# shared upload layer.
_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_./ -]+")

# Upper bound for judge-trace search. mlflow.search_traces auto-paginates up to
# this, so it removes the old silent 1000-trace cap without truly unbounded reads.
_JUDGE_TRACE_SEARCH_LIMIT = 1_000_000


_SPAN_KIND_TO_MLFLOW: dict[CaptureSpanKind, str] = {
    CaptureSpanKind.AGENT: SpanType.AGENT,
    CaptureSpanKind.CHAT_MODEL: SpanType.CHAT_MODEL,
    CaptureSpanKind.LLM: SpanType.LLM,
    CaptureSpanKind.RERANKER: SpanType.RERANKER,
    CaptureSpanKind.RETRIEVER: SpanType.RETRIEVER,
    CaptureSpanKind.TASK: SpanType.TASK,
    CaptureSpanKind.TOOL: SpanType.TOOL,
    CaptureSpanKind.UNKNOWN: SpanType.UNKNOWN,
}


# Session id + case-level metadata active for the current case-grouping context.
# When set, each ``start_span`` inside the context tags its trace with MLflow's
# ``mlflow.trace.session`` metadata (so the Sessions UI groups a case's repeats
# as turns) plus the case-level name/attributes, which have no parent span in
# session mode. Held in ContextVars, not instance state, so two concurrent
# ``execute_dataset`` calls (each an asyncio Task with its own copied context)
# don't cross-tag each other's traces through the shared backend singleton.
_active_session_id: ContextVar[str | None] = ContextVar("ragpill_mlflow_active_session_id", default=None)
_active_session_metadata: ContextVar[dict[str, str]] = ContextVar("ragpill_mlflow_active_session_metadata", default={})


class MLflowBackend(RemoteQueryMixin):
    """Adapter forwarding to ``mlflow.*``.

    Implements ``TraceCaptureBackend``, ``TraceQueryBackend``,
    ``ResultsBackend``, and ``LifecycleBackend``. The combined ``Backend``
    protocol is the natural shape.
    """

    # MLflow's by-id lookup returns the full span tree at once, so the shared
    # await_trace polls without the span-set-stability check.
    _stable_span_set = False

    # MLflow can track to a local SQLite store, so the execution layer may
    # synthesize a temp-directory URI when no destination is given.
    supports_local_file_store: ClassVar[bool] = True

    def _client(self) -> MlflowClient:
        """Fresh MlflowClient bound to the current tracking URI. Not cached:
        callers (including our own tests) can repoint mlflow's global
        tracking URI at any time via ``mlflow.set_tracking_uri``, and a
        cached client would silently keep talking to the old store."""
        from mlflow import MlflowClient

        return MlflowClient()

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
        span_type: CaptureSpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        inner = mlflow.start_span(name=name, span_type=_SPAN_KIND_TO_MLFLOW[span_type])
        span_attributes = dict(attributes) if attributes else None
        session_id = _active_session_id.get()
        session_metadata = dict(_active_session_metadata.get())

        @contextmanager
        def wrapped() -> Generator[Any, None, None]:
            with inner as span:
                # Apply open-time attributes for parity with the Phoenix/Langfuse
                # adapters (callers may still use ``set_attribute`` afterwards).
                if span_attributes:
                    for key, value in span_attributes.items():
                        span.set_attribute(key, value)
                # Promote the judge marker to a *trace tag* so cleanup can filter
                # server-side (``tags.ragpill_is_judge_trace = 'true'``) instead
                # of downloading every task trace's payload to inspect a root span.
                if span_attributes and span_attributes.get(JUDGE_TRACE_TAG):
                    mlflow.update_current_trace(tags={JUDGE_TRACE_TAG: "true"})
                # When inside a case-grouping context, tag this span's trace
                # with the MLflow session id (so the Sessions UI groups repeats
                # of the same case as turns) plus the case-level metadata,
                # which has no parent span to live on in session mode. Done
                # lazily on enter so we operate on the actual active trace.
                if session_id is not None:
                    mlflow.update_current_trace(metadata={**session_metadata, "mlflow.trace.session": session_id})
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

        There is no parent span in session mode, so the case-level ``name``
        and ``attributes`` are carried as trace metadata on every per-repeat
        trace instead (``inputs`` are already recorded on each per-repeat
        span by the execution layer).
        """
        _ = inputs
        metadata = {"ragpill.case_name": name}
        metadata.update({k: str(v) for k, v in (attributes or {}).items()})

        @contextmanager
        def cm() -> Generator[CaseGroupingHandle, None, None]:
            id_token = _active_session_id.set(case_id)
            meta_token = _active_session_metadata.set(metadata)
            try:
                yield CaseGroupingHandle(mode="session", session_id=case_id, case_trace_id=None)
            finally:
                _active_session_id.reset(id_token)
                _active_session_metadata.reset(meta_token)

        return cm()

    # ------------------------------------------------------------------
    # TraceQueryBackend
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> NeutralTrace | None:
        # Returns the vendor-neutral ragpill.trace.Trace (converted here), not the
        # raw mlflow.entities.Trace — every backend converts its own native trace
        # so the execution layer stays backend-agnostic. See ADR-0017.
        from mlflow.exceptions import MlflowException

        from ragpill.trace import from_mlflow_trace

        try:
            native = self._client().get_trace(trace_id)
        except MlflowException as e:
            # A not-yet-exported or unknown trace is a legitimate miss — the
            # polling loop retries. Anything else (auth, connection, server
            # error) must surface so it isn't mistaken for an in-flight trace
            # and silently burned as a poll timeout.
            if is_http_not_found(e):
                return None
            raise
        # mlflow's stub types this non-Optional, but a not-yet-exported trace can
        # come back falsy at runtime — keep the guard.
        if not native:
            return None
        return from_mlflow_trace(native)

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        if not trace_ids:
            return
        self._client().delete_traces(experiment_id=experiment_id, trace_ids=trace_ids)

    def delete_judge_traces(self, experiment_id: str, run_id: str) -> None:
        # Judge spans promote the ``ragpill_is_judge_trace`` marker to a trace
        # tag (see start_span), so we filter server-side and get back *only* the
        # judge traces — no downloading of every task trace's span payload to
        # inspect a root span. Only ``trace.info.trace_id`` is read, so
        # ``include_spans=False`` skips the span payloads entirely.
        # ``mlflow.search_traces`` auto-paginates internally up to
        # ``max_results`` (a large cap avoids the old silent 1000 cap).
        traces: list[Any] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
            return_type="list",
            run_id=run_id,
            locations=[experiment_id],
            filter_string=f"tags.{JUDGE_TRACE_TAG} = 'true'",
            max_results=_JUDGE_TRACE_SEARCH_LIMIT,
            include_spans=False,
        )
        judge_trace_ids: list[str] = [trace.info.trace_id for trace in traces]
        if judge_trace_ids:
            logger.info("Deleting %d judge trace(s) from run %s.", len(judge_trace_ids), run_id)
            self.delete_traces(experiment_id=experiment_id, trace_ids=judge_trace_ids)
            return
        # Traces persisted by pre-tag ragpill versions carry the judge marker
        # only as a root-span *attribute*, which the tag filter can never match.
        # One-shot fallback sweep: it only fires when nothing is tagged, so the
        # normal path stays cheap. This search needs the span payloads
        # (``include_spans=True``) to inspect root-span attributes; native
        # introspection (``trace.data._get_root_span`` is not public API) is
        # confined here — only this adapter knows MLflow's trace shape.
        legacy_traces: list[Any] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
            return_type="list",
            run_id=run_id,
            locations=[experiment_id],
            max_results=_JUDGE_TRACE_SEARCH_LIMIT,
            include_spans=True,
        )
        legacy_trace_ids: list[str] = []
        for trace in legacy_traces:
            root = trace.data._get_root_span()
            if root and root.attributes.get(JUDGE_TRACE_TAG):
                legacy_trace_ids.append(trace.info.trace_id)
        if legacy_trace_ids:
            logger.info(
                "Cleaned %d legacy judge trace(s) from run %s "
                "(pre-tag ragpill marked judge traces via a root-span attribute only).",
                len(legacy_trace_ids),
                run_id,
            )
        self.delete_traces(experiment_id=experiment_id, trace_ids=legacy_trace_ids)

    # ------------------------------------------------------------------
    # ResultsBackend
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float) -> None:
        # Sanitize to MLflow's metric-name charset; other backends have their
        # own naming rules, so callers pass raw names.
        mlflow.log_metric(_METRIC_NAME_RE.sub("_", name), value)

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

    def set_run_tag(self, run_id: str, key: str, value: str) -> None:
        self._client().set_tag(run_id, key, value)

    def get_run_tag(self, run_id: str, key: str) -> str | None:
        from mlflow.exceptions import MlflowException

        try:
            run: Any = self._client().get_run(run_id)
        except MlflowException as e:
            if is_http_not_found(e):
                return None
            raise
        tags: dict[str, str] = dict(run.data.tags or {})
        return tags.get(key)

    def delete_run_artifact(self, run_id: str, artifact_path: str) -> None:
        from mlflow.exceptions import MlflowException
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        client: Any = self._client()
        try:
            existing = {str(f.path) for f in client.list_artifacts(run_id)}
        except MlflowException as e:
            if is_http_not_found(e):
                return
            raise
        if artifact_path not in existing:
            return
        info: Any = client.get_run(run_id).info
        repo: Any = get_artifact_repository(info.artifact_uri)
        repo.delete_artifacts(artifact_path)

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
