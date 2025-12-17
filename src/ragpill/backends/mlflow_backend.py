"""MLflow adapter for the tracking backend protocols.

Phase 1: pure forwarder. Each method maps to a single ``mlflow.*`` call (or
a tiny piece of glue) so behaviour is identical to the inline calls being
replaced. Call sites still talk to ``mlflow`` directly today; this adapter
is in place so subsequent steps can switch them over without changing
behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Feedback, SpanType, Trace as MLflowTrace

from ragpill.backends._types import Assessment, RunHandle, SpanKind

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
        cm = mlflow.start_span(name=name, span_type=_SPAN_KIND_TO_MLFLOW[span_type])
        if attributes:
            # MLflow's context manager exposes attribute methods on the yielded
            # span. Stash the attributes for the caller to apply after enter.
            # Keeping the protocol simple: callers using attributes use the
            # ``set_attribute`` method on the yielded span directly today, so
            # we don't pre-set anything here in Step 1.
            pass
        return cm

    def autolog_pydantic_ai(self) -> None:
        mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

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

    def get_trace(self, trace_id: str) -> MLflowTrace | None:
        from mlflow import MlflowClient

        try:
            return MlflowClient().get_trace(trace_id)
        except Exception:
            return None

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
