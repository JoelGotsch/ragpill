"""Langfuse adapter for the tracking backend protocols.

Implements the tracking backend protocols against Langfuse's v4 (OTel-based)
Python SDK. Selectable via ``configure_backend(LangfuseBackend)``.

All ``langfuse`` imports are **lazy** (inside methods) so this module imports
cleanly without the ``ragpill[langfuse]`` extra; only using a backend method
requires it.

Mapping highlights (see ``designs/langfuse-integration.md``):
- Assessments -> Langfuse **scores** (``create_score``), with data-type
  inferred from the value (BOOLEAN / NUMERIC / CATEGORICAL).
- Trace deletion **is** supported (``api.trace.delete``), unlike Phoenix.
- Langfuse has no native run/metric/param/table/artifact concept; those no-op
  with a one-time warning, per ``plans/multi-backend-tracking.md``.
- Reads convert Langfuse observations to the neutral ``ragpill.trace.Trace`` by
  direct field mapping (a dedicated langfuse trace dialect adapter is a later
  OTel-ingestion phase).

Scope note: the live path is covered by an env-gated integration test, not the
unit suite.
"""

from __future__ import annotations

# The langfuse SDK is an optional extra and isn't installed in the default
# type-check environment, so its imports and return types are unresolved here.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedImport=false
import warnings
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any

from ragpill.backends._types import Assessment, CaseGroupingHandle, RunHandle, SpanKind
from ragpill.trace.model import Span as RagpillSpan, SpanKind as IngestSpanKind, Trace as RagpillTrace

if TYPE_CHECKING:
    import pandas as pd

_INSTALL_HINT = (
    "The Langfuse backend requires the 'langfuse' extra. Install it with "
    "`pip install ragpill[langfuse]` and set LANGFUSE_PUBLIC_KEY / "
    "LANGFUSE_SECRET_KEY (and LANGFUSE_HOST for self-hosted)."
)

# ragpill write-side SpanKind -> Langfuse observation as_type.
_AS_TYPE: dict[SpanKind, str] = {
    SpanKind.AGENT: "agent",
    SpanKind.CHAT_MODEL: "generation",
    SpanKind.LLM: "generation",
    SpanKind.RERANKER: "span",
    SpanKind.RETRIEVER: "retriever",
    SpanKind.TASK: "chain",
    SpanKind.TOOL: "tool",
    SpanKind.UNKNOWN: "span",
}

# Langfuse observation type -> neutral ingest-side SpanKind.
_OBS_TYPE_TO_KIND: dict[str, IngestSpanKind] = {
    "GENERATION": IngestSpanKind.LLM,
    "SPAN": IngestSpanKind.CHAIN,
    "EVENT": IngestSpanKind.UNKNOWN,
    "AGENT": IngestSpanKind.AGENT,
    "TOOL": IngestSpanKind.TOOL,
    "RETRIEVER": IngestSpanKind.RETRIEVER,
    "CHAIN": IngestSpanKind.CHAIN,
}


def _require_langfuse() -> None:
    try:
        import langfuse  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_HINT) from exc


class _SpanHandle:
    """Wraps a Langfuse observation to expose the surface ragpill reads:
    ``span_id`` / ``request_id`` (trace id) + set_attribute/inputs/outputs."""

    def __init__(self, span: Any) -> None:
        self._span = span

    @property
    def span_id(self) -> str:
        return str(getattr(self._span, "id", "") or "")

    @property
    def request_id(self) -> str:
        return str(getattr(self._span, "trace_id", "") or "")

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.update(metadata={key: value})

    def set_inputs(self, value: Any) -> None:
        self._span.update(input=value)

    def set_outputs(self, value: Any) -> None:
        self._span.update(output=value)


class LangfuseBackend:
    """Adapter implementing the tracking backend protocols against Langfuse."""

    def __init__(self) -> None:
        self._host: str | None = None
        self._project_name = "ragpill"
        self._client: Any = None
        self._run_active = False
        self._warned: set[str] = set()

    def _warn_unsupported(self, capability: str) -> None:
        if capability not in self._warned:
            self._warned.add(capability)
            warnings.warn(
                f"LangfuseBackend: '{capability}' has no native Langfuse equivalent and is a no-op. "
                "See plans/multi-backend-tracking.md.",
                stacklevel=2,
            )

    def _get_client(self) -> Any:
        if self._client is None:
            _require_langfuse()
            from langfuse import Langfuse

            self._client = Langfuse(host=self._host) if self._host else Langfuse()
        return self._client

    # ------------------------------------------------------------------
    # TraceCaptureBackend
    # ------------------------------------------------------------------

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        # Langfuse has no "experiment"; the name is kept for tagging/grouping.
        self._host = uri
        self._project_name = experiment_name
        self._client = None  # rebuilt lazily with the new host
        self._get_client()

    def start_run(self, run_id: str | None = None, description: str | None = None) -> RunHandle:
        _ = description
        self._run_active = True
        rid = run_id or self._project_name
        return RunHandle(run_id=rid, experiment_id=self._project_name)

    def end_run(self) -> None:
        self._run_active = False
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

    def start_span(
        self,
        name: str,
        span_type: SpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        client = self._get_client()

        @contextmanager
        def cm() -> Generator[Any, None, None]:
            with client.start_as_current_observation(name=name, as_type=_AS_TYPE[span_type]) as span:
                if attributes:
                    span.update(metadata=dict(attributes))
                yield _SpanHandle(span)

        return cm()

    def autolog_pydantic_ai(self) -> None:
        _require_langfuse()
        # Langfuse ingests via OTel; pydantic-ai emits OTel natively when
        # instrumented. Enable global pydantic-ai instrumentation so its spans
        # flow to the Langfuse tracer provider. Best-effort across versions.
        try:
            from pydantic_ai.agent import Agent

            Agent.instrument_all()
        except Exception:
            pass

    def start_case_grouping(
        self,
        case_id: str,
        name: str,
        inputs: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[CaseGroupingHandle]:
        """Span mode: open a parent observation; per-repeat observations nest
        under it. The case id is also recorded as the trace session id."""
        client = self._get_client()

        @contextmanager
        def cm() -> Generator[CaseGroupingHandle, None, None]:
            with client.start_as_current_observation(name=name, as_type="chain", input=inputs) as span:
                meta = {"session_id": case_id, **(dict(attributes) if attributes else {})}
                span.update(metadata=meta)
                trace_id = str(getattr(span, "trace_id", "") or "")
                yield CaseGroupingHandle(mode="span", case_trace_id=trace_id, session_id=case_id)

        return cm()

    # ------------------------------------------------------------------
    # TraceQueryBackend
    # ------------------------------------------------------------------

    def search_traces(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        max_results: int = 1000,
    ) -> list[Any]:
        # Native traces are only consumed by the MLflow-specific judge-trace
        # cleanup path; for Langfuse, suppression is the exporter-filter route.
        _ = run_id, experiment_id, max_results
        return []

    def get_trace(self, trace_id: str) -> RagpillTrace | None:
        client = self._get_client()
        try:
            native = client.api.trace.get(trace_id)
        except Exception:
            return None
        return _trace_from_langfuse(native)

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> RagpillTrace | None:
        import time

        del run_id, experiment_id
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            trace = self.get_trace(trace_id)
            if trace is not None and trace.spans:
                return trace
            if time.monotonic() >= deadline:
                return trace
            time.sleep(poll_interval_s)

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        _ = experiment_id
        if not trace_ids:
            return
        client = self._get_client()
        for tid in trace_ids:
            try:
                client.api.trace.delete(tid)
            except Exception:
                continue

    # ------------------------------------------------------------------
    # ResultsBackend
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float) -> None:
        _ = name, value
        self._warn_unsupported("log_metric")

    def log_params(self, params: Mapping[str, str]) -> None:
        _ = params
        self._warn_unsupported("log_params")

    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None:
        _ = df, artifact_file
        self._warn_unsupported("log_table")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        _ = local_path, artifact_path
        self._warn_unsupported("log_artifact")

    def log_assessment(self, trace_id: str, assessment: Assessment) -> None:
        client = self._get_client()
        value: Any = assessment.value
        if isinstance(value, bool):
            data_type, score_value = "BOOLEAN", 1 if value else 0
        elif isinstance(value, (int, float)):
            data_type, score_value = "NUMERIC", value
        else:
            data_type, score_value = "CATEGORICAL", str(value)
        client.create_score(
            name=assessment.name,
            value=score_value,
            trace_id=trace_id,
            data_type=data_type,
            comment=assessment.rationale,
            metadata=dict(assessment.metadata) or None,
        )

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        # Langfuse tags are set at trace creation; post-hoc tagging of an
        # existing trace isn't a first-class SDK op. Surface as a categorical
        # score so the information is still attached to the trace.
        self._get_client().create_score(name=key, value=value, trace_id=trace_id, data_type="CATEGORICAL")

    # ------------------------------------------------------------------
    # LifecycleBackend
    # ------------------------------------------------------------------

    def resolve_experiment_id(self, experiment_name: str) -> str:
        return experiment_name

    def get_tracking_uri(self) -> str | None:
        return self._host

    def set_tracking_uri(self, uri: str) -> None:
        self._host = uri
        self._client = None

    def is_run_active(self) -> bool:
        return self._run_active


# ---------------------------------------------------------------------------
# Langfuse trace -> neutral trace (direct field mapping; no dialect adapter yet)
# ---------------------------------------------------------------------------


def _trace_from_langfuse(native: Any) -> RagpillTrace | None:
    observations = list(getattr(native, "observations", None) or [])
    trace_id = str(getattr(native, "id", "") or "")
    spans = [_observation_to_span(obs, trace_id) for obs in observations]
    if not spans:
        return None
    return RagpillTrace(trace_id=trace_id, spans=spans, dialect="langfuse")


def _observation_to_span(obs: Any, trace_id: str) -> RagpillSpan:
    obs_type = str(getattr(obs, "type", "") or "").upper()
    usage_details = getattr(obs, "usage_details", None) or {}
    from ragpill.trace.model import Usage

    usage = Usage(
        input_tokens=usage_details.get("input"),
        output_tokens=usage_details.get("output"),
        total_tokens=usage_details.get("total"),
    )
    return RagpillSpan(
        span_id=str(getattr(obs, "id", "") or ""),
        parent_id=(str(getattr(obs, "parent_observation_id", "")) or None),
        trace_id=trace_id,
        name=str(getattr(obs, "name", "") or ""),
        kind=_OBS_TYPE_TO_KIND.get(obs_type, IngestSpanKind.UNKNOWN),
        start_time_ns=0,
        end_time_ns=0,
        inputs=getattr(obs, "input", None),
        outputs=getattr(obs, "output", None),
        model=getattr(obs, "model", None),
        usage=usage,
        attributes=dict(getattr(obs, "metadata", None) or {}),
        dialect="langfuse",
    )
