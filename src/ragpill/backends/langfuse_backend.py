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
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from ragpill.backends._common import NoopResultsMixin, SyntheticRunMixin, is_http_not_found, logger, poll_for_trace
from ragpill.backends._types import Assessment, CaseGroupingHandle, SpanKind
from ragpill.trace.model import Span as RagpillSpan, SpanKind as IngestSpanKind, Trace as RagpillTrace

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
    """Wraps a Langfuse observation to satisfy the ``SpanHandle`` protocol."""

    def __init__(self, span: Any) -> None:
        self._span = span

    @property
    def span_id(self) -> str:
        return str(getattr(self._span, "id", "") or "")

    @property
    def trace_id(self) -> str:
        return str(getattr(self._span, "trace_id", "") or "")

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.update(metadata={key: value})

    def set_inputs(self, value: Any) -> None:
        self._span.update(input=value)

    def set_outputs(self, value: Any) -> None:
        self._span.update(output=value)


class LangfuseBackend(SyntheticRunMixin, NoopResultsMixin):
    """Adapter implementing the tracking backend protocols against Langfuse."""

    # Langfuse is a remote service; the execution layer must not hand it a
    # temp SQLite URI — ``uri=None`` lets the client fall back to LANGFUSE_HOST.
    supports_local_file_store = False

    def __init__(self) -> None:
        self._host: str | None = None
        self._project_name = "ragpill"
        self._client: Any = None
        self._run_active = False

    def _get_client(self) -> Any:
        if self._client is None:
            _require_langfuse()
            from langfuse import Langfuse

            self._client = Langfuse(host=self._host) if self._host else Langfuse()
        return self._client

    def _flush(self) -> None:
        if self._client is not None:
            self._client.flush()

    # ------------------------------------------------------------------
    # TraceCaptureBackend
    # ------------------------------------------------------------------

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        # Langfuse has no "experiment"; the name is kept for tagging/grouping.
        self._host = uri
        self._project_name = experiment_name
        self._client = None  # rebuilt lazily with the new host
        self._get_client()

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

    def get_trace(self, trace_id: str) -> RagpillTrace | None:
        client = self._get_client()
        try:
            native = client.api.trace.get(trace_id)
        except Exception as exc:
            # 404 / NotFound is a legitimate "poll again" miss; anything else
            # (auth, connection, 5xx) is a real error that must surface instead
            # of masquerading as an in-flight trace.
            if is_http_not_found(exc):
                return None
            logger.warning("LangfuseBackend.get_trace failed for %s: %s", trace_id, exc)
            raise
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
        del run_id, experiment_id
        # Observations arrive in independent ingestion batches, so a readable
        # trace can still be missing in-flight spans — require a stable span set.
        return poll_for_trace(
            lambda: self.get_trace(trace_id),
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            stable_span_set=True,
        )

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        _ = experiment_id
        if not trace_ids:
            return
        client = self._get_client()
        for tid in trace_ids:
            try:
                client.api.trace.delete(tid)
            except Exception as exc:
                # Don't abort the batch on one failure, but don't lose it either.
                logger.warning("LangfuseBackend.delete_traces: failed to delete %s: %s", tid, exc)
                continue

    def delete_judge_traces(self, experiment_id: str, run_id: str) -> None:
        # Langfuse can delete traces, but locating judge traces requires a
        # server-side metadata query that is a later phase; warn rather than
        # silently skipping.
        _ = experiment_id, run_id
        self._warn_unsupported("delete_judge_traces")

    # ------------------------------------------------------------------
    # ResultsBackend — metrics/params/tables/artifacts are warn-once no-ops
    # (NoopResultsMixin); only assessments and tags have native equivalents.
    # ------------------------------------------------------------------

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
        parent_id=(str(getattr(obs, "parent_observation_id", None) or "") or None),
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
