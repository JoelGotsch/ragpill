"""Loader: convert vendor trace objects into ``ragpill.trace.Trace``.

The loader owns all *input-format* normalisation. It turns a vendor trace
(here, ``mlflow.entities.Trace``) into the normalised OTLP-JSON span dicts
that :class:`~ragpill.trace.adapters.SpanAdapter` implementations consume,
dispatches each span through the right adapter, and assembles the
:class:`~ragpill.trace.Trace`.

Phase 1 ships only the MLflow input slice (:func:`from_mlflow_trace`). The
multi-format ``parse_otel(source, dialect="auto")`` entry point with the
entry-point registry and per-span auto-detection lands in Phase 3 (see
``designs/otel-trace-ingestion.md`` §6.4). Until then, MLflow is the only
capture format ragpill produces, so a single typed entry point is enough
and keeps the surface honest about what actually works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ragpill.trace.adapters.mlflow_adapter import MLflowAdapter
from ragpill.trace.model import Span, Trace

if TYPE_CHECKING:
    from mlflow.entities import Span as MLflowSpan, Trace as MLflowTrace


def _mlflow_span_to_dict(span: MLflowSpan) -> dict[str, Any]:
    """Normalise one ``mlflow.entities.Span`` into the adapter's span-dict shape.

    ``span.attributes`` already returns **decoded** values (MLflow stores the
    raw OTel attributes as JSON strings but the ``Span`` accessor decodes
    them), so the dict carries plain Python values, not JSON text.
    """
    status = getattr(span, "status", None)
    code = "UNSET"
    message: str | None = None
    if status is not None:
        # MLflow's SpanStatus exposes an OTel-proto status-code name.
        status_code = getattr(status, "status_code", None)
        if status_code is not None and hasattr(status_code, "to_otel_proto_status_code_name"):
            code = status_code.to_otel_proto_status_code_name()
        message = getattr(status, "description", None) or None

    # mlflow.entities ships no type stubs, so cast the event list to Any to
    # keep the comprehension readable without Unknown propagating outward.
    raw_events = cast("list[Any]", getattr(span, "events", None) or [])
    events = [
        {
            "name": getattr(ev, "name", ""),
            "time_unix_nano": getattr(ev, "timestamp", None),
            "attributes": dict(getattr(ev, "attributes", {}) or {}),
        }
        for ev in raw_events
    ]

    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_id,
        "name": span.name,
        "kind": None,  # OTel SpanKind isn't surfaced by MLflow; LLM kind comes from attributes.
        "start_time_unix_nano": span.start_time_ns,
        "end_time_unix_nano": span.end_time_ns,
        "attributes": dict(span.attributes or {}),
        "events": events,
        "status": {"code": code, "message": message},
    }


def from_mlflow_trace(mlflow_trace: MLflowTrace) -> Trace:
    """Convert a ``mlflow.entities.Trace`` into a vendor-neutral ``Trace``.

    Every span is routed through :class:`MLflowAdapter`. Trace-level fields are
    filled in best-effort from MLflow's ``TraceInfo`` (tags / metadata are
    surfaced as-is so users can still read them).
    """
    info = mlflow_trace.info
    raw_spans = mlflow_trace.data.spans or []
    spans: list[Span] = [MLflowAdapter.from_otel(_mlflow_span_to_dict(s)) for s in raw_spans]

    tags = dict(getattr(info, "tags", None) or {})
    return Trace(
        trace_id=getattr(info, "trace_id", "") or "",
        spans=spans,
        tags=sorted(tags.keys()),
        metadata=tags,
        dialect=MLflowAdapter.name,
    )
