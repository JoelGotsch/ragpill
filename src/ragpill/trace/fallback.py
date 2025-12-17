"""Universal best-effort span extractor.

Used by :func:`ragpill.trace.loader.parse_otel` when no registered adapter
recognises a span. Pulls whatever common fields are present (ids, timing,
generic ``input.value`` / ``output.value`` blobs) into a :class:`Span` tagged
``dialect="unknown"`` so the span still appears, just weakly typed.
"""

from __future__ import annotations

from typing import Any

from ragpill.trace.model import Span, SpanKind


def universal_span(span: dict[str, Any]) -> Span:
    """Best-effort conversion of an unrecognised span dict into a ``Span``."""
    attributes: dict[str, Any] = dict(span.get("attributes") or {})
    status: dict[str, Any] = span.get("status") or {}
    # Generic OpenInference-ish / common payload keys, if present.
    inputs = attributes.get("input.value")
    outputs = attributes.get("output.value")
    return Span(
        span_id=str(span.get("span_id", "")),
        parent_id=span.get("parent_span_id"),
        trace_id=str(span.get("trace_id", "")),
        name=str(span.get("name", "")),
        kind=SpanKind.UNKNOWN,
        start_time_ns=int(span.get("start_time_unix_nano") or 0),
        end_time_ns=int(span.get("end_time_unix_nano") or 0),
        status=str(status.get("code", "UNSET")),
        status_message=status.get("message"),
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        events=list(span.get("events") or []),
        dialect="unknown",
    )
