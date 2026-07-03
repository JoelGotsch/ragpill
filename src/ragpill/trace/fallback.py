"""Universal best-effort span extractor.

Used by :func:`ragpill.trace.loader.parse_otel` when no registered adapter
recognises a span. Pulls whatever common fields are present (ids, timing,
generic ``input.value`` / ``output.value`` blobs) into a :class:`Span` tagged
``dialect="unknown"`` so the span still appears, just weakly typed.
"""

from __future__ import annotations

from typing import Any

from ragpill.trace.adapters._base import common_span_fields
from ragpill.trace.model import Span, SpanKind


def universal_span(span: dict[str, Any]) -> Span:
    """Best-effort conversion of an unrecognised span dict into a ``Span``.

    Unlike the dialect adapters, this never declines — a missing span_id
    coerces to an empty string so the span still appears in the trace.
    """
    attributes: dict[str, Any] = dict(span.get("attributes") or {})
    return Span(
        **common_span_fields(span, dialect="unknown"),
        kind=SpanKind.UNKNOWN,
        # Generic OpenInference-ish / common payload keys, if present.
        inputs=attributes.get("input.value"),
        outputs=attributes.get("output.value"),
        attributes=attributes,
    )
