"""``SpanAdapter`` base class (Option C — see ``designs/otel-trace-ingestion.md`` §6.2).

Each dialect adapter converts a single **normalised OTLP-JSON span dict**
into a :class:`ragpill.trace.Span`. Adapters never touch vendor SDK trace
objects — the loader (:mod:`ragpill.trace.loader`) is responsible for
normalising every input format (``mlflow.entities.Trace``, OTLP-JSON,
Phoenix/Langfuse exports, …) into the span-dict shape below before handing
spans to an adapter. This keeps adapters small, format-agnostic, and
independently shippable (a third party can register one via entry points
without forking ragpill).

The normalised span dict has these keys (the loader guarantees them):

    {
      "trace_id": str,
      "span_id": str,
      "parent_span_id": str | None,
      "name": str,
      "kind": str | None,                 # OTel SpanKind, *not* the LLM kind
      "start_time_unix_nano": int,
      "end_time_unix_nano": int,
      "attributes": dict[str, Any],       # decoded values (not JSON strings)
      "events": list[dict[str, Any]],
      "status": {"code": str, "message": str | None},
    }

``attributes`` holds **decoded** values (the loader does any vendor-specific
JSON-string decoding). An adapter reads its dialect's attribute keys from
this dict and lifts the ones ragpill cares about to first-class
:class:`~ragpill.trace.Span` fields, passing everything else through on
``Span.attributes``.

Phase 1 ships only :class:`~ragpill.trace.adapters.MLflowAdapter`. The
``signature_attributes`` / registry / auto-detect machinery that consumes
it (``detect.py``, ``registry.py``, ``parse_otel``) lands in Phase 3; the
adapter interface is finalised here so adapters written now don't get
rewritten then.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ragpill.trace.model import Span as RagpillSpan


class AdapterDeclined(Exception):
    """Raised by :meth:`SpanAdapter.from_otel` when the signature matched but
    the span is malformed for this dialect.

    The loader catches this and falls back to the universal extractor (Phase 3)
    rather than producing a wrong span. Raising it is always preferable to
    returning a half-populated :class:`~ragpill.trace.Span`.
    """


def require_span_id(span: dict[str, Any], dialect: str) -> str:
    """Return the span's id, raising :class:`AdapterDeclined` when missing."""
    span_id = span.get("span_id")
    if not span_id:
        raise AdapterDeclined(f"{dialect} span dict is missing 'span_id'")
    return str(span_id)


def common_span_fields(span: dict[str, Any], *, dialect: str) -> dict[str, Any]:
    """The ``Span`` constructor kwargs every dialect lifts identically.

    Covers ids, name, timing, status, raw events, and the dialect tag —
    adapters add only their dialect-specific fields (kind, I/O, messages,
    documents, usage, attributes) on top.
    """
    status: dict[str, Any] = span.get("status") or {}
    return {
        "span_id": str(span.get("span_id", "")),
        "parent_id": span.get("parent_span_id"),
        "trace_id": str(span.get("trace_id", "")),
        "name": str(span.get("name", "")),
        "start_time_ns": int(span.get("start_time_unix_nano") or 0),
        "end_time_ns": int(span.get("end_time_unix_nano") or 0),
        "status": str(status.get("code", "UNSET")),
        "status_message": status.get("message"),
        "events": list(span.get("events") or []),
        "dialect": dialect,
    }


class SpanAdapter(ABC):
    """Convert a normalised OTLP-JSON span dict into a ``ragpill.trace.Span``."""

    name: str  # "mlflow" | "openinference" | "openllmetry" | …

    @classmethod
    @abstractmethod
    def signature_attributes(cls) -> tuple[str, ...]:
        """Attribute keys whose presence signals this dialect.

        Used by ``detect_dialect`` (Phase 3) to pick an adapter without parsing
        the whole span. A span matches this adapter when *all* of these keys are
        present in its ``attributes``. Order is irrelevant; ties between
        adapters are broken by registry precedence.
        """

    @classmethod
    @abstractmethod
    def from_otel(cls, span: dict[str, Any]) -> RagpillSpan:
        """Parse one normalised OTLP-JSON span dict into a ``ragpill.trace.Span``.

        Raise :class:`AdapterDeclined` when the signature matched but the span
        is malformed for this dialect, so the loader can fall back rather than
        emit a wrong span.
        """
