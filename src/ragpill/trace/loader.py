"""Loader: convert vendor trace objects into ``ragpill.trace.Trace``.

The loader owns all *input-format* normalisation. It turns a vendor trace
(here, ``mlflow.entities.Trace``) into the normalised OTLP-JSON span dicts
that :class:`~ragpill.trace.adapters.SpanAdapter` implementations consume,
dispatches each span through the right adapter, and assembles the
:class:`~ragpill.trace.Trace`.

Two entry points:

- :func:`from_mlflow_trace` — the dedicated MLflow-capture path (what
  ``execute_dataset`` uses; MLflow is the backend ragpill captures with).
- :func:`parse_otel` — the generic, dialect-agnostic path: takes OTLP-JSON
  (or a list of already-normalised span dicts), auto-detects each span's
  dialect via the registry, and assembles a :class:`~ragpill.trace.Trace`.
  Used to ingest traces produced elsewhere (Phoenix / Langfuse exports, raw
  OTLP-JSON). Phase 3 supports the JSON forms only (see
  ``designs/otel-trace-ingestion.md`` §11 Q3); proto / file / vendor-object
  inputs are Phase 4+.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from ragpill.trace.adapters._base import AdapterDeclined
from ragpill.trace.adapters.mlflow_adapter import MLflowAdapter
from ragpill.trace.fallback import universal_span
from ragpill.trace.model import Span, Trace
from ragpill.trace.registry import adapter_by_name, select_adapter

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


# ---------------------------------------------------------------------------
# Generic OTLP-JSON path
# ---------------------------------------------------------------------------


def _decode_otlp_value(value: dict[str, Any]) -> Any:
    """Decode one OTLP-JSON ``AnyValue`` (``{"stringValue": ...}`` etc.)."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_decode_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {kv["key"]: _decode_otlp_value(kv["value"]) for kv in value["kvlistValue"].get("values", [])}
    return value.get("stringValue")


def _normalize_otlp_span(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw OTLP-JSON span object into the adapter span-dict shape."""
    attributes = {kv["key"]: _decode_otlp_value(kv["value"]) for kv in raw.get("attributes", [])}
    events = [
        {
            "name": ev.get("name", ""),
            "time_unix_nano": ev.get("timeUnixNano"),
            "attributes": {kv["key"]: _decode_otlp_value(kv["value"]) for kv in ev.get("attributes", [])},
        }
        for ev in raw.get("events", [])
    ]
    raw_status: dict[str, Any] = raw.get("status") or {}
    # OTLP-JSON encodes nano timestamps as strings; a missing/empty value means
    # the exporter had no real timestamp — keep it None rather than fabricate 0.
    start = raw.get("startTimeUnixNano")
    end = raw.get("endTimeUnixNano")
    return {
        "trace_id": raw.get("traceId", ""),
        "span_id": raw.get("spanId", ""),
        "parent_span_id": raw.get("parentSpanId") or None,
        "name": raw.get("name", ""),
        "kind": raw.get("kind"),
        "start_time_unix_nano": int(start) if start else None,
        "end_time_unix_nano": int(end) if end else None,
        "attributes": attributes,
        "events": events,
        "status": {"code": raw_status.get("code", "UNSET"), "message": raw_status.get("message")},
    }


def _span_dicts_from_source(source: Any) -> list[dict[str, Any]]:
    """Coerce a ``parse_otel`` source into a list of normalised span dicts.

    Accepts a list of already-normalised span dicts, or an OTLP-JSON
    ``ResourceSpans`` payload (``{"resourceSpans": [{"scopeSpans": [...]}]}``).
    """
    if isinstance(source, list):
        return cast("list[dict[str, Any]]", source)
    if isinstance(source, dict) and "resourceSpans" in source:
        out: list[dict[str, Any]] = []
        src_dict = cast("dict[str, Any]", source)
        resource_spans = cast("list[dict[str, Any]]", src_dict.get("resourceSpans", []))
        for rs in resource_spans:
            for ss in cast("list[dict[str, Any]]", rs.get("scopeSpans", [])):
                for raw in cast("list[dict[str, Any]]", ss.get("spans", [])):
                    out.append(_normalize_otlp_span(raw))
        return out
    raise ValueError(
        "parse_otel: unsupported source. Pass a list of normalised span dicts or "
        "an OTLP-JSON ResourceSpans payload (a dict with a 'resourceSpans' key)."
    )


def _try_adapter(adapter: Any, span_dict: dict[str, Any]) -> Span | None:
    """Run ``adapter.from_otel`` if ``adapter`` is set; ``None`` on miss/decline."""
    if adapter is None:
        return None
    try:
        return adapter.from_otel(span_dict)
    except AdapterDeclined:
        return None


def parse_otel(
    source: Any,
    *,
    dialect: str | None = None,
    fallback_dialect: str | None = None,
) -> Trace:
    """Parse OTLP-bearing input into a vendor-neutral :class:`Trace`.

    Args:
        source: Either a list of already-normalised span dicts (the shape
            :class:`~ragpill.trace.adapters.SpanAdapter` consumes) or an
            OTLP-JSON ``ResourceSpans`` payload.
        dialect: ``"auto"`` auto-detects each span's dialect via the registry;
            an explicit adapter name (e.g. ``"openinference"``) forces every span
            through that adapter. ``None`` (default) reads
            :class:`~ragpill.settings.RagpillTraceSettings` (env
            ``RAGPILL_TRACE_DIALECT``), which itself defaults to ``"auto"``.
        fallback_dialect: adapter used when ``dialect="auto"`` matches nothing
            for a span. ``None`` reads ``RagpillTraceSettings`` (env
            ``RAGPILL_TRACE_FALLBACK_DIALECT``, default ``"gen_ai"``). If even the
            fallback adapter declines, the universal best-effort extractor is
            used and a single warning is emitted.

    Returns:
        A :class:`Trace`. ``trace.dialect`` is set to the most common per-span
        dialect; individual spans keep their own ``dialect``.
    """
    if dialect is None or fallback_dialect is None:
        from ragpill.settings import RagpillTraceSettings

        settings = RagpillTraceSettings()  # pyright: ignore[reportCallIssue]
        dialect = dialect if dialect is not None else settings.dialect
        fallback_dialect = fallback_dialect if fallback_dialect is not None else settings.fallback_dialect

    span_dicts = _span_dicts_from_source(source)
    forced = None if dialect == "auto" else adapter_by_name(dialect)
    if dialect != "auto" and forced is None:
        raise ValueError(f"parse_otel: unknown dialect {dialect!r}")
    fallback_adapter = adapter_by_name(fallback_dialect)

    spans: list[Span] = []
    fell_back = 0
    trace_id = ""
    for sd in span_dicts:
        trace_id = trace_id or str(sd.get("trace_id", ""))
        adapter = forced or select_adapter(sd)
        span = _try_adapter(adapter, sd)
        if span is not None:
            spans.append(span)
            continue
        # Auto-detection matched nothing (or the matched adapter declined). Try
        # the configured fallback adapter, then the universal best-effort
        # extractor. Either way it counts as "not recognised" for the warning.
        fell_back += 1
        spans.append(_try_adapter(fallback_adapter, sd) or universal_span(sd))

    if fell_back:
        warnings.warn(
            f"parse_otel: {fell_back} span(s) matched no dialect adapter and were "
            "extracted best-effort. See designs/otel-trace-ingestion.md.",
            stacklevel=2,
        )

    dialects = [s.dialect for s in spans if s.dialect != "unknown"]
    primary = max(set(dialects), key=dialects.count) if dialects else "unknown"
    return Trace(trace_id=trace_id, spans=spans, dialect=primary)
