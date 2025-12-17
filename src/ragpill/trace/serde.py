"""JSON-friendly (de)serialisation for the vendor-neutral trace model.

Kept out of ``model.py`` so the dataclasses stay pure data. ``trace_to_dict``
produces a plain ``dict`` of JSON-safe primitives (``SpanKind`` becomes its
string value); ``trace_from_dict`` is the inverse. Unknown/extra keys in a
payload are ignored so the schema can grow additively.

This is the on-disk representation used by ``EvaluationOutput`` run JSON
(schema_version 2). The pass-through ``attributes`` / ``events`` bags survive
the round trip, so a span carries everything the adapter didn't lift to a
first-class field.
"""

from __future__ import annotations

from typing import Any, cast

from ragpill.trace.model import Document, Message, Span, SpanKind, Trace, Usage


def _dicts(value: Any) -> list[dict[str, Any]]:
    """Coerce a payload list field to ``list[dict]`` for typed iteration."""
    return cast("list[dict[str, Any]]", value or [])


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": m.tool_calls,
        "tool_call_id": m.tool_call_id,
    }


def _message_from_dict(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content"),
        tool_calls=list(d.get("tool_calls") or []),
        tool_call_id=d.get("tool_call_id"),
    )


def _document_to_dict(doc: Document) -> dict[str, Any]:
    return {"content": doc.content, "id": doc.id, "score": doc.score, "metadata": doc.metadata}


def _document_from_dict(d: dict[str, Any]) -> Document:
    return Document(
        content=d["content"],
        id=d.get("id"),
        score=d.get("score"),
        metadata=dict(d.get("metadata") or {}),
    )


def _usage_to_dict(u: Usage) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "cost_usd": u.cost_usd,
    }


def _usage_from_dict(d: dict[str, Any] | None) -> Usage:
    d = d or {}
    return Usage(
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        total_tokens=d.get("total_tokens"),
        cost_usd=d.get("cost_usd"),
    )


def _span_to_dict(span: Span) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "trace_id": span.trace_id,
        "name": span.name,
        "kind": span.kind.value,
        "start_time_ns": span.start_time_ns,
        "end_time_ns": span.end_time_ns,
        "status": span.status,
        "status_message": span.status_message,
        "inputs": span.inputs,
        "outputs": span.outputs,
        "messages_in": [_message_to_dict(m) for m in span.messages_in],
        "messages_out": [_message_to_dict(m) for m in span.messages_out],
        "documents": [_document_to_dict(d) for d in span.documents],
        "model": span.model,
        "model_parameters": span.model_parameters,
        "usage": _usage_to_dict(span.usage),
        "attributes": span.attributes,
        "events": span.events,
        "dialect": span.dialect,
    }


def _span_kind_from_value(value: Any) -> SpanKind:
    """Coerce a stored kind string to :class:`SpanKind`, unknown -> ``UNKNOWN``.

    Honors the module's additive-schema promise: a run JSON written by a newer
    ragpill that added a span kind stays readable here instead of raising.
    """
    try:
        return SpanKind(value)
    except ValueError:
        return SpanKind.UNKNOWN


def _span_from_dict(d: dict[str, Any]) -> Span:
    return Span(
        span_id=d["span_id"],
        parent_id=d.get("parent_id"),
        trace_id=d.get("trace_id", ""),
        name=d.get("name", ""),
        kind=_span_kind_from_value(d.get("kind", SpanKind.UNKNOWN.value)),
        start_time_ns=d.get("start_time_ns", 0),
        end_time_ns=d.get("end_time_ns", 0),
        status=d.get("status", "UNSET"),
        status_message=d.get("status_message"),
        inputs=d.get("inputs"),
        outputs=d.get("outputs"),
        messages_in=[_message_from_dict(m) for m in _dicts(d.get("messages_in"))],
        messages_out=[_message_from_dict(m) for m in _dicts(d.get("messages_out"))],
        documents=[_document_from_dict(x) for x in _dicts(d.get("documents"))],
        model=d.get("model"),
        model_parameters=dict(d.get("model_parameters") or {}),
        usage=_usage_from_dict(d.get("usage")),
        attributes=dict(d.get("attributes") or {}),
        events=list(d.get("events") or []),
        dialect=d.get("dialect", "unknown"),
    )


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    """Serialise a :class:`~ragpill.trace.Trace` to a JSON-safe ``dict``."""
    return {
        "trace_id": trace.trace_id,
        "spans": [_span_to_dict(s) for s in trace.spans],
        "session_id": trace.session_id,
        "user_id": trace.user_id,
        "tags": trace.tags,
        "metadata": trace.metadata,
        "dialect": trace.dialect,
    }


def trace_from_dict(d: dict[str, Any]) -> Trace:
    """Reconstruct a :class:`~ragpill.trace.Trace` from :func:`trace_to_dict` output."""
    return Trace(
        trace_id=d.get("trace_id", ""),
        spans=[_span_from_dict(s) for s in _dicts(d.get("spans"))],
        session_id=d.get("session_id"),
        user_id=d.get("user_id"),
        tags=list(d.get("tags") or []),
        metadata=dict(d.get("metadata") or {}),
        dialect=d.get("dialect", "unknown"),
    )
