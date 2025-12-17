"""OpenInference (Arize / Phoenix) adapter.

OpenInference is the broadest LLM-span dialect and the one ragpill's evaluators
map most cleanly to — in particular ``retrieval.documents.*.document.content``
feeds the source-based evaluators. Always-on (pure attribute reading).

Quirk handled here: OpenInference encodes lists as **indexed flat keys**
(``llm.input_messages.0.message.role``,
``retrieval.documents.0.document.content``), so this adapter reconstructs lists
by parsing key suffixes. See ``designs/otel-trace-ingestion.md`` §3.2.

**Deviation from design §6.3:** ships always-on rather than behind a
``ragpill[openinference]`` extra. The adapter only reads string-keyed
attributes, so the ``openinference-semantic-conventions`` package would supply
named constants but no behaviour; gating a pure-attribute reader behind an extra
is complexity for no runtime benefit.
"""

from __future__ import annotations

import re
from typing import Any, cast

from ragpill.trace.adapters._base import AdapterDeclined, SpanAdapter
from ragpill.trace.model import Document, Message, Span, SpanKind, Usage

_KIND_KEY = "openinference.span.kind"

_KIND_MAP: dict[str, SpanKind] = {
    "LLM": SpanKind.LLM,
    "RETRIEVER": SpanKind.RETRIEVER,
    "EMBEDDING": SpanKind.EMBEDDING,
    "RERANKER": SpanKind.RERANKER,
    "AGENT": SpanKind.AGENT,
    "TOOL": SpanKind.TOOL,
    "GUARDRAIL": SpanKind.GUARDRAIL,
    "CHAIN": SpanKind.CHAIN,
    # OpenInference EVALUATOR has no ragpill equivalent; weakly typed.
    "EVALUATOR": SpanKind.UNKNOWN,
    "UNKNOWN": SpanKind.UNKNOWN,
}


def _collect_indexed(attributes: dict[str, Any], prefix: str, item_key: str) -> list[dict[str, Any]]:
    """Reconstruct a list from indexed flat keys.

    e.g. for ``prefix="llm.input_messages"`` and ``item_key="message"``, gathers
    ``{prefix}.{i}.{item_key}.{field}`` into ``result[i][field]``. Indices may be
    sparse; the result is ordered by index.
    """
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.{re.escape(item_key)}\.(.+)$")
    by_index: dict[int, dict[str, Any]] = {}
    for key, value in attributes.items():
        m = pat.match(key)
        if m:
            idx = int(m.group(1))
            by_index.setdefault(idx, {})[m.group(2)] = value
    return [by_index[i] for i in sorted(by_index)]


class OpenInferenceAdapter(SpanAdapter):
    """Map OpenInference spans onto ``ragpill.trace.Span``."""

    name = "openinference"

    @classmethod
    def signature_attributes(cls) -> tuple[str, ...]:
        return (_KIND_KEY,)

    @classmethod
    def from_otel(cls, span: dict[str, Any]) -> Span:
        attributes: dict[str, Any] = dict(span.get("attributes") or {})
        span_id = span.get("span_id")
        if not span_id:
            raise AdapterDeclined("openinference span dict is missing 'span_id'")

        kind = _KIND_MAP.get(str(attributes.get(_KIND_KEY)), SpanKind.UNKNOWN)

        messages_in = [
            Message(role=str(m.get("role", "")), content=m.get("content"))
            for m in _collect_indexed(attributes, "llm.input_messages", "message")
        ]
        messages_out = [
            Message(role=str(m.get("role", "")), content=m.get("content"))
            for m in _collect_indexed(attributes, "llm.output_messages", "message")
        ]
        documents: list[Document] = []
        for d in _collect_indexed(attributes, "retrieval.documents", "document"):
            md = d.get("metadata")
            documents.append(
                Document(
                    content=str(d.get("content", "")),
                    id=d.get("id"),
                    score=d.get("score"),
                    metadata=cast("dict[str, Any]", md) if isinstance(md, dict) else {},
                )
            )

        usage = Usage(
            input_tokens=attributes.get("llm.token_count.prompt"),
            output_tokens=attributes.get("llm.token_count.completion"),
            total_tokens=attributes.get("llm.token_count.total"),
        )
        status: dict[str, Any] = span.get("status") or {}

        return Span(
            span_id=str(span_id),
            parent_id=span.get("parent_span_id"),
            trace_id=str(span.get("trace_id", "")),
            name=str(span.get("name", "")),
            kind=kind,
            start_time_ns=int(span.get("start_time_unix_nano") or 0),
            end_time_ns=int(span.get("end_time_unix_nano") or 0),
            status=str(status.get("code", "UNSET")),
            status_message=status.get("message"),
            inputs=attributes.get("input.value"),
            outputs=attributes.get("output.value"),
            messages_in=messages_in,
            messages_out=messages_out,
            documents=documents,
            model=attributes.get("llm.model_name"),
            usage=usage,
            attributes=attributes,
            events=list(span.get("events") or []),
            dialect=cls.name,
        )
