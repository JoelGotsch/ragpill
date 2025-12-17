"""OpenTelemetry GenAI semantic-conventions adapter (``gen_ai.*``).

Reads the OTel GenAI convention emitted natively by pydantic-ai, the OpenAI /
Anthropic OTel instrumentations, Logfire, etc. Always-on (pure attribute/event
reading, no extra dependency).

Quirk handled here: GenAI encodes chat **messages as span events**
(``gen_ai.user.message``, ``gen_ai.assistant.message``, ``gen_ai.choice``, …),
not as attributes — so this adapter reads ``span["events"]``, which an
attribute-only parser would miss. See ``designs/otel-trace-ingestion.md`` §3.1.
"""

from __future__ import annotations

from typing import Any, cast

from ragpill.trace.adapters._base import AdapterDeclined, SpanAdapter
from ragpill.trace.model import Message, Span, SpanKind, Usage

_SYSTEM_KEY = "gen_ai.system"

# event name -> chat role; choices are outputs, the rest are inputs.
_INPUT_EVENT_ROLES = {
    "gen_ai.user.message": "user",
    "gen_ai.system.message": "system",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
_OUTPUT_EVENT_NAMES = {"gen_ai.choice", "gen_ai.assistant.message"}


def _event_content(ev: dict[str, Any]) -> Any:
    """Pull the message content out of a GenAI event's attributes/body."""
    attrs: dict[str, Any] = ev.get("attributes") or {}
    for key in ("content", "gen_ai.event.content", "body", "message"):
        if key in attrs:
            return attrs[key]
    return attrs or None


class GenAIAdapter(SpanAdapter):
    """Map OTel GenAI-convention spans onto ``ragpill.trace.Span``."""

    name = "gen_ai"

    @classmethod
    def signature_attributes(cls) -> tuple[str, ...]:
        return (_SYSTEM_KEY,)

    @classmethod
    def from_otel(cls, span: dict[str, Any]) -> Span:
        attributes: dict[str, Any] = dict(span.get("attributes") or {})
        span_id = span.get("span_id")
        if not span_id:
            raise AdapterDeclined("gen_ai span dict is missing 'span_id'")

        messages_in: list[Message] = []
        messages_out: list[Message] = []
        events = cast("list[dict[str, Any]]", span.get("events") or [])
        for ev in events:
            ev_name = ev.get("name", "")
            content = _event_content(ev)
            if ev_name in _OUTPUT_EVENT_NAMES:
                messages_out.append(Message(role="assistant", content=content))
            elif ev_name in _INPUT_EVENT_ROLES:
                messages_in.append(Message(role=_INPUT_EVENT_ROLES[ev_name], content=content))

        usage = Usage(
            input_tokens=attributes.get("gen_ai.usage.input_tokens"),
            output_tokens=attributes.get("gen_ai.usage.output_tokens"),
        )
        model_parameters = {
            k: attributes[f"gen_ai.request.{k}"]
            for k in ("temperature", "top_p", "max_tokens", "stop_sequences")
            if f"gen_ai.request.{k}" in attributes
        }
        status: dict[str, Any] = span.get("status") or {}

        return Span(
            span_id=str(span_id),
            parent_id=span.get("parent_span_id"),
            trace_id=str(span.get("trace_id", "")),
            name=str(span.get("name", "")),
            kind=SpanKind.LLM,
            start_time_ns=int(span.get("start_time_unix_nano") or 0),
            end_time_ns=int(span.get("end_time_unix_nano") or 0),
            status=str(status.get("code", "UNSET")),
            status_message=status.get("message"),
            messages_in=messages_in,
            messages_out=messages_out,
            model=attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model"),
            model_parameters=model_parameters,
            usage=usage,
            attributes=attributes,
            events=list(span.get("events") or []),
            dialect=cls.name,
        )
