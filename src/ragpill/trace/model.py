"""Internal trace dataclasses (vendor-neutral).

Shape matches ``designs/otel-trace-ingestion.md`` §4. Two notable choices:

1. **Optional fields, not abstract methods.** A retriever span that
   produced no documents simply has ``documents=[]``. Evaluators check
   the field, not the kind.
2. **``attributes`` carries pass-through.** Anything the adapter didn't
   recognise (e.g. an OpenInference ``metadata.user_query_intent`` set
   by the user) lands there so users can write custom evaluators against
   raw vendor attributes when they need to.

The ``SpanKind`` enum is intentionally richer than
:class:`ragpill.backends.SpanKind` (which is write-side only — what
ragpill itself opens via ``backend.start_span``). The trace model's
SpanKind covers everything we might *ingest* from any of the LLM-OTel
dialects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    """Subset of OpenInference / OpenTelemetry GenAI span kinds we ingest.

    Adapter implementations map their dialect-specific kind attribute
    (``mlflow.spanType``, ``openinference.span.kind``,
    ``traceloop.span.kind``, ``langfuse.observation.type``, …) onto these
    values. Unknown kinds map to ``UNKNOWN`` rather than raising.
    """

    AGENT = "AGENT"
    CHAIN = "CHAIN"
    CHAT_MODEL = "CHAT_MODEL"
    EMBEDDING = "EMBEDDING"
    GUARDRAIL = "GUARDRAIL"
    LLM = "LLM"
    PARSER = "PARSER"
    RERANKER = "RERANKER"
    RETRIEVER = "RETRIEVER"
    TASK = "TASK"
    TOOL = "TOOL"
    UNKNOWN = "UNKNOWN"


@dataclass
class Message:
    """One message in an LLM chat exchange.

    ``content`` is typed ``Any`` because providers emit it as either a
    plain string or a list of content parts (text + images + tool calls).
    Adapter implementations decode the vendor-specific shape into one of
    those two forms.
    """

    role: str
    content: Any
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Document:
    """A retrieved document — output of a retriever / reranker span."""

    content: str
    id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token / cost counters for an LLM span. All fields optional."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class Span:
    """A single span in a captured trace, dialect-normalised.

    First-class fields cover the data ragpill's renderer and evaluators
    actually read. Vendor-specific extras land in ``attributes`` and
    survive JSON round-trip so users can write custom evaluators against
    raw attributes when needed.
    """

    span_id: str
    parent_id: str | None
    trace_id: str
    name: str
    kind: SpanKind
    start_time_ns: int
    end_time_ns: int
    status: str = "UNSET"
    """``"OK" | "ERROR" | "UNSET"``."""

    status_message: str | None = None
    inputs: Any | None = None
    outputs: Any | None = None
    messages_in: list[Message] = field(default_factory=list)
    messages_out: list[Message] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    model: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    attributes: dict[str, Any] = field(default_factory=dict)
    """Pass-through bag for anything the adapter didn't lift to a
    first-class field. Survives JSON round-trip."""

    events: list[dict[str, Any]] = field(default_factory=list)
    """Raw OTel events (used by the ``gen_ai`` dialect where messages
    are encoded as events). Pass-through unless the adapter promoted
    them to ``messages_in`` / ``messages_out``."""

    dialect: str = "unknown"
    """The dialect adapter that produced this span (``"mlflow"`` |
    ``"openinference"`` | ``"openllmetry"`` | ``"gen_ai"`` |
    ``"langfuse"`` | ``"logfire"`` | ``"unknown"``)."""


@dataclass
class Trace:
    """A captured trace — a collection of spans plus trace-level metadata."""

    trace_id: str
    spans: list[Span]
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    dialect: str = "unknown"
    """Primary dialect detected on this trace. Spans may differ if the
    trace mixes dialects; in that case the spans carry their own
    ``dialect`` while the trace tag reflects the most common one."""
