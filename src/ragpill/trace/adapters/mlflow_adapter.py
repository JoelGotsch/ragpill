"""MLflow dialect adapter.

Reads MLflow's ``mlflow.span*`` attribute keys off a normalised OTLP-JSON
span dict (produced by :func:`ragpill.trace.loader.from_mlflow_trace`) and
lifts them onto a vendor-neutral :class:`ragpill.trace.Span`.

This adapter imports **no** ``mlflow`` symbols — it only knows MLflow's
attribute-key conventions. All ``mlflow.entities`` handling lives in the
loader, so this stays importable without the ``mlflow`` extra and mirrors
how every other dialect adapter will be written.
"""

from __future__ import annotations

from typing import Any

from ragpill.trace.adapters._base import SpanAdapter, common_span_fields, require_span_id
from ragpill.trace.model import Span, SpanKind

_SPAN_TYPE_KEY = "mlflow.spanType"
_INPUTS_KEY = "mlflow.spanInputs"
_OUTPUTS_KEY = "mlflow.spanOutputs"

# Keys MLflow uses for its own bookkeeping. They are lifted to first-class
# Span fields (kind/inputs/outputs) and so are dropped from the pass-through
# ``attributes`` bag to avoid duplicating them.
_INTERNAL_ATTRS: frozenset[str] = frozenset(
    {
        "mlflow.traceRequestId",
        _SPAN_TYPE_KEY,
        _INPUTS_KEY,
        _OUTPUTS_KEY,
        "mlflow.spanFunctionName",
    }
)

# MLflow SpanType string -> our ingest-side SpanKind. MLflow kinds without a
# neutral equivalent (EVALUATOR, MEMORY, WORKFLOW, …) fall back to UNKNOWN
# rather than raising.
_KIND_BY_SPAN_TYPE: dict[str, SpanKind] = {
    "LLM": SpanKind.LLM,
    "CHAT_MODEL": SpanKind.CHAT_MODEL,
    "CHAIN": SpanKind.CHAIN,
    "AGENT": SpanKind.AGENT,
    "TOOL": SpanKind.TOOL,
    "RETRIEVER": SpanKind.RETRIEVER,
    "RERANKER": SpanKind.RERANKER,
    "EMBEDDING": SpanKind.EMBEDDING,
    "PARSER": SpanKind.PARSER,
    "TASK": SpanKind.TASK,
    "GUARDRAIL": SpanKind.GUARDRAIL,
    "UNKNOWN": SpanKind.UNKNOWN,
}


class MLflowAdapter(SpanAdapter):
    """Map MLflow-instrumented spans onto ``ragpill.trace.Span``."""

    name = "mlflow"

    @classmethod
    def signature_attributes(cls) -> tuple[str, ...]:
        # ``mlflow.spanType`` is set on every MLflow span and on no other
        # dialect's spans, so its presence uniquely identifies MLflow output.
        return (_SPAN_TYPE_KEY,)

    @classmethod
    def from_otel(cls, span: dict[str, Any]) -> Span:
        require_span_id(span, cls.name)
        attributes: dict[str, Any] = dict(span.get("attributes") or {})

        span_type = attributes.get(_SPAN_TYPE_KEY)
        kind = _KIND_BY_SPAN_TYPE.get(str(span_type), SpanKind.UNKNOWN)
        passthrough = {k: v for k, v in attributes.items() if k not in _INTERNAL_ATTRS}

        return Span(
            **common_span_fields(span, dialect=cls.name),
            kind=kind,
            inputs=attributes.get(_INPUTS_KEY),
            outputs=attributes.get(_OUTPUTS_KEY),
            attributes=passthrough,
        )
