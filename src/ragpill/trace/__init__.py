"""Vendor-neutral trace model + dialect adapters.

Phase 1 of ``designs/otel-trace-ingestion.md``: ships the data classes
that the renderer, evaluators, and JSON layer will eventually consume,
plus the MLflow input loader (``from_mlflow_trace``: ``mlflow.entities.Trace``
→ ``ragpill.trace.Trace``) built on the Option-C ``SpanAdapter`` interface.

The renderer, evaluators, and run-JSON layer consume this model directly
(the flip landed in 0.5.0). The remaining dialect adapters (gen_ai /
openinference / langfuse / logfire) land in later phases of
``designs/otel-trace-ingestion.md``.
"""

from __future__ import annotations

from ragpill.trace.detect import detect_dialect
from ragpill.trace.loader import from_mlflow_trace, parse_otel
from ragpill.trace.model import Document, Message, Span, SpanKind, Trace, Usage
from ragpill.trace.ops import filter_to_subtree
from ragpill.trace.serde import trace_from_dict, trace_to_dict

__all__ = [
    "Document",
    "Message",
    "Span",
    "SpanKind",
    "Trace",
    "Usage",
    "detect_dialect",
    "filter_to_subtree",
    "from_mlflow_trace",
    "parse_otel",
    "trace_from_dict",
    "trace_to_dict",
]
