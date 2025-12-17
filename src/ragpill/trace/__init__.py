"""Vendor-neutral trace model + dialect adapters.

Phase 1 of ``designs/otel-trace-ingestion.md``: ships the data classes
that the renderer, evaluators, and JSON layer will eventually consume,
plus the MLflow input loader (``from_mlflow_trace``: ``mlflow.entities.Trace``
→ ``ragpill.trace.Trace``) built on the Option-C ``SpanAdapter`` interface.

Phase 1 is purely additive — the renderer, evaluators, and JSON
serialization in ``ragpill.execution`` and ``ragpill.report`` still
operate on ``mlflow.entities.Trace`` via the Phase-1 type-alias
shortcut documented in ``plans/multi-backend-tracking.md``. The flip to
this model (and the ``compat.to_mlflow_trace`` back-compat shim for users
with custom ``SpanBaseEvaluator`` subclasses) happens in Phase 2; the
remaining dialect adapters (gen_ai / openinference / langfuse / logfire)
land in Phases 3-4.
"""

from __future__ import annotations

from ragpill.trace.loader import from_mlflow_trace
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
    "filter_to_subtree",
    "from_mlflow_trace",
    "trace_from_dict",
    "trace_to_dict",
]
