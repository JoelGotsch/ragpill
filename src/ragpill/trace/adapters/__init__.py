"""Dialect adapters — convert normalised OTLP-JSON span dicts into ``ragpill.trace.Span``.

Phase 1 ships only the MLflow adapter. Phase 3 of the OTel ingestion design
adds ``gen_ai`` + ``openinference`` and Phase 4 adds ``openllmetry`` +
``langfuse`` + ``logfire``, registered via entry points. See
``designs/otel-trace-ingestion.md`` §6 for the registry + entry-point model.
"""

from __future__ import annotations

from ragpill.trace.adapters._base import AdapterDeclined, SpanAdapter
from ragpill.trace.adapters.mlflow_adapter import MLflowAdapter

__all__ = ["AdapterDeclined", "MLflowAdapter", "SpanAdapter"]
