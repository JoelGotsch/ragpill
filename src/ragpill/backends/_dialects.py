"""Per-backend trace-dialect details.

The trace-rendering layer reads span attributes by key. Each backend's
auto-instrumentation produces a different set of "internal" keys (mlflow's
own bookkeeping, OpenInference's `llm.*` keys, Langfuse's, …) which the
renderer should *not* dump into the attribute bullet list because they're
surfaced via dedicated fields (`inputs`, `outputs`, `span_type`).

Phase 1 ships only the MLflow dialect because that's the only backend in
tree. Phase 2 (alongside `designs/otel-trace-ingestion.md`) lets each
backend register its own set so the renderer becomes dialect-neutral.
"""

from __future__ import annotations

MLFLOW_INTERNAL_SPAN_ATTRS: frozenset[str] = frozenset(
    {
        "mlflow.traceRequestId",
        "mlflow.spanType",
        "mlflow.spanInputs",
        "mlflow.spanOutputs",
        "mlflow.spanFunctionName",
    }
)
"""Span attribute keys MLflow uses internally and ragpill renders via
dedicated paths (``span.inputs``, ``span.outputs``, ``span.span_type``).
Excluded from the generic-attribute bullet so they're not duplicated."""
