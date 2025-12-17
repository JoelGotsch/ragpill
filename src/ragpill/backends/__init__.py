"""Pluggable tracking backends.

Phase 1 of ``plans/multi-backend-tracking.md``. This package defines the
small protocols ragpill needs from any tracking backend (MLflow today,
Langfuse and Arize Phoenix in Phase 2) and ships an in-tree
:class:`MLflowBackend` adapter that forwards directly to ``mlflow.*``.

Step 1 — what this commit lands:

- the four capability protocols (``TraceCaptureBackend``,
  ``TraceQueryBackend``, ``ResultsBackend``, ``LifecycleBackend``) plus the
  combined ``Backend`` protocol;
- vendor-neutral data types (``Assessment``, ``RunHandle``, ``SpanKind``);
- :class:`MLflowBackend` that satisfies all four protocols;
- a registry (``get_backend`` / ``configure_backend``) that returns the
  MLflow backend by default.

No existing call sites change in this commit; the suite still passes
unchanged. Subsequent commits switch ``upload.py``, ``execution.py`` and
``evaluators.py`` over to use the registry.
"""

from __future__ import annotations

from ragpill.backends._base import (
    Backend,
    LifecycleBackend,
    ResultsBackend,
    TraceCaptureBackend,
    TraceQueryBackend,
)
from ragpill.backends._registry import configure_backend, get_backend, reset_backend
from ragpill.backends._types import Assessment, RunHandle, SpanKind

__all__ = [
    "Assessment",
    "Backend",
    "LifecycleBackend",
    "ResultsBackend",
    "RunHandle",
    "SpanKind",
    "TraceCaptureBackend",
    "TraceQueryBackend",
    "configure_backend",
    "get_backend",
    "reset_backend",
]
