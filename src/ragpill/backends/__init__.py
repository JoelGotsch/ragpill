"""Pluggable tracking backends.

This package defines the small protocols ragpill needs from any tracking
backend (see ``plans/multi-backend-tracking.md``) and the in-tree adapters:
MLflow (default), Langfuse, and Arize Phoenix.

Contents:

- the four capability protocols (``TraceCaptureBackend``,
  ``TraceQueryBackend``, ``ResultsBackend``, ``LifecycleBackend``) plus the
  combined ``Backend`` protocol and the ``SpanHandle`` span contract;
- vendor-neutral data types (``Assessment``, ``RunHandle``, ``SpanKind``);
- a registry (``get_backend`` / ``configure_backend``) that returns the
  MLflow backend by default;
- shared adapter scaffolding in ``_common`` (polling, no-op mixins).
"""

from __future__ import annotations

from ragpill.backends._base import (
    Backend,
    LifecycleBackend,
    ResultsBackend,
    SpanHandle,
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
    "SpanHandle",
    "SpanKind",
    "TraceCaptureBackend",
    "TraceQueryBackend",
    "configure_backend",
    "get_backend",
    "reset_backend",
]
