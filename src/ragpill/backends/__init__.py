"""Pluggable tracking backends.

This package defines the small protocols ragpill needs from any tracking
backend (see ``plans/multi-backend-tracking.md``) and the in-tree adapters:
MLflow (default), Langfuse, and Arize Phoenix.

Contents:

- the four capability protocols (``TraceCaptureBackend``,
  ``TraceQueryBackend``, ``ResultsBackend``, ``LifecycleBackend``) plus the
  combined ``Backend`` protocol and the ``SpanHandle`` span contract;
- vendor-neutral data types (``Assessment``, ``RunHandle``, ``CaptureSpanKind``);
- a registry (``get_backend`` / ``configure_backend``) that returns the
  MLflow backend by default;
- shared adapter scaffolding in ``_common`` (polling, no-op mixins).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragpill.backends._base import (
    Backend,
    LifecycleBackend,
    ResultsBackend,
    SpanHandle,
    TraceCaptureBackend,
    TraceQueryBackend,
)
from ragpill.backends._registry import configure_backend, get_backend, reset_backend
from ragpill.backends._types import Assessment, CaptureSpanKind, RunHandle

if TYPE_CHECKING:
    from ragpill.backends.langfuse_backend import LangfuseBackend
    from ragpill.backends.mlflow_backend import MLflowBackend
    from ragpill.backends.phoenix_backend import PhoenixBackend

# Adapter classes are imported lazily so ``import ragpill.backends`` stays cheap
# and free of the optional extras: pulling in MLflowBackend must not require the
# phoenix/langfuse SDKs, and vice versa.
_LAZY_BACKENDS = {
    "MLflowBackend": "ragpill.backends.mlflow_backend",
    "LangfuseBackend": "ragpill.backends.langfuse_backend",
    "PhoenixBackend": "ragpill.backends.phoenix_backend",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_BACKENDS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


__all__ = [
    "Assessment",
    "Backend",
    "CaptureSpanKind",
    "LangfuseBackend",
    "LifecycleBackend",
    "MLflowBackend",
    "PhoenixBackend",
    "ResultsBackend",
    "RunHandle",
    "SpanHandle",
    "TraceCaptureBackend",
    "TraceQueryBackend",
    "configure_backend",
    "get_backend",
    "reset_backend",
]
