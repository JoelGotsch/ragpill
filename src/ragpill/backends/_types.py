"""Vendor-neutral data types shared by every tracking backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SpanKind(StrEnum):
    """Subset of OpenTelemetry / OpenInference span kinds ragpill actually uses.

    Each adapter maps these to its backend-native enum (e.g. MLflow's
    ``SpanType``, OpenInference's span-kind attribute). Unknown kinds map to
    ``UNKNOWN`` rather than raising — render-only consumers can degrade
    gracefully.
    """

    AGENT = "AGENT"
    CHAT_MODEL = "CHAT_MODEL"
    LLM = "LLM"
    RERANKER = "RERANKER"
    RETRIEVER = "RETRIEVER"
    TOOL = "TOOL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RunHandle:
    """Identifiers returned by :meth:`TraceCaptureBackend.start_run`.

    Just enough to reattach the run later or look it up server-side. Adapter
    implementations may carry richer state internally; only ``run_id`` and
    ``experiment_id`` are part of the public surface.
    """

    run_id: str
    experiment_id: str


@dataclass
class Assessment:
    """A single evaluator verdict, ready for backend persistence.

    Mirrors the fields MLflow's ``mlflow.entities.Feedback`` exposes plus a
    free-form ``metadata`` bag for backend-specific extras. Adapters convert
    to/from their native types at the boundary.
    """

    name: str
    value: bool | int | float | str
    source_type: str
    """``"LLM_JUDGE"`` or ``"CODE"`` — matches the existing convention in upload.py."""

    source_id: str
    rationale: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
