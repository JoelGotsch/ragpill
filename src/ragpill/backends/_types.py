"""Vendor-neutral data types shared by every tracking backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class CaptureSpanKind(StrEnum):
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
    TASK = "TASK"
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
class CaseGroupingHandle:
    """Returned by :meth:`TraceCaptureBackend.start_case_grouping`.

    Tells the execution layer how the backend chose to relate the per-repeat
    traces produced inside the case's loop, so it can assemble
    :class:`~ragpill.execution.CaseRunOutput` correctly.

    Two modes:

    - ``"session"`` — the backend tagged each per-repeat top-level trace with
      a native session id (MLflow ``mlflow.trace.session`` metadata,
      Langfuse ``session_id``, OpenInference ``session.id``). The Sessions
      UI then shows one session per case with each repeat as a turn.
      ``case_trace_id`` is ``None``; each ``TaskRunOutput`` carries its own
      trace_id and the execution layer fetches them individually.
    - ``"span"`` — the backend opened a parent span instead. Repeats nest as
      child spans of that parent (the pre-0.5 behaviour). Used as the
      fallback for any future adapter that lacks a sessions concept.
      ``case_trace_id`` is the parent's trace_id so the execution layer can
      filter the case-level trace to per-repeat subtrees as before.
    """

    mode: Literal["session", "span"]
    case_trace_id: str | None = None
    session_id: str | None = None


@dataclass
class Assessment:
    """A single evaluator verdict, ready for backend persistence.

    Mirrors the fields MLflow's ``mlflow.entities.Feedback`` exposes plus a
    free-form ``metadata`` bag for backend-specific extras. Adapters convert
    to/from their native types at the boundary.
    """

    name: str
    value: bool | int | float | str
    source_type: Literal["CODE", "LLM_JUDGE"]
    """Declared by the evaluator class via ``BaseEvaluator.source_type``."""

    source_id: str
    rationale: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
