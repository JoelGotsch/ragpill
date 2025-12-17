"""Courtesy back-compat shim: ``ragpill.trace.Trace`` -> MLflow-shaped duck type.

For external code with custom ``SpanBaseEvaluator`` subclasses that still read
the ``mlflow.entities.Trace`` / ``Span`` surface directly. :func:`to_mlflow_trace`
returns a lightweight duck-typed object exposing the subset that ragpill and
common custom evaluators use:

    trace.data.spans
    trace.search_spans(span_type=...)
    span.span_id / parent_id / name / span_type / inputs / outputs / attributes
    span.start_time_ns / end_time_ns

It is **not** a real ``mlflow.entities.Trace`` and is unused internally — the
codebase consumes the neutral model. This exists only so a one-line
``to_mlflow_trace(ctx.trace)`` keeps un-migrated custom evaluators working.
Backward compatibility is not a project target; prefer migrating to
``ragpill.trace.Span``.
"""

from __future__ import annotations

from typing import Any

from ragpill.trace.model import Span, Trace

# MLflow exposed span type/inputs/outputs both as first-class accessors and as
# these attribute keys. We lifted them to Span fields and dropped the keys; the
# shim re-adds them so `span.attributes["mlflow.spanType"]` still resolves.
_SPAN_TYPE_KEY = "mlflow.spanType"
_INPUTS_KEY = "mlflow.spanInputs"
_OUTPUTS_KEY = "mlflow.spanOutputs"


class _CompatSpan:
    """Duck-typed stand-in for ``mlflow.entities.Span``."""

    def __init__(self, span: Span) -> None:
        self._span = span

    @property
    def span_id(self) -> str:
        return self._span.span_id

    @property
    def parent_id(self) -> str | None:
        return self._span.parent_id

    @property
    def name(self) -> str:
        return self._span.name

    @property
    def span_type(self) -> str:
        return self._span.kind.value

    @property
    def inputs(self) -> Any:
        return self._span.inputs

    @property
    def outputs(self) -> Any:
        return self._span.outputs

    @property
    def start_time_ns(self) -> int:
        return self._span.start_time_ns

    @property
    def end_time_ns(self) -> int:
        return self._span.end_time_ns

    @property
    def attributes(self) -> dict[str, Any]:
        # Pass-through bag plus the MLflow bookkeeping keys re-materialised, so
        # custom evaluators that read raw keys keep working.
        attrs = dict(self._span.attributes)
        attrs[_SPAN_TYPE_KEY] = self._span.kind.value
        if self._span.inputs is not None:
            attrs[_INPUTS_KEY] = self._span.inputs
        if self._span.outputs is not None:
            attrs[_OUTPUTS_KEY] = self._span.outputs
        return attrs


class _CompatTraceData:
    def __init__(self, spans: list[_CompatSpan]) -> None:
        self.spans = spans


class _CompatTrace:
    """Duck-typed stand-in for ``mlflow.entities.Trace``."""

    def __init__(self, trace: Trace) -> None:
        self._trace = trace
        self.data = _CompatTraceData([_CompatSpan(s) for s in trace.spans])

    def search_spans(self, span_type: Any = None, name: str | None = None) -> list[_CompatSpan]:
        """Filter spans by ``span_type`` (string or MLflow ``SpanType``) and/or ``name``."""
        result = self.data.spans
        if span_type is not None:
            wanted = str(span_type)
            result = [s for s in result if s.span_type == wanted]
        if name is not None:
            result = [s for s in result if s.name == name]
        return result


def to_mlflow_trace(trace: Trace) -> _CompatTrace:
    """Wrap a :class:`~ragpill.trace.Trace` in an MLflow-shaped duck type.

    Convenience escape hatch for un-migrated custom ``SpanBaseEvaluator``
    subclasses. Prefer migrating to ``ragpill.trace.Span``.
    """
    return _CompatTrace(trace)
