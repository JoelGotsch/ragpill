"""Tests for _filter_trace_to_subtree."""

from unittest.mock import MagicMock

from ragpill.evaluators import _filter_trace_to_subtree


def _make_span(span_id: str, parent_id: str | None = None, **attrs: object) -> MagicMock:
    span = MagicMock()
    span.span_id = span_id
    span.parent_id = parent_id
    span.attributes = dict(attrs)
    span.inputs = None
    span.outputs = None
    return span


def _make_trace(spans: list[MagicMock]) -> MagicMock:
    trace = MagicMock()
    trace.data.spans = spans
    trace.info.trace_id = "trace-1"
    return trace


# ---------------------------------------------------------------------------
# _filter_trace_to_subtree
# ---------------------------------------------------------------------------


def test_returns_only_subtree():
    root = _make_span("root")
    run1 = _make_span("run-1", parent_id="root")
    llm1 = _make_span("llm-1", parent_id="run-1")
    ret1 = _make_span("ret-1", parent_id="run-1")
    run2 = _make_span("run-2", parent_id="root")
    llm2 = _make_span("llm-2", parent_id="run-2")

    trace = _make_trace([root, run1, llm1, ret1, run2, llm2])
    filtered = _filter_trace_to_subtree(trace, "run-1")

    filtered_ids = {s.span_id for s in filtered.data.spans}
    assert filtered_ids == {"run-1", "llm-1", "ret-1"}


def test_single_span():
    leaf = _make_span("leaf", parent_id="parent")
    trace = _make_trace([leaf])
    filtered = _filter_trace_to_subtree(trace, "leaf")
    assert len(filtered.data.spans) == 1
    assert filtered.data.spans[0].span_id == "leaf"


def test_preserves_span_data():
    span = _make_span("s1", input_key="k1")
    trace = _make_trace([span])
    filtered = _filter_trace_to_subtree(trace, "s1")
    assert filtered.data.spans[0].attributes["input_key"] == "k1"


def test_unknown_span_id():
    span = _make_span("s1")
    trace = _make_trace([span])
    filtered = _filter_trace_to_subtree(trace, "nonexistent")
    assert len(filtered.data.spans) == 0
