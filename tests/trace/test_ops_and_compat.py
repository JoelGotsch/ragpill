"""Tests for ``ragpill.trace.ops.filter_to_subtree`` and the compat shim."""

from __future__ import annotations

from ragpill.trace import filter_to_subtree
from ragpill.trace.compat import to_mlflow_trace
from ragpill.trace.model import Span, SpanKind, Trace


def _span(span_id: str, parent_id: str | None, kind: SpanKind = SpanKind.CHAIN, **kw) -> Span:
    return Span(
        span_id=span_id,
        parent_id=parent_id,
        trace_id="tr",
        name=span_id,
        kind=kind,
        start_time_ns=0,
        end_time_ns=1,
        **kw,
    )


def _tree() -> Trace:
    # root -> a -> a1 ; root -> b
    return Trace(
        trace_id="tr",
        spans=[
            _span("root", None),
            _span("a", "root"),
            _span("a1", "a"),
            _span("b", "root"),
        ],
    )


def test_filter_to_subtree_keeps_only_descendants():
    sub = filter_to_subtree(_tree(), "a")
    assert sub is not None
    assert {s.span_id for s in sub.spans} == {"a", "a1"}


def test_filter_to_subtree_root_returns_all():
    sub = filter_to_subtree(_tree(), "root")
    assert sub is not None
    assert {s.span_id for s in sub.spans} == {"root", "a", "a1", "b"}


def test_filter_to_subtree_missing_returns_none():
    assert filter_to_subtree(_tree(), "nope") is None


def test_filter_to_subtree_does_not_mutate_original():
    original = _tree()
    filter_to_subtree(original, "a")
    assert len(original.spans) == 4


def test_compat_exposes_mlflow_surface():
    trace = Trace(
        trace_id="tr",
        spans=[
            _span("r", None, kind=SpanKind.RETRIEVER, outputs=["doc1"]),
            _span("t", None, kind=SpanKind.TOOL, inputs={"q": 1}),
            _span("c", None, kind=SpanKind.CHAIN, attributes={"ragpill_x": 1}),
        ],
    )
    compat = to_mlflow_trace(trace)
    # .data.spans and per-span surface
    assert len(compat.data.spans) == 3
    retr = compat.search_spans(span_type="RETRIEVER")
    assert len(retr) == 1 and retr[0].outputs == ["doc1"]
    # internal mlflow keys re-materialised on the attribute bag
    chain = compat.search_spans(name="c")[0]
    assert chain.attributes["mlflow.spanType"] == "CHAIN"
    assert chain.attributes["ragpill_x"] == 1
    tool = compat.search_spans(span_type="TOOL")[0]
    assert tool.attributes["mlflow.spanInputs"] == {"q": 1}
