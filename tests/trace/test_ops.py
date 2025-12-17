"""Tests for ``ragpill.trace.ops.filter_to_subtree``."""

from __future__ import annotations

from functools import partial

from conftest import make_span

from ragpill.trace import filter_to_subtree
from ragpill.trace.model import Trace

_span = partial(make_span, trace_id="tr", end_time_ns=1)


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
