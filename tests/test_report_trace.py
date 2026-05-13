"""Unit tests for ``ragpill.report._trace.render_spans``.

Traces are materialized via the public ``mlflow.start_span`` API against a
temporary SQLite backend so the test exercises the same span shape that
production code produces.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import mlflow
import pytest
from mlflow.entities import SpanType, Trace

from ragpill.report._trace import REDACTED, render_spans


@pytest.fixture(autouse=True)
def _isolated_mlflow_backend() -> Iterator[None]:
    """Each test gets a private SQLite backend; restore the previous URI on teardown."""
    previous = mlflow.get_tracking_uri()
    tmp = tempfile.mkdtemp(prefix="ragpill_report_trace_")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 'mlflow.db')}")
    mlflow.set_experiment(f"exp-{os.path.basename(tmp)}")
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _build_trace(setup) -> Trace:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    """Run ``setup()`` inside a fresh mlflow run and return the resulting Trace."""
    with mlflow.start_run():
        setup()
    traces: list[Trace] = mlflow.search_traces(return_type="list", max_results=1)  # pyright: ignore[reportAssignmentType]
    assert traces, "expected at least one trace"
    return traces[0]


def _simple_trace() -> Trace:
    def build() -> None:
        with mlflow.start_span(name="root", span_type=SpanType.AGENT) as s:
            s.set_inputs({"q": "hi"})
            s.set_outputs("answer")
            with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as r:
                r.set_outputs(["doc1", "doc2"])
            with mlflow.start_span(name="generate", span_type=SpanType.LLM) as g:
                g.set_inputs("prompt")
                g.set_outputs("out")

    return _build_trace(build)


def test_render_spans_returns_empty_for_none_trace():
    assert render_spans(None) == ""


def test_render_spans_renders_full_tree_by_default():
    out = render_spans(_simple_trace(), max_chars=10_000)
    # All three spans appear with their types.
    assert "root (AGENT" in out
    assert "retrieve (RETRIEVER" in out
    assert "generate (LLM" in out
    # Children are indented under the root.
    lines = out.splitlines()
    root_line = next(line for line in lines if "root" in line)
    child_lines = [line for line in lines if "retrieve" in line or "generate" in line]
    assert all(line.startswith("  ") for line in child_lines)
    assert not root_line.startswith(" ")


def test_render_spans_filter_types_drops_unmatched_spans():
    out = render_spans(_simple_trace(), max_chars=10_000, filter_types={"LLM"})
    assert "generate (LLM" in out
    assert "retrieve" not in out
    assert "root" not in out


def test_render_spans_filter_keeps_spans_with_ragpill_attributes():
    def build() -> None:
        with mlflow.start_span(name="root", span_type=SpanType.AGENT) as s:
            s.set_attribute("ragpill_marker", "yes")
            s.set_inputs("hi")
        # No LLM/RETRIEVER spans at all.

    out = render_spans(_build_trace(build), filter_types={"LLM"})
    # The AGENT span isn't in the filter set but carries a ragpill_* attribute.
    assert "root (AGENT" in out


def test_render_spans_subtree_root():
    trace = _simple_trace()
    retriever_id = next(s.span_id for s in trace.data.spans if s.name == "retrieve")
    out = render_spans(trace, root_span_id=retriever_id, max_chars=10_000)
    assert "retrieve" in out
    assert "root" not in out
    assert "generate" not in out


def test_render_spans_unknown_subtree_root_returns_empty():
    out = render_spans(_simple_trace(), root_span_id="does-not-exist")
    assert out == ""


def test_render_spans_redacts_default_secret_keys():
    def build() -> None:
        with mlflow.start_span(name="call", span_type=SpanType.LLM) as s:
            s.set_attribute("api_key", "sk-secret")
            s.set_attribute("safe", "ok")
            s.set_inputs({"authorization": "Bearer xxx", "prompt": "hi"})

    out = render_spans(_build_trace(build), max_chars=10_000)
    assert "sk-secret" not in out
    assert "Bearer xxx" not in out
    assert REDACTED in out
    # Non-secret values still render.
    assert "ok" in out
    assert "prompt" in out


def test_render_spans_redact_can_be_disabled():
    def build() -> None:
        with mlflow.start_span(name="call", span_type=SpanType.LLM) as s:
            s.set_attribute("api_key", "sk-secret")

    out = render_spans(_build_trace(build), redact=False, max_chars=10_000)
    assert "sk-secret" in out
    assert REDACTED not in out


def test_render_spans_custom_redact_patterns():
    def build() -> None:
        with mlflow.start_span(name="call", span_type=SpanType.LLM) as s:
            s.set_attribute("password", "p@ss")
            s.set_attribute("api_key", "sk-secret")

    out = render_spans(_build_trace(build), redact_patterns=[r"(?i)password"], max_chars=10_000)
    # Only the custom pattern hits — api_key is no longer redacted.
    assert "p@ss" not in out
    assert "sk-secret" in out


def test_render_spans_budget_truncation_emits_overflow_marker():
    def build() -> None:
        with mlflow.start_span(name="root", span_type=SpanType.AGENT):
            for i in range(10):
                with mlflow.start_span(name=f"child-{i}", span_type=SpanType.LLM) as c:
                    c.set_inputs(f"prompt-{i}")
                    c.set_outputs(f"out-{i}")

    out = render_spans(_build_trace(build), max_chars=200)
    assert len(out) <= 250  # trailing "more spans" marker can push slightly over
    assert "more spans" in out
