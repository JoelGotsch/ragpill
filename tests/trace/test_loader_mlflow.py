"""Tests for ``ragpill.trace.loader.from_mlflow_trace``.

Traces are materialised via the public ``mlflow.start_span`` API against a
temporary SQLite backend so the loader sees the same span shape production
code produces. Mirrors ``tests/test_report_trace.py``'s fixture style.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import mlflow
import pytest
from mlflow.entities import SpanType, Trace

from ragpill.trace import Trace as RagpillTrace, from_mlflow_trace
from ragpill.trace.model import SpanKind


@pytest.fixture(autouse=True)
def _isolated_mlflow_backend() -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    tmp = tempfile.mkdtemp(prefix="ragpill_trace_loader_")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 'mlflow.db')}")
    mlflow.set_experiment(f"exp-{os.path.basename(tmp)}")
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _build_trace(setup) -> Trace:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
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


def test_from_mlflow_trace_returns_ragpill_trace():
    rt = from_mlflow_trace(_simple_trace())
    assert isinstance(rt, RagpillTrace)
    assert rt.dialect == "mlflow"
    assert rt.trace_id
    assert len(rt.spans) == 3


def test_span_kinds_and_names_preserved():
    rt = from_mlflow_trace(_simple_trace())
    by_name = {s.name: s for s in rt.spans}
    assert by_name["root"].kind is SpanKind.AGENT
    assert by_name["retrieve"].kind is SpanKind.RETRIEVER
    assert by_name["generate"].kind is SpanKind.LLM
    # Every span is tagged with the dialect that produced it.
    assert all(s.dialect == "mlflow" for s in rt.spans)


def test_parent_child_links_preserved():
    rt = from_mlflow_trace(_simple_trace())
    by_name = {s.name: s for s in rt.spans}
    root = by_name["root"]
    assert root.parent_id is None
    # retrieve and generate nest under root.
    assert by_name["retrieve"].parent_id == root.span_id
    assert by_name["generate"].parent_id == root.span_id


def test_inputs_outputs_lifted():
    rt = from_mlflow_trace(_simple_trace())
    by_name = {s.name: s for s in rt.spans}
    assert by_name["root"].inputs == {"q": "hi"}
    assert by_name["root"].outputs == "answer"
    assert by_name["generate"].inputs == "prompt"
    assert by_name["retrieve"].outputs == ["doc1", "doc2"]


def test_mlflow_internal_attrs_not_in_passthrough():
    rt = from_mlflow_trace(_simple_trace())
    for s in rt.spans:
        assert "mlflow.spanType" not in s.attributes
        assert "mlflow.spanInputs" not in s.attributes
        assert "mlflow.spanOutputs" not in s.attributes
