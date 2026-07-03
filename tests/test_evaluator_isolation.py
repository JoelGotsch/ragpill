"""Evaluator trace-isolation semantics on the neutral trace model.

``SpanBaseEvaluator.get_trace`` must scope evaluation to the run's own
subtree. When ``ctx.run_span_id`` is missing from the trace (e.g. the run's
spans were still in flight when the case trace was fetched), it returns an
empty span set — never the full case trace, which would silently score spans
from other repeats of the same case.
"""

from __future__ import annotations

from ragpill.base import EvaluatorMetadata
from ragpill.eval_types import EvaluatorContext
from ragpill.evaluators import RegexInSourcesEvaluator
from ragpill.trace import Document, Span, SpanKind, Trace


def _span(span_id: str, parent_id: str | None, kind: SpanKind = SpanKind.CHAIN, **overrides) -> Span:
    base: dict = {
        "span_id": span_id,
        "parent_id": parent_id,
        "trace_id": "tr-1",
        "name": span_id,
        "kind": kind,
        "start_time_ns": 0,
        "end_time_ns": 0,
    }
    base.update(overrides)
    return Span(**base)


def _ctx(trace: Trace, run_span_id: str) -> EvaluatorContext:
    return EvaluatorContext(
        inputs="in",
        output="out",
        metadata=EvaluatorMetadata(expected=True),
        name="test",
        expected_output=None,
        duration=0,
        attributes={},
        metrics={},
        trace=trace,
        run_span_id=run_span_id,
    )


def _evaluator() -> RegexInSourcesEvaluator:
    return RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="anything")


def test_get_trace_filters_to_run_subtree():
    trace = Trace(
        trace_id="tr-1",
        spans=[
            _span("case-root", None),
            _span("run-0", "case-root"),
            _span("run-0-child", "run-0"),
            _span("run-1", "case-root"),
        ],
    )
    scoped = _evaluator().get_trace(_ctx(trace, "run-0"))
    assert {s.span_id for s in scoped.spans} == {"run-0", "run-0-child"}


def test_unknown_run_span_id_yields_empty_spans_not_full_trace():
    trace = Trace(
        trace_id="tr-1",
        spans=[
            _span("case-root", None),
            _span("run-0", "case-root"),
        ],
    )
    scoped = _evaluator().get_trace(_ctx(trace, "run-9-missing"))
    # The other repeats' spans must NOT leak into this run's evaluation.
    assert scoped.spans == []


def test_get_documents_reads_neutral_documents_field():
    # OpenInference-style span: documents lifted to Span.documents, outputs a raw string.
    retriever = _span(
        "ret-0",
        "run-0",
        kind=SpanKind.RETRIEVER,
        outputs='{"raw": "json string"}',
        documents=[Document(content="lifted doc", metadata={"source": "s"})],
    )
    trace = Trace(trace_id="tr-1", spans=[_span("run-0", None), retriever])
    docs = _evaluator().get_documents(_ctx(trace, "run-0"))
    assert [d.content for d in docs] == ["lifted doc"]


def test_get_documents_falls_back_to_page_content_outputs():
    # MLflow/LangChain-style span: documents only present as page_content dicts in outputs.
    retriever = _span(
        "ret-0",
        "run-0",
        kind=SpanKind.RETRIEVER,
        outputs=[{"page_content": "legacy doc", "metadata": {"source": "s"}}],
    )
    trace = Trace(trace_id="tr-1", spans=[_span("run-0", None), retriever])
    docs = _evaluator().get_documents(_ctx(trace, "run-0"))
    assert [d.content for d in docs] == ["legacy doc"]
