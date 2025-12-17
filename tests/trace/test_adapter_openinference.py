"""Unit tests for ``OpenInferenceAdapter.from_otel`` (indexed flat keys)."""

from __future__ import annotations

import pytest

from ragpill.trace.adapters import AdapterDeclined
from ragpill.trace.adapters.openinference import OpenInferenceAdapter
from ragpill.trace.model import SpanKind


def _span(attributes: dict, **overrides) -> dict:
    base = {
        "trace_id": "tr",
        "span_id": "s1",
        "parent_span_id": None,
        "name": "span",
        "start_time_unix_nano": 1,
        "end_time_unix_nano": 2,
        "attributes": attributes,
        "events": [],
        "status": {"code": "OK", "message": None},
    }
    base.update(overrides)
    return base


def test_signature_and_kind():
    span = OpenInferenceAdapter.from_otel(_span({"openinference.span.kind": "RETRIEVER"}))
    assert OpenInferenceAdapter.signature_attributes() == ("openinference.span.kind",)
    assert span.kind is SpanKind.RETRIEVER
    assert span.dialect == "openinference"


def test_evaluator_kind_maps_to_unknown():
    span = OpenInferenceAdapter.from_otel(_span({"openinference.span.kind": "EVALUATOR"}))
    assert span.kind is SpanKind.UNKNOWN


def test_indexed_messages_reconstructed():
    span = OpenInferenceAdapter.from_otel(
        _span(
            {
                "openinference.span.kind": "LLM",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "hi",
                "llm.input_messages.1.message.role": "system",
                "llm.input_messages.1.message.content": "be brief",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "hello",
            }
        )
    )
    assert [(m.role, m.content) for m in span.messages_in] == [("user", "hi"), ("system", "be brief")]
    assert [(m.role, m.content) for m in span.messages_out] == [("assistant", "hello")]


def test_retrieval_documents_reconstructed():
    span = OpenInferenceAdapter.from_otel(
        _span(
            {
                "openinference.span.kind": "RETRIEVER",
                "retrieval.documents.0.document.content": "doc one",
                "retrieval.documents.0.document.id": "d0",
                "retrieval.documents.0.document.score": 0.9,
                "retrieval.documents.0.document.metadata": {"source": "a.txt"},
                "retrieval.documents.1.document.content": "doc two",
            }
        )
    )
    assert len(span.documents) == 2
    assert span.documents[0].content == "doc one"
    assert span.documents[0].id == "d0"
    assert span.documents[0].score == 0.9
    assert span.documents[0].metadata == {"source": "a.txt"}
    assert span.documents[1].content == "doc two"
    assert span.documents[1].metadata == {}


def test_io_model_and_usage():
    span = OpenInferenceAdapter.from_otel(
        _span(
            {
                "openinference.span.kind": "LLM",
                "input.value": "the prompt",
                "output.value": "the answer",
                "llm.model_name": "gpt-4o",
                "llm.token_count.prompt": 12,
                "llm.token_count.completion": 7,
                "llm.token_count.total": 19,
            }
        )
    )
    assert span.inputs == "the prompt"
    assert span.outputs == "the answer"
    assert span.model == "gpt-4o"
    assert (span.usage.input_tokens, span.usage.output_tokens, span.usage.total_tokens) == (12, 7, 19)


def test_missing_span_id_declines():
    with pytest.raises(AdapterDeclined):
        OpenInferenceAdapter.from_otel(_span({"openinference.span.kind": "LLM"}, span_id=None))
