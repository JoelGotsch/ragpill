"""Unit tests for ``GenAIAdapter.from_otel`` (OTel GenAI convention)."""

from __future__ import annotations

import pytest

from ragpill.trace.adapters import AdapterDeclined
from ragpill.trace.adapters.gen_ai import GenAIAdapter
from ragpill.trace.model import SpanKind


def _span(**overrides) -> dict:
    base = {
        "trace_id": "tr",
        "span_id": "s1",
        "parent_span_id": None,
        "name": "chat",
        "start_time_unix_nano": 1,
        "end_time_unix_nano": 2,
        "attributes": {"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"},
        "events": [],
        "status": {"code": "OK", "message": None},
    }
    base.update(overrides)
    return base


def test_signature_is_gen_ai_system():
    assert GenAIAdapter.signature_attributes() == ("gen_ai.system",)


def test_kind_is_llm_and_model_read():
    span = GenAIAdapter.from_otel(_span())
    assert span.kind is SpanKind.LLM
    assert span.model == "gpt-4o"
    assert span.dialect == "gen_ai"


def test_response_model_preferred_over_request():
    span = GenAIAdapter.from_otel(
        _span(attributes={"gen_ai.system": "openai", "gen_ai.request.model": "req", "gen_ai.response.model": "resp"})
    )
    assert span.model == "resp"


def test_messages_read_from_events():
    span = GenAIAdapter.from_otel(
        _span(
            events=[
                {"name": "gen_ai.system.message", "attributes": {"content": "be brief"}},
                {"name": "gen_ai.user.message", "attributes": {"content": "hi"}},
                {"name": "gen_ai.choice", "attributes": {"content": "hello"}},
            ]
        )
    )
    assert [m.role for m in span.messages_in] == ["system", "user"]
    assert span.messages_in[1].content == "hi"
    assert len(span.messages_out) == 1
    assert span.messages_out[0].content == "hello"


def test_usage_and_params():
    span = GenAIAdapter.from_otel(
        _span(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.request.temperature": 0.0,
                "gen_ai.request.max_tokens": 256,
            }
        )
    )
    assert span.usage.input_tokens == 10
    assert span.usage.output_tokens == 5
    assert span.model_parameters == {"temperature": 0.0, "max_tokens": 256}


def test_missing_span_id_declines():
    with pytest.raises(AdapterDeclined):
        GenAIAdapter.from_otel(_span(span_id=None))
