"""Round-trip tests for ``ragpill.trace.serde``."""

from __future__ import annotations

import json

from ragpill.trace import Trace, trace_from_dict, trace_to_dict
from ragpill.trace.model import Document, Message, Span, SpanKind, Usage


def _rich_trace() -> Trace:
    return Trace(
        trace_id="tr-1",
        spans=[
            Span(
                span_id="s0",
                parent_id=None,
                trace_id="tr-1",
                name="root",
                kind=SpanKind.AGENT,
                start_time_ns=1,
                end_time_ns=2,
                status="OK",
                inputs={"q": "hi"},
                outputs="answer",
                attributes={"ragpill_tag": "x", "n": 3},
            ),
            Span(
                span_id="s1",
                parent_id="s0",
                trace_id="tr-1",
                name="llm",
                kind=SpanKind.LLM,
                start_time_ns=3,
                end_time_ns=4,
                status="ERROR",
                status_message="boom",
                messages_in=[Message(role="user", content="hi")],
                messages_out=[Message(role="assistant", content="yo", tool_call_id="t1")],
                documents=[Document(content="doc", id="d1", score=0.9, metadata={"k": "v"})],
                model="gpt-4o",
                model_parameters={"temperature": 0.0},
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01),
                events=[{"name": "e", "attributes": {"a": 1}}],
            ),
        ],
        session_id="sess-1",
        tags=["t1"],
        metadata={"m": "v"},
        dialect="mlflow",
    )


def test_round_trip_preserves_everything():
    original = _rich_trace()
    restored = trace_from_dict(trace_to_dict(original))
    assert restored == original


def test_dict_is_json_serialisable():
    payload = json.dumps(trace_to_dict(_rich_trace()))
    restored = trace_from_dict(json.loads(payload))
    assert restored == _rich_trace()


def test_kind_serialises_as_string():
    d = trace_to_dict(_rich_trace())
    assert d["spans"][0]["kind"] == "AGENT"
    assert isinstance(d["spans"][0]["kind"], str)


def test_unknown_keys_ignored():
    d = trace_to_dict(_rich_trace())
    d["future_field"] = "ignored"
    d["spans"][0]["future_span_field"] = "ignored"
    restored = trace_from_dict(d)
    assert restored == _rich_trace()
