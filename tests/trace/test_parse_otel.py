"""Tests for ``ragpill.trace.parse_otel`` and ``RagpillTraceSettings``."""

from __future__ import annotations

import warnings

import pytest

from ragpill.settings import RagpillTraceSettings
from ragpill.trace import parse_otel
from ragpill.trace.model import SpanKind


def _norm_span(span_id: str, attrs: dict, **kw) -> dict:
    base = {
        "trace_id": "tr",
        "span_id": span_id,
        "parent_span_id": None,
        "name": span_id,
        "start_time_unix_nano": 1,
        "end_time_unix_nano": 2,
        "attributes": attrs,
        "events": [],
        "status": {"code": "OK", "message": None},
    }
    base.update(kw)
    return base


def test_auto_detects_per_span_dialect():
    trace = parse_otel(
        [
            _norm_span("a", {"openinference.span.kind": "RETRIEVER"}),
            _norm_span("b", {"gen_ai.system": "openai"}),
        ],
        dialect="auto",
    )
    by_id = {s.span_id: s for s in trace.spans}
    assert by_id["a"].dialect == "openinference"
    assert by_id["b"].dialect == "gen_ai"
    assert trace.trace_id == "tr"


def test_forced_dialect_overrides_detection():
    # An mlflow-shaped span forced through gen_ai.
    trace = parse_otel([_norm_span("a", {"mlflow.spanType": "LLM", "gen_ai.system": "x"})], dialect="gen_ai")
    assert trace.spans[0].dialect == "gen_ai"


def test_unknown_dialect_raises():
    with pytest.raises(ValueError, match="unknown dialect"):
        parse_otel([_norm_span("a", {"gen_ai.system": "x"})], dialect="nope")


def test_unrecognised_span_falls_back_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # No signature matches → the configured fallback adapter (gen_ai) parses
        # it best-effort, and a warning is emitted.
        trace = parse_otel([_norm_span("a", {"unrecognised.key": 1})], dialect="auto")
    assert trace.spans[0].dialect == "gen_ai"
    assert any("matched no dialect adapter" in str(w.message) for w in caught)


def test_unrecognised_span_uses_universal_when_no_fallback_adapter():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # fallback_dialect names no registered adapter → universal extractor,
        # which keeps the generic input.value/output.value payload.
        trace = parse_otel(
            [_norm_span("a", {"unrecognised.key": 1, "input.value": "x"})],
            dialect="auto",
            fallback_dialect="none",
        )
    assert trace.spans[0].dialect == "unknown"
    assert trace.spans[0].inputs == "x"


def test_universal_span_extracts_generic_payload():
    from ragpill.trace.fallback import universal_span

    span = universal_span(_norm_span("a", {"input.value": "in", "output.value": "out", "k": 1}))
    assert span.dialect == "unknown"
    assert span.kind is SpanKind.UNKNOWN
    assert span.inputs == "in"
    assert span.outputs == "out"
    assert span.attributes["k"] == 1


def test_otlp_json_resource_spans_decoded():
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "sp1",
                                "name": "llm",
                                "startTimeUnixNano": "100",
                                "endTimeUnixNano": "200",
                                "attributes": [
                                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                                    {"key": "llm.token_count.prompt", "value": {"intValue": "12"}},
                                ],
                                "status": {"code": "OK"},
                            }
                        ]
                    }
                ]
            }
        ]
    }
    trace = parse_otel(payload, dialect="auto")
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.span_id == "sp1"
    assert span.kind is SpanKind.LLM
    assert span.usage.input_tokens == 12
    assert span.start_time_ns == 100


def test_bad_source_raises():
    with pytest.raises(ValueError, match="unsupported source"):
        parse_otel("not a span source")


def test_trace_settings_defaults_and_env(monkeypatch):
    assert RagpillTraceSettings().dialect == "auto"
    assert RagpillTraceSettings().fallback_dialect == "gen_ai"
    monkeypatch.setenv("RAGPILL_TRACE_DIALECT", "openinference")
    assert RagpillTraceSettings().dialect == "openinference"
