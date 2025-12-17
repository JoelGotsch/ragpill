"""Unit tests for ``MLflowAdapter.from_otel`` — pure dict in, ragpill Span out.

These exercise the adapter in isolation (no ``mlflow`` import): the adapter's
contract is a normalised OTLP-JSON span dict, so the tests build those dicts
by hand. The loader's mlflow->dict half is covered in ``test_loader_mlflow``.
"""

from __future__ import annotations

import pytest

from ragpill.trace.adapters import AdapterDeclined, MLflowAdapter
from ragpill.trace.model import SpanKind


def _span_dict(**overrides) -> dict:
    base = {
        "trace_id": "tr-1",
        "span_id": "sp-1",
        "parent_span_id": None,
        "name": "root",
        "kind": None,
        "start_time_unix_nano": 1000,
        "end_time_unix_nano": 2000,
        "attributes": {"mlflow.spanType": "AGENT"},
        "events": [],
        "status": {"code": "OK", "message": None},
    }
    base.update(overrides)
    return base


def test_signature_attribute_is_span_type():
    assert MLflowAdapter.signature_attributes() == ("mlflow.spanType",)


def test_from_otel_maps_core_fields():
    span = MLflowAdapter.from_otel(_span_dict(parent_span_id="parent-x"))
    assert span.span_id == "sp-1"
    assert span.parent_id == "parent-x"
    assert span.trace_id == "tr-1"
    assert span.name == "root"
    assert span.kind is SpanKind.AGENT
    assert span.start_time_ns == 1000
    assert span.end_time_ns == 2000
    assert span.status == "OK"
    assert span.dialect == "mlflow"


@pytest.mark.parametrize(
    ("span_type", "expected"),
    [
        ("LLM", SpanKind.LLM),
        ("RETRIEVER", SpanKind.RETRIEVER),
        ("CHAT_MODEL", SpanKind.CHAT_MODEL),
        ("something-weird", SpanKind.UNKNOWN),
    ],
)
def test_span_type_maps_to_kind(span_type, expected):
    span = MLflowAdapter.from_otel(_span_dict(attributes={"mlflow.spanType": span_type}))
    assert span.kind is expected


def test_inputs_and_outputs_lifted_from_attributes():
    span = MLflowAdapter.from_otel(
        _span_dict(
            attributes={
                "mlflow.spanType": "LLM",
                "mlflow.spanInputs": {"q": "hi"},
                "mlflow.spanOutputs": "answer",
            }
        )
    )
    assert span.inputs == {"q": "hi"}
    assert span.outputs == "answer"


def test_internal_keys_dropped_from_passthrough_but_user_attrs_kept():
    span = MLflowAdapter.from_otel(
        _span_dict(
            attributes={
                "mlflow.spanType": "TOOL",
                "mlflow.spanInputs": {"a": 1},
                "mlflow.spanFunctionName": "search",
                "mlflow.traceRequestId": "req-1",
                "ragpill_tag": "keep-me",
                "custom.user.key": 42,
            }
        )
    )
    # Bookkeeping keys are surfaced via first-class fields, not the bag.
    assert "mlflow.spanType" not in span.attributes
    assert "mlflow.spanInputs" not in span.attributes
    assert "mlflow.spanFunctionName" not in span.attributes
    assert "mlflow.traceRequestId" not in span.attributes
    # Everything else passes through untouched.
    assert span.attributes["ragpill_tag"] == "keep-me"
    assert span.attributes["custom.user.key"] == 42


def test_status_message_carried():
    span = MLflowAdapter.from_otel(_span_dict(status={"code": "ERROR", "message": "boom"}))
    assert span.status == "ERROR"
    assert span.status_message == "boom"


def test_missing_span_id_declines():
    with pytest.raises(AdapterDeclined):
        MLflowAdapter.from_otel(_span_dict(span_id=None))
