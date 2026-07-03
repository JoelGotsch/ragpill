"""Tests for adapter registry selection + dialect detection."""

from __future__ import annotations

from ragpill.trace import detect_dialect
from ragpill.trace.registry import adapter_by_name, select_adapter


def _span(attrs: dict) -> dict:
    return {"span_id": "s", "attributes": attrs}


def test_detect_each_dialect():
    assert detect_dialect(_span({"mlflow.spanType": "LLM"})) == "mlflow"
    assert detect_dialect(_span({"openinference.span.kind": "LLM"})) == "openinference"
    assert detect_dialect(_span({"gen_ai.system": "openai"})) == "gen_ai"


def test_no_match_returns_none():
    assert detect_dialect(_span({"some.other.key": 1})) is None
    assert select_adapter(_span({})) is None


def test_priority_mlflow_beats_gen_ai_on_mixed_span():
    # MLflow autolog of pydantic-ai co-emits gen_ai events; mlflow wins.
    adapter = select_adapter(_span({"mlflow.spanType": "LLM", "gen_ai.system": "openai"}))
    assert adapter is not None and adapter.name == "mlflow"


def test_adapter_by_name():
    assert adapter_by_name("openinference").name == "openinference"  # type: ignore[union-attr]
    assert adapter_by_name("nope") is None
