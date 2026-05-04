"""Unit tests for the local evaluation primitives in ``ragpill.eval_types``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ragpill.base import BaseEvaluator
from ragpill.eval_types import (
    Case,
    Dataset,
    EvaluationReason,
    EvaluationResult,
    EvaluatorContext,
    EvaluatorSource,
)


def test_evaluation_reason_holds_value_and_optional_reason():
    reason = EvaluationReason(value=True)
    assert reason.value is True
    assert reason.reason is None

    reason2 = EvaluationReason(value=False, reason="failed")
    assert reason2.value is False
    assert reason2.reason == "failed"


def test_evaluator_source_holds_name_and_arguments():
    source = EvaluatorSource(name="my_evaluator", arguments={"k": "v"})
    assert source.name == "my_evaluator"
    assert source.arguments == {"k": "v"}

    source_default = EvaluatorSource(name="bare")
    assert source_default.arguments == {}


def test_evaluation_result_composes_evaluator_source():
    source = EvaluatorSource(name="x")
    result = EvaluationResult(name="assertion1", value=True, reason="ok", source=source)
    assert result.source is source
    assert result.name == "assertion1"


def test_case_and_dataset_defaults():
    case = Case(inputs="hi")
    assert case.inputs == "hi"
    assert case.name is None
    assert case.metadata is None
    assert case.expected_output is None
    assert case.evaluators == []

    ds = Dataset[str, str, Any](cases=[case])
    assert ds.cases == [case]
    assert ds.evaluators == []


def test_evaluator_context_is_kw_only_and_has_expected_fields():
    ctx = EvaluatorContext(
        name="case-1",
        inputs="hello",
        metadata=None,
        expected_output="hi",
        output="hi",
        duration=0.5,
    )
    assert ctx.name == "case-1"
    assert ctx.inputs == "hello"
    assert ctx.expected_output == "hi"
    assert ctx.output == "hi"
    assert ctx.duration == 0.5
    assert ctx.attributes == {}
    assert ctx.metrics == {}
    # trace/run_span_id default to None (populated in Phase 1+)
    assert ctx.trace is None
    assert ctx.run_span_id is None


def test_evaluator_context_rejects_positional_args():
    with pytest.raises(TypeError):
        EvaluatorContext("case-1", "hello", None, "hi", "hi", 0.5)  # type: ignore[call-arg]


@dataclass
class _ExampleEvaluator(BaseEvaluator):
    """Plain evaluator used for serialization tests."""

    pattern: str = "default"
    other: list[str] = field(default_factory=list)


def test_build_serialization_arguments_returns_only_non_default_fields():
    ev = _ExampleEvaluator(pattern="custom", tags={"a"})
    args = ev.build_serialization_arguments()
    # Defaults are skipped; evaluation_name is a UUID default_factory so not included
    assert args.get("pattern") == "custom"
    assert args.get("tags") == {"a"}
    assert "other" not in args
    assert "expected" not in args


def test_get_serialization_name_returns_class_name():
    ev = _ExampleEvaluator(pattern="x")
    assert ev.get_serialization_name() == "_ExampleEvaluator"
    assert _ExampleEvaluator.get_serialization_name() == "_ExampleEvaluator"
