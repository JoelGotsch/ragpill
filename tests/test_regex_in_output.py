"""Tests for RegexInOutputEvaluator."""
import asyncio
import pytest

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import RegexInOutputEvaluator
from pydantic_evals.evaluators import EvaluatorContext


def create_test_context(inputs: str, output: str) -> EvaluatorContext:
    """Helper to build an EvaluatorContext for tests."""
    return EvaluatorContext(
        inputs=inputs,
        output=output,
        metadata=EvaluatorMetadata(expected=True, mandatory=True),
        name="test",
        expected_output=None,
        duration=0,
        _span_tree=None,
        attributes={},
        metrics={},
    )


class TestRun:
    def test_matches_pattern(self):
        evaluator = RegexInOutputEvaluator(pattern="success", expected=True, mandatory=True, tags=set())
        ctx = create_test_context("input", "operation success")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is True
        assert "matched output" in result.reason

    def test_no_match(self):
        evaluator = RegexInOutputEvaluator(pattern="error", expected=True, mandatory=True, tags=set())
        ctx = create_test_context("input", "all good here")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is False
        assert "did not match output" in result.reason

    def test_inline_flags(self):
        evaluator = RegexInOutputEvaluator(pattern="(?i)pass", expected=True, mandatory=True, tags=set())
        ctx = create_test_context("input", "PASSED")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is True
        
class TestFromCSVLine:
    def test_from_csv_plain_string(self):
        evaluator = RegexInOutputEvaluator.from_csv_line(
            expected=True,
            mandatory=False,
            tags={"t"},
            check="done|ok",
        )

        assert evaluator.pattern == "done|ok"
        assert evaluator.expected is True
        assert evaluator.mandatory is False
        assert evaluator.tags == {"t"}

    def test_from_csv_json(self):
        evaluator = RegexInOutputEvaluator.from_csv_line(
            expected=True,
            mandatory=True,
            tags={"json"},
            check='{"pattern": "(?i)success"}',
        )

        assert evaluator.pattern == "(?i)success"

    def test_from_csv_empty_raises(self):
        with pytest.raises(ValueError):
            RegexInOutputEvaluator.from_csv_line(
                expected=True,
                mandatory=True,
                tags=set(),
                check="  ",
            )


class TestValidation:
    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError):
            RegexInOutputEvaluator(pattern="[", expected=True, mandatory=True, tags=set())
