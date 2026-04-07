"""Tests for RegexInOutputEvaluator."""

import asyncio

import pytest
from pydantic_evals.evaluators import EvaluatorContext

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import RegexInOutputEvaluator


def create_test_context(inputs: str, output: str) -> EvaluatorContext:
    """Helper to build an EvaluatorContext for tests."""
    return EvaluatorContext(
        inputs=inputs,
        output=output,
        metadata=EvaluatorMetadata(expected=True),
        name="test",
        expected_output=None,
        duration=0,
        _span_tree=None,
        attributes={},
        metrics={},
    )


class TestRun:
    def test_matches_pattern(self):
        evaluator = RegexInOutputEvaluator(pattern="success", expected=True, tags=set())
        ctx = create_test_context("input", "operation success")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is True
        assert "matched output" in result.reason

    def test_no_match(self):
        evaluator = RegexInOutputEvaluator(pattern="error", expected=True, tags=set())
        ctx = create_test_context("input", "all good here")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is False
        assert "did not match output" in result.reason

    def test_inline_flags(self):
        evaluator = RegexInOutputEvaluator(pattern="(?i)pass", expected=True, tags=set())
        ctx = create_test_context("input", "PASSED")

        result = asyncio.run(evaluator.run(ctx))

        assert result.value is True


class TestFromCSVLine:
    def test_from_csv_plain_string(self):
        evaluator = RegexInOutputEvaluator.from_csv_line(
            expected=True,
            tags={"t"},
            check="done|ok",
        )

        assert evaluator.pattern == "done|ok"
        assert evaluator.expected is True
        assert evaluator.tags == {"t"}

    def test_from_csv_json(self):
        evaluator = RegexInOutputEvaluator.from_csv_line(
            expected=True,
            tags={"json"},
            check='{"pattern": "(?i)success"}',
        )

        assert evaluator.pattern == "(?i)success"

    def test_from_csv_empty_raises(self):
        with pytest.raises(ValueError):
            RegexInOutputEvaluator.from_csv_line(
                expected=True,
                tags=set(),
                check="  ",
            )


class TestComprehensiveNormalization:
    """One comprehensive test exercising all normalization behaviour end-to-end.

    _normalize_text is applied to *both* the pattern and the output.
    This test would have caught the original bug where output was not normalized.
    """

    @pytest.mark.parametrize(
        "pattern, output, should_match, desc",
        [
            # Case-folding — (?i) is redundant
            ("success", "Operation SUCCESS completed", True, "casefold output"),
            ("SUCCESS", "operation success completed", True, "casefold pattern"),
            # Unicode NFKC
            ("uf6", "UF₆ compound detected", True, "NFKC subscript digit"),
            ("uf₆", "UF6 compound detected", True, "NFKC subscript in pattern"),
            # Whitespace collapsing
            ("hello world", "hello   \t  world", True, "whitespace collapse in output"),
            ("hello   world", "hello world", True, "whitespace collapse in pattern"),
            # Quote normalization (curly → straight single quote)
            ("he said 'hello'", 'he said \u201chello\u201d', True, "curly double quotes in output"),
            ("he said \u201chello\u201d", "he said 'hello'", True, "curly double quotes in pattern"),
            # Markdown subscript stripping
            ("uf6", "UF~6~ gas", True, "tilde subscript in output"),
            # Trailing period stripping
            ("the end", "the end.", True, "trailing period in output"),
            ("the end.", "the end", True, "trailing period in pattern"),
            # No match
            ("alpha", "beta gamma", False, "genuine mismatch"),
            # Regex features still work
            ("(?s)start.*end", "start\nmiddle\nend", True, "dotall flag"),
            (r"\d{3}-\d{4}", "call 555-1234 now", True, "digit pattern"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_normalization_end_to_end(self, pattern, output, should_match, desc):
        evaluator = RegexInOutputEvaluator.from_csv_line(expected=True, tags=set(), check=pattern)
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        assert result.value is should_match, f"Failed: {desc}"


class TestValidation:
    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError):
            RegexInOutputEvaluator(pattern="[", expected=True, tags=set())
