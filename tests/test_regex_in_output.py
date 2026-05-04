"""Tests for RegexInOutputEvaluator."""

import pytest

from ragpill.base import EvaluatorMetadata
from ragpill.eval_types import EvaluatorContext
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
        attributes={},
        metrics={},
    )


# ---------------------------------------------------------------------------
# Evaluator run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_matches_pattern():
    evaluator = RegexInOutputEvaluator(pattern="success", expected=True, tags=set())
    ctx = create_test_context("input", "operation success")

    result = await evaluator.run(ctx)

    assert result.value is True
    assert "matched output" in result.reason


@pytest.mark.anyio
async def test_no_match():
    evaluator = RegexInOutputEvaluator(pattern="error", expected=True, tags=set())
    ctx = create_test_context("input", "all good here")

    result = await evaluator.run(ctx)

    assert result.value is False
    assert "did not match output" in result.reason


@pytest.mark.anyio
async def test_inline_flags():
    evaluator = RegexInOutputEvaluator(pattern="(?i)pass", expected=True, tags=set())
    ctx = create_test_context("input", "PASSED")

    result = await evaluator.run(ctx)

    assert result.value is True


# ---------------------------------------------------------------------------
# from_csv_line
# ---------------------------------------------------------------------------


def test_from_csv_plain_string():
    evaluator = RegexInOutputEvaluator.from_csv_line(
        expected=True,
        tags={"t"},
        check="done|ok",
    )

    assert evaluator.pattern == "done|ok"
    assert evaluator.expected is True
    assert evaluator.tags == {"t"}


def test_from_csv_json():
    evaluator = RegexInOutputEvaluator.from_csv_line(
        expected=True,
        tags={"json"},
        check='{"pattern": "(?i)success"}',
    )

    assert evaluator.pattern == "(?i)success"


def test_from_csv_empty_raises():
    with pytest.raises(ValueError):
        RegexInOutputEvaluator.from_csv_line(
            expected=True,
            tags=set(),
            check="  ",
        )


# ---------------------------------------------------------------------------
# Normalization (parametrized end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern, output, should_match, desc",
    [
        ("success", "Operation SUCCESS completed", True, "casefold output"),
        ("SUCCESS", "operation success completed", True, "casefold pattern"),
        ("uf6", "UF₆ compound detected", True, "NFKC subscript digit"),
        ("uf₆", "UF6 compound detected", True, "NFKC subscript in pattern"),
        ("hello world", "hello   \t  world", True, "whitespace collapse in output"),
        ("hello   world", "hello world", True, "whitespace collapse in pattern"),
        ("he said 'hello'", "he said \u201chello\u201d", True, "curly double quotes in output"),
        ("he said \u201chello\u201d", "he said 'hello'", True, "curly double quotes in pattern"),
        ("uf6", "UF~6~ gas", True, "tilde subscript in output"),
        ("the end", "the end.", True, "trailing period in output"),
        ("the end.", "the end", True, "trailing period in pattern"),
        ("alpha", "beta gamma", False, "genuine mismatch"),
        ("(?s)start.*end", "start\nmiddle\nend", True, "dotall flag"),
        (r"\d{3}-\d{4}", "call 555-1234 now", True, "digit pattern"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.anyio
async def test_normalization_end_to_end(pattern, output, should_match, desc):
    evaluator = RegexInOutputEvaluator.from_csv_line(expected=True, tags=set(), check=pattern)
    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)
    assert result.value is should_match, f"Failed: {desc}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_regex_raises():
    with pytest.raises(ValueError):
        RegexInOutputEvaluator(pattern="[", expected=True, tags=set())
