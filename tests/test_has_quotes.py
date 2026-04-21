"""Test HasQuotesEvaluator."""

import pytest
from pydantic_evals.evaluators import EvaluatorContext

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import HasQuotesEvaluator


def create_test_context(inputs: str, output: str) -> EvaluatorContext:
    """Helper to create an EvaluatorContext for testing."""
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


@pytest.fixture
def evaluator():
    """Create a HasQuotesEvaluator instance with default settings."""
    return HasQuotesEvaluator(
        min_quotes=1,
        expected=True,
        tags={"quotation", "format"},
    )


# ---------------------------------------------------------------------------
# Evaluator run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_has_minimum_quotes_default(evaluator):
    output = """The report states:
> "This is a quote."
Other text without quotes.
End of text."""

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is True
    assert "Found 1 quote(s)" in result.reason
    assert "minimum required: 1" in result.reason


@pytest.mark.anyio
async def test_has_more_than_minimum():
    evaluator = HasQuotesEvaluator(min_quotes=2, expected=True)
    output = """Multiple quotes:
> First quote

> Second quote

> Third quote
"""

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is True
    assert "Found 3 quote(s)" in result.reason
    assert "minimum required: 2" in result.reason


@pytest.mark.anyio
async def test_insufficient_quotes():
    evaluator = HasQuotesEvaluator(min_quotes=3, expected=True)
    output = """Only one quote:
> Just this one
"""

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is False
    assert "Found only 1 quote(s)" in result.reason
    assert "3 required" in result.reason


@pytest.mark.anyio
async def test_no_quotes_when_required():
    evaluator = HasQuotesEvaluator(min_quotes=1, expected=True)
    output = "This text has no quotes at all."

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is False
    assert "Found only 0 quote(s)" in result.reason
    assert "1 required" in result.reason


@pytest.mark.anyio
async def test_zero_minimum_always_passes():
    evaluator = HasQuotesEvaluator(min_quotes=0, expected=True)
    output = "No quotes here."

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is True
    assert "Found 0 quote(s)" in result.reason
    assert "minimum required: 0" in result.reason


# ---------------------------------------------------------------------------
# from_csv_line
# ---------------------------------------------------------------------------


def test_from_csv_line_with_number():
    evaluator = HasQuotesEvaluator.from_csv_line(
        expected=True,
        tags={"tag1", "tag2"},
        check="5",
    )

    assert evaluator.min_quotes == 5
    assert evaluator.expected is True
    assert evaluator.tags == {"tag1", "tag2"}


def test_from_csv_line_with_attributes():
    evaluator = HasQuotesEvaluator.from_csv_line(
        expected=False,
        tags={"verification"},
        check="2",
        custom_attr="custom_value",
    )

    assert evaluator.min_quotes == 2
    assert evaluator.attributes["custom_attr"] == "custom_value"


# ---------------------------------------------------------------------------
# expected=False
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_expected_false_no_quotes_passes():
    evaluator = HasQuotesEvaluator(min_quotes=1, expected=False)
    output = "No quotes in this text."

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is False


@pytest.mark.anyio
async def test_expected_false_with_quotes_fails():
    evaluator = HasQuotesEvaluator(min_quotes=1, expected=False)
    output = "> This has a quote"

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_only_quote_markers():
    evaluator = HasQuotesEvaluator(min_quotes=1, expected=True)
    output = """>
>
>
"""

    ctx = create_test_context("input", output)
    result = await evaluator.run(ctx)

    assert result.value is False
    assert "Found only 0 quote(s)" in result.reason
