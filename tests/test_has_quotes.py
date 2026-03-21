"""Test HasQuotesEvaluator."""
import asyncio
from unittest.mock import patch

import pytest

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import HasQuotesEvaluator
from pydantic_evals.evaluators import EvaluatorContext


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


class TestEvaluatorRun:
    """Test the main evaluator run method."""

    def test_has_minimum_quotes_default(self, evaluator):
        """Test when output has the default minimum (1) quote."""
        output = """The report states:
> "This is a quote."
Other text without quotes.
End of text."""
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is True
        assert "Found 1 quote(s)" in result.reason
        assert "minimum required: 1" in result.reason

    def test_has_more_than_minimum(self):
        """Test when output has more quotes than required."""
        evaluator = HasQuotesEvaluator(min_quotes=2, expected=True)
        output = """Multiple quotes:
> First quote

> Second quote

> Third quote
"""
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is True
        assert "Found 3 quote(s)" in result.reason
        assert "minimum required: 2" in result.reason

    def test_insufficient_quotes(self):
        """Test when output has fewer quotes than required."""
        evaluator = HasQuotesEvaluator(min_quotes=3, expected=True)
        output = """Only one quote:
> Just this one
"""
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is False
        assert "Found only 1 quote(s)" in result.reason
        assert "3 required" in result.reason

    def test_no_quotes_when_required(self):
        """Test when no quotes are present but they are required."""
        evaluator = HasQuotesEvaluator(min_quotes=1, expected=True)
        output = "This text has no quotes at all."
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is False
        assert "Found only 0 quote(s)" in result.reason
        assert "1 required" in result.reason

    def test_zero_minimum_always_passes(self):
        """Test that min_quotes=0 always passes."""
        evaluator = HasQuotesEvaluator(min_quotes=0, expected=True)
        output = "No quotes here."
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is True
        assert "Found 0 quote(s)" in result.reason
        assert "minimum required: 0" in result.reason


class TestFromCSVLine:
    """Test creating evaluator from CSV line."""

    def test_from_csv_line_with_number(self):
        """Test creating evaluator from CSV with specific minimum."""
        evaluator = HasQuotesEvaluator.from_csv_line(
            expected=True,
            tags={"tag1", "tag2"},
            check="5",
        )
        
        assert evaluator.min_quotes == 5
        assert evaluator.expected is True
        assert evaluator.tags == {"tag1", "tag2"}


    def test_from_csv_line_with_attributes(self):
        """Test creating evaluator from CSV with additional attributes."""
        evaluator = HasQuotesEvaluator.from_csv_line(
            expected=False,
            tags={"verification"},
            check="2",
            custom_attr="custom_value",
        )
        
        assert evaluator.min_quotes == 2
        assert evaluator.attributes["custom_attr"] == "custom_value"


class TestExpectedFalse:
    """Test evaluator with expected=False (verifying quotes are NOT present)."""

    def test_expected_false_no_quotes_passes(self):
        """Test that expected=False passes when no quotes are present."""
        evaluator = HasQuotesEvaluator(
            min_quotes=1,
            expected=False,  # Expect the check to fail (no quotes)
        )
        output = "No quotes in this text."
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        # The evaluator returns False (not enough quotes), which matches expected=False
        assert result.value is False

    def test_expected_false_with_quotes_fails(self):
        """Test that expected=False fails when quotes are present."""
        evaluator = HasQuotesEvaluator(
            min_quotes=1,
            expected=False,  # Expect the check to fail
        )
        output = "> This has a quote"
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        # The evaluator returns True (has quotes), which doesn't match expected=False
        assert result.value is True


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_only_quote_markers(self):
        """Test output with only > markers but no content."""
        evaluator = HasQuotesEvaluator(min_quotes=1, expected=True)
        output = """>
>
>
"""
        
        ctx = create_test_context("input", output)
        result = asyncio.run(evaluator.run(ctx))
        
        assert result.value is False
        assert "Found only 0 quote(s)" in result.reason



