"""Tests for RegexInSourcesEvaluator."""
import asyncio
from unittest.mock import patch

import pytest
from mlflow.entities import Document
from pydantic_evals.evaluators import EvaluatorContext

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import RegexInSourcesEvaluator


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


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This document contains important information about nuclear safeguards.",
            metadata={"source": "safeguards.txt"}
        ),
        Document(
            page_content="The verification process requires multiple inspection rounds.",
            metadata={"source": "verification.txt"}
        ),
        Document(
            page_content="""Multi-line content here.
Line 2 with some data.
Line 3 contains SECRET information.
Final line.""",
            metadata={"source": "multiline.txt"}
        ),
        Document(
            page_content="Technical specifications: Model X-100, Version 2.5.1",
            metadata={"source": "specs.txt"}
        ),
    ]


@pytest.fixture
def evaluator():
    """Create a basic RegexInSourcesEvaluator instance."""
    return RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags={"regex", "sources"},
        check="important",
    )


class TestRun:
    """Test the main evaluator run method."""

    def test_pattern_found_in_document(self, evaluator, sample_documents):
        """Test when the regex pattern is found in a document."""
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True
            assert "found" in result.reason.lower()

    def test_pattern_not_found(self, sample_documents):
        """Test when the regex pattern is not found in any document."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="nonexistent_pattern_xyz",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False
            assert "not found" in result.reason.lower()

    def test_case_insensitive_with_inline_flag(self, sample_documents):
        """Test case-insensitive matching using inline flag."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="(?i)SECRET",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_multiline_pattern(self, sample_documents):
        """Test regex matching pattern that spans content from different lines."""
        # Note: Content is normalized (whitespace collapsed), so multiline flags
        # like (?m) with ^ anchors don't work as expected. This tests that
        # patterns still match content that originally spanned multiple lines.
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="line 2 with some data",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_dotall_flag(self, sample_documents):
        """Test dotall matching using inline flag for multi-line content."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="(?s)multi-line.*final",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_combined_flags(self, sample_documents):
        """Test combining multiple inline flags."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="(?ims)MULTI-LINE.*SECRET",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_regex_alternation(self, sample_documents):
        """Test regex with alternation (OR pattern)."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="safeguards|verification",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_regex_with_version_pattern(self, sample_documents):
        """Test regex matching version numbers."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check=r"version \d+\.\d+\.\d+",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_empty_documents_list(self):
        """Test when there are no documents to check."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="anything",
        )
        with patch.object(evaluator, 'get_documents', return_value=[]):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False

    def test_pattern_in_first_document(self, sample_documents):
        """Test that pattern is found when in the first document only."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="nuclear safeguards",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_pattern_in_last_document(self, sample_documents):
        """Test that pattern is found when in the last document only."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="model x-100",
        )
        with patch.object(evaluator, 'get_documents', return_value=sample_documents):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True


class TestFromCSVLine:
    """Test creating evaluator from CSV line."""

    def test_from_csv_basic(self):
        """Test basic creation from CSV line."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags={"tag1"},
            check="pattern",
        )
        assert evaluator.pattern == "pattern"
        assert evaluator.expected is True

        assert evaluator.tags == {"tag1"}

    def test_from_csv_with_special_chars(self):
        """Test creating evaluator with special regex characters."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check=r"test\.pattern\?",
        )
        assert evaluator.pattern == r"test\.pattern\?"

    def test_from_csv_with_kwargs(self):
        """Test creating evaluator with additional attributes."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=False,

            tags={"test"},
            check="pattern",
            custom_attr="custom_value",
            another_attr=123,
        )
        assert evaluator.expected is False
        assert evaluator.attributes["custom_attr"] == "custom_value"
        assert evaluator.attributes["another_attr"] == 123

    def test_from_csv_normalizes_pattern(self):
        """Test that the pattern is normalized (unicode NFKC and lowercased)."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="UPPERCASE Pattern",
        )
        # Pattern should be normalized (lowercased)
        assert evaluator.pattern == "uppercase pattern"

    def test_from_csv_empty_tags(self):
        """Test creating evaluator with empty tags."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="pattern",
        )
        assert evaluator.tags == set()

    def test_from_csv_multiple_tags(self):
        """Test creating evaluator with multiple tags."""
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags={"tag1", "tag2", "tag3"},
            check="pattern",
        )
        assert evaluator.tags == {"tag1", "tag2", "tag3"}


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_document(self):
        """Test with a single document."""
        docs = [
            Document(
                page_content="Single document with test content.",
                metadata={"source": "single.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="test content",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_empty_document_content(self):
        """Test with documents that have empty content."""
        docs = [
            Document(page_content="", metadata={"source": "empty.txt"}),
            Document(page_content="   ", metadata={"source": "whitespace.txt"}),
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="pattern",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False

    def test_special_regex_characters_in_content(self):
        """Test matching content with special regex characters."""
        docs = [
            Document(
                page_content="Price: $100.00 (50% off!)",
                metadata={"source": "prices.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check=r"\$\d+\.\d{2}",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_unicode_content(self):
        """Test matching Unicode content."""
        docs = [
            Document(
                page_content="The formula is E=mc² and π≈3.14159.",
                metadata={"source": "science.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="e=mc2",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_very_long_pattern(self):
        """Test with a very long regex pattern."""
        long_content = "This is a very long sentence that we want to match exactly in the document."
        docs = [
            Document(
                page_content=long_content,
                metadata={"source": "long.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="very long sentence",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_word_boundary_pattern(self):
        """Test regex with word boundaries."""
        docs = [
            Document(
                page_content="The test passed successfully.",
                metadata={"source": "test.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check=r"\btest\b",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_pattern_with_groups(self):
        """Test regex pattern with capture groups."""
        docs = [
            Document(
                page_content="Date: 2025-01-15",
                metadata={"source": "date.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check=r"(\d{4})-(\d{2})-(\d{2})",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True


class TestNormalization:
    """Test text normalization behavior."""

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized in content."""
        docs = [
            Document(
                page_content="Multiple   spaces   here",
                metadata={"source": "spaces.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="multiple spaces here",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_case_normalization(self):
        """Test that case is normalized (lowercase)."""
        docs = [
            Document(
                page_content="UPPERCASE TEXT HERE",
                metadata={"source": "upper.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="uppercase text",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_realistic_case_normalization(self):
        """Test that case is normalized (lowercase)."""
        docs = [
            Document(
                page_content="7.  Following the change of Government in Syria towards the end of 2024,\n\n    the Director General contacted the new Syrian Minister of Foreign\n\n    Affairs and Expatriates, HE Mr Asaad Hassan al-Shaybani, in a letter\n\n    dated 14 January 2025, to convey the importance of continuing and\n\n    reinforcing cooperation between Syria and the Agency to address\n\n    unresolved safeguards issues related to Syria's past nuclear\n\n    activities.\n8.  Syria, in its reply dated 30 April 2025, invited the Director\n\n    General to visit Syria in early June 2025 and indicated that it had\n\n    no objection to the Agency's request to conduct an \"exceptional\n\n    visit\" to one of the three locations, as specified by the Agency.",
                metadata={"source": "syria-report.txt"}
            )
        ]
        evaluator = RegexInSourcesEvaluator.from_csv_line(
            expected=True,

            tags=set(),
            check="the director general contacted the new syrian minister of foreign affairs and expatriates, he mr asaad hassan al-shaybani, in a letter dated 14 january 2025, to convey the importance of continuing and reinforcing cooperation between syria and the agency to address unresolved safeguards issues related to syria’s past nuclear activities.",
        )
        with patch.object(evaluator, 'get_documents', return_value=docs):
            ctx = create_test_context("some input", "some output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
