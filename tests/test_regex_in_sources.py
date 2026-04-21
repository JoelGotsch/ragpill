"""Tests for RegexInSourcesEvaluator."""

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
            metadata={"source": "safeguards.txt"},
        ),
        Document(
            page_content="The verification process requires multiple inspection rounds.",
            metadata={"source": "verification.txt"},
        ),
        Document(
            page_content="""Multi-line content here.
Line 2 with some data.
Line 3 contains SECRET information.
Final line.""",
            metadata={"source": "multiline.txt"},
        ),
        Document(page_content="Technical specifications: Model X-100, Version 2.5.1", metadata={"source": "specs.txt"}),
    ]


@pytest.fixture
def evaluator():
    """Create a basic RegexInSourcesEvaluator instance."""
    return RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags={"regex", "sources"},
        check="important",
    )


# ---------------------------------------------------------------------------
# Evaluator run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pattern_found_in_document(evaluator, sample_documents):
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True
        assert "found" in result.reason.lower()


@pytest.mark.anyio
async def test_pattern_not_found(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="nonexistent_pattern_xyz",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is False
        assert "not found" in result.reason.lower()


@pytest.mark.anyio
async def test_case_insensitive_with_inline_flag(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="(?i)SECRET",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_multiline_pattern(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="line 2 with some data",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_dotall_flag(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="(?s)multi-line.*final",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_combined_flags(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="(?ims)MULTI-LINE.*SECRET",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_regex_alternation(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="safeguards|verification",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_regex_with_version_pattern(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check=r"version \d+\.\d+\.\d+",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_empty_documents_list():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="anything",
    )
    with patch.object(evaluator, "get_documents", return_value=[]):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_pattern_in_first_document(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="nuclear safeguards",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_pattern_in_last_document(sample_documents):
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="model x-100",
    )
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


# ---------------------------------------------------------------------------
# from_csv_line
# ---------------------------------------------------------------------------


def test_from_csv_basic():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags={"tag1"},
        check="pattern",
    )
    assert evaluator.pattern == "pattern"
    assert evaluator.expected is True
    assert evaluator.tags == {"tag1"}


def test_from_csv_with_special_chars():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check=r"test\.pattern\?",
    )
    assert evaluator.pattern == r"test\.pattern\?"


def test_from_csv_with_kwargs():
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


def test_from_csv_normalizes_pattern():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="UPPERCASE Pattern",
    )
    assert evaluator.pattern == "uppercase pattern"


def test_from_csv_empty_tags():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="pattern",
    )
    assert evaluator.tags == set()


def test_from_csv_multiple_tags():
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags={"tag1", "tag2", "tag3"},
        check="pattern",
    )
    assert evaluator.tags == {"tag1", "tag2", "tag3"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_single_document():
    docs = [Document(page_content="Single document with test content.", metadata={"source": "single.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="test content",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_empty_document_content():
    docs = [
        Document(page_content="", metadata={"source": "empty.txt"}),
        Document(page_content="   ", metadata={"source": "whitespace.txt"}),
    ]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="pattern",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_special_regex_characters_in_content():
    docs = [Document(page_content="Price: $100.00 (50% off!)", metadata={"source": "prices.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check=r"\$\d+\.\d{2}",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_unicode_content():
    docs = [Document(page_content="The formula is E=mc² and π≈3.14159.", metadata={"source": "science.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="e=mc2",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_very_long_pattern():
    long_content = "This is a very long sentence that we want to match exactly in the document."
    docs = [Document(page_content=long_content, metadata={"source": "long.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="very long sentence",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_word_boundary_pattern():
    docs = [Document(page_content="The test passed successfully.", metadata={"source": "test.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check=r"\btest\b",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_pattern_with_groups():
    docs = [Document(page_content="Date: 2025-01-15", metadata={"source": "date.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check=r"(\d{4})-(\d{2})-(\d{2})",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_whitespace_normalization():
    docs = [Document(page_content="Multiple   spaces   here", metadata={"source": "spaces.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="multiple spaces here",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_case_normalization():
    docs = [Document(page_content="UPPERCASE TEXT HERE", metadata={"source": "upper.txt"})]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="uppercase text",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_realistic_case_normalization():
    docs = [
        Document(
            page_content="7.  Following the change of Government in Syria towards the end of 2024,\n\n    the Director General contacted the new Syrian Minister of Foreign\n\n    Affairs and Expatriates, HE Mr Asaad Hassan al-Shaybani, in a letter\n\n    dated 14 January 2025, to convey the importance of continuing and\n\n    reinforcing cooperation between Syria and the Agency to address\n\n    unresolved safeguards issues related to Syria's past nuclear\n\n    activities.\n8.  Syria, in its reply dated 30 April 2025, invited the Director\n\n    General to visit Syria in early June 2025 and indicated that it had\n\n    no objection to the Agency's request to conduct an \"exceptional\n\n    visit\" to one of the three locations, as specified by the Agency.",
            metadata={"source": "syria-report.txt"},
        )
    ]
    evaluator = RegexInSourcesEvaluator.from_csv_line(
        expected=True,
        tags=set(),
        check="the director general contacted the new syrian minister of foreign affairs and expatriates, he mr asaad hassan al-shaybani, in a letter dated 14 january 2025, to convey the importance of continuing and reinforcing cooperation between syria and the agency to address unresolved safeguards issues related to syria\u2019s past nuclear activities.",
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", "some output")
        result = await evaluator.run(ctx)
        assert result.value is True


# ---------------------------------------------------------------------------
# Comprehensive normalization (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern, page_content, should_match, desc",
    [
        ("important", "IMPORTANT notice", True, "casefold content"),
        ("IMPORTANT", "important notice", True, "casefold pattern"),
        ("uf6", "UF₆ compound", True, "NFKC subscript digit in content"),
        ("uf₆", "UF6 compound", True, "NFKC subscript digit in pattern"),
        ("hello world", "hello   \t  world", True, "whitespace collapse in content"),
        ("hello   world", "hello world", True, "whitespace collapse in pattern"),
        ("he said 'hello'", "he said \u201chello\u201d", True, "curly double quotes in content"),
        ("he said \u201chello\u201d", "he said 'hello'", True, "curly double quotes in pattern"),
        ("uf6", "UF~6~ gas", True, "tilde subscript in content"),
        ("the end", "the end.", True, "trailing period in content"),
        ("the end.", "the end", True, "trailing period in pattern"),
        ("alpha", "beta gamma", False, "genuine mismatch"),
        (r"\d{3}-\d{4}", "call 555-1234 now", True, "digit pattern"),
        ("anything", "", False, "empty content"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.anyio
async def test_normalization_end_to_end(pattern, page_content, should_match, desc):
    evaluator = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check=pattern)
    docs = [Document(page_content=page_content, metadata={"source": "test.txt"})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is should_match, f"Failed: {desc}"
