"""Test LiteralQuoteEvaluator"""

from unittest.mock import patch

import pytest
from mlflow.entities import Document

from ragpill.base import EvaluatorMetadata
from ragpill.eval_types import EvaluatorContext
from ragpill.evaluators import LiteralQuoteEvaluator


def create_test_context(inputs: str, output: str) -> EvaluatorContext:
    """Helper to create an EvaluatorContext for testing."""
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


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="'no longer outstanding at this stage' does not mean 'resolved'.",
            metadata={"source": "31-May-2025_GOV-2025-25.txt"},
        ),
        Document(
            page_content="Another document with different content about nuclear verification processes.",
            metadata={"source": "other-document.txt"},
        ),
        Document(
            page_content="""This document contains the phrase exact match test for validation purposes. This is a very long quote
that spans multiple lines with Different CAPITALIZATION.""",
            metadata={"source": "test-doc.txt"},
        ),
        Document(
            page_content="7.  Following the change of Government in Syria towards the end of 2024,\n\n    the Director General contacted the new Syrian Minister of Foreign\n\n    Affairs and Expatriates, HE Mr Asaad Hassan al-Shaybani, in a letter\n\n    dated 14 January 2025, to convey the importance of continuing and\n\n    reinforcing cooperation between Syria and the Agency to address\n\n    unresolved safeguards issues related to Syria's past nuclear\n\n    activities.\n8.  Syria, in its reply dated 30 April 2025, invited the Director\n\n    General to visit Syria in early June 2025 and indicated that it had\n\n    no objection to the Agency's request to conduct an \"exceptional\n\n    visit\" to one of the three locations, as specified by the Agency.",
            metadata={"source": "syria-report.txt"},
        ),
    ]


@pytest.fixture
def evaluator():
    """Create a LiteralQuoteEvaluator evaluator instance."""
    return LiteralQuoteEvaluator(
        expected=True,
        tags={"quotation", "verification"},
    )


# ---------------------------------------------------------------------------
# Evaluator run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_all_quotes_found(evaluator, sample_documents):
    output = """The report states:
> "'no longer outstanding at this stage' does not mean 'resolved'."
(File: [31-May-2025_GOV-2025-25.txt](link), Paragraph: 38)"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_insensitive_quote_matching(evaluator, sample_documents):
    output = """The report states:
> '"No longer outstanding at this stage" does not mean "Resolved".'
(File: [31-May-2025_GOV-2025-25.txt](link), Paragraph: 38)"""
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_case_insensitive_chat_output(evaluator, sample_documents):
    output = """The report states:
> "'No longer outstanding at this stage' does not mean 'Resolved'."
(File: [31-May-2025_GOV-2025-25.txt](link), Paragraph: 38)"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_multi_line_quote(evaluator, sample_documents):
    output = """The report includes the following statement:
> "this is a very long quote that spans multiple lines with Different capitalization."
(File: [test-doc.txt](link))"""
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_quote_from_wrong_source(evaluator, sample_documents):
    """Quote is from the wrong source but still evaluates to True."""
    output = """The report states:
> "'no longer outstanding at this stage' does not mean 'resolved'."
(File: [fake.txt](link), Paragraph: 38)"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_block_quote(evaluator, sample_documents):
    output = """The report includes the following statement:
> "this is a very long Quote
> that spans Multiple Lines with different capitalization."
(File: [test-doc.txt](link))"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_quote_not_found(evaluator, sample_documents):
    output = """The report claims:
> "This quote does not exist in any document."
(File: [nonexistent.txt](link))"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is False
        assert "this quote does not exist in any document" in result.reason
        assert "nonexistent.txt" in result.reason


@pytest.mark.anyio
async def test_partial_quotes_found(evaluator, sample_documents):
    output = """First quote:
> "'no longer outstanding at this stage' does not mean 'resolved'."

Second quote:
> "This quote is not in any document."
(File: [fake.txt](link))"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is False
        assert "this quote is not in any document" in result.reason


@pytest.mark.anyio
async def test_syria_quote(evaluator, sample_documents):
    output = "The Director General highlighted in his letter to Syria's new Minister of Foreign Affairs and Expatriates, HE Mr Asaad Hassan al-Shaybani, dated 14 January 2025, the importance of continuing and reinforcing cooperation between Syria and the Agency to address unresolved safeguards issues related to Syria's past nuclear activities.\n\n> \"The Director General contacted the new Syrian Minister of Foreign Affairs and Expatriates, HE Mr Asaad Hassan al-Shaybani, in a letter dated 14 January 2025, to convey the importance of continuing and reinforcing cooperation between Syria and the Agency to address unresolved safeguards issues related to Syria's past nuclear activities.\"\n(File: 01-september-2025_gov-2025-52.txt, Para: 7)"
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_no_quotes_in_output(evaluator, sample_documents):
    output = "This is just regular text without any quotes."

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True
        assert "No quotes found" in result.reason


@pytest.mark.anyio
async def test_ellipsis_in_quote(evaluator, sample_documents):
    output = """The report states:
> "... document with different ... verification processes..."
(File: [other-document.txt](link))"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_whitespace_differences_ignored(evaluator, sample_documents):
    output = """Quote with different whitespace:
>   \t"Another     document\t with different Content"
"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_multiple_quotes_some_not_found(evaluator, sample_documents):
    output = """First quote (exists):
> "exact match test for validation purposes"

Second quote (missing):
> "This is completely made up"
(File: [fake-source.txt](link))

Third quote (missing without file):
> "Another fake quote"
"""

    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is False
        assert "this is completely made up" in result.reason
        assert "fake-source.txt" in result.reason
        assert "another fake quote" in result.reason


# ---------------------------------------------------------------------------
# from_csv_line
# ---------------------------------------------------------------------------


def test_from_csv_line_basic():
    evaluator = LiteralQuoteEvaluator.from_csv_line(
        expected=True,
        tags={"tag1", "tag2"},
        check="",
    )

    assert evaluator.expected is True
    assert evaluator.tags == {"tag1", "tag2"}


def test_from_csv_line_with_attributes():
    evaluator = LiteralQuoteEvaluator.from_csv_line(
        expected=False,
        tags={"verification"},
        check="",
        custom_attr="custom_value",
    )

    assert evaluator.expected is False
    assert evaluator.attributes["custom_attr"] == "custom_value"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_empty_documents(evaluator):
    output = """Quote:
> "Some quote"
"""

    with patch.object(evaluator, "get_documents", return_value=[]):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_quote_with_special_characters(evaluator):
    docs = [Document(page_content="The formula is E=mc² and π≈3.14159.", metadata={"source": "science.txt"})]

    output = """Scientific quote:
> "E=mc² and π≈3.14159"
"""

    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_empty_output(evaluator, sample_documents):
    with patch.object(evaluator, "get_documents", return_value=sample_documents):
        ctx = create_test_context("some input", "")
        result = await evaluator.run(ctx)
        assert result.value is True
        assert "No quotes found" in result.reason


@pytest.mark.anyio
async def test_real_iran_quote(evaluator):
    docs = [
        Document(
            page_content="51. No new information was provided by Iran with respect to the issue of\n\n    testing of centrifuges using nuclear material until October 2003. In\n\n    its letter of 21 October 2003, Iran acknowledged that, in order to\n\n    ensure the performance of centrifuge machines, a limited number of\n\n    tests using small amounts of UF~6~ imported in 1991 had been carried\n\n    out at the Kalaye Electric Company. According to Iran, the first\n\n    test of the centrifuges was conducted in 1998 using an inert gas\n\n    (xenon). Series of tests using UF~6~ were performed between 1999\n\n    and 2002. In the course of the last series of tests, an enrichment\n\n    level of 1.2% U-235 was achieved.",
            metadata={"source": "science.txt"},
        )
    ]

    output = """The first known use of uranium hexafluoride (UF₆) by Iran occurred between **1999 and 2002**, as confirmed by Iran itself in its letter to the IAEA on 21 October 2003. During this period, Iran conducted a series of tests using UF₆ at the Kalaye Electric Company in Tehran, following earlier tests with inert gases (xenon) in 1998. These tests achieved an enrichment level of 1.2% U-235.

> "According to Iran, the first test of the centrifuges was conducted in 1998 using an inert gas (xenon). Series of tests using UF~6~ were performed between 1999 and 2002. In the course of the last series of tests, an enrichment level of 1.2% U-235 was achieved."
(source: [GOV/2003/75](https://portal.sg.iaea.org/sg/SGVI/OfficeLAN/Chat%20SG%20%20Iran%20reports/PROD/10-November-2003_GOV-2003-75.docx?web=1), paragraph 51)

This marks the earliest documented use of UF₆ in Irans nuclear program. While Iran had imported UF₆ as early as 1991, the first actual testing of centrifuges with UF₆ began in 1999.
"""

    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("some input", output)
        result = await evaluator.run(ctx)
        assert result.value is True
