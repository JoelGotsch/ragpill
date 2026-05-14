# ruff: noqa: RUF001
"""Regression tests for LiteralQuoteEvaluator improvements (issue #8).

Each test exercises one of the nine documented false-positive categories
from issue #8 using neutral synthetic content. Categories A, B, C, D, G, I
map to the failure inventory in
``plans/literal-quote-evaluator-improvements.md``.

The Category I parametrize block intentionally uses visually-ambiguous
Unicode characters (en/em/figure dashes, NBSP, soft hyphen, zero-width
space) — exactly the situation the normalization is meant to collapse.
File-scope RUF001 noqa silences the (correct-in-other-files) warning.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mlflow.entities import Document

from ragpill.base import EvaluatorMetadata
from ragpill.eval_types import EvaluatorContext
from ragpill.evaluators import LiteralQuoteEvaluator
from ragpill.utils import _extract_markdown_quotes, _normalize_text  # pyright: ignore[reportPrivateUsage]


def _ctx(output: str) -> EvaluatorContext[str, str, EvaluatorMetadata]:
    return EvaluatorContext(
        inputs="some input",
        output=output,
        metadata=EvaluatorMetadata(expected=True),
        name="test",
        expected_output=None,
        duration=0,
        attributes={},
        metrics={},
    )


# ---------------------------------------------------------------------------
# Category A — pandoc / LaTeX / markdown artifacts in source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, agent_quote",
    [
        # markdown tilde subscript (already supported, regression guard)
        ("Acme delivered H~2~O to the lab.", "acme delivered h2o to the lab"),
        # LaTeX math subscript variants
        ("feeding distilled ${h}_{2}$o into module-1", "feeding distilled h2o into module-1"),
        ("feeding distilled $h_{2}$o into module-1", "feeding distilled h2o into module-1"),
        ("feeding distilled $h2o$ into module-1", "feeding distilled h2o into module-1"),
        # pandoc-escaped brackets (Footnote case)
        (
            "The lab did not detect contaminants.\\[Footnote: trace amounts present\\]",
            "the lab did not detect contaminants. [footnote: trace amounts present]",
        ),
        # markdown bold in source
        ("verified **3.6 g** of sample mass", "verified 3.6 g of sample mass"),
        # markdown italic in source
        ("verified *3.6 g* of sample mass", "verified 3.6 g of sample mass"),
        # pandoc table separator row (annex case)
        (
            "ANNEX 1\n\n----- -----\nlocation status\n----- -----\nSite-A active",
            "annex 1 location status site-a active",
        ),
    ],
)
def test_category_a_normalization_strips_artifacts(source: str, agent_quote: str) -> None:
    assert _normalize_text(agent_quote) in _normalize_text(source)


# ---------------------------------------------------------------------------
# Category B — agent inserts bracketed paraphrase markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, agent_blockquote",
    [
        # [and] insertion
        (
            "Alpha, Bravo, and Charlie were unregistered",
            "alpha, [and] bravo, and charlie were unregistered",
        ),
        # [.*] elision marker
        (
            "intermittently feeding sample into module-1, module-2m, module-4. On 15 February 2024, batch 5",
            "intermittently feeding sample into module-1, module-2m, module-4 [.*] on 15 february 2024, batch 5",
        ),
        # [...] three-dot bracketed elision
        (
            "On 25 June 2024, the lab introduced sample into the first module and on 19 August 2024 began testing",
            "on 25 june 2024, the lab introduced sample [...] on 19 august 2024 began testing",
        ),
        # [note: ...] gloss
        (
            "The supplier agreed on 14 July 2024 with consortium A",
            "the supplier agreed on 14 july 2024 [note: with consortium B]",
        ),
    ],
)
def test_category_b_bracketed_paraphrase_markers_become_wildcards(source: str, agent_blockquote: str) -> None:
    """After extraction, bracketed elision markers become `.*` and the regex matches the source."""
    output = f"The report says:\n> {agent_blockquote}\n(File: foo.txt, Para: 1)"
    quotes = _extract_markdown_quotes(output)
    assert len(quotes) == 1
    quote, _ = quotes[0]
    assert ".*" in quote
    import re

    pattern = re.escape(quote).replace(r"\.\*", ".*")
    assert re.search(pattern, _normalize_text(source)) is not None


# ---------------------------------------------------------------------------
# Category C — inline attribution boilerplate leaks into quote text
# ---------------------------------------------------------------------------


def test_category_c_inline_referenced_file_marker_is_stripped() -> None:
    output = (
        "> the inspector verified that the lab was feeding (Referenced file: REPORT/2024/12) up to 1044 module-1 units"
    )
    quotes = _extract_markdown_quotes(output)
    quote, _ = quotes[0]
    assert "referenced file" not in quote
    assert "report/2024/12" not in quote


def test_category_c_inline_file_marker_is_stripped() -> None:
    output = "> the verification step (File: foo.txt) completed successfully"
    quotes = _extract_markdown_quotes(output)
    quote, _ = quotes[0]
    assert "file:" not in quote
    assert "foo.txt" not in quote


# ---------------------------------------------------------------------------
# Category D — zero retrieved sources
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_category_d_distinguishes_zero_sources_from_quote_mismatch() -> None:
    """When no documents at all were retrieved, the error must not blame the quote text."""
    evaluator = LiteralQuoteEvaluator()
    output = "> Agreement-7 was finalized on 14 July 2024\n(File: foo, Para: 1)"
    with patch.object(evaluator, "get_documents", return_value=[]):
        result = await evaluator.run(_ctx(output))
    assert result.value is False
    reason_lower = (result.reason or "").lower()
    assert (
        "no retriever" in reason_lower
        or "no documents" in reason_lower
        or "skipped retrieval" in reason_lower
        or "no sources" in reason_lower
    ), f"Unexpected reason: {result.reason!r}"


# ---------------------------------------------------------------------------
# Category G — stray quote chars survive cleanup
# ---------------------------------------------------------------------------


def test_category_g_stray_leading_quote_is_trimmed() -> None:
    output = ">'On 22 February 2025, the inspector verified at Site-A'\n(File: foo, Para: 1)"
    quotes = _extract_markdown_quotes(output)
    quote, _ = quotes[0]
    assert not quote.startswith("'")
    assert not quote.endswith("'")


# ---------------------------------------------------------------------------
# Category I — dash and space variants
#
# Intentionally uses visually-ambiguous Unicode (en/em/figure dashes, NBSP,
# soft hyphen, zero-width space). RUF001 noqa applied at file scope above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, agent_quote",
    [
        ("2–3 October 2024", "2-3 october 2024"),  # en-dash U+2013
        ("2—3 October 2024", "2-3 october 2024"),  # em-dash U+2014
        ("2‐3 October 2024", "2-3 october 2024"),  # hyphen U+2010
        ("2‒3 October 2024", "2-3 october 2024"),  # figure dash U+2012
        ("Acme research program", "acme research program"),  # NBSP -> space
        ("inter­national", "international"),  # soft hyphen dropped
        ("zero​width", "zerowidth"),  # zero-width space dropped
    ],
)
def test_category_i_dash_and_space_variants_normalize(source: str, agent_quote: str) -> None:
    assert _normalize_text(agent_quote) in _normalize_text(source)


# ---------------------------------------------------------------------------
# Category H — trailing-punctuation widening is intentionally NOT done here.
#
# Widening beyond `.` would break regex evaluators that flow patterns through
# ``_normalize_text`` (e.g. a pattern ending in ``\?`` would lose its ``?``).
# This is documented as a future split: a separate normalization function for
# literal-text contexts could safely strip ``,;:!?``.
# ---------------------------------------------------------------------------


def test_category_h_trailing_period_still_stripped() -> None:
    """Regression: trailing period stripping must keep working."""
    assert not _normalize_text("ends with a period.").endswith(".")


# ---------------------------------------------------------------------------
# Deduplication — identical quotes collapse before reporting
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_duplicate_quotes_appear_once_in_failure_message() -> None:
    """When the agent emits the same quote twice and neither is found, report it once."""
    evaluator = LiteralQuoteEvaluator()
    output = """
> The same paragraph appears twice in the answer.
(File: foo, Para: 1)

> The same paragraph appears twice in the answer.
(File: foo, Para: 2)
"""
    docs = [Document(page_content="totally unrelated content", metadata={"source": "x"})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        result = await evaluator.run(_ctx(output))
    assert result.value is False
    # The quote text should appear exactly once in the reason — count occurrences.
    assert (result.reason or "").lower().count("the same paragraph appears twice") == 1


# ---------------------------------------------------------------------------
# Regression: synthetic end-to-end inputs reproducing the documented cases now pass
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_real_failure_pandoc_footnote_now_passes() -> None:
    """End-to-end: Footnote case with pandoc-escaped brackets in the source."""
    evaluator = LiteralQuoteEvaluator()
    docs = [
        Document(
            page_content=(
                "The lab did not detect contaminants.\\[Footnote: trace amounts of organics "
                "were detected.\\] Further investigation is required."
            ),
            metadata={"source": "lab-report.txt"},
        )
    ]
    output = (
        "The report explains:\n"
        "> The lab did not detect contaminants. [Footnote: trace amounts of organics "
        "were detected.]\n"
        "(File: lab-report.txt, Para: 4)"
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        result = await evaluator.run(_ctx(output))
    assert result.value is True, result.reason


@pytest.mark.anyio
async def test_real_failure_referenced_file_marker_now_passes() -> None:
    """End-to-end: agent's quote includes inline (Referenced file: …) that the source doesn't have."""
    evaluator = LiteralQuoteEvaluator()
    docs = [
        Document(
            page_content="the inspector verified that the lab was feeding up to 1044 module-1 units at Site-A",
            metadata={"source": "site-a.txt"},
        )
    ]
    output = (
        "Per the report:\n"
        "> the inspector verified that the lab was feeding (Referenced file: REPORT/2024/12) "
        "up to 1044 module-1 units at Site-A\n"
        "(File: site-a.txt, Para: 7)"
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        result = await evaluator.run(_ctx(output))
    assert result.value is True, result.reason


@pytest.mark.anyio
async def test_real_failure_bracketed_elision_now_passes() -> None:
    """End-to-end: agent uses [.*] marker that should become a regex wildcard."""
    evaluator = LiteralQuoteEvaluator()
    docs = [
        Document(
            page_content=(
                "intermittently feeding sample into module-1, module-2m, module-4 and module-6 units. "
                "On 15 February 2024, batch 5 was reconfigured."
            ),
            metadata={"source": "module-2m.txt"},
        )
    ]
    output = (
        "The inspector reports:\n"
        "> intermittently feeding sample into module-1, module-2m, module-4 [.*] on 15 february 2024, batch 5\n"
        "(File: module-2m.txt, Para: 12)"
    )
    with patch.object(evaluator, "get_documents", return_value=docs):
        result = await evaluator.run(_ctx(output))
    assert result.value is True, result.reason
