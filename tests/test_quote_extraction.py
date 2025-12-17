"""Regression tests for ``extract_markdown_quotes`` (quote_extraction_regression.md).

Locks the pre-0.4.2 extraction contract: the extractor produces the
agent's quote text verbatim (lean normalization only), with elision
markers converted to ``.*`` placeholders. The aggressive normalization
that was briefly merged into ``_normalize_text`` in 0.4.2 — citation
stripping, markdown emphasis stripping, dash folding — now lives in
:func:`ragpill._text.normalize_for_quote_comparison` and is applied
**only inside** :class:`~ragpill.evaluators.LiteralQuoteEvaluator.run`.
"""

from __future__ import annotations

from ragpill._text import extract_markdown_quotes

# ---------------------------------------------------------------------------
# Case 1 — bare ellipsis preserves surrounding whitespace
# ---------------------------------------------------------------------------


def test_collapse_multiple_dots_keeps_space_after_wildcard():
    """``ellipsis... and`` -> ``ellipsis.* and`` — the space before ``and`` survives."""
    output = "> This quote has ellipsis... and mid...section..."
    quote, _ = extract_markdown_quotes(output)[0]
    assert quote == "this quote has ellipsis.* and mid.*section"


# ---------------------------------------------------------------------------
# Case 2 — internal single quotes stay (only balanced outer quotes are stripped)
# ---------------------------------------------------------------------------


def test_extract_preserves_internal_single_quotes():
    """The outer ``"..."`` is stripped; the inner ``'...'`` pairs that are
    *content* are not."""
    output = "> \"'no longer outstanding at this stage' does not mean 'resolved'.\""
    quote, _ = extract_markdown_quotes(output)[0]
    assert quote == "'no longer outstanding at this stage' does not mean 'resolved'"


# ---------------------------------------------------------------------------
# Case 3 — nested-quote ``(source: …)`` markers survive extraction
# ---------------------------------------------------------------------------


def test_nested_quote_source_markers_survive_extraction():
    """When a blockquote contains a nested subquote, ``_extract_quotes``
    collapses the subquote into a single-line representation and appends
    a ``(source: …)`` marker so downstream readers can identify the
    nested origin. The extractor must not silently strip those markers."""
    output = (
        "> outer text\n"
        "> > nested quote text\n"
        "> > (File: [nested.txt](https://example.test/nested))\n"
        "> trailing outer text\n"
        "(File: [outer.txt](https://example.test/outer))\n"
    )
    quote, source = extract_markdown_quotes(output)[0]
    assert source == "outer.txt"
    # The nested subquote's source marker is preserved inline in the
    # collapsed-quote text.
    assert "source: nested.txt" in quote


# ---------------------------------------------------------------------------
# Case 4 — markdown emphasis markers stay (no silent stripping)
# ---------------------------------------------------------------------------


def test_extract_preserves_markdown_emphasis():
    """``**bold**`` / ``*italic*`` are content, not paraphrase markers —
    the extractor leaves them verbatim. Comparison-time normalization
    inside ``LiteralQuoteEvaluator.run`` handles the asymmetry vs source
    documents that lack the formatting."""
    output = "> This quote has **bold** and *italic* text\n> and also `code` formatting"
    quote, _ = extract_markdown_quotes(output)[0]
    assert "**bold**" in quote
    assert "*italic*" in quote


# ---------------------------------------------------------------------------
# Case 5 — markdown link text is content, not a paraphrase marker
# ---------------------------------------------------------------------------


def test_extract_preserves_markdown_link_text():
    """``[link text](url)`` is *not* substituted to ``.*``. The link text
    is the human-meaningful content; the URL stays in parens. The
    bracketed-gloss form of ``_AGENT_ELISION_RE`` excludes any ``[...]``
    followed immediately by ``(`` to leave links alone."""
    output = "> Check out [this link](https://example.com) for details\n> And also [another link](https://test.org)"
    quote, _ = extract_markdown_quotes(output)[0]
    assert "this link" in quote
    assert "another link" in quote
    # The links themselves stay intact.
    assert "(https://example.com)" in quote
    assert "(https://test.org)" in quote
    # No spurious ``.*`` injected from the bracketed text.
    assert ".*" not in quote


def test_sibling_subquote_after_multiline_nested_is_not_corrupted():
    """Regression (round-2 F4): a sibling nested subquote following a multiline
    nested one must not have its indices shifted by the earlier splice."""
    output = "\n".join(
        [
            "> outer start",
            ">> first nested line1",
            ">> first nested line2",
            "> middle text",
            ">> second nested",
            "> outer end",
        ]
    )
    quotes = extract_markdown_quotes(output)
    assert len(quotes) == 1
    text, _ref = quotes[0]
    # The raw '> second nested' line must not survive, its cleaned form must not
    # be duplicated, and 'outer end' must not be dropped.
    assert "> second nested" not in text
    assert text.count("second nested") == 1
    assert "outer end" in text
    assert "outer start" in text and "middle text" in text
