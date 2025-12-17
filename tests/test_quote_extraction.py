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

import pytest

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


# ---------------------------------------------------------------------------
# Round-3 R5 — source attribution happens per block, at that block's own
# nesting level. A ``(source: …)`` ref on a nested (``>``-prefixed) trailing
# line belongs to the nested block; only a same-level trailing line (or the
# after-block plain-text line) may claim the enclosing block. Expected outputs
# below were derived by running the pre-rewrite parser
# (``git show 2ca3707^:src/ragpill/_text.py``) — except ``no-trailing-marker``,
# where the old parser itself mis-attributed the nested ref to the parent (and
# dropped the nested line); there the fixed level-aware behavior is pinned
# instead. Validated against the old parser with a 30k-input fuzz comparison
# (one-off script: old-vs-new ``extract_markdown_quotes`` over random nested
# blockquote shapes); all remaining divergences classified as old-parser bugs
# (F4 splice corruption, R5-style source steals, per-level quote-strip
# mangling).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(
            '> line with "inner" quotes\n>> nested a\n>> nested (file: [n.md])\n>',
            [("line with 'inner' quotes 'nested a' (source: n.md)", None)],
            id="r5-repro-nested-trailing-source-stays-nested",
        ),
        pytest.param(
            "> a\n>> nested a\n>> nested (file: [n.md])",
            [("a 'nested a' (source: n.md)", None)],
            id="no-trailing-marker-nested-source-stays-nested",
        ),
        pytest.param(
            "> a\n>> nested (file: [n.md])\n>> nested b\n> tail",
            [("a 'nested (file:.*) nested b' tail", None)],
            id="nested-source-on-non-last-line-stays-inline",
        ),
        pytest.param(
            "> a\n>> nested x\n> (file: [p.md])",
            [("a 'nested x'", "p.md")],
            id="same-level-trailing-source-claims-parent",
        ),
        pytest.param(
            "> a\n>> n1 (file: [one.md])\n> mid\n>> n2\n>> (file: [two.md])\n> end\n(file: [outer.md])",
            [("a '' (source: one.md) mid 'n2' (source: two.md) end", "outer.md")],
            id="multiple-nested-blocks-each-keep-their-source",
        ),
        pytest.param(
            "> a (file: [x.md])\n>",
            [("a (file:.*)", None)],
            id="trailing-blank-marker-keeps-source-inline",
        ),
    ],
)
def test_nested_source_attribution_stays_at_own_level(output: str, expected: list[tuple[str, str | None]]):
    """Regression (round-3 R5): the tail-source check must not run on a block's
    raw pre-nesting content — a nested quote's trailing source ref was deleted
    from the parent and re-attributed to the parent block."""
    assert extract_markdown_quotes(output) == expected


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
