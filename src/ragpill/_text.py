"""Text and markdown-quote normalization / extraction helpers.

Split out of ``utils.py`` so the quote-parsing and normalization machinery lives
in one cohesive, stdlib-only module. The public functions
(:func:`normalize_text`, :func:`normalize_for_quote_comparison`,
:func:`extract_markdown_quotes`, :func:`to_text`) are consumed by the
evaluators, backends, and report layer; the rest are internal helpers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass


def to_text(value: object) -> str:
    """Best-effort single string form of an arbitrary value.

    The one canonical value→text helper (strings pass through; everything
    else is JSON with sorted keys, ``str`` fallback for non-serializable
    members, and non-ASCII preserved). Backends use it for span I/O
    attributes and the report layer builds :func:`render_value` on it, so
    the same value reads identically in traces and reports.

    Args:
        value: Any value.

    Returns:
        The string itself, its JSON encoding, or ``str(value)`` when the
        value cannot be JSON-encoded at all.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _clean_quote_text(text: str, quote_char: str | None = None) -> tuple[str, str | None]:
    """
    Recursively clean quote text by detecting quote characters and ensuring proper nesting and alternation.
    Returns cleaned text and the detected quote char of the outermost level if any.
    If quote_char is provided, this function ensures that this quote_char is used for this level
    and replaces any other quote chars with the provided one, while alternating quote chars for nested levels.
    """
    QUOTES = ('"', "'")

    def alternate_quote(q: str) -> str:
        return "'" if q == '"' else '"'

    def find_matching_quote(s: str, start: int, q: str) -> int:
        """Find the index of the matching closing quote, or -1 if not found."""
        i = start + 1
        while i < len(s):
            if s[i] == q:
                return i
            i += 1
        return -1

    # Strip whitespace first
    text = text.strip()

    # Check if entire text is wrapped in matching quotes - if so, strip them
    if len(text) >= 2 and text[0] in QUOTES and text[-1] == text[0]:
        # Check if these are truly the outer quotes (not just coincidental)
        outer_quote = text[0]
        match_idx = find_matching_quote(text, 0, outer_quote)
        if match_idx == len(text) - 1:
            # Strip outer quotes and recurse
            inner_text, inner_quote_char = _clean_quote_text(text[1:-1], quote_char)
            return inner_text.strip(), inner_quote_char

    # Check for unmatched leading quote
    if len(text) >= 1 and text[0] in QUOTES:
        leading_quote = text[0]
        match_idx = find_matching_quote(text, 0, leading_quote)
        if match_idx == -1:
            # Unmatched leading quote - strip it
            inner_text, _ = _clean_quote_text(text[1:], quote_char or leading_quote)
            return inner_text.strip(), quote_char or leading_quote

    def normalize_quotes_to(out_quote: str) -> str:
        """Rewrite every matched quoted span to ``out_quote``, alternating the
        quote character for nested levels (recursively cleaned)."""
        alt = alternate_quote(out_quote)
        result: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in QUOTES:
                match_idx = find_matching_quote(text, i, ch)
                if match_idx != -1:
                    inner_content = text[i + 1 : match_idx]
                    cleaned_inner, _ = _clean_quote_text(inner_content, alt)
                    result.append(out_quote)
                    result.append(cleaned_inner)
                    result.append(out_quote)
                    i = match_idx + 1
                    continue
            # Not a quote, or an unmatched quote: keep the character verbatim.
            result.append(ch)
            i += 1
        return "".join(result)

    # Normalize to the provided quote char if given, else to the first quote
    # character found in the text; if there are no quotes, return unchanged.
    if quote_char is not None:
        return normalize_quotes_to(quote_char), quote_char
    detected_quote = next((c for c in text if c in QUOTES), None)
    if detected_quote is not None:
        return normalize_quotes_to(detected_quote), detected_quote
    return text, None


def _get_source(line: str) -> str | None:
    """Extract source reference from a line if it exists."""
    file_match = re.search(r"(?i)\(?(?:file|source):\s*\[([^\]]+)\]\)?", line)
    if file_match:
        return file_match.group(1)
    return None


_QUOTE_LINE_RE = re.compile(r"^(\s*)>(.*)$")
_QUOTE_CHARS = ('"', "'")


@dataclass
class _QuoteBlock:
    """One markdown blockquote, as a tree node.

    ``items`` is the block's content in document order: plain text lines
    (``str``) interleaved with nested :class:`_QuoteBlock` children. ``source``
    is the ``(source: …)`` / ``(file: …)`` reference attached to this block, if
    any. Built in a single pass by :func:`_parse_blocks` and flattened to a
    quote string by :func:`_render_block`.
    """

    items: list[str | _QuoteBlock]
    source: str | None = None


def _parse_blocks(lines: list[str]) -> list[str | _QuoteBlock]:
    """Parse ``lines`` into document-order plain lines and blockquote trees.

    A block is a maximal run of consecutive ``>``-prefixed lines sharing the
    same leading indent. Each line contributes its content (one ``>`` peeled
    off, whitespace stripped); the collected content is parsed recursively, so a
    content line that itself begins with ``>`` becomes a nested child *in place*.

    This tree replaces the old index-splicing recursion, which recorded
    subquote positions against a list it then mutated — a multiline nested quote
    followed by a sibling shifted the sibling's indices and corrupted it
    (round-2 F4). Building children in place makes that class of bug
    unrepresentable.
    """
    items: list[str | _QuoteBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        match = _QUOTE_LINE_RE.match(lines[i])
        if match is None:
            items.append(lines[i])
            i += 1
            continue

        indent = len(match.group(1))
        content: list[str] = []
        tail_text = ""  # peeled content of the run's last raw line ("" if blank)
        while i < n:
            inner = _QUOTE_LINE_RE.match(lines[i])
            if inner is None or len(inner.group(1)) != indent:
                break
            tail_text = inner.group(2).strip()
            if tail_text:
                content.append(tail_text)
            i += 1

        # A run of empty ``>`` markers with no content is not a quote.
        if not content:
            continue

        # Source: the line immediately after the block, else a trailing
        # ``(source: …)`` on the block's own last line (which is then dropped
        # from the quote text). The trailing line only counts when it sits at
        # *this* block's nesting level: a source ref on a ``>``-prefixed
        # trailing line belongs to the nested block and is claimed by the
        # recursive parse below (round-3 R5 — checking the raw content
        # re-attributed nested sources to the parent). A trailing blank ``>``
        # marker likewise leaves the ref inline rather than promoting it.
        source = _get_source(lines[i]) if i < n else None
        if (
            source is None
            and tail_text
            and _QUOTE_LINE_RE.match(tail_text) is None
            and (tail := _get_source(tail_text)) is not None
        ):
            source = tail
            # Drop the claimed line before recursing so a preceding nested
            # block cannot also claim it via its after-block lookup.
            content = content[:-1]

        items.append(_QuoteBlock(items=_parse_blocks(content), source=source))
    return items


def _render_block(block: _QuoteBlock, depth: int) -> str:
    """Flatten a block to a single quote string.

    Plain lines join with spaces; each nested child collapses to
    ``{q}{child}{q} (source: …)``. The wrapping quote char ``q`` alternates with
    ``depth`` and is chosen to differ from the child's own outermost quote so
    the two levels stay visually distinct.
    """
    parts: list[str] = []
    for item in block.items:
        if isinstance(item, str):
            parts.append(item)
            continue
        child_text = _render_block(item, depth + 1)
        child_text, inner_quote_char = _clean_quote_text(child_text)
        candidates = [qc for qc in _QUOTE_CHARS if qc != inner_quote_char]
        quote_char = candidates[depth % len(candidates)]
        src_str = f" (source: {item.source})" if item.source else ""
        parts.append(f"{quote_char}{child_text}{quote_char}{src_str}")
    return " ".join(parts)


def _strip_balanced_outer(text: str) -> str:
    """Strip one pair of balanced outer quote characters from a top-level block.

    Unlike :func:`_clean_quote_text` (used for nested quotes) this does not
    rewrite interior quote characters — an outer ``"…"`` is peeled but interior
    ``'…'`` content pairs are left verbatim.
    """
    stripped = text.strip()
    for quote_char in _QUOTE_CHARS:
        if len(stripped) >= 2 and stripped.startswith(quote_char) and stripped.endswith(quote_char):
            return stripped[1:-1]
    return text


# Compiled once at module load. Applied symmetrically to both the agent's
# quote and the source ``page_content`` so the canonical forms match.
_SUBSCRIPT_TILDE_RE = re.compile(r"~([0-9A-Za-z]{1,10})~")
_SUPERSCRIPT_CARET_RE = re.compile(r"\^([0-9A-Za-z]{1,10})\^")
# LaTeX subscript/superscript braces: ``_{6}``, ``^{2}``.
_LATEX_BRACE_RE = re.compile(r"[_^]\{([0-9A-Za-z]{1,10})\}")
# Compact LaTeX braces around identifiers: ``${uf}``. Restricted to
# letter-first identifiers so regex quantifiers like ``\d{3}`` (which
# also flow through ``normalize_text`` in the regex evaluators) keep
# their meaning.
_LATEX_IDENT_BRACE_RE = re.compile(r"\{([A-Za-z][0-9A-Za-z]{0,9})\}")
# LaTeX math wrapper: keep the inner text, drop the delimiters.
_LATEX_MATH_RE = re.compile(r"\$([^$]{1,80})\$")
# Pandoc-escaped meta chars: ``\[ \] \_ \* \( \) \# \- \\``.
_PANDOC_ESCAPE_RE = re.compile(r"\\([\[\]_\*\(\)\#\-\\])")
# Markdown emphasis: ``**bold**``, ``*italic*``, ``__bold__``, ``_italic_``.
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1|(\*|_)(.+?)\3")
# Dashed table separator rows.
_TABLE_SEP_RE = re.compile(r"^[\s\-\|\=\+\:]+$", re.MULTILINE)
# Inline attribution markers that LLMs sometimes inline into quote text.
_INLINE_CITATION_RE = re.compile(
    r"\((?:referenced\s+file|file|source|skill|para(?:graph)?)\s*:[^)]*\)",
    re.IGNORECASE,
)
# Trailing punctuation tolerated at the very end of a quote. Kept narrow
# (period + whitespace) to avoid mangling regex patterns ending in escaped
# meta chars such as ``\?`` or ``\!`` that flow through this function via
# the regex evaluators' ``from_csv_line``.
_TRAIL_PUNCT = ". "

# Dash and zero-width / NBSP variants that NFKC does not collapse.
_DASH_TRANSLATIONS: dict[int, int | None] = {
    0x2010: 0x2D,  # hyphen
    0x2011: 0x2D,  # non-breaking hyphen
    0x2012: 0x2D,  # figure dash
    0x2013: 0x2D,  # en dash
    0x2014: 0x2D,  # em dash
    0x2212: 0x2D,  # minus sign
    0x00AD: None,  # soft hyphen — delete
    0x200B: None,  # zero-width space — delete
    0x00A0: 0x20,  # non-breaking space — space
}


def normalize_text(text: str) -> str:
    """Lean normalization: NFKC + casefold + whitespace + quote-char map.

    Used everywhere ragpill compares strings — regex evaluators, DataFrame
    text columns, etc. Intentionally conservative: does NOT strip markdown
    emphasis, citation markers, LaTeX wrappers, or dash variants, because
    those are sometimes legitimate content (nested-quote source markers,
    content with ``**bold**`` that should survive in the runs table, etc.).

    The richer comparison-only normalization used by
    :class:`~ragpill.evaluators.LiteralQuoteEvaluator` lives in
    :func:`normalize_for_quote_comparison`.

    Aligns visually similar but byte-different text: ``UF₆`` vs ``UF6``
    (via NFKC), curly vs straight quotes, collapsed whitespace, trailing
    periods.
    """
    # Strip single-tilde markdown subscripts (e.g., UF~6~ -> UF6).
    text = _SUBSCRIPT_TILDE_RE.sub(r"\1", text)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Normalize all quote-like characters to straight single quote:
    # - Straight double quote: " (U+0022)
    # - Curly double quotes: " " (U+201C, U+201D)
    # - Curly single quotes/apostrophes: ' ' (U+2018, U+2019)
    # - Low quotes: „ ‚ (U+201E, U+201A)  # noqa: RUF003
    # - Guillemets: « » ‹ › (U+00AB, U+00BB, U+2039, U+203A)  # noqa: RUF003
    # - Prime symbols: ′ ″ (U+2032, U+2033)  # noqa: RUF003
    # - Grave/acute accents: ` ´ (U+0060, U+00B4)  # noqa: RUF003
    normalized = re.sub(
        r'["\u201C\u201D\u201E\'\u2018\u2019\u201A\u00AB\u00BB\u2039\u203A\u2032\u2033`\u00B4]', "'", normalized
    )
    return normalized.strip(".")


def normalize_for_quote_comparison(text: str) -> str:
    """Aggressive normalization used only for LiteralQuoteEvaluator matching.

    Applies the full dialect-stripping pipeline (LaTeX math/braces,
    pandoc-escaped meta chars, markdown emphasis, table separators,
    inline ``(Referenced file: \u2026)`` markers, dash / soft-hyphen / NBSP
    variants, bracket-padding normalization) on top of the lean
    :func:`normalize_text`.

    Symmetric: applying it to both the agent's quote and the source
    document yields canonical forms that should match for content
    differing only in markdown formatting or citation noise. The regex
    evaluators and DataFrame text columns keep the lean
    :func:`normalize_text` so legitimate content like ``**bold**``
    and nested-quote ``(source: \u2026)`` markers survive in the runs view.
    """
    text = _SUBSCRIPT_TILDE_RE.sub(r"\1", text)
    text = _SUPERSCRIPT_CARET_RE.sub(r"\1", text)
    text = _LATEX_BRACE_RE.sub(r"\1", text)
    text = _LATEX_MATH_RE.sub(r"\1", text)
    text = _LATEX_IDENT_BRACE_RE.sub(r"\1", text)
    text = _PANDOC_ESCAPE_RE.sub(r"\1", text)
    text = _MD_EMPHASIS_RE.sub(lambda m: m.group(2) or m.group(4) or "", text)
    text = _TABLE_SEP_RE.sub("", text)
    text = _INLINE_CITATION_RE.sub("", text)
    text = text.translate(_DASH_TRANSLATIONS)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Collapse whitespace adjacent to brackets so ``X[Y]`` and ``X [Y]``
    # canonicalize the same way \u2014 pandoc-escaped ``\[`` (which loses any
    # surrounding space when unescaped) lines up with the agent's ``[``.
    normalized = re.sub(r"\s*\[\s*", " [", normalized)
    normalized = re.sub(r"\s*\]\s*", "] ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(
        r'["\u201C\u201D\u201E\'\u2018\u2019\u201A\u00AB\u00BB\u2039\u203A\u2032\u2033`\u00B4]', "'", normalized
    )
    return normalized.strip(_TRAIL_PUNCT)


# Bracketed elision / gloss markers an agent inserts inside a blockquote.
# Cases handled (each substituted with `.*` by the matcher):
#   "..."           ellipsis           -> .*  (no whitespace absorption)
#   ".."            two dots           -> .*
#   "[...]"         bracketed ellipsis -> .*  (absorbs surrounding whitespace)
#   "[..]"          bracketed two dots -> .*
#   "[.*]"          author elision     -> .*
#   "[ ... ]"       padded             -> .*
#   "[and]" "[note: ...]" "[edited]"  -> .*  (bracketed gloss starting with a
#                                              letter, ≤80 chars, NOT followed
#                                              by ``(`` so markdown links like
#                                              ``[link text](url)`` survive)
_AGENT_ELISION_RE = re.compile(
    r"""
    \s* \[[\s\.\*]{1,5}\] \s*                  # bracketed dots / asterisks
    |
    \s* \[[A-Za-z][^\[\]]{0,79}\] (?!\() \s*   # bracketed gloss, not a markdown link
    |
    \.{2,}                                     # bare ellipsis — preserves whitespace
    """,
    re.VERBOSE,
)


def extract_markdown_quotes(output: str) -> list[tuple[str, str | None]]:
    """Extract and normalize markdown quotes and their file references from output.

    Only lines that start with '>' (after leading whitespace) are considered
    markdown quotes. Regular quoted text is ignored. Quotation marks are
    stripped by :func:`_clean_quote_text`. Quote text is normalized with
    the lean :func:`normalize_text` (NFKC + casefold + whitespace +
    quote-char map + trailing-period strip) — content like ``**bold**``,
    ``[link text](url)``, nested-quote ``(source: …)`` markers, and
    inline citations all survive verbatim so the runs DataFrame and
    triage view see the agent's original words.

    Bracketed paraphrase markers (``[and]``, ``[.*]``, ``[...]``,
    ``[note: …]``) and bare ellipsis (``..+``) are converted to a regex
    ``.*`` placeholder so the matcher in
    :class:`~ragpill.evaluators.LiteralQuoteEvaluator` can accept the
    elision. Markdown link text ``[link text](url)`` is NOT substituted —
    that's content, not paraphrase.

    Args:
        output: The text to extract quotes from

    Returns:
        List of tuples: (quote_text_without_quotation_marks, referenced_filename_or_none)
    """

    quotes: list[tuple[str, str | None]] = []
    for item in _parse_blocks(output.split("\n")):
        if not isinstance(item, _QuoteBlock):
            continue
        quote_text = _render_block(item, depth=0)
        quote_text = _strip_balanced_outer(quote_text)
        quote_text = normalize_text(quote_text)
        # Convert agent elisions / bare ellipsis to regex ``.*``. Markdown
        # links ``[text](url)`` are excluded by the regex.
        quote_text = _AGENT_ELISION_RE.sub(".*", quote_text)
        quotes.append((quote_text, item.source))
    return quotes
