# Plan: Improve `LiteralQuoteEvaluator`

**Status:** Proposed
**Date:** 2026-05-14
**Tracks:** [github.com/JoelGotsch/ragpill#8](https://github.com/JoelGotsch/ragpill/issues/8)

## Context

In a recent end-to-end evaluation run, `LiteralQuoteEvaluator` failed
14 of 15 cases. Triage showed the majority were **false positives**
caused by normalization gaps, not by the agent fabricating quotes.
Because the evaluator is the citation-hallucination guardrail, false
positives erode its signal — developers stop trusting it and route
around it.

This change strengthens the three pieces involved in the comparison
(`_normalize_text`, `_extract_markdown_quotes`,
`LiteralQuoteEvaluator.run`) so the evaluator stops flagging the nine
recurring failure categories identified in the issue.

## Failure categories (recap from #8)

| Code | Category | Example |
|---|---|---|
| A | Pandoc/LaTeX/markdown artifacts survive normalization | `${uf}_{6}$`, `\[Footnote: …\]`, `**bold**`, dashed table separators |
| B | Agent inserts bracketed paraphrase markers | `[and]`, `[.*]`, `[...]`, `[note: …]` |
| C | Inline attribution boilerplate leaks into quote text | `(Referenced file: GOV/2025/24)`, `(Source: …)` |
| D | Zero retrieval — agent skipped retrieval | empty documents list |
| E | Whitespace edge cases (PDF runs of spaces, hard line wraps) | already covered by `re.sub(r"\s+", " ", …)` |
| F | Genuine disagreement (TRUE positive — leave as failure) | wording differs from any source |
| G | Stray quote chars after cleanup | leading `'` survives |
| H | Trailing punctuation strip is too narrow | only `.` stripped today |
| I | Dash / soft-hyphen / zero-width variants | en/em dashes, U+00AD, U+200B, U+00A0 |

## Approach (single PR, TDD)

### Phase 1 — Add failing tests
New file `tests/test_literal_quote_improvements.py` with parametrized
tests for each category, ported from the issue body. Run; confirm red.

### Phase 2 — Strengthen `_normalize_text` (covers A, C, E, G, H, I)
In `src/ragpill/utils.py`:

- Compile new patterns at module load:
  - `_LATEX_MATH_RE = re.compile(r"\$([^$]{1,80})\$")` — LaTeX `$math$`.
  - `_LATEX_BRACE_RE = re.compile(r"[_^]\{([0-9A-Za-z]{1,10})\}")` —
    `_{6}` / `^{2}` subscript/superscript braces.
  - `_SUPERSCRIPT_CARET_RE = re.compile(r"\^([0-9A-Za-z]{1,10})\^")` —
    markdown superscript twin of the existing tilde subscript.
  - `_PANDOC_ESCAPE_RE = re.compile(r"\\([\[\]_\*\(\)\#\-\\])")` —
    unescape pandoc meta chars.
  - `_MD_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1|(\*|_)(.+?)\3")` —
    strip bold/italic markers, keep inner text.
  - `_TABLE_SEP_RE = re.compile(r"^[\s\-\|\=\+\:]+$", re.MULTILINE)` —
    drop dashed table separator rows.
  - `_INLINE_CITATION_RE = re.compile(r"\((?:referenced\s+file|file|source|skill|para(?:graph)?)\s*:[^)]*\)", re.IGNORECASE)` —
    drop inline `(File: …)`, `(Source: …)`, `(Referenced file: …)`,
    `(Skill: …)`, `(Para[graph]: …)` markers.
- New `_strip_dash_variants(text)` helper translates `U+2010..U+2014`,
  `U+2212` → `-`, drops `U+00AD` (soft hyphen) and `U+200B`
  (zero-width), and maps `U+00A0` (NBSP) → space.
- Widen trailing-punctuation strip from `.` to `.,;:!? `.
- Order matters: pandoc/LaTeX before NFKC (NFKC folds some of them);
  emphasis/citation/table strips before whitespace collapse; dash
  variants before quote-char normalization.

The function stays idempotent and symmetric — applied to both quotes
and document content so they end up in the same canonical form.

### Phase 3 — Improve `_extract_markdown_quotes` (covers B, G)
- `_AGENT_ELISION_RE` catches `..+`, `[..]`/`[...]`/`[.*]`/`[ . ]`, and
  any bracketed gloss up to 80 chars (e.g. `[and]`, `[note: …]`).
  Convert to `.*` regex placeholder.
- Replace the current `re.sub(r"\.{2,}", ".*", quote_text)` with the
  new combined regex.
- Add a defensive trim of leading/trailing stray quote chars
  (`' " \``) and trailing soft punctuation after `_normalize_text`.

### Phase 4 — `LiteralQuoteEvaluator.run` (covers D, plus dedup; diagnostics if cheap)
- **D** — when `documents` is empty AND quotes exist, return a
  distinct reason that identifies retrieval-skipped rather than
  blaming the quote text.
- **Dedup** — collapse identical normalized quotes before iterating so
  the same "not found" doesn't appear twice in the failure message.
- **Diagnostics (only if implementable in < 50 LOC total)** — when a
  quote is not found, append a short preview of the closest matching
  substring from any document, using
  `difflib.SequenceMatcher.get_matching_blocks` (stdlib, no new dep).
  Cap the preview at ~80 chars. If a clean implementation creeps past
  50 LOC, drop the hint and ship just the dedup + zero-source
  distinction.

### Phase 5 — Verify
- `uv run pytest tests/test_literal_quote_improvements.py -v` — all
  new tests pass.
- `uv run pytest tests/test_literal_quotes.py -v` — existing tests
  stay green.
- `uv run pytest` — full suite (402 + new) passes.
- `uv run ruff check src tests` + `format --check` + `basedpyright`
  — clean.

## Critical files to modify

| File | Change |
|---|---|
| `src/ragpill/utils.py` | Strengthen `_normalize_text`, extend `_extract_markdown_quotes` with the new elision regex, add `_strip_dash_variants` + closest-window helper |
| `src/ragpill/evaluators.py` (`LiteralQuoteEvaluator`) | Zero-source distinction, dedup, closest-window hint, optional `fuzzy_threshold` |
| `tests/test_literal_quote_improvements.py` (new) | Phase-1 failing tests covering categories A, B, C, D, G, I and dedup |
| `pyproject.toml` | Version bump 0.4.1 → 0.4.2 (additive, no breaking API) |
| `docs/api/evaluators.md` | Auto-picks up the new constructor arg via mkdocstrings — no manual edit needed unless the docstring is updated for new behaviour |

## Existing code to reuse (not re-implement)

- `_clean_quote_text` (`utils.py:27`) — already handles nested quote
  stripping. New defensive trim is post-pass only.
- `_extract_quotes` (`utils.py:139`) — already handles blockquote
  collection; no change.
- `re.search(re.escape(quote).replace(r'\.\*', '.*'))` (`evaluators.py:645`) —
  existing regex shape works; we just feed it more `.*` placeholders.
- `difflib.SequenceMatcher` — stdlib; no new dep.

## Verification — end-to-end smoke

```python
from ragpill.evaluators import LiteralQuoteEvaluator
from ragpill.eval_types import EvaluatorContext
# Build ctx with documents=[...] containing pandoc-escaped Footnote.
# Build output containing > "quote with [and] elision".
result = await LiteralQuoteEvaluator().run(ctx)
assert result.value is True
```

## Out of scope (deferred)

- **Fuzzy / threshold-based matching.** Belongs in a *separate*
  evaluator (e.g. `FuzzyQuoteEvaluator`), not as a knob on
  `LiteralQuoteEvaluator`. Mixing strict and fuzzy under one class
  confuses the semantics — "literal" should mean literal. A future PR
  can add the fuzzy variant with its own constructor, tests, and
  docs; this PR keeps `LiteralQuoteEvaluator` strict.
- Replacing substring with embedding-similarity matching. Same
  reasoning — separate evaluator territory.
- Cleaning the upstream corpus. Done elsewhere
  (`rag-mcp-example`); the evaluator should be robust on its own.

## ADR

This is the next sequential ADR in date order (after the 0010 from
the previous PR), per `plans/adr-system.md`. Worth capturing as
Medium: the change widens the canonical comparison form materially
and downstream callers who depend on `_normalize_text` semantics
need to know the new behaviour.
