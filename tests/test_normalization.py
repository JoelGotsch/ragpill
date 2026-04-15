"""Tests for normalization helpers in evaluators."""

from ragpill.utils import _normalize_text


def test_normalize_text_nfkc_and_whitespace():
    # Collapses whitespace and normalizes Unicode (UF₆ -> UF6)
    assert _normalize_text("  UF₆\n  sample\t") == "uf6 sample"
    assert _normalize_text("Another     document \n\n\t with different") == "another document with different"


def test_normalize_text_preserves_plain_text():
    assert _normalize_text("Simple text") == "simple text"


def test_normalize_real():
    input = "51. No new information was provided by Iran with respect to the issue of\n\n    testing of centrifuges using nuclear material until October 2003. In\n\n    its letter of 21 October 2003, Iran acknowledged that, in order to\n\n    ensure the performance of centrifuge machines, a limited number of\n\n    tests using small amounts of UF~6~ imported in 1991 had been carried\n\n    out at the Kalaye Electric Company. According to Iran, the first\n\n    test of the centrifuges was conducted in 1998 using an inert gas\n\n    (xenon). Series of tests using UF~6~ were performed between 1999\n\n    and 2002. In the course of the last series of tests, an enrichment\n\n    level of 1.2% U-235 was achieved."
    # Note: trailing period is stripped by normalization
    expected = "51. no new information was provided by iran with respect to the issue of testing of centrifuges using nuclear material until october 2003. in its letter of 21 october 2003, iran acknowledged that, in order to ensure the performance of centrifuge machines, a limited number of tests using small amounts of uf6 imported in 1991 had been carried out at the kalaye electric company. according to iran, the first test of the centrifuges was conducted in 1998 using an inert gas (xenon). series of tests using uf6 were performed between 1999 and 2002. in the course of the last series of tests, an enrichment level of 1.2% u-235 was achieved"
    assert _normalize_text(input) == expected


def test_normalize_ellipsis_and_midsection():
    # its up to the regex/pattern parsing to substitute ellipsis with .* for matching, but
    # normalization should not remove or alter the ellipsis character itself,
    # as it may be used in patterns.
    input = "...This quote has ellipsis... and mid...section..."
    expected = "this quote has ellipsis... and mid...section"
    assert _normalize_text(input) == expected


def test_normalize_text_strips_single_tilde_subscripts_up_to_10_chars():
    assert _normalize_text("UF~6~") == "uf6"
    assert _normalize_text("UF~1234567890~") == "uf1234567890"


def test_normalize_quotes():
    assert _normalize_text('He said, "This is a quote."') == _normalize_text("he said, 'this is a quote.'")
    assert _normalize_text("syria\u2019s past nuclear activities") == _normalize_text(
        "Syria\u2019s past nuclear\n\n    activities."
    )


def test_normalize_text_leaves_long_tilde_segments():
    assert _normalize_text("UF~12345678901~") == "uf~12345678901~"


def test_normalize_text_adds_case_insensitive_and_nfkc():
    normalized = _normalize_text("UF₆ status")
    assert normalized == "uf6 status"


def test_normalize_text_preserves_existing_case_flag():
    normalized = _normalize_text("Error|Failure")
    assert normalized == "error|failure"


def test_normalize_text_trims_and_collapses_spaces():
    normalized = _normalize_text("   foo   bar   ")
    assert normalized == "foo bar"


def test_normalize_text_starts_with_multiline_only():
    normalized = _normalize_text("(?m)^start")
    assert normalized == "(?m)^start"


def test_normalize_text_with_existing_si_flags():
    normalized = _normalize_text("(?si)Already")
    assert normalized == "(?si)already"


def test_normalize_text_with_literal_parentheses():
    normalized = _normalize_text("This (is a test)")
    assert normalized == "this (is a test)"
