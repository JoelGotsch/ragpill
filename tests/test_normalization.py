"""Tests for normalization helpers in evaluators."""

from ragpill.utils import _normalize_text


def test_normalize_text_preserves_plain_text():
    assert _normalize_text("Simple text") == "simple text"


def test_normalize_ellipsis_and_midsection():
    # its up to the regex/pattern parsing to substitute ellipsis with .* for matching, but 
    # normalization should not remove or alter the ellipsis character itself,
    # as it may be used in patterns.
    input = "...This quote has ellipsis... and mid...section..."
    expected = "this quote has ellipsis... and mid...section"
    assert _normalize_text(input) == expected



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
