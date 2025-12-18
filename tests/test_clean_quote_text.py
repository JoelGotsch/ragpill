"""Unit tests for `_clean_quote_text` helper."""

import pytest

from ragpill.utils import _clean_quote_text


@pytest.mark.parametrize(
    "text, expected_text, expected_quote",
    [
        ("\"hello\"", "hello", None),
        ("'hello world'", "hello world", None),
    ],
)
def test_clean_quote_text_basic(text, expected_text, expected_quote):
    cleaned, quote_char = _clean_quote_text(text)

    assert cleaned == expected_text
    assert quote_char == expected_quote


def test_clean_quote_text_unmatched_closing():
    cleaned, quote_char = _clean_quote_text("'leading only")

    assert cleaned == "leading only"
    assert quote_char == "'"

def test_deep_mixed_quotes():
    cleaned, quote_char = _clean_quote_text("\"quotes\" and 'another \"quote within\" one'")

    assert cleaned == '"quotes" and "another \'quote within\' one"'
    assert quote_char == '"'

