"""Unit tests for `_clean_quote_text` helper."""

import pytest

from ragpill.utils import _clean_quote_text


@pytest.mark.parametrize(
    "text, expected_text, expected_quote",
    [
        ('"hello"', "hello", None),
        ("'hello world'", "hello world", None),
        ('  " spaced content "  ', "spaced content", None),
        ("no quotes here", "no quotes here", None),
        ("\"outer 'inner' quote\"", "outer 'inner' quote", "'"),
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


def test_clean_quote_text_unmatched_trailing():
    cleaned, quote_char = _clean_quote_text("unmatched'")

    assert cleaned == "unmatched'"
    assert quote_char == "'"


def test_quotes_in_the_middle():
    cleaned, quote_char = _clean_quote_text("This is a 'quote' in the middle")

    assert cleaned == "This is a 'quote' in the middle"
    assert quote_char == "'"


def test_change_quote_char():
    cleaned, quote_char = _clean_quote_text('This is a "quote" with mixed', quote_char="'")

    assert cleaned == "This is a 'quote' with mixed"
    assert quote_char == "'"


def test_nested_1():
    cleaned, quote_char = _clean_quote_text("'outer \"inner\"'")

    assert cleaned == 'outer "inner"'
    assert quote_char == '"'


def test_deep_mixed_quotes():
    cleaned, quote_char = _clean_quote_text('"quotes" and \'another "quote within" one\'')

    assert cleaned == '"quotes" and "another \'quote within\' one"'
    assert quote_char == '"'


def test_nested_in_the_middle():
    cleaned, quote_char = _clean_quote_text("Start 'outer \"inner\"' end")

    assert cleaned == "Start 'outer \"inner\"' end"
    assert quote_char == "'"


def test_nested_2():
    cleaned, quote_char = _clean_quote_text("\"outer 'inner'\"")

    assert cleaned == "outer 'inner'"
    assert quote_char == "'"
