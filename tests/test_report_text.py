"""Unit tests for ``ragpill.report._text`` primitives."""

from __future__ import annotations

from ragpill.report._text import indent, render_value, truncate


def test_truncate_under_budget_returns_input_unchanged():
    assert truncate("hello", 100) == "hello"


def test_truncate_at_exact_budget_returns_input_unchanged():
    assert truncate("hello", 5) == "hello"


def test_truncate_over_budget_appends_count_suffix():
    out = truncate("0123456789abcdef", 12)
    assert out.endswith(")")
    assert "+" in out
    assert len(out) <= 12


def test_truncate_zero_budget_yields_empty():
    assert truncate("anything", 0) == ""


def test_truncate_tiny_budget_falls_back_to_hard_clip():
    # No room for the "… (+N)" suffix at all.
    assert truncate("abcdef", 3) == "abc"


def test_render_value_collapses_whitespace_in_strings():
    assert render_value("hello\n\nworld\t  again") == "hello world again"


def test_render_value_serializes_dict_with_sorted_keys():
    result = render_value({"b": 1, "a": 2})
    assert result == '{"a": 2, "b": 1}'


def test_render_value_clips_long_strings():
    long_str = "x" * 1000
    out = render_value(long_str, max_chars=50)
    assert len(out) <= 50
    assert out.endswith(")")


def test_render_value_falls_back_to_str_for_non_serializable():
    class Obj:
        def __str__(self) -> str:
            return "obj-repr"

    out = render_value(Obj())
    assert "obj-repr" in out


def test_indent_adds_two_spaces_per_level():
    assert indent("a\nb", 2) == "    a\n    b"


def test_indent_zero_level_is_noop():
    assert indent("hello", 0) == "hello"


def test_indent_preserves_blank_lines():
    assert indent("a\n\nb", 1) == "  a\n\n  b"
