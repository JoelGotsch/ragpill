"""Tests for ragpill.base (metadata merging, repeat/threshold validation)."""

import pytest

from ragpill.base import TestCaseMetadata, merge_metadata
from ragpill.evaluators import RegexInOutputEvaluator

# ---------------------------------------------------------------------------
# Metadata expected inheritance
# ---------------------------------------------------------------------------


def test_inherits_case_expected_or_uses_explicit():
    case_meta = TestCaseMetadata(expected=False)
    inherits = RegexInOutputEvaluator(pattern="2")
    explicit = RegexInOutputEvaluator(pattern="3", expected=True)

    merged_inherits = merge_metadata(case_meta, inherits.metadata)
    merged_explicit = merge_metadata(case_meta, explicit.metadata)

    assert merged_inherits.expected is False, "Should inherit expected=False from case metadata"
    assert merged_explicit.expected is True, "Should keep explicit expected=True"


# ---------------------------------------------------------------------------
# TestCaseMetadata validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("repeat", 0),
        ("repeat", -1),
        ("threshold", 1.1),
        ("threshold", -0.1),
    ],
    ids=["repeat_zero", "repeat_negative", "threshold_above_one", "threshold_negative"],
)
def test_metadata_rejects_invalid_values(field, value):
    with pytest.raises(Exception):
        TestCaseMetadata(**{field: value})
