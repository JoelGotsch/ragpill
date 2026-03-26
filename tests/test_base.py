"""Tests for ragpill.base (metadata merging, evaluator inheritance)."""

from ragpill.base import TestCaseMetadata, merge_metadata
from ragpill.evaluators import RegexInOutputEvaluator


class TestMergeMetadataExpectedInheritance:
    """Evaluators without explicit `expected` inherit from case metadata;
    evaluators with explicit `expected` keep their own value."""

    def test_inherits_case_expected_or_uses_explicit(self):
        case_meta = TestCaseMetadata(expected=False)
        inherits = RegexInOutputEvaluator(pattern="2")
        explicit = RegexInOutputEvaluator(pattern="3", expected=True)

        merged_inherits = merge_metadata(case_meta, inherits.metadata)
        merged_explicit = merge_metadata(case_meta, explicit.metadata)

        assert merged_inherits.expected is False, "Should inherit expected=False from case metadata"
        assert merged_explicit.expected is True, "Should keep explicit expected=True"
