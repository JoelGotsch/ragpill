"""Tests for RegexInDocumentMetadataEvaluator."""

import asyncio
from unittest.mock import patch

import pytest
from mlflow.entities import Document
from pydantic_evals.evaluators import EvaluatorContext

from ragpill.base import EvaluatorMetadata
from ragpill.evaluators import RegexInDocumentMetadataEvaluator


def create_test_context(inputs: str, output: str) -> EvaluatorContext:
    """Helper to build an EvaluatorContext for tests."""
    return EvaluatorContext(
        inputs=inputs,
        output=output,
        metadata=EvaluatorMetadata(expected=True),
        name="test",
        expected_output=None,
        duration=0,
        _span_tree=None,
        attributes={},
        metrics={},
    )


class TestFromCSVLine:
    def test_basic_json(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True,
            tags={"meta"},
            check='{"pattern": "chapter.*", "key": "source"}',
        )
        assert evaluator.pattern == "chapter.*"
        assert evaluator.metadata_key == "source"
        assert evaluator.expected is True
        assert evaluator.tags == {"meta"}

    def test_missing_key_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            RegexInDocumentMetadataEvaluator.from_csv_line(
                expected=True, tags=set(), check='{"pattern": "x"}'
            )

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            RegexInDocumentMetadataEvaluator.from_csv_line(
                expected=True, tags=set(), check="not json"
            )

    def test_kwargs_stored(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=False,
            tags=set(),
            check='{"pattern": "x", "key": "k"}',
            priority="high",
        )
        assert evaluator.attributes["priority"] == "high"


class TestRun:
    def test_match_in_metadata(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True, tags=set(), check='{"pattern": "report", "key": "source"}'
        )
        docs = [Document(page_content="text", metadata={"source": "annual-report.pdf"})]
        with patch.object(evaluator, "get_documents", return_value=docs):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True

    def test_no_match_in_metadata(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True, tags=set(), check='{"pattern": "secret", "key": "source"}'
        )
        docs = [Document(page_content="text", metadata={"source": "public.pdf"})]
        with patch.object(evaluator, "get_documents", return_value=docs):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False

    def test_key_not_present(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True, tags=set(), check='{"pattern": "x", "key": "missing_key"}'
        )
        docs = [Document(page_content="text", metadata={"source": "file.pdf"})]
        with patch.object(evaluator, "get_documents", return_value=docs):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False

    def test_empty_documents(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True, tags=set(), check='{"pattern": "x", "key": "source"}'
        )
        with patch.object(evaluator, "get_documents", return_value=[]):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is False

    def test_match_across_multiple_documents(self):
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
            expected=True, tags=set(), check='{"pattern": "target", "key": "tag"}'
        )
        docs = [
            Document(page_content="a", metadata={"tag": "irrelevant"}),
            Document(page_content="b", metadata={"tag": "target-doc"}),
        ]
        with patch.object(evaluator, "get_documents", return_value=docs):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is True


class TestComprehensiveNormalization:
    """One comprehensive test exercising all normalization behaviour end-to-end.

    _normalize_text is applied to both the pattern and metadata values.
    """

    @pytest.mark.parametrize(
        "pattern, metadata_value, should_match, desc",
        [
            # Case-folding — (?i) is redundant
            ("report", "ANNUAL REPORT", True, "casefold metadata value"),
            ("REPORT", "annual report", True, "casefold pattern"),
            # Unicode NFKC
            ("uf6", "UF₆-spec.pdf", True, "NFKC subscript digit in metadata"),
            ("uf₆", "UF6-spec.pdf", True, "NFKC subscript digit in pattern"),
            # Whitespace collapsing
            ("annual report", "annual   report", True, "whitespace collapse in metadata"),
            ("annual   report", "annual report", True, "whitespace collapse in pattern"),
            # Quote normalization (curly → straight single quote)
            ("file 'a'", 'file \u201ca\u201d', True, "curly quotes in metadata"),
            # Markdown subscript stripping
            ("h2o", "H~2~O", True, "tilde subscript in metadata"),
            # Trailing period stripping
            ("the end", "the end.", True, "trailing period in metadata"),
            # No match
            ("alpha", "beta", False, "genuine mismatch"),
            # Regex features
            (r"\d{4}", "report-2025.pdf", True, "digit pattern"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_normalization_end_to_end(self, pattern, metadata_value, should_match, desc):
        import json

        check = json.dumps({"pattern": pattern, "key": "source"})
        evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(expected=True, tags=set(), check=check)
        docs = [Document(page_content="text", metadata={"source": metadata_value})]
        with patch.object(evaluator, "get_documents", return_value=docs):
            ctx = create_test_context("input", "output")
            result = asyncio.run(evaluator.run(ctx))
            assert result.value is should_match, f"Failed: {desc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
