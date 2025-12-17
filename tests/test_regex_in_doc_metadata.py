"""Tests for RegexInDocumentMetadataEvaluator."""

import json
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


# ---------------------------------------------------------------------------
# from_csv_line
# ---------------------------------------------------------------------------


def test_basic_json():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True,
        tags={"meta"},
        check='{"pattern": "chapter.*", "key": "source"}',
    )
    assert evaluator.pattern == "chapter.*"
    assert evaluator.metadata_key == "source"
    assert evaluator.expected is True
    assert evaluator.tags == {"meta"}


def test_missing_key_raises():
    with pytest.raises((ValueError, AssertionError)):
        RegexInDocumentMetadataEvaluator.from_csv_line(expected=True, tags=set(), check='{"pattern": "x"}')


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        RegexInDocumentMetadataEvaluator.from_csv_line(expected=True, tags=set(), check="not json")


def test_kwargs_stored():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=False,
        tags=set(),
        check='{"pattern": "x", "key": "k"}',
        priority="high",
    )
    assert evaluator.attributes["priority"] == "high"


# ---------------------------------------------------------------------------
# Evaluator run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_match_in_metadata():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True, tags=set(), check='{"pattern": "report", "key": "source"}'
    )
    docs = [Document(page_content="text", metadata={"source": "annual-report.pdf"})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is True


@pytest.mark.anyio
async def test_no_match_in_metadata():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True, tags=set(), check='{"pattern": "secret", "key": "source"}'
    )
    docs = [Document(page_content="text", metadata={"source": "public.pdf"})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_key_not_present():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True, tags=set(), check='{"pattern": "x", "key": "missing_key"}'
    )
    docs = [Document(page_content="text", metadata={"source": "file.pdf"})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_empty_documents():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True, tags=set(), check='{"pattern": "x", "key": "source"}'
    )
    with patch.object(evaluator, "get_documents", return_value=[]):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is False


@pytest.mark.anyio
async def test_match_across_multiple_documents():
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(
        expected=True, tags=set(), check='{"pattern": "target", "key": "tag"}'
    )
    docs = [
        Document(page_content="a", metadata={"tag": "irrelevant"}),
        Document(page_content="b", metadata={"tag": "target-doc"}),
    ]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is True


# ---------------------------------------------------------------------------
# Normalization (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern, metadata_value, should_match, desc",
    [
        ("report", "ANNUAL REPORT", True, "casefold metadata value"),
        ("REPORT", "annual report", True, "casefold pattern"),
        ("uf6", "UF₆-spec.pdf", True, "NFKC subscript digit in metadata"),
        ("uf₆", "UF6-spec.pdf", True, "NFKC subscript digit in pattern"),
        ("annual report", "annual   report", True, "whitespace collapse in metadata"),
        ("annual   report", "annual report", True, "whitespace collapse in pattern"),
        ("file 'a'", "file \u201ca\u201d", True, "curly quotes in metadata"),
        ("h2o", "H~2~O", True, "tilde subscript in metadata"),
        ("the end", "the end.", True, "trailing period in metadata"),
        ("alpha", "beta", False, "genuine mismatch"),
        (r"\d{4}", "report-2025.pdf", True, "digit pattern"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.anyio
async def test_normalization_end_to_end(pattern, metadata_value, should_match, desc):
    check = json.dumps({"pattern": pattern, "key": "source"})
    evaluator = RegexInDocumentMetadataEvaluator.from_csv_line(expected=True, tags=set(), check=check)
    docs = [Document(page_content="text", metadata={"source": metadata_value})]
    with patch.object(evaluator, "get_documents", return_value=docs):
        ctx = create_test_context("input", "output")
        result = await evaluator.run(ctx)
        assert result.value is should_match, f"Failed: {desc}"
