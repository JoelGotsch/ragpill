"""Test CSV testset creation."""

from pathlib import Path
from typing import Any

from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from ragpill.csv.testset import load_testset
from ragpill.evaluators import LLMJudge

# Prevent actual model requests during testing
models.ALLOW_MODEL_REQUESTS = False


def test_load_testset():
    """Test creating a dataset from the CSV file."""
    csv_path = Path(__file__).parent / "data" / "testset.csv"

    # Use TestModel for testing without making actual API calls
    llm_model = TestModel()

    # Create a test wrapper that injects the model
    class TestLLMJudge(LLMJudge):
        @classmethod
        def from_csv_line(cls, **kwargs: Any):
            kwargs.setdefault("model", llm_model)
            return super().from_csv_line(**kwargs)

    # Define available evaluator classes
    evaluator_classes = {
        "LLMJudge": TestLLMJudge,
        # Add more evaluator types as needed:
        # 'REGEX': RegexEvaluator,
        # 'CountSentences': CountSentencesEvaluator,
    }

    # Create dataset - no evaluator_kwargs needed!
    dataset = load_testset(
        csv_path=csv_path,
        evaluator_classes=evaluator_classes,
        skip_unknown_evaluators=True,  # Skip evaluators we haven't implemented yet
    )

    # Verify the dataset was created
    assert dataset is not None, "Dataset should not be None"
    assert len(dataset.cases) > 0, "Dataset should have at least one case"

    # Verify each case has the expected structure
    for case in dataset.cases:
        assert case.inputs, "Case should have inputs"
        assert isinstance(case.evaluators, list), "Case should have evaluators list"
        assert len(case.evaluators) > 0, f"Case '{case.inputs}' should have at least one evaluator"
        assert hasattr(case, "metadata"), "Case should have metadata"

        # Verify each evaluator has required attributes
        for evaluator in case.evaluators:
            assert hasattr(evaluator, "expected"), "Evaluator should have 'expected' attribute"
            assert hasattr(evaluator, "tags"), "Evaluator should have 'tags' attribute"
            assert hasattr(evaluator, "attributes"), "Evaluator should have 'attributes' attribute"
            assert isinstance(evaluator.tags, set), "Evaluator tags should be a set"
            assert isinstance(evaluator.attributes, dict), "Evaluator attributes should be a dict"

    # Find a case with multiple evaluators to verify tag/attribute distribution
    multi_eval_case = None
    for case in dataset.cases:
        if len(case.evaluators) > 1:
            multi_eval_case = case
            break

    if multi_eval_case:
        # Verify case metadata exists
        assert hasattr(multi_eval_case.metadata, "tags"), "Case metadata should have tags"
        assert hasattr(multi_eval_case.metadata, "attributes"), "Case metadata should have attributes"
        assert isinstance(multi_eval_case.metadata.tags, set), "Case tags should be a set"
        assert isinstance(multi_eval_case.metadata.attributes, dict), "Case attributes should be a dict"

    # Verify the first case
    first_case = dataset.cases[0]
    assert first_case.inputs, "First case should have inputs"
    assert len(first_case.evaluators) > 0, "First case should have evaluators"

    # Verify LLMJudge evaluators have required fields
    for evaluator in first_case.evaluators:
        if isinstance(evaluator, LLMJudge):
            assert hasattr(evaluator, "rubric"), "LLMJudge should have 'rubric' attribute"
            assert hasattr(evaluator, "model"), "LLMJudge should have 'model' attribute"
            assert evaluator.rubric, "LLMJudge rubric should not be empty"


def test_load_testset_with_global_evaluators():
    """Test creating a dataset with global evaluators (rows with empty questions)."""
    csv_path = Path(__file__).parent / "data" / "testset_with_global.csv"

    # Use TestModel for testing without making actual API calls
    llm_model = TestModel()

    # Create a test wrapper that injects the model
    class TestLLMJudge(LLMJudge):
        @classmethod
        def from_csv_line(cls, **kwargs: Any):
            kwargs.setdefault("model", llm_model)
            return super().from_csv_line(**kwargs)

    # Define available evaluator classes
    evaluator_classes = {
        "LLMJudge": TestLLMJudge,
    }

    # Create dataset with global evaluators
    dataset = load_testset(
        csv_path=csv_path,
        evaluator_classes=evaluator_classes,
    )

    # Verify the dataset was created
    assert dataset is not None, "Dataset should not be None"
    # Expected: 3 questions in the CSV (Python, France, for loop)
    # Each should have global evaluators + their own evaluators
    assert len(dataset.cases) == 3, f"Expected 3 cases, got {len(dataset.cases)}"

    # Check first case (Python question)
    python_case = None
    for case in dataset.cases:
        if "Python" in case.inputs:
            python_case = case
            break

    assert python_case is not None, "Should find Python question case"

    # Python question has 2 specific evaluators + 2 global evaluators = 4 total
    assert len(python_case.evaluators) == 2, (
        f"Python case should have 2 specific evaluators, got {len(python_case.evaluators)}"
    )
    assert len(dataset.evaluators) == 2, f"Dataset should have 2 global evaluators, got {len(dataset.evaluators)}"

    # Verify France case has fewer specific evaluators but still has globals
    france_case = None
    for case in dataset.cases:
        if "France" in case.inputs:
            france_case = case
            break

    assert france_case is not None, "Should find France question case"
    # France has 1 specific evaluator + 2 global = 3 total
    assert len(france_case.evaluators) == 1, (
        f"France case should have 1 specific evaluator, got {len(france_case.evaluators)}"
    )


if __name__ == "__main__":
    test_load_testset()
    test_load_testset_with_global_evaluators()
