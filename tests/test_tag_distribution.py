"""Quick test to verify tag and attribute distribution."""
from pathlib import Path
from typing import Any

from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from ragpill.csv.testset import load_testset
from ragpill.evaluators import LLMJudge

# Prevent actual model requests during testing
models.ALLOW_MODEL_REQUESTS = False


def test_tag_and_attribute_distribution():
    """Test that tags and attributes are correctly distributed between case and evaluators."""
    csv_path = Path(__file__).parent / 'data' / 'testset_example.csv'
    
    # Use TestModel for testing without making actual API calls
    llm_model = TestModel()
    
    # Create a test wrapper that injects the model
    class TestLLMJudge(LLMJudge):
        @classmethod
        def from_csv_line(cls, **kwargs: Any):
            # Override get_llm to return our test model
            kwargs['get_llm'] = lambda: llm_model
            return super().from_csv_line(**kwargs)
    
    evaluator_classes = {'LLMJudge': TestLLMJudge}
    
    dataset = load_testset(
        csv_path=csv_path,
        evaluator_classes=evaluator_classes,
    )
    
    # Verify the first case (Washington)
    washington_case = dataset.cases[0]
    assert washington_case.inputs == "Tell me about Washington"
    assert len(washington_case.evaluators) == 2
    
    # "Ambiguity" should be at case level (common to both)
    assert "Ambiguity" in washington_case.metadata.tags
    
    # "Critical" and "Style" should be at evaluator level only
    assert "Critical" not in washington_case.metadata.tags
    assert "Style" not in washington_case.metadata.tags
    assert "Critical" in washington_case.evaluators[0].tags
    assert "Style" in washington_case.evaluators[1].tags
    
    # Both have importance=high and category=factual, so should be at case level
    assert washington_case.metadata.attributes["importance"] == "high"
    assert washington_case.metadata.attributes["category"] == "factual"
    
    # Evaluators should not have these attributes since they're at case level
    assert "importance" not in washington_case.evaluators[0].attributes
    assert "category" not in washington_case.evaluators[0].attributes



if __name__ == '__main__':
    test_tag_and_attribute_distribution()
