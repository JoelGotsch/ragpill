# CSV Adapter

The CSV adapter provides a flexible way to define test cases and evaluators in CSV format.

The main function we use, is [load_testset](../api/csv.md#load_testset).

## Overview

The CSV adapter allows you to:

- Define test cases in a simple tabular format, e.g in Excel
- Configure multiple evaluators per test case
- Version control your test data
- Collaborate with non-technical team members

## CSV Format

### Required Columns

While the column names can be customized, by default the CSV adapter expects the following columns:
- **Question**: The input question or prompt
- **test_type**: Type of evaluator (e.g., 'LLMJudge'). Here is a list of [built-in evaluators](evaluators.md#built-in-evaluators). And here is [how to create your own custom evaluator](evaluators.md#creating-custom-evaluators)
- **expected**: Boolean (`true` or `false`) indicating whether this check should pass
  - Use `true` (default) for normal tests
  - Use `false` for negative tests (e.g., checking the LLM does NOT hallucinate a specific link)
- **check**: The evaluation criteria
  - For **LLMJudge**: This becomes the `rubric` - the criteria the LLM judge uses to evaluate
  - For other evaluators: The specific check type or validation rule
  - Can contain json if multiple arguments are used. In general, the logic for parsing has to be
  implemented in the evaluator's [`from_csv_line`](../api/base.md#baseevaluator)

### Optional Columns

- **mandatory**: Whether this check is mandatory (`true`/`false`, defaults to `true`)
- **tags**: Comma separated tags

### Understanding Column Mapping

For **LLMJudge** evaluator:
- `check` → `rubric` parameter (what to judge against)
- `expected` → `expected` parameter (should the check pass or fail?)

Example:
```csv
Question,test_type,expected,mandatory,tags,check
What is Python?,LLMJudge,true,true,tech,Should mention it's a programming language
What is Python?,LLMJudge,false,true,tech,Mentions snakes or reptiles
```

In the second row, `expected=false` means we're checking that the output does NOT contain something (negative test).

### Example CSV

```csv
Question,test_type,expected,mandatory,tags,check
What is the capital of France?,LLMJudge,true,true,"geography,factual",The capital of France is Paris
What is 2+2?,LLMJudge,true,true,"math,arithmetic",The answer should be 4
What is 2+2?,LLMJudge,false,true,"math,hallucination",Mentions calculus or complex mathematics
Name a primary color,LLMJudge,true,true,"art,colors",Should name red blue or yellow
```

Note the third row uses `expected=false` to check that the output does NOT contain something (negative test for hallucinations).

## Loading Test Sets

```python
from pathlib import Path
from ragpill.csv.testset import load_testset, default_evaluator_classes

# Load from CSV using default evaluator classes
# default_evaluator_classes includes: LLMJudge, RegexInSourcesEvaluator, etc.
dataset = load_testset(
    csv_path=Path("testset.csv"),
    evaluator_classes=default_evaluator_classes,
)

print(f"Loaded {len(dataset.cases)} test cases")
```

**Note:** The `default_evaluator_classes` includes all built-in evaluators like `LLMJudge`. For LLMJudge to work, you need to set environment variables for the model configuration:
- `RAGPILL_LLMJUDGE_API_KEY`
- `RAGPILL_LLMJUDGE_BASE_URL`
- `RAGPILL_LLMJUDGE_MODEL_NAME`

## Multiple Evaluators Per Question

You can attach multiple evaluators to the same question by adding multiple rows:

```csv
Question,test_type,expected,mandatory,tags,check
Explain photosynthesis,LLMJudge,true,true,"biology,science",Should mention sunlight
Explain photosynthesis,LLMJudge,true,false,"biology,quality",Should be clear and understandable
Explain photosynthesis,LLMJudge,false,true,"biology,accuracy",contain scientific errors
```

This creates a single test case with three evaluators.

## Global Evaluators

The CSV adapter supports **global evaluators** - a convenience feature that applies specific evaluators to ALL test cases. Define them by leaving the `Question` column empty:

```csv
Question,test_type,expected,mandatory,tags,check
,LLMJudge,true,true,global_politeness,Response should be polite and professional
,LLMJudge,true,false,global_safety,Should not contain harmful content
,LiteralQuotationTest,true,false,global_formatting,
What is Python?,LLMJudge,true,true,tech,Should mention it's a programming language
What is Python?,LLMJudge,false,true,tech,Should mention it's a snake
What is the capital of France?,LLMJudge,true,true,geography,Should answer Paris
```

In this example:
- The first three rows (with empty Question) define global evaluators
- These global evaluators will be automatically added to EVERY test case
- The "What is Python?" case will have **5 evaluators**: the 3 global ones + two specific ones
- The "What is the capital of France?" case will have **4 evaluators**: the 3 global ones + its specific one

**Use Cases for Global Evaluators:**
- **Safety checks**: Ensure all responses avoid harmful content
- **Tone/politeness**: Verify all responses maintain appropriate tone
- **Format requirements**: Check all responses follow a specific format
- **Company policies**: Verify all responses comply with organizational guidelines


## Custom Evaluators

You can create custom evaluators with the CSV adapter. There are two patterns for parameterizing them:

### Pattern 1: Environment Variables (for shared configuration)

Use this when all instances of your evaluator need the same configuration (e.g., API keys, base URLs, thresholds).

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from ragpill.base import BaseEvaluator, EvaluatorMetadata
from ragpill.csv.testset import load_testset, default_evaluator_classes
from pydantic_evals.evaluators import EvaluationReason
from pydantic_evals.evaluators.context import EvaluatorContext

class MyEvaluatorSettings(BaseSettings):
    """Settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_prefix='MY_EVALUATOR_')
    
    api_key: SecretStr
    threshold: float = 0.8

class MySharedConfigEvaluator(BaseEvaluator):
    """Evaluator with shared configuration via environment variables."""
    
    settings: MyEvaluatorSettings
    
    @classmethod
    def from_csv_line(
        cls,
        expected: bool,
        mandatory: bool,
        tags: str,
        check: str,
        **kwargs: Any
    ):
        """Create evaluator from CSV row data.
        
        The 'check' parameter contains the evaluation criteria.
        Settings are loaded from environment variables.
        """
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
            settings=MyEvaluatorSettings(),
        )
    
    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        # Use self.settings.api_key, self.settings.threshold, etc.
        passed = len(ctx.output) > self.settings.threshold
        
        return EvaluationReason(
            value=passed,
            reason=f"Output length {len(ctx.output)} (threshold: {self.settings.threshold})",
        )

# Set environment variables before running:
# export MY_EVALUATOR_API_KEY=your_key
# export MY_EVALUATOR_THRESHOLD=0.9
```
