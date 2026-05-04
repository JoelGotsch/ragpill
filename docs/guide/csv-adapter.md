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

- **tags**: Comma separated tags
- **repeat**: Number of times to run this test case (integer, e.g. `3`). Empty or absent defers to `MLFlowSettings.ragpill_repeat` (default: 1).
- **threshold**: Minimum fraction of runs that must pass (float, e.g. `0.8`). Empty or absent defers to `MLFlowSettings.ragpill_threshold` (default: 1.0).

!!! note
    All rows for the same question must have the same `repeat` and `threshold` values. Inconsistent values will raise a `ValueError`.

### Understanding Column Mapping

For **LLMJudge** evaluator:
- `check` → `rubric` parameter (what to judge against)
- `expected` → `expected` parameter (should the check pass or fail?)

Example:
```csv
Question,test_type,expected,tags,check
What is Python?,LLMJudge,true,tech,Should mention it's a programming language
What is Python?,LLMJudge,false,tech,Mentions snakes or reptiles
```

In the second row, `expected=false` means we're checking that the output does NOT contain something (negative test).

### Example CSV

```csv
Question,test_type,expected,tags,check
What is the capital of France?,LLMJudge,true,"geography,factual",The capital of France is Paris
What is 2+2?,LLMJudge,true,"math,arithmetic",The answer should be 4
What is 2+2?,LLMJudge,false,"math,hallucination",Mentions calculus or complex mathematics
Name a primary color,LLMJudge,true,"art,colors",Should name red blue or yellow
```

Note the third row uses `expected=false` to check that the output does NOT contain something (negative test for hallucinations).

### Example CSV with Repeated Runs

```csv
Question,test_type,expected,tags,check,repeat,threshold
What is the capital of France?,RegexInOutputEvaluator,true,"geography,factual",paris,3,0.8
What is 2+2?,RegexInOutputEvaluator,true,"math,arithmetic",4,,
```

- The geography question runs 3 times, requiring 80% pass rate
- The math question uses global defaults (repeat=1, threshold=1.0)

See [Repeated Runs](repeated-runs.md) for full documentation.

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
Question,test_type,expected,tags,check
Explain photosynthesis,LLMJudge,true,"biology,science",Should mention sunlight
Explain photosynthesis,LLMJudge,true,"biology,quality",Should be clear and understandable
Explain photosynthesis,LLMJudge,false,"biology,accuracy",contain scientific errors
```

This creates a single test case with three evaluators.

## Global Evaluators

The CSV adapter supports **global evaluators** - a convenience feature that applies specific evaluators to ALL test cases. Define them by leaving the `Question` column empty:

```csv
Question,test_type,expected,tags,check
,LLMJudge,true,global_politeness,Response should be polite and professional
,LLMJudge,true,global_safety,Should not contain harmful content
,LiteralQuoteEvaluator,true,global_formatting,
What is Python?,LLMJudge,true,tech,Should mention it's a programming language
What is Python?,LLMJudge,false,tech,Should mention it's a snake
What is the capital of France?,LLMJudge,true,geography,Should answer Paris
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
from ragpill.eval_types import EvaluationReason, EvaluatorContext

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
        tags: set[str],
        check: str,
        **kwargs: Any
    ):
        """Create evaluator from CSV row data.
        
        The 'check' parameter contains the evaluation criteria.
        Settings are loaded from environment variables.
        """
        return cls(
            expected=expected,
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

### Pattern 2: JSON in check Column (for per-instance configuration)

Use this when different test cases need different parameters (e.g., different regex patterns, metadata keys).

```python
import json
from ragpill.base import BaseEvaluator, EvaluatorMetadata
from ragpill.csv.testset import load_testset, default_evaluator_classes
from ragpill.eval_types import EvaluationReason, EvaluatorContext

class MyJsonConfigEvaluator(BaseEvaluator):
    """Evaluator with per-instance configuration via JSON in check column."""
    
    pattern: str
    metadata_key: str
    
    @classmethod
    def from_csv_line(
        cls,
        expected: bool,
        tags: set[str],
        check: str,
        **kwargs: Any
    ):
        """Create evaluator from CSV row data.
        
        The 'check' parameter should be a JSON string with required parameters.
        Example: {"pattern": ".*error.*", "metadata_key": "status"}
        """
        try:
            check_dict = json.loads(check)
            if not isinstance(check_dict, dict):
                raise ValueError("check must be a JSON object")
            
            pattern = check_dict.get('pattern')
            metadata_key = check_dict.get('metadata_key')
            
            if not pattern or not metadata_key:
                raise ValueError("check must contain 'pattern' and 'metadata_key'")
            
        except json.JSONDecodeError:
            raise ValueError(
                f"MyJsonConfigEvaluator requires 'check' to be a JSON string "
                f"with 'pattern' and 'metadata_key' keys. Got: {check}"
            )
        
        return cls(
            expected=expected,
            tags=tags,
            attributes=kwargs,
            pattern=pattern,
            metadata_key=metadata_key,
        )
    
    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        import re
        # Use per-instance configuration
        regex = re.compile(self.pattern)
        matched = bool(regex.search(ctx.output))
        
        return EvaluationReason(
            value=matched,
            reason=f"Pattern '{self.pattern}' {'found' if matched else 'not found'}",
        )

# CSV example for this evaluator:
# Question,test_type,expected,tags,check
# What is the status?,MyJsonConfig,true,test,"{\"pattern\": \".*success.*\", \"metadata_key\": \"status\"}"
```

### Using Custom Evaluators

```python
from ragpill.csv.testset import load_testset, default_evaluator_classes

# Extend default evaluators with your custom evaluators
dataset = load_testset(
    csv_path="testset.csv",
    evaluator_classes=default_evaluator_classes | {
        'MySharedConfig': MySharedConfigEvaluator,
        'MyJsonConfig': MyJsonConfigEvaluator,
    },
)
```

### When to Use Which Pattern

- **Pattern 1 (Environment Variables)**: 
  - API keys, base URLs, model names
  - Global thresholds or limits
  - Configuration shared across all test cases
  - Example: `LLMJudge` uses this for model configuration

- **Pattern 2 (JSON in check column)**:
  - Different regex patterns per test case
  - Different metadata keys to check
  - Test-specific parameters
  - Example: `RegexInDocumentMetadataEvaluator` uses this for pattern and key

## Encoding Support

The CSV adapter automatically handles different encodings:

- UTF-8
- UTF-8 with BOM
- Latin-1
- Windows-1252 (CP1252)

You don't need to specify the encoding manually.

## Best Practices

### 1. Version Control

Keep your CSV files in version control alongside your code:

```
project/
├── src/
├── tests/
│   └── data/
│       ├── testset.csv
```

### 2. Realistic Test Cases

Use questions verbetim like a user would ask.

```csv
# Good
capital of france?,Paris,...
2+2,4,...
```

### 3. Meaningful Tags

Use tags to organize and filter tests:

```csv
Question,test_type,expected,tags,check
...,"geography,european,capitals",...
...,"math,arithmetic,basic",...
...,"biology,science,photosynthesis",...
```

### 4. Separate Concerns

Consider separate CSV files for different domains:

```
testsets/
├── geography_tests.csv
├── math_tests.csv
└── science_tests.csv
```

### 5. Document Expected Values

Be specific about what you expect:

```csv
# Good - specific rubric in check column
check: "The answer should mention Paris as the capital city"

# Too vague
check: "Correct answer"
```

## See Also

- [TestSets Guide](testsets.md)
- [CSV Module API](../api/csv.md)
