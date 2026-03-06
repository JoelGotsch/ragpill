# Evaluators

## What are Evaluators?

Evaluators are the core components that check whether an LLM output (this can include the sources that were used to create the output, tool-calls,etc.) meets specified criteria. Each evaluator performs a specific type of check and returns a pass/fail result with reasoning.

## Built-in Evaluators

See [Evaluators](../api/evaluators.md)

## Creating Custom Evaluators

### Basic Pattern

All custom evaluators must:
1. Inherit from [`BaseEvaluator`](../api/base.md#baseevaluator) or one of the other useful [Base Evaluators](../api/evaluators.md#base-evaluators)
2. Implement `from_csv_line()` class method with standard signature
3. Implement `async run()` method

```python
from typing import Any
from ragpill.base import BaseEvaluator, EvaluatorMetadata
from pydantic_evals.evaluators import EvaluationReason
from pydantic_evals.evaluators.context import EvaluatorContext

class MyEvaluator(BaseEvaluator):
    """Description of what this evaluator checks."""
    
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
        
        This class method is required for CSV integration.
        The signature must be exactly this - do not add custom parameters here.
        Use the 'check' parameter for per-instance config (see examples below).
        """
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
        )
    
    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        # Your evaluation logic
        passed = self._check_condition(ctx.output)
        
        return EvaluationReason(
            value=passed,
            reason=f"Explanation of why it {'passed' if passed else 'failed'}",
        )
    
    def _check_condition(self, output: str) -> bool:
        # Helper method
        return True
```

### Parameterization Patterns

There are two ways to parameterize custom evaluators:

#### Pattern 1: Environment Variables (for shared configuration)

Use this for configuration shared across all instances (API keys, global thresholds, etc.):

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class LengthEvaluatorSettings(BaseSettings):
    """Settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_prefix='LENGTH_EVAL_')
    
    api_key: SecretStr
    min_length: int = 10
    max_length: int = 1000

class LengthEvaluator(BaseEvaluator):
    """Checks if output length is within bounds from settings."""
    
    settings: LengthEvaluatorSettings
    
    @classmethod
    def from_csv_line(cls, expected: bool, mandatory: bool, tags: str, check: str, **kwargs: Any):
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
            settings=LengthEvaluatorSettings(),
        )
    
    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        length = len(ctx.output)
        passed = self.settings.min_length <= length <= self.settings.max_length
        
        return EvaluationReason(
            value=passed,
            reason=f"Length {length} (range: {self.settings.min_length}-{self.settings.max_length})",
        )

# Set environment variables:
# export LENGTH_EVAL_MIN_LENGTH=50
# export LENGTH_EVAL_MAX_LENGTH=500
```

#### Pattern 2: JSON in check Column (for per-instance configuration)

Use this for parameters that vary per test case (regex patterns, specific values, etc.):

```python
import json

class RegexEvaluator(BaseEvaluator):
    """Checks if output matches a regex pattern from check column."""
    
    pattern: str
    
    @classmethod
    def from_csv_line(cls, expected: bool, mandatory: bool, tags: str, check: str, **kwargs: Any):
        """Parse pattern from check column (plain text or JSON)."""
        try:
            check_dict = json.loads(check)
            if isinstance(check_dict, dict):
                pattern = check_dict.get('pattern', check)
            else:
                pattern = check
        except json.JSONDecodeError:
            # Plain text - use as pattern
            pattern = check
        
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
            pattern=pattern,
        )
    
    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        import re
        regex = re.compile(self.pattern, re.IGNORECASE)
        match = regex.search(ctx.output)
        passed = match is not None
        
        return EvaluationReason(
            value=passed,
            reason=f"Pattern '{self.pattern}' {'found' if passed else 'not found'}",
        )

# CSV examples:
# Plain text pattern:
# Question,test_type,expected,mandatory,tags,check
# What is Python?,RegexEvaluator,true,true,tech,programming language
#
# JSON pattern with additional config:
# What is Python?,RegexEvaluator,true,true,tech,"{\"pattern\": \".*programming.*\"}"
```

### Real-World Example: Built-in Evaluator

See the built-in `RegexInDocumentMetadataEvaluator` for a complete example that uses JSON configuration:

```python
# From evaluators.py
@classmethod
def from_csv_line(cls, expected: bool, mandatory: bool, tags: str, check: str, **kwargs: Any):
    """Create evaluator from CSV with JSON in check column."""
    try:
        check_dict = json.loads(check)
        if isinstance(check_dict, dict):
            pattern = check_dict.get('pattern')
            metadata_key = check_dict.get('key')
        else:
            raise ValueError("check must be a JSON object")
    except json.JSONDecodeError:
        raise ValueError(
            f"RegexInDocumentMetadataEvaluator requires 'check' to be a JSON string "
            f"with 'pattern' and 'key'. Got: {check}"
        )
    
    return cls(
        expected=expected,
        mandatory=mandatory,
        tags=tags,
        attributes=kwargs,
        pattern=pattern,
        metadata_key=metadata_key,
    )

# CSV usage:
# Question,test_type,expected,mandatory,tags,check
# Query docs,RegexInDocumentMetadata,true,true,retrieval,"{\"pattern\": \".*2024.*\", \"key\": \"date\"}"
```

## Custom Attributes

You can add custom attributes to evaluators by adding columns to your CSV:

```csv
Question,test_type,expected,mandatory,tags,check,priority,category
What is X?,LLMJudge,true,true,factual,answer_correctness,high,science
What is Y?,RegexEvaluator,false,false,format,email_format,low,validation
```

These custom columns (like `priority` and `category`) are automatically:
1. Passed to each evaluator's `attributes` dict via the `**kwargs` in `from_csv_line()`
2. Available in your evaluator through `self.attributes`

**Important:** If all evaluators for a given question have the **same value** for an attribute, that attribute becomes part of the Test Case metadata and will be visible in MLflow tracking.

```python
# In code - extend default evaluators with your custom class
from ragpill.csv.testset import load_testset, default_evaluator_classes

evaluator_classes = default_evaluator_classes | {
    'MyEvaluator': MyEvaluator,
}

dataset = load_testset(
    csv_path="testset.csv",
    evaluator_classes=evaluator_classes,
)

# Access attributes in your evaluator
class MyEvaluator(BaseEvaluator):
    @classmethod
    def from_csv_line(cls, expected: bool, mandatory: bool, tags: str, check: str, **kwargs: Any):
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,  # Contains {priority: "high", category: "science"}
        )
    
    async def run(self, ctx):
        priority = self.attributes.get('priority', 'medium')
        # Use the attribute in your logic
        ...
```

## Multiple Evaluators

You can attach multiple evaluators to a single test case:

```python
case = Case(
    input=BaseTestInput(metadata=metadata),
    evaluators=[
        LLMJudge(...),  # Check correctness
        LengthEvaluator(...),  # Check length
        RegexEvaluator(...),  # Check format
    ],
)
```

## Evaluator Results

### EvaluationReason Structure

Each evaluator's `run()` method returns an `EvaluationReason`:

```python
EvaluationReason(
    value=True,                  # Whether the check passed
    reason="Explanation...",     # Human-readable explanation
)
```

## Best Practices

### 1. Clear Naming

Use descriptive names for your evaluators:

```python
# Good
class EmailFormatEvaluator(BaseEvaluator): ...
class SentimentPositivityEvaluator(BaseEvaluator): ...

# Avoid
class Evaluator1(BaseEvaluator): ...
class CheckThing(BaseEvaluator): ...
```

### 2. Informative Reasons

Provide helpful explanations in `reason`:

```python
# Good
reason = f"Found 3 of 5 required keywords: {found_keywords}"

# Less helpful
reason = "Failed"
```

### 3. Deterministic When Possible

Prefer deterministic checks over LLM judges when possible:

- Regex for format validation
- Length for size constraints
- Keyword matching for required terms

Use LLM judges for:
- Semantic correctness
- Tone and style

### 5. Document Your Evaluators

Add docstrings explaining what and why:

```python
class KeywordEvaluator(BaseEvaluator):
    """Checks if specific keywords are present in the output.
    
    Useful for ensuring important terms are mentioned without
    requiring exact phrasing. Can check for any keyword (OR logic)
    or all keywords (AND logic).
    
    Args:
        keywords: List of keywords to check for
        require_all: If True, all keywords must be present (AND).
                    If False, any keyword is sufficient (OR).
    """
```

## Handling of Evaluator Failures

By default, if an evaluator raises an exception during `run()`, it will be treated as a failure (i.e., `value=False`) and the exception message will be recorded in the reason. This ensures that unexpected errors in evaluators do not crash the entire evaluation process and are properly logged for debugging.