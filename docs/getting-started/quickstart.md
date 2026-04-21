# Quick Start

Get started with ragpill in just a few minutes!

## Basic Usage

### 1. Prepare Your Test Data

Create a CSV file with your test cases. Here's a simple example:

```csv
Question,test_type,expected,tags,check
capital of france?,LLMJudge,true,geography,The answer is Paris
2+2?,LLMJudge,true,math,The answer should be 4
```

For more details on the csv structure, see [csv-adapter](../guide/csv-adapter.md#csv-format)

### 2. Prepare env variables

We recommend a `.env` file which is automatically detected by pydantic-settings. 

You need those environment variables for mlflow:

```
EVAL_MLFLOW_
```

If you are using LLMJudge. you need at least the API_KEY:

```
RAGPILL_LLMJUDGE_API_KEY=<your-api-key>
RAGPILL_LLMJUDGE_BASE_URL=<optional>
RAGPILL_LLMJUDGE_MODEL_NAME=<optional, defaults to 'gpt-4o'>
```



### 2. Load the TestSet

```python
from pathlib import Path
from ragpill.csv.testset import load_testset, default_evaluator_classes

# Define your CSV path
csv_path = Path("testset.csv")

# Create the dataset using default evaluators
dataset = load_testset(
    csv_path=csv_path,
    evaluator_classes=default_evaluator_classes,
)

print(f"✅ Created dataset with {len(dataset.cases)} test cases")
```

**Note:** For LLMJudge to work, set these environment variables:
- `RAGPILL_LLMJUDGE_API_KEY`
- `RAGPILL_LLMJUDGE_BASE_URL` 
- `RAGPILL_LLMJUDGE_MODEL_NAME`

**Note:** For mlflow tracking to work, you can either pass the 

### 3. Run Evaluation

```python
# Define your agent or function to test
async def my_agent(question: str) -> str:
    # Your agent logic here
    # For this example, we'll use a simple mock
    return "Paris"

# Run evaluation
from pydantic_evals import eval_

results = await eval_(
    dataset=dataset,
    callable=my_agent,
)

# Print results
print(f"\n📊 Evaluation Results:")
print(f"Total cases: {len(results.results)}")
```

## Repeated Runs

LLM outputs are non-deterministic. Run each test case multiple times for statistical confidence:

```python
from ragpill import evaluate_testset_with_mlflow
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator
from pydantic_evals import Case, Dataset

case = Case(
    inputs="What is the capital of France?",
    metadata=TestCaseMetadata(repeat=3, threshold=0.8),
    evaluators=[RegexInOutputEvaluator(pattern="paris", expected=True)],
)
testset = Dataset(cases=[case])

result = await evaluate_testset_with_mlflow(testset=testset, task=my_agent)
print(result.summary)  # One row per case: passed, pass_rate, threshold
```

See the [Repeated Runs Guide](../guide/repeated-runs.md) for details on `task_factory`, threshold semantics, and MLflow integration.

## CSV Format Guide

Your CSV file should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `Question` | The input question/prompt | Yes |
| `test_type` | Type of evaluator (e.g., LLMJudge) | Yes |
| `expected` | Boolean (true/false) - should this check pass? | Yes |
| `tags` | Comma-separated tags | No |
| `check` | Evaluation criteria (for LLMJudge: the rubric text) | Yes |

## Multiple Evaluators Per Question

You can add multiple rows with the same question to apply multiple evaluators:

```csv
Question,test_type,expected,tags,check
What is the capital of France?,LLMJudge,true,"geography,factual",Should mention Paris
What is the capital of France?,LLMJudge,false,quality,Should NOT mention historical irrelevant details
```

## Next Steps

- Learn more about [Loading TestSets](../tutorials/02-defining-testset.ipynb) in detail
- Explore [Custom Evaluators](../how-to/custom-evaluator.ipynb)
- Set up [MLflow Integration](../api/mlflow.md) for tracking
