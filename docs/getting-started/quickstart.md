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

Settings are read from the environment (and from a `.env` file in the working
directory, which the settings classes load automatically).

Tracking settings use the `RAGPILL_` prefix. All are optional — when
`RAGPILL_TRACKING_URI` is unset, a zero-server run uses a private temp store:

```
RAGPILL_TRACKING_URI=http://localhost:5000   # optional; omit for zero-server
RAGPILL_EXPERIMENT_NAME=my_project_evaluation # optional
```

If you use the LLM judge, set at least the API key (base URL and model name are
optional and fall back to the OpenAI defaults / `OPENAI_API_KEY`):

```
RAGPILL_LLMJUDGE_API_KEY=<your-api-key>
RAGPILL_LLMJUDGE_BASE_URL=<optional, defaults to the OpenAI base URL>
RAGPILL_LLMJUDGE_MODEL_NAME=<optional, defaults to 'gpt-4o'>
```

### 3. Load the TestSet

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

### 4. Run Evaluation (zero-server)

The quickest path needs no tracking server: `execute_dataset` captures to a
private temp store and `evaluate_results` scores the outputs. Both are async, so
wrap the call in `asyncio.run(...)` from a script.

```python
import asyncio

from ragpill import execute_dataset, evaluate_results


# Define your agent or function to test
async def my_agent(question: str) -> str:
    # Your agent logic here; a mock for this example.
    return "Paris"


async def main():
    run = await execute_dataset(dataset, task=my_agent)  # zero-server temp store
    results = await evaluate_results(run, dataset)
    print("\n📊 Evaluation Results:")
    print(results.summary)


asyncio.run(main())
```

!!! tip "Persisting to a tracking server"
    To log results to MLflow (or Langfuse / Phoenix), set `RAGPILL_TRACKING_URI`
    (or pass `settings=TrackingSettings(tracking_uri=...)`) and use the one-call
    `evaluate_testset(testset=dataset, task=my_agent)`, which chains
    execute → evaluate → upload. It requires a tracking URI; the zero-server path
    above is `execute_dataset` + `evaluate_results` used directly. See the
    [Layered Architecture Guide](../guide/layered-architecture.md).


## Repeated Runs

LLM outputs are non-deterministic. Run each test case multiple times for statistical confidence:

```python
from ragpill import Case, Dataset, execute_dataset, evaluate_results
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator

case = Case(
    inputs="What is the capital of France?",
    metadata=TestCaseMetadata(repeat=3, threshold=0.8),
    evaluators=[RegexInOutputEvaluator(pattern="paris", expected=True)],
)
testset = Dataset(cases=[case])

run = await execute_dataset(testset, task=my_agent)
result = await evaluate_results(run, testset)
print(result.summary)  # One row per case: passed, pass_rate, threshold
```

See the [Repeated Runs Guide](../guide/repeated-runs.md) for details on `task_factory`, threshold semantics, and server integration.

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

- Learn more about [Loading TestSets](../guide/csv-adapter.md) in detail
- Explore [Custom Evaluators](../how-to/custom-evaluator.ipynb)
- Set up [MLflow Integration](../api/mlflow.md) for tracking
