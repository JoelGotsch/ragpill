# Evaluate Historical Outputs

The three-layer architecture lets you decouple task execution from evaluation.
This how-to shows how to **capture task outputs now** and **run evaluators
against them later** — useful when:

- Running the task is expensive (large LLM calls) and you want to try several
  evaluator versions against the same outputs.
- You want to run the task on a machine without access to your MLflow server
  and upload results later.
- You want to regression-test historic outputs when you change an evaluator.

## 1. Capture outputs offline

Run `execute_dataset` with the default local-temp backend. The
`DatasetRunOutput` is fully JSON-serializable — including captured MLflow
`Trace` objects.

```python
from ragpill import Case, Dataset, TestCaseMetadata, execute_dataset

async def my_task(question: str) -> str:
    # your real task here
    return await some_agent.run(question)

dataset = Dataset(
    cases=[
        Case(inputs="What is the capital of France?", metadata=TestCaseMetadata()),
        Case(inputs="What is 2+2?", metadata=TestCaseMetadata()),
    ]
)

run_output = await execute_dataset(dataset, task=my_task)

with open("run.json", "w") as f:
    f.write(run_output.to_json())
```

## 2. Load and evaluate later

```python
from ragpill import Dataset, Case, TestCaseMetadata, evaluate_results
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.execution import DatasetRunOutput

with open("run.json") as f:
    run_output = DatasetRunOutput.from_json(f.read())

# Attach evaluators to the dataset — the `inputs`/`metadata` must align with
# what was used at capture time.
dataset = Dataset(
    cases=[
        Case(
            inputs="What is the capital of France?",
            metadata=TestCaseMetadata(),
            evaluators=[RegexInOutputEvaluator(pattern="paris", expected=True)],
        ),
        Case(
            inputs="What is 2+2?",
            metadata=TestCaseMetadata(),
            evaluators=[RegexInOutputEvaluator(pattern="4", expected=True)],
        ),
    ]
)

eval_output = await evaluate_results(run_output, dataset)
print(eval_output.summary)
```

## 3. Upload to MLflow (optional)

When you're ready to persist to MLflow, pass `upload_traces=True` so the
offline-captured trace data also lives on the server.

```python
from ragpill import upload_to_mlflow
from ragpill.settings import MLFlowSettings

upload_to_mlflow(eval_output, MLFlowSettings(), upload_traces=True)
```

## Try different evaluators against the same run

Because `evaluate_results` is a pure function, you can call it multiple times
with different evaluator sets:

```python
dataset_v1 = Dataset(cases=[...with old evaluators...])
dataset_v2 = Dataset(cases=[...with new evaluators...])

eval_v1 = await evaluate_results(run_output, dataset_v1)
eval_v2 = await evaluate_results(run_output, dataset_v2)

# Compare without re-running the task:
print("V1 summary:", eval_v1.summary["passed"].mean())
print("V2 summary:", eval_v2.summary["passed"].mean())
```

## See Also

- [Layered Architecture Guide](../guide/layered-architecture.md) — the full
  model and other use cases.
- [Execution API](../api/execution.md) — `execute_dataset`, `DatasetRunOutput`.
- [Evaluation API](../api/evaluation.md) — `evaluate_results`.
- [Upload API](../api/upload.md) — `upload_to_mlflow`.
