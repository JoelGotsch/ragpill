# Layered Architecture

ragpill splits evaluation into three independent layers. Each layer has a
single responsibility and can be used standalone — you mix and match them to
fit your workflow.

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Execute    │ → │  Evaluate    │ → │  Upload         │
│             │   │              │   │                 │
│ execute_    │   │ evaluate_    │   │ upload_to_      │
│   dataset() │   │   results()  │   │   mlflow()      │
└─────────────┘   └──────────────┘   └─────────────────┘
      │                  │                    │
      ▼                  ▼                    ▼
DatasetRunOutput   EvaluationOutput    (persisted in MLflow)
```

## The three layers

### 1. Execute — `execute_dataset()`

Runs tasks against every case in a dataset and captures MLflow traces. It is
the only layer that interacts with MLflow **during task execution**. Two
tracing backends are supported:

- **Local temp SQLite** (default; `mlflow_tracking_uri=None`): a private temp
  DB is created for the run and deleted when execution completes. Traces are
  copied into the returned `DatasetRunOutput` before cleanup.
- **Direct server tracing**: when you pass `mlflow_tracking_uri=<uri>`, traces
  go directly to that server.

Output: [`DatasetRunOutput`](../api/execution.md#datasetrunoutput) — JSON-serializable.

### 2. Evaluate — `evaluate_results()`

Runs every evaluator against the captured outputs. **No MLflow server is
required** — this layer is pure with respect to dataclass inputs and outputs,
which makes it trivial to unit-test.

Output: [`EvaluationOutput`](../api/types.md) — `runs` / `cases` DataFrames
plus the structured `case_results`.

### 3. Upload — `upload_to_mlflow()`

Persists the aggregated results (runs table, metrics, per-trace assessments,
tags) to an MLflow server. When traces were captured offline, passing
`upload_traces=True` additionally writes the serialized traces as an MLflow
artifact.

## The combined shortcut

For the common case where you want all three layers in one call, use
[`evaluate_testset_with_mlflow()`](../api/mlflow.md) — it chains the three
layers internally with sensible defaults.

## Four use cases

### 1. Run once, evaluate many

Execute the task once and reuse the captured traces with different evaluator
sets. Useful when running the task is expensive (LLM calls) and evaluation is
cheap.

```python
from ragpill import execute_dataset, evaluate_results

run_output = await execute_dataset(testset_with_evaluators_a, task=my_task)

# First evaluation pass:
eval_a = await evaluate_results(run_output, testset_with_evaluators_a)

# Different evaluator set, same run output — no re-execution of the task:
eval_b = await evaluate_results(run_output, testset_with_evaluators_b)
```

### 2. CI without a server

Run the full pipeline in CI with no MLflow server — traces stay in the local
temp backend and are discarded when the process exits. You still get the
`EvaluationOutput` DataFrames to assert against.

```python
from ragpill import execute_dataset, evaluate_results

run_output = await execute_dataset(testset, task=my_task)  # no URI → local temp
eval_output = await evaluate_results(run_output, testset)
assert eval_output.summary["passed"].all()
```

### 3. Serialize and share

Execute on one machine, ship the results, evaluate elsewhere. The
`DatasetRunOutput` round-trips through JSON.

```python
# On machine A:
run_output = await execute_dataset(testset, task=my_task)
with open("run.json", "w") as f:
    f.write(run_output.to_json())

# On machine B:
from ragpill import DatasetRunOutput, evaluate_results
with open("run.json") as f:
    run_output = DatasetRunOutput.from_json(f.read())
eval_output = await evaluate_results(run_output, testset)
```

### 4. Disconnected execution + upload later

Run offline (no server), then upload once the server is reachable. Pass
`upload_traces=True` so the offline-captured trace data lives alongside the
aggregated results on the server.

```python
from ragpill import execute_dataset, evaluate_results, upload_to_mlflow

# Offline phase — local temp backend:
run_output = await execute_dataset(testset, task=my_task)
eval_output = await evaluate_results(run_output, testset)

# Later, when the server is reachable:
upload_to_mlflow(eval_output, settings, upload_traces=True)
```

## Async-only

The library is async-only. If your caller cannot use `await`, wrap calls in
`asyncio.run(...)` at your top-level entry point.

## Migration notes

- `evaluate_testset_with_mlflow_sync()` was removed. Use `asyncio.run(evaluate_testset_with_mlflow(...))`.
- `WrappedPydanticEvaluator` was removed along with the `pydantic_evals`
  dependency. If you relied on wrapping pydantic-evals evaluators, implement
  the logic as a direct `BaseEvaluator` subclass.

## See Also

- [Execution API](../api/execution.md)
- [Evaluation API](../api/evaluation.md)
- [Upload API](../api/upload.md)
- [Combined entry point](../api/mlflow.md)
