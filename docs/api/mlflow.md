# MLflow Helper

MLflow integration utilities for experiment tracking and result management.

!!! tip "Recommendation"
    Create dedicated experiments for evaluations. Don't mix with production traces.

## Async (preferred)

::: ragpill.evaluate_testset_with_mlflow
    options:
      show_root_heading: true
      show_source: true

## Sync wrapper

Use this when `await` is not available (plain scripts, CLI tools, synchronous test suites).
It runs the async version in a dedicated thread, so it is safe to call from both sync and
async contexts — including Jupyter notebooks and FastAPI route handlers.

::: ragpill.evaluate_testset_with_mlflow_sync
    options:
      show_root_heading: true
      show_source: true

## See Also

- [Result Types](types.md) - `EvaluationOutput`, `CaseResult`, `RunResult`, `AggregatedResult`

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Repeated Runs Guide](../guide/repeated-runs.md) - Multi-run evaluation with aggregation
- [Task Factory How-To](../how-to/task-factory.md) - Stateful tasks with repeat
