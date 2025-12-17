# MLflow Helper

High-level MLflow entry point that chains the three layers of the ragpill
pipeline. For direct access to individual layers, see
[Execution](execution.md), [Evaluation](evaluation.md), and [Upload](upload.md).

!!! tip "Recommendation"
    Create dedicated MLflow experiments for evaluations. Don't mix with production traces.

## evaluate_testset

::: ragpill.evaluate_testset
    options:
      show_root_heading: true
      show_source: true

## See Also

- [Layered Architecture Guide](../guide/layered-architecture.md) - When to use
  the combined entry point vs. the individual layers.
- [Execution Layer](execution.md) - `execute_dataset`
- [Evaluation Layer](evaluation.md) - `evaluate_results`
- [Upload Layer](upload.md) - `upload_results`
- [Result Types](types.md) - `EvaluationOutput`, `CaseResult`, `RunResult`, `AggregatedResult`
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Repeated Runs Guide](../guide/repeated-runs.md) - Multi-run evaluation with aggregation
- [Task Factory How-To](../how-to/task-factory.md) - Stateful tasks with repeat
