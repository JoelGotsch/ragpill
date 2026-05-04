# Evaluation Layer

The evaluate layer runs evaluators against a captured `DatasetRunOutput` and
produces an `EvaluationOutput` with `runs` / `cases` DataFrames.

This layer has **no MLflow server dependency** — it is pure in the sense that
it consumes dataclass inputs and returns dataclass outputs. This makes
evaluation easy to unit-test without any external infrastructure.

## evaluate_results

::: ragpill.evaluation.evaluate_results
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [Execution Layer](execution.md) — produce the `DatasetRunOutput` this layer consumes.
- [Upload Layer](upload.md) — persist the `EvaluationOutput` produced here.
- [Layered Architecture Guide](../guide/layered-architecture.md).
- [Result Types](types.md) — `EvaluationOutput`, `CaseResult`, `RunResult`, etc.
