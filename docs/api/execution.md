# Execution Layer

The execute layer runs tasks against a dataset and captures traces. It is the
first of three independent layers in the ragpill pipeline:

1. **Execute** — this module.
2. **Evaluate** — [`ragpill.evaluation`](evaluation.md).
3. **Upload** — [`ragpill.upload`](upload.md).

## execute_dataset

::: ragpill.execution.execute_dataset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## DatasetRunOutput

::: ragpill.execution.DatasetRunOutput
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## CaseRunOutput

::: ragpill.execution.CaseRunOutput
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## TaskRunOutput

::: ragpill.execution.TaskRunOutput
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [Evaluation Layer](evaluation.md) — run evaluators against a `DatasetRunOutput`.
- [Upload Layer](upload.md) — persist results to MLflow.
- [Layered Architecture Guide](../guide/layered-architecture.md) — the three-layer model
  and use cases.
