# Upload Layer

The upload layer persists an `EvaluationOutput` to a running MLflow server —
runs table, metrics, assessments, and optionally the captured traces as a JSON
artifact.

## upload_to_mlflow

::: ragpill.upload.upload_to_mlflow
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [Execution Layer](execution.md) — capture traces (with or without a server).
- [Evaluation Layer](evaluation.md) — produce the `EvaluationOutput`.
- [Layered Architecture Guide](../guide/layered-architecture.md) — the
  "disconnected execution + upload later" workflow uses `upload_traces=True`.
