"""Upload layer: persist an ``EvaluationOutput`` to a tracking backend.

This layer is synchronous and isolated from task execution and evaluation. It
talks to whichever tracking backend is configured via
:func:`ragpill.backends.get_backend` — MLflow by default; Langfuse / Arize
Phoenix when the corresponding adapter is registered.

Two modes:

- ``upload_traces=False`` (default in the standard pipeline): Traces are
  already on the server because :func:`ragpill.execution.execute_dataset`
  wrote them directly. We only upload aggregated results (runs table, metrics,
  assessments, tags).
- ``upload_traces=True``: When a :class:`~ragpill.execution.DatasetRunOutput`
  was captured offline (local temp backend + JSON round-trip), we additionally
  serialize captured traces as a backend artifact so the full payload lives on
  the server.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ragpill.backends import Assessment, get_backend
from ragpill.settings import MLFlowSettings
from ragpill.types import CaseResult, EvaluationOutput

# ---------------------------------------------------------------------------
# Run lifecycle helpers
# ---------------------------------------------------------------------------


def _reattach_run(settings: MLFlowSettings, run_id: str | None) -> tuple[str | None, str]:
    """Reattach to an existing run (from execute_dataset) or start a new one.

    Args:
        settings: Tracking configuration.
        run_id: Optional existing run id. When provided, the active run is
            re-opened; otherwise a new run is started.

    Returns:
        ``(previous_tracking_uri, active_run_id)``. The previous URI lets the
        caller restore the tracking destination; the active run id is needed
        for trace-level operations such as judge-trace cleanup.
    """
    backend = get_backend()
    previous_uri = backend.get_tracking_uri()
    backend.set_destination(settings.ragpill_tracking_uri, settings.ragpill_experiment_name)
    if run_id:
        handle = backend.start_run(run_id=run_id)
    else:
        handle = backend.start_run(description=settings.ragpill_run_description)
    return previous_uri, handle.run_id


def _delete_llm_judge_traces(experiment_id: str, run_id: str) -> None:
    """Remove LLM-judge evaluation-only traces from the tracking UI.

    Called at the end of upload so the tracing UI only shows task traces.
    Walks the run's traces and deletes any whose root span carries the
    ``ragpill_is_judge_trace`` attribute (set by :class:`LLMJudge`).
    """
    backend = get_backend()
    traces = backend.search_traces(run_id=run_id, experiment_id=experiment_id)
    delete_trace_ids: list[str] = []
    for trace in traces:
        root = trace.data._get_root_span()
        if root and root.attributes.get("ragpill_is_judge_trace"):
            delete_trace_ids.append(trace.info.trace_id)
    if delete_trace_ids:
        backend.delete_traces(experiment_id=experiment_id, trace_ids=delete_trace_ids)


# ---------------------------------------------------------------------------
# Metric + assessment helpers
# ---------------------------------------------------------------------------


_MLFLOW_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_./ -]+")


def _slug(name: str) -> str:
    """Slug a string to MLflow's metric-name charset (alphanum, `_`, `.`, `/`, space, `-`)."""
    return _MLFLOW_METRIC_NAME_RE.sub("_", name)


def _log_accuracy_metrics(prefix: str, scores: dict[str, float]) -> None:
    """Log each entry of ``scores`` as ``f"{prefix}_{slug(key)}"``."""
    backend = get_backend()
    for name, value in scores.items():
        backend.log_metric(f"{prefix}_{_slug(name)}", value)


def _log_table_and_metrics(
    evaluation: EvaluationOutput,
    model_params: dict[str, str] | None,
) -> None:
    """Log the runs DataFrame as a tracking-backend table plus overall + per-tag + per-attribute accuracy."""
    backend = get_backend()
    backend.log_table(evaluation.runs, "evaluation_results.json")
    if model_params:
        backend.log_params(model_params)
    if evaluation.runs.empty:
        return

    eval_df: Any = evaluation.runs
    df_valid = eval_df[eval_df["evaluator_result"].notna()]
    if len(df_valid) > 0:
        overall_accuracy: float = float(df_valid["evaluator_result"].mean())
        backend.log_metric("overall_accuracy", overall_accuracy)
    _log_accuracy_metrics("accuracy_tag", evaluation.per_tag_accuracy())
    for attr_key, value_map in evaluation.per_attribute_accuracy_all().items():
        _log_accuracy_metrics(f"accuracy_attr_{_slug(attr_key)}", value_map)


def _log_assessments_and_tags(case_results: list[CaseResult]) -> None:
    """Log per-run + aggregate assessments and trace tags derived from metadata."""
    backend = get_backend()
    for cr in case_results:
        trace_id = cr.trace_id
        repeat = len(cr.run_results)

        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                source_type = "LLM_JUDGE" if "LLMJudge" in eval_result.source.name else "CODE"
                assessment = Assessment(
                    name=f"run-{rr.run_index}_{eval_name}",
                    value=eval_result.value,
                    source_type=source_type,
                    source_id=eval_result.source.name,
                    rationale=str(eval_result.reason),
                )
                if trace_id:
                    backend.log_assessment(trace_id, assessment)

        if repeat > 1 and trace_id:
            for eval_name, eval_pass_rate in cr.aggregated.per_evaluator_pass_rates.items():
                agg_passed = eval_pass_rate >= cr.aggregated.threshold
                rationale = (
                    f"Aggregate: "
                    f"{sum(1 for r in cr.run_results if eval_name in r.assertions and r.assertions[eval_name].value is True)}"
                    f"/{repeat} runs passed (threshold={cr.aggregated.threshold})"
                )
                agg_assessment = Assessment(
                    name=f"agg_{eval_name}",
                    value=agg_passed,
                    source_type="CODE",
                    source_id="ragpill_aggregation",
                    rationale=rationale,
                )
                backend.log_assessment(trace_id, agg_assessment)

        if trace_id:
            for key, value in cr.metadata.attributes.items():
                backend.set_trace_tag(trace_id, key, str(value))
            for tag in cr.metadata.tags:
                backend.set_trace_tag(trace_id, f"tag_{tag}", "true")


def _log_traces_as_artifact(evaluation: EvaluationOutput) -> None:
    """Serialize captured traces from ``evaluation.dataset_run`` and upload as an artifact."""
    if evaluation.dataset_run is None:
        return
    import os
    import tempfile

    payload = evaluation.dataset_run.to_json()
    tmp_dir = tempfile.mkdtemp(prefix="ragpill_upload_")
    try:
        path = os.path.join(tmp_dir, "dataset_run.json")
        with open(path, "w") as f:
            f.write(payload)
        get_backend().log_artifact(path, artifact_path="ragpill_traces")
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_experiment_id(settings: MLFlowSettings) -> str:
    """Return the experiment id for the configured experiment name."""
    return get_backend().resolve_experiment_id(settings.ragpill_experiment_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_to_mlflow(
    evaluation: EvaluationOutput,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
    upload_traces: bool = False,
) -> None:
    """Persist an :class:`EvaluationOutput` to an MLflow server.

    Args:
        evaluation: Output of :func:`ragpill.evaluation.evaluate_results`.
        mlflow_settings: Server connection + experiment info. When omitted, a
            default :class:`MLFlowSettings` is loaded from environment vars.
        model_params: Optional model parameters to log for reproducibility.
        upload_traces: When ``True``, serialize
            ``evaluation.dataset_run.to_json()`` and upload it as an MLflow
            artifact. Use this for the "disconnected execution + upload later"
            workflow where traces were captured offline and are not yet on the
            server.

    Example:
        ```python
        run_output = await execute_dataset(testset, task=my_task)
        eval_output = await evaluate_results(run_output, testset)
        upload_to_mlflow(eval_output, settings, upload_traces=False)
        ```

    See Also:
        [`execute_dataset`][ragpill.execution.execute_dataset]: Phase 1.
        [`evaluate_results`][ragpill.evaluation.evaluate_results]: Phase 2.
    """
    settings = mlflow_settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]
    backend = get_backend()
    dataset_run = evaluation.dataset_run
    run_id: str | None = dataset_run.mlflow_run_id if (dataset_run and dataset_run.mlflow_run_id) else None

    previous_uri, active_run_id = _reattach_run(settings, run_id)
    try:
        _log_table_and_metrics(evaluation, model_params)
        _log_assessments_and_tags(evaluation.case_results)
        if upload_traces:
            _log_traces_as_artifact(evaluation)

        # Strip LLM-judge traces created during evaluation (they only clutter the UI).
        experiment_id = _resolve_experiment_id(settings)
        if backend.is_run_active():
            _delete_llm_judge_traces(experiment_id, active_run_id)
    finally:
        if backend.is_run_active():
            backend.end_run()
        if previous_uri is not None:
            backend.set_tracking_uri(previous_uri)


def upload_dataset_run_json(path: str) -> EvaluationOutput:
    """Load a ``DatasetRunOutput`` JSON file and return it wrapped as an empty EvaluationOutput.

    Helper for scripts that serialize a run offline and want to load it back
    before calling :func:`evaluate_results` + :func:`upload_to_mlflow`.

    Args:
        path: Filesystem path to a JSON file produced by
            :meth:`ragpill.execution.DatasetRunOutput.to_json`.

    Returns:
        An :class:`EvaluationOutput` containing only ``dataset_run``; ``runs``
        and ``cases`` are empty DataFrames until the caller runs evaluation.
    """
    from ragpill.execution import DatasetRunOutput

    with open(path) as f:
        payload = f.read()
    dataset_run = DatasetRunOutput.from_json(payload)
    return EvaluationOutput(
        runs=pd.DataFrame(),
        cases=pd.DataFrame(),
        case_results=[],
        dataset_run=dataset_run,
    )


# Make json import non-unused when we want it for future expansion.
_ = json
