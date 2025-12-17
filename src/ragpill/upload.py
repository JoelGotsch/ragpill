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

from typing import Any

import pandas as pd

from ragpill.backends import Assessment, get_backend
from ragpill.llm_judge import JUDGE_PROMPT_VERSION, judge_prompt_hash
from ragpill.settings import TrackingSettings
from ragpill.types import CaseResult, EvaluationOutput

# ---------------------------------------------------------------------------
# Run lifecycle helpers
# ---------------------------------------------------------------------------


def _reattach_run(settings: TrackingSettings, run_id: str | None, tracking_uri: str | None) -> tuple[str | None, str]:
    """Reattach to an existing run (from execute_dataset) or start a new one.

    Args:
        settings: Tracking configuration.
        run_id: Optional existing run id. When provided, the active run is
            re-opened; otherwise a new run is started.
        tracking_uri: The resolved destination to point the backend at (see
            :func:`upload_results` for the precedence rule).

    Returns:
        ``(previous_tracking_uri, active_run_id)``. The previous URI lets the
        caller restore the tracking destination; the active run id is needed
        for trace-level operations such as judge-trace cleanup.
    """
    backend = get_backend()
    previous_uri = backend.get_tracking_uri()
    backend.set_destination(tracking_uri, settings.experiment_name)
    if run_id:
        handle = backend.start_run(run_id=run_id)
    else:
        handle = backend.start_run(description=settings.run_description)
    return previous_uri, handle.run_id


# ---------------------------------------------------------------------------
# Metric + assessment helpers
# ---------------------------------------------------------------------------


def _log_accuracy_metrics(prefix: str, scores: dict[str, float]) -> None:
    """Log each entry of ``scores`` as ``f"{prefix}_{key}"``.

    Names are passed raw: each backend applies its own naming rules in
    ``log_metric`` (e.g. MLflow's metric-name charset).
    """
    backend = get_backend()
    for name, value in scores.items():
        backend.log_metric(f"{prefix}_{name}", value)


def _log_table_and_metrics(
    evaluation: EvaluationOutput,
    model_params: dict[str, str] | None,
) -> None:
    """Log the runs DataFrame as a tracking-backend table plus overall + per-tag + per-attribute accuracy."""
    backend = get_backend()
    backend.log_table(evaluation.runs, _RESULTS_TABLE_ARTIFACT)
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
        _log_accuracy_metrics(f"accuracy_attr_{attr_key}", value_map)


def _log_assessments_and_tags(case_results: list[CaseResult]) -> None:
    """Log per-run + aggregate assessments and trace tags derived from metadata.

    Per-run assessments target the run's own trace (``RunResult.trace_id``,
    falling back to the case-level id). Aggregates and tags target the
    case-level trace when one exists (span-mode grouping); in session mode
    there is no case trace — each repeat is its own trace — so they are logged
    to every per-run trace instead.
    """
    backend = get_backend()
    for cr in case_results:
        trace_id = cr.trace_id
        repeat = len(cr.run_results)
        # Case-level targets: the case trace in span mode, else the distinct
        # per-run traces (session mode; order-preserving dedup).
        case_level_ids = (
            [trace_id] if trace_id else list(dict.fromkeys(rr.trace_id for rr in cr.run_results if rr.trace_id))
        )

        for rr in cr.run_results:
            run_trace_id = rr.trace_id or trace_id
            for eval_name, eval_result in rr.assertions.items():
                assessment = Assessment(
                    name=f"run-{rr.run_index}_{eval_name}",
                    value=eval_result.value,
                    source_type=eval_result.source.source_type,
                    source_id=eval_result.source.name,
                    rationale=str(eval_result.reason),
                )
                if run_trace_id:
                    backend.log_assessment(run_trace_id, assessment)

        if repeat > 1:
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
                for tid in case_level_ids:
                    backend.log_assessment(tid, agg_assessment)

        for tid in case_level_ids:
            for key, value in cr.metadata.attributes.items():
                backend.set_trace_tag(tid, key, str(value))
            for tag in cr.metadata.tags:
                backend.set_trace_tag(tid, f"tag_{tag}", "true")


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


def _resolve_experiment_id(settings: TrackingSettings) -> str:
    """Return the experiment id for the configured experiment name."""
    return get_backend().resolve_experiment_id(settings.experiment_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Run tag recording upload progress, so a re-run is idempotent.
_UPLOAD_STATE_TAG = "ragpill_upload_state"
_RESULTS_TABLE_ARTIFACT = "evaluation_results.json"


def upload_results(
    evaluation: EvaluationOutput,
    settings: TrackingSettings | None = None,
    *,
    tracking_uri: str | None = None,
    model_params: dict[str, str] | None = None,
    upload_traces: bool = False,
    overwrite: bool = False,
) -> None:
    """Persist an :class:`EvaluationOutput` to the configured tracking backend.

    Behaviour is backend-driven via :func:`ragpill.backends.get_backend`
    (MLflow by default, Langfuse / Arize Phoenix when their adapter is
    registered). The upload is **idempotent**: it marks the run's upload state
    and refuses to re-upload a run that already completed (unless
    ``overwrite=True``), and on a retry it replaces the results-table artifact
    instead of appending duplicate rows.

    Args:
        evaluation: Output of :func:`ragpill.evaluation.evaluate_results`.
        settings: Backend connection + experiment info. When omitted, a
            default :class:`TrackingSettings` is loaded from environment vars.
        tracking_uri: Explicit destination override. When omitted, the
            destination is resolved as: this argument > the run's recorded
            ``DatasetRunOutput.tracking_uri`` (so an upload can't reattach the
            run against a *different* server than it was captured on) >
            ``settings.tracking_uri``.
        model_params: Optional model parameters to log for reproducibility.
        upload_traces: When ``True``, serialize
            ``evaluation.dataset_run.to_json()`` and upload it as a run
            artifact. Use this for the "disconnected execution + upload later"
            workflow where traces were captured offline and are not yet on the
            server.
        overwrite: Re-upload a run whose upload state is already ``complete``.
            Without it, a completed run raises rather than duplicating data.

    Raises:
        RuntimeError: If the run was already uploaded and ``overwrite`` is False.

    Example:
        ```python
        run_output = await execute_dataset(testset, task=my_task)
        eval_output = await evaluate_results(run_output, testset)
        upload_results(eval_output, settings, upload_traces=False)
        ```

    See Also:
        [`execute_dataset`][ragpill.execution.execute_dataset]: Phase 1.
        [`evaluate_results`][ragpill.evaluation.evaluate_results]: Phase 2.
    """
    settings = settings or TrackingSettings()  # pyright: ignore[reportCallIssue]
    backend = get_backend()
    dataset_run = evaluation.dataset_run
    run_id: str | None = dataset_run.run_id if (dataset_run and dataset_run.run_id) else None

    # Destination precedence: explicit arg > the run's recorded URI > settings.
    recorded_uri = dataset_run.tracking_uri if (dataset_run and dataset_run.tracking_uri) else None
    dest_uri = tracking_uri or recorded_uri or settings.tracking_uri

    # Record which judge prompts produced these scores, so a ragpill upgrade
    # that edits a judge system prompt (and thus shifts every score) is visible
    # on the run rather than silent. Only when the run actually used a judge.
    params: dict[str, str] = dict(model_params or {})
    runs: Any = evaluation.runs
    if "source_type" in runs.columns and bool((runs["source_type"] == "LLM_JUDGE").any()):
        params.setdefault("ragpill_judge_prompt_version", str(JUDGE_PROMPT_VERSION))
        params.setdefault("ragpill_judge_prompt_hash", judge_prompt_hash())

    previous_uri, active_run_id = _reattach_run(settings, run_id, dest_uri)
    try:
        # Idempotency: refuse to re-upload a completed run; on a prior partial or
        # (overwrite) complete run, replace the append-only results table so
        # retries don't duplicate rows. Mark ``partial`` before writing and
        # ``complete`` only after everything succeeds.
        prior_state = backend.get_run_tag(active_run_id, _UPLOAD_STATE_TAG)
        if prior_state == "complete" and not overwrite:
            raise RuntimeError(
                f"Run {active_run_id!r} was already uploaded (upload state 'complete'). "
                "Pass overwrite=True to re-upload."
            )
        backend.set_run_tag(active_run_id, _UPLOAD_STATE_TAG, "partial")
        if prior_state in ("partial", "complete"):
            backend.delete_run_artifact(active_run_id, _RESULTS_TABLE_ARTIFACT)

        _log_table_and_metrics(evaluation, params or None)
        _log_assessments_and_tags(evaluation.case_results)
        if upload_traces:
            _log_traces_as_artifact(evaluation)

        # Strip LLM-judge traces created during evaluation (they only clutter the UI).
        experiment_id = _resolve_experiment_id(settings)
        if backend.is_run_active():
            backend.delete_judge_traces(experiment_id, active_run_id)

        backend.set_run_tag(active_run_id, _UPLOAD_STATE_TAG, "complete")
    finally:
        if backend.is_run_active():
            backend.end_run()
        if previous_uri is not None:
            backend.set_tracking_uri(previous_uri)


def upload_dataset_run_json(path: str) -> EvaluationOutput:
    """Load a ``DatasetRunOutput`` JSON file and return it wrapped as an empty EvaluationOutput.

    Helper for scripts that serialize a run offline and want to load it back
    before calling :func:`evaluate_results` + :func:`upload_results`.

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
