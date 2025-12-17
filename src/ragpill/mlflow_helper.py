"""Thin orchestrator: execute + evaluate + upload.

This module used to contain the full evaluation pipeline. After the layered
refactor, it delegates to:

- :func:`ragpill.execution.execute_dataset` — task execution + trace capture.
- :func:`ragpill.evaluation.evaluate_results` — evaluator application.
- :func:`ragpill.upload.upload_results` — backend persistence (runs table,
  metrics, assessments).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragpill.base import CaseMetadataT
from ragpill.eval_types import Dataset
from ragpill.evaluation import evaluate_results
from ragpill.execution import TaskType, execute_dataset
from ragpill.settings import TrackingSettings
from ragpill.types import EvaluationOutput
from ragpill.upload import upload_results

__all__ = ["evaluate_testset"]


async def evaluate_testset(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    settings: TrackingSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    """Run the full evaluation pipeline against the configured tracking backend.

    Chains the three layers of the refactored architecture:

    1. :func:`~ragpill.execution.execute_dataset` runs the task against every
       case and captures traces directly to the configured tracking backend.
    2. :func:`~ragpill.evaluation.evaluate_results` runs every evaluator
       against the captured outputs.
    3. :func:`~ragpill.upload.upload_results` persists aggregated results
       (tables, metrics, assessments) to the run created by step 1.

    Because it uploads to a server, this function requires an explicit
    ``tracking_uri`` (set ``RAGPILL_TRACKING_URI`` or pass ``settings``). For a
    zero-server run, use :func:`~ragpill.execution.execute_dataset` +
    :func:`~ragpill.evaluation.evaluate_results` directly — those default to a
    private temp store and skip upload.

    Args:
        testset: The dataset to evaluate.
        task: The task callable. Mutually exclusive with ``task_factory``.
        task_factory: A zero-arg callable returning a fresh task instance per
            run. Mutually exclusive with ``task``.
        settings: Tracking configuration. Falls back to environment vars.
        model_params: Optional model parameters to log for reproducibility.

    Returns:
        :class:`EvaluationOutput` with ``.runs``, ``.cases``, ``.summary``
        DataFrames and ``.case_results``.

    Raises:
        ValueError: If both or neither of ``task`` and ``task_factory`` are
            provided, or if no ``tracking_uri`` is configured.

    Example:
        ```python
        from ragpill import evaluate_testset

        result = await evaluate_testset(
            testset=my_dataset,
            task=my_task,
            settings=my_settings,
        )
        print(result.summary)
        ```
    """
    # Validate task/factory first (mirrors execute_dataset) so those errors take
    # precedence, then require a destination — upload has nowhere to go without one.
    if task is not None and task_factory is not None:
        raise ValueError("Provide either 'task' or 'task_factory', not both.")
    if task is None and task_factory is None:
        raise ValueError("Provide either 'task' or 'task_factory'.")

    settings = settings or TrackingSettings()  # pyright: ignore[reportCallIssue]
    if settings.tracking_uri is None:
        raise ValueError(
            "evaluate_testset uploads results to a tracking server, so a tracking URI is "
            "required. Set RAGPILL_TRACKING_URI (or pass settings=TrackingSettings(tracking_uri=...)). "
            "For a zero-server run, use execute_dataset() + evaluate_results() directly."
        )

    run_output = await execute_dataset(
        testset,
        task=task,
        task_factory=task_factory,
        settings=settings,
        tracking_uri=settings.tracking_uri,
        capture_traces=True,
    )
    eval_output = await evaluate_results(run_output, testset, settings=settings)
    upload_results(eval_output, settings, model_params=model_params, upload_traces=False)
    return eval_output
