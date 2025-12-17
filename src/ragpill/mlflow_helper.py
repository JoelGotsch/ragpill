"""Thin orchestrator: execute + evaluate + upload.

This module used to contain the full evaluation pipeline. After the layered
refactor, it delegates to:

- :func:`ragpill.execution.execute_dataset` — task execution + trace capture.
- :func:`ragpill.evaluation.evaluate_results` — evaluator application.
- :func:`ragpill.upload.upload_to_mlflow` — MLflow persistence (runs table,
  metrics, assessments).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragpill.base import CaseMetadataT
from ragpill.eval_types import Dataset
from ragpill.evaluation import evaluate_results
from ragpill.execution import TaskType, execute_dataset
from ragpill.settings import MLFlowSettings
from ragpill.types import EvaluationOutput
from ragpill.upload import upload_to_mlflow

__all__ = ["evaluate_testset_with_mlflow"]


async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    """Run the full evaluation pipeline against an MLflow server.

    Chains the three layers of the refactored architecture:

    1. :func:`~ragpill.execution.execute_dataset` runs the task against every
       case and captures traces directly to the configured MLflow server.
    2. :func:`~ragpill.evaluation.evaluate_results` runs every evaluator
       against the captured outputs.
    3. :func:`~ragpill.upload.upload_to_mlflow` persists aggregated results
       (tables, metrics, assessments) to the MLflow run created by step 1.

    Args:
        testset: The dataset to evaluate.
        task: The task callable. Mutually exclusive with ``task_factory``.
        task_factory: A zero-arg callable returning a fresh task instance per
            run. Mutually exclusive with ``task``.
        mlflow_settings: MLflow configuration. Falls back to environment vars.
        model_params: Optional model parameters to log for reproducibility.

    Returns:
        :class:`EvaluationOutput` with ``.runs``, ``.cases``, ``.summary``
        DataFrames and ``.case_results``.

    Raises:
        ValueError: If both or neither of ``task`` and ``task_factory`` are
            provided.

    Example:
        ```python
        from ragpill import evaluate_testset_with_mlflow

        result = await evaluate_testset_with_mlflow(
            testset=my_dataset,
            task=my_task,
            mlflow_settings=my_settings,
        )
        print(result.summary)
        ```
    """
    settings = mlflow_settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]

    run_output = await execute_dataset(
        testset,
        task=task,
        task_factory=task_factory,
        settings=settings,
        mlflow_tracking_uri=settings.ragpill_tracking_uri,
        capture_traces=True,
    )
    eval_output = await evaluate_results(run_output, testset, settings=settings)
    upload_to_mlflow(eval_output, settings, model_params=model_params, upload_traces=False)
    return eval_output
