"""Tests for task/task_factory parameter validation."""

import pytest

from ragpill.mlflow_helper import evaluate_testset_with_mlflow


@pytest.mark.anyio
async def test_both_task_and_factory_raises():
    with pytest.raises(ValueError, match="not both"):
        await evaluate_testset_with_mlflow(
            testset=None,  # type: ignore[arg-type]
            task=lambda x: x,
            task_factory=lambda: lambda x: x,
        )


@pytest.mark.anyio
async def test_neither_task_nor_factory_raises():
    with pytest.raises(ValueError, match="Provide either"):
        await evaluate_testset_with_mlflow(
            testset=None,  # type: ignore[arg-type]
            task=None,
            task_factory=None,
        )
