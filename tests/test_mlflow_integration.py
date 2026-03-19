"""Integration tests for evaluate_testset_with_mlflow (async).

These tests require a running MLflow server and are skipped by default.
Set the environment variable RUN_MLFLOW_INTEGRATION_TESTS=1 to enable them.
"""

import os

import pandas as pd
import pytest
from pydantic_evals import Case, Dataset
from dotenv import load_dotenv

from ragpill import evaluate_testset_with_mlflow, evaluate_testset_with_mlflow_sync
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.settings import MLFlowSettings

load_dotenv()

SKIP_REASON = "Set RUN_MLFLOW_INTEGRATION_TESTS=1 to run MLflow integration tests"
skip_unless_enabled = pytest.mark.skipif(
    not os.getenv("RUN_MLFLOW_INTEGRATION_TESTS"),
    reason=SKIP_REASON,
)


def _make_minimal_testset() -> Dataset:
    """Minimal dataset with one evaluator — enough to exercise the MLflow wiring."""
    evaluator = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = Case(inputs="hello", metadata=TestCaseMetadata(), evaluators=[evaluator])
    return Dataset(cases=[case])


def _make_mlflow_settings() -> MLFlowSettings:
    return MLFlowSettings(
        ragpill_tracking_uri=os.getenv("EVAL_MLFLOW_TRACKING_URI", "http://localhost:5000"),
        ragpill_experiment_name="ragpill_integration_test",
    )


async def _dummy_task(question: str) -> str:
    return f"answer: {question}"


# ---------------------------------------------------------------------------
# Async version
# ---------------------------------------------------------------------------


@skip_unless_enabled
@pytest.mark.anyio
async def test_evaluate_testset_with_mlflow_async():
    testset = _make_minimal_testset()
    result = await evaluate_testset_with_mlflow(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Sync version
# ---------------------------------------------------------------------------


@skip_unless_enabled
def test_evaluate_testset_with_mlflow_sync():
    testset = _make_minimal_testset()
    result = evaluate_testset_with_mlflow_sync(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, pd.DataFrame)
