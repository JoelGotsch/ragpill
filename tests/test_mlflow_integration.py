"""Integration tests for evaluate_testset_with_mlflow (async).

These tests require a running MLflow server and are skipped by default.
Set the environment variable RUN_MLFLOW_INTEGRATION_TESTS=1 to enable them.
"""

import os

import pytest
from dotenv import load_dotenv
from pydantic_evals import Case, Dataset

from ragpill import evaluate_testset_with_mlflow, evaluate_testset_with_mlflow_sync
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.settings import MLFlowSettings
from ragpill.types import EvaluationOutput

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


def _dummy_task_sync(question: str) -> str:
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
    assert isinstance(result, EvaluationOutput)
    assert len(result.runs) > 0


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
    assert isinstance(result, EvaluationOutput)
    assert len(result.runs) > 0


# ---------------------------------------------------------------------------
# Repeat integration (MLflow required)
# ---------------------------------------------------------------------------


@skip_unless_enabled
@pytest.mark.anyio
async def test_repeat_with_mlflow_async():
    """repeat=3, threshold=0.6, stateless task -> EvaluationOutput with correct structure."""
    evaluator = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = Case(
        inputs="hello",
        metadata=TestCaseMetadata(repeat=3, threshold=0.6),
        evaluators=[evaluator],
    )
    testset = Dataset(cases=[case])
    result = await evaluate_testset_with_mlflow(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, EvaluationOutput)
    # 3 runs * 1 evaluator = 3 rows in runs
    assert len(result.runs) == 3
    # 1 case * 1 evaluator = 1 row in cases
    assert len(result.cases) == 1
    # Summary has 1 row
    assert len(result.summary) == 1


@skip_unless_enabled
def test_repeat_with_mlflow_sync():
    evaluator = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = Case(
        inputs="hello",
        metadata=TestCaseMetadata(repeat=3, threshold=0.6),
        evaluators=[evaluator],
    )
    testset = Dataset(cases=[case])
    result = evaluate_testset_with_mlflow_sync(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, EvaluationOutput)
    assert len(result.runs) == 3


# ---------------------------------------------------------------------------
# Task factory with MLflow
# ---------------------------------------------------------------------------


@skip_unless_enabled
@pytest.mark.anyio
async def test_task_factory_with_mlflow():
    """task_factory returns a fresh dummy task per run."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return _dummy_task

    evaluator = RegexInOutputEvaluator(pattern="hello", expected=True, tags=set())
    case = Case(
        inputs="hello",
        metadata=TestCaseMetadata(repeat=2),
        evaluators=[evaluator],
    )
    testset = Dataset(cases=[case])
    result = await evaluate_testset_with_mlflow(
        testset=testset,
        task_factory=factory,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, EvaluationOutput)
    assert call_count == 2  # factory called once per run


# ---------------------------------------------------------------------------
# Single repeat backward compatibility
# ---------------------------------------------------------------------------


@skip_unless_enabled
@pytest.mark.anyio
async def test_single_repeat_with_mlflow():
    """repeat=1 (default) -> EvaluationOutput with 1 row per evaluator."""
    testset = _make_minimal_testset()
    result = await evaluate_testset_with_mlflow(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    assert isinstance(result, EvaluationOutput)
    assert len(result.runs) == 1
    assert len(result.cases) == 1


# ---------------------------------------------------------------------------
# Mixed repeat values across cases
# ---------------------------------------------------------------------------


@skip_unless_enabled
@pytest.mark.anyio
async def test_mixed_repeat_with_mlflow():
    """case A has repeat=1, case B has repeat=3."""
    ev = RegexInOutputEvaluator(pattern="answer", expected=True, tags=set())
    case_a = Case(inputs="hello", metadata=TestCaseMetadata(repeat=1), evaluators=[ev])
    case_b = Case(inputs="world", metadata=TestCaseMetadata(repeat=3), evaluators=[ev])
    testset = Dataset(cases=[case_a, case_b])
    result = await evaluate_testset_with_mlflow(
        testset=testset,
        task=_dummy_task,
        mlflow_settings=_make_mlflow_settings(),
    )
    # case_a: 1 run * 1 eval = 1 row; case_b: 3 runs * 1 eval = 3 rows
    assert len(result.runs) == 4
    assert len(result.cases) == 2
