"""Guard the documented zero-server quickstart path so it can't silently rot.

Mirrors docs/getting-started/quickstart.md step 4 (execute_dataset +
evaluate_results, no tracking server) and the repeated-runs snippet, using the
same public imports the docs show.
"""

from __future__ import annotations

import pytest

from ragpill import Case, Dataset, evaluate_results, execute_dataset
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator


async def my_agent(question: str) -> str:
    return "Paris"


@pytest.mark.anyio
async def test_quickstart_zero_server_flow():
    case = Case(
        inputs="What is the capital of France?",
        metadata=TestCaseMetadata(),
        evaluators=[RegexInOutputEvaluator(pattern="paris", expected=True)],
    )
    dataset = Dataset(cases=[case])

    run = await execute_dataset(dataset, task=my_agent)  # zero-server temp store
    results = await evaluate_results(run, dataset)

    assert results.summary is not None
    assert results.case_results[0].aggregated.passed is True


@pytest.mark.anyio
async def test_quickstart_repeated_runs_flow():
    case = Case(
        inputs="What is the capital of France?",
        metadata=TestCaseMetadata(repeat=3, threshold=0.8),
        evaluators=[RegexInOutputEvaluator(pattern="paris", expected=True)],
    )
    testset = Dataset(cases=[case])

    run = await execute_dataset(testset, task=my_agent)
    result = await evaluate_results(run, testset)
    assert len(result.case_results[0].run_results) == 3
    assert result.case_results[0].aggregated.pass_rate == 1.0
