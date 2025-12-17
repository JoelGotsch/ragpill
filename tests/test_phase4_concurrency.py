"""Phase 4: opt-in timeouts and bounded-concurrency evaluation.

Timeouts are caller-specified (never imposed by ragpill), and raising the
evaluation concurrency must not change results.
"""

from __future__ import annotations

import anyio
import pytest

from ragpill.base import TestCaseMetadata, default_input_to_key
from ragpill.eval_types import Case, Dataset
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput, execute_dataset


@pytest.mark.anyio
async def test_task_timeout_records_timeout_error():
    async def slow(_q: str) -> str:
        await anyio.sleep(1.0)
        return "done"

    ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="q", metadata=TestCaseMetadata())])
    out = await execute_dataset(ds, task=slow, capture_traces=False, task_timeout_s=0.05)
    tr = out.cases[0].task_runs[0]
    assert tr.error is not None
    assert "TimeoutError" in tr.error


@pytest.mark.anyio
async def test_no_timeout_by_default():
    # Without task_timeout_s, a slow-ish task completes — ragpill imposes no budget.
    async def slowish(q: str) -> str:
        await anyio.sleep(0.05)
        return q

    ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="q", metadata=TestCaseMetadata())])
    out = await execute_dataset(ds, task=slowish, capture_traces=False)
    assert out.cases[0].task_runs[0].output == "q"
    assert out.cases[0].task_runs[0].error is None


def _case_run(inputs: str, outputs: list[str]) -> CaseRunOutput:
    runs = [TaskRunOutput(run_index=i, input_key=f"k_{i}", output=o, duration=0.01) for i, o in enumerate(outputs)]
    return CaseRunOutput(
        case_name=inputs,
        inputs=inputs,
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
        base_input_key=default_input_to_key(inputs),
        trace=None,
        trace_id="",
        task_runs=runs,
    )


@pytest.mark.anyio
async def test_max_concurrency_produces_identical_results():
    # Two cases, multiple repeats each, a deterministic evaluator.
    cases = [
        Case(inputs="a", metadata=TestCaseMetadata(), evaluators=[RegexInOutputEvaluator(pattern="x", tags=set())]),
        Case(inputs="b", metadata=TestCaseMetadata(), evaluators=[RegexInOutputEvaluator(pattern="y", tags=set())]),
    ]
    testset = Dataset[str, str, TestCaseMetadata](cases=cases)
    run = DatasetRunOutput(cases=[_case_run("a", ["x", "nope", "x"]), _case_run("b", ["y", "y"])])

    seq = await evaluate_results(run, testset, max_concurrency=1)
    conc = await evaluate_results(run, testset, max_concurrency=4)

    def _summary(out):
        return [(cr.case_name, cr.aggregated.passed, round(cr.aggregated.pass_rate, 6)) for cr in out.case_results]

    assert _summary(seq) == _summary(conc)
    assert len(seq.runs) == len(conc.runs)


def test_llm_judge_has_optional_timeout_field():
    import dataclasses

    from ragpill.evaluators import LLMJudge

    fields = {f.name for f in dataclasses.fields(LLMJudge)}
    assert "timeout_s" in fields
