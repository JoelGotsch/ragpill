"""Tests for ragpill.types (RunResult, EvaluationOutput)."""

import pandas as pd

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    RunResult,
)


def _make_assertion(name: str, value: bool, reason: str = "") -> EvaluationResult:
    return EvaluationResult(
        name=name,
        value=value,
        reason=reason,
        source=EvaluatorSource(name="CODE", arguments={"evaluation_name": "test"}),
    )


def _make_case_result(case_name: str, passed: bool, pass_rate: float, threshold: float) -> CaseResult:
    return CaseResult(
        case_name=case_name,
        inputs=case_name,
        metadata=TestCaseMetadata(),
        base_input_key=f"key_{case_name}",
        trace_id="trace1",
        run_results=[],
        aggregated=AggregatedResult(
            passed=passed,
            pass_rate=pass_rate,
            threshold=threshold,
            summary=f"{pass_rate}",
            per_evaluator_pass_rates={},
        ),
    )


# ---------------------------------------------------------------------------
# RunResult.all_passed
# ---------------------------------------------------------------------------


def test_all_passed_true():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s1",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True), "e2": _make_assertion("e2", True)},
    )
    assert rr.all_passed is True


def test_all_passed_false_assertion():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s1",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True), "e2": _make_assertion("e2", False)},
    )
    assert rr.all_passed is False


def test_all_passed_false_error():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s1",
        output=None,
        duration=0.0,
        assertions={"e1": _make_assertion("e1", True)},
        error=RuntimeError("boom"),
    )
    assert rr.all_passed is False


def test_all_passed_empty_assertions():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s1",
        output="out",
        duration=1.0,
        assertions={},
    )
    assert rr.all_passed is True


# ---------------------------------------------------------------------------
# EvaluationOutput.summary
# ---------------------------------------------------------------------------


def test_summary_property():
    cr1 = _make_case_result("A", True, 1.0, 0.6)
    cr2 = _make_case_result("B", False, 0.3, 0.6)
    eo = EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=[cr1, cr2])
    summary = eo.summary
    assert len(summary) == 2
    assert summary.iloc[0]["case_name"] == "A"
    assert summary.iloc[0]["passed"] == True  # noqa: E712
    assert summary.iloc[1]["case_name"] == "B"
    assert summary.iloc[1]["passed"] == False  # noqa: E712


def test_runs_and_cases_shapes():
    """When repeat > 1, runs has more rows than cases."""
    rr1 = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s1",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True)},
    )
    rr2 = RunResult(
        run_index=1,
        input_key="k_1",
        run_span_id="s2",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True)},
    )
    cr = CaseResult(
        case_name="A",
        inputs="A",
        metadata=TestCaseMetadata(),
        base_input_key="key_A",
        trace_id="t1",
        run_results=[rr1, rr2],
        aggregated=AggregatedResult(
            passed=True,
            pass_rate=1.0,
            threshold=0.5,
            summary="2/2",
            per_evaluator_pass_rates={"e1": 1.0},
        ),
    )
    runs_rows = [
        {"case_id": "key_A", "run_index": 0, "evaluator_name": "e1"},
        {"case_id": "key_A", "run_index": 1, "evaluator_name": "e1"},
    ]
    cases_rows = [
        {"case_id": "key_A", "evaluator_name": "e1", "pass_rate": 1.0},
    ]
    eo = EvaluationOutput(
        runs=pd.DataFrame(runs_rows),
        cases=pd.DataFrame(cases_rows),
        case_results=[cr],
    )
    assert len(eo.runs) == 2
    assert len(eo.cases) == 1
