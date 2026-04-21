"""Tests for _create_runs_dataframe and _create_cases_dataframe."""

from pydantic_evals.evaluators.evaluator import EvaluationResult, EvaluatorSpec

from ragpill.base import TestCaseMetadata
from ragpill.mlflow_helper import _create_cases_dataframe, _create_runs_dataframe
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult


def _make_assertion(name: str, value: bool) -> EvaluationResult:
    return EvaluationResult(
        name=name,
        value=value,
        reason=f"{name} reason",
        source=EvaluatorSpec(name="CODE", arguments={"evaluation_name": "test"}),
    )


def _make_case_results() -> list[CaseResult]:
    """2 cases, 3 runs each, 2 evaluators -> 12 run rows, 4 cases rows."""
    results = []
    for case_idx in range(2):
        run_results = []
        for run_idx in range(3):
            rr = RunResult(
                run_index=run_idx,
                input_key=f"k{case_idx}_{run_idx}",
                run_span_id=f"s{case_idx}_{run_idx}",
                output=f"out_{case_idx}_{run_idx}",
                duration=1.0 + run_idx * 0.1,
                assertions={
                    "e1": _make_assertion("e1", True),
                    "e2": _make_assertion("e2", run_idx < 2),
                },
            )
            run_results.append(rr)

        cr = CaseResult(
            case_name=f"case_{case_idx}",
            inputs=f"input_{case_idx}",
            metadata=TestCaseMetadata(),
            base_input_key=f"k{case_idx}",
            trace_id=f"trace_{case_idx}",
            run_results=run_results,
            aggregated=AggregatedResult(
                passed=True,
                pass_rate=2 / 3,
                threshold=0.5,
                summary="2/3 passed",
                per_evaluator_pass_rates={"e1": 1.0, "e2": 2 / 3},
            ),
        )
        results.append(cr)
    return results


# ---------------------------------------------------------------------------
# _create_runs_dataframe
# ---------------------------------------------------------------------------


def test_runs_columns():
    df = _create_runs_dataframe(_make_case_results())
    expected_cols = {
        "case_id",
        "case_name",
        "run_index",
        "repeat_total",
        "threshold",
        "evaluator_name",
        "evaluator_result",
        "evaluator_reason",
        "inputs",
        "output",
        "input_key",
        "trace_id",
    }
    assert expected_cols.issubset(set(df.columns))


def test_runs_row_count():
    df = _create_runs_dataframe(_make_case_results())
    # 2 cases * 3 runs * 2 evaluators = 12 rows
    assert len(df) == 12


def test_runs_single_repeat():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s0",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True)},
    )
    cr = CaseResult(
        case_name="c",
        inputs="i",
        metadata=TestCaseMetadata(),
        base_input_key="k",
        trace_id="t",
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=True,
            pass_rate=1.0,
            threshold=1.0,
            summary="1/1",
            per_evaluator_pass_rates={"e1": 1.0},
        ),
    )
    df = _create_runs_dataframe([cr])
    assert all(df["run_index"] == 0)


def test_runs_input_key_format():
    df = _create_runs_dataframe(_make_case_results())
    for _, row in df.iterrows():
        assert f"_{int(row['run_index'])}" in str(row["input_key"])


# ---------------------------------------------------------------------------
# _create_cases_dataframe
# ---------------------------------------------------------------------------


def test_cases_columns():
    df = _create_cases_dataframe(_make_case_results())
    expected_cols = {"case_id", "case_name", "evaluator_name", "pass_rate", "passed", "threshold"}
    assert expected_cols.issubset(set(df.columns))


def test_cases_row_count():
    df = _create_cases_dataframe(_make_case_results())
    # 2 cases * 2 evaluators = 4 rows
    assert len(df) == 4


def test_cases_pass_rate_correct():
    df = _create_cases_dataframe(_make_case_results())
    e2_rows = df[df["evaluator_name"] == "e2"]
    # e2 passes on runs 0, 1 but fails on run 2 -> 2/3
    for _, row in e2_rows.iterrows():
        assert abs(row["pass_rate"] - 2 / 3) < 0.01


def test_cases_single_repeat_matches_runs():
    rr = RunResult(
        run_index=0,
        input_key="k_0",
        run_span_id="s0",
        output="out",
        duration=1.0,
        assertions={"e1": _make_assertion("e1", True)},
    )
    cr = CaseResult(
        case_name="c",
        inputs="i",
        metadata=TestCaseMetadata(),
        base_input_key="k",
        trace_id="t",
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=True,
            pass_rate=1.0,
            threshold=1.0,
            summary="1/1",
            per_evaluator_pass_rates={"e1": 1.0},
        ),
    )
    cases_df = _create_cases_dataframe([cr])
    runs_df = _create_runs_dataframe([cr])
    assert cases_df.iloc[0]["passed"] == runs_df.iloc[0]["evaluator_result"]


# ---------------------------------------------------------------------------
# EvaluationOutput.summary
# ---------------------------------------------------------------------------


def test_summary_one_row_per_case():
    case_results = _make_case_results()
    eo = EvaluationOutput(
        runs=_create_runs_dataframe(case_results),
        cases=_create_cases_dataframe(case_results),
        case_results=case_results,
    )
    summary = eo.summary
    assert len(summary) == 2


def test_summary_overall_passed():
    case_results = _make_case_results()
    eo = EvaluationOutput(
        runs=_create_runs_dataframe(case_results),
        cases=_create_cases_dataframe(case_results),
        case_results=case_results,
    )
    summary = eo.summary
    # Both cases have threshold=0.5, pass_rate=2/3 -> both pass
    assert all(summary["passed"])
