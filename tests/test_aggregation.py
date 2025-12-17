"""Tests for _aggregate_runs() aggregation logic."""

from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.mlflow_helper import _aggregate_runs
from ragpill.types import RunResult


def _make_assertion(name: str, value: bool, reason: str = "") -> EvaluationResult:
    return EvaluationResult(
        name=name,
        value=value,
        reason=reason,
        source=EvaluatorSource(name="CODE", arguments={"evaluation_name": "test"}),
    )


def _make_run(index: int, passed: bool, error: Exception | None = None) -> RunResult:
    """Make a simple RunResult with one evaluator that passes or fails."""
    return RunResult(
        run_index=index,
        input_key=f"k_{index}",
        run_span_id=f"s{index}",
        output="out" if not error else None,
        duration=1.0,
        assertions={"eval1": _make_assertion("eval1", passed)},
        error=error,
    )


def _make_run_multi_eval(index: int, eval_results: dict[str, bool]) -> RunResult:
    """Make a RunResult with multiple evaluators."""
    assertions = {name: _make_assertion(name, val) for name, val in eval_results.items()}
    return RunResult(
        run_index=index,
        input_key=f"k_{index}",
        run_span_id=f"s{index}",
        output="out",
        duration=1.0,
        assertions=assertions,
    )


# ---------------------------------------------------------------------------
# Basic aggregation
# ---------------------------------------------------------------------------


def test_all_pass():
    runs = [_make_run(i, True) for i in range(3)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert result.passed is True
    assert result.pass_rate == 1.0


def test_all_fail():
    runs = [_make_run(i, False) for i in range(3)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert result.passed is False
    assert result.pass_rate == 0.0


def test_partial_above_threshold():
    runs = [_make_run(0, True), _make_run(1, True), _make_run(2, False)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert result.passed is True
    assert abs(result.pass_rate - 2 / 3) < 0.01


def test_partial_below_threshold():
    runs = [_make_run(0, True), _make_run(1, False), _make_run(2, False)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert result.passed is False
    assert abs(result.pass_rate - 1 / 3) < 0.01


def test_exact_threshold():
    runs = [_make_run(0, True), _make_run(1, True), _make_run(2, True), _make_run(3, False), _make_run(4, False)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert result.passed is True  # 0.6 >= 0.6


def test_just_below_threshold():
    runs = [_make_run(0, True), _make_run(1, True), _make_run(2, False), _make_run(3, False), _make_run(4, False)]
    result = _aggregate_runs(runs, threshold=0.5)
    assert result.passed is False  # 0.4 < 0.5


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------


def test_threshold_zero():
    runs = [_make_run(0, False), _make_run(1, False)]
    result = _aggregate_runs(runs, threshold=0.0)
    assert result.passed is True  # 0.0 >= 0.0


def test_threshold_one():
    runs = [_make_run(0, True), _make_run(1, False)]
    result = _aggregate_runs(runs, threshold=1.0)
    assert result.passed is False  # 0.5 < 1.0


def test_threshold_one_all_pass():
    runs = [_make_run(0, True), _make_run(1, True)]
    result = _aggregate_runs(runs, threshold=1.0)
    assert result.passed is True


def test_single_run_pass():
    result = _aggregate_runs([_make_run(0, True)], threshold=1.0)
    assert result.passed is True


def test_single_run_fail():
    result = _aggregate_runs([_make_run(0, False)], threshold=1.0)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Per-evaluator pass rates
# ---------------------------------------------------------------------------


def test_per_evaluator_rates():
    runs = [
        _make_run_multi_eval(0, {"e1": True, "e2": True}),
        _make_run_multi_eval(1, {"e1": True, "e2": False}),
        _make_run_multi_eval(2, {"e1": False, "e2": False}),
    ]
    result = _aggregate_runs(runs, threshold=0.5)
    assert abs(result.per_evaluator_pass_rates["e1"] - 2 / 3) < 0.01
    assert abs(result.per_evaluator_pass_rates["e2"] - 1 / 3) < 0.01


def test_per_evaluator_one_always_fails():
    runs = [
        _make_run_multi_eval(0, {"e1": True, "e2": False}),
        _make_run_multi_eval(1, {"e1": True, "e2": False}),
    ]
    result = _aggregate_runs(runs, threshold=0.5)
    assert result.per_evaluator_pass_rates["e1"] == 1.0
    assert result.per_evaluator_pass_rates["e2"] == 0.0


# ---------------------------------------------------------------------------
# Task errors
# ---------------------------------------------------------------------------


def test_with_task_error():
    runs = [
        _make_run(0, True),
        _make_run(1, False, error=RuntimeError("boom")),
        _make_run(2, True),
    ]
    result = _aggregate_runs(runs, threshold=0.5)
    # run 1 has error -> all_passed is False, so 2/3 pass
    assert abs(result.pass_rate - 2 / 3) < 0.01
    assert result.passed is True


def test_all_task_errors():
    runs = [_make_run(i, False, error=RuntimeError("boom")) for i in range(3)]
    result = _aggregate_runs(runs, threshold=0.5)
    assert result.pass_rate == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def test_summary_passed_includes_ratio():
    runs = [_make_run(0, True), _make_run(1, True), _make_run(2, False)]
    result = _aggregate_runs(runs, threshold=0.6)
    assert "2/3" in result.summary
    assert "passed" in result.summary


def test_summary_failed_includes_details():
    runs = [_make_run(0, True), _make_run(1, False), _make_run(2, False)]
    result = _aggregate_runs(runs, threshold=0.8)
    assert "1/3" in result.summary
    assert "Failed" in result.summary
    assert "run-1" in result.summary
