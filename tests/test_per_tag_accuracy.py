"""Unit tests for ``EvaluationOutput.per_tag_accuracy``."""

from __future__ import annotations

import math

import pandas as pd

from ragpill.types import EvaluationOutput


def _eval_output(rows: list[dict[str, object]]) -> EvaluationOutput:
    """Build an EvaluationOutput with only the ``runs`` DataFrame populated.

    The method under test only consults ``self.runs``, so we keep fixtures
    minimal — no case_results or cases DataFrame required.
    """
    return EvaluationOutput(runs=pd.DataFrame(rows), cases=pd.DataFrame(), case_results=[])


def test_returns_empty_when_runs_dataframe_is_empty():
    eo = EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=[])
    assert eo.per_tag_accuracy() == {}


def test_returns_empty_when_runs_has_no_evaluator_result_column():
    eo = EvaluationOutput(
        runs=pd.DataFrame([{"tags": {"a"}, "other": 1}]),
        cases=pd.DataFrame(),
        case_results=[],
    )
    assert eo.per_tag_accuracy() == {}


def test_returns_empty_when_all_results_are_nan():
    eo = _eval_output(
        [
            {"tags": {"a"}, "evaluator_result": math.nan},
            {"tags": {"b"}, "evaluator_result": math.nan},
        ]
    )
    assert eo.per_tag_accuracy() == {}


def test_two_tags_on_same_row_both_reported_with_same_value():
    eo = _eval_output(
        [
            {"tags": {"alpha", "beta"}, "evaluator_result": True},
            {"tags": {"alpha", "beta"}, "evaluator_result": False},
        ]
    )
    result = eo.per_tag_accuracy()
    assert set(result) == {"alpha", "beta"}
    assert result["alpha"] == 0.5
    assert result["beta"] == 0.5


def test_separate_tags_compute_independent_pass_rates():
    eo = _eval_output(
        [
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"beta"}, "evaluator_result": False},
            {"tags": {"beta"}, "evaluator_result": True},
        ]
    )
    result = eo.per_tag_accuracy()
    assert result["alpha"] == 1.0
    assert result["beta"] == 0.5


def test_untagged_rows_dropped_silently():
    eo = _eval_output(
        [
            {"tags": set(), "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": False},
        ]
    )
    result = eo.per_tag_accuracy()
    assert result == {"alpha": 0.5}


def test_nan_rows_excluded_from_pass_rate_denominator():
    eo = _eval_output(
        [
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": math.nan},
        ]
    )
    # Only the True row counts: 1/1, not 1/2.
    assert eo.per_tag_accuracy() == {"alpha": 1.0}


def test_numeric_evaluator_results_are_averaged_alongside_booleans():
    eo = _eval_output(
        [
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": 0.5},
        ]
    )
    # mean(1.0, 0.5) == 0.75
    assert eo.per_tag_accuracy() == {"alpha": 0.75}


def test_return_type_is_plain_dict_with_python_floats():
    eo = _eval_output([{"tags": {"alpha"}, "evaluator_result": True}])
    result = eo.per_tag_accuracy()
    assert isinstance(result, dict)
    [(tag, value)] = result.items()
    assert isinstance(tag, str)
    assert isinstance(value, float)


def test_accuracy_values_are_rounded_to_three_decimals():
    """Repeating decimals like 1/3 are rounded; not full-float precision."""
    eo = _eval_output(
        [
            {"tags": {"alpha"}, "evaluator_result": True},
            {"tags": {"alpha"}, "evaluator_result": False},
            {"tags": {"alpha"}, "evaluator_result": False},  # 1/3 -> 0.333
            {"tags": {"beta"}, "evaluator_result": True},
            {"tags": {"beta"}, "evaluator_result": True},
            {"tags": {"beta"}, "evaluator_result": False},  # 2/3 -> 0.667
        ]
    )
    assert eo.per_tag_accuracy() == {"alpha": 0.333, "beta": 0.667}
