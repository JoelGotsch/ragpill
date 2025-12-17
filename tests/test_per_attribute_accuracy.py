"""Unit tests for ``EvaluationOutput.per_attribute_accuracy`` / ``per_attribute_accuracy_all``."""

from __future__ import annotations

import pandas as pd

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.types import AggregatedResult, CaseResult, EvaluationOutput, RunResult


def _result(name: str, value: bool | float) -> EvaluationResult:
    return EvaluationResult(name=name, value=value, reason=None, source=EvaluatorSource(name=name))


def _run(idx: int, assertions: dict[str, EvaluationResult]) -> RunResult:
    return RunResult(
        run_index=idx,
        input_key=f"k{idx}",
        run_span_id="",
        output="out",
        duration=0.1,
        assertions=assertions,
    )


def _agg() -> AggregatedResult:
    return AggregatedResult(
        passed=True,
        pass_rate=1.0,
        threshold=0.8,
        summary="ok",
        per_evaluator_pass_rates={},
    )


def _case(base_key: str, attributes: dict[str, object], runs: list[RunResult]) -> CaseResult:
    return CaseResult(
        case_name=base_key,
        inputs="in",
        metadata=TestCaseMetadata(attributes=attributes),
        base_input_key=base_key,
        trace_id="",
        run_results=runs,
        aggregated=_agg(),
    )


def _eo(cases: list[CaseResult]) -> EvaluationOutput:
    return EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=cases)


def test_returns_empty_when_no_case_results() -> None:
    eo = _eo([])
    assert eo.per_attribute_accuracy("difficulty") == {}
    assert eo.per_attribute_accuracy_all() == {}


def test_returns_empty_when_attribute_absent_from_every_case() -> None:
    eo = _eo([_case("c1", {"domain": "chem"}, [_run(0, {"e": _result("e", True)})])])
    assert eo.per_attribute_accuracy("difficulty") == {}


def test_two_values_of_one_attribute_compute_independent_pass_rates() -> None:
    eo = _eo(
        [
            _case("c1", {"difficulty": "easy"}, [_run(0, {"e": _result("e", True)})]),
            _case("c2", {"difficulty": "easy"}, [_run(0, {"e": _result("e", True)})]),
            _case("c3", {"difficulty": "hard"}, [_run(0, {"e": _result("e", False)})]),
            _case("c4", {"difficulty": "hard"}, [_run(0, {"e": _result("e", True)})]),
        ]
    )
    result = eo.per_attribute_accuracy("difficulty")
    assert result == {"easy": 1.0, "hard": 0.5}


def test_cases_missing_the_attribute_are_skipped_not_counted_as_failures() -> None:
    eo = _eo(
        [
            _case("c1", {"difficulty": "easy"}, [_run(0, {"e": _result("e", True)})]),
            _case("c2", {}, [_run(0, {"e": _result("e", False)})]),  # no `difficulty` key
        ]
    )
    # c2 must not appear under any difficulty bucket and must not drag c1's score down.
    assert eo.per_attribute_accuracy("difficulty") == {"easy": 1.0}


def test_multiple_runs_per_case_each_contribute() -> None:
    eo = _eo(
        [
            _case(
                "c1",
                {"difficulty": "easy"},
                [
                    _run(0, {"e": _result("e", True)}),
                    _run(1, {"e": _result("e", False)}),
                ],
            ),
        ]
    )
    assert eo.per_attribute_accuracy("difficulty") == {"easy": 0.5}


def test_multiple_evaluators_per_run_each_contribute() -> None:
    eo = _eo(
        [
            _case(
                "c1",
                {"difficulty": "easy"},
                [_run(0, {"a": _result("a", True), "b": _result("b", False)})],
            ),
        ]
    )
    assert eo.per_attribute_accuracy("difficulty") == {"easy": 0.5}


def test_numeric_evaluator_results_are_averaged_alongside_booleans() -> None:
    eo = _eo(
        [
            _case(
                "c1",
                {"difficulty": "easy"},
                [_run(0, {"a": _result("a", True), "b": _result("b", 0.5)})],
            ),
        ]
    )
    # mean(1.0, 0.5) == 0.75
    assert eo.per_attribute_accuracy("difficulty") == {"easy": 0.75}


def test_unhashable_attribute_value_is_stringified_silently() -> None:
    # Attribute value is a list — Python normally can't use it as a dict key.
    eo = _eo([_case("c1", {"tags_list": ["a", "b"]}, [_run(0, {"e": _result("e", True)})])])
    result = eo.per_attribute_accuracy("tags_list")
    assert result == {"['a', 'b']": 1.0}


def test_int_and_bool_values_keep_their_typed_str_form() -> None:
    eo = _eo(
        [
            _case("c1", {"n": 3}, [_run(0, {"e": _result("e", True)})]),
            _case("c2", {"n": 4}, [_run(0, {"e": _result("e", False)})]),
        ]
    )
    assert eo.per_attribute_accuracy("n") == {"3": 1.0, "4": 0.0}


def test_per_attribute_accuracy_all_auto_discovers_keys() -> None:
    eo = _eo(
        [
            _case("c1", {"difficulty": "easy", "domain": "chem"}, [_run(0, {"e": _result("e", True)})]),
            _case("c2", {"difficulty": "hard", "domain": "bio"}, [_run(0, {"e": _result("e", False)})]),
        ]
    )
    out = eo.per_attribute_accuracy_all()
    assert set(out) == {"difficulty", "domain"}
    assert out["difficulty"] == {"easy": 1.0, "hard": 0.0}
    assert out["domain"] == {"chem": 1.0, "bio": 0.0}


def test_per_attribute_accuracy_all_drops_attributes_with_no_usable_rows() -> None:
    """An attribute whose every value-bucket is empty (no usable assertions) is omitted."""
    eo = _eo([_case("c1", {"empty_attr": "v"}, [_run(0, {})])])  # zero assertions
    assert eo.per_attribute_accuracy_all() == {}


def test_api_returns_full_float_precision() -> None:
    """The Python API gives callers raw means — rounding lives at upload/triage."""
    import math

    eo = _eo(
        [
            _case("c1", {"difficulty": "hard"}, [_run(0, {"e": _result("e", True)})]),
            _case("c2", {"difficulty": "hard"}, [_run(0, {"e": _result("e", False)})]),
            _case("c3", {"difficulty": "hard"}, [_run(0, {"e": _result("e", False)})]),
        ]
    )
    assert math.isclose(eo.per_attribute_accuracy("difficulty")["hard"], 1 / 3)
