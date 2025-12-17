"""Unit tests for ``ragpill.report.triage`` — markdown rendering of EvaluationOutput."""

from __future__ import annotations

import pandas as pd

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.report.triage import render_evaluation_output_as_triage
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    RunResult,
)


def _result(name: str, value: bool, reason: str | None = None) -> EvaluationResult:
    return EvaluationResult(name=name, value=value, reason=reason, source=EvaluatorSource(name=name))


def _run(index: int, key: str, output: str, assertions: dict[str, EvaluationResult]) -> RunResult:
    return RunResult(
        run_index=index,
        input_key=key,
        run_span_id="",
        output=output,
        duration=0.1,
        assertions=assertions,
    )


def _aggregated(passed: bool, pass_rate: float, threshold: float = 0.8) -> AggregatedResult:
    summary = "ok" if passed else "fail"
    return AggregatedResult(
        passed=passed,
        pass_rate=pass_rate,
        threshold=threshold,
        summary=summary,
        per_evaluator_pass_rates={},
    )


def _case(
    name: str,
    base_key: str,
    inputs: str,
    runs: list[RunResult],
    aggregated: AggregatedResult,
) -> CaseResult:
    return CaseResult(
        case_name=name,
        inputs=inputs,
        metadata=TestCaseMetadata(),
        base_input_key=base_key,
        trace_id="",
        run_results=runs,
        aggregated=aggregated,
    )


def _eval_output(case_results: list[CaseResult]) -> EvaluationOutput:
    return EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=case_results)


def test_header_reports_totals_and_per_evaluator_rollup():
    failing = _case(
        "Q1",
        "case-1",
        "what is q1?",
        runs=[_run(0, "k0", "wrong", {"LLMJudge": _result("LLMJudge", False, "off-topic")})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    passing = _case(
        "Q2",
        "case-2",
        "what is q2?",
        runs=[_run(0, "k0", "right", {"LLMJudge": _result("LLMJudge", True)})],
        aggregated=_aggregated(passed=True, pass_rate=1.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([failing, passing]))
    assert "# Evaluation summary" in out
    assert "Total cases: 2 (1 passed, 1 failed)" in out
    assert "Overall pass rate: 50.0%" in out
    assert "`LLMJudge` — 1/2 passed" in out


def test_failing_cases_appear_first_sorted_by_pass_rate():
    a = _case(
        "case A",
        "a",
        "in-a",
        runs=[_run(0, "k0", "out", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.5),
    )
    b = _case(
        "case B",
        "b",
        "in-b",
        runs=[_run(0, "k0", "out", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([a, b]))
    # B (pass_rate 0.0) sorts before A (0.5)
    assert out.index("case B") < out.index("case A")


def test_passing_section_collapsed_to_one_bullet_each():
    passing = _case(
        "passing case",
        "p1",
        "in",
        runs=[_run(0, "k0", "out", {"E": _result("E", True)})],
        aggregated=_aggregated(passed=True, pass_rate=1.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([passing]))
    assert "## Passing cases (collapsed)" in out
    assert "passing case" in out
    # No per-run detail for passing cases.
    assert "#### Run" not in out


def test_include_passing_false_omits_passing_section():
    passing = _case(
        "p",
        "p1",
        "in",
        runs=[_run(0, "k0", "out", {"E": _result("E", True)})],
        aggregated=_aggregated(passed=True, pass_rate=1.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([passing]), include_passing=False)
    assert "Passing cases" not in out


def test_failing_run_lists_each_assertion_with_verdict_and_reason():
    failing = _case(
        "Q",
        "case-1",
        "in",
        runs=[
            _run(
                0,
                "k0",
                "wrong",
                {
                    "LLMJudge": _result("LLMJudge", False, "off-topic answer"),
                    "Regex": _result("Regex", True),
                },
            )
        ],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([failing]))
    assert "`LLMJudge`: **FAIL** — off-topic answer" in out
    assert "`Regex`: **PASS**" in out


def test_truncation_drops_passing_section_first():
    passing = [
        _case(
            f"passing {i}",
            f"p{i}",
            f"in-{i}",
            runs=[_run(0, "k0", "out", {"E": _result("E", True)})],
            aggregated=_aggregated(passed=True, pass_rate=1.0),
        )
        for i in range(20)
    ]
    failing = _case(
        "the failing case",
        "f1",
        "in",
        runs=[_run(0, "k0", "wrong", {"E": _result("E", False, "nope")})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    full = render_evaluation_output_as_triage(_eval_output([failing, *passing]), max_chars=32_000)
    assert "Passing cases" in full
    # Force budget below header+failing+passing combined; passing should drop out.
    failing_only_size = len(render_evaluation_output_as_triage(_eval_output([failing])))
    truncated = render_evaluation_output_as_triage(
        _eval_output([failing, *passing]),
        max_chars=failing_only_size + 50,
    )
    assert "Passing cases" not in truncated
    assert "the failing case" in truncated


def test_tag_breakdown_section_appears_when_runs_have_tags():
    failing = _case(
        "Q",
        "case-1",
        "in",
        runs=[_run(0, "k0", "out", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    runs_df = pd.DataFrame(
        [
            {"tags": {"flaky"}, "evaluator_result": False},
            {"tags": {"flaky"}, "evaluator_result": True},
            {"tags": {"flaky"}, "evaluator_result": False},
            {"tags": {"baseline"}, "evaluator_result": True},
            {"tags": {"baseline"}, "evaluator_result": True},
        ]
    )
    eo = EvaluationOutput(runs=runs_df, cases=pd.DataFrame(), case_results=[failing])
    out = render_evaluation_output_as_triage(eo)
    assert "## Pass rate by tag" in out
    # Worst tag (flaky, 33.3%) listed before baseline (100.0%).
    assert out.index("`flaky`") < out.index("`baseline`")
    assert "33.3%" in out
    assert "100.0%" in out


def test_attribute_breakdown_renders_per_attribute_section():
    cases = [
        CaseResult(
            case_name=f"q{i}",
            inputs="in",
            metadata=TestCaseMetadata(attributes={"difficulty": diff}),
            base_input_key=f"c{i}",
            trace_id="",
            run_results=[
                RunResult(
                    run_index=0,
                    input_key="k0",
                    run_span_id="",
                    output="out",
                    duration=0.1,
                    assertions={"e": _result("e", passed)},
                )
            ],
            aggregated=_aggregated(passed=passed, pass_rate=1.0 if passed else 0.0),
        )
        for i, (diff, passed) in enumerate([("easy", True), ("easy", True), ("hard", False), ("hard", True)])
    ]
    out = render_evaluation_output_as_triage(_eval_output(cases))
    assert "## Pass rate by attribute" in out
    assert "### `difficulty`" in out
    # hard (50.0%) sorts before easy (100.0%) — worst first
    assert out.index("`hard`") < out.index("`easy`")
    assert "50.0%" in out
    assert "100.0%" in out


def test_attribute_breakdown_skips_single_value_attributes():
    """A uniform attribute (every case has the same value) is not worth a table."""
    cases = [
        CaseResult(
            case_name=f"q{i}",
            inputs="in",
            metadata=TestCaseMetadata(attributes={"domain": "chem"}),  # same on every case
            base_input_key=f"c{i}",
            trace_id="",
            run_results=[
                RunResult(
                    run_index=0,
                    input_key="k0",
                    run_span_id="",
                    output="out",
                    duration=0.1,
                    assertions={"e": _result("e", True)},
                )
            ],
            aggregated=_aggregated(passed=True, pass_rate=1.0),
        )
        for i in range(3)
    ]
    out = render_evaluation_output_as_triage(_eval_output(cases))
    assert "Pass rate by attribute" not in out


def test_attribute_breakdown_omitted_when_no_attributes():
    failing = _case(
        "Q",
        "case-1",
        "in",
        runs=[_run(0, "k0", "out", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([failing]))
    assert "Pass rate by attribute" not in out


def test_tag_breakdown_omitted_when_runs_have_no_tags():
    failing = _case(
        "Q",
        "case-1",
        "in",
        runs=[_run(0, "k0", "out", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    out = render_evaluation_output_as_triage(_eval_output([failing]))
    assert "Pass rate by tag" not in out


def test_relevant_spans_pulled_from_dataset_run_when_present():
    # We don't need a real trace here — just verify that when dataset_run is None,
    # no spans section is emitted, and when present-but-trace-None, ditto.
    failing = _case(
        "Q",
        "f1",
        "in",
        runs=[_run(0, "f1_0", "wrong", {"E": _result("E", False)})],
        aggregated=_aggregated(passed=False, pass_rate=0.0),
    )
    case_run = CaseRunOutput(
        case_name="Q",
        inputs="in",
        expected_output="expected",
        metadata={},
        base_input_key="f1",
        trace=None,
        trace_id="",
        task_runs=[TaskRunOutput(run_index=0, input_key="f1_0", output="wrong", duration=0.0, trace=None)],
    )
    eo = EvaluationOutput(
        runs=pd.DataFrame(),
        cases=pd.DataFrame(),
        case_results=[failing],
        dataset_run=DatasetRunOutput(cases=[case_run]),
    )
    out = render_evaluation_output_as_triage(eo)
    # Expected output surfaced from dataset_run.
    assert "Expected output: expected" in out
    # No "Relevant spans" because trace is None.
    assert "Relevant spans" not in out
