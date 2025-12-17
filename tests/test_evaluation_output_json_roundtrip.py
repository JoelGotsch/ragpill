"""Round-trip tests for ``EvaluationOutput.to_json`` / ``from_json``."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.entities import SpanType, Trace

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.trace import from_mlflow_trace
from ragpill.trace.model import Document, Span, SpanKind, Trace as NeutralTrace
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    EvaluatorFailureInfo,
    RunResult,
)


@pytest.fixture
def _isolated_mlflow_backend() -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    tmp = tempfile.mkdtemp(prefix="ragpill_eo_roundtrip_")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 'mlflow.db')}")
    mlflow.set_experiment(f"exp-{os.path.basename(tmp)}")
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _runs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"case_id": "c1", "evaluator_name": "E", "passed": True, "pass_rate": 1.0},
            {"case_id": "c1", "evaluator_name": "F", "passed": False, "pass_rate": 0.0},
        ]
    )


def _cases_df() -> pd.DataFrame:
    return pd.DataFrame([{"case_id": "c1", "evaluator_name": "E", "pass_rate": 1.0, "passed": True}])


def _build_case_results() -> list[CaseResult]:
    rr = RunResult(
        run_index=0,
        input_key="c1_0",
        run_span_id="span-1",
        output="hello",
        duration=0.42,
        assertions={
            "E": EvaluationResult(
                name="E",
                value=True,
                reason="ok",
                source=EvaluatorSource(name="E", arguments={"k": 1}, source_type="LLM_JUDGE"),
            ),
            "F": EvaluationResult(name="F", value=False, reason="bad", source=EvaluatorSource(name="F")),
        },
        evaluator_failures=[EvaluatorFailureInfo(name="X", error_message="boom", error_stacktrace="...")],
        error=ValueError("task crashed"),
    )
    cr = CaseResult(
        case_name="C1",
        inputs={"q": "hi"},
        metadata=TestCaseMetadata(threshold=0.5),
        base_input_key="c1",
        trace_id="t1",
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=False,
            pass_rate=0.5,
            threshold=0.5,
            summary="1/1 runs ok",
            per_evaluator_pass_rates={"E": 1.0, "F": 0.0},
        ),
    )
    return [cr]


def test_roundtrip_preserves_dataframes():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())
    pd.testing.assert_frame_equal(restored.runs.reset_index(drop=True), eo.runs.reset_index(drop=True))
    pd.testing.assert_frame_equal(restored.cases.reset_index(drop=True), eo.cases.reset_index(drop=True))


def test_roundtrip_preserves_case_results_structure():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())

    assert len(restored.case_results) == 1
    cr = restored.case_results[0]
    assert cr.base_input_key == "c1"
    assert cr.aggregated.pass_rate == 0.5
    assert cr.aggregated.per_evaluator_pass_rates == {"E": 1.0, "F": 0.0}
    rr = cr.run_results[0]
    assert rr.assertions["E"].value is True
    assert rr.assertions["F"].reason == "bad"
    assert rr.assertions["E"].source.arguments == {"k": 1}
    # source_type must survive the round trip (regression guard for the serde gap).
    assert rr.assertions["E"].source.source_type == "LLM_JUDGE"
    assert rr.assertions["F"].source.source_type == "CODE"
    assert len(rr.evaluator_failures) == 1
    assert rr.evaluator_failures[0].name == "X"
    # Exceptions become RuntimeError after round trip — content is preserved as a string.
    assert rr.error is not None
    assert "task crashed" in str(rr.error)


def test_roundtrip_with_dataset_run(_isolated_mlflow_backend: None) -> None:
    with mlflow.start_run():
        with mlflow.start_span(name="root", span_type=SpanType.AGENT) as root:
            root.set_inputs("hi")
            root.set_outputs("bye")
            run_span_id = root.span_id
    traces: list[Trace] = mlflow.search_traces(return_type="list", max_results=1)  # pyright: ignore[reportAssignmentType]
    trace = from_mlflow_trace(traces[0])
    case = CaseRunOutput(
        case_name="C1",
        inputs="hi",
        expected_output="bye",
        metadata={},
        base_input_key="c1",
        trace=trace,
        trace_id="t1",
        task_runs=[
            TaskRunOutput(
                run_index=0, input_key="c1_0", output="bye", duration=0.0, trace=trace, run_span_id=run_span_id
            )
        ],
    )
    dr = DatasetRunOutput(cases=[case], tracking_uri="x", run_id="r", experiment_id="e")
    eo = EvaluationOutput(
        runs=_runs_df(),
        cases=_cases_df(),
        case_results=_build_case_results(),
        dataset_run=dr,
    )
    restored = EvaluationOutput.from_json(eo.to_json())
    assert restored.dataset_run is not None
    assert restored.dataset_run.tracking_uri == "x"
    rcase = restored.dataset_run.cases[0]
    assert rcase.trace is not None
    orig_ids = sorted(s.span_id for s in trace.spans)
    new_ids = sorted(s.span_id for s in rcase.trace.spans)
    assert orig_ids == new_ids


def test_roundtrip_when_dataset_run_is_none():
    eo = EvaluationOutput(runs=_runs_df(), cases=_cases_df(), case_results=_build_case_results())
    restored = EvaluationOutput.from_json(eo.to_json())
    assert restored.dataset_run is None


# ---------------------------------------------------------------------------
# Golden wire-format contract (round-3 F11)
# ---------------------------------------------------------------------------
#
# ``tests/data/golden_evaluation_output_v3.json`` is a checked-in, human-reviewed
# ``EvaluationOutput`` serialization (run-JSON schema_version 3). The round-trip
# tests above push data through the same adapter in both directions, so a wire-
# format change is invisible to them; these tests pin the on-disk format itself
# against a fixed file. If one of them breaks, you changed the wire format:
# bump the schema version and announce the break (CHANGELOG / ADR) — do NOT
# silently regenerate the golden file to make the test pass. (This is the guard
# that would have caught the original ``source_type`` serde loss.)

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden_evaluation_output_v3.json"


def _golden_evaluation_output() -> EvaluationOutput:
    """Construct the exact object stored in the golden file.

    Shape: one case, two runs — run 0 passes with two assertions (one CODE, one
    LLM_JUDGE), run 1 is error-state (no verdicts, only an evaluator_failure) —
    plus the ADR-0018 aggregate fields and a dataset_run carrying a small
    neutral trace.
    """
    question = {"question": "What is the capital of France?"}
    answer = "Paris is the capital of France."
    passing_run = RunResult(
        run_index=0,
        input_key="case-1_0",
        run_span_id="span-run-0",
        output=answer,
        duration=1.25,
        assertions={
            "ContainsCapital": EvaluationResult(
                name="ContainsCapital",
                value=True,
                reason="found 'Paris' in the answer",
                source=EvaluatorSource(name="ContainsCapital", arguments={"check": "Paris"}, source_type="CODE"),
            ),
            "FaithfulnessJudge": EvaluationResult(
                name="FaithfulnessJudge",
                value=True,
                reason="answer is grounded in the retrieved context",
                source=EvaluatorSource(name="FaithfulnessJudge", source_type="LLM_JUDGE"),
            ),
        },
        trace_id="trace-run-0",
    )
    # Error-state run: no assertions, no task error, only evaluator failures —
    # excluded from pass_rate_evaluated and counted as a non-pass in pass_rate.
    error_run = RunResult(
        run_index=1,
        input_key="case-1_1",
        run_span_id="span-run-1",
        output=answer,
        duration=0.98,
        assertions={},
        evaluator_failures=[
            EvaluatorFailureInfo(
                name="ContainsCapital",
                error_message="TraceUnavailableError: trace was not exported before the deadline",
                error_stacktrace="Traceback (most recent call last):\n  ...\nTraceUnavailableError",
            ),
            EvaluatorFailureInfo(
                name="FaithfulnessJudge",
                error_message="TraceUnavailableError: trace was not exported before the deadline",
                error_stacktrace="Traceback (most recent call last):\n  ...\nTraceUnavailableError",
            ),
        ],
        trace_id="trace-run-1",
    )
    case_result = CaseResult(
        case_name="capital-of-france",
        inputs=question,
        metadata=TestCaseMetadata(attributes={"team": "geo"}, tags={"smoke"}, repeat=2, threshold=1.0),
        base_input_key="case-1",
        trace_id="",
        run_results=[passing_run, error_run],
        aggregated=AggregatedResult(
            passed=False,
            pass_rate=0.5,
            threshold=1.0,
            summary="1/2 runs passed (1 run could not be evaluated)",
            per_evaluator_pass_rates={"ContainsCapital": 1.0, "FaithfulnessJudge": 1.0},
            error_counts={"ContainsCapital": 1, "FaithfulnessJudge": 1},
            pass_rate_evaluated=1.0,
            runs_evaluated=1,
            runs_infra_error=1,
        ),
    )
    trace = NeutralTrace(
        trace_id="trace-run-0",
        session_id="case-1",
        dialect="mlflow",
        spans=[
            Span(
                span_id="span-run-0",
                parent_id=None,
                trace_id="trace-run-0",
                name="agent",
                kind=SpanKind.AGENT,
                start_time_ns=1_700_000_000_000_000_000,
                end_time_ns=1_700_000_001_250_000_000,
                status="OK",
                inputs=question,
                outputs=answer,
                dialect="mlflow",
            ),
            Span(
                span_id="span-retrieve",
                parent_id="span-run-0",
                trace_id="trace-run-0",
                name="retrieve",
                kind=SpanKind.RETRIEVER,
                start_time_ns=1_700_000_000_100_000_000,
                end_time_ns=1_700_000_000_400_000_000,
                status="OK",
                inputs="capital of France",
                documents=[
                    Document(
                        content="Paris is the capital and most populous city of France.",
                        id="doc-1",
                        score=0.93,
                        metadata={"source": "wiki"},
                    )
                ],
                dialect="mlflow",
            ),
        ],
    )
    dataset_run = DatasetRunOutput(
        cases=[
            CaseRunOutput(
                case_name="capital-of-france",
                inputs=question,
                expected_output="Paris",
                metadata={
                    "attributes": {"team": "geo"},
                    "tags": ["smoke"],
                    "expected": None,
                    "repeat": 2,
                    "threshold": 1.0,
                },
                base_input_key="case-1",
                trace=None,
                trace_id="",
                task_runs=[
                    TaskRunOutput(
                        run_index=0,
                        input_key="case-1_0",
                        output=answer,
                        duration=1.25,
                        trace=trace,
                        run_span_id="span-run-0",
                        trace_id="trace-run-0",
                        trace_status="ok",
                    ),
                    TaskRunOutput(
                        run_index=1,
                        input_key="case-1_1",
                        output=answer,
                        duration=0.98,
                        trace=None,
                        run_span_id="span-run-1",
                        trace_id="trace-run-1",
                        trace_status="unavailable",
                    ),
                ],
            )
        ],
        tracking_uri="http://tracking.example:5000",
        run_id="run-42",
        experiment_id="7",
    )
    runs = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "case_name": "capital-of-france",
                "run_index": 0,
                "evaluator_name": "ContainsCapital",
                "evaluator_result": True,
                "source_type": "CODE",
            },
            {
                "case_id": "case-1",
                "case_name": "capital-of-france",
                "run_index": 0,
                "evaluator_name": "FaithfulnessJudge",
                "evaluator_result": True,
                "source_type": "LLM_JUDGE",
            },
        ]
    )
    cases = pd.DataFrame(
        [
            {"case_id": "case-1", "evaluator_name": "ContainsCapital", "pass_rate": 1.0, "passed": True},
            {"case_id": "case-1", "evaluator_name": "FaithfulnessJudge", "pass_rate": 1.0, "passed": True},
        ]
    )
    return EvaluationOutput(runs=runs, cases=cases, case_results=[case_result], dataset_run=dataset_run)


def test_golden_file_deserializes_to_expected_object():
    loaded = EvaluationOutput.from_json(_GOLDEN_PATH.read_text())
    expected = _golden_evaluation_output()

    pd.testing.assert_frame_equal(loaded.runs.reset_index(drop=True), expected.runs.reset_index(drop=True))
    pd.testing.assert_frame_equal(loaded.cases.reset_index(drop=True), expected.cases.reset_index(drop=True))

    assert len(loaded.case_results) == 1
    got, want = loaded.case_results[0], expected.case_results[0]
    assert got.case_name == want.case_name
    assert got.inputs == want.inputs
    assert got.metadata == want.metadata  # pydantic model equality (tags set restored)
    assert got.base_input_key == want.base_input_key
    assert got.trace_id == want.trace_id
    assert got.aggregated == want.aggregated  # includes the ADR-0018 fields
    for got_rr, want_rr in zip(got.run_results, want.run_results, strict=True):
        assert got_rr.assertions == want_rr.assertions  # values, reasons, sources, source_type
        assert got_rr.evaluator_failures == want_rr.evaluator_failures
        assert got_rr.error is None
        assert (got_rr.run_index, got_rr.input_key, got_rr.run_span_id, got_rr.trace_id) == (
            want_rr.run_index,
            want_rr.input_key,
            want_rr.run_span_id,
            want_rr.trace_id,
        )
        assert (got_rr.output, got_rr.duration) == (want_rr.output, want_rr.duration)
    assert got.run_results[0].all_passed is True
    assert got.run_results[1].is_error_state is True

    assert loaded.dataset_run is not None and expected.dataset_run is not None
    got_dr, want_dr = loaded.dataset_run, expected.dataset_run
    assert (got_dr.tracking_uri, got_dr.run_id, got_dr.experiment_id) == (
        want_dr.tracking_uri,
        want_dr.run_id,
        want_dr.experiment_id,
    )
    # Dataclass equality covers the whole nested tree, including the neutral trace.
    assert got_dr.cases == want_dr.cases


def test_golden_file_reserializes_to_identical_wire_format():
    golden_text = _GOLDEN_PATH.read_text()
    loaded = EvaluationOutput.from_json(golden_text)
    # Compare parsed JSON (not strings): key order / whitespace are not part of
    # the wire contract, field names and values are.
    assert json.loads(loaded.to_json()) == json.loads(golden_text)
