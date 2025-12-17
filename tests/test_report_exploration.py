"""Unit tests for ``ragpill.report.exploration`` — DatasetRunOutput markdown view."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import mlflow
import pytest
from mlflow.entities import SpanType, Trace

from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.report.exploration import render_dataset_run_as_exploration


@pytest.fixture(autouse=True)
def _isolated_mlflow_backend() -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    tmp = tempfile.mkdtemp(prefix="ragpill_report_explore_")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 'mlflow.db')}")
    mlflow.set_experiment(f"exp-{os.path.basename(tmp)}")
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _trace_with_two_spans() -> tuple[Trace, str]:
    with mlflow.start_run():
        with mlflow.start_span(name="root", span_type=SpanType.AGENT) as root:
            root.set_inputs("hi")
            root.set_outputs("bye")
            root_id = root.span_id
            with mlflow.start_span(name="llm", span_type=SpanType.LLM) as c:
                c.set_inputs("prompt")
                c.set_outputs("out")
    traces: list[Trace] = mlflow.search_traces(return_type="list", max_results=1)  # pyright: ignore[reportAssignmentType]
    return traces[0], root_id


def _make_dataset_run(with_trace: bool = True) -> DatasetRunOutput:
    if with_trace:
        trace, root_id = _trace_with_two_spans()
    else:
        trace, root_id = None, ""
    case = CaseRunOutput(
        case_name="example case",
        inputs="what is q?",
        expected_output="answer",
        metadata={},
        base_input_key="case-1",
        trace=trace,
        trace_id="t-1",
        task_runs=[
            TaskRunOutput(
                run_index=0,
                input_key="case-1_0",
                output="answer",
                duration=1.234,
                trace=trace,
                run_span_id=root_id,
            )
        ],
    )
    return DatasetRunOutput(cases=[case], tracking_uri="sqlite:///x.db", mlflow_run_id="r1", mlflow_experiment_id="e1")


def test_header_lists_run_metadata():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=False))
    assert "# Dataset run" in out
    assert "Cases: 1" in out
    assert "Tracking URI: sqlite:///x.db" in out
    assert "MLflow run: r1" in out
    assert "MLflow experiment: e1" in out


def test_per_case_section_includes_inputs_and_expected():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=False))
    assert "## Case `case-1`: example case" in out
    assert "Inputs: what is q?" in out
    assert "Expected output: answer" in out
    assert "Runs: 1" in out


def test_per_run_section_includes_duration_and_output():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=False))
    assert "### Run 0 (duration=1.23s)" in out
    assert "Output: answer" in out


def test_trace_tree_rendered_when_include_spans_true():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=True))
    assert "root (AGENT" in out
    assert "llm (LLM" in out
    # The LLM span is one level deeper than the root.
    lines = out.splitlines()
    root_line = next(line for line in lines if "root (AGENT" in line)
    llm_line = next(line for line in lines if "llm (LLM" in line)
    assert root_line.lstrip(" -") == "root" + root_line.lstrip(" -")[len("root") :]
    assert (len(llm_line) - len(llm_line.lstrip(" "))) > (len(root_line) - len(root_line.lstrip(" ")))


def test_include_spans_false_skips_trace():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=True), include_spans=False)
    assert "AGENT" not in out
    assert "LLM" not in out


def test_total_max_chars_is_respected():
    out = render_dataset_run_as_exploration(_make_dataset_run(with_trace=True), max_chars=200)
    assert len(out) <= 200
