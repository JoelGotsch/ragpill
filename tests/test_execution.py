"""Unit tests for ``ragpill.execution`` — no MLflow server required."""

from __future__ import annotations

import pytest

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import Case, Dataset
from ragpill.execution import (
    CaseRunOutput,
    DatasetRunOutput,
    TaskRunOutput,
    execute_dataset,
)


def _make_minimal_dataset(repeat: int | None = None) -> Dataset[str, str, TestCaseMetadata]:
    meta = TestCaseMetadata(repeat=repeat) if repeat else TestCaseMetadata()
    case: Case[str, str, TestCaseMetadata] = Case(inputs="hello", metadata=meta)
    return Dataset[str, str, TestCaseMetadata](cases=[case])


# ---------------------------------------------------------------------------
# Dataclass shape tests
# ---------------------------------------------------------------------------


def test_task_run_output_shape():
    tr = TaskRunOutput(
        run_index=0,
        input_key="k_0",
        output="out",
        duration=0.01,
    )
    assert tr.trace is None
    assert tr.run_span_id == ""
    assert tr.error is None


def test_case_run_output_shape():
    cr = CaseRunOutput(
        case_name="c",
        inputs="i",
        expected_output=None,
        metadata={},
        base_input_key="k",
        trace=None,
        trace_id="",
        task_runs=[],
    )
    assert cr.task_runs == []


def test_dataset_run_output_defaults_to_empty():
    dr = DatasetRunOutput()
    assert dr.cases == []
    assert dr.tracking_uri == ""


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------


def test_to_json_and_from_json_round_trip_empty():
    dr = DatasetRunOutput()
    payload = dr.to_json()
    dr2 = DatasetRunOutput.from_json(payload)
    assert dr2.cases == []
    assert dr2.tracking_uri == ""


def test_to_json_round_trip_with_task_runs_but_no_traces():
    tr = TaskRunOutput(run_index=0, input_key="k_0", output="out", duration=0.5)
    cr = CaseRunOutput(
        case_name="c",
        inputs="hello",
        expected_output=None,
        metadata={"attributes": {"k": "v"}, "tags": []},
        base_input_key="k",
        trace=None,
        trace_id="",
        task_runs=[tr],
    )
    dr = DatasetRunOutput(cases=[cr])
    payload = dr.to_json()
    dr2 = DatasetRunOutput.from_json(payload)
    assert len(dr2.cases) == 1
    assert dr2.cases[0].task_runs[0].output == "out"
    assert dr2.cases[0].task_runs[0].duration == 0.5
    assert dr2.cases[0].metadata == {"attributes": {"k": "v"}, "tags": []}


# ---------------------------------------------------------------------------
# execute_dataset — no tracing
# ---------------------------------------------------------------------------


async def _echo_task(q: str) -> str:
    return f"echo:{q}"


@pytest.mark.anyio
async def test_execute_dataset_without_tracing_returns_outputs():
    ds = _make_minimal_dataset()
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=False)
    assert isinstance(run_output, DatasetRunOutput)
    assert len(run_output.cases) == 1
    case_out = run_output.cases[0]
    assert case_out.task_runs[0].output == "echo:hello"
    assert case_out.task_runs[0].trace is None
    assert case_out.task_runs[0].run_span_id == ""
    assert case_out.trace is None


@pytest.mark.anyio
async def test_execute_dataset_captures_task_error_without_raising():
    async def broken(q: str) -> str:
        raise RuntimeError(f"broken for {q}")

    ds = _make_minimal_dataset()
    run_output = await execute_dataset(ds, task=broken, capture_traces=False)
    tr = run_output.cases[0].task_runs[0]
    assert tr.output is None
    assert tr.error is not None
    assert "broken for hello" in tr.error


@pytest.mark.anyio
async def test_execute_dataset_respects_per_case_repeat():
    meta = TestCaseMetadata(repeat=3)
    case: Case[str, str, TestCaseMetadata] = Case(inputs="x", metadata=meta)
    ds = Dataset[str, str, TestCaseMetadata](cases=[case])
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=False)
    runs = run_output.cases[0].task_runs
    assert len(runs) == 3
    assert [r.run_index for r in runs] == [0, 1, 2]
    # input_keys differ per run
    assert len({r.input_key for r in runs}) == 3


@pytest.mark.anyio
async def test_execute_dataset_rejects_both_task_and_factory():
    ds = _make_minimal_dataset()
    with pytest.raises(ValueError):
        await execute_dataset(ds, task=_echo_task, task_factory=lambda: _echo_task)


@pytest.mark.anyio
async def test_execute_dataset_rejects_neither_task_nor_factory():
    ds = _make_minimal_dataset()
    with pytest.raises(ValueError):
        await execute_dataset(ds)
