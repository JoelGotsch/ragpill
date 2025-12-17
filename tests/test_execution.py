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


# ---------------------------------------------------------------------------
# _fetch_trace — delegates to the backend's polling await_trace (no fallback)
# ---------------------------------------------------------------------------


def test_fetch_trace_delegates_to_await_trace():
    from unittest.mock import MagicMock, patch

    from ragpill.execution import _fetch_trace

    backend = MagicMock()
    # await_trace already returns the neutral ragpill.trace.Trace (the backend
    # converts its own native trace), so _fetch_trace passes it straight through.
    neutral = object()
    backend.await_trace.return_value = neutral
    with patch("ragpill.execution.get_backend", return_value=backend):
        result = _fetch_trace("exp-1", "run-1", "trace-1", timeout_s=7.0, poll_interval_s=0.25)
    assert result is neutral
    backend.await_trace.assert_called_once_with(
        "trace-1",
        run_id="run-1",
        experiment_id="exp-1",
        timeout_s=7.0,
        poll_interval_s=0.25,
    )
    # No best-effort search_traces fallback that could return the wrong trace.
    backend.search_traces.assert_not_called()


def test_fetch_trace_returns_none_on_miss():
    from unittest.mock import MagicMock, patch

    from ragpill.execution import _fetch_trace

    backend = MagicMock()
    backend.await_trace.return_value = None
    with patch("ragpill.execution.get_backend", return_value=backend):
        result = _fetch_trace("exp-1", "run-1", "trace-1", timeout_s=1.0, poll_interval_s=0.1)
    assert result is None


def test_setup_local_tracing_passes_none_uri_to_remote_backends():
    """Backends without a local file store (Langfuse/Phoenix) must not receive
    the MLflow-specific temp SQLite URI — they get None and fall back to their
    own env-derived destination."""
    from unittest.mock import MagicMock, patch

    from ragpill.backends import RunHandle
    from ragpill.execution import _setup_tracing
    from ragpill.settings import TrackingSettings

    backend = MagicMock()
    backend.supports_local_file_store = False
    backend.get_tracking_uri.return_value = None
    backend.start_run.return_value = RunHandle(run_id="r", experiment_id="e")
    with patch("ragpill.execution.get_backend", return_value=backend):
        ctx = _setup_tracing(None, TrackingSettings())
    (uri, _experiment), _ = backend.set_destination.call_args
    assert uri is None
    assert ctx.temp_dir is None


def test_setup_local_tracing_builds_temp_sqlite_for_file_store_backends():
    from unittest.mock import MagicMock, patch

    from ragpill.backends import RunHandle
    from ragpill.execution import _setup_tracing, _teardown_tracing
    from ragpill.settings import TrackingSettings

    backend = MagicMock()
    backend.supports_local_file_store = True
    backend.get_tracking_uri.return_value = None
    backend.start_run.return_value = RunHandle(run_id="r", experiment_id="e")
    with patch("ragpill.execution.get_backend", return_value=backend):
        ctx = _setup_tracing(None, TrackingSettings())
        (uri, _experiment), _ = backend.set_destination.call_args
        assert uri is not None and uri.startswith("sqlite:///")
        assert ctx.temp_dir is not None
        _teardown_tracing(ctx)
