"""Integration tests for ``ragpill.execution.execute_dataset``.

These run end-to-end against a real local SQLite MLflow backend. No external
server is required — the module uses ``tempfile.mkdtemp`` internally for the
local-temp backend path.
"""

from __future__ import annotations

import glob
import os
import tempfile

import mlflow
import pytest

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import Case, Dataset
from ragpill.execution import DatasetRunOutput, execute_dataset


async def _echo_task(q: str) -> str:
    return f"echo:{q}"


def _case(inputs: str, repeat: int | None = None) -> Case[str, str, TestCaseMetadata]:
    return Case(inputs=inputs, metadata=TestCaseMetadata(repeat=repeat) if repeat else TestCaseMetadata())


# ---------------------------------------------------------------------------
# Local temp backend
# ---------------------------------------------------------------------------


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_local_temp_backend_creates_traces():
    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=True)
    assert isinstance(run_output, DatasetRunOutput)
    case_out = run_output.cases[0]
    # Temp backend is cleaned up — tracking_uri on the output is empty.
    assert run_output.tracking_uri == ""
    # Case-level trace captured
    assert case_out.trace is not None
    assert case_out.trace_id != ""
    # Each task run captured a span id (subtree-filtered trace)
    tr = case_out.task_runs[0]
    assert tr.run_span_id != ""
    assert tr.trace is not None


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_local_temp_backend_cleans_up_temp_dir():
    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
    # Snapshot temp dirs before
    tmp_root = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp_root, "ragpill_exec_*")))
    await execute_dataset(ds, task=_echo_task, capture_traces=True)
    after = set(glob.glob(os.path.join(tmp_root, "ragpill_exec_*")))
    assert before == after, f"Temp dirs leaked: {after - before}"


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_local_temp_backend_cleans_up_even_on_error():
    async def broken(q: str) -> str:
        raise RuntimeError(f"broken for {q}")

    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
    tmp_root = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp_root, "ragpill_exec_*")))
    run_output = await execute_dataset(ds, task=broken, capture_traces=True)
    after = set(glob.glob(os.path.join(tmp_root, "ragpill_exec_*")))
    assert before == after
    # Task error was captured, not raised.
    assert run_output.cases[0].task_runs[0].error is not None


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_repeat_3_produces_distinct_run_span_ids():
    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello", repeat=3)])
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=True)
    runs = run_output.cases[0].task_runs
    assert len(runs) == 3
    span_ids = [r.run_span_id for r in runs]
    assert all(sid for sid in span_ids)
    assert len(set(span_ids)) == 3


# ---------------------------------------------------------------------------
# Dual-backend: explicit URI (second temp sqlite used as the "server")
# ---------------------------------------------------------------------------


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_explicit_uri_traces_to_that_uri():
    # Create a second temp SQLite URI that will act as the "server".
    server_dir = tempfile.mkdtemp(prefix="ragpill_server_")
    server_uri = f"sqlite:///{os.path.join(server_dir, 'server.db')}"
    try:
        ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
        run_output = await execute_dataset(
            ds,
            task=_echo_task,
            capture_traces=True,
            mlflow_tracking_uri=server_uri,
        )
        assert run_output.tracking_uri == server_uri

        # Trace survived on the "server"
        mlflow.set_tracking_uri(server_uri)
        traces = mlflow.search_traces(return_type="list")
        assert len(traces) >= 1
    finally:
        import shutil

        shutil.rmtree(server_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Round trip: execute → to_json → from_json preserves trace data
# ---------------------------------------------------------------------------


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_json_roundtrip_preserves_trace_spans():
    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=True)
    payload = run_output.to_json()
    restored = DatasetRunOutput.from_json(payload)

    assert len(restored.cases) == 1
    orig_case = run_output.cases[0]
    new_case = restored.cases[0]
    assert new_case.trace is not None and orig_case.trace is not None
    assert len(new_case.trace.data.spans) == len(orig_case.trace.data.spans)
    # Span ids survive
    orig_ids = sorted(s.span_id for s in orig_case.trace.data.spans)
    new_ids = sorted(s.span_id for s in new_case.trace.data.spans)
    assert orig_ids == new_ids
