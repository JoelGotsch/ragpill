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
    # Each task run captured its own span (session mode: own trace; span
    # mode: subtree-filtered case trace).
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
    """In session mode each per-repeat trace is its own top-level trace; in
    span mode the case-level trace is filtered per run. Either way the
    per-run trace survives the JSON round-trip."""
    ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello")])
    run_output = await execute_dataset(ds, task=_echo_task, capture_traces=True)
    payload = run_output.to_json()
    restored = DatasetRunOutput.from_json(payload)

    assert len(restored.cases) == 1
    orig_run = run_output.cases[0].task_runs[0]
    new_run = restored.cases[0].task_runs[0]
    assert new_run.trace is not None and orig_run.trace is not None
    assert len(new_run.trace.data.spans) == len(orig_run.trace.data.spans)
    # Span ids survive
    orig_ids = sorted(s.span_id for s in orig_run.trace.data.spans)
    new_ids = sorted(s.span_id for s in new_run.trace.data.spans)
    assert orig_ids == new_ids
    # The trace_id captured at span-open is preserved too.
    assert new_run.trace_id == orig_run.trace_id


# ---------------------------------------------------------------------------
# Sessions UI grouping — mlflow.trace.session metadata
# ---------------------------------------------------------------------------


@pytest.mark.anyio(backends=["asyncio"])
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_repeats_share_mlflow_session_id():
    """In session mode each repeat opens as its own top-level trace, tagged
    with ``mlflow.trace.session`` = case base_input_key so the Sessions UI
    groups repeats of one case as turns of one session."""
    server_dir = tempfile.mkdtemp(prefix="ragpill_server_")
    server_uri = f"sqlite:///{os.path.join(server_dir, 'server.db')}"
    try:
        ds = Dataset[str, str, TestCaseMetadata](cases=[_case("hello", repeat=3)])
        run_output = await execute_dataset(
            ds,
            task=_echo_task,
            capture_traces=True,
            mlflow_tracking_uri=server_uri,
        )
        runs = run_output.cases[0].task_runs
        assert len(runs) == 3
        # Every repeat has its own trace_id (top-level traces, not nested).
        trace_ids = [r.trace_id for r in runs]
        assert all(tid for tid in trace_ids)
        assert len(set(trace_ids)) == 3

        # Each per-repeat trace carries the same session metadata, equal to
        # the case base_input_key.
        mlflow.set_tracking_uri(server_uri)
        expected_session_id = run_output.cases[0].base_input_key
        seen_sessions: set[str] = set()
        for tid in trace_ids:
            from mlflow import MlflowClient

            trace = MlflowClient().get_trace(tid)
            assert trace is not None
            session_md = trace.info.trace_metadata.get("mlflow.trace.session")
            assert session_md == expected_session_id, (
                f"trace {tid}: expected session {expected_session_id!r}, got {session_md!r}"
            )
            seen_sessions.add(session_md)
        assert seen_sessions == {expected_session_id}
    finally:
        import shutil

        shutil.rmtree(server_dir, ignore_errors=True)
