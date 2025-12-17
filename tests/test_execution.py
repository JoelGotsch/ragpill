"""Unit tests for ``ragpill.execution`` — no MLflow server required."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from ragpill.backends import RunHandle, configure_backend, reset_backend
from ragpill.backends._types import CaseGroupingHandle
from ragpill.base import TestCaseMetadata
from ragpill.eval_types import Case, Dataset
from ragpill.execution import (
    CaseRunOutput,
    DatasetRunOutput,
    TaskRunOutput,
    execute_dataset,
)
from ragpill.trace import Document, Span, SpanKind, Trace


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
    from unittest.mock import MagicMock

    from ragpill.execution import _fetch_trace

    backend = MagicMock()
    # await_trace returns the (trace, stable) 2-tuple where the trace is
    # already the neutral ragpill.trace.Trace (the backend converts its own
    # native trace), so _fetch_trace passes the pair straight through.
    neutral = object()
    backend.await_trace.return_value = (neutral, True)
    trace, stable = _fetch_trace(backend, "exp-1", "run-1", "trace-1", timeout_s=7.0, poll_interval_s=0.25)
    assert trace is neutral
    assert stable is True
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
    from unittest.mock import MagicMock

    from ragpill.execution import _fetch_trace

    backend = MagicMock()
    # Deadline hit with nothing exported: (None, stable=False).
    backend.await_trace.return_value = (None, False)
    result = _fetch_trace(backend, "exp-1", "run-1", "trace-1", timeout_s=1.0, poll_interval_s=0.1)
    assert result == (None, False)


def test_setup_local_tracing_passes_none_uri_to_remote_backends():
    """Backends without a local file store (Langfuse/Phoenix) must not receive
    the MLflow-specific temp SQLite URI — they get None and fall back to their
    own env-derived destination."""
    from unittest.mock import MagicMock

    from ragpill.backends import RunHandle
    from ragpill.execution import _setup_tracing
    from ragpill.settings import TrackingSettings

    backend = MagicMock()
    backend.supports_local_file_store = False
    backend.get_tracking_uri.return_value = None
    backend.start_run.return_value = RunHandle(run_id="r", experiment_id="e")
    ctx = _setup_tracing(backend, None, TrackingSettings())
    (uri, _experiment), _ = backend.set_destination.call_args
    assert uri is None
    assert ctx.temp_dir is None


def test_setup_local_tracing_builds_temp_sqlite_for_file_store_backends():
    from unittest.mock import MagicMock

    from ragpill.backends import RunHandle
    from ragpill.execution import _setup_tracing, _teardown_tracing
    from ragpill.settings import TrackingSettings

    backend = MagicMock()
    backend.supports_local_file_store = True
    backend.get_tracking_uri.return_value = None
    backend.start_run.return_value = RunHandle(run_id="r", experiment_id="e")
    ctx = _setup_tracing(backend, None, TrackingSettings())
    (uri, _experiment), _ = backend.set_destination.call_args
    assert uri is not None and uri.startswith("sqlite:///")
    assert ctx.temp_dir is not None
    _teardown_tracing(backend, ctx)


# ---------------------------------------------------------------------------
# R7 — empty run_span_id must not blanket-error runs whose case trace exists
# ---------------------------------------------------------------------------


class _NoIdSpan:
    """Span handle that cannot provide ids — the SpanHandle protocol
    explicitly permits the empty string (the Langfuse adapter returns one)."""

    span_id = ""
    trace_id = ""

    def set_attribute(self, *a, **k):
        pass

    def set_inputs(self, *a, **k):
        pass

    def set_outputs(self, *a, **k):
        pass


class _EmptySpanIdBackend:
    """Span-mode backend whose per-run span handles have no span id but whose
    case-level trace fetch succeeds (with a retriever span carrying a doc)."""

    supports_local_file_store = False

    def get_tracking_uri(self):
        return None

    def set_tracking_uri(self, uri):
        pass

    def set_destination(self, uri, experiment_name):
        pass

    def autolog_pydantic_ai(self):
        pass

    def start_run(self, run_id=None, description=None):
        return RunHandle(run_id="r", experiment_id="e")

    def end_run(self):
        pass

    def is_run_active(self):
        return True

    @contextmanager
    def start_span(self, name, span_type, attributes=None):
        yield _NoIdSpan()

    @contextmanager
    def start_case_grouping(self, case_id, name, inputs=None, attributes=None):
        yield CaseGroupingHandle(mode="span", case_trace_id="case-trace")

    def await_trace(self, trace_id, *, run_id=None, experiment_id=None, timeout_s=10.0, poll_interval_s=0.5):
        root = Span(
            span_id="root",
            parent_id=None,
            trace_id="case-trace",
            name="root",
            kind=SpanKind.CHAIN,
            start_time_ns=0,
            end_time_ns=0,
        )
        retriever = Span(
            span_id="ret",
            parent_id="root",
            trace_id="case-trace",
            name="retriever",
            kind=SpanKind.RETRIEVER,
            start_time_ns=0,
            end_time_ns=0,
            documents=[Document(content="the needle is in here")],
        )
        return Trace(trace_id="case-trace", spans=[root, retriever]), True


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_empty_run_span_id_keeps_trace_ok_with_case_trace_fallback(anyio_backend):
    """A backend that can't provide per-run span ids must not blanket-error
    the run: the case trace was fetched, so trace_status stays "ok", the run
    trace stays None, and span evaluators fall back to the full case trace."""
    from ragpill.evaluation import evaluate_results
    from ragpill.evaluators import RegexInSourcesEvaluator

    configure_backend(_EmptySpanIdBackend)
    try:
        ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="needle")
        ds = Dataset(cases=[Case(inputs="q", metadata=TestCaseMetadata(), evaluators=[ev])])
        out = await execute_dataset(ds, task=_echo_task, capture_traces=True, tracking_uri="http://server")

        tr = out.cases[0].task_runs[0]
        assert tr.run_span_id == ""
        assert tr.trace_status == "ok"  # the trace IS available, just not run-scoped
        assert tr.trace is None  # no per-run subtree — evaluation falls back
        assert out.cases[0].trace is not None  # the case trace was captured

        eval_out = await evaluate_results(out, ds)
        rr = eval_out.case_results[0].run_results[0]
        # The span evaluator saw the full case trace: a real verdict, not a
        # TraceUnavailableError failure.
        assert rr.evaluator_failures == []
        assert len(rr.assertions) == 1
        assert next(iter(rr.assertions.values())).value is True
    finally:
        reset_backend()


# ---------------------------------------------------------------------------
# backend= parameter — an explicit backend bypasses the process registry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_explicit_backend_param_is_used_without_registry(anyio_backend):
    def _poisoned_factory():
        raise AssertionError("the process registry must not be consulted when backend= is passed")

    calls: list[str] = []

    class _RecordingBackend(_EmptySpanIdBackend):
        def start_run(self, run_id=None, description=None):
            calls.append("start_run")
            return RunHandle(run_id="r", experiment_id="e")

    configure_backend(_poisoned_factory)
    try:
        ds = _make_minimal_dataset()
        out = await execute_dataset(
            ds, task=_echo_task, capture_traces=True, tracking_uri="http://server", backend=_RecordingBackend()
        )
    finally:
        reset_backend()
    assert calls == ["start_run"]  # the explicit backend did the capture
    assert out.cases[0].task_runs[0].output == "echo:hello"
