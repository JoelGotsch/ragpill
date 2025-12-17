"""Regression tests for the round-2 review findings (F1-F8, plus cleanups).

Each test reproduces a specific finding and pins the fix.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
from conftest import make_span as _span

from ragpill.backends._common import to_unix_nano  # pyright: ignore[reportPrivateUsage]
from ragpill.base import TestCaseMetadata, default_input_to_key
from ragpill.eval_types import Case, Dataset, EvaluatorContext
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import RegexInSourcesEvaluator, TraceUnavailableError
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput, execute_dataset
from ragpill.trace import SpanKind, Trace

# ---------------------------------------------------------------------------
# F1 — a transient trace-fetch error must not destroy the whole run
# ---------------------------------------------------------------------------


class _FakeSpan:
    span_id = "s"
    trace_id = "t"

    def set_attribute(self, *a, **k):
        pass

    def set_inputs(self, *a, **k):
        pass

    def set_outputs(self, *a, **k):
        pass


@pytest.fixture
def _use_raising_backend(make_fake_backend):
    # The shared fake with an await_trace that raises (a transient 5xx), used
    # to prove execute_dataset survives and records the run as trace-unavailable.
    def _raise_503(*_a, **_kw):
        raise RuntimeError("503 Service Unavailable")

    make_fake_backend(await_trace=_raise_503)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_transient_trace_error_does_not_destroy_run(_use_raising_backend, anyio_backend):
    ran = []

    async def task(q):
        ran.append(q)
        return f"ans:{q}"

    ds = Dataset(cases=[Case(inputs="a", metadata=TestCaseMetadata()), Case(inputs="b", metadata=TestCaseMetadata())])
    # capture_traces=True -> the fetch will raise, but the run must survive.
    out = await execute_dataset(ds, task=task, capture_traces=True, tracking_uri="http://server")

    assert ran == ["a", "b"]  # both cases ran despite the trace-store 503
    assert len(out.cases) == 2
    for case in out.cases:
        tr = case.task_runs[0]
        assert tr.output == f"ans:{case.inputs}"  # output preserved
        assert tr.trace_status == "unavailable"  # recorded, not lost
        assert tr.trace is None


# ---------------------------------------------------------------------------
# F2 — task_timeout_s must work for sync (thread-offloaded) tasks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_sync_task_timeout_is_effective(anyio_backend):
    def blocking(_q):
        time.sleep(2.0)
        return "done"

    ds = Dataset(cases=[Case(inputs="q", metadata=TestCaseMetadata())])
    t0 = time.perf_counter()
    out = await execute_dataset(ds, task=blocking, capture_traces=False, task_timeout_s=0.1)
    elapsed = time.perf_counter() - t0

    tr = out.cases[0].task_runs[0]
    assert elapsed < 1.0  # released at the deadline, not after the full 2 s
    assert tr.error is not None and "TimeoutError" in tr.error


# ---------------------------------------------------------------------------
# F3 — identity guard must not reject valid saved runs for non-str inputs
# ---------------------------------------------------------------------------


class _StructuredInput:
    def __init__(self, q):
        self.q = q

    # No stable __repr__/__eq__ on purpose: str() embeds the memory address.


@pytest.mark.anyio
async def test_identity_guard_tolerates_unstable_str_inputs():
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="x")
    # The run was captured with one instance; "another process" reconstructs an
    # equivalent instance whose str() differs (different address).
    captured = _StructuredInput("hello")
    reconstructed = _StructuredInput("hello")
    dataset_run = DatasetRunOutput(
        cases=[
            CaseRunOutput(
                case_name="c",
                inputs=captured,
                expected_output=None,
                metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
                base_input_key=default_input_to_key(captured),
                trace=None,
                trace_id="",
                task_runs=[TaskRunOutput(run_index=0, input_key="k_0", output="out", duration=0.01)],
            )
        ]
    )
    testset = Dataset(cases=[Case(inputs=reconstructed, metadata=TestCaseMetadata(), evaluators=[ev])])
    # Must NOT raise: the key was never stable, so identity verification is skipped.
    out = await evaluate_results(dataset_run, testset)
    assert len(out.case_results) == 1


# ---------------------------------------------------------------------------
# F5 — to_unix_nano(NaT) must be None, not int64-min
# ---------------------------------------------------------------------------


def test_to_unix_nano_handles_nat():
    assert to_unix_nano(pd.NaT) is None
    assert to_unix_nano(None) is None
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    assert to_unix_nano(ts) == ts.value


# ---------------------------------------------------------------------------
# F7/F8 — partial outage: the case and per-evaluator surfaces agree
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_partial_outage_surfaces_agree():
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="anything")
    # 3 repeats: run 0 has a trace whose subtree is present (evaluator errors on
    # empty retrieval -> value False is a real verdict here). To model the F7
    # partial outage, give run 0 a usable trace and runs 1..2 unavailable traces.
    runs = []
    for i in range(3):
        tr = TaskRunOutput(run_index=i, input_key=f"k_{i}", output="out", duration=0.01, run_span_id=f"run-{i}")
        if i == 0:
            # Trace present with a retriever span containing the pattern -> passes.
            retr = _span("ret", "run-0", SpanKind.RETRIEVER)
            retr.documents = []  # no docs -> Sources returns False (a real verdict)
            tr.trace = Trace(trace_id="t", spans=[_span("run-0", None), retr])
            tr.trace_status = "ok"
        else:
            tr.trace_status = "unavailable"
        runs.append(tr)
    case_run = CaseRunOutput(
        case_name="c",
        inputs="q",
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": 3, "threshold": None},
        base_input_key=default_input_to_key("q"),
        trace=None,
        trace_id="",
        task_runs=runs,
    )
    testset = Dataset(cases=[Case(inputs="q", metadata=TestCaseMetadata(repeat=3, threshold=0.8), evaluators=[ev])])
    out = await evaluate_results(DatasetRunOutput(cases=[case_run]), testset)
    cr = out.case_results[0]

    # Only run 0 was evaluable (produced a verdict); runs 1..2 are error-state and
    # excluded from every denominator.
    assert cr.aggregated.error_counts  # the outage is surfaced, not hidden
    # The case pass_rate and the evaluator's pass_rate use the same (evaluable)
    # denominator, so they agree rather than one green + one red.
    ev_name = next(iter(cr.aggregated.per_evaluator_pass_rates))
    ev_rate = cr.aggregated.per_evaluator_pass_rates[ev_name]
    agg_passed = ev_rate >= cr.aggregated.threshold
    assert cr.aggregated.passed == agg_passed


@pytest.mark.anyio
async def test_full_outage_is_not_green():
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="x")
    tr = TaskRunOutput(run_index=0, input_key="k_0", output="out", duration=0.01, run_span_id="run-0")
    tr.trace_status = "unavailable"
    case_run = CaseRunOutput(
        case_name="c",
        inputs="q",
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
        base_input_key=default_input_to_key("q"),
        trace=None,
        trace_id="",
        task_runs=[tr],
    )
    testset = Dataset(cases=[Case(inputs="q", metadata=TestCaseMetadata(), evaluators=[ev])])
    out = await evaluate_results(DatasetRunOutput(cases=[case_run]), testset)
    cr = out.case_results[0]
    # A full trace outage must not read as a passing case.
    assert cr.aggregated.passed is False
    assert cr.run_results[0].is_error_state is True


def test_incomplete_trace_status_raises_before_scoring():
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="foo")
    # Root span only present (child retriever spans still in an un-flushed batch)
    # with trace_status="incomplete" -> must raise, not score False.
    trace = Trace(trace_id="t", spans=[_span("run-0", None)])
    ctx: EvaluatorContext = EvaluatorContext(
        name="c",
        inputs="i",
        metadata=None,
        expected_output=None,
        output="o",
        duration=0.0,
        trace=trace,
        run_span_id="run-0",
        trace_status="incomplete",
    )
    with pytest.raises(TraceUnavailableError):
        ev.get_trace(ctx)


def test_literal_quote_evaluator_is_picklable():
    """Round-2 F12/M11: the old placeholder lambda made it unpicklable."""
    import pickle

    from ragpill.evaluators import LiteralQuoteEvaluator

    ev = LiteralQuoteEvaluator(expected=True, tags={"q"})
    restored = pickle.loads(pickle.dumps(ev))
    assert isinstance(restored, LiteralQuoteEvaluator)
    assert restored.expected is True


# ---------------------------------------------------------------------------
# F9 — concurrent traced execute_dataset must not overlap (global URI safety)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_concurrent_traced_runs_are_serialized(anyio_backend, make_fake_backend):
    # Records the max number of traced runs active at once. With the tracking
    # lock, two concurrent execute_dataset calls must never overlap (depth <= 1).
    import anyio

    depth = {"active": 0, "max": 0}

    def _enter(uri, experiment_name):
        depth["active"] += 1
        depth["max"] = max(depth["max"], depth["active"])

    def _leave():
        depth["active"] -= 1

    make_fake_backend(on_set_destination=_enter, on_end_run=_leave)

    async def one(q):
        async def task(_):
            await anyio.sleep(0.02)  # give the other run a chance to overlap
            return "x"

        ds = Dataset(cases=[Case(inputs=q, metadata=TestCaseMetadata())])
        await execute_dataset(ds, task=task, capture_traces=True, tracking_uri=f"http://{q}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(one, "a")
        tg.start_soon(one, "b")

    assert depth["max"] == 1  # never two traced runs active at once


# ---------------------------------------------------------------------------
# F16 — max_concurrency parallelizes evaluators *within* a run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_max_concurrency_overlaps_evaluators_within_a_run(anyio_backend):
    import anyio as _anyio

    from ragpill.base import BaseEvaluator
    from ragpill.eval_types import EvaluationReason

    class _SlowEval(BaseEvaluator):
        @classmethod
        def get_serialization_name(cls) -> str:
            return f"Slow{id(cls) % 1000}"

        async def run(self, ctx: EvaluatorContext) -> EvaluationReason:  # type: ignore[type-arg]
            await _anyio.sleep(0.1)
            return EvaluationReason(value=True, reason="ok")

    # Distinct classes so their assertion names differ.
    evs = [type(f"S{i}", (_SlowEval,), {})(expected=True, tags=set()) for i in range(4)]
    case = Case(inputs="q", metadata=TestCaseMetadata(), evaluators=evs)
    testset = Dataset(cases=[case])
    run = DatasetRunOutput(
        cases=[
            CaseRunOutput(
                case_name="q",
                inputs="q",
                expected_output=None,
                metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
                base_input_key=default_input_to_key("q"),
                trace=None,
                trace_id="",
                task_runs=[TaskRunOutput(run_index=0, input_key="k_0", output="out", duration=0.0)],
            )
        ]
    )

    t0 = time.perf_counter()
    out = await evaluate_results(run, testset, max_concurrency=4)
    elapsed = time.perf_counter() - t0
    # 4 evaluators * 0.1s each: overlapped they finish in ~0.1s, not ~0.4s.
    assert elapsed < 0.3
    assert len(out.case_results[0].run_results[0].assertions) == 4


# ---------------------------------------------------------------------------
# F15 — judge-trace cleanup filters server-side by a real trace tag
# (validated against a local sqlite MLflow store — no server, no LLM needed)
# ---------------------------------------------------------------------------


def test_delete_judge_traces_against_real_sqlite_store(tmp_path):
    import mlflow

    from ragpill.backends import CaptureSpanKind
    from ragpill.backends._common import JUDGE_TRACE_TAG  # pyright: ignore[reportPrivateUsage]
    from ragpill.backends.mlflow_backend import MLflowBackend

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    backend = MLflowBackend()
    backend.set_destination(None, "f15_e2e")
    handle = backend.start_run()
    try:
        # A judge span carries the marker -> promoted to a trace tag.
        with backend.start_span("llm-judge", CaptureSpanKind.LLM, attributes={JUDGE_TRACE_TAG: True}) as s:
            s.set_outputs({"pass": True})
        # A plain task span -> no tag.
        with backend.start_span("task", CaptureSpanKind.TASK) as s:
            s.set_inputs({"q": "hi"})
    finally:
        backend.end_run()

    exp_id = backend.resolve_experiment_id("f15_e2e")

    def _names(traces):
        return {t.info.tags.get("mlflow.traceName", "") for t in traces}

    before = mlflow.search_traces(locations=[exp_id], run_id=handle.run_id, return_type="list")
    assert _names(before) == {"llm-judge", "task"}  # both captured under the run

    # The server-side tag filter returns only the judge trace; deleting it must
    # leave the task trace untouched.
    backend.delete_judge_traces(exp_id, handle.run_id)

    after = mlflow.search_traces(locations=[exp_id], run_id=handle.run_id, return_type="list")
    assert _names(after) == {"task"}  # judge trace deleted, task trace kept
