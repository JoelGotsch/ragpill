"""Phase 4: opt-in timeouts and bounded-concurrency evaluation.

Timeouts are caller-specified (never imposed by ragpill), and raising the
evaluation concurrency must not change results. Traced execution and upload
serialize process-wide on the shared tracking-state lock (round-3 R2).
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager

import anyio
import pandas as pd
import pytest

from ragpill.backends import RunHandle, configure_backend, reset_backend
from ragpill.backends._types import CaseGroupingHandle
from ragpill.base import TestCaseMetadata, default_input_to_key
from ragpill.eval_types import Case, Dataset
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import RegexInOutputEvaluator
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput, execute_dataset


@pytest.mark.anyio
async def test_task_timeout_records_timeout_error():
    async def slow(_q: str) -> str:
        await anyio.sleep(1.0)
        return "done"

    ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="q", metadata=TestCaseMetadata())])
    out = await execute_dataset(ds, task=slow, capture_traces=False, task_timeout_s=0.05)
    tr = out.cases[0].task_runs[0]
    assert tr.error is not None
    assert "TimeoutError" in tr.error


@pytest.mark.anyio
async def test_no_timeout_by_default():
    # Without task_timeout_s, a slow-ish task completes — ragpill imposes no budget.
    async def slowish(q: str) -> str:
        await anyio.sleep(0.05)
        return q

    ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="q", metadata=TestCaseMetadata())])
    out = await execute_dataset(ds, task=slowish, capture_traces=False)
    assert out.cases[0].task_runs[0].output == "q"
    assert out.cases[0].task_runs[0].error is None


def _case_run(inputs: str, outputs: list[str]) -> CaseRunOutput:
    runs = [TaskRunOutput(run_index=i, input_key=f"k_{i}", output=o, duration=0.01) for i, o in enumerate(outputs)]
    return CaseRunOutput(
        case_name=inputs,
        inputs=inputs,
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
        base_input_key=default_input_to_key(inputs),
        trace=None,
        trace_id="",
        task_runs=runs,
    )


@pytest.mark.anyio
async def test_max_concurrency_produces_identical_results():
    # Two cases, multiple repeats each, a deterministic evaluator.
    cases = [
        Case(inputs="a", metadata=TestCaseMetadata(), evaluators=[RegexInOutputEvaluator(pattern="x", tags=set())]),
        Case(inputs="b", metadata=TestCaseMetadata(), evaluators=[RegexInOutputEvaluator(pattern="y", tags=set())]),
    ]
    testset = Dataset[str, str, TestCaseMetadata](cases=cases)
    run = DatasetRunOutput(cases=[_case_run("a", ["x", "nope", "x"]), _case_run("b", ["y", "y"])])

    seq = await evaluate_results(run, testset, max_concurrency=1)
    conc = await evaluate_results(run, testset, max_concurrency=4)

    def _summary(out):
        return [(cr.case_name, cr.aggregated.passed, round(cr.aggregated.pass_rate, 6)) for cr in out.case_results]

    assert _summary(seq) == _summary(conc)
    assert len(seq.runs) == len(conc.runs)


@pytest.mark.anyio
async def test_evaluator_failure_order_is_declaration_order_under_concurrency():
    # Two failing evaluators with inverted latencies: completion order is
    # slow-then-fast at concurrency 1 but fast-then-slow at 4. The recorded
    # failure order must be declaration order either way (round-3 R8).
    from dataclasses import dataclass

    from ragpill.base import BaseEvaluator

    @dataclass(kw_only=True)
    class SlowFail(BaseEvaluator):
        async def run(self, ctx):
            await anyio.sleep(0.1)
            raise RuntimeError("boom")

    @dataclass(kw_only=True)
    class FastFail(BaseEvaluator):
        async def run(self, ctx):
            raise RuntimeError("boom")

    cases = [
        Case(
            inputs="a",
            metadata=TestCaseMetadata(),
            evaluators=[SlowFail(tags=set()), FastFail(tags=set())],
        )
    ]
    testset = Dataset[str, str, TestCaseMetadata](cases=cases)
    run = DatasetRunOutput(cases=[_case_run("a", ["out"])])

    for concurrency in (1, 4):
        result = await evaluate_results(run, testset, max_concurrency=concurrency)
        names = [f.name for f in result.case_results[0].run_results[0].evaluator_failures]
        assert names == ["SlowFail", "FastFail"], f"max_concurrency={concurrency}: {names}"


def test_llm_judge_has_optional_timeout_field():
    import dataclasses

    from ragpill.evaluators import LLMJudge

    fields = {f.name for f in dataclasses.fields(LLMJudge)}
    assert "timeout_s" in fields


# ---------------------------------------------------------------------------
# R2 — traced runs serialize process-wide: across threads AND event loops
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


class _ConcurrencyRecordingBackend:
    """Records how many traced capture sections (set_destination → end_run)
    are active at once. Class-level counters: the registry hands the same
    instance to every thread, and the guard makes the bookkeeping race-free."""

    supports_local_file_store = False
    _guard = threading.Lock()
    active = 0
    max_active = 0

    def get_tracking_uri(self):
        return None

    def set_tracking_uri(self, uri):
        pass

    def set_destination(self, uri, experiment_name):
        cls = type(self)
        with cls._guard:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        time.sleep(0.05)  # widen the window so an unlocked overlap is caught

    def autolog_pydantic_ai(self):
        pass

    def start_run(self, run_id=None, description=None):
        return RunHandle(run_id="r", experiment_id="e")

    def end_run(self):
        cls = type(self)
        with cls._guard:
            cls.active -= 1

    def is_run_active(self):
        return True

    @contextmanager
    def start_span(self, name, span_type, attributes=None):
        yield _FakeSpan()

    @contextmanager
    def start_case_grouping(self, case_id, name, inputs=None, attributes=None):
        yield CaseGroupingHandle(mode="session", session_id=case_id, case_trace_id=None)

    def await_trace(self, trace_id, *, run_id=None, experiment_id=None, timeout_s=10.0, poll_interval_s=0.5):
        return None, False


def test_traced_runs_serialize_across_threads_and_event_loops():
    """Round-3 R2 reproduction: two OS threads, each with its own event loop
    (asyncio.run), both tracing. The old per-loop anyio.Lock let them overlap
    (2 concurrent capture sections); the process-global threading.Lock must
    keep the max at 1."""
    _ConcurrencyRecordingBackend.active = 0
    _ConcurrencyRecordingBackend.max_active = 0
    configure_backend(_ConcurrencyRecordingBackend)
    errors: list[BaseException] = []
    try:

        def run_one(name: str) -> None:
            async def task(_q):
                await asyncio.sleep(0.02)
                return "x"

            ds = Dataset(cases=[Case(inputs=name, metadata=TestCaseMetadata())])
            try:
                asyncio.run(execute_dataset(ds, task=task, capture_traces=True, tracking_uri=f"http://{name}"))
            except BaseException as exc:  # surfaced to the main thread below
                errors.append(exc)

        threads = [threading.Thread(target=run_one, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        reset_backend()

    assert errors == []
    assert _ConcurrencyRecordingBackend.max_active == 1  # never two capture sections at once


# ---------------------------------------------------------------------------
# R2 — upload_results holds the SAME tracking-state lock for its whole
# reattach → write → restore section
# ---------------------------------------------------------------------------


class _RecordingLock:
    """Wrapper with the context-manager surface upload_results uses; records
    how often it was entered and lets the fake backend check whether it is
    held at mutate/restore time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False

    def locked(self):
        return self._lock.locked()


class _UploadRecordingBackend:
    """No-op upload backend recording, for the destination mutation and the
    URI restore, whether the given lock was held at the time of the call."""

    supports_local_file_store = False

    def __init__(self, lock: _RecordingLock, held_during: list[tuple[str, bool]]):
        self._lock = lock
        self._held_during = held_during

    def get_tracking_uri(self):
        return "http://previous"

    def set_tracking_uri(self, uri):
        self._held_during.append(("set_tracking_uri", self._lock.locked()))

    def set_destination(self, uri, experiment_name):
        self._held_during.append(("set_destination", self._lock.locked()))

    def start_run(self, run_id=None, description=None):
        return RunHandle(run_id="r", experiment_id="e")

    def end_run(self):
        pass

    def is_run_active(self):
        return True

    def get_run_tag(self, run_id, key):
        return None

    def set_run_tag(self, run_id, key, value):
        pass

    def delete_run_artifact(self, run_id, artifact_path):
        pass

    def log_table(self, df, artifact_file):
        pass

    def log_metric(self, name, value):
        pass

    def log_params(self, params):
        pass

    def log_artifact(self, local_path, artifact_path=None):
        pass

    def log_assessment(self, trace_id, assessment):
        pass

    def set_trace_tag(self, trace_id, key, value):
        pass

    def resolve_experiment_id(self, experiment_name):
        return "e"

    def delete_judge_traces(self, experiment_id, run_id):
        pass


def test_upload_results_holds_tracking_state_lock(monkeypatch):
    import ragpill.upload as upload_mod
    from ragpill.types import EvaluationOutput
    from ragpill.upload import upload_results

    rec = _RecordingLock()
    monkeypatch.setattr(upload_mod, "tracking_state_lock", rec)
    held_during: list[tuple[str, bool]] = []

    evaluation = EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=[])
    upload_results(evaluation, tracking_uri="http://server", backend=_UploadRecordingBackend(rec, held_during))

    # Both the destination mutation (via _reattach_run) and the URI restore
    # happened while the shared lock was held, in one continuous section.
    assert ("set_destination", True) in held_during
    assert ("set_tracking_uri", True) in held_during
    assert rec.enter_count == 1
