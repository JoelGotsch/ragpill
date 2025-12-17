"""Execute layer: run tasks against a dataset and capture traces.

The execute layer is one of three independent layers in ragpill's pipeline:

1. **Execute** — this module. Runs tasks, captures traces, returns a
   :class:`DatasetRunOutput`.
2. **Evaluate** — in :mod:`ragpill.evaluation` (Phase 2). Consumes a
   :class:`DatasetRunOutput` and a :class:`ragpill.eval_types.Dataset` of
   evaluators and returns an :class:`ragpill.types.EvaluationOutput`.
3. **Upload** — in :mod:`ragpill.upload` (Phase 3). Persists evaluation
   results to the configured tracking backend.

This module is backend-agnostic: it drives whichever backend is registered via
:func:`ragpill.backends.get_backend` (MLflow by default). Two capture modes:

- **Local temp store** (default) when ``tracking_uri`` is ``None``. For a
  file-store backend (MLflow) a private temp SQLite database is used and
  deleted when execution completes — a zero-server capture path.
- **Direct server tracing** when an explicit ``tracking_uri`` is provided —
  traces go straight to that destination and can later be uploaded by layer 3.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread

from ragpill.backends import CaptureSpanKind, get_backend
from ragpill.base import (
    CaseMetadataT,
    TestCaseMetadata,
    default_input_to_key,
    resolve_repeat,
)
from ragpill.eval_types import Case, Dataset
from ragpill.settings import TrackingSettings
from ragpill.trace import filter_to_subtree, trace_from_dict, trace_to_dict
from ragpill.utils import _fix_evaluator_global_flag  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from ragpill.trace import Trace

# Run-JSON schema version. v2 stores the vendor-neutral ragpill.trace.Trace
# (v1 stored mlflow.entities.Trace JSON). No v1 migrator — see ADR-0014.
_RUN_JSON_SCHEMA_VERSION = 2

TaskType = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskRunOutput:
    """Output of a single task execution (one run of one case).

    Attributes:
        run_index: Zero-based index of this run within the case's repeat sequence.
        input_key: Unique key for this run, formatted as ``{base_hash}_{run_index}``.
        output: Return value of the task, or ``None`` if the task raised.
        duration: Wall-clock seconds the task took to run.
        trace: Captured vendor-neutral ``ragpill.trace.Trace`` scoped to this
            run. In session-mode backends (e.g. MLflow with
            ``mlflow.trace.session`` metadata set) this is the run's own
            top-level trace; in span-mode backends it's the per-run subtree
            filtered out of the case-level trace. ``None`` when tracing is
            disabled.
        run_span_id: Span ID captured at the per-run span open. In
            session mode this is the trace's root span id; in span mode
            it's the child span id under the case's parent.
        trace_id: Backend trace id captured at the per-run span open.
            Populated in session-mode backends so the post-loop can
            fetch each repeat's own trace; empty string in span mode
            (use the case-level ``CaseRunOutput.trace_id`` instead).
        error: String representation of any exception the task raised. ``None``
            on success. We store the string (not the exception) so the
            dataclass stays JSON-serializable.
    """

    run_index: int
    input_key: str
    output: Any
    duration: float
    trace: Trace | None = None
    run_span_id: str = ""
    trace_id: str = ""
    error: str | None = None


@dataclass
class CaseRunOutput:
    """All runs for a single case, plus the case-level trace.

    Attributes:
        case_name: Display name (falls back to ``str(inputs)``).
        inputs: The task inputs for this case.
        expected_output: The case's expected output, if any.
        metadata: Case metadata as a plain dict (for JSON round-trip).
        base_input_key: Hash of the inputs (no run-index suffix).
        trace: Full case-level ``ragpill.trace.Trace`` (including all run
            subtrees). ``None`` when tracing is disabled.
        trace_id: Backend trace id string.
        task_runs: One :class:`TaskRunOutput` per repeat.
    """

    case_name: str
    inputs: Any
    expected_output: Any
    metadata: dict[str, Any]
    base_input_key: str
    trace: Trace | None
    trace_id: str
    task_runs: list[TaskRunOutput]


@dataclass
class DatasetRunOutput:
    """Top-level output of :func:`execute_dataset`.

    Attributes:
        cases: One :class:`CaseRunOutput` per case in the dataset.
        tracking_uri: The tracking URI that was active during execution.
            Empty string when the local temp backend was used (the temp DB is
            cleaned up after execution).
        run_id: Backend run ID under which traces were captured. Empty when
            tracing was disabled or the temp backend was used.
        experiment_id: Backend experiment ID under which the run was created.
            Empty when tracing was disabled or the temp backend was used.
    """

    cases: list[CaseRunOutput] = field(default_factory=list)
    tracking_uri: str = ""
    run_id: str = ""
    experiment_id: str = ""

    def to_json(self) -> str:
        """Serialize this output to a JSON string.

        Traces are serialized via the neutral ``ragpill.trace`` model
        (``schema_version`` 2). Other fields are preserved as-is.

        Returns:
            A JSON string that ``from_json`` can round-trip back into an
            equivalent :class:`DatasetRunOutput`.

        Example:
            ```python
            run_output = await execute_dataset(dataset, task=my_task)
            with open("run.json", "w") as f:
                f.write(run_output.to_json())
            ```
        """
        return json.dumps(_dataset_run_to_dict(self), default=_json_fallback)

    @classmethod
    def from_json(cls, s: str) -> DatasetRunOutput:
        """Deserialize a :class:`DatasetRunOutput` produced by :meth:`to_json`.

        Args:
            s: JSON string produced by :meth:`to_json`.

        Returns:
            A :class:`DatasetRunOutput` equivalent to the one that was
            serialized (trace data survives the round trip).

        Example:
            ```python
            with open("run.json") as f:
                run_output = DatasetRunOutput.from_json(f.read())
            ```
        """
        return _dataset_run_from_dict(json.loads(s))

    def to_llm_text(
        self,
        *,
        max_chars: int = 16_000,
        include_spans: bool = True,
        redact: bool = True,
        redact_patterns: list[str] | None = None,
    ) -> str:
        """Render an exploration-focused markdown view of this run.

        See :func:`ragpill.report.exploration.render_dataset_run_as_exploration`.
        """
        from ragpill.report.exploration import render_dataset_run_as_exploration

        return render_dataset_run_as_exploration(
            self,
            max_chars=max_chars,
            include_spans=include_spans,
            redact=redact,
            redact_patterns=redact_patterns,
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_fallback(obj: Any) -> str:
    """``json.dumps`` ``default`` hook for non-serializable task outputs.

    Task outputs are ``Any``; a RAG pipeline commonly returns a pydantic model,
    dataclass, or datetime. Rather than crash the "save run to disk" workflow
    *after* the expensive execution completed, coerce with ``str()`` and warn —
    the value survives for human inspection but won't round-trip to its
    original type.
    """
    warnings.warn(
        f"DatasetRunOutput.to_json: value of type {type(obj).__name__!r} is not "
        "JSON-serializable; stored as str(). It will not round-trip to the original type.",
        stacklevel=2,
    )
    return str(obj)


def _task_run_to_dict(tr: TaskRunOutput) -> dict[str, Any]:
    return {
        "run_index": tr.run_index,
        "input_key": tr.input_key,
        "output": tr.output,
        "duration": tr.duration,
        "trace": trace_to_dict(tr.trace) if tr.trace is not None else None,
        "run_span_id": tr.run_span_id,
        "trace_id": tr.trace_id,
        "error": tr.error,
    }


def _task_run_from_dict(d: dict[str, Any]) -> TaskRunOutput:
    trace_payload = d.get("trace")
    trace = trace_from_dict(trace_payload) if trace_payload else None
    return TaskRunOutput(
        run_index=d["run_index"],
        input_key=d["input_key"],
        output=d.get("output"),
        duration=d.get("duration", 0.0),
        trace=trace,
        run_span_id=d.get("run_span_id", ""),
        trace_id=d.get("trace_id", ""),
        error=d.get("error"),
    )


def _case_run_to_dict(cr: CaseRunOutput) -> dict[str, Any]:
    return {
        "case_name": cr.case_name,
        "inputs": cr.inputs,
        "expected_output": cr.expected_output,
        "metadata": cr.metadata,
        "base_input_key": cr.base_input_key,
        "trace": trace_to_dict(cr.trace) if cr.trace is not None else None,
        "trace_id": cr.trace_id,
        "task_runs": [_task_run_to_dict(tr) for tr in cr.task_runs],
    }


def _case_run_from_dict(d: dict[str, Any]) -> CaseRunOutput:
    trace_payload = d.get("trace")
    trace = trace_from_dict(trace_payload) if trace_payload else None
    return CaseRunOutput(
        case_name=d["case_name"],
        inputs=d.get("inputs"),
        expected_output=d.get("expected_output"),
        metadata=d.get("metadata", {}),
        base_input_key=d["base_input_key"],
        trace=trace,
        trace_id=d.get("trace_id", ""),
        task_runs=[_task_run_from_dict(tr) for tr in d.get("task_runs", [])],
    )


def _dataset_run_to_dict(dr: DatasetRunOutput) -> dict[str, Any]:
    return {
        "schema_version": _RUN_JSON_SCHEMA_VERSION,
        "cases": [_case_run_to_dict(c) for c in dr.cases],
        "tracking_uri": dr.tracking_uri,
        "run_id": dr.run_id,
        "experiment_id": dr.experiment_id,
    }


def _dataset_run_from_dict(d: dict[str, Any]) -> DatasetRunOutput:
    version = d.get("schema_version", 1)
    if version != _RUN_JSON_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run-JSON schema_version {version!r}; this build reads "
            f"v{_RUN_JSON_SCHEMA_VERSION}. v1 files store mlflow-shaped traces and are not "
            "migrated (the trace model changed in 0.5.0 — see CHANGELOG / ADR-0014). "
            "Re-run the evaluation to produce a v2 file."
        )
    return DatasetRunOutput(
        cases=[_case_run_from_dict(c) for c in d.get("cases", [])],
        tracking_uri=d.get("tracking_uri", ""),
        run_id=d.get("run_id", ""),
        experiment_id=d.get("experiment_id", ""),
    )


# ---------------------------------------------------------------------------
# Tracing backend setup/teardown
# ---------------------------------------------------------------------------


@dataclass
class _TracingContext:
    """Bookkeeping for the tracing backend used during a single run."""

    tracking_uri: str
    experiment_id: str
    run_id: str
    previous_uri: str | None
    temp_dir: str | None  # set for local-temp backend, used to rm -rf on teardown
    trace_fetch_timeout_s: float  # how long to poll for a trace to be exported
    trace_fetch_poll_interval_s: float  # interval between readiness polls


def _setup_tracing(uri: str | None, settings: TrackingSettings) -> _TracingContext:
    """Configure the tracing destination and open a run.

    With an explicit ``uri``, traces go straight to that server. Without one,
    file-store backends (``supports_local_file_store``, e.g. MLflow) get a
    temporary local SQLite store — the temp directory (containing
    ``mlflow.db`` and the ``mlartifacts`` folder) is deleted by
    :func:`_teardown_tracing` — while remote-service backends (Langfuse,
    Phoenix) get ``uri=None`` and fall back to their own environment-derived
    destination.
    """
    backend = get_backend()
    previous_uri = backend.get_tracking_uri()
    temp_dir: str | None = None
    if uri is None and getattr(backend, "supports_local_file_store", False):
        temp_dir = tempfile.mkdtemp(prefix="ragpill_exec_")
        db_path = os.path.join(temp_dir, "mlflow.db")
        artifacts_path = os.path.join(temp_dir, "mlartifacts")
        os.makedirs(artifacts_path, exist_ok=True)
        uri = f"sqlite:///{db_path}"
    backend.set_destination(uri, settings.experiment_name)
    backend.autolog_pydantic_ai()
    handle = backend.start_run(description=settings.run_description)
    return _TracingContext(
        tracking_uri=uri or "",
        experiment_id=handle.experiment_id,
        run_id=handle.run_id,
        previous_uri=previous_uri,
        temp_dir=temp_dir,
        trace_fetch_timeout_s=settings.trace_fetch_timeout_s,
        trace_fetch_poll_interval_s=settings.trace_fetch_poll_interval_s,
    )


def _teardown_tracing(ctx: _TracingContext | None) -> None:
    """End the active run and restore the previous tracking URI.

    For the local-temp backend, also removes the temp directory.
    """
    if ctx is None:
        return
    backend = get_backend()
    try:
        if backend.is_run_active():
            backend.end_run()
    finally:
        if ctx.previous_uri is not None:
            backend.set_tracking_uri(ctx.previous_uri)
        if ctx.temp_dir is not None and os.path.isdir(ctx.temp_dir):
            shutil.rmtree(ctx.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_trace(
    experiment_id: str,
    run_id: str,
    parent_trace_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> Trace | None:
    """Fetch the trace ``parent_trace_id`` and return it as a neutral ``Trace``.

    Spans flush to the backend asynchronously, so a fetch issued right after
    the case grouping context closes can miss a trace still in flight. The
    backend polls by id and returns ``None`` (never a different trace) on
    timeout — a miss leaves the SpanBaseEvaluators without a trace rather than
    silently scoring the wrong one. ``await_trace`` returns the vendor-neutral
    ``ragpill.trace.Trace`` (the backend converts its own native trace), so
    nothing here is MLflow-specific.
    """
    return get_backend().await_trace(
        parent_trace_id,
        run_id=run_id,
        experiment_id=experiment_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )


async def _execute_case_runs(
    case: Case[Any, Any, Any],
    task_factory: Callable[[], TaskType],
    input_to_key: Callable[[Any], str],
    repeat: int,
    capture_traces: bool,
    tracing: _TracingContext | None,
    task_timeout_s: float | None = None,
) -> CaseRunOutput:
    """Execute all ``repeat`` runs for a single case and return its output.

    Tracing-mode branches on the backend's ``start_case_grouping`` handle:

    - **session mode** (e.g. MLflow with ``mlflow.trace.session`` metadata,
      Langfuse with ``session_id``): each repeat opens as its own top-level
      trace, tagged with the case's id. The UI then shows one session per
      case, one turn per repeat. ``CaseRunOutput.trace`` is ``None``; each
      ``TaskRunOutput.trace`` is the repeat's own trace.

    - **span mode** (fallback for backends without sessions): the case
      opens a parent span and repeats nest beneath it. The post-loop fetches
      the case-level trace once and filters per-repeat subtrees, as in
      pre-0.5 versions.
    """
    metadata = case.metadata
    assert metadata is None or isinstance(metadata, TestCaseMetadata)
    base_key = input_to_key(case.inputs)

    task_runs: list[TaskRunOutput] = []
    case_trace_id = ""
    grouping_mode: str = "span"

    if capture_traces and tracing is not None:
        backend = get_backend()
        with backend.start_case_grouping(
            case_id=base_key,
            name=(case.name or str(case.inputs))[:60],
            inputs=case.inputs,
            attributes={"input_key": base_key, "n_runs": repeat},
        ) as case_handle:
            grouping_mode = case_handle.mode
            case_trace_id = case_handle.case_trace_id or ""
            for i in range(repeat):
                task_runs.append(
                    await _execute_single_run(
                        case, task_factory, base_key, i, capture_traces=True, task_timeout_s=task_timeout_s
                    )
                )
    else:
        for i in range(repeat):
            task_runs.append(
                await _execute_single_run(
                    case, task_factory, base_key, i, capture_traces=False, task_timeout_s=task_timeout_s
                )
            )

    # Attach traces after spans have been committed. ``await_trace`` is a
    # synchronous polling call (time.sleep between readiness checks), so it is
    # offloaded to a worker thread rather than run on the event loop.
    case_trace: Trace | None = None
    if capture_traces and tracing is not None:
        if grouping_mode == "span" and case_trace_id:
            # Span mode: one case trace, filtered per repeat. ``await_trace`` is a
            # synchronous polling call, so it is offloaded to a worker thread
            # (via anyio, portable across asyncio and trio) rather than blocking
            # the event loop.
            case_trace = await anyio.to_thread.run_sync(
                functools.partial(
                    _fetch_trace,
                    tracing.experiment_id,
                    tracing.run_id,
                    case_trace_id,
                    timeout_s=tracing.trace_fetch_timeout_s,
                    poll_interval_s=tracing.trace_fetch_poll_interval_s,
                )
            )
            if case_trace is not None:
                for tr in task_runs:
                    if tr.run_span_id:
                        tr.trace = filter_to_subtree(case_trace, tr.run_span_id)
        elif grouping_mode == "session":
            # Session mode: each repeat already produced its own trace; fetch
            # them individually by the trace_id captured at span open. The
            # fetches are independent, so they run concurrently — worst case
            # is one timeout, not one per repeat.
            pending = [tr for tr in task_runs if tr.trace_id]
            fetched: list[Trace | None] = [None] * len(pending)

            async def _fetch_into(idx: int, tid: str) -> None:
                fetched[idx] = await anyio.to_thread.run_sync(
                    functools.partial(
                        _fetch_trace,
                        tracing.experiment_id,
                        tracing.run_id,
                        tid,
                        timeout_s=tracing.trace_fetch_timeout_s,
                        poll_interval_s=tracing.trace_fetch_poll_interval_s,
                    )
                )

            async with anyio.create_task_group() as tg:
                for i, tr in enumerate(pending):
                    tg.start_soon(_fetch_into, i, tr.trace_id)
            for tr, trace in zip(pending, fetched):
                tr.trace = trace

    return CaseRunOutput(
        case_name=case.name or str(case.inputs),
        inputs=case.inputs,
        expected_output=case.expected_output,
        metadata=metadata.model_dump(mode="json") if metadata is not None else {},
        base_input_key=base_key,
        trace=case_trace,
        trace_id=case_trace_id,
        task_runs=task_runs,
    )


async def _execute_single_run(
    case: Case[Any, Any, Any],
    task_factory: Callable[[], TaskType],
    base_key: str,
    run_index: int,
    capture_traces: bool,
    task_timeout_s: float | None = None,
) -> TaskRunOutput:
    """Execute one repeat of a case; capture output, duration, span id, error."""
    input_key = f"{base_key}_{run_index}"
    fresh_task = task_factory()
    run_span_id = ""
    trace_id = ""
    duration = 0.0
    output: Any = None
    error_str: str | None = None

    async def _call() -> Any:
        call = fresh_task
        # ``iscoroutinefunction`` is False for a callable *instance* whose
        # ``__call__`` is async — the shape ``task_factory`` exists for. Detect
        # both so async tasks run on the loop and don't leak an un-awaited
        # coroutine as the output.
        is_async = inspect.iscoroutinefunction(call) or inspect.iscoroutinefunction(getattr(call, "__call__", None))
        result: Any
        if is_async:
            result = call(case.inputs)
        else:
            # A truly synchronous task runs in a worker thread so a blocking
            # client (e.g. a ``requests``-based RAG call) doesn't stall the event
            # loop and the tracking exporters running on it. anyio keeps this
            # portable across the asyncio and trio backends.
            result = await anyio.to_thread.run_sync(call, case.inputs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _invoke() -> Any:
        # Timeouts are opt-in and caller-specified: ragpill imposes no default
        # budget on the client's task. ``None`` means no timeout.
        if task_timeout_s is not None:
            with anyio.fail_after(task_timeout_s):
                return await _call()
        return await _call()

    if capture_traces:
        try:
            with get_backend().start_span(name=f"run-{run_index}", span_type=CaptureSpanKind.TASK) as run_span:
                run_span_id = run_span.span_id
                trace_id = run_span.trace_id
                run_span.set_attribute("run_index", run_index)
                run_span.set_attribute("input_key", input_key)
                run_span.set_inputs(case.inputs)
                t0 = time.perf_counter()
                try:
                    output = await _invoke()
                finally:
                    # Record duration even when the task raises, so a failed run
                    # reports its real latency (instant crash vs slow timeout)
                    # instead of a misleading 0.0.
                    duration = time.perf_counter() - t0
                run_span.set_outputs(output)
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
    else:
        t0 = time.perf_counter()
        try:
            output = await _invoke()
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
        finally:
            duration = time.perf_counter() - t0

    return TaskRunOutput(
        run_index=run_index,
        input_key=input_key,
        output=output,
        duration=duration,
        trace=None,  # filled in after the span has closed (session or span mode)
        run_span_id=run_span_id,
        trace_id=trace_id,
        error=error_str,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_dataset(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    *,
    settings: TrackingSettings | None = None,
    tracking_uri: str | None = None,
    capture_traces: bool = True,
    task_timeout_s: float | None = None,
) -> DatasetRunOutput:
    """Run every case in a dataset and return the captured outputs + traces.

    The function is pure with respect to the dataset: evaluators attached to
    cases are not invoked. That is the Phase 2 evaluator's job. What happens
    here is task execution and trace capture.

    Tracing backends (selected by ``tracking_uri``):

    - ``None`` — use a private temp SQLite database. The database is removed
      after the call, but captured ``Trace`` objects are copied into the
      returned :class:`DatasetRunOutput` before cleanup.
    - A URI string — trace directly to that server (dual-backend model).

    Args:
        testset: The dataset to execute.
        task: The task callable. Mutually exclusive with ``task_factory``.
        task_factory: A zero-arg callable that returns a fresh task instance
            per run (use for stateful tasks). Mutually exclusive with ``task``.
        settings: MLflow settings; falls back to environment variables.
        tracking_uri: Override the tracking URI. When ``None``, a
            temp SQLite backend is spun up and torn down.
        capture_traces: When ``False``, tasks are run without capturing spans;
            all ``Trace`` fields in the result will be ``None`` and
            ``run_span_id`` will be empty. Use for fast non-traced runs.
        task_timeout_s: Optional per-task wall-clock timeout in seconds. ``None``
            (default) imposes no timeout — the client owns its latency budget. A
            task exceeding the budget is recorded as a ``TimeoutError`` run and
            execution continues with the next repeat/case.

    Note:
        Cases and repeats run sequentially during capture: trace correctness
        depends on process-global tracking state (e.g. MLflow's active session),
        so the concurrency ceiling here is a deliberate constraint, not an
        oversight. Bounded-concurrent *evaluation* is available separately via
        ``evaluate_results(max_concurrency=...)``.

    Returns:
        A :class:`DatasetRunOutput` with one :class:`CaseRunOutput` per case
        and (when ``capture_traces=True``) attached ``Trace`` objects.

    Raises:
        ValueError: If both or neither of ``task`` and ``task_factory`` are
            provided.

    Example:
        ```python
        from ragpill import Case, Dataset, TestCaseMetadata
        from ragpill.execution import execute_dataset

        ds = Dataset(cases=[Case(inputs="hi", metadata=TestCaseMetadata())])

        async def my_task(q: str) -> str:
            return f"answer: {q}"

        run_output = await execute_dataset(ds, task=my_task)
        assert run_output.cases[0].task_runs[0].output == "answer: hi"
        ```

    See Also:
        [`DatasetRunOutput`][ragpill.execution.DatasetRunOutput]: the return type.
        [`ragpill.evaluation.evaluate_results`][ragpill.evaluation.evaluate_results]:
            Phase 2 — run evaluators against a ``DatasetRunOutput``.
    """
    if task is not None and task_factory is not None:
        raise ValueError("Provide either 'task' or 'task_factory', not both.")
    if task is None and task_factory is None:
        raise ValueError("Provide either 'task' or 'task_factory'.")

    _factory: Callable[[], TaskType]
    if task is not None:
        _task = task
        _factory = lambda: _task  # noqa: E731
    else:
        assert task_factory is not None
        _factory = task_factory

    _settings = settings or TrackingSettings()  # pyright: ignore[reportCallIssue]
    _fix_evaluator_global_flag(testset)

    tracing: _TracingContext | None = None
    try:
        if capture_traces:
            tracing = _setup_tracing(tracking_uri or None, _settings)

        case_outputs: list[CaseRunOutput] = []
        for case in testset.cases:
            case_metadata: TestCaseMetadata | None = (
                case.metadata if isinstance(case.metadata, TestCaseMetadata) else None
            )
            repeat, _ = resolve_repeat(case_metadata, _settings)
            case_output = await _execute_case_runs(
                case,
                _factory,
                default_input_to_key,
                repeat,
                capture_traces=capture_traces,
                tracing=tracing,
                task_timeout_s=task_timeout_s,
            )
            case_outputs.append(case_output)

        return DatasetRunOutput(
            cases=case_outputs,
            tracking_uri=tracing.tracking_uri if (tracing and tracing.temp_dir is None) else "",
            run_id=tracing.run_id if (tracing and tracing.temp_dir is None) else "",
            experiment_id=tracing.experiment_id if (tracing and tracing.temp_dir is None) else "",
        )
    finally:
        _teardown_tracing(tracing)


__all__ = [
    "CaseRunOutput",
    "DatasetRunOutput",
    "TaskRunOutput",
    "execute_dataset",
]
