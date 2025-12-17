"""Execute layer: run tasks against a dataset and capture traces.

The execute layer is one of three independent layers in ragpill's pipeline:

1. **Execute** — this module. Runs tasks, captures traces, returns a
   :class:`DatasetRunOutput`.
2. **Evaluate** — in :mod:`ragpill.evaluation` (Phase 2). Consumes a
   :class:`DatasetRunOutput` and a :class:`ragpill.eval_types.Dataset` of
   evaluators and returns an :class:`ragpill.types.EvaluationOutput`.
3. **Upload** — in :mod:`ragpill.upload` (Phase 3). Persists evaluation
   results to an MLflow server.

This module is intentionally dependency-light on MLflow: it sets the tracking
URI it is told to use and restores the previous URI when finished. It supports
two tracing backends:

- **Local temp SQLite** (default) when ``mlflow_tracking_uri`` is ``None``. The
  temp database is deleted when execution completes.
- **Direct server tracing** when an explicit URI is provided — traces are
  written directly to that server and can later be uploaded by layer 3.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import Trace

from ragpill.backends import SpanKind, get_backend
from ragpill.base import (
    CaseMetadataT,
    TestCaseMetadata,
    default_input_to_key,
    resolve_repeat,
)
from ragpill.eval_types import Case, Dataset
from ragpill.settings import MLFlowSettings
from ragpill.utils import _fix_evaluator_global_flag  # pyright: ignore[reportPrivateUsage]

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
        trace: MLflow ``Trace`` scoped to this run (filtered to the run's
            subtree). ``None`` when tracing is disabled.
        run_span_id: Span ID of the per-run parent span inside the case trace.
            Empty string when tracing is disabled.
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
        trace: Full case-level MLflow ``Trace`` (including all run subtrees).
            ``None`` when tracing is disabled.
        trace_id: MLflow trace id string.
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
        tracking_uri: The MLflow tracking URI that was active during execution.
            Empty string when the local temp backend was used (the temp DB is
            cleaned up after execution).
        mlflow_run_id: MLflow run ID under which traces were captured. Empty
            when tracing was disabled or the temp backend was used.
        mlflow_experiment_id: MLflow experiment ID under which the run was
            created. Empty when tracing was disabled or the temp backend was used.
    """

    cases: list[CaseRunOutput] = field(default_factory=list)
    tracking_uri: str = ""
    mlflow_run_id: str = ""
    mlflow_experiment_id: str = ""

    def to_json(self) -> str:
        """Serialize this output to a JSON string.

        Traces are serialized via ``Trace.to_json()``. Other fields are
        preserved as-is.

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
        return json.dumps(_dataset_run_to_dict(self))

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


def _task_run_to_dict(tr: TaskRunOutput) -> dict[str, Any]:
    return {
        "run_index": tr.run_index,
        "input_key": tr.input_key,
        "output": tr.output,
        "duration": tr.duration,
        "trace": tr.trace.to_json() if tr.trace is not None else None,
        "run_span_id": tr.run_span_id,
        "error": tr.error,
    }


def _task_run_from_dict(d: dict[str, Any]) -> TaskRunOutput:
    trace_json: str | None = d.get("trace")
    trace = Trace.from_json(trace_json) if trace_json else None
    return TaskRunOutput(
        run_index=d["run_index"],
        input_key=d["input_key"],
        output=d.get("output"),
        duration=d.get("duration", 0.0),
        trace=trace,
        run_span_id=d.get("run_span_id", ""),
        error=d.get("error"),
    )


def _case_run_to_dict(cr: CaseRunOutput) -> dict[str, Any]:
    return {
        "case_name": cr.case_name,
        "inputs": cr.inputs,
        "expected_output": cr.expected_output,
        "metadata": cr.metadata,
        "base_input_key": cr.base_input_key,
        "trace": cr.trace.to_json() if cr.trace is not None else None,
        "trace_id": cr.trace_id,
        "task_runs": [_task_run_to_dict(tr) for tr in cr.task_runs],
    }


def _case_run_from_dict(d: dict[str, Any]) -> CaseRunOutput:
    trace_json: str | None = d.get("trace")
    trace = Trace.from_json(trace_json) if trace_json else None
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
        "cases": [_case_run_to_dict(c) for c in dr.cases],
        "tracking_uri": dr.tracking_uri,
        "mlflow_run_id": dr.mlflow_run_id,
        "mlflow_experiment_id": dr.mlflow_experiment_id,
    }


def _dataset_run_from_dict(d: dict[str, Any]) -> DatasetRunOutput:
    return DatasetRunOutput(
        cases=[_case_run_from_dict(c) for c in d.get("cases", [])],
        tracking_uri=d.get("tracking_uri", ""),
        mlflow_run_id=d.get("mlflow_run_id", ""),
        mlflow_experiment_id=d.get("mlflow_experiment_id", ""),
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


def _setup_local_tracing(experiment_name: str) -> _TracingContext:
    """Configure a temporary local SQLite backend.

    Used when no server URI is provided. The temp directory (containing
    ``mlflow.db`` and the ``mlartifacts`` folder) is deleted by
    :func:`_teardown_tracing`. The SQLite URI is MLflow-specific; backends
    that don't use a local-file store will ignore the path and fall back to
    their own destination logic.
    """
    backend = get_backend()
    previous_uri = backend.get_tracking_uri()
    temp_dir = tempfile.mkdtemp(prefix="ragpill_exec_")
    db_path = os.path.join(temp_dir, "mlflow.db")
    artifacts_path = os.path.join(temp_dir, "mlartifacts")
    os.makedirs(artifacts_path, exist_ok=True)
    uri = f"sqlite:///{db_path}"
    backend.set_destination(uri, experiment_name)
    backend.autolog_pydantic_ai()
    handle = backend.start_run()
    return _TracingContext(
        tracking_uri=uri,
        experiment_id=handle.experiment_id,
        run_id=handle.run_id,
        previous_uri=previous_uri,
        temp_dir=temp_dir,
    )


def _setup_server_tracing(uri: str, settings: MLFlowSettings) -> _TracingContext:
    """Configure the tracking URI to point at an existing tracking server."""
    backend = get_backend()
    previous_uri = backend.get_tracking_uri()
    backend.set_destination(uri, settings.ragpill_experiment_name)
    backend.autolog_pydantic_ai()
    handle = backend.start_run(description=settings.ragpill_run_description)
    return _TracingContext(
        tracking_uri=uri,
        experiment_id=handle.experiment_id,
        run_id=handle.run_id,
        previous_uri=previous_uri,
        temp_dir=None,
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


def _filter_trace_to_subtree(trace: Trace, root_span_id: str) -> Trace | None:
    """Return a copy of ``trace`` containing only the subtree rooted at
    ``root_span_id``, or ``None`` when that span is not present."""
    from copy import copy

    all_spans = trace.data.spans or []
    if not any(s.span_id == root_span_id for s in all_spans):
        return None
    included: set[str] = set()
    queue = [root_span_id]
    while queue:
        current = queue.pop()
        included.add(current)
        for span in all_spans:
            if span.parent_id == current:
                queue.append(span.span_id)
    filtered_data = copy(trace.data)
    filtered_data.spans = [s for s in all_spans if s.span_id in included]
    return Trace(info=trace.info, data=filtered_data)


def _fetch_trace(experiment_id: str, run_id: str, parent_trace_id: str) -> Trace | None:
    """Fetch the trace whose request_id matches ``parent_trace_id``."""
    traces = get_backend().search_traces(run_id=run_id, experiment_id=experiment_id)
    for t in traces:
        if t.info.trace_id == parent_trace_id:
            return t
    # Fallback: if only one trace exists for the run, use it.
    return traces[0] if len(traces) == 1 else None


async def _execute_case_runs(
    case: Case[Any, Any, Any],
    task_factory: Callable[[], TaskType],
    input_to_key: Callable[[Any], str],
    repeat: int,
    capture_traces: bool,
    tracing: _TracingContext | None,
) -> CaseRunOutput:
    """Execute all ``repeat`` runs for a single case and return its output."""
    metadata = case.metadata
    assert metadata is None or isinstance(metadata, TestCaseMetadata)
    base_key = input_to_key(case.inputs)

    task_runs: list[TaskRunOutput] = []
    parent_trace_id = ""

    if capture_traces and tracing is not None:
        with get_backend().start_span(
            name=(case.name or str(case.inputs))[:60], span_type=SpanKind.TASK
        ) as parent_span:
            parent_span.set_inputs(case.inputs)
            parent_span.set_attribute("input_key", base_key)
            parent_span.set_attribute("n_runs", repeat)
            parent_trace_id = parent_span.request_id

            for i in range(repeat):
                task_runs.append(await _execute_single_run(case, task_factory, base_key, i, capture_traces=True))
    else:
        for i in range(repeat):
            task_runs.append(await _execute_single_run(case, task_factory, base_key, i, capture_traces=False))

    # Attach traces after spans have been committed.
    case_trace: Trace | None = None
    if capture_traces and tracing is not None and parent_trace_id:
        case_trace = _fetch_trace(tracing.experiment_id, tracing.run_id, parent_trace_id)
        if case_trace is not None:
            for tr in task_runs:
                if tr.run_span_id:
                    tr.trace = _filter_trace_to_subtree(case_trace, tr.run_span_id)

    return CaseRunOutput(
        case_name=case.name or str(case.inputs),
        inputs=case.inputs,
        expected_output=case.expected_output,
        metadata=metadata.model_dump(mode="json") if metadata is not None else {},
        base_input_key=base_key,
        trace=case_trace,
        trace_id=parent_trace_id,
        task_runs=task_runs,
    )


async def _execute_single_run(
    case: Case[Any, Any, Any],
    task_factory: Callable[[], TaskType],
    base_key: str,
    run_index: int,
    capture_traces: bool,
) -> TaskRunOutput:
    """Execute one repeat of a case; capture output, duration, span id, error."""
    input_key = f"{base_key}_{run_index}"
    fresh_task = task_factory()
    run_span_id = ""
    duration = 0.0
    output: Any = None
    error_str: str | None = None

    async def _call() -> Any:
        if inspect.iscoroutinefunction(fresh_task):
            return await fresh_task(case.inputs)
        return fresh_task(case.inputs)

    if capture_traces:
        try:
            with get_backend().start_span(name=f"run-{run_index}", span_type=SpanKind.TASK) as run_span:
                run_span_id = run_span.span_id
                run_span.set_attribute("run_index", run_index)
                run_span.set_attribute("input_key", input_key)
                run_span.set_inputs(case.inputs)
                t0 = time.perf_counter()
                output = await _call()
                duration = time.perf_counter() - t0
                run_span.set_outputs(output)
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
    else:
        try:
            t0 = time.perf_counter()
            output = await _call()
            duration = time.perf_counter() - t0
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"

    return TaskRunOutput(
        run_index=run_index,
        input_key=input_key,
        output=output,
        duration=duration,
        trace=None,  # filled in after the parent span has closed
        run_span_id=run_span_id,
        error=error_str,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_dataset(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    settings: MLFlowSettings | None = None,
    mlflow_tracking_uri: str | None = None,
    capture_traces: bool = True,
) -> DatasetRunOutput:
    """Run every case in a dataset and return the captured outputs + traces.

    The function is pure with respect to the dataset: evaluators attached to
    cases are not invoked. That is the Phase 2 evaluator's job. What happens
    here is task execution and trace capture.

    Tracing backends (selected by ``mlflow_tracking_uri``):

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
        mlflow_tracking_uri: Override the tracking URI. When ``None``, a
            temp SQLite backend is spun up and torn down.
        capture_traces: When ``False``, tasks are run without MLflow spans;
            all ``Trace`` fields in the result will be ``None`` and
            ``run_span_id`` will be empty. Use for fast non-traced runs.

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

    _settings = settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]
    _fix_evaluator_global_flag(testset)

    tracing: _TracingContext | None = None
    try:
        if capture_traces:
            if mlflow_tracking_uri:
                tracing = _setup_server_tracing(mlflow_tracking_uri, _settings)
            else:
                tracing = _setup_local_tracing(_settings.ragpill_experiment_name)

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
            )
            case_outputs.append(case_output)

        return DatasetRunOutput(
            cases=case_outputs,
            tracking_uri=tracing.tracking_uri if (tracing and tracing.temp_dir is None) else "",
            mlflow_run_id=tracing.run_id if (tracing and tracing.temp_dir is None) else "",
            mlflow_experiment_id=tracing.experiment_id if (tracing and tracing.temp_dir is None) else "",
        )
    finally:
        _teardown_tracing(tracing)


__all__ = [
    "CaseRunOutput",
    "DatasetRunOutput",
    "TaskRunOutput",
    "execute_dataset",
]
