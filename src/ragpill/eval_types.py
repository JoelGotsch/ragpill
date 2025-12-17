"""Local evaluation primitives replacing the pydantic_evals dependency."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Generic, Literal

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from ragpill.base import BaseEvaluator
    from ragpill.trace import Trace


InputsT = TypeVar("InputsT", default=Any)
OutputT = TypeVar("OutputT", default=Any)
MetadataT = TypeVar("MetadataT", default=Any)

# Trace-availability of a run, recorded so downstream evaluators/reporting can
# distinguish an infrastructure failure from a real result. "ok" = complete
# trace; "incomplete" = a trace was read but the export was still settling at
# the fetch deadline (possibly missing spans); "unavailable" = no trace at all
# (fetch timed out empty, backend errored, or the run's subtree was absent).
TraceStatus = Literal["ok", "incomplete", "unavailable"]


@dataclass
class EvaluationReason:
    """Result of running an evaluator with optional explanation.

    Attributes:
        value: The evaluator's verdict. Typically a bool; evaluators may return
            int, float, or str for non-binary assertions.
        reason: Optional human-readable explanation of the verdict.
    """

    value: bool | int | float | str
    reason: str | None = None


@dataclass
class EvaluatorSource:
    """Provenance information for an evaluation result.

    Attributes:
        name: Identifier of the evaluator that produced the result
            (typically the class name).
        arguments: Arbitrary metadata describing the evaluator instance.
        source_type: ``"CODE"`` or ``"LLM_JUDGE"`` — declared by the evaluator
            class (:attr:`ragpill.base.BaseEvaluator.source_type`) rather than
            inferred from the class name.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source_type: Literal["CODE", "LLM_JUDGE"] = "CODE"


@dataclass
class EvaluationResult:
    """Details of an individual evaluation result.

    Attributes:
        name: The evaluator's name as it appears in the aggregated results.
        value: The verdict value (bool/int/float/str).
        reason: Optional explanation of the verdict.
        source: Provenance describing which evaluator produced this result.
    """

    name: str
    value: bool | int | float | str
    reason: str | None
    source: EvaluatorSource


@dataclass(kw_only=True)
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context passed to evaluators during evaluation.

    Attributes:
        name: Optional case name.
        inputs: The task inputs for this case.
        metadata: Case-level metadata (typically TestCaseMetadata), or None.
        expected_output: The expected output for this case, if any.
        output: The actual task output.
        duration: Wall-clock seconds the task took to run.
        attributes: Arbitrary attributes attached to the run.
        metrics: Arbitrary numeric metrics attached to the run.
        trace: Vendor-neutral ``ragpill.trace.Trace`` captured during the run.
            Populated by the execute layer and consumed by span-based
            evaluators. None when tracing was disabled.
        run_span_id: Span ID of the per-run parent span inside ``trace``.
            Used by span-based evaluators to restrict attention to the current
            run's subtree when multiple runs share a trace. Empty string or
            None when tracing was disabled.
        trace_status: Trace availability for this run (see
            :data:`~ragpill.eval_types.TraceStatus`).
    """

    name: str | None
    inputs: InputsT
    metadata: MetadataT | None
    expected_output: OutputT | None
    output: OutputT
    duration: float
    attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
    trace: Trace | None = None
    run_span_id: str | None = None
    # Span-based evaluators treat anything other than "ok" as an infrastructure
    # failure (raise TraceUnavailableError) rather than an evaluation result.
    trace_status: TraceStatus = "ok"


@dataclass
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single test case in a dataset.

    Attributes:
        inputs: The inputs to pass to the task under test.
        name: Optional display name for the case.
        metadata: Case-level metadata (typically TestCaseMetadata).
        expected_output: Optional expected output for comparison.
        evaluators: Case-level evaluators (each a ragpill BaseEvaluator).
    """

    inputs: InputsT
    name: str | None = None
    metadata: MetadataT | None = None
    expected_output: OutputT | None = None
    evaluators: list[BaseEvaluator] = field(default_factory=list)


@dataclass
class Dataset(Generic[InputsT, OutputT, MetadataT]):
    """A collection of test cases with optional global evaluators.

    Attributes:
        cases: The test cases in this dataset.
        evaluators: Dataset-level evaluators applied to every case.
    """

    cases: list[Case[InputsT, OutputT, MetadataT]] = field(default_factory=list)
    evaluators: list[BaseEvaluator] = field(default_factory=list)


def _build_serialization_arguments(instance: Any) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """Return non-default field values as a dict.

    Mirrors ``pydantic_evals.evaluators._base.BaseEvaluator.build_serialization_arguments``
    so downstream behavior (serialization, repr) remains consistent after the
    pydantic_evals removal. Called from ``ragpill.base.BaseEvaluator``.
    """
    result: dict[str, Any] = {}
    for f in fields(instance):
        value = getattr(instance, f.name)
        if f.default is not MISSING:
            if value == f.default:
                continue
        if f.default_factory is not MISSING:
            if value == f.default_factory():
                continue
        result[f.name] = value
    return result
