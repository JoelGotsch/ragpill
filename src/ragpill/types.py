"""Result types for multi-run evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult

if TYPE_CHECKING:
    from ragpill.execution import DatasetRunOutput


@dataclass
class EvaluatorFailureInfo:
    """Information about an evaluator that raised an exception during evaluation."""

    name: str
    error_message: str
    error_stacktrace: str


@dataclass
class RunResult:
    """Result of a single task execution (one run of one test case).

    Attributes:
        run_index: Zero-based index of this run within the case's repeat sequence.
        input_key: Unique key for this run, formatted as ``{base_hash}_{run_index}``.
        run_span_id: MLflow span ID captured during Phase 1, used to set ContextVar in Phase 2.
        output: The task's return value, or None if the task raised an exception.
        duration: Wall-clock seconds the task took to execute.
        assertions: Evaluator name -> EvaluationResult mapping for this run.
        evaluator_failures: Evaluators that raised exceptions (not pass/fail, but code errors).
        error: The exception raised by the task, or None if it succeeded.
    """

    run_index: int
    input_key: str
    run_span_id: str
    output: Any
    duration: float
    assertions: dict[str, EvaluationResult]
    evaluator_failures: list[EvaluatorFailureInfo] = field(default_factory=list)
    error: Exception | None = None

    @property
    def all_passed(self) -> bool:
        """True if the task succeeded and every assertion passed."""
        if self.error is not None:
            return False
        if not self.assertions:
            return True
        return all(r.value is True for r in self.assertions.values())


@dataclass
class AggregatedResult:
    """Aggregated pass/fail verdict across multiple runs of the same test case.

    Attributes:
        passed: Whether ``pass_rate >= threshold``.
        pass_rate: Fraction of runs where ``all_passed`` was True (0.0 to 1.0).
        threshold: The minimum pass_rate required to pass.
        summary: Human-readable summary string (e.g. "2/3 runs passed").
        per_evaluator_pass_rates: Per-evaluator pass rates across runs.
    """

    passed: bool
    pass_rate: float
    threshold: float
    summary: str
    per_evaluator_pass_rates: dict[str, float]


@dataclass
class CaseResult:
    """Result for a single test case across all its runs.

    Attributes:
        case_name: Display name or string representation of the case inputs.
        inputs: The original test case inputs.
        metadata: The TestCaseMetadata for this case.
        base_input_key: Hash of the inputs (without run index suffix).
        trace_id: MLflow trace ID for the parent span.
        run_results: One RunResult per repeat execution.
        aggregated: Aggregated verdict across all runs.
    """

    case_name: str
    inputs: Any
    metadata: TestCaseMetadata
    base_input_key: str
    trace_id: str
    run_results: list[RunResult]
    aggregated: AggregatedResult


@dataclass
class EvaluationOutput:
    """Top-level output returned by ``evaluate_testset_with_mlflow``.

    Provides three views of the evaluation data:
    - ``.runs``: One row per (run x evaluator) — the most granular view.
    - ``.cases``: One row per (case x evaluator) — aggregated across runs.
    - ``.summary``: One row per case — overall pass/fail with pass_rate.
    - ``.case_results``: The structured ``CaseResult`` objects for programmatic access.

    Attributes:
        runs: DataFrame with one row per (run x evaluator).
        cases: DataFrame with one row per (case x evaluator), aggregated.
        case_results: List of CaseResult objects.
    """

    runs: pd.DataFrame
    cases: pd.DataFrame
    case_results: list[CaseResult]
    dataset_run: DatasetRunOutput | None = None

    @property
    def summary(self) -> pd.DataFrame:
        """One row per case with overall pass/fail and pass_rate."""
        rows: list[dict[str, Any]] = []
        for cr in self.case_results:
            rows.append(
                {
                    "case_id": cr.base_input_key,
                    "case_name": cr.case_name,
                    "passed": cr.aggregated.passed,
                    "pass_rate": cr.aggregated.pass_rate,
                    "threshold": cr.aggregated.threshold,
                    "summary": cr.aggregated.summary,
                }
            )
        return pd.DataFrame(rows)
