"""Result types for multi-run evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING, Any

import pandas as pd

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult, EvaluatorSource

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

    def per_tag_accuracy(self) -> dict[str, float]:
        """Mean ``evaluator_result`` grouped by tag, across all runs and evaluators.

        Tags appear on both case metadata (``TestCaseMetadata.tags``) and
        evaluator metadata (``BaseEvaluator.tags``); these are union-merged
        during evaluation so each row in ``self.runs`` carries the full tag
        set. Rows with NaN ``evaluator_result`` (evaluator failures) are
        excluded.

        For boolean evaluators (the common case) the returned value is the
        pass rate. For numeric evaluators the mean of the raw values is
        returned alongside, since both share the ``evaluator_result``
        column.

        Returns:
            Mapping from tag -> mean evaluator result in [0.0, 1.0]. Empty
            dict when ``self.runs`` is empty, has no ``evaluator_result``
            column, or has no tagged rows.
        """
        if self.runs.empty or "evaluator_result" not in self.runs.columns:
            return {}
        runs: Any = self.runs
        df_valid = runs[runs["evaluator_result"].notna()]
        if df_valid.empty:
            return {}
        grouped = df_valid.explode("tags").groupby("tags")["evaluator_result"].mean()
        return {str(tag): float(acc) for tag, acc in grouped.items() if pd.notna(tag)}  # pyright: ignore[reportUnknownMemberType]

    def to_llm_text(
        self,
        *,
        max_chars: int = 32_000,
        include_passing: bool = True,
        include_spans: bool = True,
        redact: bool = True,
        redact_patterns: list[str] | None = None,
    ) -> str:
        """Render a triage-focused markdown view suitable for LLM input.

        See :func:`ragpill.report.triage.render_evaluation_output_as_triage`.
        """
        from ragpill.report.triage import render_evaluation_output_as_triage

        return render_evaluation_output_as_triage(
            self,
            max_chars=max_chars,
            include_passing=include_passing,
            include_spans=include_spans,
            redact=redact,
            redact_patterns=redact_patterns,
        )

    def to_json(self) -> str:
        """Serialize this :class:`EvaluationOutput` to a JSON string.

        DataFrames are encoded via ``pandas.to_json(orient="split")`` and
        traces via ``Trace.to_json()``. ``from_json`` is the inverse.
        """
        return json.dumps(_evaluation_output_to_dict(self))

    @classmethod
    def from_json(cls, s: str) -> EvaluationOutput:
        """Deserialize an :class:`EvaluationOutput` produced by :meth:`to_json`."""
        return _evaluation_output_from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _df_to_json(df: pd.DataFrame) -> str:
    # ``orient="table"`` includes a JSON Table Schema so dtypes survive the
    # round trip (``orient="split"`` collapses int-valued floats to int).
    if df.empty:
        return df.to_json(orient="split", date_format="iso", default_handler=str)  # pyright: ignore[reportUnknownMemberType, reportReturnType]
    return df.to_json(orient="table", date_format="iso", default_handler=str)  # pyright: ignore[reportUnknownMemberType, reportReturnType]


def _df_from_json(s: str) -> pd.DataFrame:
    payload = json.loads(s)
    if "schema" in payload:
        return pd.read_json(StringIO(s), orient="table")
    return pd.read_json(StringIO(s), orient="split")


def _evaluator_source_to_dict(src: EvaluatorSource) -> dict[str, Any]:
    return {"name": src.name, "arguments": src.arguments}


def _evaluator_source_from_dict(d: dict[str, Any]) -> EvaluatorSource:
    return EvaluatorSource(name=d["name"], arguments=dict(d.get("arguments", {})))


def _evaluation_result_to_dict(er: EvaluationResult) -> dict[str, Any]:
    return {
        "name": er.name,
        "value": er.value,
        "reason": er.reason,
        "source": _evaluator_source_to_dict(er.source),
    }


def _evaluation_result_from_dict(d: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        name=d["name"],
        value=d["value"],
        reason=d.get("reason"),
        source=_evaluator_source_from_dict(d["source"]),
    )


def _evaluator_failure_to_dict(ef: EvaluatorFailureInfo) -> dict[str, Any]:
    return {"name": ef.name, "error_message": ef.error_message, "error_stacktrace": ef.error_stacktrace}


def _evaluator_failure_from_dict(d: dict[str, Any]) -> EvaluatorFailureInfo:
    return EvaluatorFailureInfo(
        name=d["name"],
        error_message=d["error_message"],
        error_stacktrace=d["error_stacktrace"],
    )


def _run_result_to_dict(rr: RunResult) -> dict[str, Any]:
    return {
        "run_index": rr.run_index,
        "input_key": rr.input_key,
        "run_span_id": rr.run_span_id,
        "output": rr.output,
        "duration": rr.duration,
        "assertions": {k: _evaluation_result_to_dict(v) for k, v in rr.assertions.items()},
        "evaluator_failures": [_evaluator_failure_to_dict(ef) for ef in rr.evaluator_failures],
        # Exceptions don't survive JSON; persist the string representation.
        "error": None if rr.error is None else f"{type(rr.error).__name__}: {rr.error}",
    }


def _run_result_from_dict(d: dict[str, Any]) -> RunResult:
    error_str = d.get("error")
    error: Exception | None = None
    if error_str is not None:
        error = RuntimeError(error_str)
    return RunResult(
        run_index=d["run_index"],
        input_key=d["input_key"],
        run_span_id=d.get("run_span_id", ""),
        output=d.get("output"),
        duration=d.get("duration", 0.0),
        assertions={k: _evaluation_result_from_dict(v) for k, v in d.get("assertions", {}).items()},
        evaluator_failures=[_evaluator_failure_from_dict(ef) for ef in d.get("evaluator_failures", [])],
        error=error,
    )


def _aggregated_to_dict(ar: AggregatedResult) -> dict[str, Any]:
    return {
        "passed": ar.passed,
        "pass_rate": ar.pass_rate,
        "threshold": ar.threshold,
        "summary": ar.summary,
        "per_evaluator_pass_rates": ar.per_evaluator_pass_rates,
    }


def _aggregated_from_dict(d: dict[str, Any]) -> AggregatedResult:
    return AggregatedResult(
        passed=d["passed"],
        pass_rate=d["pass_rate"],
        threshold=d["threshold"],
        summary=d["summary"],
        per_evaluator_pass_rates=dict(d.get("per_evaluator_pass_rates", {})),
    )


def _case_result_to_dict(cr: CaseResult) -> dict[str, Any]:
    metadata_dict = cr.metadata.model_dump(mode="json")
    return {
        "case_name": cr.case_name,
        "inputs": cr.inputs,
        "metadata": metadata_dict,
        "base_input_key": cr.base_input_key,
        "trace_id": cr.trace_id,
        "run_results": [_run_result_to_dict(rr) for rr in cr.run_results],
        "aggregated": _aggregated_to_dict(cr.aggregated),
    }


def _case_result_from_dict(d: dict[str, Any]) -> CaseResult:
    metadata = TestCaseMetadata.model_validate(d.get("metadata") or {})
    return CaseResult(
        case_name=d["case_name"],
        inputs=d.get("inputs"),
        metadata=metadata,
        base_input_key=d["base_input_key"],
        trace_id=d.get("trace_id", ""),
        run_results=[_run_result_from_dict(rr) for rr in d.get("run_results", [])],
        aggregated=_aggregated_from_dict(d["aggregated"]),
    )


def _evaluation_output_to_dict(eo: EvaluationOutput) -> dict[str, Any]:
    from ragpill.execution import _dataset_run_to_dict  # pyright: ignore[reportPrivateUsage]

    return {
        "runs": _df_to_json(eo.runs),
        "cases": _df_to_json(eo.cases),
        "case_results": [_case_result_to_dict(cr) for cr in eo.case_results],
        "dataset_run": _dataset_run_to_dict(eo.dataset_run) if eo.dataset_run is not None else None,
    }


def _evaluation_output_from_dict(d: dict[str, Any]) -> EvaluationOutput:
    from ragpill.execution import _dataset_run_from_dict  # pyright: ignore[reportPrivateUsage]

    dataset_run_dict = d.get("dataset_run")
    return EvaluationOutput(
        runs=_df_from_json(d["runs"]),
        cases=_df_from_json(d["cases"]),
        case_results=[_case_result_from_dict(cr) for cr in d.get("case_results", [])],
        dataset_run=_dataset_run_from_dict(dataset_run_dict) if dataset_run_dict is not None else None,
    )
