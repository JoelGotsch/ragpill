"""Result types for multi-run evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Annotated, Any

import pandas as pd
from pydantic import PlainSerializer, PlainValidator, TypeAdapter

from ragpill.base import TestCaseMetadata
from ragpill.eval_types import EvaluationResult
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
        trace_id: Backend trace id captured at the run's span open. In
            session-mode backends each repeat is its own trace, so this is
            where per-run assessments are logged; in span mode it matches the
            case-level ``CaseResult.trace_id``. Empty when tracing was off.
    """

    run_index: int
    input_key: str
    run_span_id: str
    output: Any
    duration: float
    assertions: dict[str, EvaluationResult]
    evaluator_failures: list[EvaluatorFailureInfo] = field(default_factory=list)
    error: ErrorField = None
    trace_id: str = ""

    @property
    def is_error_state(self) -> bool:
        """True when the run could not be evaluated at all — it produced no
        assertions and no task error, only evaluator failures (e.g. the trace
        was unavailable). Such runs are excluded from pass-rate denominators so
        an infrastructure outage neither passes nor fails a case."""
        return not self.assertions and self.error is None and bool(self.evaluator_failures)

    @property
    def all_passed(self) -> bool:
        """True if the task succeeded and every assertion passed.

        An error-state run (see :attr:`is_error_state`) is not a pass — "no
        data" must never read green (ADR-0018). Runs where *some* evaluators
        produced verdicts and others failed are judged on the verdicts they did
        produce (the failures are surfaced separately in the triage report and
        the per-evaluator error counts).
        """
        if self.error is not None:
            return False
        if not self.assertions:
            return not self.is_error_state
        return all(r.value is True for r in self.assertions.values())


@dataclass
class AggregatedResult:
    """Aggregated pass/fail verdict across multiple runs of the same test case.

    Metric semantics follow ADR-0018: numbers a pipeline can gate on are
    conservative (infra failures only ever push them down), evaluated-only
    numbers are diagnostic and always travel with their coverage counts.

    Attributes:
        passed: The gate: ``pass_rate >= threshold`` AND no infra-degraded
            runs. An infra failure can only ever block a promotion, never
            improve the reported result.
        pass_rate: **Conservative lower bound** — passing runs divided by
            *all* runs, error-state (infra-degraded) runs counted as
            non-passes. Safe to gate on.
        pass_rate_evaluated: **Diagnostic** — passing runs divided by runs
            that could actually be evaluated. Answers "how good is the agent
            on the runs we could score"; never use it for gating without
            checking ``runs_infra_error``.
        runs_evaluated: Number of runs that produced verdicts (denominator of
            ``pass_rate_evaluated``).
        runs_infra_error: Number of error-state runs (trace unavailable etc.)
            excluded from ``pass_rate_evaluated`` and counted as non-passes in
            ``pass_rate``.
        threshold: The minimum ``pass_rate`` required to pass.
        summary: Human-readable summary string (e.g. "2/3 runs passed").
        per_evaluator_pass_rates: Per-evaluator pass rates across runs
            (diagnostic, evaluated-only). The denominator for each evaluator
            is the number of runs in which it produced a verdict; read it
            together with ``error_counts``.
        error_counts: Per-evaluator count of runs in which the evaluator
            errored (could not produce a verdict). Surfaces an outage instead
            of hiding or diluting it.
    """

    passed: bool
    pass_rate: float
    threshold: float
    summary: str
    per_evaluator_pass_rates: dict[str, float]
    error_counts: dict[str, int] = field(default_factory=dict)
    pass_rate_evaluated: float = 0.0
    runs_evaluated: int = 0
    runs_infra_error: int = 0


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
    """Top-level output returned by ``evaluate_testset``.

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

    runs: DataFrameField
    cases: DataFrameField
    case_results: list[CaseResult]
    dataset_run: DatasetRunField = None

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
                    "pass_rate_evaluated": cr.aggregated.pass_rate_evaluated,
                    "runs_infra_error": cr.aggregated.runs_infra_error,
                    "threshold": cr.aggregated.threshold,
                    "summary": cr.aggregated.summary,
                }
            )
        return pd.DataFrame(rows)

    def per_attribute_accuracy(self, attribute: str) -> dict[str, float]:
        """Mean evaluator result grouped by value of a single attribute key.

        Iterates ``self.case_results`` and looks up
        ``cr.metadata.attributes[attribute]`` for each case. Cases missing
        the key are skipped (not counted as failures). For each surviving
        case, every assertion across every run contributes its value
        (``True`` -> 1.0, ``False`` -> 0.0, numeric passthrough). Returns
        an empty dict when no case carries the key or no usable
        assertions exist. Attribute values are stringified for dict-key
        safety (covers unhashable values such as lists).
        """
        buckets: dict[str, list[float]] = {}
        for cr in self.case_results:
            attrs = cr.metadata.attributes
            if attribute not in attrs:
                continue
            value_key = str(attrs[attribute])
            bucket = buckets.setdefault(value_key, [])
            for rr in cr.run_results:
                for r in rr.assertions.values():
                    v = r.value
                    if isinstance(v, bool):
                        bucket.append(1.0 if v else 0.0)
                    elif isinstance(v, (int, float)):
                        bucket.append(float(v))
        return {k: sum(vs) / len(vs) for k, vs in buckets.items() if vs}

    def per_attribute_accuracy_all(self) -> dict[str, dict[str, float]]:
        """Auto-discovered per-attribute breakdown across the dataset.

        Returns a ``{attribute_key: {value: mean_score}}`` mapping for every
        attribute key that appears on at least one case and yields at least
        one usable score. Useful for surfacing every attribute breakdown
        in a single call (e.g. the triage report).
        """
        keys: set[str] = set()
        for cr in self.case_results:
            keys.update(cr.metadata.attributes.keys())
        out: dict[str, dict[str, float]] = {}
        for k in sorted(keys):
            scores = self.per_attribute_accuracy(k)
            if scores:
                out[k] = scores
        return out

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

        DataFrames are encoded via ``pandas.to_json`` and nested traces via the
        neutral trace model; the whole tree is walked by a pydantic
        ``TypeAdapter`` so the encoding can't drift from the dataclass fields.
        ``from_json`` is the inverse.
        """
        # One pass: ``dump_json`` serializes straight to JSON bytes (nested
        # pydantic ``TestCaseMetadata``'s ``set`` of tags canonicalizes to
        # JSON-native types exactly as ``dump_python(mode="json")`` did),
        # without materializing an intermediate Python tree for ``json.dumps``.
        return _EVALUATION_OUTPUT_ADAPTER.dump_json(self).decode()

    @classmethod
    def from_json(cls, s: str) -> EvaluationOutput:
        """Deserialize an :class:`EvaluationOutput` produced by :meth:`to_json`."""
        return _EVALUATION_OUTPUT_ADAPTER.validate_python(json.loads(s))


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


# --- Field serde for the types pydantic can't round-trip on its own ---------
#
# A ``TypeAdapter`` walks the ``EvaluationOutput`` tree and handles every plain
# field (and nested pydantic ``TestCaseMetadata``) automatically, so the leaf
# to/from-dict helpers are gone (round-2 F11 — no more field drift). Only three
# field shapes need explicit conversion, expressed as reusable ``Annotated``
# aliases so the fields keep their real static types.


def _df_from_payload(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else _df_from_json(value)


def _error_to_payload(error: Exception | None) -> str | None:
    # Exceptions don't survive JSON; persist the string representation.
    return None if error is None else f"{type(error).__name__}: {error}"


def _error_from_payload(value: Any) -> Exception | None:
    if value is None or isinstance(value, Exception):
        return value
    return RuntimeError(value)


def _dataset_run_to_payload(run: DatasetRunOutput | None) -> dict[str, Any] | None:
    return run.to_dict() if run is not None else None


def _dataset_run_from_payload(value: Any) -> DatasetRunOutput | None:
    if value is None or isinstance(value, DatasetRunOutput):
        return value
    return DatasetRunOutput.from_dict(value)


# A DataFrame that serializes through ``pandas.to_json`` (dtype-preserving
# ``orient="table"``, ``orient="split"`` for the empty frame). The validator
# precedes the serializer and the serializer declares ``return_type`` because
# pandas' C-level ``DataFrame`` is not pydantic-introspectable — without both,
# pydantic silently drops the serializer for such types.
DataFrameField = Annotated[
    pd.DataFrame,
    PlainValidator(_df_from_payload),
    PlainSerializer(_df_to_json, return_type=str, when_used="always"),
]
# A task exception, stored as ``"Type: message"``; restored as a RuntimeError
# carrying that string (the original type/traceback cannot be reconstructed).
ErrorField = Annotated[
    Exception | None,
    PlainValidator(_error_from_payload),
    PlainSerializer(_error_to_payload, return_type=str, when_used="always"),
]
# A nested run output, delegated to its own schema-versioned ``to_dict`` /
# ``from_dict`` rather than re-walked by this adapter.
DatasetRunField = Annotated[
    DatasetRunOutput | None,
    PlainSerializer(_dataset_run_to_payload),
    PlainValidator(_dataset_run_from_payload),
]

_EVALUATION_OUTPUT_ADAPTER: TypeAdapter[EvaluationOutput] = TypeAdapter(EvaluationOutput)
