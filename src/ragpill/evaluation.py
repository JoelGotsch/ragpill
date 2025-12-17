"""Evaluate layer: run evaluators against a captured ``DatasetRunOutput``.

This layer has no MLflow server dependency. It consumes:

- A :class:`~ragpill.execution.DatasetRunOutput` produced by Phase 1
  (``execute_dataset``).
- A :class:`~ragpill.eval_types.Dataset` of evaluators (case-level and
  dataset-level).

It returns an :class:`~ragpill.types.EvaluationOutput` with ``runs`` and
``cases`` DataFrames plus the structured ``case_results`` for downstream
Phase 3 upload.
"""

from __future__ import annotations

import re
import traceback
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import anyio
import pandas as pd
from pydantic import TypeAdapter

from ragpill.backends import Backend
from ragpill.base import (
    BaseEvaluator,
    CaseMetadataT,
    EvaluatorMetadata,
    TestCaseMetadata,
    default_input_to_key,
    merge_metadata,
    resolve_repeat,
)
from ragpill.eval_types import (
    Case,
    Dataset,
    EvaluationResult,
    EvaluatorContext,
    EvaluatorSource,
)
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
from ragpill.settings import TrackingSettings
from ragpill.types import (
    AggregatedResult,
    CaseResult,
    EvaluationOutput,
    EvaluatorFailureInfo,
    RunResult,
)

_ta = TypeAdapter(dict[str, Any])

# Signature of the default ``object.__repr__`` (``<Foo object at 0x…>``), which
# makes ``str(inputs)`` — and therefore the input hash — process-unstable.
_DEFAULT_REPR_RE = re.compile(r" object at 0x[0-9a-fA-F]+")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_runs(run_results: list[RunResult], threshold: float) -> AggregatedResult:
    """Aggregate multiple :class:`RunResult` objects into a pass/fail verdict.

    Args:
        run_results: One :class:`RunResult` per run for a single case.
        threshold: Minimum fraction of passing runs required for the case to pass.

    Returns:
        :class:`AggregatedResult` with pass/fail, pass_rate, and per-evaluator rates.
    """
    # ADR-0018: the headline pass_rate is a conservative lower bound — every
    # run counts in the denominator and an error-state (infra-degraded) run
    # counts as a non-pass, so infra trouble can only push the number down.
    # pass_rate_evaluated is the diagnostic companion over scoreable runs.
    evaluable = [r for r in run_results if not r.is_error_state]
    total_runs = len(run_results)
    runs_infra_error = total_runs - len(evaluable)
    passed_count = sum(1 for r in evaluable if r.all_passed)
    pass_rate = passed_count / total_runs if total_runs > 0 else 0.0
    pass_rate_evaluated = passed_count / len(evaluable) if evaluable else 0.0
    # The gate fails on any infra-degraded run: an outage can block a
    # promotion, never improve one.
    passed = pass_rate >= threshold and runs_infra_error == 0

    # Every evaluator seen either as a verdict or as a failure.
    evaluator_names: set[str] = set()
    error_counts: dict[str, int] = {}
    for r in run_results:
        evaluator_names.update(r.assertions.keys())
        for f in r.evaluator_failures:
            evaluator_names.add(f.name)
            error_counts[f.name] = error_counts.get(f.name, 0) + 1

    # Per-evaluator denominator = runs in which that evaluator produced a
    # verdict; runs where it errored are excluded (not counted as failures).
    per_evaluator_pass_rates: dict[str, float] = {}
    for eval_name in sorted(evaluator_names):
        produced = [r for r in run_results if eval_name in r.assertions]
        eval_passed = sum(1 for r in produced if r.assertions[eval_name].value is True)
        per_evaluator_pass_rates[eval_name] = eval_passed / len(produced) if produced else 0.0

    infra_note = ""
    if runs_infra_error:
        infra_note = (
            f"; {runs_infra_error} run(s) infra-degraded"
            f" (pass_rate_evaluated={pass_rate_evaluated:.2f} over {len(evaluable)} evaluated)"
        )
    if passed:
        summary = f"{passed_count}/{total_runs} runs passed (threshold={threshold}){infra_note}"
    else:
        failed_details: list[str] = []
        for r in evaluable:
            if not r.all_passed:
                if r.error:
                    failed_details.append(f"run-{r.run_index}: task error: {r.error}")
                else:
                    failed_evals = [
                        f"{name}: {res.reason}" for name, res in r.assertions.items() if res.value is not True
                    ]
                    failed_details.append(f"run-{r.run_index}: {'; '.join(failed_evals)}")
        if runs_infra_error and pass_rate_evaluated >= threshold:
            failed_details.append("insufficient evaluated coverage (infra-degraded runs block the verdict)")
        summary = (
            f"{passed_count}/{total_runs} runs passed (threshold={threshold}){infra_note}. "
            f"Failed: {'; '.join(failed_details)}"
        )

    return AggregatedResult(
        passed=passed,
        pass_rate=pass_rate,
        threshold=threshold,
        summary=summary,
        per_evaluator_pass_rates=per_evaluator_pass_rates,
        error_counts=error_counts,
        pass_rate_evaluated=pass_rate_evaluated,
        runs_evaluated=len(evaluable),
        runs_infra_error=runs_infra_error,
    )


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


def _assign_unique_names(evaluators: list[BaseEvaluator]) -> list[str]:
    """Assign a stable assertion name to each evaluator, aligned by index.

    Duplicate class names get ``_2``, ``_3`` suffixes in evaluator order. The
    names are computed from the *full* evaluator list up front — independent of
    which evaluators raise at run time — so a judge that rate-limits on one run
    can't shift another judge's identity across runs (which would make
    cross-run aggregation blend distinct rubrics). Counting is by exact class
    name, so ``Regex`` and ``RegexInOutputEvaluator`` never collide.
    """
    seen: dict[str, int] = {}
    names: list[str] = []
    for ev in evaluators:
        base = ev.get_serialization_name()
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def _source_for(evaluator: BaseEvaluator) -> EvaluatorSource:
    return EvaluatorSource(
        name=evaluator.get_serialization_name(),
        arguments={"evaluation_name": str(evaluator.evaluation_name)},
        source_type=evaluator.source_type,
    )


def _build_ctx(
    case: Case[Any, Any, Any], task_run: TaskRunOutput, case_run: CaseRunOutput
) -> EvaluatorContext[Any, Any, Any]:
    return EvaluatorContext(
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=task_run.output,
        duration=task_run.duration,
        attributes={},
        metrics={},
        trace=task_run.trace if task_run.trace is not None else case_run.trace,
        run_span_id=task_run.run_span_id,
        trace_status=task_run.trace_status,
    )


async def _run_one_evaluator(
    evaluator: BaseEvaluator, eval_name: str, ctx: EvaluatorContext[Any, Any, Any]
) -> tuple[EvaluationResult | None, EvaluatorFailureInfo | None]:
    """Run a single evaluator; return either its result or a failure record.

    This is the unit of concurrency: the dominant latency (LLM judges) is one
    ``evaluate`` call, so parallelizing at this granularity — not just per run —
    is what makes ``max_concurrency`` speed up judge-heavy testsets.
    """
    try:
        result = await evaluator.evaluate(ctx)
        return (
            EvaluationResult(name=eval_name, value=result.value, reason=result.reason, source=_source_for(evaluator)),
            None,
        )
    except Exception as e:
        return None, EvaluatorFailureInfo(name=eval_name, error_message=str(e), error_stacktrace=traceback.format_exc())


@dataclass
class _RunSlot:
    """Mutable accumulator for one ``(case, run)`` while its evaluators run."""

    cpos: int
    task_run: TaskRunOutput
    eval_names: list[str]
    assertions: dict[str, EvaluationResult] = field(default_factory=dict)
    failures: dict[str, EvaluatorFailureInfo] = field(default_factory=dict)
    error: Exception | None = None

    def to_run_result(self) -> RunResult:
        # Rebuild assertions AND failures in the evaluators' declared order so
        # output is deterministic regardless of the order concurrent workers
        # finished in.
        ordered = {name: self.assertions[name] for name in self.eval_names if name in self.assertions}
        ordered_failures = [self.failures[name] for name in self.eval_names if name in self.failures]
        return RunResult(
            run_index=self.task_run.run_index,
            input_key=self.task_run.input_key,
            run_span_id=self.task_run.run_span_id,
            trace_id=self.task_run.trace_id,
            output=None if self.error is not None else self.task_run.output,
            duration=self.task_run.duration,
            assertions=ordered,
            evaluator_failures=ordered_failures,
            error=self.error,
        )


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def _get_eval_metadata_for_case(
    cr: CaseResult,
    eval_result: EvaluationResult,
    metadata_by_eval_id: Mapping[str, EvaluatorMetadata] | None = None,
) -> EvaluatorMetadata:
    """Look up :class:`~ragpill.base.EvaluatorMetadata` for a given evaluator result.

    The evaluator is located via the ``evaluation_name`` uuid recorded on the
    result's source, so its own tags/attributes/expected participate in the
    merge. Falls back to a case-metadata-only default when the evaluator can't
    be located (e.g. results deserialized without the testset).
    """
    eval_uuid: str = ""
    if eval_result.source:
        eval_uuid = str(eval_result.source.arguments.get("evaluation_name", ""))
    found = (metadata_by_eval_id or {}).get(eval_uuid)
    if found is not None:
        return found
    return EvaluatorMetadata(
        expected=True,
        attributes=cr.metadata.attributes,
        tags=cr.metadata.tags,
        is_global_evaluator=False,
        other_evaluator_data=f"eval_uuid={eval_uuid}",
    )


def _create_runs_dataframe(
    case_results: list[CaseResult],
    metadata_by_eval_id: Mapping[str, EvaluatorMetadata] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with one row per ``(run, evaluator)``.

    ``metadata_by_eval_id`` maps ``str(evaluator.evaluation_name)`` to the
    evaluator's own metadata so tags/attributes merge per ``merge_metadata``;
    without it, rows carry case metadata only.
    """
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for rr in cr.run_results:
            for eval_name, eval_result in rr.assertions.items():
                eval_metadata_map = _get_eval_metadata_for_case(cr, eval_result, metadata_by_eval_id)
                merged_metadata = merge_metadata(cr.metadata, eval_metadata_map)
                source_type = eval_result.source.source_type
                rows.append(
                    {
                        "inputs": str(cr.inputs),
                        "output": str(rr.output),
                        "evaluator_result": eval_result.value,
                        "evaluator_data": merged_metadata.other_evaluator_data,
                        "evaluator_reason": eval_result.reason,
                        "expected": merged_metadata.expected,
                        "attributes": _ta.dump_json(merged_metadata.attributes),
                        "tags": merged_metadata.tags,
                        "task_duration": rr.duration,
                        "evaluator_name": eval_name,
                        "case_name": cr.case_name,
                        "case_id": cr.base_input_key,
                        "run_index": rr.run_index,
                        "repeat_total": len(cr.run_results),
                        "threshold": cr.aggregated.threshold,
                        "source_type": source_type,
                        "source_id": eval_result.source.name,
                        "input_key": rr.input_key,
                        "trace_id": cr.trace_id,
                    }
                )
            for ef in rr.evaluator_failures:
                rows.append(
                    {
                        "inputs": str(cr.inputs),
                        "output": str(rr.output),
                        # None (NaN), not False: an evaluator that could not run
                        # (raised, or its trace was unavailable) must be excluded
                        # from accuracy denominators — matching RunResult.all_passed
                        # and per_tag_accuracy — so infra failures don't depress the
                        # score. The failure is still surfaced via this row's reason
                        # and the triage report.
                        "evaluator_result": None,
                        "evaluator_data": "",
                        "evaluator_reason": f"Evaluator failed: {ef.error_message}\n\n{ef.error_stacktrace}",
                        "expected": True,
                        "attributes": _ta.dump_json(cr.metadata.attributes),
                        "tags": cr.metadata.tags,
                        "task_duration": rr.duration,
                        "evaluator_name": ef.name,
                        "case_name": cr.case_name,
                        "case_id": cr.base_input_key,
                        "run_index": rr.run_index,
                        "repeat_total": len(cr.run_results),
                        "threshold": cr.aggregated.threshold,
                        "source_type": "CODE",
                        "source_id": ef.name,
                        "input_key": rr.input_key,
                        "trace_id": cr.trace_id,
                    }
                )
    return pd.DataFrame(rows)


def _create_cases_dataframe(case_results: list[CaseResult]) -> pd.DataFrame:
    """Build a DataFrame with one row per ``(case, evaluator)``, aggregated across runs."""
    rows: list[dict[str, Any]] = []
    for cr in case_results:
        assert isinstance(cr.metadata, TestCaseMetadata)
        for eval_name, rate_evaluated in cr.aggregated.per_evaluator_pass_rates.items():
            durations = [rr.duration for rr in cr.run_results]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            errored = cr.aggregated.error_counts.get(eval_name, 0)
            produced = sum(1 for rr in cr.run_results if eval_name in rr.assertions)
            passed_n = sum(
                1 for rr in cr.run_results if rr.assertions.get(eval_name) and rr.assertions[eval_name].value is True
            )
            # ADR-0018: "pass_rate" is the conservative, gateable number
            # (errored runs count against); the evaluated-only rate is
            # diagnostic and carries its coverage in the adjacent columns.
            attempts = produced + errored
            rate_conservative = passed_n / attempts if attempts else 0.0
            rows.append(
                {
                    "case_id": cr.base_input_key,
                    "case_name": cr.case_name,
                    "repeat_total": len(cr.run_results),
                    "threshold": cr.aggregated.threshold,
                    "inputs": str(cr.inputs),
                    "evaluator_name": eval_name,
                    "pass_rate": rate_conservative,
                    "pass_rate_evaluated": rate_evaluated,
                    "runs_evaluated": produced,
                    "runs_errored": errored,
                    "passed": rate_conservative >= cr.aggregated.threshold and errored == 0,
                    "aggregated_reason": cr.aggregated.summary,
                    "expected": True,
                    "attributes": _ta.dump_json(cr.metadata.attributes),
                    "tags": cr.metadata.tags,
                    "avg_task_duration": avg_duration,
                    "trace_id": cr.trace_id,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def evaluate_results(
    dataset_run: DatasetRunOutput,
    testset: Dataset[Any, Any, CaseMetadataT],
    settings: TrackingSettings | None = None,
    *,
    max_concurrency: int = 1,
    backend: Backend | None = None,
) -> EvaluationOutput:
    """Run evaluators against a captured :class:`DatasetRunOutput`.

    For every ``(case, task_run)`` pair, builds an
    :class:`~ragpill.eval_types.EvaluatorContext` with the captured trace and
    runs every case-level + dataset-level evaluator. Results are aggregated
    per case using ``threshold`` from settings (or per-case override).

    Args:
        dataset_run: Output of :func:`ragpill.execution.execute_dataset`.
        testset: The dataset whose cases align one-for-one with
            ``dataset_run.cases``. Evaluators come from ``case.evaluators`` and
            ``testset.evaluators``.
        settings: Global :class:`TrackingSettings`. Only ``repeat`` and
            ``threshold`` are consulted — no MLflow connection is made.
        max_concurrency: Upper bound on how many ``(case, run)`` evaluations run
            concurrently. Defaults to ``1`` (fully sequential, unchanged
            behaviour). Raise it to overlap judge-heavy testsets — evaluation
            has no capture-time ordering constraint. Results are identical
            regardless of the value.
        backend: Accepted for pipeline symmetry with
            :func:`ragpill.execution.execute_dataset` and
            :func:`ragpill.upload.upload_results`. Evaluation itself never
            contacts the tracking backend (traces are read from
            ``dataset_run``), so the value is currently unused; span capture
            inside LLM-judge evaluators still resolves the process registry.

    Returns:
        :class:`EvaluationOutput` with ``.runs``, ``.cases``, and
        ``.case_results``.

    Example:
        ```python
        run_output = await execute_dataset(testset, task=my_task)
        eval_output = await evaluate_results(run_output, testset)
        print(eval_output.summary)
        ```

    See Also:
        [`execute_dataset`][ragpill.execution.execute_dataset]: Phase 1.
        [`upload_results`][ragpill.upload.upload_results]: Phase 3.
    """
    _settings = settings or TrackingSettings()  # pyright: ignore[reportCallIssue]
    # Reserved seam (see the docstring): evaluation reads captured traces from
    # ``dataset_run`` and never contacts the tracking backend, so ``backend``
    # is accepted but not resolved — resolving it here would force a backend to
    # exist for a deliberately server-free layer.
    _ = backend

    if len(dataset_run.cases) != len(testset.cases):
        raise ValueError(f"dataset_run has {len(dataset_run.cases)} cases but testset has {len(testset.cases)}")

    # Guard against silent misalignment: the disconnected workflow (execute →
    # save JSON → later evaluate against a re-loaded CSV) pairs runs to cases by
    # index. A reordered or edited testset would otherwise judge each output
    # against the wrong case's evaluators/expected/rubric. Both sides carry the
    # input hash, so verify identity per case before trusting the zip.
    #
    # The hash is ``md5(str(inputs))``, which is only stable when the input has a
    # deterministic ``str()``. For objects using the default ``object.__repr__``
    # (embedding the memory address, e.g. ``<Foo object at 0x…>``) the key was
    # never stable across processes, so a mismatch there is meaningless — skip
    # the check for those cases (with a one-time warning) rather than hard-fail a
    # valid saved run.
    mismatches: list[str] = []
    skipped_unstable = 0
    for idx, (case_run, case) in enumerate(zip(dataset_run.cases, testset.cases)):
        if _DEFAULT_REPR_RE.search(str(case.inputs)) is not None:
            skipped_unstable += 1
            continue
        expected_key = default_input_to_key(case.inputs)
        if case_run.base_input_key != expected_key:
            mismatches.append(
                f"  index {idx}: run captured inputs keyed {case_run.base_input_key!r} "
                f"(case {case_run.case_name!r}) but testset case {case.name or str(case.inputs)!r} "
                f"hashes to {expected_key!r}"
            )
    if skipped_unstable:
        warnings.warn(
            f"evaluate_results: input-identity verification skipped for {skipped_unstable} case(s) whose "
            "inputs have no stable str()/repr (default object repr embeds a memory address). Give such "
            "inputs a deterministic __repr__ (or use a dataclass/pydantic model) to re-enable the alignment "
            "check for the disconnected execute→save→evaluate workflow.",
            stacklevel=2,
        )
    if mismatches:
        raise ValueError(
            "dataset_run cases do not align with testset cases by input identity — the "
            "testset was likely reordered or edited since the run was captured. "
            "Re-run execute_dataset against this testset, or restore the original testset.\n" + "\n".join(mismatches)
        )

    # Per-case setup (synchronous): resolve evaluators, metadata, and threshold.
    # Evaluator metadata by evaluation_name uuid lets the runs DataFrame merge
    # each evaluator's own tags/attributes into its rows.
    metadata_by_eval_id: dict[str, EvaluatorMetadata] = {}
    per_case: list[tuple[Any, Case[Any, Any, Any], list[BaseEvaluator], TestCaseMetadata | None, float]] = []
    for case_run, case in zip(dataset_run.cases, testset.cases):
        evaluators: list[BaseEvaluator] = [*case.evaluators, *testset.evaluators]
        metadata_by_eval_id.update({str(ev.evaluation_name): ev.metadata for ev in evaluators})
        case_metadata: TestCaseMetadata | None = case.metadata if isinstance(case.metadata, TestCaseMetadata) else None
        _, threshold = resolve_repeat(case_metadata, _settings)
        per_case.append((case_run, case, evaluators, case_metadata, threshold))

    # Build one slot per (case, run) and flatten to a per-evaluator job list.
    # Task-error runs short-circuit every evaluator to failure with no jobs.
    slots: list[_RunSlot] = []
    jobs: list[tuple[_RunSlot, BaseEvaluator, str, EvaluatorContext[Any, Any, Any]]] = []
    for cpos, (case_run, case, evaluators, _cm, _th) in enumerate(per_case):
        eval_names = _assign_unique_names(evaluators)
        for task_run in case_run.task_runs:
            slot = _RunSlot(cpos=cpos, task_run=task_run, eval_names=eval_names)
            if task_run.error is not None:
                slot.error = RuntimeError(task_run.error)
                for evaluator, eval_name in zip(evaluators, eval_names):
                    slot.assertions[eval_name] = EvaluationResult(
                        name=eval_name,
                        value=False,
                        reason=f"Task execution failed: {task_run.error}",
                        source=_source_for(evaluator),
                    )
            else:
                ctx = _build_ctx(case, task_run, case_run)
                for evaluator, eval_name in zip(evaluators, eval_names):
                    jobs.append((slot, evaluator, eval_name, ctx))
            slots.append(slot)

    # Run every evaluator job bounded by ``max_concurrency`` (one shared limiter,
    # so per-run evaluators overlap too). ``max_concurrency=1`` runs one at a
    # time; slots collect results by name, so output is order-independent.
    limiter = anyio.CapacityLimiter(max(1, max_concurrency))

    async def _worker(
        slot: _RunSlot, evaluator: BaseEvaluator, eval_name: str, ctx: EvaluatorContext[Any, Any, Any]
    ) -> None:
        async with limiter:
            result, failure = await _run_one_evaluator(evaluator, eval_name, ctx)
        if result is not None:
            slot.assertions[eval_name] = result
        if failure is not None:
            slot.failures[eval_name] = failure

    async with anyio.create_task_group() as tg:
        for slot, evaluator, eval_name, ctx in jobs:
            tg.start_soon(_worker, slot, evaluator, eval_name, ctx)

    runs_by_case: dict[int, list[RunResult]] = {i: [] for i in range(len(per_case))}
    for slot in slots:
        runs_by_case[slot.cpos].append(slot.to_run_result())

    case_results: list[CaseResult] = []
    for cpos, (case_run, case, evaluators, case_metadata, threshold) in enumerate(per_case):
        run_results = runs_by_case[cpos]
        aggregated = _aggregate_runs(run_results, threshold)
        # CaseResult demands a TestCaseMetadata; fall back to an empty one.
        metadata_obj = case_metadata or TestCaseMetadata()
        case_results.append(
            CaseResult(
                case_name=case_run.case_name,
                inputs=case_run.inputs,
                metadata=metadata_obj,
                base_input_key=case_run.base_input_key,
                trace_id=case_run.trace_id,
                run_results=run_results,
                aggregated=aggregated,
            )
        )

    runs_df = _create_runs_dataframe(case_results, metadata_by_eval_id)
    cases_df = _create_cases_dataframe(case_results)
    return EvaluationOutput(
        runs=runs_df,
        cases=cases_df,
        case_results=case_results,
        dataset_run=dataset_run,
    )


__all__ = ["evaluate_results"]
