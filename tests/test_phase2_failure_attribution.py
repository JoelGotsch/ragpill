"""Phase 2: a trace that could not be read is an evaluator *error*, not a fail.

Pins B3: when a span-based evaluator's trace is unavailable (fetch timed out,
backend error, or the run's spans were still in flight), the run must be
recorded as an evaluator failure and excluded from accuracy — never scored as a
``False`` verdict that looks like a real regression.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ragpill.base import TestCaseMetadata, default_input_to_key
from ragpill.eval_types import Case, Dataset
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import RegexInSourcesEvaluator, TraceUnavailableError
from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput


def _case_run_without_trace(inputs: str) -> CaseRunOutput:
    # run_span_id set but no trace attached: mimics a fetch that timed out.
    tr = TaskRunOutput(run_index=0, input_key="k_0", output="out", duration=0.01, trace=None, run_span_id="run-0")
    return CaseRunOutput(
        case_name=inputs,
        inputs=inputs,
        expected_output=None,
        metadata={"attributes": {}, "tags": [], "expected": None, "repeat": None, "threshold": None},
        base_input_key=default_input_to_key(inputs),
        trace=None,
        trace_id="",
        task_runs=[tr],
    )


@pytest.mark.anyio
async def test_unavailable_trace_is_error_not_false():
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="anything")
    testset = Dataset[str, str, TestCaseMetadata](
        cases=[Case(inputs="q", metadata=TestCaseMetadata(), evaluators=[ev])]
    )
    dataset_run = DatasetRunOutput(cases=[_case_run_without_trace("q")])

    output = await evaluate_results(dataset_run, testset)
    rr = output.case_results[0].run_results[0]

    # Recorded as an evaluator error, not a passing/failing assertion.
    assert rr.assertions == {}
    assert len(rr.evaluator_failures) == 1
    assert "TraceUnavailableError" in rr.evaluator_failures[0].error_stacktrace

    # The runs-DataFrame row is NaN (excluded from accuracy), not False.
    row = output.runs.iloc[0]
    assert pd.isna(row["evaluator_result"])

    # Overall accuracy therefore has no valid rows to drag down.
    assert output.per_tag_accuracy() == {} or all(0.0 <= v <= 1.0 for v in output.per_tag_accuracy().values())


def test_get_trace_raises_on_missing_subtree_not_empty():
    from dataclasses import replace as _dc_replace

    from ragpill.base import EvaluatorMetadata
    from ragpill.eval_types import EvaluatorContext
    from ragpill.trace import Span, SpanKind, Trace

    root = Span(
        span_id="run-0", parent_id=None, trace_id="t", name="run-0", kind=SpanKind.CHAIN, start_time_ns=0, end_time_ns=0
    )
    trace = Trace(trace_id="t", spans=[root])
    ctx: EvaluatorContext[str, str, EvaluatorMetadata] = EvaluatorContext(
        name="c",
        inputs="i",
        metadata=EvaluatorMetadata(expected=True),
        expected_output=None,
        output="o",
        duration=0.0,
        trace=trace,
        run_span_id="missing-span",
    )
    ev = RegexInSourcesEvaluator.from_csv_line(expected=True, tags=set(), check="x")
    with pytest.raises(TraceUnavailableError):
        ev.get_trace(ctx)
    # Sanity: a present subtree returns normally (no false positive).
    present = _dc_replace(ctx, run_span_id="run-0")
    assert ev.get_trace(present).spans[0].span_id == "run-0"
