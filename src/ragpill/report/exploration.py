"""Render a :class:`~ragpill.execution.DatasetRunOutput` as an exploration markdown view.

The exploration view answers "what did the agent do?" — it surfaces inputs,
outputs, and the trace tree per run, with no pass/fail opinion.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ragpill.report._text import render_value, truncate
from ragpill.report._trace import render_spans

if TYPE_CHECKING:
    from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput


_PER_CASE_BUDGET = 2_000
_PER_SPAN_VALUE_BUDGET = 300
_PER_RUN_SPAN_BUDGET = 500


def render_dataset_run_as_exploration(
    dr: DatasetRunOutput,
    *,
    max_chars: int = 16_000,
    include_spans: bool = True,
    redact: bool = True,
    redact_patterns: Iterable[str] | None = None,
) -> str:
    """Render a :class:`DatasetRunOutput` as an exploration-focused markdown document.

    Args:
        dr: The dataset run output to render.
        max_chars: Total character budget. When per-case content overflows
            ``_PER_CASE_BUDGET`` runs collapse to one-line summaries.
        include_spans: Include trace tree per run. ``False`` skips trace data.
        redact: Apply redaction to span inputs/outputs.
        redact_patterns: Override the default redaction regex list.

    Returns:
        A markdown string at most ``max_chars`` characters long.
    """
    header = _render_header(dr)
    case_blocks = [
        _render_case(c, include_spans=include_spans, redact=redact, redact_patterns=redact_patterns) for c in dr.cases
    ]
    document = "\n\n".join([header, *case_blocks])
    return truncate(document, max_chars)


def _render_header(dr: DatasetRunOutput) -> str:
    lines = [
        "# Dataset run",
        "",
        f"- Cases: {len(dr.cases)}",
        f"- Tracking URI: {dr.tracking_uri or '(none)'}",
        f"- Run: {dr.mlflow_run_id or '(none)'}",
        f"- Experiment: {dr.mlflow_experiment_id or '(none)'}",
    ]
    return "\n".join(lines)


def _render_case(
    case: CaseRunOutput,
    *,
    include_spans: bool,
    redact: bool,
    redact_patterns: Iterable[str] | None,
) -> str:
    title = render_value(case.case_name, max_chars=200)
    head = [
        f"## Case `{case.base_input_key}`: {title}",
        "",
        f"- Inputs: {render_value(case.inputs)}",
    ]
    if case.expected_output is not None:
        head.append(f"- Expected output: {render_value(case.expected_output)}")
    head.append(f"- Runs: {len(case.task_runs)}")

    full = list(head)
    for tr in case.task_runs:
        full.append("")
        full.extend(_render_run(tr, include_spans=include_spans, redact=redact, redact_patterns=redact_patterns))
    rendered = "\n".join(full)
    if len(rendered) <= _PER_CASE_BUDGET:
        return rendered

    # Collapse: one-line per run.
    collapsed = list(head)
    collapsed.append("")
    for tr in case.task_runs:
        collapsed.append(
            f"- Run {tr.run_index} (duration={tr.duration:.2f}s): {render_value(tr.output, max_chars=120)}"
        )
    return truncate("\n".join(collapsed), _PER_CASE_BUDGET)


def _render_run(
    tr: TaskRunOutput,
    *,
    include_spans: bool,
    redact: bool,
    redact_patterns: Iterable[str] | None,
) -> list[str]:
    output_summary = render_value(tr.output, max_chars=200)
    lines = [
        f"### Run {tr.run_index} (duration={tr.duration:.2f}s)",
        "",
        f"- Output: {output_summary}",
    ]
    if tr.error is not None:
        lines.append(f"- Error: `{tr.error}`")
    if include_spans and tr.trace is not None:
        spans_md = render_spans(
            tr.trace,
            root_span_id=tr.run_span_id or None,
            max_chars=_PER_RUN_SPAN_BUDGET,
            filter_types=None,
            per_span_chars=_PER_SPAN_VALUE_BUDGET,
            redact=redact,
            redact_patterns=redact_patterns,
        )
        if spans_md:
            lines.append("")
            lines.append(spans_md)
    return lines
