"""Render an :class:`~ragpill.types.EvaluationOutput` as a triage-focused markdown document.

The output is optimized for "why did this evaluation fail?" — failing cases
appear first with per-evaluator reasons and the relevant trace subtree;
passing cases collapse to a single bullet each.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import pandas as pd

from ragpill.backends import SpanKind
from ragpill.report._text import render_value, truncate
from ragpill.report._trace import render_spans

if TYPE_CHECKING:
    from ragpill.execution import CaseRunOutput, DatasetRunOutput, TaskRunOutput
    from ragpill.types import CaseResult, EvaluationOutput, RunResult


# Span types we surface in the "Relevant spans" subsection of a failing run.
DEFAULT_TRIAGE_SPAN_TYPES: tuple[str, ...] = (
    str(SpanKind.RETRIEVER),
    str(SpanKind.TOOL),
    str(SpanKind.LLM),
    str(SpanKind.RERANKER),
    str(SpanKind.CHAT_MODEL),
    str(SpanKind.AGENT),
)

_PER_RUN_SPAN_BUDGET = 1500
_PER_SPAN_VALUE_BUDGET = 300


def render_evaluation_output_as_triage(
    eo: EvaluationOutput,
    *,
    max_chars: int = 32_000,
    include_passing: bool = True,
    include_spans: bool = True,
    redact: bool = True,
    redact_patterns: Iterable[str] | None = None,
) -> str:
    """Render an :class:`EvaluationOutput` as a triage-focused markdown document.

    Args:
        eo: The evaluation output to render.
        max_chars: Total character budget. When exceeded the renderer drops
            sections in this order: passing-case section, run-detail past
            index 5, then trailing failing cases.
        include_passing: Include the collapsed passing-case section.
        include_spans: Include "Relevant spans" subsections for failing runs.
            When ``False`` no trace data is touched.
        redact: Apply redaction to span inputs/outputs (see
            :func:`ragpill.report._trace.render_spans`).
        redact_patterns: Override the default redaction regex list.

    Returns:
        A markdown string at most ``max_chars`` characters long.
    """
    failing, passing = _split_cases(eo.case_results)
    failing.sort(key=lambda c: (c.aggregated.pass_rate, c.base_input_key))

    case_traces = _build_case_trace_index(eo.dataset_run)

    # Render in stages so we can shed sections under budget pressure.
    header = _render_header(eo)
    failing_blocks = [
        _render_failing_case(
            cr,
            case_traces.get(cr.base_input_key),
            include_spans=include_spans,
            collapse_runs=False,
            redact=redact,
            redact_patterns=redact_patterns,
        )
        for cr in failing
    ]
    passing_block = _render_passing_section(passing) if include_passing and passing else ""

    document = _assemble(header, failing_blocks, passing_block)
    if len(document) <= max_chars:
        return document

    # Step 1: drop passing-case section.
    if passing_block:
        document = _assemble(header, failing_blocks, "")
        if len(document) <= max_chars:
            return document

    # Step 2: collapse run detail on failing cases past index 5.
    collapsed = list(failing_blocks)
    for i in range(5, len(failing)):
        collapsed[i] = _render_failing_case(
            failing[i],
            case_traces.get(failing[i].base_input_key),
            include_spans=False,
            collapse_runs=True,
            redact=redact,
            redact_patterns=redact_patterns,
        )
    document = _assemble(header, collapsed, "")
    if len(document) <= max_chars:
        return document

    # Step 3: drop trailing failing cases.
    kept: list[str] = []
    running = len(header) + 2  # header trailing newlines
    failing_header = "\n## Failing cases\n\n"
    running += len(failing_header)
    for i, block in enumerate(collapsed):
        if running + len(block) + 50 > max_chars:
            kept.append(f"\n_… ({len(collapsed) - i} additional failing cases not shown)_\n")
            break
        kept.append(block)
        running += len(block)
    return truncate(_assemble(header, kept, "", _no_split_failing=True), max_chars)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_cases(case_results: list[CaseResult]) -> tuple[list[CaseResult], list[CaseResult]]:
    failing = [c for c in case_results if not c.aggregated.passed]
    passing = [c for c in case_results if c.aggregated.passed]
    return failing, passing


def _render_header(eo: EvaluationOutput) -> str:
    total_cases = len(eo.case_results)
    passed_cases = sum(1 for c in eo.case_results if c.aggregated.passed)
    failed_cases = total_cases - passed_cases
    pass_rate = (passed_cases / total_cases * 100) if total_cases else 0.0

    rollup = _per_evaluator_rollup(eo.case_results)
    lines = [
        "# Evaluation summary",
        "",
        f"- Total cases: {total_cases} ({passed_cases} passed, {failed_cases} failed)",
        f"- Overall pass rate: {pass_rate:.1f}% (threshold applied per-case)",
    ]
    if rollup:
        lines.append("- Evaluator rollup:")
        for name, (passed, total) in rollup.items():
            lines.append(f"  - `{name}` — {passed}/{total} passed")

    tag_block = _render_tag_breakdown(eo)
    if tag_block:
        lines.append("")
        lines.append(tag_block)
    attr_block = _render_attribute_breakdown(eo)
    if attr_block:
        lines.append("")
        lines.append(attr_block)
    return "\n".join(lines)


def _render_attribute_breakdown(eo: EvaluationOutput) -> str:
    """Render per-attribute-value pass-rate tables, one section per attribute.

    Skips attributes with fewer than two distinct values across the dataset —
    a single value carries no breakdown signal. Empty string when no
    qualifying attribute exists.
    """
    per_attr = eo.per_attribute_accuracy_all()
    if not per_attr:
        return ""
    counts = _attribute_run_counts(eo)
    sections: list[str] = []
    for attr_key in sorted(per_attr):
        value_map = per_attr[attr_key]
        if len(value_map) < 2:
            continue
        rows = sorted(value_map.items(), key=lambda item: (item[1], item[0]))
        block = [f"### `{attr_key}`", "", "| Value | Pass rate | n |", "|---|---|---|"]
        for value, accuracy in rows:
            n = counts.get(attr_key, {}).get(value, 0)
            block.append(f"| `{value}` | {accuracy * 100:.1f}% | {n} |")
        sections.append("\n".join(block))
    if not sections:
        return ""
    return "## Pass rate by attribute\n\n" + "\n\n".join(sections)


def _attribute_run_counts(eo: EvaluationOutput) -> dict[str, dict[str, int]]:
    """Return ``{attribute_key: {value: number_of_scoreable_rows}}``."""
    counts: dict[str, dict[str, int]] = {}
    for cr in eo.case_results:
        for attr_key, raw_value in cr.metadata.attributes.items():
            value = str(raw_value)
            n_scoreable = sum(
                1 for rr in cr.run_results for r in rr.assertions.values() if isinstance(r.value, (bool, int, float))
            )
            counts.setdefault(attr_key, {}).setdefault(value, 0)
            counts[attr_key][value] += n_scoreable
    return counts


def _render_tag_breakdown(eo: EvaluationOutput) -> str:
    """Render a markdown table of pass rate per tag, sorted worst-first.

    Returns an empty string when the dataset has no tags or no usable runs.
    """
    per_tag = eo.per_tag_accuracy()
    if not per_tag:
        return ""
    counts = _tag_run_counts(eo)
    rows = sorted(per_tag.items(), key=lambda item: (item[1], item[0]))
    lines = ["## Pass rate by tag", "", "| Tag | Pass rate | n |", "|---|---|---|"]
    for tag, accuracy in rows:
        n = counts.get(tag, 0)
        lines.append(f"| `{tag}` | {accuracy * 100:.1f}% | {n} |")
    return "\n".join(lines)


def _tag_run_counts(eo: EvaluationOutput) -> dict[str, int]:
    """Return ``{tag: number_of_rows}`` after exploding the runs ``tags`` column."""
    if eo.runs.empty or "evaluator_result" not in eo.runs.columns:
        return {}
    runs: Any = eo.runs
    df_valid = runs[runs["evaluator_result"].notna()]
    if df_valid.empty:
        return {}
    grouped = df_valid.explode("tags").groupby("tags").size()
    return {str(tag): int(count) for tag, count in grouped.items() if pd.notna(tag)}  # pyright: ignore[reportUnknownMemberType]


def _per_evaluator_rollup(case_results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """Return ``{evaluator_name: (passed_runs, total_runs)}`` across the dataset."""
    counts: dict[str, list[int]] = {}
    for cr in case_results:
        for rr in cr.run_results:
            for name, result in rr.assertions.items():
                bucket = counts.setdefault(name, [0, 0])
                bucket[1] += 1
                if result.value is True:
                    bucket[0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(counts.items())}


def _build_case_trace_index(
    dataset_run: DatasetRunOutput | None,
) -> dict[str, CaseRunOutput]:
    if dataset_run is None:
        return {}
    return {c.base_input_key: c for c in dataset_run.cases}


def _render_failing_case(
    cr: CaseResult,
    case_run: CaseRunOutput | None,
    *,
    include_spans: bool,
    collapse_runs: bool,
    redact: bool,
    redact_patterns: Iterable[str] | None,
) -> str:
    lines: list[str] = []
    title = render_value(cr.case_name, max_chars=200)
    pass_count = sum(1 for rr in cr.run_results if rr.all_passed)
    total = len(cr.run_results)
    lines.append(f"### Case `{cr.base_input_key}`: {title}")
    lines.append("")
    lines.append(f"- Pass rate: {pass_count}/{total} runs")
    lines.append(f"- Inputs: {render_value(cr.inputs)}")
    expected = _expected_output_for(cr, case_run)
    if expected is not None:
        lines.append(f"- Expected output: {render_value(expected)}")
    lines.append(f"- Threshold: {cr.aggregated.threshold:.2f}")

    if collapse_runs:
        lines.append(f"- Summary: {cr.aggregated.summary}")
        return "\n".join(lines)

    failing_runs = [rr for rr in cr.run_results if not rr.all_passed]
    for rr in failing_runs:
        lines.append("")
        lines.extend(
            _render_failing_run(
                rr, case_run, include_spans=include_spans, redact=redact, redact_patterns=redact_patterns
            )
        )
    return "\n".join(lines)


def _expected_output_for(_cr: CaseResult, case_run: CaseRunOutput | None) -> Any:
    if case_run is None:
        return None
    return case_run.expected_output


def _render_failing_run(
    rr: RunResult,
    case_run: CaseRunOutput | None,
    *,
    include_spans: bool,
    redact: bool,
    redact_patterns: Iterable[str] | None,
) -> list[str]:
    failing_count = sum(1 for r in rr.assertions.values() if r.value is not True)
    total_count = len(rr.assertions)
    lines: list[str] = [
        f"#### Run {rr.run_index} — FAIL ({total_count} assertions; {failing_count} failing)",
        "",
    ]
    if rr.error is not None:
        lines.append(f"- Task error: `{type(rr.error).__name__}: {rr.error}`")
    lines.append(f"- Output: {render_value(rr.output)}")
    for name, result in rr.assertions.items():
        verdict = "PASS" if result.value is True else "FAIL"
        reason = f" — {render_value(result.reason)}" if result.reason else ""
        lines.append(f"- `{name}`: **{verdict}**{reason}")

    if include_spans and case_run is not None:
        task_run = _find_task_run(case_run, rr)
        spans_md = _render_relevant_spans(task_run, redact=redact, redact_patterns=redact_patterns)
        if spans_md:
            lines.append("")
            lines.append("##### Relevant spans")
            lines.append("")
            lines.append(spans_md)
    return lines


def _find_task_run(case_run: CaseRunOutput, rr: RunResult) -> TaskRunOutput | None:
    for tr in case_run.task_runs:
        if tr.input_key == rr.input_key:
            return tr
    if 0 <= rr.run_index < len(case_run.task_runs):
        return case_run.task_runs[rr.run_index]
    return None


def _render_relevant_spans(
    task_run: TaskRunOutput | None,
    *,
    redact: bool,
    redact_patterns: Iterable[str] | None,
) -> str:
    if task_run is None or task_run.trace is None:
        return ""
    return render_spans(
        task_run.trace,
        root_span_id=task_run.run_span_id or None,
        max_chars=_PER_RUN_SPAN_BUDGET,
        filter_types=DEFAULT_TRIAGE_SPAN_TYPES,
        per_span_chars=_PER_SPAN_VALUE_BUDGET,
        redact=redact,
        redact_patterns=redact_patterns,
    )


def _render_passing_section(passing: list[CaseResult]) -> str:
    lines = ["## Passing cases (collapsed)", ""]
    for cr in passing:
        passed = sum(1 for rr in cr.run_results if rr.all_passed)
        total = len(cr.run_results)
        title = render_value(cr.case_name, max_chars=200)
        lines.append(f"- `{cr.base_input_key}`: {title} — {passed}/{total} runs passed")
    return "\n".join(lines)


def _assemble(
    header: str,
    failing_blocks: list[str],
    passing_block: str,
    *,
    _no_split_failing: bool = False,
) -> str:
    parts = [header]
    if failing_blocks:
        parts.append("")
        parts.append("## Failing cases")
        parts.append("")
        parts.append("\n\n".join(failing_blocks))
    if passing_block:
        parts.append("")
        parts.append(passing_block)
    return "\n".join(parts)
