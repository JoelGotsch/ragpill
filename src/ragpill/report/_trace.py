"""Render MLflow traces (or subtrees) as nested markdown bullets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ragpill.report._text import render_value, truncate

if TYPE_CHECKING:
    from mlflow.entities import Span, Trace


DEFAULT_REDACT_PATTERNS: tuple[str, ...] = (
    r"(?i)api[_-]?key",
    r"(?i)authorization",
)
"""Regex patterns matched against attribute keys / dict keys when ``redact=True``.

Kept intentionally narrow — secret redaction is opt-out, not a security barrier.
"""

REDACTED = "<redacted>"

# Span attributes that are MLflow internals; we surface them through dedicated
# rendering paths (inputs/outputs/type) rather than dumping them in the bullet.
_INTERNAL_ATTR_KEYS = {
    "mlflow.traceRequestId",
    "mlflow.spanType",
    "mlflow.spanInputs",
    "mlflow.spanOutputs",
    "mlflow.spanFunctionName",
}


def render_spans(
    trace: Trace | None,
    *,
    root_span_id: str | None = None,
    max_chars: int = 1500,
    filter_types: Iterable[str] | None = None,
    per_span_chars: int = 300,
    redact: bool = True,
    redact_patterns: Iterable[str] | None = None,
) -> str:
    """Render a trace (or a subtree) as nested markdown bullets.

    Args:
        trace: The MLflow trace to render. Returns an empty string if ``None``.
        root_span_id: When set, only render the subtree rooted at this span.
            Falls back to rendering nothing if no span with that id exists.
        max_chars: Total character budget for the rendered output. When the
            budget is hit a ``… (+N more spans)`` line is appended.
        filter_types: Optional iterable of MLflow ``SpanType`` values (as
            strings) to keep. Spans with non-matching types are dropped *unless*
            they expose at least one ``ragpill_*`` attribute. ``None`` keeps
            every span.
        per_span_chars: Per-span budget for inputs/outputs lines.
        redact: When True (default), values whose key matches one of
            ``redact_patterns`` are replaced with ``<redacted>``.
        redact_patterns: Override regex list for redaction. ``None`` uses
            :data:`DEFAULT_REDACT_PATTERNS`.

    Returns:
        A multi-line markdown string. Empty when ``trace`` is ``None`` or has
        no matching spans.
    """
    if trace is None or not trace.data.spans:
        return ""

    spans = list(trace.data.spans)
    children: dict[str | None, list[Span]] = {}
    for sp in spans:
        children.setdefault(sp.parent_id, []).append(sp)
    for kids in children.values():
        kids.sort(key=lambda s: s.start_time_ns)

    if root_span_id is not None:
        roots = [s for s in spans if s.span_id == root_span_id]
        if not roots:
            return ""
    else:
        roots = children.get(None, [])

    type_filter: set[str] | None = {str(t) for t in filter_types} if filter_types is not None else None
    patterns = tuple(redact_patterns) if redact_patterns is not None else DEFAULT_REDACT_PATTERNS
    compiled = [re.compile(p) for p in patterns] if redact else []

    rendered_lines: list[str] = []
    skipped = 0
    used = 0
    overflow = False

    def write(line: str) -> bool:
        nonlocal used, overflow
        candidate = line + "\n"
        if used + len(candidate) > max_chars:
            overflow = True
            return False
        rendered_lines.append(line)
        used += len(candidate)
        return True

    def visit(span: Span, depth: int) -> None:
        nonlocal skipped
        if overflow:
            skipped += 1
            return
        if not _span_kept(span, type_filter):
            skipped += 1
            for child in children.get(span.span_id, []):
                visit(child, depth)
            return
        pad = "  " * depth
        header = _format_span_header(span)
        if not write(f"{pad}- {header}"):
            skipped += 1
            return
        for sub in _format_span_body(span, compiled, per_span_chars):
            if not write(f"{pad}  {sub}"):
                # Body truncation isn't fatal — keep walking children.
                break
        for child in children.get(span.span_id, []):
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)

    if overflow and skipped > 0:
        # Best-effort tail; ignore budget here since the caller's outer truncate
        # may still trim the document.
        rendered_lines.append(f"… (+{skipped} more spans)")

    return "\n".join(rendered_lines)


def _span_kept(span: Span, type_filter: set[str] | None) -> bool:
    if type_filter is None:
        return True
    span_type = str(span.span_type) if span.span_type else ""
    if span_type in type_filter:
        return True
    # Keep spans that carry ragpill-managed attributes even if type doesn't match.
    for key in span.attributes or {}:
        if key.startswith("ragpill_"):
            return True
    return False


def _format_span_header(span: Span) -> str:
    span_type = str(span.span_type) if span.span_type else "UNKNOWN"
    duration_ms = (
        max(0, (span.end_time_ns - span.start_time_ns) // 1_000_000) if span.end_time_ns and span.start_time_ns else 0
    )
    return f"{span.name} ({span_type}, {duration_ms}ms)"


def _format_span_body(span: Span, redact_compiled: list[re.Pattern[str]], per_span_chars: int) -> list[str]:
    out: list[str] = []
    if span.inputs is not None:
        out.append(f"input: {render_value(_redact(span.inputs, redact_compiled), max_chars=per_span_chars)}")
    if span.outputs is not None:
        out.append(f"output: {render_value(_redact(span.outputs, redact_compiled), max_chars=per_span_chars)}")
    extras = _interesting_attributes(span.attributes or {}, redact_compiled)
    if extras:
        out.append(
            truncate("attrs: " + render_value(extras, max_chars=per_span_chars), per_span_chars + len("attrs: "))
        )
    return out


def _interesting_attributes(attrs: dict[str, Any], redact_compiled: list[re.Pattern[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in _INTERNAL_ATTR_KEYS:
            continue
        if any(p.search(k) for p in redact_compiled):
            result[k] = REDACTED
        else:
            result[k] = _redact(v, redact_compiled)
    return result


def _redact(value: Any, redact_compiled: list[re.Pattern[str]]) -> Any:
    if not redact_compiled:
        return value
    if isinstance(value, dict):
        items: list[tuple[Any, Any]] = list(value.items())  # pyright: ignore[reportUnknownArgumentType]
        return {
            k: REDACTED if any(p.search(str(k)) for p in redact_compiled) else _redact(v, redact_compiled)
            for k, v in items
        }
    if isinstance(value, list):
        return [_redact(item, redact_compiled) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value
