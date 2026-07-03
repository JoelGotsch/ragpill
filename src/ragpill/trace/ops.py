"""Operations over the vendor-neutral trace model.

Currently just :func:`filter_to_subtree`, the neutral replacement for the
MLflow-specific ``execution._filter_trace_to_subtree``. Works purely off the
``parent_id`` links on :class:`~ragpill.trace.Span`, so it has no backend
dependency.
"""

from __future__ import annotations

from dataclasses import replace

from ragpill.trace.model import Trace


def filter_to_subtree(trace: Trace, root_span_id: str) -> Trace | None:
    """Return a copy of ``trace`` with only the subtree rooted at ``root_span_id``.

    Returns ``None`` when no span with that id is present so callers can tell
    "span missing" apart from "empty subtree" and decide their own fallback.
    """
    spans = trace.spans
    if not any(s.span_id == root_span_id for s in spans):
        return None

    children: dict[str | None, list[str]] = {}
    for s in spans:
        children.setdefault(s.parent_id, []).append(s.span_id)

    included: set[str] = set()
    queue = [root_span_id]
    while queue:
        current = queue.pop()
        if current in included:
            continue
        included.add(current)
        queue.extend(children.get(current, []))

    return replace(trace, spans=[s for s in spans if s.span_id in included])
