"""Text primitives shared by the markdown renderers."""

from __future__ import annotations

import json
from typing import Any


def truncate(s: str, max_chars: int) -> str:
    """Clip ``s`` to ``max_chars`` with a visible suffix indicating bytes dropped.

    The suffix itself counts toward the budget so the returned string is
    always at most ``max_chars`` characters long.
    """
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    suffix_template = "… (+{n})"
    # Reserve space for the suffix; recompute dropped count for the actual cut.
    for reserve in range(8, max_chars):
        cut = max_chars - reserve
        if cut < 0:
            break
        suffix = suffix_template.format(n=len(s) - cut)
        if len(suffix) <= reserve:
            return s[:cut] + suffix
    # max_chars too small to fit any suffix — just hard-clip.
    return s[:max_chars]


def render_value(v: Any, max_chars: int = 300) -> str:
    """Stringify an arbitrary value into a single line, then clip to ``max_chars``.

    Strings are returned with surrounding whitespace collapsed; everything
    else is JSON-encoded with sorted keys (falling back to ``str()`` for
    non-serializable types). The result has no embedded newlines.
    """
    if isinstance(v, str):
        text = " ".join(v.split())
    else:
        try:
            text = json.dumps(v, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(v)
        text = " ".join(text.split())
    return truncate(text, max_chars)


def indent(s: str, levels: int) -> str:
    """Indent every non-empty line of ``s`` by ``levels`` levels (two spaces each)."""
    if levels <= 0 or not s:
        return s
    pad = "  " * levels
    return "\n".join(pad + line if line else line for line in s.splitlines())
