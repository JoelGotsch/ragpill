"""Dialect detection for a single normalised span dict."""

from __future__ import annotations

from typing import Any

from ragpill.trace.registry import select_adapter


def detect_dialect(span: dict[str, Any]) -> str | None:
    """Return the dialect name of the adapter that matches ``span``, or ``None``.

    Thin wrapper over :func:`ragpill.trace.registry.select_adapter` that returns
    just the adapter's ``name`` (e.g. ``"mlflow"``, ``"openinference"``,
    ``"gen_ai"``). ``None`` means no registered adapter recognised the span.
    """
    adapter = select_adapter(span)
    return adapter.name if adapter is not None else None
