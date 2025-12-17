"""Adapter registry + per-span dialect selection.

Holds the known :class:`~ragpill.trace.adapters.SpanAdapter` implementations in
priority order and picks one for a given normalised span dict by matching its
``signature_attributes()``. Built-ins are registered in code (no reinstall
needed); third-party adapters are additionally discovered from the
``ragpill.trace_adapters`` entry-point group.

Priority order (highest first) follows ``designs/otel-trace-ingestion.md`` §6.4:
the more specific / wrapping dialects win ties. Phase 3 ships mlflow,
openinference, and gen_ai; the langfuse / openllmetry / logfire slots land in
Phase 4 and simply insert into this list.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from ragpill.trace.adapters._base import SpanAdapter
from ragpill.trace.adapters.gen_ai import GenAIAdapter
from ragpill.trace.adapters.mlflow_adapter import MLflowAdapter
from ragpill.trace.adapters.openinference import OpenInferenceAdapter

# Highest priority first. mlflow wraps other dialects (its autolog can co-emit
# gen_ai events), so it is matched before gen_ai; openinference is more specific
# than the bare gen_ai convention.
_BUILTIN_PRIORITY: tuple[type[SpanAdapter], ...] = (
    MLflowAdapter,
    OpenInferenceAdapter,
    GenAIAdapter,
)


def _discover_entry_point_adapters() -> list[type[SpanAdapter]]:
    """Load third-party adapters registered under ``ragpill.trace_adapters``.

    Best-effort: a broken entry point is skipped rather than failing the import.
    Discovered adapters are appended after the built-ins (lower priority).
    """
    from importlib.metadata import entry_points

    found: list[type[SpanAdapter]] = []
    try:
        eps = entry_points(group="ragpill.trace_adapters")
    except Exception:  # pragma: no cover - importlib API variance across versions
        return found
    for ep in eps:
        try:
            adapter = ep.load()
        except Exception:  # pragma: no cover - a bad plugin shouldn't break ingestion
            continue
        if isinstance(adapter, type) and issubclass(adapter, SpanAdapter):
            found.append(adapter)
    return found


@cache
def _entry_point_adapters_cached() -> tuple[type[SpanAdapter], ...]:
    # Entry-point discovery scans installed distributions — expensive to repeat.
    # parse_otel calls select_adapter once per span, so without caching a
    # 500-span trace triggers 500 importlib scans. The plugin set is fixed for
    # the process; clear_adapter_cache() resets it for tests.
    return tuple(_discover_entry_point_adapters())


def clear_adapter_cache() -> None:
    """Drop the cached entry-point adapter discovery (tests register plugins)."""
    _entry_point_adapters_cached.cache_clear()


def adapters_in_priority_order() -> list[type[SpanAdapter]]:
    """Return built-in adapters (priority order) followed by any plugins."""
    return [*_BUILTIN_PRIORITY, *_entry_point_adapters_cached()]


def adapter_by_name(name: str) -> type[SpanAdapter] | None:
    """Return the registered adapter whose ``name`` matches, or ``None``."""
    for adapter in adapters_in_priority_order():
        if adapter.name == name:
            return adapter
    return None


def select_adapter(span: dict[str, Any]) -> type[SpanAdapter] | None:
    """Pick the highest-priority adapter whose signature matches ``span``.

    A span matches when every key in the adapter's ``signature_attributes()`` is
    present in ``span["attributes"]``. Returns ``None`` when nothing matches (the
    caller falls back to the universal extractor).
    """
    attributes: dict[str, Any] = span.get("attributes") or {}
    for adapter in adapters_in_priority_order():
        sig = adapter.signature_attributes()
        if sig and all(key in attributes for key in sig):
            return adapter
    return None
