"""Backend registry.

Single source of truth for "which adapter does this process talk to?".
Defaults to ``MLflowBackend`` when MLflow is importable; raises a clear
"install ``ragpill[mlflow]`` or call ``configure_backend(...)``" error
otherwise. Callers swap implementations via :func:`configure_backend`.

Step 1 ships the registry but no production code calls
:func:`get_backend` yet; later steps switch upload/execution/evaluators
over to it.
"""

from __future__ import annotations

from collections.abc import Callable

from ragpill.backends._base import Backend

_factory: Callable[[], Backend] | None = None
_instance: Backend | None = None


def configure_backend(factory: Callable[[], Backend]) -> None:
    """Install a backend factory.

    The factory is called the first time :func:`get_backend` runs after a
    reset (typically per-process). Pass the class itself for adapters with
    no constructor args (``configure_backend(MLflowBackend)``).
    """
    global _factory, _instance
    _factory = factory
    _instance = None


def reset_backend() -> None:
    """Forget the configured factory and cached instance.

    Mostly for tests; production code calls this once at startup via
    :func:`configure_backend` and never again.
    """
    global _factory, _instance
    _factory = None
    _instance = None


def get_backend() -> Backend:
    """Return the active backend, building it from the configured factory.

    Falls back to :class:`MLflowBackend` when no factory is registered.
    Raises ``RuntimeError`` with an actionable message when MLflow isn't
    installed and no alternative has been configured.
    """
    global _instance
    if _instance is not None:
        return _instance
    factory = _factory or _default_factory
    _instance = factory()
    return _instance


def _default_factory() -> Backend:
    try:
        from ragpill.backends.mlflow_backend import MLflowBackend
    except ImportError as exc:  # pragma: no cover - only hit when extra is missing
        raise RuntimeError(
            "No tracking backend is configured and the MLflow adapter is not importable. "
            "Either install the MLflow extra (`pip install ragpill[mlflow]`) or call "
            "`ragpill.backends.configure_backend(...)` with an alternative adapter."
        ) from exc
    return MLflowBackend()
