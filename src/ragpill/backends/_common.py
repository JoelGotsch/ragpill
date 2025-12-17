"""Shared scaffolding for backend adapters.

Behaviour every remote-service adapter (Langfuse, Phoenix, …) needs but that
has no backend-specific content: one-time "unsupported capability" warnings,
no-op ResultsBackend methods, synthetic run bookkeeping for backends without a
native run concept, and the trace-export polling loop. Keeping these here means
a new adapter only writes the code that actually differs per backend.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ragpill.backends._types import RunHandle

if TYPE_CHECKING:
    import pandas as pd

    from ragpill.trace import Trace as NeutralTrace

logger = logging.getLogger("ragpill.backends")


def require_extra(import_name: str, install_hint: str) -> None:
    """Import-probe an optional SDK, raising ``install_hint`` if it's missing.

    Shared by the remote adapters so each ``_require_*`` isn't a copy of the
    same try/except.
    """
    import importlib

    try:
        importlib.import_module(import_name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(install_hint) from exc


def to_text(value: object) -> str:
    """Best-effort string form of a value (str as-is; else JSON with ``str``
    fallback). Shared span-I/O stringify for the remote adapters."""
    import json

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def to_unix_nano(val: object) -> int | None:
    """Coerce a timestamp to Unix nanoseconds, or ``None`` when absent/unknown.

    Handles the shapes the remote adapters see: a pandas ``Timestamp`` (has an
    integer ``.value`` in nanos), a ``datetime`` (has ``.timestamp()`` in
    seconds), and missing values (``None`` / ``NaN`` / ``NaT``). Returns ``None``
    (not ``0``) for missing values — a not-yet-ingested Phoenix row has
    ``end_time = NaT``, whose ``.value`` is ``int64`` *min*; treating that as a
    timestamp would yield an absurd negative duration. ``None`` signals "unknown"
    so ordering and duration rendering degrade gracefully.
    """
    if val is None:
        return None
    # NaT / NaN: pandas scalar missing-value. Guard before reading ``.value``
    # (pd.NaT.value is int64-min, which must NOT be treated as a real timestamp).
    try:
        import pandas as pd

        if bool(pd.isna(val)):  # pyright: ignore[reportUnknownMemberType]
            return None
    except (TypeError, ValueError):
        pass  # not a pandas-recognized scalar; fall through
    # pandas Timestamp: .value is integer nanoseconds since the epoch.
    value = getattr(val, "value", None)
    if isinstance(value, int):
        return value
    # datetime / anything exposing .timestamp() in seconds.
    ts = getattr(val, "timestamp", None)
    if callable(ts):
        try:
            seconds: Any = ts()
            return int(seconds * 1_000_000_000)
        except Exception:
            return None
    return None


def is_http_not_found(exc: BaseException) -> bool:
    """Best-effort 404/not-found detection across httpx-based SDK clients.

    Remote backends (Langfuse, Phoenix) wrap httpx; a genuinely missing trace
    surfaces as a 404 (or a ``NotFoundError``-named exception), which is a
    legitimate "poll again" miss. Any other error (auth, connection, 5xx) is a
    real failure that read paths should surface rather than mistake for an
    in-flight trace.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    return type(exc).__name__ in {"NotFoundError", "NotFound"}


def poll_for_trace(
    fetch: Callable[[], NeutralTrace | None],
    *,
    timeout_s: float,
    poll_interval_s: float,
    stable_span_set: bool,
) -> tuple[NeutralTrace | None, bool]:
    """Poll ``fetch`` until the trace is exported, up to ``timeout_s``.

    Args:
        fetch: Zero-arg callable returning the neutral trace or ``None``.
        timeout_s: Total time to poll before giving up.
        poll_interval_s: Sleep between polls.
        stable_span_set: Readiness criterion. ``False`` returns on the first
            non-``None`` fetch — for backends whose by-id lookup is atomic
            (MLflow returns a trace only once its full span tree is stored).
            ``True`` returns only once the span set is unchanged across two
            consecutive polls — for backends that ingest spans in independent
            export batches (Langfuse, Phoenix), where a readable trace can
            still be missing in-flight spans.

    Returns:
        ``(trace, stable)``. ``stable`` is ``True`` only when the readiness
        criterion was met before the deadline — the trace is complete. When the
        deadline is hit first, returns whatever the final fetch produced
        (possibly partial or ``None``) with ``stable=False`` so the caller can
        record the run as ``incomplete``/``unavailable`` rather than trusting a
        half-exported trace. Never a different trace.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    previous_span_ids: set[str] | None = None
    while True:
        trace = fetch()
        if trace is not None:
            if not stable_span_set:
                return trace, True
            if trace.spans:
                span_ids = {s.span_id for s in trace.spans}
                if span_ids == previous_span_ids:
                    return trace, True
                previous_span_ids = span_ids
        if time.monotonic() >= deadline:
            return trace, False
        time.sleep(poll_interval_s)


class UnsupportedCapabilityWarner:
    """Mixin: warn once per capability a backend has no native equivalent for."""

    _warned: set[str]

    def _warn_unsupported(self, capability: str) -> None:
        warned = getattr(self, "_warned", None)
        if warned is None:
            warned = self._warned = set()
        if capability not in warned:
            warned.add(capability)
            display = type(self).__name__.removesuffix("Backend")
            warnings.warn(
                f"{type(self).__name__}: '{capability}' has no native {display} equivalent and is a no-op. "
                "See plans/multi-backend-tracking.md.",
                stacklevel=3,
            )


class NoopResultsMixin(UnsupportedCapabilityWarner):
    """``ResultsBackend`` metric/param/table/artifact methods for backends with
    no native run-artifact concept — each is a warn-once no-op."""

    def log_metric(self, name: str, value: float) -> None:
        _ = name, value
        self._warn_unsupported("log_metric")

    def log_params(self, params: Mapping[str, str]) -> None:
        _ = params
        self._warn_unsupported("log_params")

    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None:
        _ = df, artifact_file
        self._warn_unsupported("log_table")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        _ = local_path, artifact_path
        self._warn_unsupported("log_artifact")

    def set_run_tag(self, run_id: str, key: str, value: str) -> None:
        _ = run_id, key, value
        self._warn_unsupported("set_run_tag")

    def get_run_tag(self, run_id: str, key: str) -> str | None:
        # No native run concept -> no stored tags. Returning None lets the
        # upload idempotency guard simply proceed for these backends.
        _ = run_id, key
        return None

    def delete_run_artifact(self, run_id: str, artifact_path: str) -> None:
        _ = run_id, artifact_path
        self._warn_unsupported("delete_run_artifact")


class SyntheticRunMixin:
    """``start_run``/``end_run``/``is_run_active`` for backends with no native
    run concept: the project/experiment name doubles as the run id, and
    ``end_run`` flushes pending exports via the ``_flush`` hook."""

    _run_active: bool = False
    _project_name: str = "ragpill"

    def start_run(self, run_id: str | None = None, description: str | None = None) -> RunHandle:
        _ = description
        self._run_active = True
        rid = run_id or self._project_name
        return RunHandle(run_id=rid, experiment_id=self._project_name)

    def end_run(self) -> None:
        self._run_active = False
        try:
            self._flush()
        except Exception:
            # A failed flush at run end means spans were dropped — surface it
            # instead of silently losing data on the async-export backends.
            logger.warning("%s: flush on end_run failed; some spans may be lost.", type(self).__name__, exc_info=True)

    def is_run_active(self) -> bool:
        return self._run_active

    def _flush(self) -> None:
        """Flush pending exports on run end. Override per backend."""


class RemoteQueryMixin:
    """Shared ``await_trace`` for backends whose reads poll for export.

    Subclasses implement :meth:`get_trace`; ``_stable_span_set`` selects the
    readiness criterion — ``True`` for batch-ingest backends (Langfuse/Phoenix),
    ``False`` for atomic-read backends (MLflow's by-id lookup returns the full
    span tree at once). Returns ``(trace, stable)`` per :func:`poll_for_trace`.
    """

    _stable_span_set: bool = True

    def get_trace(self, trace_id: str) -> NeutralTrace | None:  # pragma: no cover - provided by concrete backend
        del trace_id
        raise NotImplementedError

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> tuple[NeutralTrace | None, bool]:
        # run_id / experiment_id are part of the protocol for backends whose
        # readiness query needs them; the poll-by-id backends do not.
        del run_id, experiment_id
        return poll_for_trace(
            lambda: self.get_trace(trace_id),
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            stable_span_set=self._stable_span_set,
        )
