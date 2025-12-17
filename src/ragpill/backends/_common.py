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
from typing import TYPE_CHECKING

from ragpill.backends._types import RunHandle

if TYPE_CHECKING:
    import pandas as pd

    from ragpill.trace import Trace as NeutralTrace

logger = logging.getLogger("ragpill.backends")


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
) -> NeutralTrace | None:
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
        The trace once ready, or whatever the final fetch produced at the
        deadline (possibly partial or ``None``). Never a different trace.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    previous_span_ids: set[str] | None = None
    while True:
        trace = fetch()
        if trace is not None:
            if not stable_span_set:
                return trace
            if trace.spans:
                span_ids = {s.span_id for s in trace.spans}
                if span_ids == previous_span_ids:
                    return trace
                previous_span_ids = span_ids
        if time.monotonic() >= deadline:
            return trace
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
