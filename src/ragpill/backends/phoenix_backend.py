"""Arize Phoenix adapter for the tracking backend protocols.

Phoenix is OpenInference-native: capture is OTel + the OpenInference
instrumentors, and traces are read back as a spans DataFrame that we convert to
the neutral model via the OpenInference dialect adapter (ADR-0013/0017).

All ``phoenix`` / ``opentelemetry`` imports are **lazy** (inside methods) so this
module imports cleanly without the ``ragpill[phoenix]`` extra; only actually
using a backend method requires it. ``phoenix`` and ``mlflow`` are not
co-installable in one environment — see ``plans/phoenix-backend-findings.md``.

Scope note: the live path (real Phoenix server) is exercised by the env-gated
integration test, not the unit suite. Methods Phoenix has no native concept for
(metrics, params, tables, artifacts, trace deletion) no-op with a one-time
warning, per ``plans/multi-backend-tracking.md``'s risk table.
"""

from __future__ import annotations

# The phoenix / openinference SDKs are an optional extra and are not installed in
# the default (mlflow) type-check environment, so their imports and return types
# are unresolved here. Relax the unknown-type reports for this adapter; the
# phoenix CI env (where the extra is installed) type-checks against the real API.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedImport=false
import warnings
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any

from ragpill.backends._types import Assessment, CaseGroupingHandle, RunHandle, SpanKind

if TYPE_CHECKING:
    import pandas as pd

    from ragpill.trace import Trace as NeutralTrace

_INSTALL_HINT = (
    "The Phoenix backend requires the 'phoenix' extra. Install it with "
    "`pip install ragpill[phoenix]` (note: not co-installable with ragpill[mlflow] "
    "in the same environment — see plans/phoenix-backend-findings.md)."
)

# ragpill write-side SpanKind -> OpenInference span.kind string.
_SPAN_KIND_TO_OI: dict[SpanKind, str] = {
    SpanKind.AGENT: "AGENT",
    SpanKind.CHAT_MODEL: "LLM",
    SpanKind.LLM: "LLM",
    SpanKind.RERANKER: "RERANKER",
    SpanKind.RETRIEVER: "RETRIEVER",
    SpanKind.TASK: "CHAIN",
    SpanKind.TOOL: "TOOL",
    SpanKind.UNKNOWN: "UNKNOWN",
}


def _require_phoenix() -> None:
    try:
        import phoenix.otel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_HINT) from exc


class _SpanHandle:
    """Wraps an OTel span to expose the set_attribute/set_inputs/set_outputs
    surface ragpill's capture code uses, mapping I/O onto OpenInference keys."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_inputs(self, value: Any) -> None:
        self._span.set_attribute("input.value", _as_text(value))

    def set_outputs(self, value: Any) -> None:
        self._span.set_attribute("output.value", _as_text(value))


def _as_text(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


class PhoenixBackend:
    """Adapter implementing the tracking backend protocols against Arize Phoenix."""

    def __init__(self) -> None:
        self._project_name = "ragpill"
        self._endpoint: str | None = None
        self._tracer_provider: Any = None
        self._tracer: Any = None
        self._run_active = False
        self._warned: set[str] = set()

    def _warn_unsupported(self, capability: str) -> None:
        if capability not in self._warned:
            self._warned.add(capability)
            warnings.warn(
                f"PhoenixBackend: '{capability}' has no native Phoenix equivalent and is a no-op. "
                "See plans/multi-backend-tracking.md.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # TraceCaptureBackend
    # ------------------------------------------------------------------

    def set_destination(self, uri: str | None, experiment_name: str) -> None:
        _require_phoenix()
        from phoenix.otel import register

        self._endpoint = uri
        self._project_name = experiment_name
        # set_global_tracer_provider=False so multiple configure cycles don't
        # clobber a process-global; we keep our own handle.
        self._tracer_provider = register(
            endpoint=uri,
            project_name=experiment_name,
            set_global_tracer_provider=False,
            verbose=False,
        )
        self._tracer = self._tracer_provider.get_tracer("ragpill")

    def start_run(self, run_id: str | None = None, description: str | None = None) -> RunHandle:
        # Phoenix has no "run" concept; a project ≈ an experiment. Synthesize a
        # handle so the execution layer's bookkeeping works.
        _ = description
        self._run_active = True
        rid = run_id or self._project_name
        return RunHandle(run_id=rid, experiment_id=self._project_name)

    def end_run(self) -> None:
        self._run_active = False
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.force_flush()
            except Exception:
                pass

    def start_span(
        self,
        name: str,
        span_type: SpanKind,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        _require_phoenix()
        tracer = self._tracer

        @contextmanager
        def cm() -> Generator[Any, None, None]:
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("openinference.span.kind", _SPAN_KIND_TO_OI[span_type])
                for k, v in (attributes or {}).items():
                    span.set_attribute(k, v)
                yield _SpanHandle(span)

        return cm()

    def autolog_pydantic_ai(self) -> None:
        _require_phoenix()
        # Phoenix relies on pydantic-ai's native OTel emission reshaped by the
        # OpenInference span processor (there is no MLflow-style monkeypatch
        # autolog). Add the processor to our tracer provider.
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

        if self._tracer_provider is not None:
            self._tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

    def start_case_grouping(
        self,
        case_id: str,
        name: str,
        inputs: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[CaseGroupingHandle]:
        """Span mode: open a parent span; per-repeat spans nest under it and the
        execution layer filters per repeat by ``run_span_id``."""
        _require_phoenix()
        tracer = self._tracer

        @contextmanager
        def cm() -> Generator[CaseGroupingHandle, None, None]:
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("session.id", case_id)
                if inputs is not None:
                    span.set_attribute("input.value", _as_text(inputs))
                for k, v in (attributes or {}).items():
                    span.set_attribute(k, v)
                trace_id_hex = format(span.get_span_context().trace_id, "032x")
                yield CaseGroupingHandle(mode="span", case_trace_id=trace_id_hex, session_id=case_id)

        return cm()

    # ------------------------------------------------------------------
    # TraceQueryBackend
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        from phoenix.client import Client

        return Client(base_url=self._endpoint) if self._endpoint else Client()

    def search_traces(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        max_results: int = 1000,
    ) -> list[Any]:
        # Native traces are only consumed by the MLflow-specific judge-trace
        # cleanup path, which Phoenix doesn't support (no deletion). Return empty.
        _ = run_id, experiment_id, max_results
        return []

    def get_trace(self, trace_id: str) -> NeutralTrace | None:
        _require_phoenix()
        try:
            df = self._client().spans.get_spans_dataframe(project_identifier=self._project_name)
        except Exception:
            return None
        return _trace_from_spans_dataframe(df, trace_id)

    def await_trace(
        self,
        trace_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> NeutralTrace | None:
        import time

        del run_id, experiment_id
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            trace = self.get_trace(trace_id)
            if trace is not None and trace.spans:
                return trace
            if time.monotonic() >= deadline:
                return trace  # may be None or empty; caller treats both as "no trace"
            time.sleep(poll_interval_s)

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> None:
        _ = experiment_id, trace_ids
        self._warn_unsupported("delete_traces")

    # ------------------------------------------------------------------
    # ResultsBackend
    # ------------------------------------------------------------------

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

    def log_assessment(self, trace_id: str, assessment: Assessment) -> None:
        """Map an assessment onto a Phoenix span annotation on the trace's root span."""
        _require_phoenix()
        root_span_id = self._root_span_id(trace_id)
        if root_span_id is None:
            return
        annotator = "LLM" if assessment.source_type.upper().startswith("LLM") else "CODE"
        score = float(assessment.value) if isinstance(assessment.value, (bool, int, float)) else None
        label = assessment.value if isinstance(assessment.value, str) else None
        self._client().spans.add_span_annotation(
            span_id=root_span_id,
            annotation_name=assessment.name,
            annotator_kind=annotator,
            label=label,
            score=score,
            explanation=assessment.rationale,
            metadata=dict(assessment.metadata) or None,
        )

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        # Phoenix tags live as root-span attributes; surface as an annotation label.
        root_span_id = self._root_span_id(trace_id)
        if root_span_id is None:
            return
        self._client().spans.add_span_annotation(
            span_id=root_span_id, annotation_name=key, annotator_kind="CODE", label=value
        )

    def _root_span_id(self, trace_id: str) -> str | None:
        try:
            df = self._client().spans.get_spans_dataframe(project_identifier=self._project_name, root_spans_only=True)
        except Exception:
            return None
        return _root_span_id_for_trace(df, trace_id)

    # ------------------------------------------------------------------
    # LifecycleBackend
    # ------------------------------------------------------------------

    def resolve_experiment_id(self, experiment_name: str) -> str:
        # Phoenix projects are identified by name.
        return experiment_name

    def get_tracking_uri(self) -> str | None:
        return self._endpoint

    def set_tracking_uri(self, uri: str) -> None:
        self._endpoint = uri

    def is_run_active(self) -> bool:
        return self._run_active


# ---------------------------------------------------------------------------
# DataFrame -> neutral trace conversion
#
# Phoenix's get_spans_dataframe returns one row per span, indexed by span id,
# with OpenInference attributes flattened into ``attributes.<key>`` columns.
# We rebuild normalised OTLP-JSON span dicts and run them through the
# OpenInference adapter. Column names follow Phoenix's documented schema; the
# env-gated integration test is the real check against a live server.
# ---------------------------------------------------------------------------


def _row_to_span_dict(span_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for col, val in row.items():
        if col.startswith("attributes.") and _present(val):
            attributes[col[len("attributes.") :]] = val
    # span_kind is a top-level column in Phoenix; OpenInference adapter reads it
    # from the attribute bag.
    kind = row.get("span_kind")
    if _present(kind):
        attributes.setdefault("openinference.span.kind", kind)
    return {
        "trace_id": str(row.get("context.trace_id", "") or ""),
        "span_id": span_id,
        "parent_span_id": (str(row["parent_id"]) if _present(row.get("parent_id")) else None),
        "name": str(row.get("name", "") or ""),
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 0,
        "attributes": attributes,
        "events": [],
        "status": {"code": str(row.get("status_code", "UNSET") or "UNSET"), "message": None},
    }


def _present(val: Any) -> bool:
    # Avoid importing pandas just for isna; treat None/NaN/empty as absent.
    if val is None:
        return False
    try:
        return not (val != val)  # NaN != NaN
    except Exception:
        return True


def _trace_from_spans_dataframe(df: pd.DataFrame, trace_id: str) -> NeutralTrace | None:
    from ragpill.trace import parse_otel

    rows = _rows_for_trace(df, trace_id)
    if not rows:
        return None
    span_dicts = [_row_to_span_dict(sid, row) for sid, row in rows]
    return parse_otel(span_dicts, dialect="openinference")


def _rows_for_trace(df: pd.DataFrame, trace_id: str) -> list[tuple[str, Mapping[str, Any]]]:
    out: list[tuple[str, Mapping[str, Any]]] = []
    for span_id, row in df.iterrows():
        record: dict[str, Any] = dict(row)
        if str(record.get("context.trace_id", "")) == trace_id:
            out.append((str(span_id), record))
    return out


def _root_span_id_for_trace(df: pd.DataFrame, trace_id: str) -> str | None:
    for span_id, row in df.iterrows():
        record: dict[str, Any] = dict(row)
        if str(record.get("context.trace_id", "")) == trace_id:
            return str(span_id)
    return None
