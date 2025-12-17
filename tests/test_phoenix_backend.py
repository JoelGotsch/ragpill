"""Unit tests for ``PhoenixBackend`` — SDK calls mocked.

Phoenix is an optional extra and isn't installed in the default test env, so the
lazy ``phoenix`` / ``openinference`` imports are satisfied by fake modules
injected into ``sys.modules``. These tests verify the adapter's glue (the calls
it makes and the dataframe->neutral-trace conversion), not the live server,
which the env-gated integration test covers.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import ragpill.backends.phoenix_backend as phoenix_backend_module
from ragpill.backends import Backend
from ragpill.backends._types import Assessment
from ragpill.backends.phoenix_backend import PhoenixBackend
from ragpill.trace import SpanKind  # ingest-side kind that span.kind carries


@pytest.fixture
def fake_phoenix() -> Iterator[dict[str, MagicMock]]:
    """Inject fake phoenix / openinference modules so lazy imports resolve."""
    register = MagicMock(name="register")
    client_cls = MagicMock(name="Client")
    span_processor_cls = MagicMock(name="OpenInferenceSpanProcessor")

    mods: dict[str, ModuleType] = {}

    def _mod(name: str, **attrs: object) -> ModuleType:
        m = ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        return m

    phoenix = _mod("phoenix")
    _mod("phoenix.otel", register=register)
    phoenix.otel = mods["phoenix.otel"]  # type: ignore[attr-defined]
    _mod("phoenix.client", Client=client_cls)
    phoenix.client = mods["phoenix.client"]  # type: ignore[attr-defined]
    oi = _mod("openinference")
    _mod("openinference.instrumentation")
    oi.instrumentation = mods["openinference.instrumentation"]  # type: ignore[attr-defined]
    _mod("openinference.instrumentation.pydantic_ai", OpenInferenceSpanProcessor=span_processor_cls)
    mods["openinference.instrumentation"].pydantic_ai = mods["openinference.instrumentation.pydantic_ai"]  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    # The SpanQuery capability probe is negative-cached at module level; reset
    # it around each test so what this test's sys.modules offer decides the
    # outcome, not a previous test's probe result.
    phoenix_backend_module._span_query_cls = phoenix_backend_module._UNPROBED  # pyright: ignore[reportPrivateUsage]
    try:
        yield {"register": register, "Client": client_cls, "SpanProcessor": span_processor_cls}
    finally:
        phoenix_backend_module._span_query_cls = phoenix_backend_module._UNPROBED  # pyright: ignore[reportPrivateUsage]
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _FakeSpanQuery:
    """Stand-in for ``phoenix.client.types.spans.SpanQuery`` recording its
    where-clause so tests can assert the server-side filter condition."""

    def __init__(self) -> None:
        self.where_condition: str | None = None

    def where(self, condition: str) -> _FakeSpanQuery:
        self.where_condition = condition
        return self


@pytest.fixture
def fake_span_query(fake_phoenix) -> Iterator[type[_FakeSpanQuery]]:
    """Inject ``phoenix.client.types.spans.SpanQuery`` (new-client capability)."""
    _ = fake_phoenix  # base phoenix modules must be present first
    spans_mod = ModuleType("phoenix.client.types.spans")
    spans_mod.SpanQuery = _FakeSpanQuery  # type: ignore[attr-defined]
    types_mod = ModuleType("phoenix.client.types")
    types_mod.spans = spans_mod  # type: ignore[attr-defined]
    names = ("phoenix.client.types", "phoenix.client.types.spans")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["phoenix.client.types"] = types_mod
    sys.modules["phoenix.client.types.spans"] = spans_mod
    try:
        yield _FakeSpanQuery
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _KeywordOnlySpansAPI:
    """``client.spans`` stub with the REAL keyword-only signature of
    ``get_spans_dataframe`` — a MagicMock happily accepts positional args and
    cannot catch the positional-``query`` regression (round-3 R1)."""

    def __init__(self, df: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self._df = df if df is not None else pd.DataFrame()
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def get_spans_dataframe(
        self,
        *,
        query: Any = None,
        start_time: Any = None,
        end_time: Any = None,
        limit: int = 1000,
        root_spans_only: Any = None,
        project_identifier: Any = None,
        project_name: Any = None,
        timeout: Any = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
                "root_spans_only": root_spans_only,
                "project_identifier": project_identifier,
                "project_name": project_name,
                "timeout": timeout,
            }
        )
        if self._error is not None:
            raise self._error
        return self._df


def _single_trace_df(trace_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "context.trace_id": trace_id,
                "parent_id": None,
                "name": "root",
                "span_kind": "CHAIN",
                "status_code": "OK",
            }
        ],
        index=["sp1"],
    )


def test_satisfies_backend_protocol():
    assert isinstance(PhoenixBackend(), Backend)


def test_require_phoenix_raises_clear_error_without_extra():
    # No fake_phoenix fixture here: the real import fails -> actionable error.
    with pytest.raises(RuntimeError, match=r"ragpill\[phoenix\]"):
        PhoenixBackend().set_destination("http://localhost:6006", "proj")


def test_set_destination_registers(fake_phoenix):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "my-proj")
    fake_phoenix["register"].assert_called_once()
    _, kwargs = fake_phoenix["register"].call_args
    assert kwargs["endpoint"] == "http://localhost:6006"
    assert kwargs["project_name"] == "my-proj"
    assert backend.get_tracking_uri() == "http://localhost:6006"
    assert backend.resolve_experiment_id("my-proj") == "my-proj"


def test_start_span_handle_exposes_ids_and_io(fake_phoenix):
    from ragpill.backends import CaptureSpanKind as WriteSpanKind

    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    span = MagicMock()
    span.get_span_context.return_value.span_id = 0xABC
    span.get_span_context.return_value.trace_id = 0x123
    backend._tracer.start_as_current_span.return_value.__enter__.return_value = span  # pyright: ignore[reportAttributeAccessIssue]

    with backend.start_span("run-0", WriteSpanKind.TASK) as handle:
        # execution reads these off the handle (run_span.span_id / .trace_id).
        assert handle.span_id == format(0xABC, "016x")
        assert handle.trace_id == format(0x123, "032x")
        handle.set_inputs("hi")
        handle.set_outputs("bye")
    span.set_attribute.assert_any_call("input.value", "hi")
    span.set_attribute.assert_any_call("output.value", "bye")


def test_start_run_synthesizes_handle():
    backend = PhoenixBackend()
    backend._project_name = "proj"  # pyright: ignore[reportPrivateUsage]
    handle = backend.start_run()
    assert handle.run_id == "proj"
    assert handle.experiment_id == "proj"
    assert backend.is_run_active() is True
    backend.end_run()
    assert backend.is_run_active() is False


def test_unsupported_methods_noop_and_warn():
    backend = PhoenixBackend()
    with pytest.warns(UserWarning, match="no native Phoenix equivalent"):
        backend.log_metric("acc", 1.0)
    # Same capability warns only once; other no-ops don't raise.
    backend.log_metric("acc", 0.5)
    backend.log_params({"k": "v"})
    backend.log_table(pd.DataFrame(), "f.json")
    backend.log_artifact("/tmp/x")
    backend.delete_traces("exp", ["t1"])


def test_log_assessment_annotates_root_span(fake_phoenix, monkeypatch):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    monkeypatch.setattr(backend, "_root_span_id", lambda _tid: "root-span")
    client = fake_phoenix["Client"].return_value
    backend.log_assessment(
        "trace-1",
        Assessment(name="LLMJudge_quality", value=True, source_type="LLM_JUDGE", source_id="judge", rationale="ok"),
    )
    client.spans.add_span_annotation.assert_called_once()
    _, kwargs = client.spans.add_span_annotation.call_args
    assert kwargs["span_id"] == "root-span"
    assert kwargs["annotation_name"] == "LLMJudge_quality"
    assert kwargs["annotator_kind"] == "LLM"
    assert kwargs["score"] == 1.0  # bool True -> 1.0
    assert kwargs["explanation"] == "ok"


def test_get_trace_converts_spans_dataframe(fake_phoenix):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    df = pd.DataFrame(
        [
            {
                "context.trace_id": "t1",
                "parent_id": None,
                "name": "retrieve",
                "span_kind": "RETRIEVER",
                "status_code": "OK",
                "attributes.retrieval.documents.0.document.content": "doc one",
            },
            {  # different trace, must be excluded
                "context.trace_id": "other",
                "parent_id": None,
                "name": "x",
                "span_kind": "LLM",
                "status_code": "OK",
                "attributes.retrieval.documents.0.document.content": "nope",
            },
        ],
        index=["sp1", "sp2"],
    )
    fake_phoenix["Client"].return_value.spans.get_spans_dataframe.return_value = df

    trace = backend.get_trace("t1")
    assert trace is not None
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.span_id == "sp1"
    assert span.kind is SpanKind.RETRIEVER
    assert span.dialect == "openinference"
    assert len(span.documents) == 1
    assert span.documents[0].content == "doc one"


def test_get_trace_keeps_nat_end_time_as_none(fake_phoenix):
    """An in-flight span (end_time=NaT) must surface ``end_time_ns=None`` from
    ``get_trace`` — not a coerced 0 (the R3 regression)."""
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    df = pd.DataFrame(
        [
            {
                "context.trace_id": "t1",
                "parent_id": None,
                "name": "inflight",
                "span_kind": "LLM",
                "status_code": "UNSET",
                "start_time": start,
                "end_time": pd.NaT,
            }
        ],
        index=["sp1"],
    )
    fake_phoenix["Client"].return_value.spans.get_spans_dataframe.return_value = df

    trace = backend.get_trace("t1")
    assert trace is not None
    (span,) = trace.spans
    assert span.start_time_ns == start.value
    assert span.end_time_ns is None


def test_get_trace_returns_none_for_missing_trace(fake_phoenix):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    fake_phoenix["Client"].return_value.spans.get_spans_dataframe.return_value = pd.DataFrame()
    assert backend.get_trace("nope") is None


# ---------------------------------------------------------------------------
# Server-side trace filtering (round-2 F14 / round-3 R1). The client's
# get_spans_dataframe is keyword-only, so these tests use a stub with the real
# signature — passing the query positionally must fail loudly, not silently
# degrade to the unfiltered full-project fetch.
# ---------------------------------------------------------------------------


def test_get_trace_passes_span_query_by_keyword_with_trace_filter(fake_phoenix, fake_span_query):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    spans_api = _KeywordOnlySpansAPI(df=_single_trace_df("t1"))
    backend._client_cache = SimpleNamespace(spans=spans_api)  # pyright: ignore[reportPrivateUsage]

    trace = backend.get_trace("t1")

    assert trace is not None and len(trace.spans) == 1
    (call,) = spans_api.calls
    assert call["query"] is not None, "server-side filter silently skipped"
    assert isinstance(call["query"], fake_span_query)
    assert call["query"].where_condition is not None
    assert "t1" in call["query"].where_condition
    assert call["project_identifier"] == "proj"
    # Explicit limit: the client defaults to 1000, which silently truncates.
    assert call["limit"] == phoenix_backend_module._SPANS_FETCH_LIMIT  # pyright: ignore[reportPrivateUsage]


def test_span_query_capability_probed_once(fake_phoenix, fake_span_query):
    _ = fake_span_query
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    spans_api = _KeywordOnlySpansAPI(df=_single_trace_df("t1"))
    backend._client_cache = SimpleNamespace(spans=spans_api)  # pyright: ignore[reportPrivateUsage]

    assert backend.get_trace("t1") is not None
    # Remove the module: a per-call re-import would now lose the capability.
    del sys.modules["phoenix.client.types.spans"]
    assert backend.get_trace("t1") is not None
    assert len(spans_api.calls) == 2
    assert all(c["query"] is not None for c in spans_api.calls)


def test_old_client_without_span_query_falls_back_unfiltered(fake_phoenix):
    # No fake_span_query fixture: the SpanQuery import raises ImportError,
    # which is the ONLY condition that selects the full-project fallback.
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    spans_api = _KeywordOnlySpansAPI(df=_single_trace_df("t1"))
    backend._client_cache = SimpleNamespace(spans=spans_api)  # pyright: ignore[reportPrivateUsage]

    trace = backend.get_trace("t1")

    assert trace is not None and len(trace.spans) == 1
    (call,) = spans_api.calls
    assert call["query"] is None
    assert call["project_identifier"] == "proj"
    assert call["limit"] == phoenix_backend_module._SPANS_FETCH_LIMIT  # pyright: ignore[reportPrivateUsage]


def test_sdk_typeerror_in_query_path_propagates(fake_phoenix, fake_span_query):
    # A genuine TypeError from the SDK must surface — re-classifying it as
    # "old client" is exactly what hid the positional-query no-op (R1).
    _ = fake_span_query
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    spans_api = _KeywordOnlySpansAPI(error=TypeError("unexpected keyword argument"))
    backend._client_cache = SimpleNamespace(spans=spans_api)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        backend.get_trace("t1")
    # It went through the filtered path, not the unfiltered fallback.
    assert len(spans_api.calls) == 1
    assert spans_api.calls[0]["query"] is not None


# ---------------------------------------------------------------------------
# Integration — requires the phoenix extra installed AND a live Phoenix server.
# Gated by RUN_PHOENIX_INTEGRATION_TESTS=1 (skipped by default / in CI without it).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("RUN_PHOENIX_INTEGRATION_TESTS"),
    reason="set RUN_PHOENIX_INTEGRATION_TESTS=1 and run a Phoenix server to exercise the live path",
)
@pytest.mark.anyio
async def test_phoenix_end_to_end_capture_and_fetch():
    """End-to-end: configure the Phoenix backend, run a dataset with trace
    capture, and assert each task run gets a neutral trace back."""
    import os

    from ragpill.backends import configure_backend, reset_backend
    from ragpill.base import TestCaseMetadata
    from ragpill.eval_types import Case, Dataset
    from ragpill.execution import execute_dataset

    async def echo(q: str) -> str:
        return f"echo:{q}"

    reset_backend()
    configure_backend(PhoenixBackend)
    try:
        ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="hello", metadata=TestCaseMetadata())])
        out = await execute_dataset(
            ds,
            task=echo,
            capture_traces=True,
            tracking_uri=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
        )
        tr = out.cases[0].task_runs[0]
        assert tr.output == "echo:hello"
        assert tr.trace is not None and tr.trace.spans
    finally:
        reset_backend()


def test_await_trace_waits_for_span_set_to_stabilize():
    """Spans arrive in independent OTLP batches: await_trace must not return
    the first non-empty snapshot, only one that is stable across two polls."""
    from unittest.mock import patch

    from ragpill.trace import Span, SpanKind, Trace

    def _trace(n_spans: int) -> Trace:
        spans = [
            Span(
                span_id=f"s{i}",
                parent_id=None,
                trace_id="t",
                name=f"s{i}",
                kind=SpanKind.CHAIN,
                start_time_ns=0,
                end_time_ns=0,
            )
            for i in range(n_spans)
        ]
        return Trace(trace_id="t", spans=spans)

    backend = PhoenixBackend()
    partial, full = _trace(1), _trace(3)
    with patch.object(PhoenixBackend, "get_trace", side_effect=[partial, full, full]) as mock_get:
        got, stable = backend.await_trace("t", timeout_s=5.0, poll_interval_s=0.01)
    assert got is not None and len(got.spans) == 3
    assert stable is True
    assert mock_get.call_count == 3
