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
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd
import pytest

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
    try:
        yield {"register": register, "Client": client_cls, "SpanProcessor": span_processor_cls}
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


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


def test_get_trace_returns_none_for_missing_trace(fake_phoenix):
    backend = PhoenixBackend()
    backend.set_destination("http://localhost:6006", "proj")
    fake_phoenix["Client"].return_value.spans.get_spans_dataframe.return_value = pd.DataFrame()
    assert backend.get_trace("nope") is None


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
