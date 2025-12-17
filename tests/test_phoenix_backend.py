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
            mlflow_tracking_uri=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
        )
        tr = out.cases[0].task_runs[0]
        assert tr.output == "echo:hello"
        assert tr.trace is not None and tr.trace.spans
    finally:
        reset_backend()
