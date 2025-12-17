"""Unit tests for ``LangfuseBackend`` — SDK calls mocked.

Langfuse is an optional extra not installed in the default test env, so the lazy
``langfuse`` import is satisfied by a fake module. These verify the adapter glue
(the calls it makes, score mapping, observation->span conversion), not the live
server, which the env-gated integration test covers.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ragpill.backends import Backend, SpanKind as WriteSpanKind
from ragpill.backends._types import Assessment
from ragpill.backends.langfuse_backend import LangfuseBackend
from ragpill.trace import SpanKind  # ingest-side kind span.kind carries


@pytest.fixture
def fake_langfuse() -> Iterator[MagicMock]:
    """Inject a fake ``langfuse`` module so the lazy import resolves."""
    langfuse_cls = MagicMock(name="Langfuse")
    mod = ModuleType("langfuse")
    mod.Langfuse = langfuse_cls  # type: ignore[attr-defined]
    saved = sys.modules.get("langfuse")
    sys.modules["langfuse"] = mod
    try:
        yield langfuse_cls
    finally:
        if saved is None:
            sys.modules.pop("langfuse", None)
        else:
            sys.modules["langfuse"] = saved


def test_satisfies_backend_protocol():
    assert isinstance(LangfuseBackend(), Backend)


def test_require_langfuse_raises_clear_error_without_extra():
    with pytest.raises(RuntimeError, match=r"ragpill\[langfuse\]"):
        LangfuseBackend().set_destination("https://cloud.langfuse.com", "proj")


def test_set_destination_builds_client_with_host(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf.example.com", "proj")
    fake_langfuse.assert_called_once()
    _, kwargs = fake_langfuse.call_args
    assert kwargs["host"] == "https://lf.example.com"
    assert backend.get_tracking_uri() == "https://lf.example.com"
    assert backend.resolve_experiment_id("proj") == "proj"


def test_start_run_and_lifecycle(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    handle = backend.start_run()
    assert handle.run_id == "proj" and handle.experiment_id == "proj"
    assert backend.is_run_active() is True
    backend.end_run()
    assert backend.is_run_active() is False
    fake_langfuse.return_value.flush.assert_called_once()


def test_start_span_handle_exposes_ids_and_io(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    span = MagicMock()
    span.id = "obs-1"
    span.trace_id = "trace-1"
    client = fake_langfuse.return_value
    client.start_as_current_observation.return_value.__enter__.return_value = span

    with backend.start_span("run-0", WriteSpanKind.TASK) as handle:
        assert handle.span_id == "obs-1"
        assert handle.trace_id == "trace-1"
        handle.set_inputs("hi")
        handle.set_outputs("bye")
    # as_type for TASK is "chain"
    _, kwargs = client.start_as_current_observation.call_args
    assert kwargs["as_type"] == "chain"
    span.update.assert_any_call(input="hi")
    span.update.assert_any_call(output="bye")


def test_unsupported_methods_noop_and_warn(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    with pytest.warns(UserWarning, match="no native Langfuse equivalent"):
        backend.log_metric("acc", 1.0)
    backend.log_params({"k": "v"})
    backend.log_table(pd.DataFrame(), "f.json")
    backend.log_artifact("/tmp/x")


@pytest.mark.parametrize(
    ("value", "data_type", "score"),
    [(True, "BOOLEAN", 1), (0.8, "NUMERIC", 0.8), ("good", "CATEGORICAL", "good")],
)
def test_log_assessment_maps_score_type(fake_langfuse, value, data_type, score):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    backend.log_assessment(
        "trace-1",
        Assessment(name="LLMJudge", value=value, source_type="LLM_JUDGE", source_id="j", rationale="why"),
    )
    client = fake_langfuse.return_value
    client.create_score.assert_called_once()
    _, kwargs = client.create_score.call_args
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["data_type"] == data_type
    assert kwargs["value"] == score
    assert kwargs["comment"] == "why"


def test_delete_traces_calls_api_per_id(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    backend.delete_traces("proj", ["t1", "t2"])
    client = fake_langfuse.return_value
    assert client.api.trace.delete.call_count == 2
    client.api.trace.delete.assert_any_call("t1")


def test_get_trace_converts_observations(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    native = SimpleNamespace(
        id="trace-1",
        observations=[
            SimpleNamespace(
                id="obs-root",
                parent_observation_id=None,
                name="root",
                type="SPAN",
                input={"q": "hi"},
                output="ans",
                model=None,
                usage_details={},
                metadata={"k": "v"},
            ),
            SimpleNamespace(
                id="obs-gen",
                parent_observation_id="obs-root",
                name="generate",
                type="GENERATION",
                input="prompt",
                output="out",
                model="gpt-4o",
                usage_details={"input": 10, "output": 5, "total": 15},
                metadata=None,
            ),
        ],
    )
    fake_langfuse.return_value.api.trace.get.return_value = native

    trace = backend.get_trace("trace-1")
    assert trace is not None and trace.dialect == "langfuse"
    by_id = {s.span_id: s for s in trace.spans}
    # A None parent must stay None (not the string "None"), or root detection breaks.
    assert by_id["obs-root"].parent_id is None
    assert by_id["obs-root"].kind is SpanKind.CHAIN
    assert by_id["obs-root"].inputs == {"q": "hi"}
    assert by_id["obs-root"].attributes == {"k": "v"}
    gen = by_id["obs-gen"]
    assert gen.kind is SpanKind.LLM
    assert gen.parent_id == "obs-root"
    assert gen.model == "gpt-4o"
    assert (gen.usage.input_tokens, gen.usage.output_tokens, gen.usage.total_tokens) == (10, 5, 15)


def test_get_trace_returns_none_when_empty(fake_langfuse):
    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    fake_langfuse.return_value.api.trace.get.return_value = SimpleNamespace(id="t", observations=[])
    assert backend.get_trace("t") is None


# ---------------------------------------------------------------------------
# Integration — requires the langfuse extra + a live Langfuse instance.
# Gated by RUN_LANGFUSE_INTEGRATION_TESTS=1.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("RUN_LANGFUSE_INTEGRATION_TESTS"),
    reason="set RUN_LANGFUSE_INTEGRATION_TESTS=1 + Langfuse creds to exercise the live path",
)
@pytest.mark.anyio
async def test_langfuse_end_to_end_capture_and_fetch():
    import os

    from ragpill.backends import configure_backend, reset_backend
    from ragpill.base import TestCaseMetadata
    from ragpill.eval_types import Case, Dataset
    from ragpill.execution import execute_dataset

    async def echo(q: str) -> str:
        return f"echo:{q}"

    reset_backend()
    configure_backend(LangfuseBackend)
    try:
        ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="hello", metadata=TestCaseMetadata())])
        out = await execute_dataset(
            ds,
            task=echo,
            capture_traces=True,
            mlflow_tracking_uri=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        tr = out.cases[0].task_runs[0]
        assert tr.output == "echo:hello"
        assert tr.trace is not None and tr.trace.spans
    finally:
        reset_backend()


def test_await_trace_waits_for_span_set_to_stabilize(fake_langfuse):
    """Observations arrive in independent batches: await_trace must not return
    the first non-empty snapshot, only one that is stable across two polls."""
    from unittest.mock import patch

    from ragpill.trace import Span, Trace

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

    backend = LangfuseBackend()
    backend.set_destination("https://lf", "proj")
    partial, full = _trace(1), _trace(3)
    with patch.object(LangfuseBackend, "get_trace", side_effect=[partial, full, full]) as mock_get:
        got = backend.await_trace("t", timeout_s=5.0, poll_interval_s=0.01)
    assert got is not None and len(got.spans) == 3
    assert mock_get.call_count == 3
