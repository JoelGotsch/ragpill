"""Phase 6: remote adapters carry real span timestamps (not a false 0ms)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ragpill.backends._common import to_unix_nano  # pyright: ignore[reportPrivateUsage]


def test_to_unix_nano_handles_datetime_timestamp_and_none():
    dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert to_unix_nano(dt) == int(dt.timestamp() * 1_000_000_000)
    # pandas Timestamp exposes integer nanos via .value.
    pd = __import__("pandas")
    ts = pd.Timestamp("2024-01-02T03:04:05Z")
    assert to_unix_nano(ts) == ts.value
    assert to_unix_nano(None) is None
    assert to_unix_nano(float("nan")) is None
    # NaT must be None, never int64-min (pd.NaT.value) or a fabricated 0.
    assert to_unix_nano(pd.NaT) is None


def test_langfuse_observation_carries_timestamps_and_status():
    from ragpill.backends.langfuse_backend import _observation_to_span  # pyright: ignore[reportPrivateUsage]

    start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 1, 0, 0, 2, tzinfo=UTC)
    obs = SimpleNamespace(
        id="o1",
        parent_observation_id=None,
        name="gen",
        type="GENERATION",
        start_time=start,
        end_time=end,
        level="ERROR",
        status_message="boom",
        input="p",
        output="o",
        model="gpt-4o",
        usage_details={},
        metadata={},
    )
    span = _observation_to_span(obs, "trace-1")
    assert span.start_time_ns == int(start.timestamp() * 1_000_000_000)
    assert span.end_time_ns == int(end.timestamp() * 1_000_000_000)
    assert span.end_time_ns > span.start_time_ns  # real duration, not 0ms
    assert span.status == "ERROR"
    assert span.status_message == "boom"


def test_phoenix_row_lifts_timestamps():
    import pandas as pd

    from ragpill.backends.phoenix_backend import _row_to_span_dict  # pyright: ignore[reportPrivateUsage]

    start = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:03Z")
    row = {
        "context.trace_id": "t1",
        "parent_id": None,
        "name": "retrieve",
        "span_kind": "RETRIEVER",
        "status_code": "OK",
        "start_time": start,
        "end_time": end,
    }
    d = _row_to_span_dict("sp1", row)
    assert d["start_time_unix_nano"] == start.value
    assert d["end_time_unix_nano"] == end.value
    assert d["end_time_unix_nano"] > d["start_time_unix_nano"]


def _nat_end_time_dataframe() -> tuple[object, object]:
    """A Phoenix-shaped spans DataFrame whose single row has ``end_time=NaT``."""
    import pandas as pd

    start = pd.Timestamp("2024-01-01T00:00:00Z")
    df = pd.DataFrame(
        [
            {
                "context.trace_id": "t1",
                "parent_id": None,
                "name": "retrieve",
                "span_kind": "RETRIEVER",
                "status_code": "UNSET",
                "start_time": start,
                "end_time": pd.NaT,
            }
        ],
        index=["sp1"],
    )
    return df, start


def test_phoenix_nat_end_time_survives_to_span_as_none():
    """A NaT end_time must reach ``Span.end_time_ns`` as None, not a coerced 0.

    Exercises the full Phoenix read path (DataFrame row -> normalised span
    dict -> OpenInference adapter -> ``Span``) — the seam where the
    ``int(... or 0)`` coercion in ``common_span_fields`` used to undo
    ``to_unix_nano``'s None.
    """
    from ragpill.backends.phoenix_backend import _trace_from_spans_dataframe  # pyright: ignore[reportPrivateUsage]

    df, start = _nat_end_time_dataframe()
    trace = _trace_from_spans_dataframe(df, "t1")  # pyright: ignore[reportArgumentType]
    assert trace is not None
    (span,) = trace.spans
    assert span.start_time_ns == start.value  # pyright: ignore[reportAttributeAccessIssue]
    assert span.end_time_ns is None
    assert span.end_time_ns != 0


def test_phoenix_nat_end_time_renders_duration_unknown():
    """The report renderer must say 'duration unknown' for a Phoenix NaT span."""
    from ragpill.backends.phoenix_backend import _trace_from_spans_dataframe  # pyright: ignore[reportPrivateUsage]
    from ragpill.report._trace import render_spans

    df, _ = _nat_end_time_dataframe()
    trace = _trace_from_spans_dataframe(df, "t1")  # pyright: ignore[reportArgumentType]
    out = render_spans(trace, max_chars=10_000)
    assert "retrieve" in out
    assert "duration unknown" in out
    assert "0ms" not in out
