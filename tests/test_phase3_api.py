"""Phase 3: the clean-break API surface and behavior fixes.

Pins the renamed/settings surface: backend-neutral TrackingSettings with the
RAGPILL_ env prefix and a None (zero-server) default, the curated public
exports, evaluate_testset requiring a destination, and the entry-point
discovery cache.
"""

from __future__ import annotations

import pytest

import ragpill
from ragpill.settings import TrackingSettings


def test_tracking_settings_env_prefix_and_default(monkeypatch: pytest.MonkeyPatch):
    # Default tracking_uri is None (zero-server temp store) — no silent localhost.
    assert TrackingSettings().tracking_uri is None  # type: ignore[call-arg]
    monkeypatch.setenv("RAGPILL_TRACKING_URI", "http://mlflow.internal:5000")
    monkeypatch.setenv("RAGPILL_EXPERIMENT_NAME", "proj_eval")
    s = TrackingSettings()  # type: ignore[call-arg]
    assert s.tracking_uri == "http://mlflow.internal:5000"
    assert s.experiment_name == "proj_eval"


def test_public_exports_present_and_pruned():
    for name in (
        "TrackingSettings",
        "CaptureSpanKind",
        "Trace",
        "load_testset",
        "configure_backend",
        "get_backend",
        "evaluate_testset",
        "upload_results",
        "TraceUnavailableError",
    ):
        assert name in ragpill.__all__, f"{name} should be exported"
        assert hasattr(ragpill, name)
    # merge_settings (a logging util) is no longer part of the public surface.
    assert "merge_settings" not in ragpill.__all__


def test_backend_classes_lazily_exported():
    from ragpill.backends import MLflowBackend

    assert MLflowBackend.__name__ == "MLflowBackend"
    with pytest.raises(AttributeError):
        import ragpill.backends as b

        _ = b.NotARealBackend  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_evaluate_testset_requires_tracking_uri():
    from ragpill import Case, Dataset, TestCaseMetadata, evaluate_testset

    ds = Dataset[str, str, TestCaseMetadata](cases=[Case(inputs="q", metadata=TestCaseMetadata())])

    async def task(q: str) -> str:
        return q

    # No RAGPILL_TRACKING_URI configured -> clear error, not a silent localhost.
    with pytest.raises(ValueError, match=r"tracking URI is required|tracking server"):
        await evaluate_testset(ds, task=task, settings=TrackingSettings())  # type: ignore[call-arg]


def test_adapter_discovery_is_cached(monkeypatch: pytest.MonkeyPatch):
    import ragpill.trace.registry as reg

    reg.clear_adapter_cache()
    calls = {"n": 0}
    real = reg._discover_entry_point_adapters  # pyright: ignore[reportPrivateUsage]

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(reg, "_discover_entry_point_adapters", counting)
    reg.clear_adapter_cache()
    for _ in range(5):
        reg.adapters_in_priority_order()
    assert calls["n"] == 1  # discovery runs once, not per call/span


def test_capture_traces_is_keyword_only():
    import inspect

    from ragpill.execution import execute_dataset

    params = inspect.signature(execute_dataset).parameters
    assert params["capture_traces"].kind == inspect.Parameter.KEYWORD_ONLY
