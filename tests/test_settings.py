"""Tests for MLFlowSettings env-var loading and resolve_repeat()."""

import pytest

from ragpill.base import TestCaseMetadata, resolve_repeat
from ragpill.settings import MLFlowSettings

# ---------------------------------------------------------------------------
# MLFlowSettings from environment
# ---------------------------------------------------------------------------


def test_repeat_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_RAGPILL_REPEAT", "5")
    settings = MLFlowSettings()  # type: ignore[call-arg]
    assert settings.ragpill_repeat == 5


def test_threshold_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_RAGPILL_THRESHOLD", "0.7")
    settings = MLFlowSettings()  # type: ignore[call-arg]
    assert settings.ragpill_threshold == 0.7


# ---------------------------------------------------------------------------
# resolve_repeat
# ---------------------------------------------------------------------------


def _settings(repeat: int = 1, threshold: float = 1.0) -> MLFlowSettings:
    return MLFlowSettings(ragpill_repeat=repeat, ragpill_threshold=threshold)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "case_repeat, case_threshold, global_repeat, global_threshold, expected_repeat, expected_threshold",
    [
        (None, None, 3, 0.8, 3, 0.8),
        (5, None, 1, 1.0, 5, 1.0),
        (None, 0.5, 1, 1.0, 1, 0.5),
        (7, 0.3, 1, 1.0, 7, 0.3),
    ],
    ids=["both_none_uses_global", "case_overrides_repeat", "case_overrides_threshold", "case_overrides_both"],
)
def test_resolve_repeat(
    case_repeat, case_threshold, global_repeat, global_threshold, expected_repeat, expected_threshold
):
    meta = TestCaseMetadata(repeat=case_repeat, threshold=case_threshold)
    r, t = resolve_repeat(meta, _settings(repeat=global_repeat, threshold=global_threshold))
    assert r == expected_repeat
    assert t == expected_threshold


def test_resolve_repeat_no_metadata():
    r, t = resolve_repeat(None, _settings(repeat=2, threshold=0.9))
    assert r == 2
    assert t == 0.9
