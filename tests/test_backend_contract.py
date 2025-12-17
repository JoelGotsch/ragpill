"""Parametrized backend-contract suite.

Runs the same conformance checks against every in-tree backend, so a new
adapter only has to add its class to ``ALL_BACKENDS`` to inherit the whole
suite. Backends instantiate without their optional SDK (imports are lazy), so
these checks run in the default environment.
"""

from __future__ import annotations

import warnings

import pytest

from ragpill.backends import (
    Backend,
    LangfuseBackend,
    LifecycleBackend,
    MLflowBackend,
    PhoenixBackend,
    ResultsBackend,
    TraceCaptureBackend,
    TraceQueryBackend,
)

ALL_BACKENDS = [MLflowBackend, LangfuseBackend, PhoenixBackend]
# Backends with no native run concept: run tags / artifacts are no-ops.
SYNTHETIC_RUN_BACKENDS = [LangfuseBackend, PhoenixBackend]


@pytest.mark.parametrize("backend_cls", ALL_BACKENDS)
def test_satisfies_all_capability_protocols(backend_cls):
    backend = backend_cls()
    assert isinstance(backend, Backend)
    assert isinstance(backend, TraceCaptureBackend)
    assert isinstance(backend, TraceQueryBackend)
    assert isinstance(backend, ResultsBackend)
    assert isinstance(backend, LifecycleBackend)


@pytest.mark.parametrize("backend_cls", ALL_BACKENDS)
def test_supports_local_file_store_is_declared_bool(backend_cls):
    assert isinstance(backend_cls().supports_local_file_store, bool)


@pytest.mark.parametrize("backend_cls", ALL_BACKENDS)
def test_run_tag_and_artifact_methods_exist(backend_cls):
    backend = backend_cls()
    for name in ("set_run_tag", "get_run_tag", "delete_run_artifact"):
        assert callable(getattr(backend, name)), f"{backend_cls.__name__} missing {name}"


@pytest.mark.parametrize("backend_cls", SYNTHETIC_RUN_BACKENDS)
def test_synthetic_run_backends_treat_run_tags_as_noops(backend_cls):
    backend = backend_cls()
    # No native run concept -> get returns None (idempotency guard just proceeds),
    # set/delete are warn-once no-ops and must not raise.
    assert backend.get_run_tag("run-1", "ragpill_upload_state") is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend.set_run_tag("run-1", "k", "v")
        backend.delete_run_artifact("run-1", "evaluation_results.json")


@pytest.mark.parametrize("backend_cls", ALL_BACKENDS)
def test_resolve_experiment_id_is_stable_for_name_based_backends(backend_cls):
    # Langfuse/Phoenix identify experiments by name; MLflow needs a server, so
    # only assert the name-based ones here (they must echo the name).
    backend = backend_cls()
    if backend.supports_local_file_store:
        pytest.skip("MLflow resolves experiment ids against a server")
    assert backend.resolve_experiment_id("proj") == "proj"
