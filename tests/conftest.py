"""Shared pytest configuration for the ragpill test suite.

Disable MLflow's async trace export by default. The unit tests create
spans with ``mlflow.start_span`` and then immediately read them back
with ``mlflow.search_traces``. Async export defers the write to a
background queue, so the search races and returns ``[]``. Forcing
synchronous export keeps unit tests deterministic; the
``mlflow-integration`` CI job exercises the async path against a real
tracking server. See ADR-2004 — pending write-up; tracked in
plans/adr-system.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
