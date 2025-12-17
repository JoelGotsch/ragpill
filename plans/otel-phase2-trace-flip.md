# Plan: OTel ingestion Phase 2 — flip onto the vendor-neutral trace model

**Status:** In progress
**Date:** 2026-06-19
**Branch:** `feature/multi-backend-step1-protocols`
**Design:** `designs/otel-trace-ingestion.md` §9 Phase 2
**Builds on:** Phase 1 (`ragpill.trace` model + MLflow loader, commit `a3d31a1`).

## Decision: clean break, no flag

The design proposed a `RAGPILL_USE_NEW_TRACE_MODEL` flag to *stage* the
cutover. That only pays off if backward compatibility during the transition
is a goal. It is **not** — breaking changes are handled by a changelog entry
and a version bump, and the internal evaluators are all adapted regardless.
So there is **no flag, no dual code path, no JSON migrator**. We convert the
captured MLflow trace into a `ragpill.trace.Trace` once at the capture
boundary; everything downstream consumes the neutral model only.

Backward compat is not a target. The one courtesy retained is
`ragpill.trace.compat.to_mlflow_trace` — a **duck-typed** wrapper exposing the
`mlflow.entities.Trace`/`Span` surface (``.data.spans``,
``.search_spans(span_type=...)``, ``span.span_type/inputs/outputs/attributes``)
for external custom `SpanBaseEvaluator` subclasses. It is convenience-only and
unused internally.

## Steps

### A. Serialization + compat wrapper (additive)
- `ragpill.trace.model`: `to_dict`/`from_dict` on `Trace` (+ nested
  Span/Message/Document/Usage). Enum-safe, round-trips `attributes`/`events`.
- `ragpill.trace.compat.to_mlflow_trace(trace)` — duck-typed wrapper.
- `ragpill.trace.ops.filter_to_subtree(trace, root_span_id)` — neutral
  reimplementation of the subtree filter (replaces the mlflow-specific
  `_filter_trace_to_subtree`).
- Tests: model round-trip; subtree filter; compat surface.

### B. Renderer migration
- `report/_trace.render_spans` consumes `ragpill.trace.Trace`: read
  `trace.spans`, `span.kind`, `span.inputs/outputs/attributes`,
  `span.span_id/parent_id/start_time_ns/end_time_ns`. `span.attributes`
  already excludes MLflow bookkeeping keys (the adapter dropped them), so the
  `_INTERNAL_ATTR_KEYS` filtering simplifies away.
- Update `report/exploration.py` if it reads span shape.
- Renderer tests now build a `ragpill.trace.Trace` (via `from_mlflow_trace`
  off a real mlflow trace, keeping the same fixtures).

### C. Evaluator migration
- `EvaluatorContext.trace` type → `ragpill.trace.Trace | None`.
- `SpanBaseEvaluator.get_trace` returns `ragpill.trace.Trace`, filtered via
  `ops.filter_to_subtree`.
- `SourcesBaseEvaluator.get_documents`: select spans by
  `kind in {RETRIEVER, TOOL, RERANKER}`, read `span.outputs`, build the
  ragpill `Document` list as today.
- `LiteralQuoteEvaluator`, `RegexInSourcesEvaluator`,
  `RegexInDocumentMetadataEvaluator` ride on the base-class change.
- Update evaluator tests / fixtures.

### D. Capture-boundary conversion + JSON flip
- `execution.py`: after `await_trace`, convert the MLflow trace to
  `ragpill.trace.Trace` via `from_mlflow_trace`. `CaseRunOutput.trace` /
  `TaskRunOutput.trace` are now `ragpill.trace.Trace`. Subtree filtering uses
  `ops.filter_to_subtree`.
- `to_json`/`from_json`: serialise `ragpill.trace.Trace` via its `to_dict`,
  stamp `schema_version=2`. No v1 migrator (documented break).
- Update execution/JSON-roundtrip tests.

### E. Release hygiene
- `pyproject.toml`: `0.5.0` → `0.6.0` (pre-1.0 breaking change).
- `CHANGELOG.md`: new file, document the breaking trace-type change
  (`EvaluatorContext.trace`, `SpanBaseEvaluator.get_trace`, run-JSON schema).
- Docs sweep per `documentation-guidelines` (public types changed).

## Acceptance
- Full suite green after each step.
- `basedpyright src/` clean.
- No `mlflow.entities` import remains in `report/_trace.py` or the migrated
  evaluator paths (only in `trace/loader.py` and `trace/compat.py`).
