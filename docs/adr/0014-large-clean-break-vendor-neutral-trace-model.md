# ADR-0014: Clean break to the vendor-neutral trace model (no feature flag)

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Large
**Related:** designs/otel-trace-ingestion.md, ADR-0013, ADR-0015

## Context
Phase 2 of `designs/otel-trace-ingestion.md` flips the renderer, the three
span-based evaluators, and the run-JSON layer from `mlflow.entities.Trace`
onto the vendor-neutral `ragpill.trace.Trace`. The design proposed a
`RAGPILL_USE_NEW_TRACE_MODEL=1` feature flag to stage the cutover with dual
code paths, plus a JSON v1 → v2 migrator so old run files would keep loading.

Staging machinery (flag, dual path, migrator) only earns its complexity if
backward compatibility *during the transition* is a goal. For this project at
this stage it is not.

## Decision
No flag, no dual code path, no JSON migrator. Convert the captured MLflow
trace to `ragpill.trace.Trace` once, at the capture boundary; everything
downstream consumes the neutral model only.

Run-JSON writes the new shape with `schema_version=2` and ships no v1
migrator. Backward compatibility is explicitly not a target. The breaking
changes are documented in a CHANGELOG and the package version is bumped
0.5.0 → 0.6.0:

- `EvaluatorContext.trace` type changes
- `SpanBaseEvaluator.get_trace` return type changes
- run-JSON on-disk schema changes (v1 → v2)

(A convenience-only compat wrapper for external custom evaluators is recorded
separately in ADR-0015.)

## Alternatives considered
- **The design's `RAGPILL_USE_NEW_TRACE_MODEL` flag + dual path + v1→v2 JSON
  migrator.** Rejected: staging and rollback only pay off when backward
  compatibility during the transition is a goal, which it is not here. The
  dual path would roughly double the surface under test for no lasting
  benefit.

## Consequences
- Single code path everywhere downstream of capture; nothing has to branch on
  which trace model is active.
- Existing on-disk run files written under the old schema will not load. This
  is acceptable and documented in the CHANGELOG.
- External custom `SpanBaseEvaluator` subclasses break unless they adopt
  `ragpill.trace.Span` (or the courtesy compat wrapper from ADR-0015). The
  type changes to `EvaluatorContext.trace` and `get_trace` are the visible
  break.
- The capture boundary is now the single conversion point from MLflow to the
  neutral model — the one place to look when reasoning about trace fidelity.

## References
- `ragpill.trace.Trace`, `ragpill.trace.Span`
- `EvaluatorContext.trace`, `SpanBaseEvaluator.get_trace`
- run-JSON layer (`schema_version=2`)
- CHANGELOG; version bump 0.5.0 → 0.6.0
- `designs/otel-trace-ingestion.md` Phase 2
- ADR-0013 (adapter interface), ADR-0015 (compat shim)
- Chat session 2026-06-19
