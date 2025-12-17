# ADR-0015: Duck-typed `to_mlflow_trace` compat shim, convenience-only

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Small
**Related:** ADR-0014, designs/otel-trace-ingestion.md

## Context
The clean break to the vendor-neutral trace model (ADR-0014) breaks external
custom `SpanBaseEvaluator` subclasses that still expect an
`mlflow.entities.Trace`. Backward compatibility is not a project goal, but a
cheap escape hatch for external evaluators authored against the old type is
worth providing — provided it stays out of the internal code path.

## Decision
Provide `ragpill.trace.compat.to_mlflow_trace` as a duck-typed wrapper that
adapts a `ragpill.trace.Trace` to the shape an MLflow-expecting evaluator
reads. It is a courtesy escape hatch for external custom `SpanBaseEvaluator`
subclasses only — convenience-only, and unused internally.

## Alternatives considered
- **A real `mlflow.entities.Trace` reconstruction for the shim.** Rejected as
  fragile: MLflow `Span`s wrap OTel `ReadableSpan`s, so faithfully rebuilding
  a true MLflow Trace is brittle and high-maintenance. A duck-typed wrapper
  that satisfies the attributes evaluators actually read is enough.
- **No compat shim at all.** Rejected in favour of the cheap courtesy
  wrapper — the wrapper is small and gives external authors a migration path
  without committing the project to backward compatibility.

## Consequences
- External custom evaluators have a one-line migration path
  (`to_mlflow_trace(trace)`) instead of an immediate rewrite.
- The shim is convenience-only and never used internally, so it imposes no
  constraint on the internal single code path (ADR-0014).
- Because it is duck-typed, it satisfies the attributes evaluators read, not
  the full `mlflow.entities.Trace` contract; an external evaluator reaching
  for an unsupported MLflow Trace method will still break.

## References
- `ragpill.trace.compat.to_mlflow_trace`
- ADR-0014 (clean break to vendor-neutral trace model)
- Chat session 2026-06-19
