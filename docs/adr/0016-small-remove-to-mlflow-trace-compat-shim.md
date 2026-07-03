# ADR-0016: Remove the `to_mlflow_trace` compat shim

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Small
**Related:** ADR-0014, ADR-0015 (superseded), designs/otel-trace-ingestion.md

## Context
ADR-0015 added `ragpill.trace.compat.to_mlflow_trace` as a convenience escape
hatch for external custom `SpanBaseEvaluator` subclasses written against
`mlflow.entities.Trace`. A subsequent backwards-compatibility sweep of the
codebase flagged it as the clearest piece of pure back-compat code: it was not
in `ragpill.trace.__all__`, had no internal callers, and existed only to ease
external migration. The project's stance is that backward compatibility is not
a target.

## Decision
Remove the compat shim entirely: delete `src/ragpill/trace/compat.py`, its test,
and all references (the `ragpill.trace` package docstring, the API docs page,
the CHANGELOG "Added" entry). External custom evaluators migrate to
`ragpill.trace.Span` directly, exactly like the internal ones did.

## Alternatives considered
- **Keep the shim (ADR-0015's position).** Rejected: a convenience-only shim is
  still standing surface to maintain and document, and keeping it contradicts
  the stated "backward compatibility is not a target" stance. The migration it
  eased is a mechanical field rename (`span.span_type` → `span.kind`,
  `trace.data.spans` → `trace.spans`).
- **Keep it but mark deprecated.** Rejected: deprecation cycles are a
  backward-compatibility mechanism the project has opted out of.

## Consequences
- ~120 lines of specialised, never-internally-used code removed.
- External custom `SpanBaseEvaluator` subclasses that read the MLflow trace
  surface must migrate to `ragpill.trace.Span` with no shim; this is documented
  in the CHANGELOG breaking-changes entry (ADR-0014).
- One fewer public surface to keep in sync as the trace model evolves.

## References
- Removed: `src/ragpill/trace/compat.py`, `tests/trace/test_ops.py` compat test
- ADR-0014 (clean break to vendor-neutral trace model)
- ADR-0015 (the now-superseded decision to add the shim)
- Chat session 2026-06-19
