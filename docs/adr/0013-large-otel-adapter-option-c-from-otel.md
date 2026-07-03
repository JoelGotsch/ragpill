# ADR-0013: OTel dialect-adapter interface — Option C (loader → normalised OTLP-JSON → `from_otel`)

**Status:** Accepted
**Date:** 2026-06-19
**Impact:** Large
**Related:** designs/otel-trace-ingestion.md, ADR-0005

## Context
The in-progress `ragpill.trace` work had a `SpanAdapter` interface
(`from_native(vendor_trace)` / `span_from_native`) that consumed vendor SDK
trace objects directly. This diverged from `designs/otel-trace-ingestion.md`
§6 (Option C), which routes every input format through a loader that produces
normalised OTLP-JSON span dicts, with adapters exposing
`signature_attributes()` + `from_otel(span_dict)` and auto-detect dispatch.

ADR-0005 already decided *that* OTel ingestion should be pluggable rather than
hard-coded to MLflow. This ADR records the *interface shape within* that
pluggable design — what an adapter actually consumes.

## Decision
Adopt Option C. Adapters consume normalised OTLP-JSON span dicts and never
touch vendor SDK objects; the loader owns all input-format normalisation.
Adapters expose `signature_attributes()` and `from_otel(span_dict)`. This is
the interface finalised in `src/ragpill/trace/adapters/_base.py`.

The MLflow adapter and `loader.from_mlflow_trace` were built this way. The
multi-format `parse_otel` entry point, the entry-point registry, and per-span
auto-detect dispatch are deferred to Phase 3.

## Alternatives considered
- **Keep `from_native(vendor_obj)` for Phase 1 simplicity.** Rejected: it
  re-introduces exactly the MLflow-object coupling Option C exists to remove,
  and would force a rewrite of the MLflow adapter once gen_ai / openinference
  dialects land.
- **Options A / B / D from `designs/otel-trace-ingestion.md`** (single
  monolithic normaliser; per-field strategy registry; single hand-written
  reader). Rejected per the design document's own analysis.

## Consequences
- The adapter interface is a public, third-party extension point: anyone can
  add a dialect adapter against normalised OTLP-JSON without depending on a
  vendor SDK.
- Adapters are decoupled from vendor object models; the loader is the single
  place that knows how to normalise a given input format.
- Phase 3 work (`parse_otel`, entry-point registry, auto-detect) is still
  outstanding; today only the MLflow path is wired (`from_mlflow_trace`).
- Adding a new dialect means writing a loader normaliser + an adapter, not
  patching a central reader.

## References
- `src/ragpill/trace/adapters/_base.py` (`signature_attributes`, `from_otel`)
- `loader.from_mlflow_trace`; MLflow adapter
- `designs/otel-trace-ingestion.md` §6 (Option C)
- ADR-0005 (pluggable OTel ingestion)
- Chat session 2026-06-19
