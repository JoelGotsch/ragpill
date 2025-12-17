# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) for `ragpill`. Each
ADR captures a single decision, its alternatives, and its consequences. The
format, naming, and impact tiers are defined in `plans/adr-system.md`.

- **Naming:** `docs/adr/NNNN-<tier>-kebab-title.md`, `NNNN` a zero-padded
  4-digit pure-sequential id, `<tier>` ∈ `large` | `medium` | `small`.
- **Template:** see [`template.md`](./template.md).
- **Numbers are immutable handles.** Code references (`# See ADR-0011`) must
  keep working forever, so impact lives in the filename suffix and the table
  below, never in the number.

## Reserved: ADR-0001 – ADR-0010 (backfill)

ADR-0001 through ADR-0010 are **reserved** for not-yet-written backfill
decisions, per the date-ordered backlog in `plans/adr-system.md`. Those files
do not exist yet; the numbers are held so the backfill can land in strict
decision-date order without colliding with newer ADRs. New ADRs therefore
start at ADR-0011.

| Number | Date | Impact | Title | Status |
|---|---|---|---|---|
| ADR-0001 | ~inception | Large | MLflow as the canonical tracing/eval backend | Reserved (backfill) |
| ADR-0002 | 2026-04-23 | Large | Three-layer architecture: execute / evaluate / upload | Reserved (backfill) |
| ADR-0003 | 2026-04-24 | Medium | Phase B (MCP server): directory-only, no live MLflow lookup | Reserved (backfill) |
| ADR-0004 | 2026-04-27 | Medium | LLM-judge trace suppression at OTel exporter layer | Reserved (backfill) |
| ADR-0005 | 2026-05-05 | Medium | Pluggable OTel trace ingestion vs. hard-coded MLflow exporter | Reserved (backfill) |
| ADR-0006 | 2026-05-05 | Medium | `EvaluationOutput.to_json` uses `orient="table"` not `"split"` | Reserved (backfill) |
| ADR-0007 | 2026-05-05 | Small | `RunResult.error` coerced to `RuntimeError` on JSON roundtrip | Reserved (backfill) |
| ADR-0008 | 2026-05-05 | Small | Trace subtree filter span-type set (expanded from design's four) | Reserved (backfill) |
| ADR-0009 | 2026-05-05 | Small | `model_params` not surfaced in triage header | Reserved (backfill) |
| ADR-0010 | 2026-05-13 | Small | Disable async MLflow trace logging in unit tests | Reserved (backfill) |

## Written ADRs (0011+)

| Number | Date | Impact | Title | Status |
|---|---|---|---|---|
| [ADR-0011](./0011-medium-await-trace-in-backend-protocol.md) | 2026-06-19 | Medium | Trace-readiness polling lives in the backend protocol (`await_trace`); configurable poll budget | Accepted |
| [ADR-0012](./0012-small-surface-evaluator-failures-in-triage.md) | 2026-06-19 | Small | Surface evaluator failures in triage without changing `all_passed` | Superseded by ADR-0018 |
| [ADR-0013](./0013-large-otel-adapter-option-c-from-otel.md) | 2026-06-19 | Large | OTel dialect-adapter interface — Option C (loader → normalised OTLP-JSON → `from_otel`) | Accepted |
| [ADR-0014](./0014-large-clean-break-vendor-neutral-trace-model.md) | 2026-06-19 | Large | Clean break to the vendor-neutral trace model (no feature flag) | Accepted |
| [ADR-0015](./0015-small-duck-typed-to-mlflow-trace-compat-shim.md) | 2026-06-19 | Small | Duck-typed `to_mlflow_trace` compat shim, convenience-only | Superseded by ADR-0016 |
| [ADR-0016](./0016-small-remove-to-mlflow-trace-compat-shim.md) | 2026-06-19 | Small | Remove the `to_mlflow_trace` compat shim | Accepted |
| [ADR-0017](./0017-medium-await-trace-returns-neutral-trace.md) | 2026-06-19 | Medium | `get_trace` / `await_trace` return the neutral trace, not native | Accepted |
| [ADR-0018](./0018-medium-conservative-gateable-metrics.md) | 2026-07-07 | Medium | Gateable metrics are conservative; infra failures fail closed | Accepted |
