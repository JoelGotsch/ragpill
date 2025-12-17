# Plan: OTel ingestion Phase 3 — more dialects + auto-detect loader

**Status:** In progress
**Date:** 2026-06-19
**Branch:** `feature/multi-backend-step1-protocols`
**Design:** `designs/otel-trace-ingestion.md` §6, §9 Phase 3
**Builds on:** Phase 1 (model + MLflow adapter), Phase 2 (the flip).

## Goal

Make trace ingestion multi-dialect: add `gen_ai` and `openinference` adapters,
a per-span auto-detect registry, and a `parse_otel(...)` entry point that turns
OTLP-JSON (or a list of normalised span dicts) into a `ragpill.trace.Trace`.
This unblocks the Phoenix backend (OpenInference-native) in multi-backend
Phase 2.

## Steps

### A. Registry + detection + fallback
- `trace/registry.py`: built-in adapter registry with a priority order
  (mlflow > openinference > gen_ai for now; langfuse/openllmetry/logfire slots
  reserved). `select_adapter(span_dict)` returns the first adapter whose
  `signature_attributes()` are all present. Also discovers third-party adapters
  via the `ragpill.trace_adapters` entry-point group (additive; built-ins are
  registered in code so no reinstall is needed).
- `trace/detect.py`: `detect_dialect(span_dict) -> str | None` thin wrapper.
- `trace/fallback.py`: `universal_span(span_dict) -> Span` best-effort extractor
  for when nothing matches; emits a single `warnings.warn` per parse.

### B. gen_ai adapter (`adapters/gen_ai.py`, always-on, no dep)
- signature: `gen_ai.system`. kind=LLM. model from `gen_ai.request.model` /
  `gen_ai.response.model`; usage from `gen_ai.usage.{input,output}_tokens`;
  model_parameters from `gen_ai.request.{temperature,top_p,max_tokens}`.
- Messages are **events**, not attributes: read `span.events`
  (`gen_ai.{user,system,assistant,tool}.message` → messages_in; `gen_ai.choice`
  / assistant → messages_out), each event body carrying `content`/`tool_calls`.

### C. openinference adapter (`adapters/openinference.py`, always-on, no dep)
- signature: `openinference.span.kind`. kind from that key (incl. GUARDRAIL).
- inputs/outputs from `input.value` / `output.value`.
- messages from indexed flat keys `llm.input_messages.N.message.{role,content}`
  / `llm.output_messages.N...` — reconstruct lists by parsing key suffixes.
- documents from `retrieval.documents.N.document.{id,score,content,metadata}`
  → `ragpill.trace.Document` (this is what the source-based evaluators read).
- model from `llm.model_name`; usage from `llm.token_count.{prompt,completion,total}`.
- **Deviation from design §6.3:** ships always-on (no `ragpill[openinference]`
  extra). The adapter only reads string-keyed attributes, so the
  `openinference-semantic-conventions` package would add named constants but no
  behaviour; gating a pure-attribute reader behind an extra is complexity for no
  runtime benefit. The extra is reserved for if we later import their constants.

### D. parse_otel loader entry point
- `loader.parse_otel(source, *, dialect="auto", fallback_dialect="gen_ai") -> Trace`.
  Phase 3 supports two `source` forms (OTLP-JSON only, per design §11 Q3):
  a normalised `list[span_dict]`, and an OTLP-JSON `ResourceSpans` dict
  (decode the `{key, value:{stringValue|intValue|...}}` attribute encoding).
  `dialect="auto"` selects per span via the registry; an explicit name forces
  one adapter. Misses fall to `fallback_dialect`.
- `from_mlflow_trace` stays the dedicated MLflow-capture path (unchanged).

### E. RagpillTraceSettings
- `settings.RagpillTraceSettings` (env `RAGPILL_TRACE_`): `dialect="auto"`,
  `fallback_dialect="gen_ai"`. `parse_otel` reads these as defaults. Not forced
  into `execute_dataset` — live capture is MLflow-native via `from_mlflow_trace`.

## Tests
- Adapter unit tests on hand-built span dicts (gen_ai events, openinference
  indexed keys + retrieval docs).
- Registry/detect: right adapter chosen by signature; priority ties; fallback.
- `parse_otel`: list + OTLP-JSON dict forms; auto vs forced dialect; mixed-dialect
  trace records per-span dialect; unknown-dialect warns once.
- Settings env parsing.

## Out of scope (Phase 4+)
- openllmetry / langfuse / logfire adapters.
- OTLP-protobuf, file, and vendor-export input formats for `parse_otel`.
- The universal fallback delegating to a full gen_ai parse (kept minimal).
