# Design: Pluggable OTel Trace Ingestion

**Status:** Draft
**Date:** 2026-05-05
**Author:** ragpill core
**Related:**
- [designs/langfuse-integration.md](langfuse-integration.md) — emits OTel into Langfuse; this doc is the symmetric *ingestion* side.
- [designs/llm-readable-outputs-and-mcp.md](llm-readable-outputs-and-mcp.md) — defines the renderer that ultimately consumes the parsed traces.
- [src/ragpill/report/_trace.py](../src/ragpill/report/_trace.py) — current MLflow-coupled span renderer; this design generalises what it consumes.
- [src/ragpill/execution.py](../src/ragpill/execution.py) — current consumer of `mlflow.entities.Trace`.

---

## 1. Goal

Today ragpill's evaluators, renderers, and JSON serialisation all operate on
`mlflow.entities.Trace` / `mlflow.entities.Span`. Concretely the renderer
reads MLflow-flavoured attribute keys (`mlflow.spanType`, `mlflow.spanInputs`,
`mlflow.spanOutputs`) and the evaluators walk a span tree shaped exactly the
way `mlflow.pydantic_ai.autolog()` produces it.

This locks us to one particular *dialect* of OpenTelemetry. The same OTel
spans, when collected by a different vendor's SDK or instrumentation library,
look semantically equivalent but use different attribute names, different
input/output encodings, and a different notion of "span kind". We want
ragpill to be able to **ingest traces from any of the popular LLM-tracing
OTel dialects** and turn them into a single normalised internal trace model
that the renderer, evaluators, and JSON layer share.

The user-visible knob:

```toml
[tool.ragpill]
otel_dialect = "auto"  # or "mlflow" | "openinference" | "openllmetry" | "gen_ai" | "langfuse" | "logfire"
```

…and an **extension point** so adding a new dialect means a new optional
extra (`pip install ragpill[phoenix]`) plus a small adapter module — no
edits to evaluators, renderer, or core types.

## 2. Non-Goals

- Becoming an OTel collector. Ingestion happens **after** spans land in
  whatever store ragpill is plugged into (MLflow trace store, an `.otlp`
  dump on disk, a Phoenix project export, etc.). The wire-format ingestion
  is OTLP-JSON or OTLP-protobuf bytes; live OTLP receivers are out of scope.
- Inventing yet another LLM-trace semantic convention. We *consume* the
  existing dialects; we don't try to push our own as canonical.
- Lossless round-trip across dialects. We extract the fields ragpill
  actually uses (kind, model, messages, tool calls, retrieved documents,
  usage, errors). Vendor-specific extras land in a free-form
  `attributes` dict and are passed through but not re-typed.
- Replacing MLflow as the *capture* backend in `execute_dataset`. Capture
  stays MLflow-default for now; this design changes only what we accept on
  the *consumption* side. Capture pluggability is a follow-up.

---

## 3. Landscape: OTel-flavoured LLM tracing dialects

Below is the inventory. For each: who emits it, the attribute namespace
they use, what concepts they encode, and the primary place ragpill is
likely to encounter it. Names below are what the current SDKs emit; they
do drift, so the adapter modules pin to a known schema version range.

### 3.1 OpenTelemetry GenAI Semantic Conventions (`gen_ai.*`)

- **Source:** Official OTel semconv working group (`semantic-conventions/docs/gen-ai/`).
- **Status (early 2026):** *Stable* for chat / completion attributes;
  *experimental* for embedding, agent, and tool-call attributes.
- **Emitters:** `pydantic-ai` (native), `openai-python` (≥1.40 with the
  OTel instrumentation), `anthropic-python` (≥0.40), `langchain-otel`,
  Vercel AI SDK's OTel emitter, Logfire under the hood.
- **Key attributes:**
  - `gen_ai.system` (e.g. `"openai"`, `"anthropic"`)
  - `gen_ai.operation.name` (`"chat"`, `"text_completion"`, `"embeddings"`)
  - `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`
  - `gen_ai.request.{temperature,top_p,max_tokens,stop_sequences}`
  - `gen_ai.usage.{input_tokens,output_tokens}`
  - Messages: emitted as **events** on the span:
    `gen_ai.user.message`, `gen_ai.system.message`, `gen_ai.assistant.message`,
    `gen_ai.tool.message`, `gen_ai.choice` — each event has a `body` JSON
    blob with `content`, `tool_calls`, etc.
- **Ragpill concepts mapped:** kind=LLM, input messages, output choices,
  model, usage. No retrieval / no agent kind in stable; experimental
  conventions add `gen_ai.tool.name`.
- **Quirk:** Because messages are *events*, not attributes, parsers that
  only look at `span.attributes` miss the actual content. Adapters must
  read `span.events`.

### 3.2 OpenInference (Arize / Phoenix)

- **Source:** [github.com/Arize-ai/openinference](https://github.com/Arize-ai/openinference).
- **Status:** Most widely-supported "span-kinds-for-LLM-apps" spec.
- **Emitters:** Phoenix's auto-instrumentation, LangChain via
  `openinference-instrumentation-langchain`, LlamaIndex via
  `openinference-instrumentation-llama-index`, Bedrock, DSPy, Haystack,
  Mistral, Groq — broadest ecosystem coverage of any dialect.
- **Key attributes:**
  - `openinference.span.kind` ∈ {`LLM`, `RETRIEVER`, `EMBEDDING`,
    `RERANKER`, `AGENT`, `TOOL`, `GUARDRAIL`, `CHAIN`, `EVALUATOR`,
    `UNKNOWN`}
  - `input.value`, `output.value` — the raw I/O blob (often JSON)
  - `input.mime_type`, `output.mime_type`
  - LLM messages: indexed flat keys —
    `llm.input_messages.0.message.role`,
    `llm.input_messages.0.message.content`,
    `llm.input_messages.0.message.tool_calls.0.tool_call.id`, etc.
    Same shape for `llm.output_messages.*`.
  - `llm.model_name`, `llm.provider`, `llm.invocation_parameters` (JSON),
    `llm.token_count.{prompt,completion,total}`
  - Retrieval: `retrieval.documents.0.document.{id,score,content,metadata}`
  - Tool: `tool.name`, `tool.description`, `tool.parameters`
  - Embedding: `embedding.embeddings.0.embedding.{vector,text}`
- **Ragpill concepts mapped:** essentially everything — this is the
  dialect ragpill's evaluators map most cleanly to. `RegexInSourcesEvaluator`
  in particular wants `retrieval.documents.*.document.content`.
- **Quirk:** "Indexed flat keys" mean the adapter has to reconstruct lists
  by parsing key suffixes. Most other dialects use JSON blobs for the
  same data.

### 3.3 OpenLLMetry / Traceloop SDK (`traceloop.*` + `llm.*`)

- **Source:** Traceloop's [openllmetry](https://github.com/traceloop/openllmetry),
  predates the GenAI semconv.
- **Emitters:** Their own SDK (`traceloop-sdk`), plus a large set of
  per-vendor instrumentations (`opentelemetry-instrumentation-openai`,
  `-anthropic`, `-cohere`, `-bedrock`, `-replicate`, `-pinecone`,
  `-chromadb`, `-qdrant`, `-haystack`, `-langchain`, `-llamaindex`).
- **Key attributes:**
  - `traceloop.workflow.name`, `traceloop.entity.name`,
    `traceloop.entity.input` (JSON), `traceloop.entity.output` (JSON)
  - `traceloop.span.kind` ∈ {`workflow`, `task`, `agent`, `tool`,
    `unknown`} — *different* enum than OpenInference
  - LLM-specific (largely overlapping with the experimental GenAI conv,
    pre-rename): `llm.request.type`, `llm.request.model`,
    `llm.response.model`, `llm.usage.total_tokens`,
    `llm.usage.prompt_tokens`, `llm.usage.completion_tokens`
  - Messages: `llm.prompts.0.role`, `llm.prompts.0.content`,
    `llm.completions.0.role`, `llm.completions.0.content`,
    `llm.completions.0.finish_reason`
  - Vector DB: `db.system` + `vector.query.top_k` + `vector.query.embedding.0`
- **Ragpill concepts mapped:** kind, messages, model, usage. No
  rich retriever-document concept (vector DB calls show as DB spans, not
  "retriever spans with documents"); evaluators that walk retrieved docs
  will have to lean on payload JSON.
- **Quirk:** Two parallel namespaces (`traceloop.*` for workflow,
  `llm.*` for model details). A single span often carries both.

### 3.4 MLflow (`mlflow.*`)

- **Source:** MLflow's tracing module ([`mlflow.tracing`](https://mlflow.org/docs/latest/llms/tracing/index.html)).
  This is what ragpill currently produces.
- **Emitters:** `mlflow.<framework>.autolog()` (pydantic_ai, langchain,
  llama_index, openai, dspy, …) + `mlflow.start_span()`.
- **Key attributes:**
  - `mlflow.spanType` ∈ {`LLM`, `CHAT_MODEL`, `RETRIEVER`, `RERANKER`,
    `TOOL`, `AGENT`, `CHAIN`, `EMBEDDING`, `PARSER`, `UNKNOWN`}
  - `mlflow.spanInputs`, `mlflow.spanOutputs` (JSON blobs)
  - `mlflow.traceRequestId`
  - `mlflow.spanFunctionName`
- **Ragpill concepts mapped:** all of the above directly via the existing
  renderer. MLflow's own `Trace` object also surfaces inputs/outputs as
  typed properties (`span.inputs`, `span.outputs`); the underlying OTel
  attributes are these `mlflow.*` keys.
- **Quirk:** When MLflow autoinstruments pydantic-ai, the resulting span
  carries *both* `mlflow.*` attributes and `gen_ai.*` events from
  pydantic-ai's native OTel emission. Either layer is sufficient; some
  evaluators currently rely on MLflow having normalised them.

### 3.5 Langfuse (`langfuse.*`)

- **Source:** Langfuse's [Python SDK v3](https://langfuse.com/docs/sdk/python/sdk-v3)
  emits OTel spans with their own attribute layer; their server also
  *ingests* OpenInference and GenAI spans via their OTel endpoint and
  maps them inward.
- **Emitters:** `langfuse-python` (≥3.0), Langfuse's OTel processor.
- **Key attributes:**
  - `langfuse.observation.type` ∈ {`SPAN`, `GENERATION`, `EVENT`}
  - `langfuse.observation.input`, `langfuse.observation.output` (JSON)
  - `langfuse.observation.model`, `langfuse.observation.model.parameters`
  - `langfuse.observation.usage.{input,output,total}`
  - `langfuse.trace.{session_id,user_id,tags,metadata}`
  - `langfuse.observation.completion_start_time`
- **Ragpill concepts mapped:** kind (their three-way split is coarser
  than OpenInference's; `GENERATION` ≈ LLM, `SPAN` ≈ everything else),
  model, usage, session/user grouping. No native retriever-document
  concept — Langfuse expects retrieval to be encoded as `SPAN` with
  payload JSON.
- **Quirk:** Sessions and user-ids are *trace-level* in Langfuse but
  often emitted as span-level OTel attributes; the adapter has to
  promote them.

### 3.6 Logfire (Pydantic) (`logfire.*` + `gen_ai.*`)

- **Source:** [Pydantic Logfire SDK](https://pydantic.dev/logfire). Built
  on OTel; their own attribute layer plus the GenAI conv for LLM calls.
- **Emitters:** `logfire` SDK; pydantic-ai under Logfire reuses GenAI
  semconv.
- **Key attributes:**
  - `logfire.msg_template`, `logfire.msg`, `logfire.span_type`,
    `logfire.level_num`, `logfire.tags`
  - For LLM calls: pure GenAI semconv (`gen_ai.*`).
- **Ragpill concepts mapped:** Almost a strict subset of "`gen_ai` for
  the LLM bits + structured logging metadata for the rest". The adapter
  is largely a thin wrapper around the `gen_ai` adapter.

### 3.7 LangSmith / LangChain tracing

- **Source:** LangSmith's tracing format. Until recently this was
  HTTP/JSON-only; LangChain's newer `langchain-otel` + LangSmith's OTel
  bridge emits OTel spans now.
- **Status:** In flux. Their OTel-flavoured spans use
  `ls_provider`, `ls_model_name`, `ls_run_type`, `ls_thread_id`. The
  more reliable path today is to use the OpenInference LangChain
  instrumentation (which emits OpenInference attributes for the same
  agent invocation) instead of LangSmith's native OTel.
- **Recommendation:** *Do not ship a LangSmith adapter in v1.* Tell users
  to plug `openinference-instrumentation-langchain` instead. Revisit if
  the LangSmith OTel format stabilises.

### 3.8 Vendor-shaped honourable mentions

These all *consume* OTel and accept it via OTLP, but have no distinct
*emission* dialect of their own that we'd plausibly receive in the wild
without one of the above being installed alongside them:

- **Honeycomb / Datadog / NewRelic / Grafana Tempo** — pure consumers.
- **Helicone** — proxy-based, doesn't speak OTel directly.
- **W&B Weave** — proprietary span format on the wire; not OTel.
- **Galileo** — has an OTel ingestion endpoint but their own SDK emits a
  proprietary format. Skip.

### 3.9 Summary table

| Dialect | Span kind attr | I/O encoding | Messages | Documents | Usage | Sessions | Notes |
|---|---|---|---|---|---|---|---|
| `gen_ai` | (none — derived from `gen_ai.operation.name`) | events on span | events `gen_ai.*.message` | — | `gen_ai.usage.*` | — | Stable for chat. |
| OpenInference | `openinference.span.kind` | `input.value` / `output.value` (JSON) | `llm.input_messages.{i}.*` | `retrieval.documents.{i}.document.*` | `llm.token_count.*` | `session.id` (experimental) | Richest. |
| OpenLLMetry | `traceloop.span.kind` | `traceloop.entity.{input,output}` (JSON) | `llm.prompts.{i}.*` / `llm.completions.{i}.*` | (DB span payload) | `llm.usage.*` | `traceloop.workflow.name` | Two namespaces. |
| MLflow | `mlflow.spanType` | `mlflow.span{Inputs,Outputs}` (JSON) | (in I/O JSON) | (in I/O JSON) | (in I/O JSON) | — | Current ragpill internal. |
| Langfuse | `langfuse.observation.type` | `langfuse.observation.{input,output}` (JSON) | (in I/O JSON) | — | `langfuse.observation.usage.*` | `langfuse.trace.session_id` | 3-way kind split. |
| Logfire | `logfire.span_type` + `gen_ai.*` | events (delegates to GenAI) | events | — | `gen_ai.usage.*` | — | Subset of GenAI. |

---

## 4. Internal model

Ragpill needs *one* normalised representation that the renderer, the
JSON layer, and the evaluators all consume. Today that role is played
implicitly by `mlflow.entities.Span`. We propose an explicit, dataclass-only
model in `ragpill.trace` that is dialect-agnostic.

```python
# src/ragpill/trace/model.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class SpanKind(str, Enum):
    LLM = "LLM"
    CHAT_MODEL = "CHAT_MODEL"
    EMBEDDING = "EMBEDDING"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    TOOL = "TOOL"
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    GUARDRAIL = "GUARDRAIL"
    PARSER = "PARSER"
    TASK = "TASK"
    UNKNOWN = "UNKNOWN"

@dataclass
class Message:
    role: str            # "user" | "system" | "assistant" | "tool"
    content: Any         # str or list of content parts
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None

@dataclass
class Document:
    id: str | None
    score: float | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

@dataclass
class Span:
    span_id: str
    parent_id: str | None
    trace_id: str
    name: str
    kind: SpanKind
    start_time_ns: int
    end_time_ns: int
    status: str          # "OK" | "ERROR" | "UNSET"
    status_message: str | None
    inputs: Any | None   # decoded JSON if available
    outputs: Any | None
    messages_in: list[Message] = field(default_factory=list)
    messages_out: list[Message] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    model: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    attributes: dict[str, Any] = field(default_factory=dict)
    # ^ everything we *didn't* extract — passed through verbatim
    events: list[dict[str, Any]] = field(default_factory=list)
    dialect: str = "unknown"
    # ^ "mlflow" | "openinference" | "openllmetry" | "gen_ai" | …

@dataclass
class Trace:
    trace_id: str
    spans: list[Span]
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # primary dialect detected on this trace; spans may differ if mixed
    dialect: str = "unknown"
```

Two notable choices:

1. **Optional fields, not abstract methods.** A retriever span that
   produced no documents simply has `documents=[]`. Evaluators check
   the field, not the kind.
2. **`attributes` carries pass-through.** Anything the adapter didn't
   recognise (e.g. an OpenInference `metadata.user_query_intent` set by
   the user) lands there so users can write custom evaluators against
   raw vendor attributes when they need to.

The renderer in [`ragpill/report/_trace.py`](../src/ragpill/report/_trace.py)
moves to consuming `ragpill.trace.Span` instead of `mlflow.entities.Span`.
That's the only renderer change needed — same shape, slightly different
field names.

---

## 5. Architecture options

We considered four architectures. They differ in **where the dialect-specific
logic lives** and **how a new dialect is added**. The tradeoffs are real;
we recommend Option C (auto-detect chain of dialect adapters, registered
via entry points) but the others are documented for context.

### 5.1 Option A — Single normaliser via "promote-to-canonical" pre-passes

```
Foreign OTel spans
   │
   ▼
Pre-normaliser (per-dialect): mutate attributes
  e.g. `traceloop.entity.input` → `gen_ai.*` + `input.value`
   │
   ▼
One canonical reader (reads OpenInference shape)
   │
   ▼
ragpill.trace.Trace
```

- **Plus:** Only one "real" reader. Adding a dialect = writing a small
  attribute-renaming function.
- **Minus:** Lossy, in-place mutation of incoming spans. Renamings collide
  (OpenLLMetry's `llm.prompts.{i}` and OpenInference's
  `llm.input_messages.{i}` overlap on the `llm.` prefix but mean different
  things). The "canonical" choice forces every dialect to bend toward one
  particular vendor's worldview.
- **Verdict:** Fragile. Skipped.

### 5.2 Option B — One-extractor-per-field strategy registry

```
Foreign OTel span
   │
   ▼
For each ragpill field (kind, model, messages_in, …):
  try registered extractors in priority order
  first hit wins
   │
   ▼
ragpill.trace.Span
```

- **Plus:** Extremely modular. Adding a dialect = registering N small
  functions. Mixed-dialect traces (one MLflow span, one OpenInference
  span) work transparently because every field is resolved
  independently.
- **Minus:** Hard to reason about for a single-dialect span — you can't
  trust that "this is the OpenInference adapter speaking", and the
  ordering of extractors becomes load-bearing. Debug noise. Also no
  natural place to record `dialect` on the resulting `Span`.
- **Verdict:** Powerful but operationally annoying. Kept as a fallback
  layer (see §5.3 step 3).

### 5.3 Option C — Auto-detect dialect → dispatch to whole-span adapter (recommended)

```
Foreign OTel span
   │
   ▼
1. Dialect detector — sniff signature attributes, return dialect name
   │
   ▼
2. Dispatch to that dialect's `SpanAdapter.from_otel(span) -> ragpill.Span`
   │
   ▼  (if the dialect adapter declines or no dialect detected)
3. Universal best-effort extractor (the Option-B strategy registry,
   used as fallback only)
   │
   ▼
ragpill.trace.Span
```

- **Detection signatures** (cheap; just attribute-key membership):
  - `mlflow.spanType` present → `mlflow`
  - `openinference.span.kind` present → `openinference`
  - `langfuse.observation.type` present → `langfuse`
  - `traceloop.span.kind` or `traceloop.entity.name` → `openllmetry`
  - `logfire.span_type` present (and at least one `gen_ai.*` event)  → `logfire`
  - any `gen_ai.*` event but none of the above → `gen_ai`
  - none of the above → `unknown` (uses fallback)
- **Plus:** Each dialect adapter is a single self-contained module —
  easy to read, test, version, and ship behind an extra. The detector is
  one short function with explicit precedence. Pass-through attributes
  preserve everything ragpill didn't extract, so users aren't blocked
  when an adapter is incomplete.
- **Minus:** Mixed-dialect *single* spans (rare — usually a span comes
  from one instrumentation library) would force an arbitrary choice;
  the precedence list above documents it.
- **Verdict:** Recommended. Works the way users mentally model the
  problem ("I'm using Phoenix, parse Phoenix").

### 5.4 Option D — Single hand-written reader (no plug-ins)

A single "do-everything" function that knows every dialect by hard-coded
branching, without a plug-in surface. Simplest to ship; impossible to
extend without modifying ragpill core; community-contributed support for
new dialects requires PRs to ragpill itself. We rejected this once we
saw how many dialects exist (§3) and how much each one's namespace
sprawls.

---

## 6. Recommended architecture (Option C, fleshed out)

### 6.1 Module layout

```
src/ragpill/trace/
  __init__.py            # Public API: parse_otel(...), Trace, Span, SpanKind, Message, Document, Usage
  model.py               # the dataclasses from §4
  detect.py              # detect_dialect(span: OtelSpan) -> str
  registry.py            # SpanAdapter registry + entry-point loader
  fallback.py            # universal best-effort extractor (Option B style)
  loader.py              # parse_otel_json, parse_otlp_proto, from_mlflow_trace,
                         #   from_phoenix_export, from_langfuse_export
  adapters/
    __init__.py
    _base.py             # SpanAdapter ABC
    mlflow_adapter.py    # always-on (current capture format)
    gen_ai.py            # always-on (no extra dependency — pure attribute reading)
    openinference.py     # extra: ragpill[openinference]
    openllmetry.py       # extra: ragpill[openllmetry]
    langfuse.py          # extra: ragpill[langfuse]
    logfire.py           # extra: ragpill[logfire]
```

### 6.2 The `SpanAdapter` interface

```python
# src/ragpill/trace/adapters/_base.py

from abc import ABC, abstractmethod
from typing import Any
from ragpill.trace.model import Span as RagpillSpan

class SpanAdapter(ABC):
    """Convert a single OTel ReadableSpan-shaped dict into a ragpill Span."""

    name: str  # "mlflow" | "openinference" | "openllmetry" | …

    @classmethod
    @abstractmethod
    def signature_attributes(cls) -> tuple[str, ...]:
        """Attribute keys whose presence signals this dialect.

        Used by `detect_dialect` to choose an adapter without parsing
        the whole span. Order does not matter inside one adapter; ties
        between adapters are broken by registry precedence.
        """

    @classmethod
    @abstractmethod
    def from_otel(cls, span: dict[str, Any]) -> RagpillSpan:
        """Parse a single OTLP/JSON span dict into a ragpill Span.

        Implementations may raise `AdapterDeclined` to signal "the
        signature matched but the actual span is malformed for this
        dialect" — the loader then falls back to the universal extractor.
        """
```

The adapter receives a normalised OTLP-JSON span dict with fields:
`trace_id`, `span_id`, `parent_span_id`, `name`, `kind` (OTel SpanKind,
not LLM kind), `start_time_unix_nano`, `end_time_unix_nano`,
`attributes` (dict), `events` (list of `{name, time_unix_nano,
attributes}`), `status` (`{code, message}`). The loader (§6.4) is
responsible for normalising ALL input formats (MLflow Trace, OTLP
protobuf, Phoenix HTTP export, Langfuse export, raw OTLP/JSON) into
this shape before handing to an adapter. Adapters never deal with input
format conversion; only with attribute interpretation.

### 6.3 Registration & dependency extras

Adapters self-register via Python entry points so a third party can
ship one without forking ragpill:

```toml
# pyproject.toml of ragpill itself
[project.entry-points."ragpill.trace_adapters"]
mlflow       = "ragpill.trace.adapters.mlflow_adapter:MLflowAdapter"
gen_ai       = "ragpill.trace.adapters.gen_ai:GenAIAdapter"
openinference = "ragpill.trace.adapters.openinference:OpenInferenceAdapter"
openllmetry  = "ragpill.trace.adapters.openllmetry:OpenLLMetryAdapter"
langfuse     = "ragpill.trace.adapters.langfuse:LangfuseAdapter"
logfire      = "ragpill.trace.adapters.logfire:LogfireAdapter"

[project.optional-dependencies]
openinference = ["openinference-semantic-conventions>=0.1.10"]
openllmetry   = ["opentelemetry-semantic-conventions-ai>=0.4"]
langfuse      = ["langfuse>=3.0,<4"]
logfire       = []   # no runtime dep — just the adapter
```

`mlflow` and `gen_ai` ship with the base install — `mlflow` because
ragpill already depends on it, `gen_ai` because it's a pure-Python
attribute-reading adapter with no extra dependency. Everything else is
opt-in via an extra. Importing
`ragpill.trace.adapters.openinference` without the extra raises
`ModuleNotFoundError("install ragpill[openinference]")` with a hint.

A third-party package may also register an adapter:

```toml
# someones-package/pyproject.toml
[project.entry-points."ragpill.trace_adapters"]
mycorp = "mycorp.ragpill_otel:MyCorpAdapter"
```

### 6.4 The loader: input formats to OTLP-JSON span dicts

```python
# src/ragpill/trace/loader.py

def parse_otel(
    source: Any,
    *,
    dialect: str = "auto",   # "auto" | adapter name
    fallback_dialect: str = "gen_ai",
) -> Trace:
    """Parse any supported OTel-bearing input into a ragpill Trace.

    `source` may be:
      - bytes (OTLP-protobuf payload) — auto-detect via magic bytes
      - str / Path (a .json or .otlp file)
      - dict (already-parsed OTLP-JSON ResourceSpans payload)
      - mlflow.entities.Trace
      - langfuse.api.Trace (when ragpill[langfuse] is installed)
      - phoenix.trace.Trace (when ragpill[phoenix] is installed,
        future)

    `dialect="auto"` runs the dialect detector per-span and dispatches
    accordingly. An explicit dialect name forces every span through
    that adapter (escape hatch for "I know my data; don't auto-detect").
    `fallback_dialect` controls which adapter the universal fallback
    delegates to when nothing matches — defaults to `gen_ai` because
    its attribute set is the smallest.
    """
```

Selection logic for `dialect="auto"`:

```python
def _select_adapter(span_dict, registry):
    for adapter in registry.in_priority_order():
        if all(attr in span_dict["attributes"] for attr in adapter.signature_attributes()):
            return adapter
    return None  # fall through to universal extractor
```

Priority order (highest first), justified in §3:
1. `mlflow` (most specific — wraps everything else)
2. `langfuse`
3. `openinference`
4. `openllmetry`
5. `logfire`
6. `gen_ai`
7. universal fallback

### 6.5 Settings integration

A new `RagpillTraceSettings` (pydantic-settings, env-prefix
`RAGPILL_TRACE_`):

```python
class RagpillTraceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGPILL_TRACE_")
    dialect: str = "auto"
    fallback_dialect: str = "gen_ai"
    # advanced: explicit adapter precedence override
    adapter_precedence: list[str] | None = None
```

Wired through `MLFlowSettings` aggregation so existing call sites pick
it up without breaking changes:

```python
@dataclass
class Settings:
    mlflow: MLFlowSettings
    trace: RagpillTraceSettings
    # langfuse: LangfuseSettings (when present)
```

`execute_dataset` and the report module read `settings.trace.dialect`
when materialising `ragpill.trace.Trace` from whatever the capture
backend produced.

### 6.6 Extension recipe (what "adding a new dialect" looks like)

Concretely, to add a hypothetical `mycorp` dialect:

1. Add `pip install opentelemetry-semantic-conventions-mycorp` (or
   nothing, if it's pure attribute reading) under
   `ragpill[mycorp]` in `pyproject.toml`.
2. Create `src/ragpill/trace/adapters/mycorp.py`:
   ```python
   class MyCorpAdapter(SpanAdapter):
       name = "mycorp"
       @classmethod
       def signature_attributes(cls): return ("mycorp.observation.kind",)
       @classmethod
       def from_otel(cls, span): ...
   ```
3. Register in `pyproject.toml`:
   ```toml
   [project.entry-points."ragpill.trace_adapters"]
   mycorp = "ragpill.trace.adapters.mycorp:MyCorpAdapter"
   ```
4. Add tests in `tests/trace/test_adapter_mycorp.py` with sample
   OTLP-JSON fixtures.

No other module changes. The renderer, evaluators, and JSON layer
already speak `ragpill.trace.Span` — the new adapter slots in.

---

## 7. Evaluator implications

Span-introspecting evaluators today (`RegexInSourcesEvaluator`,
`RegexInDocumentMetadataEvaluator`, `LiteralQuoteEvaluator`) walk
`mlflow.entities.Trace` and read `span.attributes` directly. After this
design lands they'd consume `ragpill.trace.Trace` and read
`span.documents`, `span.messages_out`, etc. instead. Concretely:

- `RegexInSourcesEvaluator`: was "iterate retriever spans, look at
  `mlflow.spanOutputs` JSON for documents". Becomes: "iterate spans
  where `span.kind == SpanKind.RETRIEVER`, scan `span.documents`."
- `LiteralQuoteEvaluator`: similar.
- `RegexInOutputEvaluator`: doesn't change — it consumes the task's
  return value, not span data.

This is a **breaking change** for users who have written custom
evaluators against `mlflow.entities.Span`. We mitigate via:

- A `ragpill.trace.compat` module exposing `to_mlflow_trace(trace)` for
  users who want to keep the old API on read paths.
- Migration doc + the v0.4.0 release-notes entry.

---

## 8. Testing strategy

### 8.1 Fixture corpus

`tests/trace/fixtures/` keeps one OTLP-JSON capture per dialect, taken
from a real run of a tiny RAG agent. We reuse the same agent topology
across dialects so the resulting `ragpill.trace.Trace` shapes are
*structurally identical*. Any per-dialect divergence is a bug.

```
tests/trace/fixtures/
  rag_agent_mlflow.otlp.json
  rag_agent_openinference.otlp.json
  rag_agent_openllmetry.otlp.json
  rag_agent_gen_ai.otlp.json
  rag_agent_langfuse.otlp.json
  rag_agent_logfire.otlp.json
```

### 8.2 Cross-dialect parity test

```python
@pytest.mark.parametrize("dialect", ["mlflow", "openinference", "openllmetry", "gen_ai", "langfuse", "logfire"])
def test_rag_agent_parity(dialect):
    trace = parse_otel(load_fixture(f"rag_agent_{dialect}.otlp.json"), dialect=dialect)
    assert any(s.kind == SpanKind.RETRIEVER for s in trace.spans)
    retriever = next(s for s in trace.spans if s.kind == SpanKind.RETRIEVER)
    assert len(retriever.documents) == 3
    assert all(d.content for d in retriever.documents)
    llm = next(s for s in trace.spans if s.kind == SpanKind.LLM)
    assert llm.model
    assert llm.usage.input_tokens > 0
    assert llm.messages_in[0].role == "user"
    assert "answer" in llm.messages_out[0].content.lower()
```

### 8.3 Auto-detect test

```python
def test_auto_detect_picks_correct_adapter(caplog):
    trace = parse_otel(load_fixture("rag_agent_openinference.otlp.json"), dialect="auto")
    assert all(s.dialect == "openinference" for s in trace.spans)
```

### 8.4 Mixed-dialect test

A fixture where one span is `mlflow.*`, one `openinference.*`, one
`gen_ai.*` — assert each gets the right dialect tag and the trace as a
whole loads cleanly.

### 8.5 Unknown-dialect fallback test

A fixture with only standard OTel attributes (no LLM-specific ones) —
assert the fallback extractor produces a valid (if sparse) `Trace`,
sets `dialect="unknown"`, and doesn't crash.

### 8.6 Adapter-extra import-error test

```python
def test_openinference_adapter_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openinference.semconv", None)
    with pytest.raises(ModuleNotFoundError, match=r"install ragpill\[openinference\]"):
        from ragpill.trace.adapters.openinference import OpenInferenceAdapter
```

---

## 9. Phased implementation

### Phase 1 — Internal model + MLflow adapter (no behaviour change)

1. Land `ragpill.trace.model` (Span/Trace/Message/Document/Usage/SpanKind).
2. Land `ragpill.trace.adapters.mlflow_adapter` — produces a
   `ragpill.trace.Trace` from `mlflow.entities.Trace`.
3. Add `ragpill.trace.compat.to_mlflow_trace` for back-compat.
4. **Don't** switch the renderer or evaluators yet. Tests:
   round-trip `mlflow_trace -> ragpill_trace -> mlflow_trace` for the
   existing fixtures.

### Phase 2 — Renderer + JSON layer flip

5. Switch `ragpill.report._trace.render_spans` to consume
   `ragpill.trace.Span`. Migrate `triage.py` and `exploration.py`
   accordingly.
6. Migrate evaluators: `RegexInSourcesEvaluator`,
   `RegexInDocumentMetadataEvaluator`, `LiteralQuoteEvaluator`.
7. Update `EvaluationOutput.to_json` / `from_json` to serialise
   `ragpill.trace.Trace` instead of `mlflow.entities.Trace`. Carry a
   schema-version bump and a one-shot `from_json_v1_to_v2`
   migrator for old run files.

### Phase 3 — Add gen_ai + openinference adapters

8. `gen_ai.py` and `openinference.py` adapters, with fixtures.
9. `parse_otel(...)` loader entry point.
10. `RagpillTraceSettings` wired through `execute_dataset`.

### Phase 4 — Remaining adapters (one extra each)

11. `openllmetry`, `langfuse`, `logfire` adapters under their respective
    extras. Each ships with a fixture + parity test.

### Phase 5 — Capture flexibility (separate design)

This design only addresses *consumption*. Letting the user pick which
backend produces traces during `execute_dataset` (Phoenix, OTLP file,
Langfuse-direct, …) is its own design — it interacts with the
[langfuse-integration design](langfuse-integration.md) §6.

Phases 1–3 deliver the bulk of the value. Phase 4 is purely
additive — each new dialect can land independently.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Vendor SDKs change attribute names between versions (e.g. OpenInference 0.1.x → 0.2.x renamed `llm.token_count` → `llm.token_count.prompt`). | Pin floor versions in extras; per-adapter attribute readers go through a small `versioned_attr(span, [v1_key, v2_key])` helper that tries names in order. Document each adapter's tested version range. |
| Auto-detect picks wrong adapter on mixed traces. | Detection is per-span, not per-trace. The dialect is recorded on each `Span`, so users can see exactly which adapter handled which span. Document the precedence list in §6.4 explicitly. |
| Universal fallback silently produces near-empty Spans for unsupported dialects. | Emit a *single* `warnings.warn(...)` per parse with the dialect signature it didn't recognise and a pointer to the docs. Tests assert the warning. |
| Round-trip dataloss on serialisation (vendor attributes the adapter didn't extract). | The pass-through `attributes` dict on `Span` carries everything we didn't lift to first-class fields. JSON round-trip preserves it. |
| Evaluator migration bug burden. | Phase 1 lands the data model without touching evaluators. Phase 2 migrates them with a feature flag (`RAGPILL_USE_NEW_TRACE_MODEL=1`) so the cutover can be staged. Compat shim (`to_mlflow_trace`) for users with custom evaluators. |
| Complexity creep — five adapters is a lot to maintain. | Keep adapters small: each is a single file, ~200 LOC, with one clear job. Shared utilities (`_safe_json`, `_extract_indexed_messages`, etc.) in `_base.py`. Cross-dialect parity test (§8.2) keeps drift visible. |
| OTel GenAI semconv stabilises and the universe converges. | Great outcome. We retire dialect adapters by marking them deprecated and pointing users at the `gen_ai` adapter. The data model doesn't change. |

---

## 11. Open questions

1. **Adapter selection at the *trace* level vs the *span* level.** Per-span
   gives correct results on mixed traces; per-trace is simpler and
   matches user mental model. This design picks per-span. Confirm.
2. **`ragpill.trace.Trace.dialect`** when spans disagree. Currently set
   to "the dialect of the most common span", with a tiebreak by
   precedence list. Alternative: `"mixed"`. Cosmetic question.
3. **OTLP-protobuf parsing.** Do we depend on `opentelemetry-proto`
   directly, or is OTLP-JSON-only sufficient v1? OTLP-JSON covers all
   currently-encountered fixtures (MLflow JSON, Phoenix exports,
   Langfuse exports). OTLP-proto is needed for live wire ingestion,
   which §2 declares out of scope. Recommend: OTLP-JSON only in v1;
   add `ragpill[otlp-proto]` extra later if a user demand surfaces.
4. **Should the LangChain/LangSmith adapter ship in v1?** §3.7 argued
   no. Confirm.
5. **Cost field.** `Usage.cost_usd` is best-effort and only Langfuse
   emits it natively. Other dialects need a per-model lookup table to
   compute it. Out of scope of this design; tracked as a follow-up
   ("attach cost during evaluate" rather than during ingest).

---

## 12. Out of scope

- Live OTLP receiver (acting as a collector ourselves).
- Pushing traces back to *another* backend in a different dialect
  (re-emission). The langfuse-integration design covers a partial form
  of this; doing it cross-dialect is a separate problem.
- Replacing `pydantic_ai`'s tracing with a ragpill-native emitter.
- Schema enforcement on user-supplied `attributes` pass-through. We
  carry through whatever the dialect emitted; users handle it.
