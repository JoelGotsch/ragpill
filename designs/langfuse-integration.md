# Design: Langfuse Integration

**Status:** Draft
**Date:** 2026-04-25
**Context:** ragpill currently treats MLflow as the canonical backend for both
trace capture and assessment/metric persistence. Langfuse is the most
feature-rich open-source LLM observability platform and a natural alternative
backend, especially for teams who want native LLM affordances (token/cost
tracking, sessions, prompt management, online evaluation). This document
maps Langfuse's surface against ragpill's, defines an integration strategy,
and identifies new capabilities the integration unlocks.

---

## 1. Goals

1. Make Langfuse a **first-class backend alongside MLflow**, not a
   replacement. ragpill's three-layer architecture (execute / evaluate /
   upload) is provider-neutral by design — adding a backend should require
   only new modules, not changes to existing layer interfaces.
2. Preserve the **offline-friendly pattern** (`DatasetRunOutput.to_json` →
   evaluate → upload-later). Langfuse must not require a live server during
   `execute_dataset` or `evaluate_results`.
3. **Surface Langfuse-only features** (session UI grouping, prompt
   management, server-side evaluations) through new optional fields/APIs.
   Don't pretend they exist when MLflow is the backend. Multi-turn cases
   themselves are not Langfuse-specific — they're expressed via the
   generic `InputsT`/`OutputT` parameters of `Case` (see §9.1) and work
   on either backend; only the threaded UI is Langfuse-specific.
4. Allow **dual-shipping** in the same evaluation run: traces → Langfuse,
   secondary copy → MLflow, or vice versa. Some teams already run both for
   different audiences.

## 2. Non-Goals

- Hiding backend differences behind a unified abstraction. The lowest common
  denominator is too thin to be useful; clients should pick a backend
  knowingly.
- Re-implementing Langfuse's UI. We rely on `langfuse.com` / self-hosted
  instance for visualization.
- Replacing MLflow's pydantic_ai autolog. We bridge OTEL where Langfuse
  needs it.

---

## 3. Langfuse Primer

### 3.1 Core entities

| Langfuse entity | Description |
|-----------------|-------------|
| **Trace** | Top-level container. Has `name`, `userId`, `sessionId`, `metadata`, `tags`, `version`, `input`, `output`, `release`. |
| **Observation** | Anything inside a trace. Three types: `SPAN` (any work unit), `GENERATION` (LLM call with `model`, `modelParameters`, `input`, `output`, `usage`, `cost`), `EVENT` (zero-duration marker). |
| **Score** | Pass/fail or numeric verdict attached to a `Trace`, `Observation`, `Session`, or `DatasetRunItem`. Has `name`, `value`, `comment`, `dataType` (`NUMERIC` \| `BOOLEAN` \| `CATEGORICAL`), `source` (`API`, `ANNOTATION`, `EVAL`, `REVIEW`). |
| **Dataset** | Server-side test set. Versioned. |
| **DatasetItem** | One case in a dataset: `input`, `expectedOutput`, `metadata`, `status` (`ACTIVE`/`ARCHIVED`). |
| **DatasetRun** | A named execution of a dataset (typically one per CI run). |
| **DatasetRunItem** | Links a `DatasetItem` to the `Trace` produced when running it. |
| **Session** | Multi-turn grouping for conversational agents. |
| **Prompt** | Versioned prompt template stored in Langfuse and fetched at runtime. |

### 3.2 Eval-relevant features

- **Online evaluations**: server-side LLM-as-judge runs against newly
  ingested traces using user-defined templates. Scores attach automatically.
- **Annotation queues**: human reviewers add manual scores via the UI.
- **Comparison view**: side-by-side runs of the same dataset.
- **Cost / latency analytics**: aggregated from `GENERATION` observations.
- **OTEL ingestion**: a Langfuse instance can ingest OpenTelemetry directly,
  so any OTEL-emitting client (including `pydantic_ai`) can target it.

### 3.3 SDK shape

`langfuse` Python SDK (v3.x):

```python
from langfuse import Langfuse, observe

lf = Langfuse(public_key=..., secret_key=..., host=...)
trace = lf.trace(name="case-1", input=question, metadata=...)
gen = trace.generation(name="llm-call", model=..., input=..., output=...)
trace.score(name="LLMJudge", value=1.0, comment="passed", data_type="NUMERIC")
lf.flush()
```

The `observe` decorator + OTEL bridge are the typical route for non-explicit
instrumentation. For pydantic-ai we'll prefer the OTEL bridge since the
agent already emits OTEL spans.

---

## 4. Functionality Coverage

### 4.1 What Langfuse covers natively

| ragpill capability | Langfuse equivalent | Coverage |
|-------------------|---------------------|---------|
| Trace capture during task execution (`execute_dataset` traces) | Trace + Observations via OTEL ingestion | **Full** — pydantic_ai's OTEL output maps cleanly to Langfuse's trace model. |
| Per-trace assessments (`upload_to_mlflow`'s `log_assessment` calls) | `trace.score(...)` | **Full** — one-to-one, with richer typing (`NUMERIC` / `BOOLEAN` / `CATEGORICAL`). |
| Run-level metadata (`mlflow.log_params` for `model_params`) | DatasetRun `metadata` + Trace `version`/`tags` | **Full** — Langfuse splits between run-level metadata and trace tags; we map both. |
| Aggregate metrics (`overall_accuracy`, per-tag accuracy via `log_metric`) | Computed by Langfuse from the scores; their dashboards show pass-rate by tag automatically | **Full + better** — we don't need to upload computed metrics; Langfuse derives them server-side. |
| Test case container (`Dataset` + `Case` + `expected_output`) | Dataset + DatasetItem | **Full** — bidirectional sync possible. |
| Dataset versioning | Dataset versions in Langfuse | **Langfuse-only** — ragpill currently has no notion of versioned datasets. |
| Multi-turn conversation grouping | Sessions | **Different model.** ragpill expresses multi-turn cases via the generic `InputsT`/`OutputT` parameters of `Case` (see §9.1). Langfuse sessions are a UI grouping; the upload layer can opt-in to splitting one ragpill case into N Langfuse traces sharing a `session_id`. |
| Prompt management | Prompts (versioned, fetched at runtime) | **Langfuse-only** — ragpill carries rubrics inline; users can opt into Langfuse-managed rubrics. |

### 4.2 What ragpill covers that Langfuse doesn't

| ragpill capability | Langfuse status |
|-------------------|-----------------|
| Layered architecture: `execute_dataset` → `evaluate_results` → `upload_to_X` with offline JSON serialization | Langfuse is online-first; their `langfuse.dataset.run()` workflow expects a live server. ragpill's "execute offline → evaluate → upload later" flow has no direct equivalent. |
| `BaseEvaluator` class hierarchy with `expected` polarity logic, `evaluation_name`, `tags`, `attributes`, `is_global` | Langfuse evaluators run server-side via templates; client-side custom evaluators are a thin scoring API. The `expected=False` (negative-test) polarity has no direct concept. |
| `RegexInSourcesEvaluator` / `RegexInDocumentMetadataEvaluator` / `LiteralQuoteEvaluator` — span-data-aware evaluators inspecting retrieved documents | These walk the trace tree client-side. Langfuse can do server-side LLM judgments but not arbitrary trace-shape introspection. |
| Per-case `repeat` + `threshold` aggregation with `pass_rate ≥ threshold` verdict | Langfuse stores per-trace scores; aggregating across N traces of the same case requires custom dashboards or client-side rollup. |
| CSV adapter (`load_testset`) | Langfuse has CSV import in the UI, but the column-to-evaluator-class mapping is custom to ragpill. |
| Combined LLM-judge and code-evaluator pipeline run in one pass | Langfuse splits manual/EVAL/API score sources; doing both in one client-side function is a ragpill pattern. |

### 4.3 Where the overlap is messy

| Concern | Detail |
|---------|--------|
| Trace-id model | MLflow: `request_id` per trace, `span_id` per span. Langfuse: `trace.id` per trace, `observation.id` per observation. The id strings are different — `DatasetRunOutput.task_runs[i].run_span_id` is currently an MLflow span id. Need a backend-agnostic identifier (or carry both). |
| Score `value` typing | MLflow `Feedback` accepts arbitrary values. Langfuse score has explicit `dataType`. We need to map `EvaluationResult.value` (bool/int/float/str) to one of `BOOLEAN`/`NUMERIC`/`CATEGORICAL` per upload. |
| LLMJudge sub-traces | Today we delete the LLMJudge's own MLflow traces post-upload (they clutter the UI). Langfuse traces are similar — we'll need an equivalent cleanup or a `tag="ragpill:judge"` filter so the UI hides them. |
| Run identifier | MLflow's run id ↔ Langfuse `DatasetRun.name`. They are not interchangeable; we need both fields on `DatasetRunOutput`. |
| `input_key` (per-run hash used by ragpill) | Has no Langfuse counterpart. Stays in the metadata. |

---

## 5. Integration Strategy

### 5.1 Architectural position

```
                    ┌──────────────────────────┐
                    │  execute_dataset(...)    │
                    │  — backend-agnostic core │
                    └───────────┬──────────────┘
                                │
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
  MLflow tracing       Langfuse tracing            local-temp tracing
  (existing)           (new, via OTEL)             (existing, no upload)
                                │
                    ┌───────────┴──────────────┐
                    │  evaluate_results(...)   │
                    │  — pure, no backend      │
                    └───────────┬──────────────┘
                                │
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
  upload_to_mlflow      upload_to_langfuse         (no upload)
  (existing)            (new)
```

The execute and evaluate layers stay backend-neutral. Only **tracing
backend selection** in `execute_dataset` and the **upload module** are
backend-specific.

### 5.2 Module boundaries

```
src/ragpill/langfuse/
  __init__.py        # public exports (LangfuseSettings, upload_to_langfuse)
  _client.py         # Langfuse SDK lifecycle (init, flush, teardown)
  _tracing.py        # OTEL bridge wiring: pydantic_ai → Langfuse
  upload.py          # upload_to_langfuse(eval_output, settings, ...)
  dataset_sync.py    # sync_dataset_to_langfuse / Dataset.from_langfuse
  settings.py        # LangfuseSettings (BaseSettings, env-prefixed)
```

Optional extra: `ragpill[langfuse]` pulling `langfuse>=3.0` and
`opentelemetry-exporter-otlp-proto-http`. Importing
`ragpill.langfuse` without the extra raises a clear hint.

### 5.3 Settings

A new pydantic-settings model parallel to `MLFlowSettings`:

```python
class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGPILL_LANGFUSE_")

    public_key: SecretStr
    secret_key: SecretStr
    host: str = "https://cloud.langfuse.com"
    release: str | None = None
    environment: str | None = None  # production / staging / dev
    flush_at: int = 15  # batch flush threshold
    flush_interval: float = 0.5
```

### 5.4 Coexistence with MLflow

We deliberately do **not** force a single backend per call. Three
configurations are explicitly supported:

1. **MLflow only** — current behavior; nothing changes for existing users.
2. **Langfuse only** — `mlflow_tracking_uri=None` and a `langfuse_client`
   passed to `execute_dataset`.
3. **Dual** — both. OTEL spans tee to both backends. `upload_to_mlflow`
   and `upload_to_langfuse` can both be called from the same evaluation
   pipeline.

The shared piece is OTEL: pydantic_ai emits one OTEL stream and we attach
one or both ingesters.

---

## 6. Tracing Layer

### 6.1 OTEL bridge

pydantic_ai already emits OpenTelemetry. MLflow's `mlflow.pydantic_ai.autolog()`
intercepts via OTEL too. Langfuse offers two ingestion paths:

- **Native OTEL endpoint** at `<host>/api/public/otel/v1/traces`. Configure
  the standard OTEL exporter to point there with appropriate auth headers.
- **Langfuse OTEL processor** (`langfuse.otel.LangfuseSpanProcessor`),
  shipped in v3+. Plug it into the `TracerProvider` directly.

We use the **processor approach** because it lets us share a single
`TracerProvider` between MLflow's autolog and Langfuse — both
processors receive the same spans without forwarding HTTP traffic twice.

### 6.2 New `execute_dataset` signature

Additive, no breaking change:

```python
async def execute_dataset(
    testset: Dataset[...],
    *,
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    settings: MLFlowSettings | None = None,
    mlflow_tracking_uri: str | None = None,
    langfuse_client: "Langfuse | None" = None,   # NEW
    capture_traces: bool = True,
) -> DatasetRunOutput:
    ...
```

When `langfuse_client` is non-None:

1. We register a `LangfuseSpanProcessor` on the active `TracerProvider`
   before the per-case loop.
2. After each case finishes, we call `langfuse_client.flush()` so the run's
   traces are persisted before we collect their ids for `DatasetRunOutput`.
3. After the loop, we deregister the processor and do a final flush.
4. We populate two new fields on `DatasetRunOutput`:
   - `langfuse_dataset_run_id: str` — the `DatasetRun.id` if the dataset
     was synced to Langfuse beforehand (see §8).
   - `langfuse_trace_ids: list[str]` — per-case Langfuse trace ids,
     allowing later score upload by id.

### 6.3 `DatasetRunOutput` extension

```python
@dataclass
class DatasetRunOutput:
    cases: list[CaseRunOutput] = field(default_factory=list)
    tracking_uri: str = ""
    mlflow_run_id: str = ""
    mlflow_experiment_id: str = ""
    # NEW
    langfuse_run_name: str = ""
    langfuse_host: str = ""
    langfuse_dataset_run_id: str = ""
    langfuse_trace_ids: dict[str, str] = field(default_factory=dict)
    # ^ keyed by CaseRunOutput.base_input_key
```

`to_json` / `from_json` extend in lockstep. JSON round-trip is preserved.

### 6.4 `CaseRunOutput` extension

A per-case Langfuse trace id is needed when uploading scores. Add:

```python
@dataclass
class CaseRunOutput:
    ...
    langfuse_trace_id: str = ""
```

---

## 7. Upload Layer

### 7.1 New module: `ragpill.langfuse.upload`

```python
def upload_to_langfuse(
    evaluation: EvaluationOutput,
    langfuse_client: Langfuse,
    *,
    upload_traces: bool = False,
    score_namespace: str = "ragpill",
    redact_attributes: list[str] | None = None,
) -> None:
    """Persist an EvaluationOutput's scores + metadata to Langfuse.

    For each (case, run, evaluator) triple, attach a Score to the
    corresponding Langfuse trace. If `upload_traces=True` and the
    EvaluationOutput contains an offline DatasetRunOutput, also
    materialize new traces via `langfuse_client.trace(...)` from the
    captured span data.
    """
```

### 7.2 Score mapping rules

| `EvaluationResult.value` | Langfuse `dataType` | `value` payload |
|---|---|---|
| `bool` | `BOOLEAN` | `1.0` if True else `0.0` (bool-ish) |
| `int` / `float` | `NUMERIC` | the value |
| `str` | `CATEGORICAL` | the value |

Names use the `score_namespace` prefix (default `ragpill`) to avoid
colliding with user-defined scores: e.g. `ragpill.LLMJudge`,
`ragpill.RegexInOutputEvaluator`. The original evaluator-and-run name
(`run-0_LLMJudge`) goes into the `comment` field along with the rationale.

### 7.3 Per-case aggregate scores

When `repeat > 1`, mirror the existing MLflow `agg_*` assessment as a
Langfuse score with name `ragpill.agg.<EvaluatorName>` and `dataType =
BOOLEAN`. Comment carries `"3/5 runs passed (threshold=0.6)"`. Source =
`API`.

### 7.4 Tag + attribute mapping

Langfuse supports `tags` (list of strings) and `metadata` (free dict) on
traces. Both are populated:

- `tags`: ragpill case `metadata.tags` (already a set of strings).
- `metadata`: `metadata.attributes` plus a `ragpill_*` namespace with
  `pass_rate`, `threshold`, `case_id`.

### 7.5 LLM-judge trace cleanup

The existing `_delete_llm_judge_traces` step strips the LLMJudge sub-traces
from MLflow before they show up in the UI. Langfuse equivalent:

- Tag those traces `ragpill:judge_internal`.
- Don't delete (Langfuse traces aren't easily delete-able programmatically
  anyway). Instead, the `tag` filter lets users hide them.

### 7.6 Upload-traces-as-artifact mode

Symmetric with `upload_to_mlflow(..., upload_traces=True)`:

- When the offline `DatasetRunOutput` was captured locally and never seen
  by Langfuse, we want to materialize traces on the server now.
- Iterate over every `CaseRunOutput`, walk its captured `Trace.data.spans`,
  and rebuild via the Langfuse SDK:
  - Top-level: `lf.trace(name=case.case_name, input=case.inputs, ...)`.
  - Per span: classify by `span_type` → `lf.span(...)` or
    `lf.generation(...)`. Preserve nesting via `parentObservationId`.
- Limitation: timing precision is exact (we have wall-clock); model
  parameters and token counts are best-effort (extracted from span
  attributes when present, otherwise omitted).

This re-trace path is the most fragile piece of the integration — we
flag it as **experimental** in the docstring and gate it on tests.

---

## 8. Dataset Sync

Two directions:

### 8.1 Push: `sync_dataset_to_langfuse`

```python
def sync_dataset_to_langfuse(
    dataset: Dataset[Any, Any, CaseMetadataT],
    langfuse_client: Langfuse,
    *,
    dataset_name: str,
    description: str | None = None,
) -> None:
    """Idempotently create/update a Langfuse Dataset matching a ragpill Dataset.

    Each ragpill Case becomes a Langfuse DatasetItem. The match key is the
    case's `base_input_key` (md5 of inputs), stored as `metadata.ragpill_id`
    on the DatasetItem so re-syncs update rather than duplicate.
    """
```

Why this matters: once a Langfuse Dataset exists, runs can be associated
with `DatasetRunItems`, unlocking Langfuse's run-comparison UI.

### 8.2 Pull: `Dataset.from_langfuse`

```python
@classmethod
def from_langfuse(
    cls,
    langfuse_client: Langfuse,
    dataset_name: str,
    evaluators: list[BaseEvaluator] | None = None,
) -> Dataset[Any, Any, TestCaseMetadata]:
    """Build a ragpill Dataset from an existing Langfuse Dataset.

    Useful when test cases live in Langfuse (curated by the team via the
    UI) and the engineer wants to run ragpill evaluators against them.
    Evaluators are passed in here because they are code, not data.
    """
```

Round-trip discipline: `sync_dataset_to_langfuse(ds, …)` followed by
`Dataset.from_langfuse(…)` must yield a dataset with the same case inputs,
expected outputs, and `base_input_key`s. Tested explicitly.

### 8.3 Linking a run to a synced dataset

When `langfuse_dataset_run_id` is set on `DatasetRunOutput`, the Langfuse
upload also creates `DatasetRunItem` rows linking each case's
`base_input_key` → trace id → dataset item. This is what populates the
Langfuse run-comparison UI.

---

## 9. Features ragpill Gains via Langfuse

Listed roughly in order of value/leverage:

### 9.1 Sessions (multi-turn agents)

Multi-turn conversations are **not** modeled with new ragpill types
(`SessionCase`, `Turn`, `SessionEvaluator`). Instead, the existing generic
`Case[InputsT, OutputT, MetadataT]` is sufficient: choose `InputsT` =
`Conversation` (a list of turns) and `OutputT` = `ConversationOutput` (a
list of per-turn responses + final). Helpers ship in a new
`ragpill.multi_turn` module:

```python
from ragpill.multi_turn import (
    Conversation,            # @dataclass: turns: list[str], metadata: dict
    ConversationOutput,      # @dataclass: turn_outputs: list[str], final_output: str
    stateless_multi_turn,    # wraps fn(turn, history) → conversation task
    stateful_multi_turn_factory,  # wraps stateful agent_factory → conversation task_factory
)
from ragpill.evaluators.conversation import (
    ContextRetentionJudge,   # LLM judge over the full transcript
    TurnCountEvaluator,      # min/max turn count
)
```

Why this works:
- **Trace shape is already correct.** A multi-turn task that calls the
  agent N times produces N nested spans inside the single case-level
  parent span. Span-introspecting evaluators
  (`RegexInSourcesEvaluator`, etc.) walk the whole subtree and
  naturally cover all turns.
- **Evaluators see typed transcripts.** `ctx.inputs: Conversation` and
  `ctx.output: ConversationOutput` give evaluators direct, indexed
  access to the turn sequence — no new context fields needed.
- **No per-turn-position granularity in the report.** One verdict per
  case per evaluator (the existing model), which matches the 90% case
  ("did the conversation as a whole succeed on dimension X").

**Langfuse-specific upload mode.** When `case.inputs` is a
`Conversation`, `upload_to_langfuse` can optionally split the single
case trace into N Langfuse traces sharing a deterministic
`session_id` (default: `f"ragpill-{case.base_input_key}"`). This
unlocks Langfuse's threaded conversation UI. When the flag is off (or
when the input isn't a `Conversation`), the case still appears as one
trace with N nested spans — fully browsable, just not threaded. MLflow
ignores `session_id` since it has no session UI; we don't emulate it.

```python
upload_to_langfuse(
    eval_output,
    langfuse_client,
    split_conversations_into_sessions=True,   # opt-in
)
```

### 9.2 Server-side LLM-as-judge

Langfuse's "evaluations" feature runs LLM judges server-side against any
trace matching a filter, on a schedule. Pattern: ragpill uploads the run,
Langfuse re-judges it overnight against an updated rubric. Implementation:
none on our side beyond a how-to doc explaining how to point the
server-side evaluator at `tags: ragpill`.

### 9.3 Annotation queues

Domain experts can label traces in Langfuse's UI (binary or rubric-graded).
Their scores show up in the same `score` table. ragpill needs no code
change; the `EvaluationOutput.summary` becomes one view of the truth, and
the Langfuse UI is the other.

### 9.4 Cost / token analytics

Langfuse `GENERATION` observations carry `usage` and `cost`. pydantic_ai
emits these in OTEL today. By routing OTEL → Langfuse we get free cost
per case, per run, per tag — something MLflow's autolog doesn't surface.

### 9.5 Prompt management

LLMJudge currently embeds the rubric as a Python field. We add an opt-in:

```python
LLMJudge.from_langfuse_prompt(client, name="rag-faithfulness", version=3)
```

The prompt body comes from Langfuse, so updates ship without redeploys.
Versioning lives in Langfuse, traceable to a specific run.

### 9.6 Dataset versioning

Langfuse versions every dataset edit. We expose
`Dataset.from_langfuse(..., version=N)` so an evaluation pinned to a
particular dataset version is reproducible without committing the testset
in the repo.

### 9.7 Run comparison UI

Langfuse natively diffs two `DatasetRun`s on the same dataset. ragpill's
own `EvaluationOutput.summary` only shows a single run; users currently
diff in pandas. Just by uploading we get the comparison view.

---

## 10. Use Cases (with code sketches)

### 10.1 CI: same evaluation, dual-shipped to MLflow + Langfuse

```python
from ragpill import execute_dataset, evaluate_results, upload_to_mlflow
from ragpill.langfuse import upload_to_langfuse, LangfuseSettings
from langfuse import Langfuse

mlflow_settings = MLFlowSettings()
lf_settings = LangfuseSettings()
lf = Langfuse(public_key=lf_settings.public_key.get_secret_value(),
              secret_key=lf_settings.secret_key.get_secret_value(),
              host=lf_settings.host)

run_output = await execute_dataset(
    testset, task=my_task,
    settings=mlflow_settings,
    mlflow_tracking_uri=mlflow_settings.ragpill_tracking_uri,
    langfuse_client=lf,
)
eval_output = await evaluate_results(run_output, testset)
upload_to_mlflow(eval_output, mlflow_settings)
upload_to_langfuse(eval_output, lf)
lf.flush()
```

### 10.2 Disconnected execution + Langfuse upload later

```python
# Offline: capture run JSON
run_output = await execute_dataset(testset, task=my_task)
with open("run.json", "w") as f:
    f.write(run_output.to_json())

# Later, with Langfuse access:
from ragpill.execution import DatasetRunOutput
run_output = DatasetRunOutput.from_json(open("run.json").read())
eval_output = await evaluate_results(run_output, testset)
upload_to_langfuse(eval_output, lf, upload_traces=True)
```

### 10.3 Curated dataset in Langfuse, code evaluators

```python
testset = Dataset.from_langfuse(lf, "rag-quality-suite",
                                evaluators=[my_global_evaluator])
result = await execute_dataset(testset, task=my_task, langfuse_client=lf)
```

### 10.4 LLMJudge with a server-versioned rubric

```python
judge = LLMJudge.from_langfuse_prompt(lf, name="rag-faithfulness", version=3)
testset = Dataset(cases=[Case(inputs=q, evaluators=[judge]) for q in questions])
```

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Langfuse SDK churn (v2→v3 rewrite was non-trivial) | Pin a floor version (`langfuse>=3.0,<4`). Treat the integration as **beta** until v3 has been stable for ≥6 months. |
| OTEL ingestion duplication when both MLflow autolog and Langfuse processor are attached | Use a single `TracerProvider` with both processors attached. Document the gotcha and add a smoke test that asserts both backends see N traces from one execution. |
| Scoring data-type mismatches (a ragpill evaluator returns an int that should be `NUMERIC`, not `BOOLEAN`) | Centralize the mapping in `ragpill.langfuse.upload._score_data_type(value)` with explicit per-type rules and tests. |
| Re-tracing offline runs into Langfuse loses fidelity (token counts/costs unavailable) | Document explicitly that `upload_traces=True` is best-effort. Tag re-traced runs with `ragpill:re_traced=true` so they're filterable. |
| Secrets leaking through trace attributes (API keys in agent input) | `redact_attributes` param on `upload_to_langfuse` defaults to `["api[_-]?key", "authorization", "secret"]` regex set. Same as the MCP design's redaction. |
| Langfuse server-side eval running concurrently with our score upload causes duplicate scores under different names | Use `score_namespace` prefix; document that user-defined Langfuse evals should use a different namespace. |
| Cost: high-volume traces with all our metadata blow up Langfuse storage quotas | Add a `metadata_strip` flag on upload to drop large fields (full doc text in retriever spans). Off by default. |

---

## 12. Phased Implementation

### Phase 1 — Tracing only (additive, low risk)

1. `ragpill/langfuse/_client.py`, `_tracing.py`, `settings.py`.
2. New `langfuse_client` arg on `execute_dataset`. Wires the OTEL processor.
3. New fields on `DatasetRunOutput` and `CaseRunOutput`. JSON round-trip.
4. Tests:
   - Unit: settings parse, client init, processor registration.
   - Integration (gated, requires Langfuse cloud creds or self-hosted in CI):
     execute → assert traces appear in Langfuse and trace ids land on the
     output.
5. Docs: `docs/guide/langfuse.md` — getting started, env vars, dual-backend
   pattern.

### Phase 2 — Scores

1. `ragpill/langfuse/upload.py` with `upload_to_langfuse(eval_output, ...)`.
2. Per-evaluator scores + per-case aggregates + tags/metadata.
3. Idempotency: if uploaded twice, scores are upserted (Langfuse's API
   supports `name+observationId+traceId` as a logical key — confirm during
   implementation).
4. Tests for `_score_data_type`, score upload, namespace handling.

### Phase 3 — Dataset sync

1. `sync_dataset_to_langfuse` and `Dataset.from_langfuse`.
2. DatasetRunItem linking on upload.
3. Round-trip test: push → pull → assert structural equality.

### Phase 4 — Optional integrations

1. `LLMJudge.from_langfuse_prompt(...)` for server-managed rubrics.
2. `split_conversations_into_sessions=True` upload flag — splits a
   `Conversation`-typed case into N Langfuse traces sharing a deterministic
   `session_id`. Depends on the `ragpill.multi_turn` helpers, which can
   land independently.
3. Doc: `docs/how-to/server-side-evaluations.md` showing the Langfuse-side
   eval template setup.

### Phase 5 — Re-tracing offline runs (experimental)

1. `upload_to_langfuse(..., upload_traces=True)` materializes traces from
   captured span data.
2. Heavy testing: feature-gated behind `experimental=True` until stable.

Phases 1 and 2 deliver the bulk of the value. Phases 3-5 are additive and
can ship over weeks/months as user demand surfaces.

---

## 13. Open Questions

These need owner decisions before implementation:

1. **Default backend.** Should `evaluate_testset_with_mlflow` get a sibling
   `evaluate_testset_with_langfuse` for symmetry? Or do we steer users
   toward calling the three layers directly when picking Langfuse?
2. **Dual-backend identifier model.** Should `DatasetRunOutput` have a
   single `traces: dict[backend_name, BackendIdentifier]` field instead of
   the current MLflow-specific + Langfuse-specific fields? Cleaner, but
   churns the existing JSON schema.
3. **OTEL ownership.** If the user has their own `TracerProvider` already
   set (e.g. for production observability), we currently overwrite/reset
   it in MLflow's autolog. Langfuse changes this risk only marginally, but
   we should document expectations and possibly add a "preserve existing
   provider" mode.
4. **Conversation traces on MLflow.** When a user runs a
   `Conversation`-typed case and the backend is MLflow, the case appears
   as one trace with N nested per-turn spans. That's fine. Confirm we
   don't need to emulate the Langfuse session UI on the MLflow side — I
   don't think we do; users who want the threaded view should pick
   Langfuse for that workload.

---

## 14. Out of Scope

- Migrating existing MLflow runs into Langfuse retroactively. Possible via
  the re-trace path but a separate one-shot tool, not part of normal flow.
- Replacing pydantic_ai's tracing with a custom emitter to reduce
  boundary cases. Too much surface for too little gain.
- Langfuse's "Playground" feature (interactive prompt iteration) — that's
  a developer workflow that lives entirely outside ragpill.
