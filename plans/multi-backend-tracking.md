# Plan: Make Tracking Backend Pluggable (MLflow / Langfuse / Arize Phoenix)

**Status:** Proposed
**Date:** 2026-05-17
**Related:**
- [designs/langfuse-integration.md](../designs/langfuse-integration.md) — already-drafted analysis of Langfuse as an alternative.
- [designs/otel-trace-ingestion.md](../designs/otel-trace-ingestion.md) — already-drafted plan to normalise span dialects (the trace-shape half of multi-backend support).
- `plans/suppress-llm-judge-traces.md` — touches the same OTel exporter plumbing.

## Context

ragpill today is tightly coupled to MLflow at every layer:

| File | Lines using `mlflow` | What for |
|---|---|---|
| `src/ragpill/execution.py` | 38 | `mlflow.pydantic_ai.autolog()`, `start_span`, `set_tracking_uri`, `set_experiment`, `start_run`, plus `DatasetRunOutput.mlflow_run_id` / `mlflow_experiment_id` fields |
| `src/ragpill/upload.py` | 30 | the entire upload layer (`log_table`, `log_metric`, `log_params`, `log_assessment`, `set_trace_tag`, `log_artifact`, `search_traces`, `MlflowClient.delete_traces`) |
| `src/ragpill/mlflow_helper.py` | 13 | bootstrap/tracking-URI helpers |
| `src/ragpill/evaluators.py` | 8 | `LLMJudge.run` opens a span; `SpanBaseEvaluator.get_trace` returns `mlflow.entities.Trace`; `_filter_trace_to_subtree` takes a `Trace` |
| `src/ragpill/report/_trace.py` | 6 | renderer reads MLflow span attribute keys (`mlflow.spanType`, `mlflow.spanInputs`, `mlflow.spanOutputs`) |
| `src/ragpill/__init__.py` | 4 | re-exports |
| `src/ragpill/settings.py` | 3 | `MLFlowSettings` |
| `src/ragpill/utils.py` | 2 | small helper |
| `src/ragpill/report/exploration.py` | 2 | trace summarisation helpers |
| `src/ragpill/types.py` | 1 | `EvaluatorContext.trace: mlflow.entities.Trace \| None` |

That's ~107 references, but they cluster into a small number of *capabilities*. The dependency itself is also heavy: `mlflow-skinny>=3.8.1` plus its transitive footprint dominates the install (SQLAlchemy, FastAPI/uvicorn via the server stub, OpenTelemetry SDK, alembic). Making it optional materially shrinks `pip install ragpill` for users who only need a different backend.

This plan tackles backend pluggability in **two phases**:

1. **Phase 1 — Decouple, prove feasibility, ship MLflow as an optional extra.** Define the small set of interfaces ragpill actually depends on, replace direct `mlflow.*` calls with interface calls, keep the existing MLflow behaviour bit-for-bit via an in-tree adapter, and gate MLflow imports behind `extras = ["mlflow"]`. The point is **no functional change**; the success criterion is `pip install ragpill[mlflow]` works identically to today and `pip install ragpill` (with no extras) imports cleanly and errors only when the user actually tries to track a run.
2. **Phase 2 — Per-backend adapters.** Concrete API analysis for Langfuse and Arize Phoenix, then implementation as `ragpill[langfuse]` and `ragpill[phoenix]` extras using the Phase-1 interfaces.

## Goals

1. **MLflow becomes optional.** `pip install ragpill` should not pull MLflow. The user opts in: `pip install ragpill[mlflow]`, `pip install ragpill[langfuse]`, etc.
2. **Adapter-shaped.** A backend is one class implementing 2–3 small protocols. Adding a new backend is a self-contained file, not a refactor.
3. **No regressions for current users.** A user on `ragpill[mlflow]` sees the same behaviour, metric names, trace shape, and CLI as today. Existing tests stay green without modification.
4. **Dual-shipping is allowed but not required.** A user can configure two backends to receive the same evaluation (one writes assessments, another captures traces). Mostly a Phase-2 concern.
5. **Don't ship a new abstraction we don't need.** The protocols are derived from the actual call sites — no speculative methods.

## Non-goals

- Building a "universal trace model" is out of scope here — [`otel-trace-ingestion.md`](../designs/otel-trace-ingestion.md) already owns that question; this plan **depends on** that work landing first (or alongside) for backends that don't produce MLflow-shaped traces.
- Removing the existing MLflow integration. It stays, just becomes an extra.
- Adding more than 3 backends. The 3 named are concrete; the design must accommodate a 4th but we don't ship one.

---

## Phase 1 — Make MLflow optional

### Step 1.1: Inventory of capabilities the codebase actually uses

Boiled down from the ~107 call sites, ragpill needs exactly these capabilities from a tracking backend:

**A. Trace capture (during `execute_dataset`)**
- Set tracking destination (URI/server/in-memory).
- Set experiment name (or equivalent grouping).
- Open a parent "run".
- Auto-instrument `pydantic-ai` so the task's LLM/tool/retriever calls land as spans.
- Open / close a manually-named span with a span type, attributes, inputs, outputs (used by `LLMJudge.run` to nest its judge call cleanly under the task span).

**B. Trace retrieval & manipulation (during `evaluate_results` and `upload_to_mlflow`)**
- Search traces in a run (by run_id, or "all of them").
- Get a trace by ID with its full span tree.
- Delete traces by request_id (used to strip judge traces).
- Filter a trace to a subtree rooted at a given span_id.

**C. Results persistence (during upload)**
- Log a metric: `(name, float)` pairs.
- Log params: `dict[str, str]`.
- Log a table artifact: a `DataFrame` + a filename.
- Log a general artifact: file path → server path.
- Log an assessment: structured `(name, value, source_type, source_id, rationale, metadata)` attached to a trace.
- Set a tag on a trace: `(trace_id, key, value)`.

**D. Run lifecycle**
- Resolve experiment id from name.
- Start a run with optional reattach-to-existing-run-id.
- End the active run.

That's **four capability buckets**, not "all of MLflow". Adapter interfaces are sized accordingly.

### Step 1.2: Proposed protocols

Four protocols in a new module `src/ragpill/backends/_base.py`:

```python
class TraceCaptureBackend(Protocol):
    """A. trace capture (only used during execute_dataset)."""
    def set_destination(self, uri: str | None, experiment_name: str) -> None: ...
    def start_run(self, run_id: str | None = None, description: str | None = None) -> RunHandle: ...
    def end_run(self) -> None: ...
    def start_span(self, name: str, span_type: SpanKind, attributes: dict[str, Any] | None = None) -> SpanHandle: ...
    def autolog_pydantic_ai(self) -> None: ...

class TraceQueryBackend(Protocol):
    """B. trace retrieval & manipulation."""
    def search_traces(self, run_id: str, max_results: int = 1000) -> list[Trace]: ...
    def get_trace(self, trace_id: str) -> Trace | None: ...
    def delete_traces(self, request_ids: list[str]) -> None: ...

class ResultsBackend(Protocol):
    """C. results persistence."""
    def log_metric(self, name: str, value: float) -> None: ...
    def log_params(self, params: dict[str, str]) -> None: ...
    def log_table(self, df: pd.DataFrame, artifact_file: str) -> None: ...
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...
    def log_assessment(self, trace_id: str, assessment: Assessment) -> None: ...
    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None: ...

class LifecycleBackend(Protocol):
    """D. run lifecycle. Often the same object as TraceCaptureBackend."""
    def resolve_experiment_id(self, experiment_name: str) -> str: ...
    def reattach_run(self, run_id: str) -> None: ...
```

`Trace`, `SpanHandle`, `Assessment`, and `SpanKind` are **vendor-neutral** dataclasses living in `ragpill.backends._types`. The actual normalisation work is the [OTel ingestion plan](../designs/otel-trace-ingestion.md); for Phase 1, we either:
- (a) reuse `mlflow.entities.Trace` directly under a type alias (`Trace = mlflow.entities.Trace`), accepting that any non-MLflow adapter must produce MLflow-shaped objects until the ingestion plan lands. Cheapest, ships first, blocks Phoenix/Langfuse adapters from being clean.
- (b) define lightweight stand-in dataclasses (`Trace`, `Span`) shaped after the renderer's actual reads, plus an `mlflow_to_internal(...)` converter on the MLflow adapter. Heavier; lets Phase 2 land cleanly.

**Recommendation:** ship Phase 1 with (a) — type alias — and label it explicitly as "Phase 1 short-cut". Phase 2's first task is to ship (b) **in tandem with** otel-trace-ingestion, so MLflow becomes one dialect among several rather than the canonical shape.

### Step 1.3: Refactor sequence

Order picked so each commit is shippable on its own and the test suite stays green between them.

1. **Add `src/ragpill/backends/` package with protocol stubs + `MLflowBackend` adapter.** The adapter just forwards to `mlflow.*`. No call sites change yet. Suite still passes.
2. **Replace direct `mlflow.*` calls in `upload.py`** with calls through a `ResultsBackend` instance. Source of the instance: a global registry / factory function `get_results_backend()` that, in Phase 1, always returns `MLflowBackend()`. This is the bulk of the refactor — `upload.py` is the most MLflow-heavy module.
3. **Replace `mlflow.*` calls in `execution.py`** with `TraceCaptureBackend` calls. Same registry pattern.
4. **Replace `mlflow.*` calls in `evaluators.py`** — `LLMJudge.run`'s span open, `SpanBaseEvaluator.get_trace`. The span context manager is the trickiest one because of the autolog-race comment at `evaluators.py:111`; the MLflow adapter must preserve that behaviour exactly. **Add a regression test** that creates a judge call and asserts there is no UNIQUE-constraint race on the SQLite backend.
5. **Update `report/_trace.py` and `report/exploration.py`** to read attribute keys via the backend's introspection rather than hard-coded `mlflow.spanType` etc. With the type-alias shortcut from Step 1.2(a), this collapses to a single helper that knows the MLflow key set; later it expands per-dialect.
6. **Gate the import.** Move all `import mlflow` lines into `src/ragpill/backends/mlflow_backend.py`. Re-exports from `__init__.py` that mention MLflow types become lazy or are dropped. Add a `try/except ImportError` at the registry boundary with a clear "install `ragpill[mlflow]` to use the MLflow backend" message.
7. **`pyproject.toml`** — move `mlflow-skinny`, `mlflow[skinny]` etc. from `dependencies` to `[project.optional-dependencies] mlflow = [...]`. Tests that import MLflow get marked `pytest.importorskip("mlflow")` so they skip gracefully when the extra isn't installed.
8. **CI matrix** — add a job `pip install ragpill` (no extras) that runs the subset of tests not gated on MLflow. Add another job for `pip install ragpill[mlflow]` that runs the full suite. Together they prove the "make optional" claim.

### Step 1.4: Feasibility check

Three places where I expect resistance:

- **`mlflow.pydantic_ai.autolog()`** is the integration that produces all task spans. Each backend's adapter must offer an equivalent — Langfuse has its own OTel-based autolog; Phoenix uses OpenInference instrumentors. The capability is there in every backend but the bootstrap call differs significantly, so `autolog_pydantic_ai()` must remain a per-adapter method (not a shared helper).
- **The `_delete_llm_judge_traces` post-hoc cleanup** is MLflow-specific (it uses `MlflowClient.delete_traces`). Langfuse can delete traces via API; Phoenix's tracing UI doesn't currently expose deletion. For Phoenix this means we need to **not generate the judge traces in the first place** — which is what `plans/suppress-llm-judge-traces.md` proposes via an OTel exporter filter. **Multi-backend support is the right time to land the exporter-filter approach** rather than continuing the post-hoc-delete one, because it works for any backend uniformly.
- **`mlflow.entities.Trace`** is currently leaked into `EvaluatorContext.trace`, the public type signature of `SpanBaseEvaluator.get_trace`, and `DatasetRunOutput.cases[i].task_runs[j].trace`. Users who write custom span-based evaluators read attributes like `span.attributes["mlflow.spanType"]` directly. Changing this is a **breaking change to public API** for anyone subclassing `SpanBaseEvaluator`. Phase 1 keeps the alias (per Step 1.2(a)) to avoid that break; Phase 2 plans the deprecation alongside otel-trace-ingestion.

**Conclusion on feasibility:** decoupling is feasible, with the caveat that one design decision (alias vs new types) determines whether Phase 2 inherits clean shapes. Recommend taking the alias short-cut and accepting that the Phoenix/Langfuse adapters will have an "adapter-side conversion" until otel-trace-ingestion lands.

### Step 1.5: Risks & mitigations

| Risk | Mitigation |
|---|---|
| Span-context-manager semantics differ per backend; `LLMJudge`'s parent-span trick relied on MLflow specifics. | Test the no-race behaviour for each adapter as a backend integration test. Mark the test as `xfail` when the adapter doesn't yet preserve the property; never ship green-but-broken. |
| Users running offline (no MLflow server, local SQLite) may not realise they're using a tracking backend at all. | Keep the "default local temp SQLite" behaviour in the MLflow adapter; if MLflow extra is installed, behaviour is unchanged. With no extras installed and no backend configured, `execute_dataset` runs but emits a warning that nothing is being captured. |
| `EvaluationOutput.to_json` / `from_json` carries trace objects. Changing trace types breaks JSON roundtrip. | Use the type-alias shortcut (Step 1.2(a)) so the on-disk schema is unchanged in Phase 1. Document the JSON-schema change explicitly in the Phase 2 release notes. |
| MLflow's `mlflow.log_table` writes `evaluation_results.json` with a specific schema MLflow's UI knows about. Other backends have no equivalent. | Adapter is allowed to no-op (or emit the same data as a CSV artifact) when the destination has no native table concept. Document the gap in the per-backend page. |

---

## Phase 2 — Per-backend adapter analysis

For each of the three backends, the analysis follows the same template; the implementation is one PR per backend after Phase 1 lands.

### Phase 2.A: MLflow adapter

(Phase 1 already produces this. Listed here for completeness.)

- Already known: MLflow uses its own tracking server (or local SQLite), OTel exporter to its own API, attribute keys `mlflow.span*`.
- Capability mapping is 1:1 since the abstraction came *from* MLflow's API.

### Phase 2.B: Langfuse adapter

See [`designs/langfuse-integration.md`](../designs/langfuse-integration.md) for the long-form analysis. Summary as it maps to the four buckets:

| Capability | Langfuse mechanism |
|---|---|
| Trace capture | `Langfuse(public_key, secret_key, host)` + their OpenTelemetry SDK shim. `langfuse.openai.OpenAI` wraps the OpenAI client; for pydantic-ai we use the OTel exporter route. |
| Auto-instrumentation of pydantic-ai | OpenTelemetry instrumentation, not a one-call `autolog`. Adapter handles the OTel wiring once at `set_destination`. |
| Trace search/retrieval | `langfuse.api.trace.list(...)` / `get(...)`. Supports filtering by user/session/tags. |
| Metrics | Langfuse has no first-class "metric" concept; mapped to **Scores** (`langfuse.score(...)`) at trace level or generation level. Aggregate metrics (`overall_accuracy`) become trace-level Scores with a synthetic trace, or are surfaced as Run-level metadata. |
| Params | Run-level metadata (`session.metadata`) or per-trace metadata. |
| Tables (DataFrame) | No native equivalent. Adapter writes it as a JSON artifact attachment on the parent trace. |
| Artifacts (generic file) | Trace attachments via the file upload API. |
| Assessments | Scores with a `comment` field for `rationale` and `dataType` for the verdict. Maps cleanly. |
| Trace tags | Native `tags: list[str]` on traces. |
| Run grouping | Sessions (UI-level threaded view). |
| Delete trace | `langfuse.api.trace.delete(...)`. Used by the judge-trace cleanup path. |

**New capabilities Langfuse unlocks** (Phase 2.B nice-to-haves, not required for adapter parity):
- Sessions UI for multi-turn conversational evaluations.
- Server-side LLM-as-judge that can re-score historical runs.
- Native cost / token analytics.
- Dataset push/pull (`sync_dataset_to_langfuse`, `Dataset.from_langfuse`).

These are out-of-scope for the adapter — they go into separate follow-on PRs (`ragpill.langfuse.sync`, `ragpill.langfuse.sessions`) so they can be reviewed independently of the adapter itself.

### Phase 2.C: Arize Phoenix adapter

Phoenix is OTel-native end-to-end. It uses **OpenInference** as its instrumentation standard, which has a different attribute namespace than MLflow (`llm.input_messages`, `llm.output_messages`, `tool.name`, etc.). Highlights:

| Capability | Phoenix mechanism |
|---|---|
| Trace capture | `phoenix.trace.LangChainInstrumentor`, `phoenix.trace.OpenAIInstrumentor`, etc. For pydantic-ai there's `openinference-instrumentation-pydantic-ai` (community-maintained — verify version at implementation time). Falls back to the OTel SDK + Phoenix-collector endpoint configuration. |
| Auto-instrumentation | Per-library instrumentor; pydantic-ai support depends on the OpenInference ecosystem. Risk: not as polished as MLflow's autolog. **Adapter implementation must verify pydantic-ai coverage before claiming parity.** |
| Trace search/retrieval | Phoenix Python client: `px.Client().get_spans_dataframe(...)`, `px.Client().get_evaluations(...)`. |
| Metrics | Phoenix has **evaluations** — scores attached to spans/traces. Maps closely to MLflow's assessment concept. |
| Params | No first-class concept. Mapped to a JSON artifact or to trace metadata. |
| Tables / artifacts | None natively. Same as Langfuse — fall back to local-file plus a span-attribute pointer. |
| Assessments | Native `phoenix.evals` package; `Evaluator.evaluate(...)` produces an `EvalResult` that can be uploaded via `px.log_evaluations(...)`. Adapter wraps our `Assessment` → `EvalResult`. |
| Trace tags | OTel attributes on the root span; surfaced in the UI as searchable fields. |
| Run grouping | Project name (`PHOENIX_PROJECT_NAME`) plus session id. Project ≈ experiment. |
| Delete trace | Phoenix doesn't officially support deletion via SDK (as of last verification). **For the judge-trace problem, the adapter must rely on the exporter-side filter (`plans/suppress-llm-judge-traces.md`) rather than post-hoc cleanup.** |

**Phoenix-specific gotchas:**
- Phoenix wants spans in OpenInference attribute shape. MLflow autolog produces `mlflow.spanType` attributes. They aren't equivalent. The OTel ingestion plan covers normalisation; until it lands, Phoenix users will see weakly-typed spans in the UI. Acceptable for Phase 2, called out in docs.
- Phoenix runs as a local web app (or via Arize cloud). The adapter should default to launching the local app on import unless `PHOENIX_COLLECTOR_ENDPOINT` is set.
- Phoenix has a `Dataset` concept that overlaps with ours. Out of scope for the adapter; potential Phase-3 work.

---

## Critical files / new modules

| File | Change | Phase |
|---|---|---|
| `src/ragpill/backends/__init__.py` (new) | Public `get_backend()` registry + protocol re-exports | 1 |
| `src/ragpill/backends/_base.py` (new) | Four `Protocol` classes (`TraceCaptureBackend`, `TraceQueryBackend`, `ResultsBackend`, `LifecycleBackend`) | 1 |
| `src/ragpill/backends/_types.py` (new) | Vendor-neutral `Assessment`, `RunHandle`, `SpanHandle`, `SpanKind` | 1 |
| `src/ragpill/backends/mlflow_backend.py` (new) | MLflow adapter (just forwards) | 1 |
| `src/ragpill/upload.py` | Replace `mlflow.log_*` calls with `backend.log_*` | 1 |
| `src/ragpill/execution.py` | Replace `mlflow.set_tracking_uri` / `start_run` / `start_span` / `autolog` with backend calls | 1 |
| `src/ragpill/evaluators.py` | `LLMJudge.run` uses `backend.start_span(...)`; `SpanBaseEvaluator.get_trace` returns the alias type | 1 |
| `src/ragpill/mlflow_helper.py` | Either merge into `mlflow_backend.py` or shrink to lazy import shim | 1 |
| `src/ragpill/__init__.py` | Drop MLflow re-exports or make them lazy | 1 |
| `src/ragpill/settings.py` | Generalise `MLFlowSettings` → `MLflowSettings` (deprecate old name) + add `LangfuseSettings`, `PhoenixSettings` | 1 (rename) / 2 (new) |
| `pyproject.toml` | `mlflow-skinny` moves to `[project.optional-dependencies]` under `mlflow = [...]`; add `langfuse = [...]`, `phoenix = [...]` slots | 1 (mlflow extra) / 2 (others) |
| `src/ragpill/backends/langfuse_backend.py` (new) | Langfuse adapter | 2.B |
| `src/ragpill/backends/phoenix_backend.py` (new) | Phoenix adapter | 2.C |
| `.github/workflows/ci.yml` | Matrix entry for "no extras" smoke job + "[mlflow]" job + later per-backend integration jobs | 1 then expand |

## Test plan

### Phase 1 acceptance

- `uv run pytest` with `pip install -e .` (no extras) — only non-MLflow tests run; all pass.
- `uv run pytest` with `pip install -e .[mlflow]` — full existing suite passes unchanged. **This is the no-regressions gate.**
- New unit test: `test_upload_uses_backend_registry` — patches `get_results_backend()` to return a `MagicMock` and asserts every existing `mlflow.log_*` call site funnels through it.
- New unit test: `test_llm_judge_span_no_unique_race` — runs `LLMJudge.run` against the in-process MLflow SQLite backend in a loop and asserts no exception. Locks the autolog-race behaviour that originally motivated the manual span.
- New CI job `test-no-extras` runs the suite without MLflow installed.

### Phase 2 per-adapter acceptance

For each adapter (Langfuse, Phoenix):

- An integration test analogous to `tests/test_mlflow_integration.py` that requires a running backend instance, gated by an env var (`RUN_LANGFUSE_INTEGRATION_TESTS=1`, etc.).
- Parity check: same `EvaluationOutput` produces equivalent server-side state (same metrics names, same number of assessments, same trace tags).
- A "switch backends mid-test" test confirming the registry resolves cleanly when reconfigured.

## Out of scope (deferred)

- **Auto-detecting which backend the user wants.** Configuration stays explicit (env var, `configure(...)` call). No magic.
- **Multi-backend dual-write** for traces beyond a single demonstrator test. The plumbing exists once the registry is generic, but the UX (what does it mean to write the same trace to two places?) deserves its own design.
- **Phoenix-side instrumentation gaps** for pydantic-ai. If the OpenInference instrumentor lags, we ship the adapter anyway with a known-limitation note.
- **Trace-shape normalisation across dialects.** Belongs to `designs/otel-trace-ingestion.md`. Phase 1 papers over it with the type-alias shortcut.

## ADR worth writing

This is a Large decision (cross-cutting architecture, public API surface, dependency choice). When the ADR system in `plans/adr-system.md` lands, this is one of the first ADRs to backfill — sequence number to be assigned per `plans/adr-system.md`'s date-ordered scheme.

## Sequencing

1. Land the Phase 1 refactor in a single PR that does **not** change runtime behaviour. This is the highest-risk PR (touches every layer) but with zero new functionality, so the diff is essentially "replace `mlflow.X(...)` with `backend.X(...)`".
2. Land the `pyproject.toml` extras change + CI matrix change in a follow-on PR. Splitting keeps the refactor reviewable.
3. Land the OTel ingestion normalisation (`designs/otel-trace-ingestion.md`) once Phase 1 is in. Required for Phase 2 to be clean.
4. Phase 2.B: Langfuse adapter PR. Largest of the three.
5. Phase 2.C: Phoenix adapter PR.
6. Optional: per-backend enhancement PRs (Langfuse sessions, Phoenix evals, etc.).
