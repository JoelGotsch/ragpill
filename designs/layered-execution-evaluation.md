# Design: Layered Execution / Evaluation Architecture

**Status:** Draft  
**Date:** 2026-04-16  
**Depends on:** [remove-pydantic-evals.md](remove-pydantic-evals.md) (removing pydantic_evals types first simplifies this refactor)

---

## 1. Motivation

The current architecture couples three concerns into a single function (`evaluate_testset_with_mlflow`):

1. **Task execution** — running the user's task against inputs, capturing outputs and traces
2. **Evaluation** — running evaluators against those outputs/traces
3. **MLflow upload** — persisting results as assessments, metrics, and tags

This makes it impossible to:
- Run tasks once and evaluate multiple times (e.g., iterate on evaluators without re-running expensive LLM calls)
- Evaluate without an MLflow server (e.g., in CI with just assertions)
- Capture traces locally and upload later (e.g., disconnected environments)
- Serialize/deserialize execution results (e.g., share runs across machines)
- Run evaluators against historical outputs

**Goal:** Split into three composable layers with well-defined data objects flowing between them.

---

## 2. Proposed Architecture

```
              Layer 1                Layer 2               Layer 3
         ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
         │   Execute     │      │   Evaluate    │      │   Upload     │
Input ──>│   tasks       │─────>│   outputs     │─────>│   to MLflow  │
         │               │      │               │      │              │
         │  DatasetRun   │      │  Evaluation   │      │  MLflow      │
         │  Output       │      │  Output       │      │  Server      │
         └──────────────┘      └──────────────┘      └──────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
         Trace capture         Evaluators run          Assessments,
         (local MLflow)        on output objects       metrics, tags
```

### Data Flow

```
Dataset + Task/TaskFactory
        │
        ▼
   execute_dataset()  ──────>  DatasetRunOutput
                                   │
                                   ▼  (+ evaluators)
                          evaluate_results()  ──>  EvaluationOutput
                                                       │
                                                       ▼
                                              upload_to_mlflow()  ──>  MLflow Server
```

Each layer is independently callable. The combined `evaluate_testset_with_mlflow()` remains as a convenience that chains all three.

---

## 3. Layer 1: Task Execution

### 3.1 Output Types

```python
@dataclass
class TaskRunOutput:
    """Output from a single task execution (one run of one case)."""
    run_index: int
    input_key: str
    output: Any                    # task return value
    duration: float                # wall-clock seconds
    trace: Trace | None            # MLflow Trace object (captured spans)
    run_span_id: str               # span ID for this run within the trace
    error: Exception | None = None # if task raised

@dataclass
class CaseRunOutput:
    """Output from executing all runs of a single test case."""
    case_name: str
    inputs: Any
    metadata: TestCaseMetadata
    base_input_key: str
    trace_id: str                  # MLflow trace ID for the parent span
    trace: Trace | None            # full Trace object for this case
    runs: list[TaskRunOutput]
    repeat: int
    threshold: float

@dataclass
class DatasetRunOutput:
    """Output from executing all cases in a dataset."""
    cases: list[CaseRunOutput]
    execution_time: float          # total wall-clock seconds
    settings: dict[str, Any]       # capture of settings used
```

### 3.2 The `Trace` Object

The key question: **can we capture traces without a running MLflow server?**

**Answer: Yes.** Validated experimentally:
- MLflow can use a local SQLite backend (`sqlite:///path/to/temp.db`) for trace storage
- `mlflow.pydantic_ai.autolog()` patches pydantic-ai classes regardless of backend — all child spans (Agent, LLM, Tool calls) are auto-captured
- After spans close, `mlflow.search_traces()` returns fully populated `Trace` objects from the local store
- `Trace` objects survive after `mlflow.end_run()` and even after changing the tracking URI — they are self-contained Python objects holding `TraceData(spans=[Span, ...])`
- `trace.search_spans(span_type=SpanType.RETRIEVER)` works on detached traces
- `Span` objects carry all data: `span_id`, `parent_id`, `name`, `span_type`, `inputs`, `outputs`, `attributes`

**Autolog integrations available in MLflow** (all work with local backend):
- `mlflow.pydantic_ai.autolog()` — Agent.run, InstrumentedModel.request, ToolManager.handle_call
- `mlflow.langchain.autolog()` — Chain, LLM, Tool, Retriever spans
- `mlflow.openai.autolog()` — Chat completions, embeddings
- `mlflow.anthropic.autolog()` — Messages API
- `mlflow.llama_index.autolog()` — Query engine, retriever, LLM spans
- Plus: crewai, dspy, haystack, semantic_kernel, smolagents, strands, etc.

### 3.3 Execution Function

```python
async def execute_dataset(
    dataset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    settings: MLFlowSettings | None = None,
    autolog: list[str] | None = None,  # e.g. ["pydantic_ai", "langchain"]
    capture_traces: bool = True,
) -> DatasetRunOutput:
    """Layer 1: Execute tasks and capture outputs + traces.
    
    Runs tasks against each case in the dataset, capturing outputs and 
    (optionally) MLflow traces. Does NOT run evaluators or upload to MLflow.
    
    Uses a local temporary SQLite backend for trace capture — no MLflow 
    server needed. Traces are extracted into the output objects and the 
    temp database is cleaned up.
    """
```

**Implementation approach:**

```python
async def execute_dataset(...) -> DatasetRunOutput:
    settings = settings or MLFlowSettings()
    
    # Set up local-only tracing (temp SQLite file)
    if capture_traces:
        _setup_local_tracing(autolog or ["pydantic_ai"])
    
    case_outputs: list[CaseRunOutput] = []
    for case in dataset.cases:
        repeat, threshold = resolve_repeat(case.metadata, settings)
        case_output = await _execute_case_runs(case, _factory, repeat, threshold, capture_traces)
        case_outputs.append(case_output)
    
    if capture_traces:
        _teardown_local_tracing()
    
    return DatasetRunOutput(cases=case_outputs, ...)
```

### 3.4 Local Trace Capture

```python
import tempfile
import mlflow

_local_db_path: str | None = None

def _setup_local_tracing(autolog_integrations: list[str]) -> None:
    """Configure MLflow to trace locally without a server."""
    global _local_db_path
    _local_db_path = tempfile.mktemp(suffix=".db")
    mlflow.set_tracking_uri(f"sqlite:///{_local_db_path}")
    mlflow.set_experiment("ragpill_local_capture")
    
    # Enable requested autolog integrations
    for integration in autolog_integrations:
        getattr(mlflow, integration).autolog()
    
    mlflow.start_run()

def _teardown_local_tracing() -> None:
    """End MLflow run and clean up temp database."""
    mlflow.end_run()
    if _local_db_path and os.path.exists(_local_db_path):
        os.unlink(_local_db_path)
```

After each case's spans are committed, we extract the `Trace` object:

```python
def _extract_trace(experiment_id: str, trace_id: str) -> Trace:
    """Extract a Trace object from the local MLflow store."""
    traces = mlflow.search_traces(
        locations=[experiment_id],
        return_type="list",
    )
    for t in traces:
        if t.info.trace_id == trace_id:
            return t
    raise ValueError(f"Trace {trace_id} not found")
```

The `Trace` object is a self-contained Python object — once extracted, it doesn't need the MLflow backend anymore.

### 3.5 Execution Without Traces

When `capture_traces=False`:
- No MLflow setup at all
- Tasks execute directly
- `TaskRunOutput.trace = None` and `run_span_id = ""`
- Span-based evaluators will raise if used against trace-less outputs
- Useful for simple input/output evaluators (regex, LLM judge) in environments where tracing overhead is unwanted

---

## 4. Layer 2: Evaluation

### 4.1 Evaluator Context: Trace from Output Object

Currently, span-based evaluators call `mlflow.search_traces()` to get trace data. In the new architecture, they receive the trace from the output object:

```python
# Current (queries MLflow backend):
class SpanBaseEvaluator(BaseEvaluator):
    def get_trace(self, inputs: Any) -> Trace:
        traces = mlflow.search_traces(
            locations=[self.mlflow_experiment_id],
            run_id=self.mlflow_run_id,
            return_type="list",
        )
        # ... match by input_key ...

# New (receives trace from output object):
class SpanBaseEvaluator(BaseEvaluator):
    def get_trace(self, ctx: EvaluatorContext) -> Trace:
        if ctx.trace is None:
            raise ValueError(
                "No trace available. Span-based evaluators require "
                "capture_traces=True during execute_dataset()."
            )
        return ctx.trace
```

### 4.2 Updated EvaluatorContext

```python
@dataclass(kw_only=True)
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context passed to evaluators during evaluation."""
    name: str | None
    inputs: InputsT
    metadata: MetadataT | None
    expected_output: OutputT | None
    output: OutputT
    duration: float
    trace: Trace | None = None         # NEW: captured trace for span-based evaluators
    run_span_id: str | None = None     # NEW: which run's subtree to evaluate
    attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
```

The `trace` and `run_span_id` fields replace the MLflow-querying mechanism. The `_current_run_span_id` ContextVar is no longer needed — the run's span ID is directly on the context.

### 4.3 SpanBaseEvaluator Refactor

```python
@dataclass(kw_only=True, repr=False)
class SpanBaseEvaluator(BaseEvaluator):
    """Base class for evaluators that inspect execution traces."""
    
    def get_trace(self, ctx: EvaluatorContext) -> Trace:
        """Get the trace for evaluation, filtered to the current run's subtree."""
        if ctx.trace is None:
            raise ValueError("No trace captured. Use capture_traces=True.")
        
        trace = ctx.trace
        if ctx.run_span_id is not None:
            trace = _filter_trace_to_subtree(trace, ctx.run_span_id)
        return trace
```

**What's removed:**
- `_mlflow_settings`, `_mlflow_experiment_id`, `_mlflow_run_id` fields
- `mlflow_settings`, `mlflow_experiment_id`, `mlflow_run_id` properties
- `mlflow.search_traces()` call
- `inputs_to_key_function` field (no longer needed — we don't search by key)

**What's gained:**
- Evaluators work on any `Trace` object, regardless of how it was captured
- No dependency on active MLflow session during evaluation
- Testability: pass a mock `Trace` directly

### 4.4 SourcesBaseEvaluator Refactor

```python
@dataclass(kw_only=True, repr=False)
class SourcesBaseEvaluator(SpanBaseEvaluator):
    """Base class that retrieves source documents from traces."""
    
    evaluation_function: Callable[[list[Document]], bool]
    custom_reason_true: str = "Evaluation function returned True."
    custom_reason_false: str = "Evaluation function returned False."
    
    def get_documents(self, ctx: EvaluatorContext) -> list[Document]:
        trace = self.get_trace(ctx)
        retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)
        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
        reranker_spans = trace.search_spans(span_type=SpanType.RERANKER)
        # ... extract Documents from span outputs (unchanged logic) ...
    
    async def run(self, ctx: EvaluatorContext) -> EvaluationReason:
        documents = self.get_documents(ctx)  # was: self.get_documents(ctx.inputs)
        result = self.evaluation_function(documents)
        return EvaluationReason(value=result, reason=...)
```

**Key change:** `get_documents()` takes `ctx` instead of `inputs`. It gets the trace from the context rather than querying MLflow.

### 4.5 Evaluation Function

```python
async def evaluate_results(
    dataset_run: DatasetRunOutput,
    dataset: Dataset[Any, Any, CaseMetadataT],
) -> EvaluationOutput:
    """Layer 2: Run evaluators against execution outputs.
    
    Takes the output from execute_dataset() and the original dataset 
    (for evaluator definitions), runs all evaluators, and returns 
    structured results with DataFrames.
    
    No MLflow server needed. No task execution. Pure evaluation.
    """
```

**Implementation:**

```python
async def evaluate_results(
    dataset_run: DatasetRunOutput,
    dataset: Dataset[Any, Any, CaseMetadataT],
) -> EvaluationOutput:
    case_results: list[CaseResult] = []
    
    for case_run, case in zip(dataset_run.cases, dataset.cases):
        all_evaluators = list(case.evaluators) + list(dataset.evaluators)
        
        run_results: list[RunResult] = []
        for task_run in case_run.runs:
            if task_run.error is not None:
                # ... mark all evaluators as failed ...
                continue
            
            ctx = EvaluatorContext(
                name=case_run.case_name,
                inputs=case_run.inputs,
                metadata=case_run.metadata,
                expected_output=case.expected_output,
                output=task_run.output,
                duration=task_run.duration,
                trace=case_run.trace,          # pass the captured trace
                run_span_id=task_run.run_span_id,  # for subtree filtering
            )
            
            run_result = await _evaluate_single_run(ctx, all_evaluators, task_run)
            run_results.append(run_result)
        
        aggregated = _aggregate_runs(run_results, case_run.threshold)
        case_results.append(CaseResult(...))
    
    runs_df = _create_runs_dataframe(case_results)
    cases_df = _create_cases_dataframe(case_results)
    return EvaluationOutput(
        runs=runs_df, cases=cases_df, case_results=case_results,
        dataset_run=dataset_run,  # keep reference to execution data
    )
```

---

## 5. Layer 3: MLflow Upload

### 5.1 Upload Function

```python
def upload_to_mlflow(
    evaluation: EvaluationOutput,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
    upload_traces: bool = True,
) -> None:
    """Layer 3: Upload evaluation results to an MLflow server.
    
    Connects to the MLflow tracking server specified in settings,
    creates a run, and uploads:
    - Traces (re-exported from captured local traces)
    - Assessments (per-run and aggregate)
    - Metrics (overall accuracy, per-tag accuracy)
    - DataFrames as tables
    - Tags from case metadata
    
    This is the only function that requires a running MLflow server.
    """
```

**Implementation outline:**

```python
def upload_to_mlflow(
    evaluation: EvaluationOutput,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
    upload_traces: bool = True,
) -> None:
    settings = mlflow_settings or MLFlowSettings()
    
    # Connect to the real MLflow server
    mlflow.set_tracking_uri(settings.ragpill_tracking_uri)
    mlflow.set_experiment(settings.ragpill_experiment_name)
    mlflow.start_run(description=settings.ragpill_run_description)
    
    if upload_traces:
        # Re-create traces on the server
        _upload_traces(evaluation.dataset_run, settings)
    
    # Upload evaluation results (assessments, metrics, tags)
    _upload_assessments(evaluation)
    _upload_metrics(evaluation, model_params)
    
    # Log DataFrames
    mlflow.log_table(evaluation.runs, "evaluation_results.json")
    
    mlflow.end_run()
```

### 5.2 Trace Re-Export

The tricky part: re-creating locally-captured traces on a remote MLflow server.

**Option A: Re-run with server tracing (simplest, but re-executes tasks)**
Not viable — defeats the purpose.

**Option B: Use MLflow client API to create traces**

```python
def _upload_traces(dataset_run: DatasetRunOutput, settings: MLFlowSettings) -> dict[str, str]:
    """Re-create traces on the MLflow server from captured trace data.
    
    Returns mapping of local_trace_id -> server_trace_id.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.ragpill_tracking_uri)
    trace_id_map: dict[str, str] = {}
    
    for case_run in dataset_run.cases:
        if case_run.trace is None:
            continue
        
        # Create a new trace on the server mirroring the local one
        # MLflow's client API allows creating traces programmatically
        root_span = case_run.trace.data.spans[0]  # root span
        
        server_span = client.start_trace(
            name=root_span.name,
            inputs=root_span.inputs,
            span_type=root_span.span_type,
        )
        server_trace_id = server_span.request_id
        trace_id_map[case_run.trace_id] = server_trace_id
        
        # Recreate child spans...
        _recreate_spans(client, case_run.trace, server_trace_id)
        
        client.end_trace(server_trace_id)
    
    return trace_id_map
```

**Option C: Replay execution with server tracing**

Run the tasks again but this time with the server as the backend. This gives perfect trace fidelity but doubles execution cost.

**Option D: Upload trace data as artifacts**

Instead of recreating traces, serialize the `Trace` objects and upload as MLflow artifacts (JSON). The MLflow UI wouldn't show them as native traces, but the data is preserved.

**Recommended: Option B with fallback to D.** Option B gives the best UX (traces appear in MLflow UI), but if the `client.start_trace` API is too limited to recreate complex span trees, fall back to serializing as artifacts.

### 5.3 Alternative: Dual-Backend Tracing

Instead of capturing locally and re-uploading, use MLflow's tracing with the real server from the start, but make it optional:

```python
async def execute_dataset(
    ...,
    mlflow_tracking_uri: str | None = None,  # None = local temp, str = remote server
) -> DatasetRunOutput:
```

- When `mlflow_tracking_uri` is `None`: use a local temp SQLite backend, extract traces into output objects
- When `mlflow_tracking_uri` is set: trace directly to the server (current behavior) AND extract into output objects

This means `upload_to_mlflow()` in the remote case only uploads assessments/metrics (traces already exist). In the local case, it does the full upload.

---

## 6. Convenience Function (Backward Compatibility)

The existing `evaluate_testset_with_mlflow()` becomes a thin orchestrator:

```python
async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> EvaluationOutput:
    """Execute, evaluate, and upload — all in one call.
    
    Equivalent to:
        run_output = await execute_dataset(testset, task, task_factory, settings)
        eval_output = await evaluate_results(run_output, testset)
        upload_to_mlflow(eval_output, settings, model_params)
        return eval_output
    """
    settings = mlflow_settings or MLFlowSettings()
    
    # Layer 1: Execute
    run_output = await execute_dataset(
        testset, task=task, task_factory=task_factory,
        settings=settings,
        mlflow_tracking_uri=settings.ragpill_tracking_uri,  # trace directly to server
    )
    
    # Layer 2: Evaluate
    eval_output = await evaluate_results(run_output, testset)
    
    # Layer 3: Upload (assessments, metrics — traces already on server)
    upload_to_mlflow(eval_output, settings, model_params, upload_traces=False)
    
    return eval_output
```

---

## 7. New Use Cases Unlocked

### 7.1 Run Once, Evaluate Many Times

```python
# Execute tasks (expensive LLM calls)
run_output = await execute_dataset(testset, task=my_agent)

# Iterate on evaluators without re-running tasks
eval_v1 = await evaluate_results(run_output, testset_v1)
eval_v2 = await evaluate_results(run_output, testset_v2)

# Upload the best version
upload_to_mlflow(eval_v2)
```

### 7.2 CI Without MLflow Server

```python
# In CI: execute + evaluate locally, assert on results
run_output = await execute_dataset(testset, task=my_agent, capture_traces=True)
eval_output = await evaluate_results(run_output, testset)

# Pure assertion — no MLflow server needed
assert eval_output.summary["passed"].all(), f"Failures:\n{eval_output.summary}"
```

### 7.3 Serialize and Share

```python
# On machine A: execute
run_output = await execute_dataset(testset, task=my_agent)
save_run_output(run_output, "run_2026-04-16.pkl")  # future: JSON serialization

# On machine B: evaluate
run_output = load_run_output("run_2026-04-16.pkl")
eval_output = await evaluate_results(run_output, testset)
upload_to_mlflow(eval_output)
```

### 7.4 Disconnected Execution

```python
# On a disconnected machine (no MLflow server)
run_output = await execute_dataset(testset, task=my_agent)
eval_output = await evaluate_results(run_output, testset)

# Later, when connected
upload_to_mlflow(eval_output, mlflow_settings=production_settings)
```

---

## 8. Impact on Span-Based Evaluators

### 8.1 What Changes

| Evaluator | Current | New |
|-----------|---------|-----|
| `SpanBaseEvaluator` | Queries `mlflow.search_traces()`, uses `input_key` matching | Gets `Trace` from `ctx.trace`, filters by `ctx.run_span_id` |
| `SourcesBaseEvaluator` | Calls `self.get_trace(inputs)` → `self.get_documents(inputs)` | Calls `self.get_trace(ctx)` → `self.get_documents(ctx)` |
| `RegexInSourcesEvaluator` | Unchanged internally | `run(ctx)` passes `ctx` to `get_documents` instead of `ctx.inputs` |
| `RegexInDocumentMetadataEvaluator` | Same | Same |
| `LiteralQuoteEvaluator` | Same | Same |

### 8.2 What's Removed from SpanBaseEvaluator

```python
# REMOVED — no longer needed:
_mlflow_settings: MLFlowSettings | None
_mlflow_experiment_id: str | None
_mlflow_run_id: str | None
inputs_to_key_function: Callable[[Any], str]

@property
def mlflow_settings(self) -> MLFlowSettings: ...
@property
def mlflow_experiment_id(self) -> str: ...
@property
def mlflow_run_id(self) -> str: ...
```

### 8.3 What's Removed Globally

- `_current_run_span_id` ContextVar in `base.py` — replaced by `ctx.run_span_id`
- `default_input_to_key()` in `base.py` — no longer needed for trace matching
- `inputs_to_key_function` concept — traces are passed directly, not looked up by key

### 8.4 Tracing Still Works for User's Libraries

The key concern: *"that would mean we have to do all the tracing that mlflow does for us right now"*

**We don't have to reimplement tracing.** MLflow's autolog system works with any backend, including a local temp SQLite file. During `execute_dataset()`:

1. We set up MLflow with a local backend
2. We enable autolog for the user's libraries (`pydantic_ai`, `langchain`, etc.)
3. The user's task executes inside `mlflow.start_span()` 
4. Autolog automatically instruments library calls, creating child spans for LLM calls, tool invocations, retriever operations, etc.
5. After execution, we extract the `Trace` object from the local backend
6. The trace contains the full span tree, exactly as if a server were running

**No reimplementation of tracing needed.** We use MLflow as a trace capture engine, not as a persistence layer.

---

## 9. Design Decisions

### 9.1 Keep MLflow Trace as the Span Data Model

**Decision:** Use `mlflow.entities.Trace` and `mlflow.entities.Span` directly in our output types, rather than creating our own span data model.

**Rationale:**
- MLflow's `Trace` object is self-contained and survives after `end_run()`
- `trace.search_spans(span_type=...)` works on detached traces
- The `Document` type extraction from span outputs is already proven
- Avoids a redundant data model translation layer
- If we later support other tracing backends (OpenTelemetry), we can add a `TraceAdapter` interface

**Trade-off:** Adds `mlflow` as a type dependency even for the execution layer. But MLflow is already a required dependency, so this is not a new constraint.

### 9.2 Autolog Configuration

**Decision:** Let users specify which autolog integrations to enable.

```python
run_output = await execute_dataset(
    testset, task=my_agent,
    autolog=["pydantic_ai", "langchain"],
)
```

**Default:** `["pydantic_ai"]` (current behavior).

### 9.3 Trace Serialization (Future)

`Trace` objects contain `Span` objects which internally hold OpenTelemetry span references (and an `RLock` that prevents `deepcopy`). For serialization, we'd need to extract span data into plain dicts.

**Deferral:** Don't implement serialization now. Hold `Trace` objects by reference in memory. Serialization can be added later with a `to_dict()` / `from_dict()` method on `DatasetRunOutput`.

### 9.4 Trace Upload Strategy

**Decision:** Use the dual-backend approach (Section 5.3).
- When the user provides a tracking URI (via `evaluate_testset_with_mlflow` or explicitly), trace directly to that server during execution
- When no URI is provided (`execute_dataset` alone), use a local temp backend
- `upload_to_mlflow()` only uploads assessments/metrics/tags when traces already exist on server; uploads trace artifacts when they were captured locally

This avoids the complexity of recreating traces via the MLflow client API.

---

## 10. Public API Summary

### New Functions

```python
# Layer 1: Execute
async def execute_dataset(
    dataset: Dataset,
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    settings: MLFlowSettings | None = None,
    autolog: list[str] | None = None,
    capture_traces: bool = True,
    mlflow_tracking_uri: str | None = None,  # None = local temp
) -> DatasetRunOutput: ...

def execute_dataset_sync(...) -> DatasetRunOutput: ...

# Layer 2: Evaluate
async def evaluate_results(
    dataset_run: DatasetRunOutput,
    dataset: Dataset,
) -> EvaluationOutput: ...

def evaluate_results_sync(...) -> EvaluationOutput: ...

# Layer 3: Upload
def upload_to_mlflow(
    evaluation: EvaluationOutput,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
    upload_traces: bool = True,
) -> None: ...
```

### New Types

```python
TaskRunOutput       # one run of one case
CaseRunOutput       # all runs of one case
DatasetRunOutput    # all cases
```

### Modified Types

```python
EvaluatorContext    # gains trace and run_span_id fields
EvaluationOutput   # gains dataset_run field
```

### Existing Functions (unchanged signatures)

```python
evaluate_testset_with_mlflow()       # chains all 3 layers
evaluate_testset_with_mlflow_sync()  # sync wrapper
load_testset()                       # CSV loading (unchanged)
```

---

## 11. Implementation Order

This builds on top of the pydantic-evals removal (see `remove-pydantic-evals.md`). Best done after that refactor.

### Phase 1: Output Types + Execute Layer
1. Define `TaskRunOutput`, `CaseRunOutput`, `DatasetRunOutput`
2. Extract `_execute_case_runs()` from `mlflow_helper.py` into a new `execution.py`
3. Implement `execute_dataset()` with local tracing support
4. Add `trace` and `run_span_id` to `EvaluatorContext`

### Phase 2: Evaluate Layer
5. Refactor `SpanBaseEvaluator.get_trace()` to use `ctx.trace`
6. Update `SourcesBaseEvaluator.get_documents()` signature
7. Remove `_current_run_span_id` ContextVar, `inputs_to_key_function`, MLflow query fields
8. Extract evaluation loop into `evaluate_results()`

### Phase 3: Upload Layer
9. Extract MLflow upload logic into `upload_to_mlflow()`
10. Implement dual-backend trace handling
11. Rewire `evaluate_testset_with_mlflow()` to chain all three

### Phase 4: Tests + Docs
12. Tests for each layer independently
13. Integration test for the full pipeline
14. Tests for `execute_dataset` without MLflow server
15. Documentation for new API

---

## 12. Risks and Open Questions

### 12.1 Trace Object Size

**Risk:** For datasets with many cases and deep span trees, holding all `Trace` objects in memory could be significant.

**Mitigation:** Traces are already in memory during the current single-function flow. The new architecture doesn't increase peak memory — it just keeps references longer. For very large datasets, add an option to discard traces after evaluation (`keep_traces=False`).

### 12.2 Trace Fidelity During Re-Upload

**Risk:** Recreating traces on a remote server may not perfectly match the original (different trace IDs, timing metadata lost).

**Mitigation:** Use the dual-backend approach — when a server URI is provided, trace directly to it. Re-upload is only needed for the "execute locally, upload later" flow, where some metadata loss is acceptable.

### 12.3 Autolog Side Effects

**Risk:** Calling `mlflow.pydantic_ai.autolog()` patches global state. If the user has their own MLflow setup, our local tracing could interfere.

**Mitigation:** 
- Save and restore the tracking URI around `execute_dataset()`
- Document that `execute_dataset()` temporarily modifies MLflow global state
- Consider using `mlflow.start_run(nested=True)` or a separate experiment

### 12.4 Thread Safety

**Risk:** MLflow global state (tracking URI, active run) is process-wide, not thread-safe.

**Mitigation:** Same as current situation — document that ragpill evaluation should not run concurrently in the same process. The `_current_run_span_id` ContextVar removal actually improves this (no global mutable state for run isolation).

### 12.5 Open Question: Should Layer 2 Also Be Sync-First?

The current evaluators are async (they `await` LLM judge calls). Should `evaluate_results()` be async?

**Recommendation:** Keep async as primary, provide sync wrapper. LLMJudge is inherently async (makes LLM API calls). Other evaluators are fast enough that the async overhead is negligible.
