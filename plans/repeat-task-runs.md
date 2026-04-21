# Plan: Repeatable Task Runs with Threshold-Based Pass/Fail

## Goal

Add per-case `repeat` and `threshold` to `TestCaseMetadata` so each test case can run N times. A case passes if the ratio of successful runs meets the threshold. This addresses LLM stochasticity where a single run may not be representative.

```python
class TestCaseMetadata(BaseModel):
    # ... existing fields ...
    repeat: int = Field(default=1, ge=1)
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)
```

---

## Context: What pydantic_evals Already Provides

pydantic_evals has a **global** `repeat` parameter on `Dataset.evaluate(repeat=N)`:
- Runs every case N times, appending `[1/N]`, `[2/N]` to case names
- Groups results via `report.case_groups()` → `ReportCaseGroup`
- Computes averages via `ReportCaseAggregate` (includes assertion pass rate)
- `ReportCase.source_case_name` links runs back to the original case

**What it does NOT provide:**
- Per-case repeat (it's global or nothing)
- Threshold / pass-fail decision
- MLflow integration for grouped traces

---

## The Core Problem

The current architecture maps traces to results via `input_key = md5(str(input))`. Repeated runs of the same input produce the same key, which breaks:
- `_get_input_key_trace_id_map` (last trace wins)
- `_get_input_key_report_case_map` (assertion: no duplicate keys)
- `SpanBaseEvaluator.get_trace()` (finds multiple traces, returns first)

Any approach must solve this identity problem.

---

## Task State & Isolation Analysis

### Current Assumptions

The current `TaskType = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]` assumes:

1. **Single instance reuse:** One `task` callable is created, captured in the MLflow wrapper closure, and called once per case. There's no mechanism to create a fresh instance per call.
2. **Implicit statelessness:** The code assumes `task(input)` can be called N times with different inputs and each call is independent. No isolation between calls.
3. **No lifecycle management:** No setup/teardown per call. No resource cleanup.

### Why This Breaks With Repeated Runs

For repeated runs to produce **independent samples** (the whole point), each run must start from identical initial state. But agents in practice are often stateful:

| Agent Pattern | State Concern | Repeat Impact |
|---------------|---------------|---------------|
| **pydantic-ai `Agent.run()`** (default) | Creates fresh conversation per call — **stateless by default** | Safe to reuse |
| **pydantic-ai with `message_history`** | Accumulates conversation history across calls | Run 1 sees Run 0's messages → not independent |
| **Agent with tool state** (DB connections, caches, counters) | Side effects persist across calls | Run 1 affected by Run 0's side effects |
| **LangChain agent with memory** | `ConversationBufferMemory` accumulates | Runs are correlated, not independent |
| **Custom agent class with `self.state`** | Instance-level mutable state | Bleeds between runs AND between cases |

The tutorial example (`docs/tutorials/full.md:131-163`) shows the typical pattern:

```python
agent = Agent(model=pyai_llm_model, system_prompt=system_prompt)

async def task(question: str) -> str:
    result = await agent.run(question)
    return result.output
```

This is safe for repeat (pydantic-ai Agent.run() is stateless per call). But as soon as a user adds memory, caching, or mutable instance state, it silently breaks the independence assumption.

### The State Isolation Problem Is Broader Than Repeat

Even without repeat (`repeat=1`), running Case A then Case B through a stateful agent means Case B's result depends on Case A. This isn't a new problem — it exists today — but repeat makes it acute because the user explicitly expects independent samples.

### Solution: Task Factory

Introduce `task_factory: Callable[[], TaskType]` — a zero-argument callable that returns a fresh task instance. The execution loop calls the factory before each run to get an isolated task.

**Design options:**

#### Option 1: Mutually exclusive parameters (explicit, recommended)

```python
async def evaluate_testset_with_mlflow(
    testset: Dataset,
    task: TaskType | None = None,
    task_factory: Callable[[], TaskType] | None = None,
    ...
) -> EvaluationOutput:
    if task is not None and task_factory is not None:
        raise ValueError("Provide either 'task' or 'task_factory', not both.")
    if task is None and task_factory is None:
        raise ValueError("Provide either 'task' or 'task_factory'.")
    
    if task is not None:
        _factory = lambda: task  # stateless: reuse same callable
    else:
        _factory = task_factory
```

- `task=my_func` — "my function is stateless, call it directly" (simple path)
- `task_factory=create_agent` — "create a fresh agent for each run" (stateful path)

Pros: Explicit intent, impossible to misuse, clear documentation path.
Cons: Two parameters where one concept exists.

#### Option 2: Single parameter, smart detection

```python
# Accept a union type with runtime detection
TaskInput = TaskType | Callable[[], TaskType]

async def evaluate_testset_with_mlflow(
    testset: Dataset,
    task: TaskInput,
    ...
):
    sig = inspect.signature(task)
    if len(sig.parameters) == 0:
        _factory = task  # zero-arg → factory
    else:
        _factory = lambda: task  # has params → stateless task
```

Pros: Single parameter, less cognitive overhead.
Cons: Fragile (what about `*args`, `**kwargs`, default params?). Ambiguous for callables like a class with `__call__`. Not type-safe.

#### Option 3: Always factory, helper for stateless

```python
async def evaluate_testset_with_mlflow(
    testset: Dataset,
    task_factory: Callable[[], TaskType],
    ...
):
    ...

# Helper for the common stateless case
def as_factory(task: TaskType) -> Callable[[], TaskType]:
    """Wrap a stateless task as a factory (returns the same callable each time)."""
    return lambda: task
```

Usage:
```python
# Stateless
results = await evaluate_testset_with_mlflow(testset, task_factory=as_factory(my_func))

# Stateful
results = await evaluate_testset_with_mlflow(testset, task_factory=create_agent)
```

Pros: Single concept, very explicit about isolation semantics.
Cons: Boilerplate for the common (stateless) case. `as_factory` feels unnecessary.

### Recommendation: Option 1

Option 1 is the most Pythonic: the common case (`task=`) is simple, the advanced case (`task_factory=`) is opt-in and self-documenting. The mutual exclusion is enforced at runtime with a clear error message.

### Factory Invocation Policy

When is the factory called?

| Policy | Description | Use Case |
|--------|-------------|----------|
| **Per run** | Factory called before each run within a case | Full isolation. Agent memory reset per run. |
| **Per case** | Factory called once per case, reused across runs | Cross-run memory within a case (e.g., "does the agent improve on retry?") |

**Default: per run.** This is what the user asked for — independent samples. Per-case reuse is a niche pattern that could be added later via a flag.

When `task` (not `task_factory`) is provided, the same callable is reused across all runs and cases — no factory invocation at all. This is correct because the user is asserting statelessness by using `task`.

### How the factory changes the execution loop

```python
async def _execute_case(
    case: Case,
    task_factory: Callable[[], TaskType],
    ...
) -> CaseResult:
    ...
    for i in range(repeat):
        fresh_task = task_factory()  # fresh instance per run
        run_result = await _execute_run(
            case=case,
            task=fresh_task,
            ...
        )
    ...
```

---

## Three Approaches

### Approach A: Expand Dataset + pydantic_evals evaluate (Pragmatic)

**Idea:** Before calling `testset.evaluate()`, expand cases with `repeat > 1` into N independent cases. Let pydantic_evals treat each as a separate case. Aggregate results post-evaluation.

**Identity solution:** Wrap inputs in a `RepeatedInput(original_input, repeat_index)` dataclass. The MLflow wrapper unwraps before calling the task but uses the index for a unique `input_key`.

```
Dataset (before expansion):
  Case "What is X?" (repeat=3)

Dataset (after expansion):
  Case "What is X? [0/3]" → RepeatedInput("What is X?", 0)
  Case "What is X? [1/3]" → RepeatedInput("What is X?", 1)
  Case "What is X? [2/3]" → RepeatedInput("What is X?", 2)
```

**MLflow traces:** One independent trace per run. Tagged with `repeat_group`, `repeat_index` for grouping. Aggregated assessment logged to first trace.

```
Trace: "What is X? [0/3]"  ← aggregated assessment attached here
Trace: "What is X? [1/3]"
Trace: "What is X? [2/3]"
(all tagged: repeat_group=abc123, repeat_total=3)
```

**Pros:**
- Minimal changes to evaluation flow — pydantic_evals does execution + evaluation
- No new orchestration code
- Concurrency, error handling, retries come for free

**Cons:**
- `RepeatedInput` wrapper leaks into evaluator context (`ctx.inputs` is a `RepeatedInput`, not the original string) — every span-based evaluator must handle unwrapping
- Flat MLflow traces (no parent-child hierarchy) — grouping is only via tags
- Evaluator instances are shared across expanded cases (same list reference) — could cause issues with stateful evaluators
- The expansion logic creates a parallel data model that diverges from pydantic_evals' own repeat concept

**Changes required:**
| File | What |
|------|------|
| `base.py` | Add `repeat`, `threshold` to `TestCaseMetadata`; add `RepeatedInput` dataclass |
| `csv/testset.py` | Parse `repeat`/`threshold` columns |
| `mlflow_helper.py` | `_expand_testset_for_repeats()`, aggregation, DataFrame updates |
| `evaluators.py` | `SpanBaseEvaluator.get_trace()` unwraps `RepeatedInput` |

---

### Approach B: Use pydantic_evals native repeat per-case (Middle Ground)

Correction: this param doesnt exist in pydantic ai
**Idea:** For each case, create a single-case Dataset and call `evaluate(repeat=case.metadata.repeat)` using pydantic_evals' built-in repeat. Use its native grouping and aggregation. Add threshold on top.

```python
for case in testset.cases:
    repeat = case.metadata.repeat
    single = Dataset(cases=[case], evaluators=testset.evaluators)
    report = await single.evaluate(wrapped_task, repeat=repeat)
    
    if repeat > 1:
        groups = report.case_groups()
        aggregate = groups[0].summary
        passed = aggregate.assertions >= case.metadata.threshold
    else:
        passed = all(r.value for r in report.cases[0].assertions.values())
    
    all_results.append((case, report, passed))
```

**Identity solution:** Use pydantic_evals' native naming (`"Case [1/3]"`, `"Case [2/3]"`). The wrapper uses an internal counter per base input_key to generate unique keys:

```python
def _mlflow_runnable_wrapper(task, input_to_key):
    run_counters: dict[str, int] = {}
    counter_lock = asyncio.Lock()
    
    async def async_runnable(input):
        base_key = input_to_key(input)
        async with counter_lock:
            idx = run_counters.get(base_key, 0)
            run_counters[base_key] = idx + 1
        input_key = f"{base_key}_{idx}"
        ...
```

**MLflow traces:** Same as Approach A (flat, tagged). The counter handles concurrency.

**Pros:**
- Uses pydantic_evals repeat natively — no wrapper types
- `ctx.inputs` is the original string (evaluators work unchanged)
- `ReportCaseAggregate.assertions` gives pass rate for free
- `case_groups()` does grouping for free

**Cons:**
- Per-case `evaluate()` calls — serializes execution across cases (no cross-case concurrency)
- Each `evaluate()` creates its own pydantic_evals experiment span — nested spans may confuse
- MLflow traces are still flat (no hierarchy)
- Counter-based key assignment requires careful async handling
- Multiple independent `evaluate()` calls complicate the MLflow run lifecycle

**Changes required:**
| File | What |
|------|------|
| `base.py` | Add `repeat`, `threshold` to `TestCaseMetadata` |
| `csv/testset.py` | Parse `repeat`/`threshold` columns |
| `mlflow_helper.py` | Per-case evaluate loop, counter-based wrapper, aggregation using pydantic_evals groups |
| `evaluators.py` | No changes needed (inputs are pristine) |

---

### Approach C: Own Orchestration Loop (Clean Design) — RECOMMENDED

**Idea:** Replace `testset.evaluate()` with our own execution loop. Use pydantic_evals evaluators directly (they're just classes with an `evaluate()` method that takes an `EvaluatorContext`). This gives full control over MLflow trace/span hierarchy.

**Why this is viable:** We already post-process ~80% of pydantic_evals' output. The `evaluate_testset_with_mlflow` function currently:
1. Calls `testset.evaluate()` ← pydantic_evals
2. Deletes judge traces ← our code
3. Maps traces to results ← our code
4. Maps results to metadata ← our code
5. Builds DataFrame ← our code
6. Uploads to MLflow ← our code

We can replace step 1 with our own loop and eliminate steps 2-4 entirely (since we control trace creation from the start).

**Architecture:**

```
MLflow Trace (one per test case)
└── Case Span: "case: What is X?" (type=CHAIN)
    ├── Run Span: "run_0" (type=TASK)
    │   ├── Agent spans (autologged by mlflow.pydantic_ai)
    │   ├── Eval: LLMJudge → pass
    │   └── Eval: RegexInOutput → pass
    ├── Run Span: "run_1" (type=TASK)
    │   ├── Agent spans (autologged)
    │   ├── Eval: LLMJudge → fail ("didn't mention capital")
    │   └── Eval: RegexInOutput → pass
    └── Run Span: "run_2" (type=TASK)
        ├── Agent spans (autologged)
        ├── Eval: LLMJudge → pass
        └── Eval: RegexInOutput → pass
    
    Case result: 2/3 passed (threshold: 0.6) → PASSED
```

**Core execution loop:**

```python
async def _execute_case(
    case: Case,
    task_factory: Callable[[], TaskType],
    dataset_evaluators: list[BaseEvaluator],
    input_to_key: Callable,
) -> CaseResult:
    metadata = case.metadata  # TestCaseMetadata
    repeat = metadata.repeat
    threshold = metadata.threshold
    base_key = input_to_key(case.inputs)
    all_evaluators = list(case.evaluators) + list(dataset_evaluators)
    
    with mlflow.start_span(name=f"case: {case.name}", span_type=SpanType.CHAIN) as case_span:
        case_span.set_inputs(case.inputs)
        case_span.set_attribute("input_key", base_key)
        case_span.set_attribute("repeat", repeat)
        case_span.set_attribute("threshold", threshold)
        
        run_results: list[RunResult] = []
        for i in range(repeat):
            fresh_task = task_factory()  # fresh instance per run
            run_result = await _execute_run(
                case=case,
                task=fresh_task,
                evaluators=all_evaluators,
                run_index=i,
                input_key=f"{base_key}_{i}",
            )
            run_results.append(run_result)
        
        aggregated = _aggregate_runs(run_results, threshold)
        case_span.set_outputs(aggregated.summary)
        return CaseResult(
            case=case,
            base_input_key=base_key,
            run_results=run_results,
            aggregated=aggregated,
            trace_id=case_span... ,  # capture trace id
        )


async def _execute_run(
    case: Case,
    task: TaskType,  # already a fresh instance from factory
    evaluators: list[BaseEvaluator],
    run_index: int,
    input_key: str,
) -> RunResult:
    with mlflow.start_span(name=f"run_{run_index}", span_type=SpanType.TASK) as run_span:
        run_span.set_inputs(case.inputs)
        run_span.set_attribute("input_key", input_key)
        run_span.set_attribute("run_index", run_index)
        
        # Execute task
        start = time.time()
        try:
            output = await _call_task(task, case.inputs)
        except Exception as e:
            return RunResult.failure(run_index, e)
        duration = time.time() - start
        
        run_span.set_outputs(output)
        
        # Run evaluators
        ctx = EvaluatorContext(
            name=case.name,
            inputs=case.inputs,
            metadata=case.metadata,
            expected_output=case.expected_output,
            output=output,
            duration=duration,
            _span_tree=...,  # from OTEL context or empty
            attributes={},
            metrics={},
        )
        
        eval_results: dict[str, EvaluationResult] = {}
        eval_failures: list[EvaluatorFailure] = []
        for evaluator in evaluators:
            try:
                result = await evaluator.evaluate(ctx)
                eval_results[str(evaluator.evaluation_name)] = EvaluationResult(
                    name=evaluator.get_serialization_name(),
                    value=result.value,
                    reason=result.reason,
                    source=EvaluatorSpec(
                        name=evaluator.get_serialization_name(),
                        arguments={"evaluation_name": evaluator.evaluation_name},
                    ),
                )
            except Exception as e:
                eval_failures.append(...)
        
        return RunResult(
            run_index=run_index,
            output=output,
            duration=duration,
            eval_results=eval_results,
            eval_failures=eval_failures,
            input_key=input_key,
        )
```

**Identity solution:** `input_key = f"{base_key}_{run_index}"` — always suffixed, no special-casing.

**Concurrency:** Use `asyncio.Semaphore` for max concurrent cases. Within a case, runs are sequential (same input, want deterministic ordering). Cross-case parallelism is preserved.

**SpanBaseEvaluator:** Since we control the execution loop, `SpanBaseEvaluator.get_trace()` can search by the run-specific `input_key` that's already on the current span. No changes to evaluation logic — `ctx.inputs` is the original string.

**Pros:**
- Cleanest MLflow structure: hierarchical traces with case → run spans
- No wrapper types — `ctx.inputs` is always the original input
- No fighting pydantic_evals assumptions
- Eliminates several post-processing steps (trace cleanup, key mapping)
- `SpanBaseEvaluator` works as-is (each run has a unique input_key on its span)
- Full control over concurrency model
- Aggregation happens at the right level with full context

**Cons:**
- Must construct `EvaluatorContext` manually (coupling to pydantic_evals internals)
- Must handle task errors, evaluator errors ourselves
- Lose pydantic_evals retry logic, progress bars, lifecycle hooks
- More code to maintain

**Mitigations:**
- `EvaluatorContext` is a simple dataclass — coupling risk is low
- Error handling is straightforward (try/except, already modeled in current code via `_handle_task_failures`)
- Can add retries/progress bars incrementally if needed
- Less total code than current approach (eliminate post-processing)

**Changes required:**
| File | What |
|------|------|
| `base.py` | Add `repeat`, `threshold` to `TestCaseMetadata` |
| `types.py` | New: `RunResult`, `AggregatedResult`, `CaseResult`, `EvaluationOutput` |
| `csv/testset.py` | Parse `repeat`/`threshold` columns |
| `mlflow_helper.py` | Replace `evaluate_testset_with_mlflow` with new orchestration. `task`/`task_factory` parameters. Per-run factory invocation. New functions: `_execute_case`, `_execute_run`, `_aggregate_runs`. Refactored DataFrame + upload. |
| `evaluators.py` | Potentially no changes (evaluators receive pristine inputs, `SpanBaseEvaluator` searches by run-specific input_key) |
| `docs/` | New guide, tutorial, how-to for repeated runs + task factory (see Documentation Plan) |

---

## Approach Comparison

| Dimension | A: Expand Dataset | B: Native Repeat | C: Own Loop |
|-----------|-------------------|-------------------|-------------|
| Per-case repeat | Yes (via expansion) | Yes (per-case evaluate) | Yes (native) |
| Threshold | Custom aggregation | Custom on top of aggregate | Custom aggregation |
| Input purity | No (`RepeatedInput` wrapper) | Yes | Yes |
| MLflow hierarchy | Flat (tagged) | Flat (tagged) | Hierarchical (case → runs) |
| Evaluator changes | Yes (unwrap inputs) | No | No |
| Cross-case concurrency | Yes (pydantic_evals) | No (sequential per case) | Yes (semaphore) |
| Task factory support | Awkward (pydantic_evals calls task) | Awkward (same) | Native (we call the factory) |
| State isolation | Hard (pydantic_evals owns the loop) | Partial (new evaluate() per case, but same task within) | Full (factory per run) |
| pydantic_evals coupling | High (expand + evaluate) | Medium (per-case evaluate) | Low (only evaluator API) |
| Complexity | Medium | Medium | Higher initial, lower long-term |
| Eliminated post-processing | None | Some | Most (trace cleanup, key mapping) |

---

## Recommendation: Approach C

Approach C is the cleanest design because:

1. **No wrapper types** — the fundamental data model stays clean
2. **Hierarchical MLflow traces** — matches the user's mental model (case with sub-runs)
3. **Less total code** — eliminates the trace cleanup, key mapping, and metadata merging dance
4. **Natural aggregation** — threshold logic lives where the execution happens, not as post-processing
5. **Evaluators stay unchanged** — `ctx.inputs` is the real input, `SpanBaseEvaluator` works because each run has a unique `input_key` on its span

The cost (constructing `EvaluatorContext`, handling errors) is modest and well-contained.

---

## Detailed Design for Approach C

### New Data Types

```python
# In base.py or a new types.py

@dataclass
class RunResult:
    """Result of a single task run within a case."""
    run_index: int
    input_key: str
    output: Any
    duration: float
    assertions: dict[str, EvaluationResult]  # evaluator_name → result
    evaluator_failures: list[EvaluatorFailure]
    error: Exception | None = None  # if task execution failed
    
    @property
    def all_passed(self) -> bool:
        if self.error:
            return False
        return all(r.value is True for r in self.assertions.values())


@dataclass
class AggregatedResult:
    """Aggregated result across repeated runs of a case."""
    passed: bool
    pass_rate: float  # successful_runs / total_runs
    threshold: float
    summary: str  # human-readable explanation
    per_evaluator_pass_rates: dict[str, float]  # evaluator_name → pass rate


@dataclass
class CaseResult:
    """Complete result for a test case, including all runs and aggregation."""
    case_name: str
    inputs: Any
    metadata: TestCaseMetadata
    base_input_key: str
    trace_id: str
    run_results: list[RunResult]
    aggregated: AggregatedResult
```

### Aggregation Logic

```python
def _aggregate_runs(
    run_results: list[RunResult],
    threshold: float,
) -> AggregatedResult:
    total = len(run_results)
    successful = sum(1 for r in run_results if r.all_passed)
    pass_rate = successful / total
    passed = pass_rate >= threshold
    
    # Per-evaluator pass rates
    evaluator_names = set()
    for r in run_results:
        evaluator_names.update(r.assertions.keys())
    
    per_eval = {}
    for name in evaluator_names:
        eval_passes = sum(
            1 for r in run_results
            if name in r.assertions and r.assertions[name].value is True
        )
        per_eval[name] = eval_passes / total
    
    # Build explanation
    lines = [f"{successful}/{total} runs passed (threshold: {threshold:.0%}). Case {'PASSED' if passed else 'FAILED'}."]
    
    if not passed:
        for r in run_results:
            if not r.all_passed:
                failed_evals = [
                    f"  - {name}: '{res.reason}'"
                    for name, res in r.assertions.items()
                    if res.value is not True
                ]
                if r.error:
                    failed_evals.append(f"  - Task error: {r.error}")
                lines.append(f"Run {r.run_index} FAILED:")
                lines.extend(failed_evals)
    
    return AggregatedResult(
        passed=passed,
        pass_rate=pass_rate,
        threshold=threshold,
        summary="\n".join(lines),
        per_evaluator_pass_rates=per_eval,
    )
```

### MLflow Trace Structure

Each test case produces exactly **one MLflow trace**:

```
Trace
└── "case: What is the capital of France?" (CHAIN)
    │ attributes: input_key, repeat, threshold
    │ assessments: [aggregated pass/fail per evaluator]
    │
    ├── "run_0" (TASK)
    │   │ attributes: input_key=abc123_0, run_index=0
    │   ├── (autologged agent spans)
    │   └── assessments: [LLMJudge: pass, Regex: pass]
    │
    ├── "run_1" (TASK)
    │   │ attributes: input_key=abc123_1, run_index=1
    │   ├── (autologged agent spans)
    │   └── assessments: [LLMJudge: fail, Regex: pass]
    │
    └── "run_2" (TASK)
        │ attributes: input_key=abc123_2, run_index=2
        ├── (autologged agent spans)
        └── assessments: [LLMJudge: pass, Regex: pass]
```

When `repeat=1`, the structure simplifies to what we have today (case span wraps a single run span — could even flatten to just the run span).

### Input Key Scheme

Always: `input_key = f"{base_hash}_{run_index}"`

No special-casing for index 0. Consistent, simple, greppable.

`case_id` in the DataFrame = `base_hash` (groups all runs of the same case).

### DataFrame Design

Two options for the output DataFrame:

#### Option 1: Single DataFrame with level column

```
| level          | case_id | run_index | evaluator_name | result | reason        | ... |
|----------------|---------|-----------|----------------|--------|---------------|-----|
| run            | abc123  | 0         | LLMJudge       | True   | "mentions..." | ... |
| run            | abc123  | 0         | Regex          | True   | "found"       | ... |
| run            | abc123  | 1         | LLMJudge       | False  | "missing..."  | ... |
| run            | abc123  | 1         | Regex          | True   | "found"       | ... |
| case_aggregate | abc123  | -         | LLMJudge       | True   | "2/3 passed"  | ... |
| case_aggregate | abc123  | -         | Regex          | True   | "3/3 passed"  | ... |
| case_aggregate | abc123  | -         | _overall       | True   | "2/3 runs..." | ... |
```

Usage:
```python
runs = df[df["level"] == "run"]
summary = df[df["level"] == "case_aggregate"]
```

#### Option 2: Return a result object with separate views (RECOMMENDED)

```python
@dataclass
class EvaluationOutput:
    """Returned by evaluate_testset_with_mlflow."""
    
    runs: pd.DataFrame        # one row per (run × evaluator) — detailed
    cases: pd.DataFrame       # one row per (case × evaluator) — aggregated
    case_results: list[CaseResult]  # full structured data
    
    @property
    def summary(self) -> pd.DataFrame:
        """One row per case with overall pass/fail."""
        ...
```

**`runs` DataFrame columns:**
```
case_id, case_name, run_index, repeat_total, threshold,
inputs, output, evaluator_name, evaluator_result, evaluator_reason,
evaluator_data, expected, attributes, tags, task_duration,
source_type, source_id, input_key, trace_id
```

**`cases` DataFrame columns:**
```
case_id, case_name, repeat_total, threshold,
inputs, evaluator_name, pass_rate, passed,
aggregated_reason, expected, attributes, tags,
avg_task_duration, trace_id
```

This is cleaner than mixing levels in one DataFrame. Users get the view they need without filtering.

### SpanBaseEvaluator Adaptation

With Approach C, `SpanBaseEvaluator.get_trace()` needs minimal changes. The evaluator runs within the run span context, so the current MLflow trace contains the run's spans. The `input_key` attribute on the run span is unique (`{hash}_{run_index}`).

The main thing: `get_trace()` currently searches all traces in the experiment for a matching `input_key`. With hierarchical traces, the trace structure changes (the root span is the case span, not the run span). Two options:

**Option i:** Search for traces where ANY span (not just root) has the matching `input_key`. This handles both flat and hierarchical structures.

**Option ii:** Pass the trace/span reference directly through the EvaluatorContext (e.g., via the `attributes` dict). This avoids the search entirely.

Option ii is cleaner — since we control the execution loop, we can inject the current trace reference into the context:

```python
ctx = EvaluatorContext(
    ...
    attributes={"_current_trace": trace, "_current_run_span": run_span},
)
```

Then `SpanBaseEvaluator` can use `ctx.attributes["_current_trace"]` directly instead of searching.

### CSV Integration

`repeat` and `threshold` are **case-level** columns. In the CSV, they're specified per row but must be consistent within a question group (like `expected` promotes to case metadata when shared).

**Treatment:** Add them as standard columns (not custom attributes). In `_create_case_from_rows`, extract from the first row and validate consistency.

```csv
Question,test_type,expected,tags,check,repeat,threshold
What is X?,LLMJudge,true,factual,"mentions X",3,0.6
What is X?,RegexInOutput,true,factual,"X",3,0.6
What is Y?,LLMJudge,true,factual,"mentions Y",,
```

Empty `repeat`/`threshold` → defaults (1 and 1.0). All rows for "What is X?" must agree on repeat/threshold.

---

## File Change Summary (Approach C)

| File | Change | Description |
|------|--------|-------------|
| `src/ragpill/base.py` | Modify | Add `repeat`, `threshold` to `TestCaseMetadata` |
| `src/ragpill/types.py` | **New** | `RunResult`, `AggregatedResult`, `CaseResult`, `EvaluationOutput` |
| `src/ragpill/csv/testset.py` | Modify | Parse `repeat`/`threshold` columns, add to `TestCaseMetadata` |
| `src/ragpill/mlflow_helper.py` | **Rewrite** | Replace `evaluate_testset_with_mlflow` internals: new execution loop, aggregation, hierarchical spans. Remove `_expand_testset_for_repeats` (not needed). Simplify trace handling (no cleanup step, no key mapping). |
| `src/ragpill/evaluators.py` | Minor | `SpanBaseEvaluator.get_trace()` — use injected trace ref or search nested spans |
| `tests/` | Add/Modify | Tests for repeat, threshold, aggregation, hierarchical traces |

## Implementation Order

1. `base.py` + `types.py` — data model foundations (`repeat`, `threshold`, result types)
2. `csv/testset.py` — CSV parsing for new columns
3. `mlflow_helper.py` — new execution loop with `task`/`task_factory`, per-run factory invocation, hierarchical MLflow spans, aggregation, DataFrame construction, upload
4. `evaluators.py` — SpanBaseEvaluator adaptation (injected trace ref)
5. Tests — unit (aggregation, threshold, factory) + integration (MLflow traces, DataFrame output)
6. Docs — guide, tutorial, how-to, API reference updates, docstrings

## Open Questions

1. **EvaluatorContext construction:** Need to verify exact constructor signature in the installed pydantic_evals version. The `_span_tree` field may need a sentinel value when we don't capture OTEL spans ourselves.

2. **Concurrency model:** Should runs within a case be sequential or concurrent? Sequential is simpler and gives deterministic ordering. Concurrent is faster but runs may interfere (e.g., shared rate limits). Recommend: sequential within case, concurrent across cases.

3. **Single-run optimization:** When `repeat=1`, should we flatten the trace structure (no case wrapper span, just the run span like today)? Or always use the hierarchical structure for consistency? Recommend: always hierarchical for consistency — the extra span level has negligible overhead and simplifies the code.

4. **Return type change:** Currently returns `pd.DataFrame`. Approach C recommends returning `EvaluationOutput`. This is a breaking change — acceptable per user's "backward-compatibility is no issue" directive.

5. **Evaluator state isolation:** Evaluators are also potentially stateful (e.g., `SpanBaseEvaluator` caches MLflow clients). Currently evaluator instances are shared across cases. With our own loop we could deep-copy evaluators per run — but this is likely overkill since evaluators are designed to be stateless per-call. Monitor and revisit if needed.

---

## Documentation Plan

Documentation is critical for this feature — it changes the public API, introduces a new concept (task factory), and requires users to reason about state isolation. The docs should be layered: quick examples for the common case, deep explanation for the advanced case.

### Docs to Create/Update

#### 1. API Reference: `evaluate_testset_with_mlflow` (update existing)

**File:** `docs/api/mlflow.md`

Update the function signature docs to cover:
- `task` parameter — when to use, statelessness assumption
- `task_factory` parameter — when to use, what it does, invocation policy (per-run)
- New return type `EvaluationOutput` — `.runs`, `.cases`, `.summary` DataFrames
- `repeat` and `threshold` on `TestCaseMetadata`

#### 2. API Reference: `TestCaseMetadata` (update existing)

**File:** `docs/api/base.md`

Add `repeat` and `threshold` fields with descriptions and examples.

#### 3. Guide: Repeated Runs (new page)

**File:** `docs/guide/repeated-runs.md`

Structure:

```markdown
# Repeated Runs

## Why Repeat?

LLMs are stochastic. A single run may pass or fail by chance. Repeating 
runs and applying a threshold gives statistical confidence.

## Quick Start

### Stateless task (most common)

Most tasks are stateless — calling them with the same input produces 
independent results each time. pydantic-ai's `Agent.run()` is stateless 
by default.

​```python
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o", system_prompt="You are helpful.")

async def my_task(question: str) -> str:
    result = await agent.run(question)
    return result.output

testset = Dataset(cases=[
    Case(
        inputs="What is the capital of France?",
        metadata=TestCaseMetadata(repeat=5, threshold=0.8),
        evaluators=[LLMJudge(rubric="Answer mentions Paris")],
    ),
])

results = await evaluate_testset_with_mlflow(
    testset=testset,
    task=my_task,  # stateless — safe to reuse
)

# Check case-level results (aggregated across 5 runs)
print(results.cases)

# Drill into individual runs
print(results.runs)
​```

### Stateful task (agents with memory)

If your agent holds state between calls (conversation history, caches,
mutable config), each repeated run would see the previous run's state. 
This defeats the purpose of repeating.

Use `task_factory` to create a fresh agent per run:

​```python
from pydantic_ai import Agent

def create_agent():
    """Factory: returns a fresh agent each time."""
    agent = Agent(
        model="openai:gpt-4o",
        system_prompt="You are helpful.",
    )
    async def run(question: str) -> str:
        result = await agent.run(question)
        return result.output
    return run

results = await evaluate_testset_with_mlflow(
    testset=testset,
    task_factory=create_agent,  # fresh agent per run
)
​```

### When do I need a factory?

| Your setup | Use `task=` | Use `task_factory=` |
|------------|-------------|---------------------|
| pydantic-ai `Agent.run()` (no message_history) | Yes | Not needed |
| pydantic-ai with `message_history` persistence | No | Yes |
| LangChain agent with `ConversationMemory` | No | Yes |
| Stateless function (API wrapper, etc.) | Yes | Not needed |
| Custom class with `self.state` | No | Yes |
| RAG pipeline with ephemeral retriever | Yes | Not needed |
| RAG pipeline with caching retriever | Depends | Yes, if cache affects results |

**Rule of thumb:** If calling your task twice with the same input produces 
different results *because of state accumulation* (not LLM randomness), 
use `task_factory`.

## Threshold Semantics

- A run is "successful" if **all** evaluators on that run pass
- The case passes if `successful_runs / total_runs >= threshold`
- `threshold=1.0` (default): all runs must pass
- `threshold=0.6`: at least 60% of runs must pass
- `threshold=0.0`: case always passes (useful for "just gather data" mode)

## Reading the Results

​```python
results = await evaluate_testset_with_mlflow(...)

# High-level: one row per (case × evaluator), aggregated
results.cases
#   case_id | evaluator_name | pass_rate | passed | aggregated_reason | ...

# Detailed: one row per (run × evaluator)
results.runs
#   case_id | run_index | evaluator_name | evaluator_result | evaluator_reason | ...

# Summary: one row per case, overall pass/fail
results.summary
#   case_id | case_name | passed | pass_rate | threshold | summary | ...
​```

## MLflow Trace Structure

Each case produces one trace with nested spans:

​```
Trace: "case: What is the capital of France?"
├── run_0: agent spans + evaluator results
├── run_1: agent spans + evaluator results
└── run_2: agent spans + evaluator results
​```

Aggregated assessments (pass rate per evaluator) are attached to the 
case-level span. Individual assessments are on each run span.

## CSV Integration

Add `repeat` and `threshold` columns to your CSV:

​```csv
Question,test_type,expected,tags,check,repeat,threshold
What is X?,LLMJudge,true,factual,"mentions X",5,0.8
What is X?,RegexInOutput,true,factual,"X",5,0.8
What is Y?,LLMJudge,true,geography,"mentions Y",,
​```

- `repeat` and `threshold` are case-level — all rows for the same 
  question must have the same values
- Empty values default to `repeat=1`, `threshold=1.0`

## Failure Explanations

When a case fails, the aggregated reason includes per-run details:

​```
2/5 runs passed (threshold: 80%). Case FAILED.
Run 1 FAILED:
  - LLMJudge: 'Answer did not mention the capital city'
  - RegexInOutput: 'Pattern "Paris" not found'
Run 3 FAILED:
  - LLMJudge: 'Answer was incomplete — mentioned Paris but not France'
Run 4 FAILED:
  - Task error: TimeoutError('LLM call timed out after 30s')
​```
```

#### 4. Tutorial: Repeated Runs End-to-End (new page)

**File:** `docs/tutorials/repeated-runs.md`

A complete notebook-style walkthrough:
1. Define a testset with varying repeat/threshold per case
2. Run with a stateless task, inspect results
3. Run with a task_factory (stateful agent), compare
4. Show the MLflow UI: how traces look, how to navigate runs within a case
5. Show DataFrame analysis: grouping, filtering, computing custom metrics
6. Show a failing case: read the aggregated explanation, drill into specific run

#### 5. How-To: Migrate from `task` to `task_factory` (new page)

**File:** `docs/how-to/task-factory.md`

Short, recipe-style:

```markdown
# Using Task Factories for Stateful Agents

## Problem
Your agent has conversation memory or mutable state. Repeated runs 
contaminate each other.

## Solution
Wrap your agent construction in a factory function.

### Before (broken with repeat)
​```python
agent = MyStatefulAgent(model="gpt-4o")

async def task(q: str) -> str:
    return await agent.ask(q)  # agent remembers previous calls!

results = await evaluate_testset_with_mlflow(testset, task=task)
​```

### After (correct)
​```python
def create_task():
    agent = MyStatefulAgent(model="gpt-4o")  # fresh agent
    async def task(q: str) -> str:
        return await agent.ask(q)
    return task

results = await evaluate_testset_with_mlflow(testset, task_factory=create_task)
​```

### With pydantic-ai Agent + message_history
​```python
def create_task():
    agent = Agent(model="openai:gpt-4o", system_prompt="...")
    history = []  # fresh history per run
    
    async def task(q: str) -> str:
        result = await agent.run(q, message_history=history)
        history.extend(result.new_messages())
        return result.output
    return task
​```

### With resource cleanup
​```python
def create_task():
    db = DatabaseConnection()  # opened fresh
    agent = Agent(model="openai:gpt-4o", system_prompt="...")
    
    async def task(q: str) -> str:
        result = await agent.run(q, deps=db)
        return result.output
    return task
    # Note: db will be garbage-collected after the run
    # For explicit cleanup, consider async context manager patterns
​```
```

#### 6. Update Existing Pages

| Page | Changes |
|------|---------|
| `docs/getting-started/quickstart.md` | Add a "Repeated Runs" section with minimal example |
| `docs/guide/overview.md` | Mention repeat/threshold in the flow diagram |
| `docs/guide/testsets.md` | Document `repeat`/`threshold` as TestCaseMetadata fields |
| `docs/guide/csv-adapter.md` | Add `repeat`/`threshold` columns to CSV format docs |
| `docs/tutorials/full.md` | Add a section showing repeat with the existing example |

### Documentation Principles

1. **Lead with the simple case.** `task=my_func` with `repeat=3, threshold=0.8` should be explainable in 5 lines. Factory comes second.
2. **Show, don't tell.** Every concept gets a runnable code example.
3. **Decision table over prose.** The "when do I need a factory?" table is more useful than paragraphs.
4. **Show the output.** Include example DataFrame output and MLflow screenshots.
5. **Explain failures.** Show what a failed aggregation looks like — this is what users will actually need to debug.

### Docstring Updates

All public API changes need updated docstrings:
- `evaluate_testset_with_mlflow()` — full Args/Returns update with examples
- `evaluate_testset_with_mlflow_sync()` — mirror the async version
- `TestCaseMetadata` — document `repeat` and `threshold` fields
- `EvaluationOutput` — document `.runs`, `.cases`, `.summary`
- `RunResult`, `AggregatedResult`, `CaseResult` — all fields documented

---

## File Change Summary (Approach C) — Updated

| File | Change | Description |
|------|--------|-------------|
| `src/ragpill/base.py` | Modify | Add `repeat`, `threshold` to `TestCaseMetadata` |
| `src/ragpill/types.py` | **New** | `RunResult`, `AggregatedResult`, `CaseResult`, `EvaluationOutput` |
| `src/ragpill/csv/testset.py` | Modify | Parse `repeat`/`threshold` columns, add to `TestCaseMetadata` |
| `src/ragpill/mlflow_helper.py` | **Rewrite** | New execution loop with `task`/`task_factory`, per-run factory invocation, hierarchical MLflow spans, aggregation, DataFrame construction, upload |
| `src/ragpill/evaluators.py` | Minor | `SpanBaseEvaluator.get_trace()` — use injected trace ref or search nested spans |
| `tests/` | Add/Modify | Tests for repeat, threshold, aggregation, factory pattern, hierarchical traces |
| `docs/guide/repeated-runs.md` | **New** | Guide covering repeat, threshold, task vs factory, result interpretation |
| `docs/tutorials/repeated-runs.md` | **New** | End-to-end tutorial with stateless + stateful examples |
| `docs/how-to/task-factory.md` | **New** | Recipe for migrating stateful agents to factory pattern |
| `docs/getting-started/quickstart.md` | Update | Add minimal repeat example |
| `docs/guide/overview.md` | Update | Mention repeat in flow diagram |
| `docs/guide/testsets.md` | Update | Document new TestCaseMetadata fields |
| `docs/guide/csv-adapter.md` | Update | Add repeat/threshold CSV columns |
| `docs/tutorials/full.md` | Update | Add repeat section to existing tutorial |
