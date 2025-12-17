# Design: Concurrent Scheduling for Runs & Evaluations

**Status:** Draft
**Date:** 2026-04-27

---

## 1. Motivation

The execution and evaluation phases are written with `async def` everywhere but contain no actual concurrency. Three nested serial `for` loops walk the case/repeat/evaluator tree depth-first, so the codebase pays async overhead without getting the benefit. On LLM-API-bound workloads this is the dominant source of wall-clock time.

The trigger for this design was a question about whether ideas from database query schedulers could fundamentally improve scheduling here. The honest answer: a few do, most don't, and the largest win is much simpler than an exotic scheduler.

---

## 2. Current State

Three nested serial loops, all `async`, none concurrent:

1. **Cases** — [execution.py:556-569](../src/ragpill/execution.py#L556-L569)
   ```python
   for case in testset.cases:
       case_output = await _execute_case_runs(...)
   ```
2. **Repeats per case** — [execution.py:387-391](../src/ragpill/execution.py#L387-L391)
   ```python
   for i in range(repeat):
       task_runs.append(await _execute_single_run(...))
   ```
3. **Evaluators per run** — [evaluation.py:154-157](../src/ragpill/evaluation.py#L154-L157)
   ```python
   for evaluator in evaluators:
       result = await evaluator.evaluate(ctx)
   ```
4. **Runs-within-case for eval** — [evaluation.py:362-365](../src/ragpill/evaluation.py#L362-L365) — same pattern.

In addition, one MLflow `search_traces` per case ([execution.py:396](../src/ragpill/execution.py#L396)) sits on the critical path as a serial network round-trip.

---

## 3. What Translates From DB Scheduling, What Doesn't

Ragpill is async I/O bound (LLM and MLflow calls), not CPU-bound query planning. Several DB-scheduler ideas map cleanly; others don't.

| DB scheduling idea | Applies here? | Concretely |
|---|---|---|
| **Pipelining / overlapping I/O** | Strong fit | `gather` across cases + repeats. Likely 3–10× on typical workloads with zero new machinery. |
| **Admission control** (bounded concurrency) | Strong fit | LLM providers will 429 if 50 cases × 3 repeats hit at once. A per-provider `Semaphore` is the right primitive. |
| **Cost-based ordering** (cheap predicates first) | Good fit for evaluators | Run regex/heuristic evaluators before LLM judges; short-circuit on assertion failure for fail-fast modes. |
| **Operator fusion / batching** | Only if providers support it | Multiple evaluator calls to the same model could be batched into one request, but needs prompt-level design. |
| **Prefetch** | Modest win | Kick off `search_traces` for case N while case N+1 executes. |
| **DAG scheduler / dependency tracking** | Overkill now | Evaluators are independent. Don't build a DAG runtime until a second producer-consumer edge exists. |
| **Work-stealing across workers** | Wrong tool | Single-process async — a `Semaphore`-gated `gather` *is* the work-stealer you need. |

---

## 4. Proposed Changes

Three concrete changes, ordered by impact-to-risk ratio.

### 4.1 Replace serial loops with bounded `gather`

Introduce two semaphores at the dataset-run boundary:

- `case_concurrency` — caps how many cases execute in parallel
- `provider_concurrency` — caps concurrent calls per LLM provider (passed to task and evaluators)

```python
case_sem = asyncio.Semaphore(settings.case_concurrency)

async def _bounded_case(case):
    async with case_sem:
        return await _execute_case_runs(...)

case_outputs = await asyncio.gather(*[_bounded_case(c) for c in testset.cases])
```

Same shape inside `_execute_case_runs` for the `repeat` loop, and inside `evaluate_dataset_run` for both the case-level and evaluator-level loops.

### 4.2 Cost-based evaluator ordering

In `_evaluate_single_run` ([evaluation.py:154-177](../src/ragpill/evaluation.py#L154-L177)), order evaluators cheap-first. Either:

- a static `cost` hint on `BaseEvaluator` (e.g. `cost: Literal["cheap", "expensive"] = "cheap"`, with LLM-judge subclasses overriding to `"expensive"`), or
- a heuristic: evaluators that don't override `__init__` with an LLM client are cheap.

Cheap evaluators run first; if a fail-fast mode is enabled and a cheap one fails, expensive ones are skipped. Without fail-fast, ordering still matters because parallelizing evaluators is bounded by `provider_concurrency` — running cheap ones first lets results stream in.

### 4.3 Move trace fetching off the critical path

Two options, in increasing complexity:

- **Per-case async fetch** — kick off `_fetch_trace` as a task at the end of `_execute_case_runs` and `await` it later when assembling `DatasetRunOutput`. Overlaps trace I/O with the next case's execution.
- **Batched fetch** — collect all `parent_trace_id`s and issue one MLflow query at the end. Requires confirming MLflow's `search_traces` supports filtering by multiple IDs.

Start with the per-case async fetch — it's a one-line change with `asyncio.create_task`.

---

## 5. Risks & Open Questions

- **MLflow span context across `gather`.** The parent span at [execution.py:381](../src/ragpill/execution.py#L381) wraps the serial repeat loop. MLflow tracing uses `contextvars`, which propagate correctly across `gather`, but this needs a smoke test before declaring done — concurrent child spans under one parent is the exact scenario to verify.
- **Provider rate limits.** Adding parallelism without `provider_concurrency` will trigger 429s. The semaphore must default to a conservative value (e.g. 4–8) and be configurable per provider.
- **Determinism.** Output order changes from input order under `gather`. Any code that assumes positional alignment between `testset.cases` and `dataset_run.cases` (e.g. [evaluation.py:349](../src/ragpill/evaluation.py#L349)) must be checked. The current `zip` works only because order is preserved — `asyncio.gather` also preserves order, so this should hold, but it's worth an explicit assertion.
- **Backpressure on MLflow tracking server.** Concurrent runs hitting MLflow simultaneously may saturate the server in self-hosted setups. Consider a separate `mlflow_concurrency` semaphore if this surfaces.
- **Error semantics.** `gather` with `return_exceptions=False` cancels siblings on first failure. That's probably wrong for batch evaluation — one bad case shouldn't cancel the others. Use `return_exceptions=True` and surface failures in the result.

---

## 6. Settings Surface

Add to `MLFlowSettings` (or a new `SchedulingSettings`):

```python
case_concurrency: int = 4
repeat_concurrency: int = 4         # within a case
evaluator_concurrency: int = 4      # within a run
provider_concurrency: dict[str, int] = {"default": 8}
fail_fast_evaluators: bool = False
```

Defaults should be conservative enough that the existing serial behavior is roughly preserved at low values, and the user opts into more parallelism explicitly.

---

## 7. Out of Scope

- Cross-process or distributed scheduling.
- Persistent queues / retry on restart.
- Cost-based scheduling that uses *measured* historical evaluator latencies (the static cheap/expensive split is enough for v1).
- DAG-shaped evaluator dependencies — revisit when a real producer-consumer edge appears.
