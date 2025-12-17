# Repeated Runs

LLM outputs are stochastic. The same input can produce different answers across runs. A single evaluation run might pass or fail by chance. Repeated runs give you statistical confidence that your agent's behavior is reliable.

## Quick Start

Run each test case 3 times and require at least 80% of runs to pass:

```python
from ragpill import evaluate_testset_with_mlflow
from ragpill.settings import MLFlowSettings
from pydantic_evals import Case, Dataset
from ragpill.base import TestCaseMetadata
from ragpill.evaluators import RegexInOutputEvaluator

evaluator = RegexInOutputEvaluator(pattern="paris", expected=True)
case = Case(
    inputs="What is the capital of France?",
    metadata=TestCaseMetadata(repeat=3, threshold=0.8),
    evaluators=[evaluator],
)
testset = Dataset(cases=[case])

result = await evaluate_testset_with_mlflow(
    testset=testset,
    task=my_agent,
    mlflow_settings=MLFlowSettings(),
)

# Three views of the data:
print(result.runs)     # One row per (run x evaluator) — most granular
print(result.cases)    # One row per (case x evaluator) — aggregated
print(result.summary)  # One row per case — overall pass/fail
```

## How It Works

When `repeat > 1`, ragpill uses a **two-phase execution** model for each test case:

**Phase 1 (Task Execution):** The task is executed N times inside an MLflow span context. Each run gets its own child span under a shared parent. All spans are committed to MLflow when Phase 1 completes.

**Phase 2 (Evaluation):** After all task spans are committed, evaluators run for each run individually. A `ContextVar` ensures span-based evaluators only see spans from their specific run, not from other runs.

```
Case: "What is the capital of France?" (repeat=3)
├── Phase 1: Execute task
│   ├── run-0: task("What is...") → "Paris is the capital"
│   ├── run-1: task("What is...") → "The capital is Paris"
│   └── run-2: task("What is...") → "France's capital: Paris"
└── Phase 2: Evaluate
    ├── run-0: RegexInOutput("paris") → True
    ├── run-1: RegexInOutput("paris") → True
    └── run-2: RegexInOutput("paris") → True
→ pass_rate = 3/3 = 1.0 ≥ threshold 0.8 → PASSED
```

## Stateful Tasks (task_factory)

If your task is stateful (e.g., an agent with message history), use `task_factory` instead of `task` to ensure each run starts with a clean state:

```python
def create_agent():
    """Return a fresh agent instance with empty history."""
    return MyAgent(history=[])

result = await evaluate_testset_with_mlflow(
    testset=testset,
    task_factory=create_agent,
)
```

See the [Task Factory How-To](../how-to/task-factory.md) for detailed guidance.

!!! warning "When do I need a factory?"
    | Scenario | Use `task=` | Use `task_factory=` |
    |----------|:-----------:|:-------------------:|
    | Stateless function (no side effects) | Yes | |
    | Agent with message history | | Yes |
    | Agent with mutable configuration | | Yes |
    | Pure function + repeat=1 | Yes | |

## Threshold Semantics

The `threshold` parameter controls how pass/fail is decided:

- **threshold=1.0** (default): All runs must pass. A single failure means the case fails.
- **threshold=0.0**: The case always passes regardless of run results.
- **threshold=0.8**: At least 80% of runs must pass.
- The comparison is `pass_rate >= threshold`, so `threshold=0.6` with 2/3 runs passing (0.667) is a pass.

A run counts as "passed" when **all** its evaluators pass. If any evaluator fails, the entire run is considered failed.

## Reading the Results

`evaluate_testset_with_mlflow` returns an `EvaluationOutput` with three DataFrame views:

### `.runs` — Per-run detail

One row per (run x evaluator). Includes `run_index`, `repeat_total`, `threshold`, and all the standard columns (`evaluator_result`, `evaluator_reason`, etc.).

### `.cases` — Aggregated per case

One row per (case x evaluator). Includes `pass_rate` and `passed` columns showing the aggregated result across runs.

### `.summary` — Overall verdict

One row per case with `passed`, `pass_rate`, `threshold`, and a human-readable `summary` string.

## Per-Case Overrides vs. Global Defaults

You can set `repeat` and `threshold` at two levels:

**Global defaults** via `MLFlowSettings`:

```python
settings = MLFlowSettings(ragpill_repeat=3, ragpill_threshold=0.8)
```

Or via environment variables:

```bash
export MLFLOW_RAGPILL_REPEAT=3
export MLFLOW_RAGPILL_THRESHOLD=0.8
```

**Per-case overrides** via `TestCaseMetadata`:

```python
case = Case(
    inputs="...",
    metadata=TestCaseMetadata(repeat=5, threshold=0.6),
    evaluators=[...],
)
```

Per-case values take precedence over global defaults. When a per-case value is `None`, the global default is used.

## CSV Integration

Add `repeat` and `threshold` columns to your CSV:

```csv
Question,test_type,expected,tags,check,repeat,threshold
What is X?,RegexInOutputEvaluator,true,factual,x,3,0.6
What is Y?,RegexInOutputEvaluator,true,geography,y,,
```

- `What is X?` will run 3 times with threshold 0.6
- `What is Y?` will use global defaults (repeat=1, threshold=1.0 unless overridden)
- Empty values defer to the global `MLFlowSettings` defaults

!!! note
    All rows for the same question must have the same `repeat` and `threshold` values. Inconsistent values will raise a `ValueError`.

## MLflow Assessment Naming

When viewing traces in the MLflow UI, assessments follow this naming convention:

- **Per-run:** `run-0_RegexInOutput`, `run-1_RegexInOutput`, `run-2_RegexInOutput`
- **Aggregate** (only when repeat > 1): `agg_RegexInOutput`

The aggregate assessment value is `pass_rate >= threshold`, with a rationale like `"Aggregate: 2/3 runs passed (threshold=0.6)"`.

## Failure Explanations

When a case fails, the `summary` field includes details about which runs failed and why:

```
1/3 runs passed (threshold=0.8). Failed: run-1: RegexInOutput: pattern "paris" did not match output; run-2: task error: TimeoutError
```
