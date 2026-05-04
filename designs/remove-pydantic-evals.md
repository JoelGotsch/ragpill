# Design: Remove pydantic-evals Dependency

**Status:** Draft  
**Date:** 2026-04-16  
**Context:** ragpill currently imports types and functions from `pydantic_evals` (shipped as part of `pydantic-ai`). The goal is to eliminate all `pydantic_evals` imports so ragpill owns its evaluation primitives and is decoupled from upstream changes in pydantic-evals.

---

## 1. Current Dependency Surface

### 1.1 Imported Types (8 symbols)

| Symbol | Source module | Used in |
|--------|-------------|---------|
| `Evaluator` | `pydantic_evals.evaluators.evaluator` | `base.py` — `BaseEvaluator(Evaluator)` inheritance |
| `EvaluationReason` | `pydantic_evals.evaluators.evaluator` | `base.py`, `evaluators.py` — return type of `run()` / `evaluate()` |
| `EvaluationResult` | `pydantic_evals.evaluators.evaluator` | `types.py`, `mlflow_helper.py` — stored in `RunResult.assertions` |
| `EvaluatorSpec` | `pydantic_evals.evaluators.evaluator` | `mlflow_helper.py` — `source` field of `EvaluationResult` |
| `EvaluatorContext` | `pydantic_evals.evaluators.context` | `base.py`, `evaluators.py`, `mlflow_helper.py` — parameter to evaluator methods |
| `Case` | `pydantic_evals` | `csv/testset.py`, `mlflow_helper.py` — test case container |
| `Dataset` | `pydantic_evals` | `csv/testset.py`, `mlflow_helper.py`, `utils.py` — testset container |
| `judge_input_output`, `judge_output` | `pydantic_evals.evaluators.llm_as_a_judge` | `evaluators.py` — LLMJudge evaluator |

### 1.2 Usage Patterns

**Inheritance:** `BaseEvaluator` inherits from `Evaluator`. The inherited surface used:
- `get_serialization_name()` — returns `cls.__name__`
- `build_serialization_arguments()` — iterates dataclass fields, returns non-default values as dict
- Abstract `evaluate()` — completely overridden

**Data containers:** `Case` and `Dataset` used as typed containers. We construct them in `csv/testset.py` and iterate them in `mlflow_helper.py`. We never call `dataset.evaluate()` — our own orchestration loop replaced it.

**Context object:** `EvaluatorContext` constructed manually in `mlflow_helper.py:_evaluate_run()`. We pass `_span_tree=None` since we use MLflow tracing, not OpenTelemetry.

**Result types:** `EvaluationResult` and `EvaluatorSpec` store evaluator output and source provenance. `EvaluationReason` is the return type of evaluators.

**LLM judge:** `judge_output()` and `judge_input_output()` are high-level functions that internally use a `pydantic_ai.Agent` with a structured prompt. They return `GradingOutput(reason, pass_, score)`.

### 1.3 Files That Import pydantic_evals

| File | Imports |
|------|---------|
| `src/ragpill/base.py` | `EvaluatorContext`, `EvaluationReason`, `Evaluator` |
| `src/ragpill/evaluators.py` | `EvaluationReason`, `Evaluator`, `EvaluatorContext`, `judge_input_output`, `judge_output` |
| `src/ragpill/mlflow_helper.py` | `Case`, `Dataset`, `EvaluationResult`, `EvaluatorSpec`, `EvaluatorContext` |
| `src/ragpill/csv/testset.py` | `Case`, `Dataset` |
| `src/ragpill/types.py` | `EvaluationResult` |
| `src/ragpill/utils.py` | `Dataset` |
| Tests (7 files) | Various combinations of the above |
| Docs (6 files) | Code examples with pydantic_evals imports |

---

## 2. Replacement Strategy

### 2.1 Overview

Every pydantic_evals type is either a simple dataclass or a thin container. None of them carry complex behavior that would be costly to reimplement. The strategy is:

1. **Create `src/ragpill/eval_types.py`** — local definitions for all evaluation primitives
2. **Rewrite `BaseEvaluator`** — standalone dataclass, no external base class
3. **Reimplement LLM judge** — use `pydantic_ai.Agent` directly (same approach pydantic_evals uses internally)
4. **Update all imports** — replace `pydantic_evals.*` with `ragpill.eval_types.*`
5. **Remove `WrappedPydanticEvaluator`** — no longer relevant without pydantic_evals evaluators to wrap

### 2.2 Dependency Change

```toml
# Before
dependencies = [
    "pydantic-ai>=1.39.1",
    "mlflow>=3.8.1",
]

# After
dependencies = [
    "pydantic-ai>=1.39.1",   # kept: OpenAI model wrappers, Agent for LLM judge
    "mlflow>=3.8.1",
]
```

`pydantic-ai` stays as a dependency. We use it for:
- `pydantic_ai.models.Model` type and `OpenAIChatModel` (in `utils.py` and `evaluators.py`)
- `pydantic_ai.Agent` (new: for reimplemented LLM judge)

`pydantic_evals` ships inside the `pydantic-ai` package, so it will still be *installed* — we just stop importing from it.

---

## 3. New Module: `src/ragpill/eval_types.py`

All replacement types live in a single new module. This keeps the diff localized and makes the migration atomic.

### 3.1 EvaluationReason

```python
@dataclass
class EvaluationReason:
    """Result of running an evaluator with optional explanation."""
    value: bool | int | float | str
    reason: str | None = None
```

Drop-in replacement. Identical to the pydantic_evals version.

### 3.2 EvaluatorSource

Replaces `EvaluatorSpec` (aliased from `pydantic_ai._spec.NamedSpec`, a Pydantic model with `name: str` and `arguments: dict | tuple | None`).

```python
@dataclass
class EvaluatorSource:
    """Provenance information for an evaluation result."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
```

Renamed from `EvaluatorSpec` to `EvaluatorSource` since "spec" implies something you construct from (a pydantic_evals serialization concept we don't need). We only use it as provenance metadata in `EvaluationResult.source`.

### 3.3 EvaluationResult

```python
@dataclass
class EvaluationResult:
    """Details of an individual evaluation result."""
    name: str
    value: bool | int | float | str
    reason: str | None
    source: EvaluatorSource
```

Same fields as before, but `source` is now `EvaluatorSource` instead of `EvaluatorSpec`.

### 3.4 EvaluatorContext

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
    attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
```

**Key difference:** No `_span_tree` field. We never used it (always passed `None`). Removing it eliminates the need for `SpanTreeRecordingError` imports and the pyright suppression comment.

### 3.5 Case

```python
@dataclass
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single test case in a dataset."""
    inputs: InputsT
    name: str | None = None
    metadata: MetadataT | None = None
    expected_output: OutputT | None = None
    evaluators: list[Any] = field(default_factory=list)  # list[BaseEvaluator] at runtime
```

Simple dataclass replacing the pydantic_evals `Case`. We only construct these in `csv/testset.py` and iterate them in `mlflow_helper.py`. The `evaluators` field uses `Any` to avoid circular imports (same pattern pydantic_evals uses with generics); runtime asserts already verify `isinstance(ev, BaseEvaluator)`.

### 3.6 Dataset

```python
@dataclass
class Dataset(Generic[InputsT, OutputT, MetadataT]):
    """A collection of test cases with optional global evaluators."""
    cases: list[Case[InputsT, OutputT, MetadataT]]
    evaluators: list[Any] = field(default_factory=list)  # list[BaseEvaluator] at runtime
```

Replaces the Pydantic BaseModel version. We never serialize/deserialize datasets — we construct them in code or from CSV. A plain dataclass is sufficient.

**Note:** `Dataset.evaluate()` is not reimplemented. We stopped using it when we built our own orchestration loop in `mlflow_helper.py`.

---

## 4. BaseEvaluator: Breaking Free from Evaluator

### 4.1 Current Inheritance

```
pydantic_evals.evaluators.evaluator.Evaluator  (abstract)
  └── ragpill.base.BaseEvaluator  (@dataclass)
        ├── LLMJudge
        ├── WrappedPydanticEvaluator
        ├── SpanBaseEvaluator
        │     └── SourcesBaseEvaluator
        │           ├── RegexInSourcesEvaluator
        │           ├── RegexInDocumentMetadataEvaluator
        │           └── LiteralQuoteEvaluator
        ├── RegexInOutputEvaluator
        └── HasQuotesEvaluator
```

### 4.2 What We Actually Use from Evaluator

| Method | What it does | Replacement |
|--------|-------------|-------------|
| `get_serialization_name()` | Returns `cls.__name__` | Copy as `@classmethod` on `BaseEvaluator` |
| `build_serialization_arguments()` | Returns dict of non-default dataclass fields | Copy as method on `BaseEvaluator` |
| `evaluate()` (abstract) | Entry point for evaluation | Already fully overridden in `BaseEvaluator` |
| `evaluate_sync()` / `evaluate_async()` | Sync/async wrappers | Not used — we always call `await evaluator.evaluate(ctx)` |
| `as_spec()` / `serialize()` | Serialization to `EvaluatorSpec` | Not used in ragpill |

### 4.3 New BaseEvaluator

```python
@dataclass
class BaseEvaluator:
    """Base class for all ragpill evaluators."""
    
    evaluation_name: uuid.UUID = field(default_factory=uuid.uuid4)
    expected: bool | None = field(default=None)
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    is_global: bool = field(default=False)
    
    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the class name for identification."""
        return cls.__name__
    
    def build_serialization_arguments(self) -> dict[str, Any]:
        """Return non-default field values as a dict (for serialization/debugging)."""
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value != f.default and value != (f.default_factory() if f.default_factory is not MISSING else MISSING):
                result[f.name] = value
        return result
    
    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> BaseEvaluator:
        # ... unchanged ...
    
    @property
    def metadata(self) -> EvaluatorMetadata:
        # ... unchanged ...
    
    async def run(self, ctx: EvaluatorContext[Any, Any, EvaluatorMetadata]) -> EvaluationReason:
        raise NotImplementedError("Subclasses must implement the run method.")
    
    async def evaluate(self, ctx: EvaluatorContext[Any, Any, EvaluatorMetadata]) -> EvaluationReason:
        # ... unchanged (handles expected logic) ...
```

The `build_serialization_arguments()` reimplementation can be simplified. Currently `LLMJudge` calls `super(BaseEvaluator, self).build_serialization_arguments()` — this needs to be updated to call `super().build_serialization_arguments()` or just `self.build_serialization_arguments()` in the override.

---

## 5. LLM Judge: Reimplementation

### 5.1 Current Dependency

`LLMJudge.run()` calls:
- `judge_input_output(ctx.inputs, ctx.output, self.rubric, self.model)` — when `include_input=True`
- `judge_output(ctx.output, self.rubric, self.model)` — when `include_input=False`

Both return `GradingOutput(reason: str, pass_: bool, score: float)`.

### 5.2 What These Functions Do Internally

The pydantic_evals judge functions:
1. Create a `pydantic_ai.Agent` with a system prompt
2. Build a user prompt with XML tags: `<Input>`, `<Output>`, `<Rubric>`
3. Run the agent to get a structured `GradingOutput` response
4. Return the result

This is ~30 lines of code. The system prompt is:

```
You are an expert evaluator. Given an output (and optionally an input), 
judge whether the output meets the given rubric. Return a JSON with:
- reason: explanation of your judgment
- pass: true/false
- score: 0.0-1.0
```

### 5.3 Replacement: `src/ragpill/llm_judge.py`

Create a new module with our own implementation using `pydantic_ai.Agent` directly:

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, models

class GradingOutput(BaseModel):
    """Structured output from LLM grading."""
    reason: str
    pass_: bool = Field(validation_alias="pass", serialization_alias="pass")
    score: float = Field(ge=0.0, le=1.0)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Your task is to judge whether 
the provided output meets the given rubric criteria.

Evaluate carefully and provide:
- A clear reason for your judgment
- Whether the output passes (true) or fails (false)  
- A confidence score from 0.0 to 1.0"""

async def judge_output(
    output: Any,
    rubric: str,
    model: models.Model,
) -> GradingOutput:
    """Judge output against a rubric using an LLM."""
    agent = Agent(model, system_prompt=JUDGE_SYSTEM_PROMPT, output_type=GradingOutput)
    result = await agent.run(
        f"<Output>{output}</Output>\n<Rubric>{rubric}</Rubric>"
    )
    return result.output

async def judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: models.Model,
) -> GradingOutput:
    """Judge output against a rubric, considering inputs."""
    agent = Agent(model, system_prompt=JUDGE_SYSTEM_PROMPT, output_type=GradingOutput)
    result = await agent.run(
        f"<Input>{inputs}</Input>\n<Output>{output}</Output>\n<Rubric>{rubric}</Rubric>"
    )
    return result.output
```

**Trade-off:** The system prompt may differ slightly from pydantic_evals' internal prompt. This could cause marginal changes in LLMJudge behavior. To mitigate: extract the exact prompt from pydantic_evals source during implementation and use it verbatim.

**Alternative:** Read the exact system prompt from the installed pydantic_evals at implementation time and hard-code it, preserving behavioral parity.

---

## 6. WrappedPydanticEvaluator: Removal

`WrappedPydanticEvaluator` exists solely to wrap pydantic-evals `Evaluator` instances for use in ragpill. Without the pydantic_evals dependency, there are no external evaluators to wrap.

**Action:** Remove the class. Remove from `csv/testset.py:default_evaluator_classes`. Remove from `__init__.py` exports.

**Migration note for users:** If users relied on `WrappedPydanticEvaluator` to use pydantic-evals built-in evaluators, they will need to reimplement those evaluators as ragpill `BaseEvaluator` subclasses. Document this as a breaking change.

---

## 7. File-by-File Change Plan

### 7.1 New Files

| File | Purpose |
|------|---------|
| `src/ragpill/eval_types.py` | `EvaluationReason`, `EvaluationResult`, `EvaluatorSource`, `EvaluatorContext`, `Case`, `Dataset` |
| `src/ragpill/llm_judge.py` | `GradingOutput`, `judge_output()`, `judge_input_output()`, `JUDGE_SYSTEM_PROMPT` |

### 7.2 Modified Files

#### `src/ragpill/base.py`
- Remove: `from pydantic_evals.evaluators.context import EvaluatorContext`
- Remove: `from pydantic_evals.evaluators.evaluator import EvaluationReason, Evaluator`
- Add: `from ragpill.eval_types import EvaluationReason, EvaluatorContext`
- Change: `class BaseEvaluator(Evaluator)` → `class BaseEvaluator`
- Add: `get_serialization_name()` classmethod
- Add: `build_serialization_arguments()` method
- Impact: All evaluator subclasses automatically get the new base (no changes needed in subclasses unless they call `super()` on these methods)

#### `src/ragpill/evaluators.py`
- Remove: `from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext`
- Remove: `from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output, judge_output`
- Add: `from ragpill.eval_types import EvaluationReason, EvaluatorContext`
- Add: `from ragpill.llm_judge import judge_input_output, judge_output`
- Remove: `WrappedPydanticEvaluator` class entirely
- Update: `LLMJudge.build_serialization_arguments()` — change `super(BaseEvaluator, self)` call
- All `EvaluatorContext` type annotations remain identical (same generic signature)

#### `src/ragpill/mlflow_helper.py`
- Remove: `from pydantic_evals import Case, Dataset`
- Remove: `from pydantic_evals.evaluators.evaluator import EvaluationResult, EvaluatorSpec`
- Remove: function-local `from pydantic_evals.evaluators.context import EvaluatorContext`
- Add: `from ragpill.eval_types import Case, Dataset, EvaluationResult, EvaluatorContext, EvaluatorSource`
- Replace: `EvaluatorSpec(name=..., arguments=...)` → `EvaluatorSource(name=..., arguments=...)`
- Remove: `_span_tree=None` from `EvaluatorContext(...)` construction (field no longer exists)
- All function signatures using `Case[...]` and `Dataset[...]` remain identical

#### `src/ragpill/csv/testset.py`
- Remove: `from pydantic_evals import Case, Dataset`
- Add: `from ragpill.eval_types import Case, Dataset`
- Remove: `WrappedPydanticEvaluator` from imports and `default_evaluator_classes`
- No other changes (Case/Dataset construction API is identical)

#### `src/ragpill/types.py`
- Remove: `from pydantic_evals.evaluators.evaluator import EvaluationResult`
- Add: `from ragpill.eval_types import EvaluationResult`
- No other changes

#### `src/ragpill/utils.py`
- Remove: `from pydantic_evals import Dataset`
- Add: `from ragpill.eval_types import Dataset`
- `_fix_evaluator_global_flag()` signature unchanged

#### `src/ragpill/__init__.py`
- Remove: `WrappedPydanticEvaluator` from imports and `__all__`
- Optionally export new types: `EvaluationReason`, `EvaluatorContext`, `Case`, `Dataset`

### 7.3 Test Files

All test files that import from `pydantic_evals` need the same pattern:
- Replace `from pydantic_evals.evaluators.evaluator import ...` with `from ragpill.eval_types import ...`
- Replace `from pydantic_evals.evaluators import EvaluatorContext` with `from ragpill.eval_types import EvaluatorContext`
- Replace `from pydantic_evals import Case, Dataset` with `from ragpill.eval_types import Case, Dataset`
- Replace `EvaluatorSpec(...)` with `EvaluatorSource(...)`

Files to update:
- `tests/test_types.py`
- `tests/test_dataframe.py`
- `tests/test_aggregation.py`
- `tests/test_mlflow_integration.py`
- `tests/test_literal_quotes.py`
- `tests/test_regex_in_sources.py`
- `tests/test_regex_in_doc_metadata.py`
- `tests/test_regex_in_output.py`
- `tests/test_has_quotes.py`

### 7.4 Documentation Files

Update all code examples in:
- `docs/guide/evaluators.md`
- `docs/guide/csv-adapter.md`
- `docs/how-to/custom-evaluator.md`
- `docs/how-to/custom-type-evaluator.md`
- `docs/getting-started/quickstart.md`
- `docs/tutorials/full.md`
- `README.md` — update the "built on pydantic-ai evals" reference

---

## 8. Migration Risks and Mitigations

### 8.1 `build_serialization_arguments()` Behavior

**Risk:** The pydantic_evals implementation iterates dataclass fields and compares against defaults. Our reimplementation must match this behavior exactly, especially for fields with `default_factory`.

**Mitigation:** Write a dedicated test that creates a `BaseEvaluator` subclass with various field types and verifies `build_serialization_arguments()` output matches expectations. The `LLMJudge` override calls `super(BaseEvaluator, self).build_serialization_arguments()` which currently skips `BaseEvaluator` and calls `Evaluator`'s version — this must be updated to `super().build_serialization_arguments()`.

### 8.2 LLM Judge Prompt Parity

**Risk:** Different system prompt → different grading behavior → existing test results may shift.

**Mitigation:** During implementation, extract the exact system prompt from `pydantic_evals/evaluators/llm_as_a_judge.py` and replicate it. Add a comment with the source version for traceability.

### 8.3 EvaluatorContext Constructor

**Risk:** We currently suppress a pyright error for `_span_tree=None`. Removing `_span_tree` is cleaner but means our `EvaluatorContext` diverges from pydantic_evals'. If any user code passes our context to pydantic_evals functions, it would break.

**Mitigation:** This is unlikely (we own the evaluation loop). If backward compat is needed, keep `_span_tree` as an optional field defaulting to `None`. Recommended: remove it.

### 8.4 `WrappedPydanticEvaluator` Removal

**Risk:** Users who wrap pydantic-evals evaluators lose this capability.

**Mitigation:** Document the removal. Users can still import pydantic-evals directly and create their own wrapper if needed — the pattern is simple (delegate `evaluate()` to the wrapped evaluator).

### 8.5 Type Checking with Generic Dataclasses

**Risk:** `Case[InputsT, OutputT, MetadataT]` and `Dataset[InputsT, OutputT, MetadataT]` are currently Pydantic-aware. Switching to plain `@dataclass` with `Generic` may cause pyright issues with generic subscript syntax.

**Mitigation:** Use `Generic` from `typing` with `@dataclass`. This is well-supported in Python 3.11+. If pyright complains about `Dataset[str, str, TestCaseMetadata](cases=...)`, switch to non-subscripted construction.

### 8.6 pydantic-ai Version Coupling

**Risk:** We still depend on `pydantic-ai` for models. Future versions might reorganize `pydantic_ai.models`, `pydantic_ai.Agent`, or `OpenAIChatModel`.

**Mitigation:** This is unchanged from the current situation. We already depend on `pydantic-ai` — removing `pydantic_evals` doesn't increase or decrease this risk.

---

## 9. Implementation Order

The removal can be done in a single pass since it's mostly import rewiring, but for reviewability:

### Phase 1: Add new modules (additive, no breakage)
1. Create `src/ragpill/eval_types.py` with all replacement types
2. Create `src/ragpill/llm_judge.py` with judge reimplementation
3. Add tests for the new types (ensure they behave identically)

### Phase 2: Switch imports (atomic swap)
4. Update `base.py` — remove `Evaluator` inheritance, add methods, switch imports
5. Update `evaluators.py` — switch imports, remove `WrappedPydanticEvaluator`, update `LLMJudge`
6. Update `mlflow_helper.py` — switch imports, `EvaluatorSpec` → `EvaluatorSource`
7. Update `csv/testset.py` — switch imports, remove `WrappedPydanticEvaluator` from defaults
8. Update `types.py` — switch import
9. Update `utils.py` — switch import
10. Update `__init__.py` — remove `WrappedPydanticEvaluator`, optionally add new exports

### Phase 3: Tests and docs
11. Update all test imports
12. Run full test suite + type checker
13. Update documentation code examples
14. Update README

### Phase 4: Cleanup
15. Verify no remaining `pydantic_evals` imports: `grep -r "pydantic_evals" src/ tests/`
16. Consider whether to add `pydantic_evals` to a banned-imports lint rule

---

## 10. What Stays, What Goes

| Component | Status | Reason |
|-----------|--------|--------|
| `pydantic-ai` dependency | **Stays** | Used for LLM model wrappers and Agent |
| `pydantic_evals` imports | **Removed** | All types replaced locally |
| `Evaluator` base class | **Removed** | `BaseEvaluator` becomes standalone |
| `EvaluationReason` | **Replaced** | Identical local dataclass |
| `EvaluationResult` | **Replaced** | Identical local dataclass |
| `EvaluatorSpec` | **Replaced** | Renamed to `EvaluatorSource` |
| `EvaluatorContext` | **Replaced** | Simplified (no `_span_tree`) |
| `Case` | **Replaced** | Plain dataclass |
| `Dataset` | **Replaced** | Plain dataclass (no `evaluate()` method) |
| `judge_output` / `judge_input_output` | **Reimplemented** | Using `pydantic_ai.Agent` directly |
| `WrappedPydanticEvaluator` | **Removed** | No pydantic_evals evaluators to wrap |
| `mlflow.pydantic_ai.autolog()` | **Stays** | This is MLflow's integration, not pydantic_evals |

---

## 11. Public API Changes

### Breaking Changes
- `WrappedPydanticEvaluator` removed from `ragpill` exports
- `EvaluatorSpec` renamed to `EvaluatorSource` (if users accessed `EvaluationResult.source`)

### Non-Breaking Changes
- All evaluator classes remain identical in behavior
- `evaluate_testset_with_mlflow()` / `evaluate_testset_with_mlflow_sync()` signatures unchanged
- `load_testset()` signature unchanged (minus `WrappedPydanticEvaluator` in defaults)
- `EvaluationOutput`, `RunResult`, `CaseResult`, `AggregatedResult` unchanged

### New Exports (optional)
- `ragpill.eval_types.Case`
- `ragpill.eval_types.Dataset`
- `ragpill.eval_types.EvaluatorContext`
- `ragpill.eval_types.EvaluationReason`
- `ragpill.eval_types.EvaluationResult`
- `ragpill.eval_types.EvaluatorSource`
