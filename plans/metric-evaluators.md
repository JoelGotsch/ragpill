# Plan: Metric-Based Evaluators (TokenUsage, AnswerLength, Duration)

**Status:** Proposed
**Date:** 2026-04-27
**Approach:** TDD per evaluator (red → green → refactor), one PR

---

## 1. Goal

Add three evaluators that check numeric bounds on signals captured by the
execute layer:

| Evaluator           | Source signal                  | Where it lives in `EvaluatorContext` |
|---|---|---|
| `DurationEvaluator`     | wall-clock seconds             | `ctx.duration`                       |
| `AnswerLengthEvaluator` | length of stringified output   | `ctx.output`                         |
| `TokenUsageEvaluator`   | LLM token counts from trace    | `ctx.trace` (via `SpanBaseEvaluator.get_trace`) |

All three share the same shape: a verdict that the measured value falls inside
`[min_value, max_value]`. They return a `bool`, so `BaseEvaluator.evaluate`'s
`expected` polarity logic works without changes ([base.py:333](../src/ragpill/base.py#L333)).

## 2. Non-goals

- No new fields on `EvaluatorContext` — duration is already there
  ([eval_types.py:91](../src/ragpill/eval_types.py#L91)) and tokens are read
  off `ctx.trace.info.token_usage` / span attributes; we are not modifying the
  execute layer.
- Not adding token capture for non-pydantic-ai backends. If
  `mlflow.pydantic_ai.autolog()` ([execution.py:276](../src/ragpill/execution.py#L276))
  is the only autolog wired up, that's the supported scope. Other backends are
  out-of-scope for this plan.
- No new aggregation metrics in the upload layer (the runs DataFrame already
  carries per-evaluator pass/fail; mean tokens/duration would be a separate
  feature).

## 3. Shared design

### 3.1 Bounds API

A single helper used by all three evaluators:

```python
@dataclass
class _Bounds:
    min_value: float | None = None  # None = unbounded below
    max_value: float | None = None  # None = unbounded above

    def check(self, value: float) -> tuple[bool, str]:
        ...  # returns (passes, reason)
```

The reason string includes the measured value, the bounds, and the unit
("s", "chars", "tokens"). Centralizing this avoids three nearly-identical
reason-string blocks.

### 3.2 CSV `check` parameter

JSON-only — the `check` field for these evaluators is *always* a JSON object.
Plain integers are deliberately **not** supported (unlike `HasQuotesEvaluator`)
because a bare number is ambiguous for limit-style checks (is `500` a min or a
max?). Forcing JSON makes intent explicit.

```
check='{"max": 5.0}'              # duration ≤ 5s
check='{"min": 50, "max": 2000}'  # 50 ≤ length ≤ 2000
check='{"max": 4000, "kind": "total"}'  # token-specific extras
```

If neither `min` nor `max` is set, `from_csv_line` raises `ValueError`. Empty
`check` strings are also rejected.

### 3.3 Polarity (`expected`)

Use the standard `expected=True` semantics: "the value should be inside the
bounds." Inversion via `expected=False` is supported by the base class — no
custom handling needed.

## 4. Per-evaluator detail

### 4.1 `DurationEvaluator`

- Reads `ctx.duration` directly. No trace required.
- `from_csv_line`: parse `min`/`max` (floats, seconds).
- Reason example: `"Duration 6.42s exceeds max 5.00s."`

### 4.2 `AnswerLengthEvaluator`

- `unit: Literal["chars", "words"] = "chars"` field on the dataclass.
- `chars`: `len(str(ctx.output))`.
- `words`: `len(str(ctx.output).split())`. (Token-count of the answer text is
  intentionally left to `TokenUsageEvaluator`, which uses real LLM-reported
  counts, not a tokenizer guess.)
- `from_csv_line` accepts `{"min": ..., "max": ..., "unit": "words"}`.

### 4.3 `TokenUsageEvaluator`

- Inherit from `SpanBaseEvaluator` ([evaluators.py:175](../src/ragpill/evaluators.py#L175))
  to reuse run-subtree filtering — when `ctx.run_span_id` is set,
  `get_trace(ctx)` returns a trace narrowed to the current run, so token sums
  exclude sibling-run spans.
- `kind: Literal["input", "output", "total"] = "total"` field.
- Source of truth: walk the filtered trace's LLM spans and sum
  `span.attributes["mlflow.chat.tokenUsage"]` (or the standard MLflow key —
  verify during Phase 1; fall back to scanning common keys
  `input_tokens` / `output_tokens` / `total_tokens`).
- `ctx.trace` is None when `capture_traces=False`. Raise a clear
  `ValueError` ("requires capture_traces=True"), mirroring
  `SpanBaseEvaluator.get_trace`'s contract
  ([evaluators.py:204](../src/ragpill/evaluators.py#L204)).
- Note in the docstring: aggregation across LLM spans means a single
  multi-step agent call is summed, which is the desired behavior for budget
  checks.

> **Phase 1 verification step:** before writing `TokenUsageEvaluator.run`, run
> a small script that calls `execute_dataset` with `capture_traces=True` on a
> trivial pydantic-ai task and prints `trace.info.token_usage` and
> `[s.attributes for s in trace.search_spans(span_type=SpanType.LLM)]`. The
> exact attribute key/shape isn't documented in this codebase yet, so confirm
> it before coding the sum.

## 5. Phases

### Phase 1 — Verify token-usage shape (no code change)

Run the verification script above; record findings as a comment in
`TokenUsageEvaluator`'s docstring. This must happen before Phase 3.

### Phase 2 — `DurationEvaluator` + `AnswerLengthEvaluator`

These don't need traces, so they're easy and unblock the shared `_Bounds`
helper.

**Tests (red):** `tests/test_duration_evaluator.py`,
`tests/test_answer_length_evaluator.py`. Pattern: hand-construct
`EvaluatorContext` with `duration=…` / `output="…"` and a `TestCaseMetadata`
([test_evaluation.py:16-48](../tests/test_evaluation.py#L16-L48) is the
template). Cover:
- min only, max only, both, neither (raises)
- value inside / below / above bounds
- `expected=False` flips the verdict
- `from_csv_line` parses JSON, rejects empty / numeric strings
- `AnswerLengthEvaluator` with `unit="words"` vs `"chars"`

**Code (green):** add `_Bounds` helper + the two classes to
[evaluators.py](../src/ragpill/evaluators.py). Export from
[__init__.py](../src/ragpill/__init__.py) `__all__`. Register in
`default_evaluator_classes` in [csv/testset.py:269](../src/ragpill/csv/testset.py#L269).

### Phase 3 — `TokenUsageEvaluator`

**Tests (red):** `tests/test_token_usage_evaluator.py`. Pattern: build a
synthetic `Trace` with two LLM spans whose attributes match whatever Phase 1
discovered. Cover:
- Sum across multiple LLM spans
- `kind` selects input/output/total correctly
- `ctx.trace=None` raises with the same wording as `SpanBaseEvaluator.get_trace`
- Run-subtree filtering: a trace with two run subtrees only counts the active
  one when `ctx.run_span_id` is set
- `from_csv_line` parses `{"max": N, "kind": "..."}`

**Code (green):** subclass `SpanBaseEvaluator`. Add to exports + CSV registry.

### Phase 4 — Docs

Per `.claude/skills/documentation-guidelines`:
- Google-style docstrings on all three classes (the dataclass-level docstring
  is the source of truth).
- Add `::: ragpill.evaluators.DurationEvaluator` etc. entries to
  `docs/api/evaluators.md` (mirror the existing pattern there).
- One short example block per evaluator showing CSV usage and programmatic
  usage.

## 6. Files touched

| File | Change |
|---|---|
| `src/ragpill/evaluators.py` | add `_Bounds`, `DurationEvaluator`, `AnswerLengthEvaluator`, `TokenUsageEvaluator` |
| `src/ragpill/__init__.py` | extend imports + `__all__` |
| `src/ragpill/csv/testset.py` | extend `default_evaluator_classes` (line 269) |
| `tests/test_duration_evaluator.py` | **NEW** |
| `tests/test_answer_length_evaluator.py` | **NEW** |
| `tests/test_token_usage_evaluator.py` | **NEW** |
| `docs/api/evaluators.md` | add three `:::` API entries |

No changes to: `eval_types.py`, `execution.py`, `evaluation.py`, `upload.py`,
`base.py`. The layered architecture from
[layered-execution-evaluation.md](layered-execution-evaluation.md) already
exposes everything we need on `EvaluatorContext`.

## 7. Open questions

- **Token-attribute key.** Resolved by Phase 1.
- **Should `AnswerLengthEvaluator` support a `tokens` unit** (counted by a
  tokenizer like `tiktoken`)? Recommendation: no — it adds a heavyweight
  optional dependency for a niche use case; users wanting that can write a
  custom evaluator. Confirm before Phase 2.
- **Should there be an `expected` default of `True` on the dataclasses?**
  Existing evaluators leave `expected=None` and let metadata fill it
  ([base.py:168](../src/ragpill/base.py#L168)). Stick with that for
  consistency.
