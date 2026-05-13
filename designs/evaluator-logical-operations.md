# Design: Logical Operations on Evaluators

**Status:** Draft
**Date:** 2026-04-27
**Context:** ragpill evaluators today are flat and independent. Each
[`BaseEvaluator`][ragpill.base.BaseEvaluator] runs alone, returns a boolean
verdict, and is aggregated only at the case level via the repeat/threshold
mechanism. There is no way to express "*pass if `RegexInOutput` matches AND
`HasQuotes` finds at least one quote*" or "*pass if any one of three regexes
matches*" without writing a bespoke evaluator class. This document proposes a
composition primitive — `LogicalEvaluator` — that lets users build boolean
expressions over existing evaluators without abandoning the
code-first principle.

---

## 1. Goals

1. **Combine existing evaluators with boolean operators** — `AND`, `OR`,
   `NOT`, and N-of-M — yielding a single composite verdict that participates
   in the existing pipeline (assertions dict, MLflow/Langfuse upload,
   threshold aggregation) exactly like a leaf evaluator.
2. **Code first, CSV second.** The Python API is the primary surface. CSV
   support is a nice-to-have driven by the same composite class — if the
   `from_csv_line` form turns out clunky, that's acceptable; users can always
   build composites in code and pass them to a `Case`.
3. **Preserve the leaf evaluator contract.** No changes required to existing
   evaluators. A composite must not require its children to know they are
   being composed.
4. **Compose-without-double-flip.** The `expected` polarity logic in
   `BaseEvaluator.evaluate()` (base.py:336) flips the result when
   `expected=False`. A composite must reason over **raw** child verdicts and
   apply polarity exactly once, at its own boundary.
5. **Short-circuit when useful.** AND/OR over expensive evaluators (LLM
   judges) should be able to skip later children once the verdict is decided.
   Opt-in, off by default for reproducibility.
6. **Transparent attribution.** When a composite fails, the user must be able
   to see which child failed. The aggregated `reason` is the surface; child
   results may also be exposed as sub-assertions.

## 2. Non-Goals

- A general expression language (precedence rules, parser, infix syntax).
  Composites are nested objects in code; we don't ship a mini-DSL.
- Numeric/categorical aggregation (e.g. average a score across evaluators).
  Logical ops work on booleans. Numeric reductions are a separate design.
- Replacing the existing repeat/threshold "*X out of N runs*" aggregation.
  That operates **across runs** of the same evaluator; this design operates
  **within a run** across multiple evaluators.
- Conditional / dependent evaluators ("*only run B if A passed*"). That's a
  control-flow feature, not a logical operator. Short-circuit is the closest
  approximation we offer.

---

## 3. Background

### 3.1 Existing evaluator contract

| Concept | Location | Behavior |
|---|---|---|
| `BaseEvaluator.run(ctx)` | [base.py:299](../src/ragpill/base.py#L299) | Subclass implements; returns `EvaluationReason(value: bool, reason: str)`. |
| `BaseEvaluator.evaluate(ctx)` | [base.py:313](../src/ragpill/base.py#L313) | Calls `run`, asserts bool, flips value if `merged_metadata.expected == False`. |
| `EvaluationReason` | [eval_types.py](../src/ragpill/eval_types.py) | Dataclass with `value` and `reason`. |
| Invocation | [evaluation.py:157](../src/ragpill/evaluation.py#L157) | Linear loop calls `evaluator.evaluate(ctx)`; failures wrapped in `EvaluatorFailureInfo`. |
| Registration | [csv/testset.py:269](../src/ragpill/csv/testset.py#L269) | `default_evaluator_classes` dict maps `test_type` string → class. |

### 3.2 Where composition fits

A composite evaluator is "*just another `BaseEvaluator`*". The runner
loop in `_evaluate_single_run` does not need to know it exists. The composite
receives the same `EvaluatorContext`, runs its children with that same
context, combines their `EvaluationReason.value`s, and emits one
`EvaluationReason`.

This keeps the runner backend-neutral (parallel to the architectural
discipline in [langfuse-integration.md §5.1](langfuse-integration.md)) and
means MLflow/Langfuse upload paths Just Work — the composite's score is one
score, named after the composite.

---

## 4. Design

### 4.1 Class hierarchy

```
BaseEvaluator
└── LogicalEvaluator                       (abstract: holds children, name, polarity)
    ├── AndEvaluator                       (all children must pass)
    ├── OrEvaluator                        (any child must pass)
    ├── NotEvaluator                       (single child; inverts)
    └── NOfMEvaluator                      (≥ k of n children pass)
```

`XorEvaluator` is a special case of N-of-M (`k=1, exactly=True`); we ship
it only if a real use case shows up. `NAndEvaluator`/`NOrEvaluator` are
sugar for `Not(And(...))` / `Not(Or(...))` and intentionally omitted.

### 4.2 `LogicalEvaluator` base

```python
@dataclass
class LogicalEvaluator(BaseEvaluator):
    """Composite evaluator. Combines child evaluator verdicts with a
    boolean operator.

    Children are run via their ``run()`` method (NOT ``evaluate()``) so that
    polarity logic is applied exactly once, at the composite boundary. A
    child's own ``expected`` field is therefore ignored when it is wrapped;
    set polarity on the composite instead.
    """

    children: list[BaseEvaluator] = field(default_factory=list)
    short_circuit: bool = False
    # Optional: name override for the composite verdict. Defaults to a
    # generated name like "And(RegexInOutput, HasQuotes)".
    name: str | None = None

    async def run(self, ctx) -> EvaluationReason:
        raise NotImplementedError  # subclass implements combine logic
```

Key contract decisions:

1. **Children's `run`, not `evaluate`.** Composites bypass each child's
   polarity flip by calling `child.run(ctx)` directly. The composite owns
   polarity; children inside a composite are pure boolean producers.
2. **Children must return `bool`.** Asserted at composite level — same
   constraint `evaluate()` already enforces (base.py:333).
3. **Children may themselves be composites.** Nesting is supported with no
   special handling because `run` returns the same `EvaluationReason` shape.
4. **`evaluation_name` collision.** Each child has its own UUID. The
   composite has its own UUID. The single assertion uploaded to MLflow is
   the composite's. Children's individual results are *not* uploaded as
   separate scores by default — they live in the composite's `reason`. (See
   §4.6 for the opt-in.)

### 4.3 Operator semantics

| Operator | Pass condition | Short-circuit on |
|---|---|---|
| `AndEvaluator` | All children pass | first child that returns `False` |
| `OrEvaluator` | At least one child passes | first child that returns `True` |
| `NotEvaluator` | Single child fails | n/a (one child) |
| `NOfMEvaluator(k)` | ≥ `k` of `n` children pass | when remaining cannot change verdict |

`NOfMEvaluator` accepts `exactly: bool = False` for the strict-equality
variant (useful for "*exactly one of these regexes should match*").

### 4.4 Reason formatting

The composite's `reason` aggregates child reasons. Format proposed:

```
AND(LLMJudge=True, RegexInOutput=False)
  - LLMJudge: passed — output mentions Paris
  - RegexInOutput: failed — pattern '\\bcite\\b' not found
```

- Top line: short, machine-parseable summary (`OPERATOR(child1=bool, ...)`).
- Indented bullets: each child's full `reason` string.
- When `short_circuit=True` and we skipped children, mark them as
  `LLMJudge=<skipped>` so the reader knows the verdict was decided early.

### 4.5 Polarity

The composite respects `expected` exactly like a leaf evaluator: its own
`evaluate()` (inherited from `BaseEvaluator`) flips the combined value if
`expected=False`. This is what the user wants for "*the disjunction should
NOT hold*" patterns.

Children's `expected` is **ignored when wrapped**. We document this
prominently and (optionally) emit a `RagpillCompositeWarning` when a child
has a non-default `expected`. Two reasons it must be ignored:

1. Avoiding double-flip — a child with `expected=False` would already flip
   its own result in `evaluate()`, but we call `run()`. If we honored child
   polarity ourselves we'd reimplement and risk drift.
2. The semantics of "*expected=False inside an OR*" are ambiguous. Forcing
   polarity to live only on the composite makes the boolean expression
   unambiguous.

Programmatic helper to make this explicit:

```python
def as_child(ev: BaseEvaluator) -> BaseEvaluator:
    """Strip polarity from an evaluator so it can be safely wrapped."""
    ev.expected = None  # raw boolean producer
    return ev
```

### 4.6 Sub-assertion exposure (opt-in)

Composites are opaque by default — only the aggregated verdict shows up in
`assertions`. Some users will want per-child visibility for debugging
without losing the composite. Opt-in via `expose_children: bool = False`:

When `True`, the runner gets back a list of `EvaluationResult`s instead of
one. Implementation: extend `BaseEvaluator.evaluate` to return
`EvaluationReason | list[EvaluationResult]`, OR (simpler) have the composite
decompose itself in a hook the runner calls after `evaluate`. Open question
in §8 — leaning toward the hook because it doesn't widen the public type.

Result names when exposed:
- Composite: `MyCompositeName`
- Children: `MyCompositeName/LLMJudge`, `MyCompositeName/RegexInOutput`

This keeps grouping visible in MLflow/Langfuse without name collisions.

### 4.7 Tags and `is_global`

| Field | Composite behavior |
|---|---|
| `tags` | Composite carries its own tags. Children's tags are visible only when `expose_children=True`. |
| `is_global` | Set on composite. Implies the *whole expression* applies to every case. Children's `is_global` is ignored when wrapped, same as `expected`. |
| `attributes` | Composite has its own `attributes` dict. Children's are not merged up. |

This is the simplest discipline — composites are first-class evaluators
that happen to delegate; their config is independent of children's config.

---

## 5. Code-first API

### 5.1 Inline operators (primary surface)

`BaseEvaluator` overloads `&`, `|`, `~` to build composites inline:

```python
class BaseEvaluator:
    def __and__(self, other):  return AndEvaluator(children=[self, other])
    def __or__(self, other):   return OrEvaluator(children=[self, other])
    def __invert__(self):      return NotEvaluator(child=self)
```

So this works directly:

```python
from ragpill.evaluators import (
    LLMJudge, RegexInOutputEvaluator, HasQuotesEvaluator,
)

ev = (RegexInOutputEvaluator(pattern=r"\bParis\b")
      & HasQuotesEvaluator(min_quotes=1)) | LLMJudge(rubric="…")
```

**Why not `and` / `or` / `not`?** Python's boolean keywords are not
overloadable. They short-circuit on `__bool__` and return one of their
operands *unchanged* — there's no hook to intercept them and build a
composite. `Evaluator1() and Evaluator2()` would call
`bool(Evaluator1())` (a category error: an evaluator's truthiness has no
meaning until a context is supplied) and then return one of the two
evaluators, never a combined object. PEP 335 (which would have allowed
overloading these) was rejected. So we use the bitwise trio, which Python
*does* let us overload, and which already has the right precedence
(`~` > `&` > `|`, matching `not` > `and` > `or`).

The constructor form below is equivalent and preferred when expressions
get long, when you need named keyword args (`short_circuit=True`,
`tags={...}`), or for N-of-M.

Auto-flattening: `a & b & c` builds `And([And([a, b]), c])` by default;
the operator overload normalizes adjacent same-operator nodes so the result
is a single flat `And([a, b, c])`. Same for `|`. This keeps the reason
output readable.

### 5.2 Constructor form

```python
from ragpill.evaluators import (
    AndEvaluator, OrEvaluator, NotEvaluator, NOfMEvaluator,
    LLMJudge, RegexInOutputEvaluator, HasQuotesEvaluator,
)

# AND: must mention Paris AND must contain at least one quote
must_cite_paris = AndEvaluator(
    children=[
        RegexInOutputEvaluator(pattern=r"\bParis\b"),
        HasQuotesEvaluator(min_quotes=1),
    ],
    tags={"requires_citation"},
)

# OR with short-circuit: cheap regex first, fall back to LLM judge
acceptable_answer = OrEvaluator(
    children=[
        RegexInOutputEvaluator(pattern=r"\bnot applicable\b"),
        LLMJudge(rubric="Output adequately addresses the question."),
    ],
    short_circuit=True,
)

# NOT: should not contain forbidden phrases
no_pii = NotEvaluator(
    child=RegexInOutputEvaluator(pattern=r"\b\d{3}-\d{2}-\d{4}\b"),
)

# N-of-M is constructor-only — no operator form
case = Case(
    inputs=q,
    evaluators=[
        NOfMEvaluator(
            k=2,
            children=[
                AndEvaluator(children=[r1, r2]),
                OrEvaluator(children=[r3, r4]),
                NotEvaluator(child=r5),
            ],
        ),
    ],
)
```

### 5.3 Reflected operators

`__rand__`, `__ror__`, and friends are also implemented for the rare case
where the left operand is something else that ends up combining with an
evaluator (e.g. via a helper that returns evaluators conditionally). They
delegate to the forward variants.

### 5.3 Registry

Add to `default_evaluator_classes`
([csv/testset.py:269](../src/ragpill/csv/testset.py#L269)):

```python
default_evaluator_classes |= {
    "And": AndEvaluator,
    "Or":  OrEvaluator,
    "Not": NotEvaluator,
    "NOfM": NOfMEvaluator,
}
```

---

## 6. CSV Support (best-effort)

The code-first principle means CSV can be lossy without being a blocker.
Two viable approaches; we ship **6.1** in v1 and treat **6.2** as a possible
later extension if demand surfaces.

### 6.1 JSON-in-`check` (v1)

The composite's `check` column contains a JSON tree describing the
expression. Children reference *registered evaluator classes by name*, with
their own `check` payload as a nested object.

```csv
Question,test_type,expected,tags,check
"Where is the Eiffel Tower?",And,true,geography,"{
  ""children"": [
    {""type"": ""RegexInOutputEvaluator"", ""check"": {""pattern"": ""\\bParis\\b""}},
    {""type"": ""HasQuotesEvaluator"",     ""check"": {""min_quotes"": 1}}
  ]
}"
```

`AndEvaluator.from_csv_line` parses the JSON, looks each child up in the
registry, calls its own `from_csv_line` with the nested `check`, and
constructs the composite.

Pros:
- One row per logical assertion. CSV reader doesn't change.
- Arbitrarily deep nesting is expressible.

Cons:
- Multi-line JSON inside a CSV cell is ugly. Most spreadsheet tools mangle
  embedded newlines on save, so users in practice end up writing single-line
  JSON. We document this limitation explicitly.
- Editing in Excel/Sheets is painful. We tell users: when expressions get
  complex, drop CSV and write the composite in code.

This is the path that fits the existing CSV adapter without churn.

### 6.2 Linked-rows form (deferred)

Add an optional `parent_id` column. A composite row declares an `id` and
its `test_type` (e.g. `And`); child rows reference it via `parent_id`. The
loader runs a two-pass build: first instantiate leaves, then composites.

Pros: each leaf row reads exactly like a normal row.
Cons: every CSV grows two columns; the parser becomes graph-shaped; ordering
between files breaks; cycles are possible. We don't ship this until the
JSON-in-check form is shown insufficient by real usage.

### 6.3 What stays the same

- `expected`, `tags` columns apply to the composite (the row).
- Extra columns become `attributes` on the composite, not children.
- Polarity inversion via `expected=false` works exactly as for leaf
  evaluators (the composite's verdict is flipped at `evaluate()` time).

---

## 7. Implementation Plan

### Phase 1 — Core class + AND/OR/NOT (code-only)

1. New module: `src/ragpill/composite.py` with `LogicalEvaluator`,
   `AndEvaluator`, `OrEvaluator`, `NotEvaluator`.
2. Reason aggregation helper.
3. Operator overloads on `BaseEvaluator` (`__and__`, `__or__`, `__invert__`).
4. Tests:
   - Unit: each operator returns the expected `EvaluationReason.value` for
     all bool combinations of two- and three-child cases.
   - Polarity: composite with `expected=False` flips correctly.
   - Polarity: child with `expected=False` is ignored (and a warning
     emitted) when wrapped.
   - Short-circuit: AND skips remaining children after the first `False`;
     OR after the first `True`. Verified with a spy child that records
     calls.
   - Nesting: `And(Or(a, b), Not(c))` agrees with Python's native boolean
     evaluation.
5. Docs: `docs/guide/composite-evaluators.md` — code-first patterns, the
   polarity-on-composite rule, short-circuit semantics.

### Phase 2 — N-of-M

1. `NOfMEvaluator(k, exactly=False)`.
2. Short-circuit logic: track running pass count and remaining-children; cut
   when `passes >= k` (and `exactly=False`) or when `passes + remaining < k`
   (failure decided).

### Phase 3 — CSV support (JSON-in-check)

1. `AndEvaluator.from_csv_line` and friends parse the nested JSON and
   delegate to child evaluators' `from_csv_line` via the registry.
2. Registry entries added.
3. Round-trip tests: compose in CSV → load → run → expect same verdict as
   the Python-constructed equivalent.
4. Doc update: [docs/guide/csv-adapter.md](../docs/guide/csv-adapter.md)
   gets a "*composite evaluators*" section. Acknowledge the multi-line-JSON
   ergonomics issue explicitly and point users at the code form for
   complex expressions.

### Phase 4 — Sub-assertion exposure (opt-in)

1. Decide between (a) widening `evaluate()` return type or (b) a new
   `decompose()` hook the runner calls. Currently leaning (b) — see §8.
2. Wire through the runner in
   [evaluation.py:157](../src/ragpill/evaluation.py#L157).
3. Verify MLflow/Langfuse upload places sub-scores under the right name
   pattern (`Composite/Child`).

---

## 8. Open Questions

1. **Sub-assertion exposure mechanism.** Two options:
   - (a) `BaseEvaluator.evaluate` returns `EvaluationReason | list[EvaluationReason]`.
     Touches every call site.
   - (b) Add `BaseEvaluator.decompose() -> list[EvaluationResult] | None`
     (returns `None` for leaves). Runner checks after `evaluate()` and
     splays results when non-None.
   Option (b) is additive and doesn't churn the public type; default
   recommendation pending review.

2. **Should `expected` on a child raise instead of warn?** A warning is
   user-friendly; a hard error prevents silent footguns. If we go with the
   `as_child(ev)` helper, raising on a wrapped non-`None` `expected` is
   the more defensive choice. Lean toward warn-in-v1, harden later.

3. **N-of-M with weighted children.** Out of scope for v1, but worth noting:
   if it lands, it generalizes to a weighted-sum-with-threshold form, which
   starts to overlap with numeric aggregation (a separate design).

4. **Footgun: `bool(evaluator)` in user code.** Because we don't define
   `__bool__`, an evaluator is truthy by default (Python's fallback). If a
   user writes `if my_evaluator: ...` expecting it to mean something, it
   silently always passes. We could define `__bool__` to raise
   `TypeError("evaluators have no truth value outside an evaluation context;
   use & | ~ to combine, or evaluate() to run")` — same trick pandas uses
   for ambiguous Series truth values. Recommended.

5. **Aggregation across runs (`repeat > 1`).** When a composite is
   evaluated `N` times across runs, the existing per-evaluator pass-rate
   logic
   ([evaluation.py:49](../src/ragpill/evaluation.py#L49)) treats it as one
   evaluator and applies the threshold. This is correct. But if
   `expose_children=True` is set, do children also get per-evaluator pass
   rates? Yes — they appear under the `Composite/Child` name and are
   treated as independent evaluators by the aggregator. Document this.

---

## 9. Out of Scope

- Numeric reductions over child scores (mean, sum, min, max). Different
  type signature; different design.
- Cross-case logical operators ("*case A passes only if case B passed*").
  ragpill cases are independent by design.
- A textual expression language (`"A AND (B OR NOT C)"` parsed at runtime).
  The constructor form and the `& | ~` sugar cover the same expressiveness
  with full IDE support and no parser to maintain.
- Conditional/short-circuit chaining beyond what AND/OR already give
  ("*run B with the result of A as input*"). That's a pipeline, not a
  logical operator.
