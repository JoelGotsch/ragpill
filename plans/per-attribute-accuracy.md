# Plan: Per-Attribute-Value Accuracy on EvaluationOutput

**Status:** Proposed
**Date:** 2026-05-15

## Context

Tags are a flat label set (you have it or you don't). **Attributes** are
key→value pairs (`{"difficulty": "hard", "domain": "chemistry"}`),
attached to both case metadata (`TestCaseMetadata.attributes`) and
evaluator metadata (`EvaluatorMetadata.attributes`), then union-merged at
evaluation time. Right now they ride along into the runs/cases DataFrames
as a JSON-serialized blob (`evaluation.py:232,256,294`) but **nobody can
group by them** without parsing that JSON themselves.

The use case is the same as per-tag accuracy: "**which value of attribute
`X` is failing?**" — e.g. "the `domain=chemistry` cases pass at 95% but
`domain=biology` only at 40%". Today the answer requires custom DataFrame
code; this change makes it a method on `EvaluationOutput` plus a triage
section.

This is the attribute-shaped twin of `per_tag_accuracy()` shipped in
0.4.1, and follows the same single-PR pattern.

## Approach

### Source of truth: `case_results`, not the DataFrame

The runs/cases DataFrames store attributes as JSON-serialized strings via
`_ta.dump_json(...)`. Parsing those back to compute groupings would force
us to round-trip Python ↔ JSON ↔ Python and would lose proper typing on
non-string values (numbers, bools, nested dicts). The structured
`self.case_results` list already carries `cr.metadata.attributes:
dict[str, Any]` in its native Python form — that's where we read from.

Per-evaluator results come from `cr.run_results[i].assertions[name].value`
which is bool-or-numeric, same as the `evaluator_result` column. We average
the same way `per_tag_accuracy` does so the numbers line up
conceptually.

### New methods on `EvaluationOutput`

```python
def per_attribute_accuracy(self, attribute: str) -> dict[Any, float]:
    """Mean evaluator result grouped by value of a single attribute key.

    Iterates ``self.case_results`` and looks up ``cr.metadata.attributes[attribute]``
    for each case. Cases that don't carry the key are skipped (not counted as
    failures). For each surviving case, every assertion across every run
    contributes its value (True->1.0, False->0.0, numeric passthrough).
    """

def per_attribute_accuracy_all(self) -> dict[str, dict[Any, float]]:
    """Same as ``per_attribute_accuracy`` but auto-discovers every attribute
    key that appears on at least one case with at least two distinct values
    across the dataset. Single-value attributes are dropped (they're not
    interesting for a breakdown — every case has the same value)."""
```

Both methods return Python-native dicts. Attribute values that aren't
hashable (lists, dicts) are stringified before being used as dict keys —
attribute-as-list is rare and we don't want to crash.

### Triage markdown — `src/ragpill/report/triage.py`

Add a "Pass rate by attribute" section in the header, sibling to "Pass
rate by tag" from 0.4.1. Render only attributes returned by
`per_attribute_accuracy_all()` (so already filtered to ≥2 distinct
values). For each such attribute, emit a small table sorted worst-first
by value:

```
## Pass rate by attribute

### `difficulty`
| Value | Pass rate | n |
|---|---|---|
| `hard` | 33% | 6 |
| `medium` | 75% | 8 |
| `easy` | 100% | 4 |

### `domain`
| Value | Pass rate | n |
|---|---|---|
| `biology` | 40% | 5 |
| `chemistry` | 95% | 13 |
```

Section omitted when no qualifying attribute exists (matches the
tag-table's degrade-silently behaviour).

### Upload layer

`upload._log_table_and_metrics` currently logs `overall_accuracy` and
`accuracy_tag_{tag}`. Add `accuracy_attr_{key}_{value}` metrics for each
(attribute, value) pair so MLflow surfaces the same breakdown. Values
get slugged (lowercase, replace non-alphanumerics with `_`) so they're
valid MLflow metric names.

### TDD order

1. Add failing tests in new file `tests/test_per_attribute_accuracy.py`:
   - Empty / no-attributes case → `{}` for both methods.
   - Single attribute, two values → method returns mapping with right
     pass rates.
   - Two attributes, mixed → `per_attribute_accuracy_all` discovers both.
   - Single-value attribute is filtered out by `_all` (still returned by
     the per-key call).
   - Numeric evaluator results averaged correctly.
   - Unhashable attribute values (e.g. a list) stringified safely.
   - Regression: triage markdown renders the new section when attributes
     present; omits when not.
2. Implement the two methods on `EvaluationOutput`.
3. Wire the renderer.
4. Wire upload metrics.
5. Full suite + lint + type-check.

### Verification

```python
result = await evaluate_testset_with_mlflow(testset, task)
result.per_attribute_accuracy("difficulty")
# {"hard": 0.33, "medium": 0.75, "easy": 1.0}
result.per_attribute_accuracy_all()
# {"difficulty": {...}, "domain": {...}}
print(result.to_llm_text())  # contains "## Pass rate by attribute"
```

## Critical files to modify

| File | Change |
|---|---|
| `src/ragpill/types.py` | Add `per_attribute_accuracy` and `per_attribute_accuracy_all` methods on `EvaluationOutput` |
| `src/ragpill/report/triage.py` | Add `_render_attribute_breakdown` helper integrated into the header |
| `src/ragpill/upload.py` | Log `accuracy_attr_{key}_{value}` metrics |
| `tests/test_per_attribute_accuracy.py` (new) | Unit tests |
| `tests/test_report_triage.py` | Regression for the new section |
| `docs/guide/llm-reports.md` | One paragraph + example |
| `pyproject.toml` | 0.4.2 → 0.4.3 (additive, no breaking API) |

## Existing code to reuse

- `cr.metadata.attributes` (`base.py`) — already the merged dict.
- The bool-or-numeric averaging convention from
  `EvaluationOutput.per_tag_accuracy` (`types.py`) — mirror it for
  consistency.
- Triage header structure (`triage.py:_render_header`) — already extended
  in 0.4.1; add a parallel call to `_render_attribute_breakdown`.

## Decisions (resolved with user 2026-05-15)

1. **`per_attribute_accuracy_all` filter** — return everything (less
   LOC). Filter single-value attributes in the **renderer** instead,
   where the cost of an empty/uniform table actually matters.
2. **MLflow metrics** — yes, log `accuracy_attr_{key}_{value}`. Apply DRY:
   extract a helper in `upload.py` that takes a prefix and a flat
   `{name: score}` mapping and logs each entry, then call it for both
   tags and attribute-value pairs.
3. **Triage section placement** — directly after "Pass rate by tag".
4. **Unhashable values** — silently `str(value)` so dict keys stay
   hashable. Doesn't crash; preserves the breakdown.

## Out of scope (deferred)

- Cross-attribute matrices (e.g. `difficulty × domain` pass rates). Useful
  but a separate, larger feature.
- Numeric attribute bucketing (auto-binning a continuous attribute). Same
  reasoning.
- Per-attribute-per-evaluator breakdown. Same shape question as
  per-tag-per-evaluator deferred in 0.4.1.
