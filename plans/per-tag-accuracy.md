# Plan: Per-Tag Accuracy on EvaluationOutput

**Status:** Proposed
**Date:** 2026-05-14

## Context

Today, per-tag accuracy is computed but **only as a side-effect of the MLflow
upload path**: `src/ragpill/upload.py:97-101` explodes the `runs` DataFrame's
`tags` column, groups by tag, and means the `evaluator_result` column. The
result is logged to MLflow as `accuracy_tag_{tag}` metrics and never surfaced
to Python callers. A user who wants to see "which tag is failing" has to
either parse `EvaluationOutput.runs` themselves or open the MLflow UI.

This change makes per-tag accuracy a first-class attribute of
`EvaluationOutput`, refactors the upload path to reuse it (DRY), and adds a
tag-breakdown section to the triage LLM-readable report so an LLM debugging
failures sees the signal immediately.

Scope per user clarifications:
- **Overall per-tag only** (not a tags×evaluators matrix). One number per tag.
- **Surface in `to_llm_text()` triage** (new "Pass rate by tag" section).
- **Refactor `upload.py`** to call the new method instead of inlining the logic.

## Approach

### 1. Add `EvaluationOutput.per_tag_accuracy()` — `src/ragpill/types.py`

Add a method right after `summary` (line ~140), before `to_llm_text` (~142):

```python
def per_tag_accuracy(self) -> dict[str, float]:
    """Mean ``evaluator_result`` grouped by tag, across all runs and evaluators.

    Tags appear on both case metadata (TestCaseMetadata.tags) and evaluator
    metadata (BaseEvaluator.tags); these are union-merged during evaluation
    (base.py:127) so each row in ``self.runs`` carries the full tag set. Rows
    with NaN ``evaluator_result`` (evaluator failures) are excluded.

    Returns:
        Mapping from tag -> mean evaluator result in [0.0, 1.0]. Empty dict
        when ``self.runs`` is empty or has no tagged rows.
    """
```

Implementation reuses the existing logic from `upload.py:93-101`:

```python
if self.runs.empty:
    return {}
df_valid = self.runs[self.runs["evaluator_result"].notna()]
if df_valid.empty:
    return {}
exploded = df_valid.explode("tags")
grouped = exploded.groupby("tags")["evaluator_result"].mean()
return {tag: float(acc) for tag, acc in grouped.items() if pd.notna(tag)}
```

Naming: `per_tag_accuracy` mirrors the existing `accuracy_tag_{tag}` metric
names in `upload.py`, so users who saw it in MLflow find the same word in
Python.

### 2. Refactor `upload.py` to call the new method — `src/ragpill/upload.py:81-101`

`_log_table_and_metrics` currently takes a `runs_df` parameter. Change its
signature to accept the full `EvaluationOutput` so it can call the new
method.

Before (`upload.py:218`): `_log_table_and_metrics(evaluation.runs, model_params)`
After: `_log_table_and_metrics(evaluation, model_params)`

Inside:

```python
def _log_table_and_metrics(
    evaluation: EvaluationOutput,
    model_params: dict[str, str] | None,
) -> None:
    mlflow.log_table(evaluation.runs, "evaluation_results.json")
    if model_params:
        mlflow.log_params(model_params)
    if evaluation.runs.empty:
        return
    df_valid = evaluation.runs[evaluation.runs["evaluator_result"].notna()]
    if len(df_valid) > 0:
        mlflow.log_metric("overall_accuracy", float(df_valid["evaluator_result"].mean()))
    for tag, accuracy in evaluation.per_tag_accuracy().items():
        mlflow.log_metric(f"accuracy_tag_{tag}", accuracy)
```

`overall_accuracy` stays inline (single line, not tag-related). Only the
explode/groupby block moves.

### 3. Add "Pass rate by tag" to triage markdown — `src/ragpill/report/triage.py`

Triage rendering today produces a header, failures, optional passing, and a
spans section. Insert a new section between the header and the first failure
listing. Render only when `per_tag_accuracy()` returns a non-empty dict —
degrades silently for untagged datasets.

Format (sorted by accuracy ascending, so problem tags surface first):

```
## Pass rate by tag

| Tag | Pass rate | n runs |
|---|---|---|
| flaky | 33% (1/3) | 3 |
| baseline | 100% (4/4) | 4 |
```

The "n runs" column needs the row count from the same groupby. The renderer
computes both inline from `evaluation.runs` rather than asking
`EvaluationOutput` for a second helper — keeps the public API minimal.

### 4. Tests — new file `tests/test_per_tag_accuracy.py`

Cover with synthetic `EvaluationOutput` objects (no MLflow involvement):

- Returns `{}` for empty runs DataFrame.
- Returns `{}` when all `evaluator_result` are NaN (all evaluators failed).
- Two tags on the same case: each appears with the right pass rate.
- Tag only on the evaluator (not on the case): still appears.
- Tag only on the case (not on the evaluator): still appears (union merge).
- Mixed bool and numeric evaluator results: numeric values are averaged
  alongside booleans (document this behavior in the docstring as well).
- One run with multiple tags expands cleanly (the explode).

Also: a regression test in `tests/test_report_triage.py` confirming the new
"Pass rate by tag" section appears when tags are present and is omitted
when not. Reuse fixtures from that file.

### 5. Docs

- `docs/api/types.md` — list the new method under `EvaluationOutput`.
- `docs/guide/llm-reports.md` — one short paragraph that the triage view now
  includes a tag breakdown.

## Critical files to modify

| File | Change |
|---|---|
| `src/ragpill/types.py` (~line 140) | Add `per_tag_accuracy()` method |
| `src/ragpill/upload.py:81-101` | Refactor `_log_table_and_metrics` to call new method |
| `src/ragpill/upload.py:218` | Update caller |
| `src/ragpill/report/triage.py` | Add "Pass rate by tag" section |
| `tests/test_per_tag_accuracy.py` (new) | Unit tests for the method |
| `tests/test_report_triage.py` | Regression for new triage section |
| `docs/api/types.md` | Document method |
| `docs/guide/llm-reports.md` | Mention new section |

## Existing code to reuse (not re-implement)

- The explode-and-mean pattern at `src/ragpill/upload.py:97-101` — this *is*
  the algorithm; just move it.
- `evaluation._create_runs_dataframe` (`src/ragpill/evaluation.py:233`) —
  already populates the `tags` set[str] column the new method depends on.
  No change needed.
- Tag union via `merge_metadata()` (`src/ragpill/base.py:127`) — already
  unions case + evaluator tags upstream. No change needed.

## Verification

End-to-end smoke from a Python shell:

```python
from ragpill import Case, Dataset, evaluate_testset_with_mlflow
# ... build a dataset with two cases, one tagged "alpha", one tagged "beta"
result = await evaluate_testset_with_mlflow(testset, task)
print(result.per_tag_accuracy())  # {"alpha": 1.0, "beta": 0.5}
print(result.to_llm_text()[:2000])  # confirm "Pass rate by tag" section
```

Test runs:

- `uv run pytest tests/test_per_tag_accuracy.py -v` — new unit tests pass.
- `uv run pytest tests/test_report_triage.py -v` — regression for new section.
- `uv run pytest` — full suite (391+ tests) stays green on both
  `py311-highest` and `py311-lowest` matrix legs.
- `RUN_MLFLOW_INTEGRATION_TESTS=1 uv run pytest tests/test_mlflow_integration.py`
  — confirm the refactored `_log_table_and_metrics` still logs
  `accuracy_tag_{tag}` metrics identically (compare run before/after on a
  tagged testset).
- `mkdocs serve` — confirm docs render without warnings.

## Out of scope (deferred)

- Tags × evaluators matrix view. Could be added later as
  `per_tag_accuracy_by_evaluator() -> pd.DataFrame` without changing the
  existing API.
- Confidence intervals / sample-size weighting. Current `evaluator_result`
  mean is the simplest defensible signal; statistical refinements are a
  separate decision.
- ADR write-up. This is a Small decision (single helper method, no API
  break). Should become the next sequential ADR in date order, per the ADR
  plan in `plans/adr-system.md`.
