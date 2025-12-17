# LLM-Readable Reports

ragpill's outputs (`DatasetRunOutput`, `EvaluationOutput`) carry rich
structured data — DataFrames, traces, per-evaluator reasons. That's great for
programmatic post-processing but unwieldy when you just want to **paste the
results into a chat session and ask "why did this fail?"**.

`to_llm_text()` renders those outputs as compact markdown tuned for LLM input:
fail-fast layout, predictable headings, aggressive truncation with visible
markers, and optional secret redaction.

---

## Two views, two questions

| View | Question it answers | Method |
|------|---------------------|--------|
| **Triage** | "Why did this evaluation fail?" | [`EvaluationOutput.to_llm_text()`](../api/types.md#evaluationoutput) |
| **Exploration** | "What did the agent do on this input?" | [`DatasetRunOutput.to_llm_text()`](../api/execution.md#datasetrunoutput) |

The triage view requires an `EvaluationOutput` (you've already run evaluators
and want to debug failures). The exploration view works on a raw
`DatasetRunOutput` — useful before you've written evaluators, or to inspect
the trace tree for a single case.

---

## Triage: debugging a failing evaluation

```python
result = await evaluate_testset_with_mlflow(
    testset=testset,
    task=my_agent,
    mlflow_settings=MLFlowSettings(),
)

# Drop into a chat tool of your choice:
print(result.to_llm_text())
```

The triage view orders failing cases first (by ascending pass rate), shows
each failing run's evaluator verdicts and reasons, and surfaces the relevant
trace subtree for each failing run — filtered to `RETRIEVER`, `TOOL`, `LLM`,
`RERANKER`, `CHAT_MODEL`, and `AGENT` spans by default. Passing cases collapse
to one bullet each. When the dataset uses tags (on cases or evaluators), a
**Pass rate by tag** table appears in the header so the worst-performing
tags surface first.

A trimmed shape:

```
# Evaluation summary

- Total cases: 12 (8 passed, 4 failed)
- Overall pass rate: 66.7% (threshold applied per-case)
- Evaluator rollup:
  - `LLMJudge` — 11/12 passed
  - `RegexInSourcesEvaluator` — 6/12 passed

## Pass rate by tag

| Tag | Pass rate | n |
|---|---|---|
| `q2-numerics` | 33% | 9 |
| `factual-recall` | 75% | 24 |
| `tone` | 100% | 12 |

## Failing cases

### Case `9f4c…`: "What does the March 5 board minutes say about Q2 targets?"

- Pass rate: 1/3 runs
- Inputs: What does the March 5 board minutes say about Q2 targets?
- Expected output: Q2 revenue target of $12M
- Threshold: 0.80

#### Run 0 — FAIL (3 assertions; 2 failing)

- Output: The board set Q2 revenue targets of $14M.
- `LLMJudge`: **FAIL** — Output claims Q2 target is $14M; minutes state $12M.
- `RegexInSourcesEvaluator(pattern="q2")`: **PASS**
- `RegexInSourcesEvaluator(pattern="12m")`: **FAIL** — Pattern '12m' not found in any source.

##### Relevant spans

- retriever (RETRIEVER, 87ms)
  input: {"q": "Q2 targets March 5"}
  output: ["minutes-2024-03-05.txt", "board-notes-q1.txt"]
- generate (LLM, 1284ms)
  input: Based on the following docs, what were the Q2 targets…
  output: The board set Q2 revenue targets of $14M.

## Passing cases (collapsed)

- `1a2b…`: "What is the company fiscal year-end?" — 3/3 runs passed
- …
```

### Tuning the output

```python
text = result.to_llm_text(
    max_chars=16_000,        # smaller budget for cheaper context windows
    include_passing=False,    # skip the passing-case section
    include_spans=False,      # skip "Relevant spans" subsections
)
```

When the document exceeds `max_chars`, the renderer sheds content in this
order: (1) the passing-case section, (2) per-run detail on failing cases past
index 5, (3) trailing failing cases (replaced with a `… (N additional cases
not shown)` marker).

### Per-tag accuracy without the markdown wrapper

The same tag breakdown is available as a plain dict for programmatic use:

```python
result.per_tag_accuracy()
# {"q2-numerics": 0.33, "factual-recall": 0.75, "tone": 1.0}
```

Tags can sit on cases (`TestCaseMetadata.tags`) or on evaluators
(`BaseEvaluator.tags`) — both are union-merged before grouping. Rows where
an evaluator raised an exception are excluded from the denominator.

---

## Exploration: looking at what the agent did

```python
run_output = await execute_dataset(testset, task=my_agent)
print(run_output.to_llm_text())
```

This view has no opinion on pass/fail — it just walks each case, each run,
and the captured trace tree:

```
# Dataset run

- Cases: 1
- Tracking URI: sqlite:///:memory:
- MLflow run: r1
- MLflow experiment: e1

## Case `1a2b…`: "What is the company fiscal year-end?"

- Inputs: What is the company fiscal year-end?
- Expected output: December 31
- Runs: 3

### Run 0 (duration=1.40s)

- Output: The fiscal year ends December 31.

- agent (AGENT, 1402ms)
  - retriever (RETRIEVER, 120ms)
    output: ["bylaws-2021.txt", "handbook-2023.pdf"]
  - llm (LLM, 1242ms)
    input: You are a compliance assistant…
    output: The fiscal year ends December 31.
```

By default every span type is included — the exploration view doesn't filter
by span type since the goal is breadth, not focus.

---

## Secret redaction

Both views run span inputs/outputs/attributes through a redaction pass before
rendering. The default pattern set is intentionally narrow:

- `(?i)api[_-]?key`
- `(?i)authorization`

Any dict key matching these regexes gets its value replaced with
`<redacted>`. To customize:

```python
text = result.to_llm_text(
    redact=True,
    redact_patterns=[r"(?i)password", r"(?i)secret", r"(?i)token"],
)
```

To disable entirely (e.g., in a private dev environment where you trust the
output):

```python
text = result.to_llm_text(redact=False)
```

Redaction is **not** a security barrier — it's a guard against pasting
obvious credentials into a chat. Treat it like `.gitignore`, not like a
firewall.

---

## Saving for later (or sharing with a tool)

`EvaluationOutput` and `DatasetRunOutput` both serialize to JSON:

```python
# Save:
with open("eval-output.json", "w") as f:
    f.write(result.to_json())

# Reload (e.g., in a different process):
from ragpill.types import EvaluationOutput

with open("eval-output.json") as f:
    result = EvaluationOutput.from_json(f.read())

print(result.to_llm_text())
```

DataFrames are encoded with the JSON Table Schema (`orient="table"`), so
dtypes survive the round trip. Traces are encoded via MLflow's own
`Trace.to_json` so span structure (including subtree filtering) is preserved.

This is the foundation for the upcoming MCP server, which will expose these
saved files to an MCP-aware agent as tools.

---

## See also

- [`render_evaluation_output_as_triage`](../api/report.md#render_evaluation_output_as_triage) — the underlying renderer (use directly for finer control).
- [`render_dataset_run_as_exploration`](../api/report.md#render_dataset_run_as_exploration).
- [Result Types](../api/types.md) — the dataclasses these views render.
- [Execution Layer](../api/execution.md) — produces `DatasetRunOutput`.
