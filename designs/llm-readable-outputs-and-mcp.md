# Design: LLM-Readable Outputs + MCP Server

**Status:** Phase A implemented (2026-05-05); Phase B (MCP server) not started. See [plans/llm-readable-outputs-and-mcp.md](../plans/llm-readable-outputs-and-mcp.md).
**Date:** 2026-04-24
**Context:** Now that `DatasetRunOutput` and `EvaluationOutput` hold rich
structured data (outputs, traces, evaluator reasons), we want to surface them
to an LLM — both for interactive debugging of failing runs and, later, as
MCP tools an agent can call. This design defines the markdown formats and the
MCP server surface.

---

## 1. Goals

1. **Two plain-text markdown views** on the existing output dataclasses:
   - **Triage view** on `EvaluationOutput` — "why did this fail?" Leads with
     failures, shows evaluator reasons, surfaces the spans that drove the
     verdict.
   - **Exploration view** on `DatasetRunOutput` — "what did the agent do?"
     Full trace tree per case, no pass/fail opinion.
2. **MCP server** (bundled as optional extra `ragpill[mcp]`) that exposes
   these views as tools an agent can call against a saved
   `DatasetRunOutput.json` or a live MLflow run.

## 2. Non-Goals

- HTML / dashboard rendering. We already have MLflow for that.
- Streaming trace data. The outputs are post-hoc snapshots.
- Remote MCP hosting. The server runs locally / in a dev environment; users
  who want a shared service can wrap it themselves.

---

## 3. Format Principles

- **Markdown, not JSON.** LLMs read markdown with less overhead and we keep
  whitespace budget for trace content.
- **Fail-fast layout.** Failing items first, passing items collapsed.
- **Budgeted truncation.** Every view exposes a `max_chars` parameter with a
  sensible default (~8k for summaries, ~32k for per-case drilldowns). Values
  exceeding the budget are replaced with `… (truncated N chars)`.
- **Stable structure.** Section headings and field names don't change
  between releases, so an LLM can learn them once. Additions go at the end of
  a section.
- **No colour / no emoji.** Rendering-neutral.

### 3.1 Common truncation helpers

Live in a new module `src/ragpill/report/_text.py`:

- `truncate(s: str, max_chars: int) -> str` — suffix with `… (+N)`.
- `render_value(v: Any, max_chars: int = 300) -> str` — stringifies, flattens
  nested dicts/lists to JSON with sorted keys, clips long strings.
- `indent(s: str, levels: int) -> str` — two spaces per level.

---

## 4. Triage View (`EvaluationOutput.to_llm_text`)

### 4.1 Target audience

An LLM or engineer asking "why did this evaluation fail?" Given an
`EvaluationOutput` produced by `evaluate_results`, return a single markdown
document optimized for answering that question.

### 4.2 Shape

```
# Evaluation summary

- Total cases: 12 (8 passed, 4 failed)
- Overall pass rate: 66.7% (threshold applied per-case)
- Evaluator rollup:
  - `LLMJudge` — 11/12 passed
  - `RegexInSourcesEvaluator` — 6/12 passed
- Model params: { "model": "gpt-4o", "temperature": "0.2" }

## Failing cases

### Case 3: "What does the board meeting minutes from March 5 2024 say about Q2 targets?"

- Pass rate: 1/3 runs
- Inputs: `…`
- Expected output: `Q2 revenue target of $12M (per minutes, page 3)`
- Threshold: 0.80

#### Run 0 — FAIL (3 assertions; 2 failing)

- `LLMJudge`: **FAIL** — "Output claims Q2 target is $14M; minutes state
  $12M."
- `RegexInSourcesEvaluator(pattern="q2")`: **PASS**
- `RegexInSourcesEvaluator(pattern="12m")`: **FAIL** — "Regex pattern
  '12m' not found in any document content."

##### Relevant spans

- `retriever` (span_id=abc123) — 4 documents retrieved
  - `minutes-2024-03-05.txt` (relevance=0.91)
  - `board-notes-q1.txt` (relevance=0.84)
  - …
- `llm-call` (span_id=def456, duration=1.3s)
  - Input: `Based on the following docs, what were the Q2 targets…`
  - Output: `The board set Q2 revenue targets of $14M.`

#### Run 1 — FAIL (similar pattern)

…

### Case 7: "…"

…

## Passing cases (collapsed)

- Case 1: "What is the company fiscal year-end?" — 3/3 runs passed
- Case 2: "…" — 3/3 runs passed
- …
```

### 4.3 Sections in order

1. **Header** — "# Evaluation summary" + bullet list (totals, pass rate,
   per-evaluator rollup, `model_params` if present).
2. **Failing cases** — one `###` section each, ordered by pass rate ascending
   then by case_id for stability. For each:
   - Case-level bullets (pass rate, inputs, expected output, threshold).
   - Per failing run (`####`) — evaluator list with PASS/FAIL, then a
     `##### Relevant spans` subsection.
   - "Relevant spans" is the trace subtree rooted at
     `run_result.run_span_id`, filtered to spans of type `RETRIEVER`, `TOOL`,
     `LLM`, `RERANKER` plus any span whose `attributes` contain
     `ragpill_*`. Other spans (function calls, timing wrappers) are dropped.
3. **Passing cases (collapsed)** — one bullet per case, no per-run detail.

### 4.4 Truncation policy

- Each run's "Relevant spans" section: 1500 chars.
- Each span input/output: 300 chars.
- Total document: 32k chars. If exceeded, drop passing-case section first,
  then collapse run detail on cases after index 5, then truncate last cases
  with `… (N additional failing cases not shown)`.

### 4.5 API

```python
class EvaluationOutput:
    def to_llm_text(
        self,
        *,
        max_chars: int = 32_000,
        include_passing: bool = True,
        include_spans: bool = True,
    ) -> str:
        """Render a triage-focused markdown summary suitable for LLM input."""
```

Implementation lives in `src/ragpill/report/triage.py`. The method on
`EvaluationOutput` is a one-line delegate so the dataclass stays small.

---

## 5. Exploration View (`DatasetRunOutput.to_llm_text`)

### 5.1 Target audience

An LLM or engineer asking "what did the agent do on this input?" Pre-
evaluation — no pass/fail opinions, just the captured traces.

### 5.2 Shape

```
# Dataset run

- Cases: 12
- Tracking URI: sqlite:///:memory: (local temp — cleaned up)
- MLflow run: (none)

## Case 1: "What is the company fiscal year-end?"

- Inputs: `What is the company fiscal year-end?`
- Expected output: `December 31`
- Runs: 3

### Run 0 (duration=1.4s, output="The fiscal year ends December 31.")

- agent (span)
  - retriever (duration=120ms) — 3 docs: bylaws-2021.txt, handbook-2023.pdf, …
  - llm-call (duration=1.2s)
    - input: "You are a compliance assistant…"
    - output: "The fiscal year ends December 31."

### Run 1 (…)
…

## Case 2: …
```

### 5.3 Sections in order

1. **Header** — "# Dataset run" + bullet list (case count, tracking URI,
   mlflow run/experiment ids if set).
2. **Per case** (`##`) — inputs, expected output, run count.
3. **Per run** (`###`) — duration, output summary, indented trace tree.

### 5.4 Trace tree rendering

- Depth-first, `  - span_name` at each level, two spaces per nesting level.
- Inline summary: `(type, duration)`.
- If the span has inputs/outputs and is an LLM/TOOL/RETRIEVER span, show them
  indented under the bullet, one per line, each truncated to 300 chars.
- Drop noisy span types (`FUNCTION`, internal opentelemetry wrappers) unless
  they have meaningful inputs/outputs.

### 5.5 Truncation policy

- Default 16k chars total.
- Per-case budget: 2000 chars; overflow collapses runs to one line.
- Per-span budget: 500 chars.

### 5.6 API

```python
class DatasetRunOutput:
    def to_llm_text(
        self,
        *,
        max_chars: int = 16_000,
        include_spans: bool = True,
    ) -> str:
        """Render an exploration-focused markdown summary."""
```

Implementation in `src/ragpill/report/exploration.py`.

---

## 6. Shared Trace Renderer

Both views need to format `mlflow.entities.Trace` spans. Extract a single
helper in `src/ragpill/report/_trace.py`:

```python
def render_spans(
    trace: Trace | None,
    *,
    root_span_id: str | None = None,
    max_chars: int = 1500,
    filter_types: Iterable[str] | None = None,
) -> str:
    """Render a trace (or a subtree) as nested markdown bullets."""
```

- When `root_span_id` is set, walk only that subtree (reuses
  `_filter_trace_to_subtree` already in `ragpill.evaluators`).
- `filter_types`: iterable of MLflow `SpanType` values to keep. When `None`,
  keep everything.

This lives in the shared helper so test coverage of the rendering logic
isn't duplicated.

---

## 7. MCP Server (`ragpill.mcp`)

### 7.1 Packaging

- Optional extra: `ragpill[mcp]`. When installed pulls `mcp` (the official
  Python SDK from `modelcontextprotocol/python-sdk`).
- Module: `src/ragpill/mcp/__init__.py` — entry point `ragpill.mcp:main`.
- Console script: `ragpill-mcp` in `[project.scripts]`.

### 7.2 Transport

- **stdio** by default (the common MCP transport for local tools).
- Optional `--http` flag for HTTP-SSE via `mcp.server.sse`. Only mention this
  in docs — not wired until somebody needs it.

### 7.3 State

The server holds:

- A filesystem path (from `--run-dir`) scanned for `*.json` files produced by
  `DatasetRunOutput.to_json()`. Files are lazy-loaded on demand.
- Optional MLflow tracking URI (from `--mlflow-uri`) — when set, the server
  can also materialize evaluation outputs from a live server run by run id.

### 7.4 Tools exposed

Each tool returns a markdown string (the triage/exploration views) or
structured JSON where appropriate.

| Tool | Args | Returns | Built on |
|------|------|---------|----------|
| `list_runs` | `limit: int = 20` | JSON array of `{run_id, path, case_count, has_eval_output}` | directory scan + optional MLflow |
| `get_run_exploration` | `run_id: str`, `max_chars: int = 16_000` | markdown | `DatasetRunOutput.to_llm_text()` |
| `get_run_triage` | `run_id: str`, `max_chars: int = 32_000`, `include_passing: bool = False` | markdown | requires an accompanying `EvaluationOutput.json`; see §7.5 |
| `get_case_trace` | `run_id: str`, `case_id: str`, `span_types: list[str] \| None = None` | markdown (deeper drilldown, ~32k budget) | `render_spans(filter_types=span_types)` |
| `get_failing_cases` | `run_id: str` | JSON array `[{case_id, pass_rate, first_failing_reason}]` | Works from `EvaluationOutput`; only when present |
| `search_spans` | `run_id: str`, `query: str`, `case_id: str \| None = None` | JSON array of matching spans (name, type, attribute subset) | regex match over span name / attributes |

### 7.5 Resources

MCP also supports "resources" (passive data URIs). Expose:

- `ragpill://run/<run_id>/raw` → the raw `DatasetRunOutput` JSON.
- `ragpill://run/<run_id>/cases` → the `EvaluationOutput.cases` DataFrame as
  JSON, when present.

Resources are pull-only — the agent can `read_resource()` without side
effects.

### 7.6 Pairing with EvaluationOutput

`EvaluationOutput` currently has no `to_json()` / `from_json()`. Add it in
the same PR as the MCP server:

- Serialize the two DataFrames via `pandas.to_json(orient="split")`.
- `case_results` via the existing dataclass → dict pattern.
- `dataset_run` delegates to `DatasetRunOutput.to_json`.

The MCP server then looks for `<run_id>.json` (dataset run) and
`<run_id>.eval.json` (evaluation output) side-by-side in the `--run-dir`.

### 7.7 Example CLI

```bash
pip install "ragpill[mcp]"

# Run an evaluation and save both JSON artifacts:
python -m my_eval_script --save-dir ~/ragpill-runs

# Launch the MCP server against that dir:
ragpill-mcp --run-dir ~/ragpill-runs
```

Claude Desktop / other MCP clients register the server in their config:

```json
{
  "mcpServers": {
    "ragpill": {
      "command": "ragpill-mcp",
      "args": ["--run-dir", "/Users/you/ragpill-runs"]
    }
  }
}
```

---

## 8. Module Layout

```
src/ragpill/report/
  __init__.py          # re-exports to_llm_text helpers
  _text.py             # truncation + value rendering primitives
  _trace.py            # render_spans() shared helper
  triage.py            # render_evaluation_output_as_triage()
  exploration.py       # render_dataset_run_as_exploration()

src/ragpill/mcp/
  __init__.py          # main(), CLI arg parsing
  server.py            # MCP server setup + tool handlers
  _store.py            # on-disk run directory + MLflow lookup
```

`EvaluationOutput.to_llm_text` and `DatasetRunOutput.to_llm_text` live as
one-line delegates on the existing dataclasses (`types.py`, `execution.py`)
to keep discoverability.

---

## 9. Changes to Existing Types

### 9.1 `EvaluationOutput` (in `types.py`)

Add:

- `to_llm_text(...) -> str` — delegates to `ragpill.report.triage`.
- `to_json() -> str` / `from_json(s) -> EvaluationOutput`.

### 9.2 `DatasetRunOutput` (in `execution.py`)

Add:

- `to_llm_text(...) -> str` — delegates to `ragpill.report.exploration`.

### 9.3 No new fields

We deliberately keep the dataclasses flat — no precomputed text. Rendering
is derived from the existing data each time.

---

## 10. Testing

Per-phase test files follow the existing
`.claude/skills/test-writing-guidelines` style.

### 10.1 Unit tests

- `tests/test_report_triage.py` — hand-built `EvaluationOutput` → assert
  expected headings, that failing cases come first, passing section is
  collapsed, truncation kicks in as advertised.
- `tests/test_report_exploration.py` — hand-built `DatasetRunOutput` with a
  synthetic `Trace` → assert tree indentation and per-span input/output
  truncation.
- `tests/test_report_trace.py` — `render_spans()` filter_types and
  subtree-root behavior.
- `tests/test_report_json_roundtrip.py` — `EvaluationOutput.to_json()` /
  `from_json()` round-trip equality on DataFrames + structured results.

### 10.2 MCP tests

- `tests/test_mcp_server.py` — spawn the server in-process via
  `mcp.client.stdio`, call each tool, assert return shapes. Gated on the
  `mcp` optional extra being installed (skip if not).
- `tests/test_mcp_store.py` — `_store` module: directory scan, lazy load,
  pairing of dataset run + eval output.

### 10.3 Integration

- `tests/test_report_integration.py` — full path: execute a dataset → save
  JSON → spawn MCP server → call `get_run_triage` → assert markdown shape.
  Skipped unless MLflow is available (reuses existing skip gate).

---

## 11. Dependencies

### 11.1 Core (no change)

`to_llm_text` uses only stdlib + `pandas` (already a dependency). No new core
dependencies.

### 11.2 Optional `[mcp]` extra

In `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.4.0"]
```

Importing `ragpill.mcp` without the extra raises a clear
`ModuleNotFoundError` with install hint. We follow the pattern from
`pyproject.toml`'s existing `[docs]` extra.

---

## 12. Phased Implementation

### Phase A — Reporting only (no MCP)

1. `src/ragpill/report/` module + all three renderers.
2. `to_llm_text` on `EvaluationOutput` and `DatasetRunOutput`.
3. `EvaluationOutput.to_json()` / `from_json()`.
4. Full unit test coverage.
5. Docs: new `docs/guide/llm-reports.md` and API pages.

### Phase B — MCP server

1. `[mcp]` optional extra + `ragpill-mcp` console script.
2. Server, tools, resources.
3. Tests + skip gating.
4. Docs: `docs/how-to/mcp-server.md` with Claude Desktop config snippet.

Phase A ships first because it's useful standalone (users can paste
`.to_llm_text()` output into any chat) and doesn't pull in the `mcp`
dependency.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `to_llm_text` formats drift between releases and break LLM prompts that assume a shape | Freeze section headings/field labels. Add a regression test that pins the shape of a reference fixture. |
| Context-window blowups on large runs | Aggressive default `max_chars`. Truncation is visible (`… (+N)`) so the LLM knows data was cut. |
| MCP SDK API not stable yet | Pin a floor version (`>=1.4.0`) and test against a single version in CI. Treat `ragpill[mcp]` as an experimental extra in the README. |
| DataFrame → JSON round trip loses dtypes | Use `pandas.to_json(orient="split")` (preserves index+columns+types) and assert equality in tests. |
| MCP server accidentally exposes secrets from traces (API keys in span attributes) | Redact span attributes matching a configurable regex (defaults include `api.?key`, `authorization`). |

---

## 14. Out of Scope / Future

- Turning `to_llm_text` output into OpenAI function-calling schemas. The
  markdown format is already LLM-friendly; structured tool-use is the job
  of the MCP server.
- A `ragpill-mcp` daemon that watches a directory and push-notifies on new
  runs. Write it once users ask.
- Multi-run comparison tools (`diff_runs(run_a, run_b)`). Easy follow-up
  once the single-run views are in.
