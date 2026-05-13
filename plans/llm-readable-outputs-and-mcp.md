# Plan: LLM-Readable Outputs + MCP Server

Implements [designs/llm-readable-outputs-and-mcp.md](../designs/llm-readable-outputs-and-mcp.md).

**Status:** Phase A complete (2026-05-05). Phase B not started.

## Decisions resolved during planning

- One plan covers both phases. Phase A is implemented; Phase B is sketched as a milestone outline and gets its own follow-up plan when picked up.
- `EvaluationOutput.to_json` / `from_json` ship in **Phase A** (not Phase B as §7.6 of the design said).
- MCP server's `--mlflow-uri` live mode is **out of scope**. Phase B is directory-only.
- Secret redaction lives in the shared `_trace.py` renderer (so triage / exploration views also benefit when pasted into chat). Default-on with a sensible regex set, **easy to disable** via parameter — not high priority, so the default regex list stays small (`api.?key`, `authorization`).
- Run-id in the MCP server = **file stem** of the JSON file in `--run-dir`. The MLflow run id (if any) is surfaced *inside* the `list_runs` payload, not used as the lookup key.

## Deviations from the original plan

Worth flagging because they're load-bearing:

- **DataFrame serialization uses `orient="table"` (not `orient="split"`).** `orient="split"` collapses int-valued floats (`1.0` → `1`) and the round-trip dtype assertion fails. `orient="table"` ships a JSON Table Schema so dtypes survive. Empty DataFrames still use `orient="split"` (table needs a non-empty schema). See [_df_to_json](../src/ragpill/types.py#L185).
- **`model_params` is not surfaced in the triage header.** The design's mock-up shows it, but `model_params` is an upload-layer parameter ([upload.py:185](../src/ragpill/upload.py#L185)) and not stored on `EvaluationOutput`. Renderer is structured so it can be added later if the field is plumbed through.
- **`RunResult.error` (a real `Exception`) becomes a `RuntimeError` after round-trip** — the original message is preserved as a string but the exception class isn't. Exceptions don't pickle into JSON; this is the pragmatic compromise.
- **Trace subtree filter for triage view** keeps `RETRIEVER`, `TOOL`, `LLM`, `RERANKER`, `CHAT_MODEL`, `AGENT`. Design only listed the first four — added the latter two because real traces from `pydantic-ai` autolog produce those types and excluding them gutted the relevant-spans section.
- **Dual `_filter_trace_to_subtree` implementations** in [evaluators.py](../src/ragpill/evaluators.py#L148) and [execution.py](../src/ragpill/execution.py#L329) were left in place. The renderer does its own subtree walk inside `render_spans` instead of importing either — independent of MLflow's internal trace structure changes.

---

## Phase A — Reporting (DONE)

Shipped 2026-05-05. 39 new tests, all green; 391/391 in the full suite. ruff + basedpyright (strict) clean.

### A1. Module scaffold (DONE)

[src/ragpill/report/](../src/ragpill/report/):

- [`__init__.py`](../src/ragpill/report/__init__.py) — re-exports the two public renderers.
- [`_text.py`](../src/ragpill/report/_text.py) — `truncate`, `render_value`, `indent`.
- [`_trace.py`](../src/ragpill/report/_trace.py) — `render_spans` with subtree walk, type filter (with `ragpill_*` attribute escape hatch), secret redaction.
- [`triage.py`](../src/ragpill/report/triage.py) — triage view.
- [`exploration.py`](../src/ragpill/report/exploration.py) — exploration view.

### A2-A5. Renderers (DONE)

Behave per design §3-§5 with the deviations noted above.

### A6. Dataclass delegates (DONE)

- [`EvaluationOutput.to_llm_text`](../src/ragpill/types.py#L142) → `triage.render_evaluation_output_as_triage`.
- [`DatasetRunOutput.to_llm_text`](../src/ragpill/execution.py#L167) → `exploration.render_dataset_run_as_exploration`.

### A7. `EvaluationOutput.to_json` / `from_json` (DONE)

[`EvaluationOutput.to_json`](../src/ragpill/types.py#L166) and [`from_json`](../src/ragpill/types.py#L175) plus the `_*_to_dict` / `_*_from_dict` helpers in `types.py`. DataFrames via `orient="table"`, traces via `Trace.to_json()`, `dataset_run` delegates to `DatasetRunOutput.to_json`.

### A8. Tests (DONE)

- [`tests/test_report_text.py`](../tests/test_report_text.py) — 12 tests.
- [`tests/test_report_trace.py`](../tests/test_report_trace.py) — 10 tests, fixture spins up a temp SQLite MLflow backend per test for real Trace objects.
- [`tests/test_report_triage.py`](../tests/test_report_triage.py) — 7 tests with hand-built dataclasses (no MLflow).
- [`tests/test_report_exploration.py`](../tests/test_report_exploration.py) — 6 tests with real Trace fixtures.
- [`tests/test_evaluation_output_json_roundtrip.py`](../tests/test_evaluation_output_json_roundtrip.py) — 4 tests covering DataFrame dtype round-trip, structured `case_results`, `dataset_run` with traces, and `dataset_run=None`.

No "shape regression" fixture pin yet — content assertions in `test_report_triage.py` cover the headings the design promises to keep stable. Add a snapshot-style test if downstream prompts start coupling to specific phrasing.

### A9. Docs (DONE)

- [`docs/guide/llm-reports.md`](../docs/guide/llm-reports.md) — guide covering both views, tuning knobs, redaction, JSON persistence.
- [`docs/api/report.md`](../docs/api/report.md) — API reference page (mkdocstrings directives).
- [mkdocs.yml](../mkdocs.yml) updated: nav entries under Guide and API Reference.
- Existing API pages ([types.md](../docs/api/types.md), [execution.md](../docs/api/execution.md)) auto-pick up the new `to_llm_text` / `to_json` methods from docstrings — no edits needed.

### A10. Acceptance (DONE)

- `uv run pytest` — 391 passed, 10 skipped.
- `uv run basedpyright` — 0 errors.
- `uv run ruff check src tests && uv run ruff format --check src tests` — clean.

---

## Phase B — MCP Server (outline, NOT STARTED)

When you pick this up, branch a fresh plan from this section. Untouched in Phase A.

### B1. Packaging

- Add `mcp = ["mcp>=1.4.0"]` to `[project.optional-dependencies]` in [pyproject.toml](../pyproject.toml).
- `[project.scripts]` entry: `ragpill-mcp = "ragpill.mcp:main"`.
- `src/ragpill/mcp/__init__.py` re-exports `main`; importing it without the extra raises a clear `ModuleNotFoundError("install ragpill[mcp]")`.

### B2. Modules

- `src/ragpill/mcp/__init__.py` — `main()`, argparse for `--run-dir`, `--http`.
- `src/ragpill/mcp/server.py` — MCP server registration: tools (§7.4) + resources (§7.5).
- `src/ragpill/mcp/_store.py` — directory scan, lazy load, pairing of `<stem>.json` (DatasetRunOutput) + `<stem>.eval.json` (EvaluationOutput).

### B3. Tools (per design §7.4)

All implemented on top of Phase A renderers + `_store`:

- `list_runs` — directory scan → `[{run_id (file stem), path, case_count, has_eval_output, mlflow_run_id?}]`.
- `get_run_exploration` → `DatasetRunOutput.to_llm_text()`.
- `get_run_triage` → `EvaluationOutput.to_llm_text()`; returns "no eval output paired" if missing.
- `get_case_trace` → `render_spans(filter_types=...)` against the case subtree.
- `get_failing_cases` → JSON from `EvaluationOutput.cases`.
- `search_spans` → regex over span name/attributes within a run (or single case).

### B4. Resources

- `ragpill://run/<stem>/raw`, `ragpill://run/<stem>/cases`. Both pull-only.

### B5. Tests (Phase B)

- `tests/test_mcp_store.py` — directory scan + pairing.
- `tests/test_mcp_server.py` — in-process stdio client; `pytest.importorskip("mcp")`.

### B6. Docs (Phase B)

- `docs/how-to/mcp-server.md` with the Claude Desktop config snippet from design §7.7.
- Update [mkdocs.yml](../mkdocs.yml) `How-To` nav.

---

## Risks & deferred items

- **Format drift breaks downstream LLM prompts** (design §13). Currently mitigated by content assertions in `test_report_triage.py`. If a third-party tool starts pinning the exact format, add a snapshot-style fixture comparison.
- **Live MLflow mode** — design §7.3, deferred. Folder watching / push notifications (design §14) — deferred.
- **Multi-run comparison** (design §14) — deferred to follow-up plan.
- **Rich exception preservation across JSON round-trip** — current behavior collapses to `RuntimeError(message)`. Worth revisiting only if a downstream tool needs to dispatch on exception class.
