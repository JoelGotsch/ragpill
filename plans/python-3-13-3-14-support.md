# Plan: Full Python 3.13 + 3.14 CI Coverage

**Status:** Proposed
**Date:** 2026-05-19

## Context

The `test` matrix in `.github/workflows/ci.yml` currently lists
`python-version: ["3.11", "3.12", "3.13"]` but support for the newer
runtimes has not actually been verified end-to-end:

- 3.13 is in the matrix but no observed runs/badges confirm clean
  passes against pinned-version highest-resolution deps; if a
  transitive (mlflow / pandas / pydantic-ai) drops 3.13 wheels we
  won't notice until release time.
- 3.14 is GA on `python.org` but absent from the matrix entirely. The
  `target-version = "py311"` line in `pyproject.toml` and the
  `pythonVersion = "3.11"` in `[tool.basedpyright]` set the lower
  bound, not the upper. We should confirm 3.14 actually works.

Goal: every supported Python in the matrix passes the full test suite,
ruff lint+format, and basedpyright. Failures get triaged immediately,
not at release time.

## Tasks

1. **Verify 3.13 locally** — `uv venv .venv313 --python 3.13 && uv pip
   install -e .[mlflow] -r dev` and run the full suite. Triage any
   `DeprecationWarning`-promoted-to-failure or import errors.
2. **Add 3.14 to the matrix** — extend `.github/workflows/ci.yml`:
   ```yaml
   matrix:
     python-version: ["3.11", "3.12", "3.13", "3.14"]
   ```
   Also add 3.14 to the tox `env_list` in `pyproject.toml`.
3. **Pin transitive dependencies that block 3.13/3.14**, if any. Most
   likely candidates: `mlflow-skinny` (uses C-extension deps),
   `pandas` (wheels per-version), `pydantic-ai-slim`.
4. **basedpyright sweep** — newer Python versions sometimes expose
   stricter type errors as the typeshed advances. Run
   `uv run basedpyright` under 3.14 and fix anything that surfaces
   (typically `reportDeprecated` for stdlib calls that gained warnings).
5. **No-extras smoke job** (already added in the multi-backend branch
   at `.github/workflows/ci.yml::no-extras-smoke`) should also be
   replicated across 3.13/3.14 once that branch lands.

## Risks

- mlflow's transitive `protobuf` / `pyarrow` may lag on 3.14 wheels;
  expect to pin a floor version once available or to mark
  `mlflow-integration` as Python-version-conditional.
- `cryptography` (now in the dev group from the 0.4.5 bug-fix PR) has
  per-version wheels; older floor may not have 3.14 ABI3 wheels.

## Out of scope

- 3.10 support. `requires-python = ">=3.11"` was set deliberately
  (uses `match`, `Self`, etc.).
- Free-threaded / no-GIL 3.13t / 3.14t. Possibly interesting if
  pydantic-ai supports it, but not required for parity.

## Success criteria

- Both 3.13 and 3.14 appear as required-status checks on PRs.
- A 3.14-specific failure (in deps or stdlib usage) produces a
  recognisable CI red, not a silent skip.
