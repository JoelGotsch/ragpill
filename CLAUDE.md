# Project conventions for AI-assisted work

## Commit messages

- **Never mention Claude, "AI", or any assistant tool in commit messages.**
  No `Co-Authored-By: Claude …` trailer, no "generated with …" notes, no
  references to the tooling used to produce the change.
- Commit messages should read as if the human author wrote them directly.
  Describe the *what* and *why* of the change; keep it terse and neutral.

## Python tooling

- Use `uv run pip ...`, never bare `pip`. Same for `uv run pytest`,
  `uv run ruff`, `uv run basedpyright`.
