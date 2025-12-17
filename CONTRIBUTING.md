# Contributing to ragpill

Thanks for your interest in contributing! The full guide — dev setup, running
tests/lint/type-checks, coding conventions, and how to add an evaluator or a
tracking backend — lives in the documentation:

**[docs/development/contributing.md](docs/development/contributing.md)**

Quick start:

```bash
uv sync --group dev --group docs
uv run pytest          # tests
uv run ruff check src/ # lint
uv run basedpyright    # type-check
```

Please open an issue to discuss substantial changes before starting, and see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
