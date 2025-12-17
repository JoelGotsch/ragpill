# Contributing

Thank you for your interest in contributing to ragpill!

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/JoelGotsch/ragpill.git
cd ragpill

# Install dependencies
uv sync --group dev --group docs

# Verify installation
uv run pytest
```

## Editor Setup (VS Code)

The repository ships with workspace settings in `.vscode/` that configure linting,
formatting, and type checking to match CI. When you open the project in VS Code
you should be prompted to install the recommended extensions. If not, you can
install them manually:

1. Open the Extensions panel (`Cmd+Shift+X` / `Ctrl+Shift+X`)
2. Search for **Ruff** (`charliermarsh.ruff`) and install it
3. Search for **basedpyright** (`detachhead.basedpyright`) and install it

With these extensions installed, the workspace settings will:

- **Sort imports and auto-fix lint issues** on every file save
- **Auto-format** with ruff on every file save
- **Show type errors** from basedpyright inline as you type

No additional configuration is needed — the `.vscode/settings.json` and
`.vscode/extensions.json` files are committed to the repo and will be
picked up automatically.

## Development Workflow

### Running Tests

```bash
# Run all tests (coverage is enabled by default via pyproject.toml)
uv run pytest

# Run specific test file
uv run pytest tests/test_clean_quote_text.py

# Regenerate the coverage badge after running tests
uv run genbadge coverage -i coverage.xml -o docs/coverage-badge.svg
```

The coverage badge in the README is generated from `coverage.xml` (produced automatically by pytest). Please regenerate it before committing if coverage changed.

### Code Quality

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [basedpyright](https://docs.basedpyright.com/) for type checking.
Both are configured in `pyproject.toml` and enforced in CI.

If you set up the recommended VS Code extensions (see above), most issues
will be caught and fixed automatically as you work. You can also run the
checks manually:

#### Linting

```bash
# Check for issues
uv run ruff check src tests

# Auto-fix issues (including import sorting)
uv run ruff check --fix src tests
```

#### Formatting

```bash
# Check formatting
uv run ruff format --check src tests

# Auto-format
uv run ruff format src tests
```

#### Type Checking

```bash
uv run basedpyright
```

#### Run All Checks (same as CI)

```bash
# Install tox if you haven't already
uv tool install tox --with tox-uv

# Run lint + type check
tox -e lint -e type
```

### Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Convert notebooks to markdown (required before build/serve)
uv run jupyter nbconvert --to markdown docs/how-to/*.ipynb docs/tutorials/*.ipynb

# Serve docs locally
uv run zensical serve

# Build docs
uv run zensical build
```

The docs will be available at http://localhost:8000

## Contribution Guidelines

### Code Style

- **Formatting** — handled by ruff (`line-length = 120`, double quotes). Don't worry about manual formatting; save the file and it's done.
- **Imports** — sorted automatically by ruff on save. Import order: stdlib, third-party, local (`from ragpill...`).
- **Type hints** — required on all public APIs. basedpyright runs in strict mode; your code must pass with zero errors.
- **Docstrings** — Google style (see example below).
- **Lint rules** — ruff enforces `RUF`, `C90` (complexity), `UP` (pyupgrade), and `I` (isort). See `pyproject.toml` for details.

### Docstring Example

```python
def load_testset(
    csv_path: Path,
    evaluator_classes: dict[str, type[BaseEvaluator]],
) -> Dataset:
    """Create a ragpill Dataset from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing test cases
        evaluator_classes: Mapping of evaluator type names to their classes
    
    Returns:
        A Dataset object containing all test cases
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    
    Example:
        ```python
        from ragpill.csv.testset import load_testset, default_evaluator_classes
        
        dataset = load_testset(
            csv_path=Path("testset.csv"),
            evaluator_classes=default_evaluator_classes,
        )
        ```
    """
```

### Commit Messages

Follow conventional commits:

```
feat: add support for custom evaluators
fix: handle empty CSV files gracefully
docs: update installation instructions
test: add tests for MLflow integration
refactor: simplify testset loading logic
```

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feat/my-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests and linting**: `uv run pytest && uv run ruff check .`
7. **Commit your changes**: `git commit -m "feat: add my feature"`
8. **Push to your fork**: `git push origin feat/my-feature`
9. **Open a pull request**

### Pull Request Checklist

- [ ] Tests pass locally (`uv run pytest`)
- [ ] New tests added for new features
- [ ] Type checking passes (`uv run basedpyright`)
- [ ] Linting and formatting pass (`uv run ruff check src tests && uv run ruff format --check src tests`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] Coverage badge regenerated if coverage changed

## Project Structure

```
ragpill/
├── src/
│   └── ragpill/
│       ├── __init__.py          # Curated public API
│       ├── base.py              # BaseEvaluator, metadata, resolve_repeat
│       ├── eval_types.py        # Case / Dataset / EvaluatorContext primitives
│       ├── evaluators.py        # Built-in evaluators (LLMJudge, Regex…, quotes)
│       ├── execution.py         # execute_dataset (capture layer)
│       ├── evaluation.py        # evaluate_results (evaluate layer)
│       ├── upload.py            # upload_results (upload layer)
│       ├── mlflow_helper.py     # evaluate_testset — chains the three layers
│       ├── settings.py          # TrackingSettings / LLMJudgeSettings
│       ├── types.py             # Result types (EvaluationOutput, …)
│       ├── utils.py             # Text/quote helpers, model construction
│       ├── backends/            # Pluggable tracking backends + protocols
│       ├── trace/               # Vendor-neutral trace model + dialect adapters
│       ├── report/              # LLM-readable / triage report renderers
│       └── csv/testset.py       # CSV -> Dataset loader
├── tests/                       # Test files
├── docs/                        # Documentation
├── pyproject.toml               # Project config
└── mkdocs.yml                   # Docs config
```

## Adding New Features

### Adding a New Evaluator

1. Create the evaluator class in `evaluators.py` (or your own module).
2. Inherit from `BaseEvaluator` — and decorate with `@dataclass(kw_only=True)`
   if you add fields, since `BaseEvaluator` is a dataclass.
3. Implement the required `from_csv_line()` classmethod and the async `run()`.
4. Register it under a `test_type` key when calling `load_testset`
   (`default_evaluator_classes | {"MyEval": MyEvaluator}`).
5. Add tests in `tests/` and API docs in `docs/api/evaluators.md`.

Example:

```python
from dataclasses import dataclass
from typing import Any

from ragpill.base import BaseEvaluator, EvaluatorMetadata
from ragpill.eval_types import EvaluationReason, EvaluatorContext


@dataclass(kw_only=True)
class MyEvaluator(BaseEvaluator):
    """Your evaluator description."""

    custom_param: str

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> "MyEvaluator":
        return cls(expected=expected, tags=tags, attributes=kwargs, custom_param=check)

    async def run(self, ctx: EvaluatorContext[Any, Any, EvaluatorMetadata]) -> EvaluationReason:
        passed = self.custom_param in str(ctx.output)
        return EvaluationReason(value=passed, reason="Reason for pass/fail")
```

### Adding a New Tracking Backend

See `src/ragpill/backends/` — implement the four capability protocols in
`backends/_base.py` (the contract docstrings are the spec), reuse the mixins in
`backends/_common.py`, and copy `backends/langfuse_backend.py` as a template.
Add your class to `ALL_BACKENDS` in `tests/test_backend_contract.py` to inherit
the conformance suite.

### Adding Documentation

1. Add markdown files to `docs/`
2. Update `mkdocs.yml` navigation (zensical reads this file)
3. Use autodoc for API references: `::: module.ClassName`
4. Add examples and usage

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: <https://joelgotsch.github.io/ragpill/>

## Releases

Releases are cut from `main` by tagging a version (`vX.Y.Z`); the `publish.yml`
workflow builds and publishes. Pre-1.0, minor versions may include breaking
changes — record them in `CHANGELOG.md` under **Breaking**.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to ragpill! 🎉
