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
│       ├── __init__.py
│       ├── base.py              # Base classes
│       ├── evaluators.py        # Pre-built evaluators
│       ├── mlflow_helper.py     # MLflow integration
│       ├── utils.py             # Utilities
│       └── csv/                 # CSV module
│           ├── testset.py       # CSV loading
│           └── constructors.py  # Constructor helpers
├── tests/                       # Test files
├── docs/                        # Documentation
├── pyproject.toml              # Project config
└── mkdocs.yml                  # Docs config (used by zensical)
```

## Adding New Features

### Adding a New Evaluator

1. Create evaluator class in `evaluators.py`
2. Inherit from `BaseEvaluator`
3. Implement `evaluate()` method
4. Add tests in `tests/`
5. Add documentation in `docs/api/evaluators.md`
6. Add example in tutorial notebook

Example:

```python
class MyEvaluator(BaseEvaluator):
    """Your evaluator description."""
    
    def __init__(
        self,
        expected: bool,
        tags: str,
        check: str,
        custom_param: str,
    ):
        super().__init__(expected, tags, check)
        self.custom_param = custom_param
    
    async def evaluate(self, input_val: str, output: str) -> EvalOutput:
        # Your logic
        passed = True  # Your check
        return EvalOutput(
            name=f"my_check_{self.check}",
            passed=passed,
            reason="Reason for pass/fail",
        )
```

### Adding Documentation

1. Add markdown files to `docs/`
2. Update `mkdocs.yml` navigation (zensical reads this file)
3. Use autodoc for API references: `::: module.ClassName`
4. Add examples and usage

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the [docs] TODO: add tfs link

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to ragpill! 🎉
