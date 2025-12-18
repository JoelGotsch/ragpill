# Agent Guidelines for ragpill

This document provides guidelines for AI agents (and human developers) working on the `ragpill` project.

## Core Principles

### 1. No Example Usage Files in Source Code

**❌ DON'T:**
- Create `example_usage.py`, `example_constructors.py`, or similar files in `src/`
- Add example code as standalone Python files

**✅ DO:**
- Put all examples in the documentation (`docs/` or `mkdocs/`)
- Use Jupyter notebooks in the documentation for interactive examples
- Include code examples in docstrings and README files

**Rationale:** Example files clutter the source code, can become outdated, and are better maintained as documentation where users actually look for them.

### 2. Always Use `uv` for Package Management

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management.

**❌ DON'T:**
- Use `pip install`
- Use `python -m pytest`
- Use `python script.py`

**✅ DO:**
- Use `uv pip install` for installing packages
- Use `uv run pytest` for running tests
- Use `uv run python script.py` for running scripts
- Use `uv sync` to sync dependencies

### 3. Always Update Documentation and Tests

After making ANY code changes, you MUST:

1. **Update Documentation:**
   - **📘 See [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md) for detailed documentation guidelines**

2. **Update Tests:**
   - Update existing tests to match the new interface
   - Add new tests for new features
   - Ensure test assertions are meaningful (not just print statements)
   - No print statements as part of tests. Exception: temporary during development of test case

3. **Run Tests:**
   - Execute tests with `uv run pytest` to verify they pass
   - Run specific test files: `uv run pytest tests/testset_csv.py`
   - Run with verbose output: `uv run pytest -v`
   - Check for errors in the modified files
   - Fix any issues before considering the work complete

## Project Structure

```
ragpill/
├── src/ragpill/          # Source code - NO example files here
│   ├── base.py            # Base evaluator classes
│   ├── evaluators.py      # Evaluator implementations
│   ├── csv/               # CSV adapter for testsets
│   │   ├── testset.py     # Core CSV functionality
│   │   └── __init__.py    # Public API
│   └── mlflow/            # MLflow integration
├── tests/                 # Test files
│   ├── testset_csv.py     # CSV testset tests
│   └── test_*.py          # Other tests
├── mkdocs/                # Documentation
│   └── src/               # Documentation source
│       ├── guide/         # User guides
│       ├── tutorials/     # Tutorial notebooks
│       └── api/           # API documentation
└── docs/                  # Additional documentation
```

## Code Standards

### Test Requirements

Every test file MUST:
- Have meaningful assertions (not just print statements)
- Test both success and failure cases where applicable
- Be runnable independently
- Use `TestModel()` to avoid external API calls

```python
# ✅ GOOD: Tests with assertions
def test_feature():
    result = some_function()
    assert result is not None, "Result should not be None"
    assert len(result) > 0, "Result should have items"

# ❌ BAD: Tests without assertions
def test_feature():
    result = some_function()
    print(f"Result: {result}")  # This doesn't test anything!
```

### Documentation Requirements

**📘 For comprehensive documentation guidelines, see [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md)**

All public APIs must have:
- Clear docstring with description using Google-style format
- Args section with type hints
- Returns section
- Example usage in docstring
- Cross-references to related functions (See Also section)
- Updated in API documentation (`docs/api/`)

**Quick Documentation Rules:**

1. **Docstrings = Single Source of Truth**
   - Write comprehensive docstrings in source code
   - Use Google-style format (Args:, Returns:, Example:, See Also:)
   - Include cross-references using `[text][module.path.to.object]` syntax

2. **API Documentation Structure**
   - `:::` directives ONLY in `docs/api/` files
   - API section must be complete (all public functions/classes)
   - Guides/tutorials link to API docs (no `:::` directives)

3. **Cross-References**
   - Link related functions in docstrings
   - Create bidirectional navigation
   - Use markdown links in guides: `[function_name](../api/module.md#function_name)`

**For detailed examples, patterns, and best practices, see [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md)**

## Workflow for Making Changes

### 1. Planning Phase
- Understand the requirements
- Check existing code and patterns
- Identify affected files (code, tests, docs)

### 2. Implementation Phase
- Make code changes
- Update or add docstrings
- Follow existing patterns and conventions

### 3. Testing Phase
- Update existing tests to match new interface
- Add new tests for new functionality
- **RUN THE TESTS** with `uv run pytest` - verify they pass
- Run specific tests: `uv run pytest tests/testset_csv.py -v`
- Fix any failures

### 4. Documentation Phase
- **Update docstrings** in source code (single source of truth)
- Update guide documentation files to reference enhanced docstrings using mkdocstrings
- Add workflow examples and tutorials in markdown (supplement the API docs)
- Add examples to docs (NOT as separate .py files)
- Update API documentation if needed
- Update README if the change affects main features

### 5. Verification Phase
- Check for errors using `get_errors` tool
- Verify all files compile/import correctly
- Ensure no broken references
- Run full test suite: `uv run pytest`

## Common Tasks

### Adding a New Evaluator

1. Create the evaluator in `src/ragpill/evaluators.py` with comprehensive docstring
2. Implement `from_csv_line()` class method. Same signature as in BaseEvaluator
3. Add to `docs/api/evaluators.md` with `:::` directive
4. Add tests in `tests/`
5. Update guide in `docs/guide/evaluators.md` with markdown links
6. Add example in tutorial notebook
7. Run tests with `uv run pytest` to verify
8. **📘 See [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md) for docstring format**

### Refactoring an API

1. Update the implementation
2. Update ALL calling code
3. Update ALL tests
4. Update ALL documentation
5. Run tests with `uv run pytest` to verify
6. Update migration guide if breaking change

### Removing Deprecated Features

1. Remove the code
2. Remove from public exports (`__init__.py`)
3. Remove or update tests
4. Update documentation to remove references
5. Add deprecation notice to changelog
6. Run tests with `uv run pytest` to verify

## File Maintenance

### Files to DELETE if found:
- `src/ragpill/csv/example_*.py` - Move to docs
- `src/ragpill/*/example_*.py` - Move to docs
- Dead code or unused imports

### Files to ALWAYS update together:
- `src/ragpill/csv/testset.py` ↔️ `tests/testset_csv.py` ↔️ `mkdocs/src/guide/csv-adapter.md`
- `src/ragpill/evaluators.py` ↔️ `tests/test_*.py` ↔️ `mkdocs/src/guide/evaluators.md`
- `src/ragpill/csv/__init__.py` ↔️ Documentation when exports change

## uv Command Reference

Common commands for this project:

```bash
# Install dependencies
uv sync

# Install a new package
uv pip install package-name

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/testset_csv.py

# Run tests with verbose output
uv run pytest -v

# Run tests and show print statements
uv run pytest -s

# Run a Python script
uv run python script.py

# Run a Python module
uv run python -m module_name
```

## Quality Checklist

Before considering work complete:

- [ ] Code changes are complete and follow project patterns
- [ ] All affected tests are updated
- [ ] Tests have been RUN with `uv run pytest` and pass
- [ ] No API documentation in `docs/api/` if new public functions/classes added
- Update guide documentation files with markdown links to API docs
- Add workflow examples and tutorials in markdown (supplement the API docs)
- Update README if the change affects main features
- **📘 Follow guidelines in [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md)**
- [ ] Public API changes are documented
- [ ] No dead code or unused imports
- [ ] Used `uv` for all package and test operations

## Common Pitfalls to Avoid

1. **Don't** create example files in source directories
2. **Don't** skip running tests after changes (use `uv run pytest`)
3. **Don't** use `pip` or `python` directly (see [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md))
6. **Don't** use `:::` directives outside `docs/api/` (see [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md))
7. **Don't** leave outdated comments or docstrings
8. **Don't** use the old `evaluator_kwargs` pattern (removed)
9. **Don't** create unnecessary wrapper functions or constructors
10. **Don't** use old-style docstrings (`:param:`, `:return:`) - use Google-style
7. **Don't** use the old `evaluator_kwargs` pattern (removed)
8. **Don't** create unnecessary wrapper functions or constructors

## Questions to Ask (Check [DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md))
- Are docstrings comprehensive and in Google-style format?
- Have I added the function to `docs/api/` if it's new and public?
- Are there example files that should be in docs instead?
- Have I run the tests with `uv run pytest`?
- Are all assertions meaningful?
- Am I using `uv` for all operations?

## Related Guidelines

- **[DOCUMENTATION-AGENT.md](DOCUMENTATION-AGENT.md)**: Comprehensive documentation guidelines for docstrings, API docs, and guidested?
- Are there example files that should be in docs instead?
- Have I run the tests with `uv run pytest`?
- Are all assertions meaningful?
- Am I using `uv` for all operations?

---

**Remember:** Code without tests and documentation is incomplete code.
