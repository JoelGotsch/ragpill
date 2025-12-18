# Documentation Agent Guidelines

This document provides comprehensive guidelines for AI agents working on documentation for the `ragpill` project.

**When to use this guide:** Follow these guidelines when:
- Writing or updating docstrings in source code
- Creating or updating documentation files in `docs/`
- Adding API documentation
- Writing guides or tutorials
- Creating cross-references between documentation

In particular, this should happen whenever a function changes its signature, a new function/ class is added, or existing documentation needs improvement.

## Core Documentation Philosophy

**Single Source of Truth:** Docstrings in source code are the authoritative source. Documentation files reference these docstrings using mkdocstrings.

**Benefits:**
- Documentation lives close to code
- Easier to keep in sync (updated when code changes)
- No duplicate maintenance
- Auto-generated API docs are always accurate
- Automatic hyperlinks between related documentation

## MkDocs Setup

This project uses **MkDocs** with `mkdocs.yml` at the project root and `docs/` as the documentation directory.

### Directory Organization

```
ragpill/
├── mkdocs.yml            # MkDocs configuration file (at project root)
├── docs/                 # All documentation (single source of truth)
│   ├── index.md          # Homepage
│   ├── api/              # API reference (uses ::: directives)
│   │   ├── base.md       # Base classes
│   │   ├── csv.md        # CSV module
│   │   ├── evaluators.md # Evaluators
│   │   ├── mlflow.md     # MLflow integration
│   │   ├── environment.md# Environment settings
│   │   └── utils.md      # Utilities
│   ├── guide/            # User guides (links to API, no ::: directives)
│   │   ├── csv-adapter.md
│   │   ├── evaluators.md
│   │   ├── overview.md
│   │   └── testsets.md
│   ├── tutorials/        # Tutorial notebooks
│   │   ├── full.ipynb
│   │   └── ...
│   ├── how-to/           # How-to notebooks
│   │   └── custom-type-evaluator.ipynb
│   ├── getting-started/
│   │   ├── installation.md
│   │   └── quickstart.md
│   └── development/
│       ├── contributing.md
│       └── roadmap.md
└── src/ragpill/         # Source code with docstrings
    ├── base.py
    ├── evaluators.py
    ├── csv/testset.py
    └── ...
```

### MkDocs Commands

```bash
# Always run from the project root
# Serve documentation locally (with live reload)
uv run mkdocs serve
# Visit http://localhost:8000

# Build static site
uv run mkdocs build

# Build with strict mode (fails on warnings)
uv run mkdocs build --strict

# Clean build (remove old files first)
uv run mkdocs build --clean

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

### Adding New Documentation

**Step 1:** Create content in `docs/`

```bash
# Create the actual documentation
echo "# New Feature" > docs/guide/new-feature.md
```

**Step 2:** Add to navigation in `mkdocs.yml`

```yaml
nav:
  - Guide:
      - New Feature: guide/new-feature.md
```

**Step 3:** Build and verify

```bash
uv run mkdocs serve
```

### Troubleshooting

**If changes aren't showing up:**

```bash
# Stop mkdocs serve (Ctrl+C), then:
rm -rf site
uv run mkdocs build --clean
uv run mkdocs serve
```

## CRITICAL RULES

### Rule 1: API Documentation Location

**Generated documentation (`::: module.path`) should ONLY be in the `docs/api/` section.**

```markdown
# ✅ CORRECT - In docs/api/csv.md
::: ragpill.csv.testset.load_testset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
```

```markdown
# ❌ WRONG - In docs/guide/csv-adapter.md
::: ragpill.csv.testset.load_testset
    options:
      show_root_heading: false
```

### Rule 2: Complete API Coverage

**The API section must contain ALL public functions and methods.**

Public = everything that doesn't start with `_`

For each module file, the API documentation must include:
- All public functions (e.g., `load_testset`, `merge_settings`)
- All public classes (e.g., `BaseEvaluator`, `LLMJudge`)
- All public constants/variables (e.g., `default_evaluator_classes`)

### Rule 3: Guides Link to API

**In guides and tutorials, ONLY link to the API section. Never use `:::` directives.**

```markdown
# ✅ CORRECT - In docs/guide/csv-adapter.md
For complete API reference, see [`load_testset`](../api/csv.md#load_testset)
in the [CSV API Documentation](../api/csv.md).

## Loading Test Sets

To load a test set from CSV, use the [`load_testset`](../api/csv.md#load_testset)
function:

```python
from ragpill.csv.testset import load_testset
```
```

## Docstring Format

### Google-Style Docstrings

All docstrings MUST use Google-style format:

```python
def load_testset(
    csv_path: str | Path,
    evaluator_classes: dict[str, Type[BaseEvaluator]] = default_evaluator_classes,
    skip_unknown_evaluators: bool = False,
) -> Dataset[str, str, TestCaseMetadata]:
    """Create a Dataset from a CSV file with evaluator configurations.

    This function reads a CSV file and creates test cases with evaluators.
    Multiple rows with the same question are grouped into a single test case
    with multiple evaluators.

    CSV Format:
        The CSV file should contain the following standard columns:

        - **Question**: The input question/prompt for the test case
        - **test_type**: Name of the evaluator class
        - **expected**: Boolean indicating expected pass/fail
        - **mandatory**: Whether this check is mandatory
        - **tags**: Comma-separated tags
        - **check**: Evaluation criteria

    Args:
        csv_path: Path to the CSV file
        evaluator_classes: Dictionary mapping test_type names to evaluator classes.
            Extend default_evaluator_classes with custom evaluators:
            `default_evaluator_classes | {'MyEval': MyEvaluator}`
        skip_unknown_evaluators: If True, skip rows with unknown evaluator types
            instead of raising an error

    Returns:
        Dataset with Cases grouped by question, each Case having multiple evaluators

    Raises:
        ValueError: If unknown evaluator type found and skip_unknown_evaluators is False
        RuntimeError: If CSV file cannot be read with supported encodings

    Example:
        ```python
        from ragpill.csv.testset import load_testset, default_evaluator_classes

        dataset = load_testset(
            csv_path='testset.csv',
            evaluator_classes=default_evaluator_classes
        )
        ```

    Note:
        Any additional columns (beyond the standard ones) will be passed to
        evaluators as attributes via **kwargs in from_csv_line().

    See Also:
        [`ragpill.base.BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]:
            Detailed descriptions of standard parameters
        [`ragpill.csv.testset.default_evaluator_classes`][ragpill.csv.testset.default_evaluator_classes]:
            Dict of built-in evaluators
    """
```

### Required Docstring Sections

Every public function/class must have:

1. **Summary**: Brief one-line description (first line)
2. **Extended Description**: Detailed explanation (optional, after blank line)
3. **Args**: All parameters with descriptions
4. **Returns**: What the function returns
5. **Raises**: Exceptions that can be raised (if applicable)
6. **Example**: Code example showing usage (highly recommended)
7. **Note**: Additional important information (optional)
8. **See Also**: Links to related functions/classes (recommended)

### Docstring Formatting Standards

- **First line**: Brief summary (one sentence, no period)
- **Blank line** after summary
- **Extended description**: Multiple paragraphs explaining details
- **Section headers**: `Args:`, `Returns:`, `Raises:`, `Example:`, `Note:`, `See Also:`
- **Indentation**: 4 spaces for section content
- **Code blocks**: Use triple backticks with language identifier
- **Lists**: Use `-` for bullet points, maintain indentation
- **Bold**: Use `**text**` for column names, important terms
- **Inline code**: Use backticks for parameter names, values, types

## Cross-References in Docstrings

### Internal Links in Docstrings

Use markdown link syntax with full module paths:

```python
def load_testset(...):
    """Create a Dataset from a CSV file.

    For detailed parameter descriptions, see
    [`BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line].

    See Also:
        [`ragpill.base.BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]:
            Standard parameter documentation
        [`ragpill.evaluators.LLMJudge`][ragpill.evaluators.LLMJudge]:
            Built-in LLM-based evaluator
    """
```

### Link Patterns

**Link to function:**
```python
[`load_testset`][ragpill.csv.testset.load_testset]
```

**Link to class:**
```python
[`BaseEvaluator`][ragpill.base.BaseEvaluator]
```

**Link to method:**
```python
[`BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]
```

**Link to class attribute:**
```python
[`default_evaluator_classes`][ragpill.csv.testset.default_evaluator_classes]
```

### Bidirectional Navigation

Create bidirectional links between related functions:

```python
# In base.py
class BaseEvaluator:
    """Base class for evaluators.

    See Also:
        [`load_testset`][ragpill.csv.testset.load_testset]:
            Create datasets from CSV files
    """

# In csv/testset.py
def load_testset(...):
    """Create a Dataset from a CSV file.

    See Also:
        [`BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]:
            Standard parameter documentation
    """
```

## API Documentation Files

### Template for API Module Documentation

```markdown
# Module Name

Brief description of the module's purpose.

## function_or_class_name

::: module.path.function_or_class_name
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## another_function_or_class

::: module.path.another_function_or_class
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
```

### Example: docs/api/csv.md

```markdown
# CSV Module

The CSV module provides functionality for loading test sets from CSV files.

## load_testset

::: ragpill.csv.testset.load_testset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## create_evaluator_from_row

::: ragpill.csv.testset.create_evaluator_from_row
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## default_evaluator_classes

::: ragpill.csv.testset.default_evaluator_classes
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
```

### mkdocstrings Options

Standard options for all API entries:

```markdown
::: module.path.name
    options:
      show_root_heading: true      # Show the function/class name as heading
      show_source: true            # Show source code
      heading_level: 3             # Use h3 (###) for the heading
```

Optional settings (use when needed):

```markdown
    options:
      members:                      # Only show specific members
        - method_name
        - another_method
      show_signature: true          # Show function signature separately
      separate_signature: true      # Put signature on separate line
      show_if_no_docstring: false   # Hide items without docstrings
```

## Guide Documentation Files

### Template for Guide Documentation

```markdown
# Guide Title

Brief introduction to the guide topic and components, the principles and ideas behind it, and how it relates to other components. For all of the components, reference the API documentation.

## Section 1

Explanation with examples.

Use the [`function_name`](../api/module.md#function_name) function:

```python
from ragpill.module import function_name

result = function_name(param="value")
```

## Section 2

More explanations. Reference other functions using
[`another_function`](../api/module.md#another_function).

## See Also

- [API Documentation](../api/module.md)
- [Related Tutorial](../tutorials/tutorial-name.ipynb)
```

### Guide Best Practices

1. **Use markdown links**: Link to API docs, never use `:::` directives
1. **Provide context**: Explain WHY and HOW, not just WHAT
1. **Include examples**: Show realistic usage patterns
1. **Link frequently**: Reference API docs when mentioning functions
1. **Progressive complexity**: Start simple, build to advanced usage

## Example Code in Documentation

### Code Blocks in Docstrings

```python
def my_function():
    """Do something useful.

    Example:
        Basic usage:

        ```python
        from ragpill import my_function

        result = my_function()
        print(result)
        ```

        Advanced usage with parameters:

        ```python
        result = my_function(
            param1="value",
            param2=42,
        )
        ```
    """
```

### Code Blocks in Guides

````markdown
## Usage Example

Here's how to use the function:

```python
from ragpill.csv.testset import load_testset, default_evaluator_classes

# Load dataset
dataset = load_testset(
    csv_path="testset.csv",
    evaluator_classes=default_evaluator_classes,
)

# Use the dataset
print(f"Loaded {len(dataset.cases)} test cases")
```

For details on parameters, see [`load_testset`](../api/csv.md#load_testset).
````

## Documentation Update Workflow

### When Adding a New Function

1. **Write comprehensive docstring** in source code
   - Include all required sections
   - Add cross-references to related functions
   - Provide clear examples

2. **Add to API documentation**
   - Add `:::` directive to appropriate `docs/api/*.md` file
   - Maintain alphabetical or logical ordering

3. **Update guides** (if user-facing)
   - Add section explaining usage in `docs/guide/*.md`
   - Link to API documentation
   - Provide context and examples

4. **Update tutorials** (if significant feature)
   - Add to relevant Jupyter notebook in `docs/tutorials/`
   - Show end-to-end usage

5. **Rebuild documentation** to verify
   ```bash
   uv run mkdocs serve
   ```

### When Updating an Existing Function

1. **Update docstring** in source code
2. **Review API docs** - Usually auto-updates via mkdocstrings
3. **Update guides** - Check if examples need updating
4. **Update tutorials** - Verify notebooks still work

### When Changing Function Signature

1. **Update docstring** with new parameters
2. **Update all examples** in docstring
3. **Update guides** with new usage patterns
4. **Update tutorials** notebooks
5. **Add migration note** if breaking change

## Common Documentation Patterns

### Documenting CSV Configuration

```python
def function_with_csv_config():
    """Function that uses CSV configuration.

    CSV Format:
        The CSV file should contain the following columns:

        - **column_name**: Description of this column
        - **another_column**: Description of another column
        - **optional_column**: Optional column description (default: value)

        Example CSV:
        ```csv
        column_name,another_column,optional_column
        value1,value2,value3
        value4,value5,value6
        ```
    """
```

### Documenting Settings/Configuration

```python
class Settings(BaseSettings):
    """Configuration settings for the module.

    Settings are loaded from environment variables with the prefix `PREFIX_`.

    Environment Variables:
        - `PREFIX_API_KEY`: API key for authentication (required)
        - `PREFIX_BASE_URL`: Base URL for API (required)
        - `PREFIX_TIMEOUT`: Request timeout in seconds (default: 30)

    Example:
        ```bash
        export PREFIX_API_KEY=your_key_here
        export PREFIX_BASE_URL=https://api.example.com
        export PREFIX_TIMEOUT=60
        ```

        ```python
        from ragpill.module import Settings

        settings = Settings()
        print(settings.api_key)
        ```
    """
```

### Documenting Return Values

```python
def complex_return():
    """Function returning complex data.

    Returns:
        pandas.DataFrame: Evaluation results with columns:
            - `input`: Test case input
            - `output`: Task output
            - `passed`: Boolean indicating pass/fail
            - `reason`: Explanation for the result
            - `duration`: Time taken in seconds
    """
```

## Quality Checklist for Documentation

Before considering documentation complete:

- [ ] All public functions have docstrings
- [ ] Docstrings use Google-style format
- [ ] All required sections present (Args, Returns, Example)
- [ ] Cross-references to related functions added
- [ ] API documentation updated in `docs/api/`
- [ ] Guide documentation uses markdown links (no `:::`)
- [ ] Examples are realistic and runnable
- [ ] Code blocks use correct language identifiers
- [ ] Links are valid and point to correct locations
- [ ] No duplicate documentation (single source of truth)
- [ ] Terminology is consistent across all docs

## Common Documentation Mistakes

### ❌ DON'T

**Don't use `:::` directives in guides:**
```markdown
# In docs/guide/csv-adapter.md - WRONG!
::: ragpill.csv.testset.load_testset
```

**Don't duplicate parameter descriptions:**
```markdown
# In docs/guide/csv-adapter.md - WRONG!
## Parameters

- csv_path: Path to the CSV file
- evaluator_classes: Dictionary of evaluators
...
```

**Don't use old-style docstrings:**
```python
# WRONG!
def function(param):
    """
    :param param: Description
    :type param: str
    :return: Result
    :rtype: bool
    """
```

**Don't forget cross-references:**
```python
# WRONG - No links to related functions
def load_testset(...):
    """Create a Dataset from a CSV file.

    Args:
        csv_path: Path to CSV file
    """
```

### ✅ DO

**Use markdown links in guides:**
```markdown
# In docs/guide/csv-adapter.md - CORRECT!
For details, see [`load_testset`](../api/csv.md#load_testset).
```

**Reference API docs instead of duplicating:**
```markdown
# In docs/guide/csv-adapter.md - CORRECT!
For parameter descriptions, see the [CSV API Documentation](../api/csv.md).
```

**Use Google-style docstrings:**
```python
# CORRECT!
def function(param: str) -> bool:
    """Do something useful.

    Args:
        param: Description of parameter

    Returns:
        Boolean indicating success
    """
```

**Include cross-references:**
```python
# CORRECT!
def load_testset(...):
    """Create a Dataset from a CSV file.

    Args:
        csv_path: Path to CSV file

    See Also:
        [`BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]:
            Standard parameter documentation
    """
```

## Tools and Commands

### Building Documentation Locally

```bash
# Run from project root
# Serve documentation locally with live reload
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Build with strict mode (fails on warnings)
uv run mkdocs build --strict

# Clean build (remove old files first)
uv run mkdocs build --clean

# Open in browser
# Navigate to http://localhost:8000
```

### Deploying Documentation

```bash
# Deploy to GitHub Pages (run from project root)
uv run mkdocs gh-deploy

# Deploy to specific branch
uv run mkdocs gh-deploy --force
```

### Checking Documentation

```bash
# Check for errors during build
uv run mkdocs build --strict

# Run pytest with doctest
uv run pytest --doctest-modules src/
```

---

**Remember:** Good documentation is as important as good code. Documentation should be clear, accurate, and always kept in sync with the code.
