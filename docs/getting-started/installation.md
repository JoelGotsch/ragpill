# Installation

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Choosing a tracking backend (extras)

Since 0.5.0 the tracking backend is an **optional extra** — a bare
`pip install ragpill` ships no backend, and the first call that needs one raises
with an actionable message. Pick one:

```bash
pip install "ragpill[mlflow]"     # default backend (local or server)
pip install "ragpill[langfuse]"   # Langfuse (co-installable with mlflow)
pip install "ragpill[phoenix]"    # Arize Phoenix
```

!!! warning "Phoenix and MLflow are not co-installable"
    `ragpill[phoenix]` pulls a newer OpenTelemetry than `ragpill[mlflow]` pins,
    so the two extras cannot live in the same environment. Install one per env.
    Langfuse coexists with either. To select a non-default backend at runtime,
    call `ragpill.configure_backend(...)` (see the
    [Backends reference](../api/backends.md)).

## Installing with uv

The recommended way to install ragpill is using the `uv` package manager:

```bash
# Add to your project, with a backend extra
uv add "ragpill[mlflow]"
```

Or if you're installing from source:

```bash
# Clone the repository
git clone https://github.com/JoelGotsch/ragpill
cd ragpill

# Sync dependencies
uv sync
```

## Development Installation

See [contributing](../development/contributing.md#development-setup)

## Verification

To verify your installation, run:

```python
from importlib.metadata import version

print(version("ragpill"))
```

Or try creating a simple dataset:

```python
from ragpill.csv.testset import load_testset
from pathlib import Path

# If you have a CSV file with test cases
dataset = load_testset(Path("testset.csv"))
```

## Next Steps

- Check out the [Quick Start Guide](quickstart.md) to create your first evaluation
- Learn about [Loading TestSets from CSV](../guide/csv-adapter.md)
