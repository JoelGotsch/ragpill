# Installation

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installing with uv

The recommended way to install ragpill is using the `uv` package manager:

```bash
# Add to your project
uv add ragpill
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
import importlib
import ragpill
print(importlib.metadata.version("ragpill"))
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
