# Installation

`meteaudata` can be installed using various Python package managers. Choose the method that best fits your workflow.

## Requirements

- Python 3.9 or higher
- pandas >= 1.4
- pydantic >= 2.0, < 3.0

## Installation Methods

### Using pip

```bash
pip install meteaudata
```

### Using Poetry

If you're using Poetry for dependency management:

```bash
poetry add meteaudata
```

### Using uv

If you're using uv as your package manager:

```bash
uv add meteaudata
```

## Development Installation

If you want to contribute to meteaudata or need the latest development version:

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/your-username/meteaudata.git
cd meteaudata
```

### 2. Install with Development Dependencies

Using uv (recommended):

```bash
uv sync --group all
uv pip install -e .
```

Using pip:

```bash
pip install -e ".[dev,docs]"
```

### 3. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

## Verify Installation

To verify that meteaudata is installed correctly, try importing it:

```python
import meteaudata
print(meteaudata.__version__)  # Should print the version number
```

Or run a quick test:

```python
from meteaudata import Signal, DataProvenance
import pandas as pd
import numpy as np

# Create a simple signal
data = pd.Series(np.random.randn(10), name="test_data")
provenance = DataProvenance(
    source_repository="Installation test",
    project="meteaudata",
    location="Test location",
    equipment="Test equipment",
    parameter="Test parameter",
    purpose="Verify installation",
    metadata_id="test"
)

signal = Signal(
    input_data=data,
    name="test_signal", 
    provenance=provenance,
    units="test_units"
)

print(f"Signal created successfully: {signal.name}")
```

## Optional Dependencies

Some features require additional packages:

- **Visualization**: `plotly` and `ipywidgets` (included by default)
- **Jupyter Support**: `ipython` (included by default)  
- **Network Graphs**: `networkx` (included by default)

## Troubleshooting

### Common Issues

**Import Error**: If you get import errors, ensure you have the correct Python version (3.9+) and all dependencies are installed.

**Version Conflicts**: If you encounter dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv meteaudata-env
source meteaudata-env/bin/activate  # On Windows: meteaudata-env\Scripts\activate
pip install meteaudata
```

**Development Issues**: For development installations, make sure you have the latest version of your package manager:

```bash
# Update uv
uv self update

# Update pip
pip install --upgrade pip
```

## Next Steps

Once installed, check out the [Quick Start](quickstart.md) guide to begin using meteaudata, or learn about the [Basic Concepts](basic-concepts.md) behind the library.
