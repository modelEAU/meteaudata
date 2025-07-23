#!/usr/bin/env python3
"""
Script to create basic content files to fix MkDocs navigation warnings.
Run this once to set up the documentation structure.
"""

from pathlib import Path

def create_file_with_content(filepath: Path, title: str, content: str = ""):
    """Create a file with basic content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    basic_content = f"""# {title}

{content if content else f'This page contains documentation for {title.lower()}.'}

!!! note "Work in Progress"
    This section is currently being developed. Check back soon for updates.

## Overview

Coming soon...

## Examples

Coming soon...
"""
    
    with open(filepath, 'w') as f:
        f.write(basic_content)
    
    print(f"Created: {filepath}")


def main():
    """Create all missing content files."""
    
    # Base docs directory
    docs_dir = Path("docs")
    
    # Files to create with their titles and optional content
    files_to_create = [
        # Getting Started
        ("getting-started/installation.md", "Installation", """
## Requirements

- Python 3.9+
- pandas >= 1.4
- pydantic >= 2.0

## Installation

### Using pip

```bash
pip install git+https://github.com/modelEAU/meteaudata.git
```

### For Development

```bash
# Clone the repository
git clone https://github.com/modelEAU/meteaudata.git
cd meteaudata

# Install with uv (recommended)
uv sync --group all
uv pip install -e .

# Or with pip
pip install -e .
```
"""),
        
        ("getting-started/quickstart.md", "Quick Start", """
## Basic Usage

```python
import pandas as pd
from meteaudata.types import DataProvenance, Signal

# Create sample data
data = pd.Series([20.1, 21.2, 22.3], name="RAW")

# Define provenance
provenance = DataProvenance(
    parameter="Temperature",
    location="Lab A",
    equipment="Sensor #1"
)

# Create signal
signal = Signal(
    input_data=data,
    name="temperature",
    units="°C",
    provenance=provenance
)

# Process the data
from meteaudata.processing_steps.univariate.resample import resample
signal.process(["temperature#1_RAW#1"], resample, "1min")
```
"""),
        
        ("getting-started/basic-concepts.md", "Basic Concepts", """
## Hierarchical Data Structure

metEAUdata organizes data in three levels:

- **Dataset**: Collection of related signals
- **Signal**: Collection of time series from same source  
- **TimeSeries**: Individual time series with processing history

## Metadata-First Approach

Every data element includes comprehensive metadata for reproducibility.
"""),
        
        # User Guide
        ("user-guide/signals.md", "Working with Signals"),
        ("user-guide/datasets.md", "Managing Datasets"),
        ("user-guide/time-series.md", "Time Series Processing"),
        ("user-guide/processing-steps.md", "Processing Steps"),
        ("user-guide/visualization.md", "Visualization"),
        
        # API Reference
        ("api-reference/index.md", "API Reference", """
## Core Types

The main classes and types in metEAUdata.

### Data Containers
- [Dataset](../metadata-dictionary/dataset.md) - Collection of signals
- [Signal](../metadata-dictionary/signal.md) - Collection of time series
- [TimeSeries](../metadata-dictionary/time-series.md) - Individual time series

### Metadata
- [DataProvenance](../metadata-dictionary/data-provenance.md) - Data origin information
- [ProcessingStep](../metadata-dictionary/processing-step.md) - Processing operation metadata
- [Parameters](../metadata-dictionary/parameters.md) - Processing parameters

## Processing Functions

- [Univariate Processing](processing/univariate.md) - Single time series operations
- [Multivariate Processing](processing/multivariate.md) - Multi-signal operations
"""),
        
        ("api-reference/processing/univariate.md", "Univariate Processing Functions"),
        ("api-reference/processing/multivariate.md", "Multivariate Processing Functions"),
        ("api-reference/display.md", "Display System"),
        
        # Examples
        ("examples/basic-workflow.md", "Basic Workflow"),
        ("examples/custom-processing.md", "Custom Processing Functions"),
        ("examples/real-world-cases.md", "Real-world Use Cases"),
        
        # Development
        ("development/contributing.md", "Contributing", """
## Contributing to metEAUdata

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/modelEAU/meteaudata.git
cd meteaudata

# Install development dependencies
uv sync --group all
uv pip install -e .

# Set up pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# Run tests
uv run pytest

# Run documentation tests
uv run mkdocs build --strict
```
"""),
        
        ("development/architecture.md", "Architecture"),
        ("development/extending.md", "Extending metEAUdata"),
    ]
    
    # Create all files
    for filepath, title, *content in files_to_create:
        full_path = docs_dir / filepath
        file_content = content[0] if content else ""
        create_file_with_content(full_path, title, file_content)
    
    print(f"\nCreated {len(files_to_create)} documentation files!")
    print("You can now run: uv run mkdocs build")


if __name__ == "__main__":
    main()