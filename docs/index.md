# metEAUdata Documentation

**A lightweight package for tracking metadata about time series to create repeatable data pipelines.**

metEAUdata is a Python library designed for comprehensive management and processing of time series data, particularly focusing on environmental data analytics. It provides tools for detailed metadata handling, data transformations, and serialization of processing steps to ensure reproducibility and clarity in data manipulation workflows.

## Key Features

- **📊 Comprehensive Metadata Management** - Track the complete lineage of your time series data
- **🔄 Reproducible Processing Pipelines** - Every transformation is documented and repeatable
- **🧪 Environmental Data Focus** - Built specifically for research and monitoring applications
- **📈 Built-in Visualization** - Generate interactive plots and dependency graphs
- **💾 Serialization Support** - Save and load complete datasets with full metadata
- **🔗 Processing Step Tracking** - Maintain detailed records of all data transformations

## Quick Start

```python
import pandas as pd
from meteaudata.types import DataProvenance, Signal

# Create some sample data
data = pd.Series([20.1, 21.2, 22.3], name="RAW")

# Define data provenance
provenance = DataProvenance(
    source_repository="my_project",
    location="Laboratory A",
    equipment="Temperature Sensor #1",
    parameter="Air Temperature",
    purpose="Environmental monitoring"
)

# Create a signal
signal = Signal(
    input_data=data,
    name="temperature",
    units="°C",
    provenance=provenance
)

# Process the data
from meteaudata.processing_steps.univariate.resample import resample
signal.process(["temperature#1_RAW#1"], resample, "1min")

# Visualize
fig = signal.plot(["temperature#1_RAW#1", "temperature#1_RESAMPLED#1"])
fig.show()
```

## Core Concepts

### 🏗️ **Hierarchical Data Structure**

metEAUdata organizes your data in a three-level hierarchy:

- **Dataset** - A collection of related signals for a project
- **Signal** - A collection of time series from the same measurement source  
- **TimeSeries** - Individual time series with complete processing history

### 📋 **Metadata-First Approach**

Every piece of data includes comprehensive metadata:

- **Data Provenance** - Where did this data come from?
- **Processing Steps** - What transformations were applied?
- **Function Information** - Which functions were used and when?
- **Parameters** - What settings were used for each processing step?

### 🔄 **Processing Pipeline Tracking**

Build processing pipelines while automatically tracking:

- Input/output relationships between time series
- Function versions and parameters
- Processing timestamps
- Step-by-step transformation history

## Documentation Sections

### 🚀 [Getting Started](getting-started/installation.md)
Installation instructions, quick start guide, and basic concepts.

### 📖 [User Guide](user-guide/signals.md)
Comprehensive guides for working with signals, datasets, and processing pipelines.

### 📚 [Metadata Dictionary](metadata-dictionary/index.md)
Official definitions for all metadata attributes and data structures.

### 🔧 [API Reference](api-reference/index.md)
Complete API documentation generated from source code.

### 💡 [Examples](examples/basic-workflow.md)
Real-world examples and use cases.

### 🛠️ [Development](development/contributing.md)
Contributing guidelines and architecture documentation.

## Why metEAUdata?

Traditional data analysis often loses track of data lineage and processing steps. metEAUdata solves this by:

- **Preserving Context** - Never lose track of where your data came from
- **Ensuring Reproducibility** - Recreate any analysis with full parameter history
- **Facilitating Collaboration** - Share datasets with complete documentation
- **Supporting Quality Assurance** - Trace errors back to their source
- **Enabling Advanced Analysis** - Build complex pipelines with confidence

## Installation

```bash
pip install meteaudata
```

## Community

- **GitHub**: [modelEAU/meteaudata](https://github.com/modelEAU/meteaudata)
- **Issues**: [Report bugs or request features](https://github.com/modelEAU/meteaudata/issues)
- **Discussions**: [Community discussions](https://github.com/modelEAU/meteaudata/discussions)

## License

This project is licensed under the CC-BY-4.0 License - see the [LICENSE](https://github.com/modelEAU/meteaudata/blob/main/LICENSE) file for details.