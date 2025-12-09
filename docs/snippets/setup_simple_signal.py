"""
Setup for simple signal examples.

Provides:
- signal: A Temperature signal with 100 hourly data points
- provenance: DataProvenance object for the signal
- data: pandas Series with the raw temperature data
- timestamps: DatetimeIndex for the data
"""

import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation, subset, replace_ranges

# Set random seed for reproducible examples
np.random.seed(42)

# Create a standard provenance for examples
provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Example",
    location="Demo Location",
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Documentation example",
    metadata_id="doc_example_001"
)

# Create simple time series data
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
data = pd.Series(np.random.randn(100) * 10 + 20, index=timestamps, name="RAW")

# Create a simple signal
signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)
