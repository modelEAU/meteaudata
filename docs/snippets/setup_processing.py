"""
Setup for processing examples with problematic data.

Provides:
- signal: A Temperature signal with gaps and outliers
- processing_provenance: DataProvenance object for the signal
- problematic_data: pandas Series with gaps and outliers
- timestamps: DatetimeIndex for the data
"""

import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation, subset, replace_ranges

# Set random seed for reproducible examples
np.random.seed(42)

# Create provenance for processing examples
processing_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Data Quality Study",
    location="Sensor Station A",
    equipment="Smart Sensor v3.0",
    parameter="Temperature",
    purpose="Demonstrate processing capabilities",
    metadata_id="processing_demo_001"
)

# Create data with issues for processing demonstrations
timestamps = pd.date_range('2024-01-01', periods=144, freq='30min')  # 30-min intervals for 3 days
base_values = 20 + 5 * np.sin(np.arange(144) * 2 * np.pi / 48) + np.random.normal(0, 0.5, 144)

# Introduce some missing values (simulate sensor issues)
missing_indices = np.random.choice(144, size=10, replace=False)
base_values[missing_indices] = np.nan

# Create some outliers
outlier_indices = np.random.choice(144, size=3, replace=False)
base_values[outlier_indices] = base_values[outlier_indices] + 20

problematic_data = pd.Series(base_values, index=timestamps, name="RAW")

# Create signal with problematic data
signal = Signal(
    input_data=problematic_data,
    name="Temperature",
    provenance=processing_provenance,
    units="°C"
)
