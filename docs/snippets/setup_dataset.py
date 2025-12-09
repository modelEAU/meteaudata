"""
Setup for dataset examples with multiple signals.

Provides:
- dataset: A Dataset with temperature, pH, and dissolved oxygen signals
- temperature_signal, ph_signal, do_signal: Individual Signal objects
- temp_data, ph_data, do_data: pandas Series with raw data
- temp_provenance, ph_provenance, do_provenance: DataProvenance objects
- timestamps: DatetimeIndex for the data
- signals: Dictionary mapping signal names to Signal objects
"""

import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)

# Create multiple time series for complex examples
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')

# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

# pH data with longer cycle
ph_data = pd.Series(
    7.2 + 0.3 * np.sin(np.arange(100) * 2 * np.pi / 48) + np.random.normal(0, 0.1, 100),
    index=timestamps,
    name="RAW"
)

# Dissolved oxygen data with some correlation to temperature
do_data = pd.Series(
    8.5 - 0.1 * (temp_data - 20) + np.random.normal(0, 0.2, 100),
    index=timestamps,
    name="RAW"
)

# Temperature signal
temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="Thermocouple Type K",
    parameter="Temperature",
    purpose="Process monitoring",
    metadata_id="temp_001"
)
temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# pH signal
ph_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="pH Sensor v1.3",
    parameter="pH",
    purpose="Process monitoring",
    metadata_id="ph_001"
)
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=ph_provenance,
    units="pH units"
)

# Dissolved oxygen signal
do_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="DO Sensor v2.0",
    parameter="Dissolved Oxygen",
    purpose="Process monitoring",
    metadata_id="do_001"
)
do_signal = Signal(
    input_data=do_data,
    name="DissolvedOxygen",
    provenance=do_provenance,
    units="mg/L"
)

# Create signals dictionary for easy access
signals = {
    "temperature": temperature_signal,
    "ph": ph_signal,
    "dissolved_oxygen": do_signal
}

# Create a complete dataset
dataset = Dataset(
    name="reactor_monitoring",
    description="Multi-parameter monitoring of reactor R-101",
    owner="Process Engineer",
    purpose="Process control and optimization",
    project="Process Monitoring Study",
    signals={
        "temperature": temperature_signal,
        "ph": ph_signal,
        "dissolved_oxygen": do_signal
    }
)
