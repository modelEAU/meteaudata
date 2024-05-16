import numpy as np
import pandas as pd
from meteaudata.processing_steps.univariate.interpolate import linear_interpolation
from meteaudata.processing_steps.univariate.resample import resample
from meteaudata.types import DataProvenance, Signal

sample_data = np.random.randn(100)
index = pd.date_range(start="2020-01-01", freq="6min", periods=100)

data = pd.Series(sample_data, name="RAW", index=index)
provenance = DataProvenance(
    source_repository="metEAUdata README snippet",
    project="metEAUdata",
    location="Primary clarifier effluent",
    equipment="S::CAN Spectro::lyser no:xyzxyz",
    parameter="Soluble Chemical Oxygen Demand",
    purpose="Demonstrating how metEAUdata works",
    metadata_id="xyz",
)
signal = Signal(input_data=data, name="CODs", provenance=provenance, units="mg/l")

# Add a processing step
signal.process(["CODs#1_RAW#1"], resample, "5min")

print(len(signal.time_series["CODs#1_RESAMPLED#1"].processing_steps))
# outputs 1

# Add another step to CODs_RESAMPLED
signal.process(["CODs#1_RESAMPLED#1"], linear_interpolation)
print(len(signal.time_series["CODs#1_LIN-INT#1"].processing_steps))
# outputs 2

# Save the resulting signal to a directory (data + metadata)
# signal.save("path/to/directory")

# Load a signal from a file
# signal = Signal.load_from_directory("path/to/directory/CODs.zip", "CODs")
