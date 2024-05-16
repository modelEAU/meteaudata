import numpy as np
import pandas as pd
from meteaudata.processing_steps.multivariate.average import average_signals
from meteaudata.types import DataProvenance, Dataset, Signal

sample_data = np.random.randn(100, 3)
index = pd.date_range(start="2020-01-01", freq="6min", periods=100)

data = pd.DataFrame(sample_data, columns=["CODs", "NH4-N", "TSS"], index=index)
provenance_cods = DataProvenance(
    source_repository="metEAUdata README snippet",
    project="metEAUdata",
    location="Primary clarifier effluent",
    equipment="S::CAN Spectro::lyser no:xxxx",
    parameter="Soluble Chemical Oxygen Demand",
    purpose="Demonstrating how metEAUdata signals work",
    metadata_id="xyz",
)
signal_cods = Signal(
    input_data=data["CODs"].rename("RAW"),
    name="CODs",
    provenance=provenance_cods,
    units="mg/l",
)
provenance_nh4n = DataProvenance(
    source_repository="metEAUdata README snippet",
    project="metEAUdata",
    location="Primary clarifier effluent",
    equipment="S::CAN Ammo::lyser no:yyyy",
    parameter="Ammonium Nitrogen",
    purpose="Demonstrating how metEAUdata signals work",
    metadata_id="xyz",
)
signal_nh4n = Signal(
    input_data=data["NH4-N"].rename("RAW"),
    name="NH4-N",
    provenance=provenance_nh4n,
    units="mg/l",
)
# Create the Dataset
dataset = Dataset(
    name="test dataset",
    description="a small dataset with randomly generated data",
    owner="Jean-David Therrien",
    purpose="Demonstrating how metEAUdata datasets work",
    project="metEAUdata",
    signals={"CODs": signal_cods, "NH4-N": signal_nh4n},
)

# create a new signal by applying a transformation to items in the dataset
dataset.process(["CODs#1_RAW#1", "NH4-N#1_RAW#1"], average_signals)

print(dataset.signals["AVERAGE#1"])
# outputs Signal(name="AVERAGE#1", ...)
# The new signal has its own raw time series
print(dataset.signals["AVERAGE#1"].time_series["AVERAGE#1_RAW#1"])
# outputs TimeSeries(..., processing_steps=[<list of all the processing steps that went into creating CODs, NH4-N, and the averaged signal>])

# Save the resulting signal to a directory (data + metadata)
# dataset.save("test directory")

# Load a signal from a file
# dataset = Dataset.load(
#    "test directory/test dataset.zip",  # path to the dataset directory or zip file
#    "test dataset",  # name of the dataset
# )
