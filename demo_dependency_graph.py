"""
Demo: cross-signal interactive dependency graph.

Three scenarios in one dataset:

  A) Divergence — RAW fans out into two resampled branches (5 min vs 10 min)
  B) Per-branch processing — each branch gets linear interpolation
  C) Cross-signal convergence — COD and NH4 interpolated series are averaged
     into a new AVERAGE signal via dataset.process()

Run:
    python demo_dependency_graph.py
"""
import numpy as np
import pandas as pd

from meteaudata.processing_steps.multivariate import average
from meteaudata.processing_steps.univariate import interpolate, replace, resample
from meteaudata.types import DataProvenance, Dataset, Signal

# ── Raw data ─────────────────────────────────────────────────────────────────

index = pd.date_range(start="2020-01-01", freq="6min", periods=200)
rng = np.random.default_rng(42)

cod_raw = pd.Series(rng.normal(50, 5, len(index)), index=index, name="RAW")
nh4_raw = pd.Series(rng.normal(10, 1, len(index)), index=index, name="RAW")

# Inject some gaps to make interpolation meaningful
cod_raw.iloc[30:40] = np.nan
nh4_raw.iloc[60:70] = np.nan

prov_cod = DataProvenance(
    source_repository="random",
    project="demo",
    location="reactor",
    equipment="probe",
    parameter="COD",
    purpose="dependency graph demo",
    metadata_id="cod-1",
)
prov_nh4 = DataProvenance(
    source_repository="random",
    project="demo",
    location="reactor",
    equipment="probe",
    parameter="NH4",
    purpose="dependency graph demo",
    metadata_id="nh4-1",
)

dataset = Dataset(
    name="demo",
    description="Dependency graph demo dataset",
    owner="demo",
    purpose="demo",
    project="demo",
    signals={
        "COD#1": Signal(input_data=cod_raw, name="COD#1", provenance=prov_cod, units="mg/L"),
        "NH4#1": Signal(input_data=nh4_raw, name="NH4#1", provenance=prov_nh4, units="mg/L"),
    },
)

# ── Scenario A: divergence — COD RAW → two resampled branches ────────────────

cod = dataset.signals["COD#1"]
cod = cod.process(["COD#1_RAW#1"], resample.resample, "5min",  output_names=["5MIN-RESAMPLED"])
cod = cod.process(["COD#1_RAW#1"], resample.resample, "10min", output_names=["10MIN-RESAMPLED"])
dataset.signals["COD#1"] = cod

# ── Scenario B: per-branch interpolation ─────────────────────────────────────

cod = cod.process(["COD#1_5MIN-RESAMPLED#1"],  interpolate.linear_interpolation, output_names=["LIN-INT-5MIN"])
cod = cod.process(["COD#1_10MIN-RESAMPLED#1"], interpolate.linear_interpolation, output_names=["LIN-INT-10MIN"])
dataset.signals["COD#1"] = cod

# NH4: simple single chain (for the convergence target)
nh4 = dataset.signals["NH4#1"]
nh4 = nh4.process(["NH4#1_RAW#1"], resample.resample, "5min")
nh4 = nh4.process(["NH4#1_RESAMPLED#1"], interpolate.linear_interpolation)
dataset.signals["NH4#1"] = nh4

# ── Scenario C: cross-signal convergence — average COD + NH4 ─────────────────

dataset = dataset.process(
    ["COD#1_LIN-INT-5MIN#1", "NH4#1_LIN-INT#1"],
    average.average_signals,
    check_units=False,
    output_signal_names=["AVG"],
)

# ── Summary ───────────────────────────────────────────────────────────────────

print("Signals in dataset:")
for sig_name, sig in dataset.signals.items():
    print(f"  {sig_name}: {list(sig.time_series.keys())}")

avg_ts_name = list(dataset.signals["AVG#1"].time_series.keys())[0]
print(f"\nAVERAGE time series name: {avg_ts_name}")

print("\n--- Scenario A+B: COD lineage for 5-min branch ---")
for dep in dataset.signals["COD#1"].build_dependency_graph("COD#1_LIN-INT-5MIN#1"):
    print(f"  {dep['origin']}  --[{dep['step']}]-->  {dep['destination']}")

print("\n--- Scenario C: full cross-signal lineage of the average ---")
for dep in dataset.build_dependency_graph(avg_ts_name):
    print(f"  {dep['origin']}  --[{dep['step']}]-->  {dep['destination']}")

# ── Interactive graphs ────────────────────────────────────────────────────────

print(f"\nOpening graph 1: lineage of {avg_ts_name} (cross-signal) ...")
dataset.plot_dependency_graph(avg_ts_name)

input("\nPress Enter to open graph 2: full dataset dependency graph ...")
dataset.plot_dependency_graph()
