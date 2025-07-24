"""
Execution contexts for meteaudata documentation code snippets.

This provides pre-built setups for common scenarios, allowing incomplete 
code snippets to be executed by providing necessary imports and setup.

The contexts are designed to be composable - higher-level contexts build
on lower-level ones to create rich, multi-object environments.
"""

import numpy as np
import pandas as pd

# Base context with common imports and setup
BASE_CONTEXT = """
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)
"""

# Individual building blocks
PROVENANCE_SETUP = """
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
"""

SIMPLE_DATA_SETUP = """
# Create simple time series data
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
data = pd.Series(np.random.randn(100) * 10 + 20, index=timestamps, name="RAW")
"""

MULTI_DATA_SETUP = """
# Create multiple time series for complex examples
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')

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
"""

PROBLEMATIC_DATA_SETUP = """
# Create data with issues for processing demonstrations
timestamps = pd.date_range('2024-01-01', periods=144, freq='30T')  # 30-min intervals for 3 days
base_values = 20 + 5 * np.sin(np.arange(144) * 2 * np.pi / 48) + np.random.normal(0, 0.5, 144)

# Introduce some missing values (simulate sensor issues)
missing_indices = np.random.choice(144, size=10, replace=False)
base_values[missing_indices] = np.nan

# Create some outliers
outlier_indices = np.random.choice(144, size=3, replace=False)  
base_values[outlier_indices] = base_values[outlier_indices] + 20

problematic_data = pd.Series(base_values, index=timestamps, name="RAW")
"""

SIGNAL_CREATION = """
# Create a simple signal
signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)
"""

MULTI_SIGNAL_CREATION = """
# Create multiple signals with different provenances

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
"""

DATASET_CREATION = """
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
"""

PROCESSED_SIGNAL_SETUP = """
# Apply some processing to demonstrate processed signals
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)
"""

CUSTOM_FUNCTION_IMPORTS = """
# Additional imports for custom processing function examples
from meteaudata.types import ProcessingStep, FunctionInfo, Parameters, ProcessingType
"""

# Composable context definitions
CONTEXTS = {
    # Basic building blocks
    "base": BASE_CONTEXT,
    
    "imports": BASE_CONTEXT,
    
    "provenance": BASE_CONTEXT + "\n\n" + PROVENANCE_SETUP,
    
    "simple_data": BASE_CONTEXT + "\n\n" + SIMPLE_DATA_SETUP,
    
    "multi_data": BASE_CONTEXT + "\n\n" + MULTI_DATA_SETUP,
    
    "problematic_data": BASE_CONTEXT + "\n\n" + PROBLEMATIC_DATA_SETUP,
    
    # Single signal contexts
    "simple_signal": BASE_CONTEXT + "\n\n" + PROVENANCE_SETUP + "\n\n" + SIMPLE_DATA_SETUP + "\n\n" + SIGNAL_CREATION,
    
    "processed_signal": BASE_CONTEXT + "\n\n" + PROVENANCE_SETUP + "\n\n" + SIMPLE_DATA_SETUP + "\n\n" + SIGNAL_CREATION + "\n\n" + PROCESSED_SIGNAL_SETUP,
    
    # Multi-signal contexts  
    "multi_signals": BASE_CONTEXT + "\n\n" + MULTI_DATA_SETUP + "\n\n" + MULTI_SIGNAL_CREATION,
    
    # Dataset contexts
    "dataset": BASE_CONTEXT + "\n\n" + MULTI_DATA_SETUP + "\n\n" + MULTI_SIGNAL_CREATION + "\n\n" + DATASET_CREATION,
    
    "simple_dataset": BASE_CONTEXT + "\n\n" + PROVENANCE_SETUP + "\n\n" + SIMPLE_DATA_SETUP + "\n\n" + SIGNAL_CREATION + "\n\n" + """
# Create a simple dataset with just one signal
dataset = Dataset(
    name="simple_monitoring",
    description="Single parameter monitoring example",
    owner="Data Analyst",
    purpose="Documentation example",
    project="Documentation Example",
    signals={"temperature": signal}
)
""",
    
    # Specialized contexts
    "visualization": BASE_CONTEXT + "\n\n" + PROVENANCE_SETUP + "\n\n" + SIMPLE_DATA_SETUP + "\n\n" + SIGNAL_CREATION + "\n\n" + PROCESSED_SIGNAL_SETUP,
    
    "processing": BASE_CONTEXT + "\n\n" + """
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
""" + "\n\n" + PROBLEMATIC_DATA_SETUP + "\n\n" + """
# Create signal with problematic data
signal = Signal(
    input_data=problematic_data,
    name="Temperature",
    provenance=processing_provenance,
    units="°C"
)
""",
    
    "custom_functions": BASE_CONTEXT + "\n\n" + CUSTOM_FUNCTION_IMPORTS + "\n\n" + PROVENANCE_SETUP + "\n\n" + SIMPLE_DATA_SETUP + "\n\n" + SIGNAL_CREATION,
    
    # Combined contexts for complex scenarios
    "full_environment": BASE_CONTEXT + "\n\n" + MULTI_DATA_SETUP + "\n\n" + MULTI_SIGNAL_CREATION + "\n\n" + DATASET_CREATION + "\n\n" + """
# Also create a simple signal for individual examples
simple_provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Example", 
    location="Demo Location",
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Documentation example",
    metadata_id="simple_example_001"
)

simple_data = pd.Series(np.random.randn(50) * 5 + 22, 
                       index=pd.date_range('2024-01-01', periods=50, freq='1H'), 
                       name="RAW")

signal = Signal(
    input_data=simple_data,
    name="SimpleTemperature",
    provenance=simple_provenance,
    units="°C"
)
""",
}

# Context composition helpers
def combine_contexts(*context_names: str) -> str:
    """
    Combine multiple contexts into one.
    
    Args:
        *context_names: Names of contexts to combine
        
    Returns:
        Combined context code
        
    Example:
        combined = combine_contexts("base", "provenance", "simple_data")
    """
    parts = []
    seen_parts = set()
    
    for context_name in context_names:
        if context_name not in CONTEXTS:
            available = ", ".join(CONTEXTS.keys())
            raise ValueError(f"Unknown context '{context_name}'. Available: {available}")
        
        context_code = CONTEXTS[context_name]
        # Simple deduplication - could be more sophisticated
        if context_code not in seen_parts:
            parts.append(context_code)
            seen_parts.add(context_code)
    
    return '\n\n'.join(parts)


def get_context(context_name: str) -> str:
    """Get the setup code for a specific context."""
    if context_name not in CONTEXTS:
        available = ", ".join(CONTEXTS.keys())
        raise ValueError(f"Unknown context '{context_name}'. Available contexts: {available}")
    
    return CONTEXTS[context_name]


def debug_context(context_name: str) -> None:
    """Debug what's in a context - useful for troubleshooting."""
    try:
        context_code = get_context(context_name)
        print(f"=== Context '{context_name}' ===")
        print(context_code)
        print(f"=== End of context '{context_name}' ===")
    except ValueError as e:
        print(f"Error: {e}")


def list_contexts() -> list:
    """List all available context names."""
    return list(CONTEXTS.keys())


def get_context_description(context_name: str) -> str:
    """Get a description of what a context provides."""
    descriptions = {
        # Basic building blocks
        "base": "Basic imports and setup for meteaudata",
        "imports": "Same as base - just the imports",
        "provenance": "Base imports + a standard DataProvenance object",
        "simple_data": "Base imports + simple time series data", 
        "multi_data": "Base imports + multiple time series (temp, pH, DO)",
        "problematic_data": "Base imports + data with gaps and outliers",
        
        # Single signal contexts
        "simple_signal": "Complete setup with one simple temperature signal",
        "processed_signal": "Simple signal + some processing steps applied",
        
        # Multi-signal contexts
        "multi_signals": "Multiple signals (temperature, pH, DO) in a dictionary",
        
        # Dataset contexts
        "dataset": "Complete dataset with multiple signals",
        "simple_dataset": "Dataset with just one signal",
        
        # Specialized contexts
        "visualization": "Signal with processing, ready for plotting examples", 
        "processing": "Signal with problematic data for processing demonstrations",
        "custom_functions": "Setup for creating and testing custom processing functions",
        
        # Combined contexts
        "full_environment": "Dataset + individual signal + all variables for complex examples",
    }
    return descriptions.get(context_name, "No description available")


def get_context_dependencies(context_name: str) -> list:
    """
    Get what objects/variables a context provides.
    
    This helps users understand what will be available after using a context.
    """
    dependencies = {
        "base": ["np", "pd", "meteaudata imports"],
        "imports": ["np", "pd", "meteaudata imports"],
        "provenance": ["np", "pd", "meteaudata imports", "provenance"],
        "simple_data": ["np", "pd", "meteaudata imports", "timestamps", "data"],
        "multi_data": ["np", "pd", "meteaudata imports", "timestamps", "temp_data", "ph_data", "do_data"],
        "problematic_data": ["np", "pd", "meteaudata imports", "timestamps", "problematic_data"],
        
        "simple_signal": ["np", "pd", "meteaudata imports", "provenance", "timestamps", "data", "signal"],
        "processed_signal": ["np", "pd", "meteaudata imports", "provenance", "timestamps", "data", "signal (with processing)"],
        
        "multi_signals": ["np", "pd", "meteaudata imports", "timestamps", "temp_data", "ph_data", "do_data", 
                         "temperature_signal", "ph_signal", "do_signal", "signals dict"],
        
        "dataset": ["np", "pd", "meteaudata imports", "timestamps", "temp_data", "ph_data", "do_data", 
                   "temperature_signal", "ph_signal", "do_signal", "signals dict", "dataset"],
        "simple_dataset": ["np", "pd", "meteaudata imports", "provenance", "timestamps", "data", "signal", "dataset"],
        
        "visualization": ["np", "pd", "meteaudata imports", "provenance", "timestamps", "data", "signal (processed)"],
        "processing": ["np", "pd", "meteaudata imports", "processing_provenance", "timestamps", "problematic_data", "signal"],
        "custom_functions": ["np", "pd", "meteaudata imports", "processing types", "provenance", "timestamps", "data", "signal"],
        
        "full_environment": ["np", "pd", "meteaudata imports", "timestamps", "temp_data", "ph_data", "do_data",
                           "temperature_signal", "ph_signal", "do_signal", "signals dict", "dataset", 
                           "simple_provenance", "simple_data", "signal"],
    }
    return dependencies.get(context_name, [])


# Usage examples and documentation
USAGE_EXAMPLES = {
    "progressive_signal_building": """
# Example: Build up a signal progressively
```python exec="base"
# Start with just imports
```

```python exec="continue" 
# Add provenance
provenance = DataProvenance(...)
```

```python exec="continue"
# Add data  
data = pd.Series(...)
```

```python exec="continue"
# Create signal
signal = Signal(input_data=data, name="Temperature", provenance=provenance, units="°C")
```
""",
    
    "use_full_context": """
# Example: Use a complete context for complex examples
```python exec="dataset"
# Now you have dataset, all signals, and all data available
print(f"Dataset has {len(dataset.signals)} signals")
```

```python exec="continue"  
# Continue building on the established environment
for name, signal in dataset.signals.items():
    print(f"Signal {name} has {len(signal.time_series)} time series")
```
""",
    
    "mix_contexts_carefully": """
# Example: Be careful when mixing contexts
```python exec="simple_signal"
# Creates: signal (simple temperature signal)
```

# DON'T do this - it will overwrite the signal:
# ```python exec="dataset"  
# # This creates different signals and dataset, losing the simple signal
# ```

# DO this instead:
```python exec="continue"
# Build on the existing signal
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
```
""",
}


def show_usage_examples():
    """Print usage examples for context composition."""
    print("Context Usage Examples:")
    print("=" * 50)
    for example_name, example_code in USAGE_EXAMPLES.items():
        print(f"\n{example_name}:")
        print(example_code)


if __name__ == "__main__":
    print("Available contexts:")
    for name in sorted(CONTEXTS.keys()):
        deps = get_context_dependencies(name)
        desc = get_context_description(name)
        print(f"\n{name:20s} - {desc}")
        indent = ' ' * 20
        print(f"{indent}   Provides: {', '.join(deps)}")
    
    print("\n" + "="*80)
    show_usage_examples()