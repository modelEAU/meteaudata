# Executable Code in Documentation

This guide explains how meteaudata's documentation system supports executable code blocks that run at build time and inject live outputs directly into the documentation.

## Overview

The meteaudata documentation includes an executable code system that:

- **Runs actual Python code** during documentation build
- **Captures real outputs** including print statements, plots, and HTML displays
- **Embeds interactive content** like Plotly charts and meteaudata rich displays
- **Maintains context** across multiple code blocks for realistic examples
- **Provides pre-built scenarios** to demonstrate meteaudata functionality

## Basic Usage

### Standard Code Blocks vs Executable Blocks

**Standard code block (static):**
```python
# This code is just displayed, not executed
signal = Signal(data, "Temperature", provenance, "°C")
print(f"Created signal: {signal.name}")
```

**Executable code block:**
```python
# This code actually runs during build and shows real output
signal = Signal(data, "Temperature", provenance, "°C")
print(f"Created signal: {signal.name}")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpwaxwx9rc.py", line 152, in <module>
    signal = Signal(data, "Temperature", provenance, "°C")
NameError: name 'data' is not defined
```

### Using Execution Contexts

**With setup context:**
```python
# Uses pre-created signal, no setup needed
print(f"Signal: {signal.name}")
print(f"Data points: {len(signal.time_series)}")
```

**Output:**
```
Signal: Temperature#1
Data points: 1
```

**Continuing from previous code:**
```python
# Continues from the previous code block's variables
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
print("Processing applied!")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpkva94dix.py", line 161, in <module>
    signal = Signal(data, "Temperature", provenance, "°C")
NameError: name 'data' is not defined
```

## Available Execution Contexts

The system provides several pre-built contexts for common scenarios:

### `simple_signal`
- **Use case**: Basic signal examples and introductory content
- **Provides**: A Temperature signal with 100 hourly data points
- **Time series**: `Temperature#1_RAW#1`
- **Best for**: Getting started guides, basic operations

### `processed_signal`
- **Use case**: Demonstrating processing pipelines
- **Provides**: Temperature signal with resampling and interpolation already applied
- **Time series**: `Temperature#1_RAW#1`, `Temperature#1_RESAMPLED#1`, `Temperature#1_LIN-INT#1`
- **Best for**: Processing examples, intermediate tutorials

### `multi_signal`
- **Use case**: Multi-parameter monitoring examples
- **Provides**: Temperature and pH signals in a `signals` dictionary
- **Signals**: `signals["temperature"]`, `signals["ph"]`
- **Best for**: Multi-variate analysis, comparison examples

### `dataset`
- **Use case**: Complete dataset workflows
- **Provides**: A `dataset` with temperature and pH signals
- **Structure**: Full Dataset object with metadata
- **Best for**: Dataset operations, complex workflows

### `visualization`
- **Use case**: Plotting and display examples
- **Provides**: Signal with processing applied for rich visualizations
- **Features**: Pre-configured for all visualization methods
- **Best for**: Plotting guides, display system demos

### `processing`
- **Use case**: Advanced processing workflows
- **Provides**: Signal with gaps, outliers, and realistic data issues
- **Features**: Includes missing values and data quality challenges
- **Best for**: Quality control, advanced processing examples

### `custom_functions`
- **Use case**: Creating custom processing functions
- **Provides**: Test signal and processing utilities
- **Features**: Includes ProcessingStep and FunctionInfo imports
- **Best for**: Advanced users, custom development

## Content Types Captured

### Text Output
```python
print(f"Signal created: {signal.name}")
print(f"Units: {signal.units}")
print(f"Time series count: {len(signal.time_series)}")
```

**Output:**
```
Signal created: Temperature#1
Units: °C
Time series count: 1
```

### Interactive Plots
```python
# Generates actual Plotly plots embedded as HTML
fig = signal.plot(["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"])
print("Interactive plot generated")
```

**Output:**
```
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_3b2b3e83.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_3b2b3e83.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_3b2b3e83.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_3b2b3e83.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_3b2b3e83.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata signal_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_3b2b3e83.html
Interactive plot generated
```

<iframe src="../assets/generated/meteaudata_timeseries_plot_3b2b3e83.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="../assets/generated/meteaudata_signal_plot_3b2b3e83.html" width="100%" height="500" frameborder="0"></iframe>

### Rich HTML Displays
```python
# Captures meteaudata's rich HTML display system
signal.display(format='html', depth=2)
```

**Output:**
```
Captured HTML display: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/display_content_f59e793e_1.html
<IPython.core.display.HTML object>
```

<iframe src="../assets/generated/display_content_f59e793e_1.html" width="100%" height="500" frameborder="0"></iframe>

### Processing Outputs
```python
# Shows real processing steps and metadata
signal.process(["Temperature#1_RAW#1"], resample, frequency="5min")
print(f"Applied processing: {len(signal.time_series)} time series now available")
```

**Output:**
```
Applied processing: 2 time series now available
```

## Technical Implementation

### Code Execution
- Uses `uv run python` for consistent environment
- Executes in isolated temporary files
- Captures stdout, stderr, and generated files
- Handles imports and dependency management

### Plot Generation
- Intercepts `plot()` method calls from meteaudata objects
- Saves Plotly figures as HTML files
- Attempts PNG export (fallback to HTML if kaleido unavailable)
- Embeds plots using iframe elements

### HTML Content Capture
- Monitors `display()` method calls with `format='html'`
- Uses meteaudata's internal `_build_html_content()` method
- Saves rich HTML to standalone files
- Embeds using iframe for interactive exploration

### Context Management
- Maintains variable scope across code blocks using `exec="continue"`
- Pre-builds execution contexts with common setups
- Injects setup code before user code execution
- Ensures reproducible examples with fixed random seeds

## File Organization

### Generated Assets
```
docs/assets/generated/
├── meteaudata_signal_plot_*.html      # Signal plots
├── meteaudata_timeseries_plot_*.html  # Time series plots  
├── meteaudata_dataset_plot_*.html     # Dataset plots
└── display_content_*.html             # Rich HTML displays
```

### Processing Scripts
```
docs/scripts/
├── exec_processor.py        # Main execution engine
├── exec_contexts.py         # Pre-built execution contexts
└── process_executable_docs.py  # MkDocs integration
```

## Best Practices

### Writing Executable Examples

**DO:**
- Use appropriate execution contexts for your content level
- Keep code blocks focused and demonstrative
- Include meaningful print statements for output
- Test examples manually before committing

**DON'T:**
- Rely on external files or network resources
- Use overly complex examples that obscure the main point
- Forget to specify execution context when needed
- Mix unrelated concepts in single code blocks

### Context Selection
- **Introductory content**: Use `simple_signal` or `basic`
- **Processing tutorials**: Use `processing` or `processed_signal`
- **Visualization guides**: Use `visualization`
- **Advanced workflows**: Use `dataset` or `multi_signal`
- **Custom development**: Use `custom_functions`

### Error Handling
- Code that fails to execute shows error output in documentation
- Use try/except blocks for expected failures
- Test all executable code blocks before publishing
- Check that context variables are available

## Integration with MkDocs

The executable code system integrates seamlessly with the existing MkDocs workflow:

1. **Build-time processing**: Runs automatically during `mkdocs build`
2. **Gen-files integration**: Uses mkdocs-gen-files plugin architecture
3. **Asset management**: Generated files stored in `docs/assets/generated/`
4. **Version control**: Generated assets can be committed for reproducibility

## Example Workflows

### Basic Tutorial Pattern
```python
# Introduction with pre-built signal
print(f"Working with signal: {signal.name}")
```

**Output:**
```
Working with signal: Temperature#1
```

```python
# Build on previous context
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
print("Processing applied successfully")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpwp0hp9_9.py", line 161, in <module>
    signal = Signal(data, "Temperature", provenance, "°C")
NameError: name 'data' is not defined
```

```python
# Continue building complexity
signal.display(format='html')
print("Rich display generated")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmp83ggc0cn.py", line 161, in <module>
    signal = Signal(data, "Temperature", provenance, "°C")
NameError: name 'data' is not defined
```

### Visualization Showcase
```python
# Show plotting capabilities
fig = signal.plot(["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"])
signal.plot_dependency_graph("Temperature#1_LIN-INT#1")
print("Multiple plots generated")
```

**Output:**
```
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_f64facee.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_f64facee.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_f64facee.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_f64facee.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_f64facee.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata signal_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_f64facee.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_dependency_graph_f64facee.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata dependency_graph saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_dependency_graph_f64facee.html
Multiple plots generated
```

<iframe src="../assets/generated/meteaudata_dependency_graph_f64facee.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="../assets/generated/meteaudata_timeseries_plot_f64facee.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="../assets/generated/meteaudata_signal_plot_f64facee.html" width="100%" height="500" frameborder="0"></iframe>

### Advanced Processing Demo
```python
# Demonstrate realistic data challenges
from meteaudata import replace_ranges, subset
from datetime import datetime

# Quality control
signal.process(["Temperature#1_RAW#1"], replace_ranges, 
               index_pairs=[[datetime(2024,1,1,10,0), datetime(2024,1,1,12,0)]],
               replace_with=np.nan)

# Show results with rich display
signal.display(format='html', depth=3)
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpuumibvql.py", line 198, in <module>
    signal.process(["Temperature#1_RAW#1"], replace_ranges, 
  File "/Users/jeandavidt/Developer/modelEAU/meteaudata/src/meteaudata/types.py", line 1157, in process
    outputs = transform_function(input_series, *args, **kwargs)
TypeError: replace_ranges() missing 1 required positional argument: 'reason'
```

This executable code system makes meteaudata's documentation truly interactive and ensures that all examples are tested and working with the current codebase.