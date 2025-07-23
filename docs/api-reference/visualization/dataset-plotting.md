# Dataset Visualization API

Collection of signals representing a complete monitoring dataset.

A Dataset groups multiple signals that are collected together as part of
a monitoring project or analysis workflow. It provides project-level
metadata and enables coordinated processing operations across multiple
parameters.

Datasets support cross-signal processing operations and maintain consistent
naming conventions across all contained signals. They provide the highest
level of organization for environmental monitoring data with complete
metadata preservation and serialization capabilities.
    

## Methods

### plot

**Signature:**

```python
def plot(self, signal_names: List[str], ts_names: List[str], title: Optional[str] = None, y_axis: Optional[str] = None, x_axis: Optional[str] = None, start: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None, end: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None) -> plotly.graph_objs._figure.Figure
```

**Description:**

Create a multi-subplot visualization comparing time series across signals.

Each signal gets its own subplot with shared x-axis (time). Only time series
that exist in each signal are plotted. Individual y-axis labels include units.

Args:
    signal_names: List of signal names to plot. Must exist in this dataset.
    ts_names: List of time series names to plot from each signal.
    title: Plot title. If None, uses "Time series plots of dataset {dataset_name}".
    y_axis: Base Y-axis label. If None, uses "Values".
    x_axis: X-axis label. If None, uses "Time".
    start: Start date for filtering data (datetime string or object).
    end: End date for filtering data (datetime string or object).
    
Returns:
    Plotly Figure object with subplots for each signal.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `signal_names` | `List` | ✓ | `—` | List of signal names to plot. Must exist in this dataset. |
| `ts_names` | `List` | ✓ | `—` | List of time series names to plot from each signal. |
| `title` | `None` | ✗ | `—` | Plot title. If None, uses "Time series plots of dataset {dataset_name}". |
| `y_axis` | `None` | ✗ | `—` | Base Y-axis label. If None, uses "Values". |
| `x_axis` | `None` | ✗ | `—` | X-axis label. If None, uses "Time". |
| `start` | `None` | ✗ | `—` | Start date for filtering data (datetime string or object). |
| `end` | `None` | ✗ | `—` | End date for filtering data (datetime string or object). |

**Returns:** `Figure`

---
