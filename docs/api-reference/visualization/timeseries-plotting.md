# TimeSeries Visualization API

Time series data with complete processing history and metadata.

This class represents a single time series with its associated pandas Series
data, complete processing history, and index metadata. It maintains a full
audit trail of all transformations applied to the data from its raw state
to the current processed form.

The class handles serialization of pandas objects and preserves critical
index information to ensure proper reconstruction. It's the fundamental
building block for environmental time series analysis workflows.

## Methods

### plot

**Signature:**

```python
def plot(self, title: Optional[str] = None, y_axis: Optional[str] = None, x_axis: Optional[str] = None, legend_name: Optional[str] = None, start: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None, end: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None) -> plotly.graph_objs._figure.Figure
```

**Description:**

Create an interactive Plotly plot of the time series data.

The plot styling is automatically determined by the processing type of the time series.
For prediction data, temporal shifting is applied to show future timestamps.

Args:
    title: Plot title. If None, uses the time series name.
    y_axis: Y-axis label. If None, uses the time series name.
    x_axis: X-axis label. If None, uses "Time".
    legend_name: Legend entry name. If None, uses the time series name.
    start: Start date for filtering data (datetime string or object).
    end: End date for filtering data (datetime string or object).
    
Returns:
    Plotly Figure object with the time series plot.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `title` | `None` | ✗ | `—` | Plot title. If None, uses the time series name. |
| `y_axis` | `None` | ✗ | `—` | Y-axis label. If None, uses the time series name. |
| `x_axis` | `None` | ✗ | `—` | X-axis label. If None, uses "Time". |
| `legend_name` | `None` | ✗ | `—` | Legend entry name. If None, uses the time series name. |
| `start` | `None` | ✗ | `—` | Start date for filtering data (datetime string or object). |
| `end` | `None` | ✗ | `—` | End date for filtering data (datetime string or object). |

**Returns:** `Figure`

---
