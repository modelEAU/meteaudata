# Signal Visualization API

Collection of related time series representing a measured parameter.

A Signal groups multiple time series that represent the same physical
parameter (e.g., temperature) at different processing stages or from
different processing paths. This enables comparison between raw and
processed data, evaluation of different processing methods, and
maintenance of data lineage.

Signals handle the naming conventions for time series, ensuring consistent
identification across processing workflows. They support processing
operations that can take multiple input time series and produce new
processed versions with complete metadata preservation.

## Methods

### plot

**Signature:**

```python
def plot(self, ts_names: List[str], title: Optional[str] = None, y_axis: Optional[str] = None, x_axis: Optional[str] = None, start: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None, end: Union[str, datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp, NoneType] = None) -> plotly.graph_objs._figure.Figure
```

**Description:**

Create an interactive Plotly plot with multiple time series from this signal.

Each time series is plotted with different colors and appropriate styling based
on their processing types. Temporal shifting is applied automatically for prediction data.

Args:
    ts_names: List of time series names to plot. Must exist in this signal.
    title: Plot title. If None, uses "Time series plot of {signal_name}".
    y_axis: Y-axis label. If None, uses "{signal_name} ({units})".
    x_axis: X-axis label. If None, uses "Time".
    start: Start date for filtering data (datetime string or object).
    end: End date for filtering data (datetime string or object).
    
Returns:
    Plotly Figure object with multiple time series traces.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ts_names` | `List` | ✓ | `—` | List of time series names to plot. Must exist in this signal. |
| `title` | `None` | ✗ | `—` | Plot title. If None, uses "Time series plot of {signal_name}". |
| `y_axis` | `None` | ✗ | `—` | Y-axis label. If None, uses "{signal_name} ({units})". |
| `x_axis` | `None` | ✗ | `—` | X-axis label. If None, uses "Time". |
| `start` | `None` | ✗ | `—` | Start date for filtering data (datetime string or object). |
| `end` | `None` | ✗ | `—` | End date for filtering data (datetime string or object). |

**Returns:** `Figure`

---

### plot_dependency_graph

**Signature:**

```python
def plot_dependency_graph(self, ts_name: str) -> plotly.graph_objs._figure.Figure
```

**Description:**

Create a dependency graph visualization showing processing lineage for a time series.

The graph displays time series as colored rectangles connected by lines representing
processing functions. The flow is temporal from left to right.

Args:
    ts_name: Name of the time series to trace dependencies for.
    
Returns:
    Plotly Figure object with the dependency graph visualization.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ts_name` | `str` | ✓ | `—` | Name of the time series to trace dependencies for. |

**Returns:** `Figure`

---
