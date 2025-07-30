# Visualization API Reference

Complete API reference for meteaudata's visualization capabilities.

## Plotting Methods

Interactive Plotly-based plotting for time series data:

- **[TimeSeries Plotting](timeseries-plotting.md)** - Individual time series visualization
- **[Signal Plotting](signal-plotting.md)** - Multi-time series and dependency graphs
- **[Dataset Plotting](dataset-plotting.md)** - Multi-signal subplot visualization

## Display System

Rich metadata exploration and visualization:

- **[Display System](display-system.md)** - Text, HTML, and interactive graph display methods

## Overview

### Plotting System

meteaudata provides three main plotting classes:

1. **TimeSeries.plot()** - Plot individual time series with automatic styling based on processing type
2. **Signal.plot()** - Plot multiple time series from a signal, with dependency graph visualization
3. **Dataset.plot()** - Plot multiple signals using subplots for comparison

All plotting methods return Plotly Figure objects that can be customized further.

### Display System

All meteaudata objects inherit rich display capabilities:

- **Text Display** - Simple text representation with configurable depth
- **HTML Display** - Rich HTML with collapsible sections (Jupyter notebooks)
- **Graph Display** - Interactive SVG visualization of metadata structure
- **Browser Display** - Full-page interactive exploration

### Key Features

**Automatic Styling:**
- Processing type-specific markers and modes
- Temporal shifting for prediction data
- Color cycling for multiple series

**Interactivity:**
- Plotly-based interactive charts
- Zoom, pan, and hover capabilities
- Exportable to HTML, PNG, PDF

**Metadata Integration:**
- Processing history visualization
- Dependency graph generation
- Complete audit trail display

## Common Parameters

### Plotting Parameters

Most plotting methods accept these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Plot title |
| `x_axis` | `str` | X-axis label |
| `y_axis` | `str` | Y-axis label |
| `start` | `str` | Start date for filtering |
| `end` | `str` | End date for filtering |

### Display Parameters

Display methods commonly accept:

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | `str` | Output format: 'text', 'html', 'graph' |
| `depth` | `int` | Display depth for text/HTML |
| `max_depth` | `int` | Maximum depth for graph display |
| `width` | `int` | Graph width in pixels |
| `height` | `int` | Graph height in pixels |
