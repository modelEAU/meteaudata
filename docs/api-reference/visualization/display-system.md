# Display System API

Complete API reference for the meteaudata display system methods.
All meteaudata objects inherit from `DisplayableBase` and provide rich visualization capabilities.

## Overview

The display system provides multiple output formats:

- **Text display** - Simple text representation
- **HTML display** - Rich HTML with expandable sections (Jupyter notebooks)
- **Graph display** - Interactive SVG graphs of metadata structure
- **Browser display** - Full-page interactive visualization

## Methods

### display

**Signature:**

```python
def display(self, format: str = 'html', depth: int = 2, max_depth: int = 4, width: int = 1200, height: int = 800) -> None
```

**Description:**

Display method with support for text, HTML, and interactive graph formats.

Args:
    format: Display format - 'text', 'html', or 'graph' 
    depth: Depth for text/html displays
    max_depth: Maximum depth for graph traversal
    width: Graph width in pixels
    height: Graph height in pixels

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | `str` | ✗ | `html` | Display format - 'text', 'html', or 'graph' |
| `depth` | `int` | ✗ | `2` | Depth for text/html displays |
| `max_depth` | `int` | ✗ | `4` | Maximum depth for graph traversal |
| `width` | `int` | ✗ | `1200` | Graph width in pixels |
| `height` | `int` | ✗ | `800` | Graph height in pixels |

**Returns:** `NoneType`

---

### show_summary

**Signature:**

```python
def show_summary(self) -> None
```

**Description:**

Convenience method to show a text summary.

**Returns:** `NoneType`

---

### show_details

**Signature:**

```python
def show_details(self, depth: int = 3) -> None
```

**Description:**

Convenience method to show detailed HTML view.

Args:
    depth: How deep to expand nested objects

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `depth` | `int` | ✗ | `3` | How deep to expand nested objects |

**Returns:** `NoneType`

---

### show_graph

**Signature:**

```python
def show_graph(self, max_depth: int = 4, width: int = 1200, height: int = 800) -> None
```

**Description:**

Convenience method to show the interactive graph.

Args:
    max_depth: Maximum depth to traverse in object hierarchy
    width: Graph width in pixels  
    height: Graph height in pixels

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_depth` | `int` | ✗ | `4` | Maximum depth to traverse in object hierarchy |
| `width` | `int` | ✗ | `1200` | Graph width in pixels |
| `height` | `int` | ✗ | `800` | Graph height in pixels |

**Returns:** `NoneType`

---

### show_graph_in_browser

**Signature:**

```python
def show_graph_in_browser(self, max_depth: int = 4, width: int = 1200, height: int = 800, title: Optional[str] = None) -> str
```

**Description:**

Render SVG graph and open in browser.

Args:
    max_depth: Maximum depth to traverse in object hierarchy
    width: Graph width in pixels
    height: Graph height in pixels
    title: Page title (auto-generated if None)
    
Returns:
    Path to the generated HTML file

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_depth` | `int` | ✗ | `4` | Maximum depth to traverse in object hierarchy |
| `width` | `int` | ✗ | `1200` | Graph width in pixels |
| `height` | `int` | ✗ | `800` | Graph height in pixels |
| `title` | `None` | ✗ | `—` | Page title (auto-generated if None) |

**Returns:** `str`

---

## Common Usage Patterns

### Quick Display Methods

```python
# Quick text summary
signal.show_summary()

# Rich HTML display (Jupyter)
signal.show_details()

# Interactive graph
signal.show_graph()
```

### Customized Display

```python
# Custom text display
signal.display(format='text', depth=3)

# Custom HTML display
signal.display(format='html', depth=4)

# Custom graph display
signal.display(format='graph', max_depth=5, width=1400, height=900)
```

### Browser Visualization

```python
# Open in browser with custom settings
html_path = signal.show_graph_in_browser(
    max_depth=4,
    width=1600,
    height=1000,
    title='Custom Visualization'
)
```
