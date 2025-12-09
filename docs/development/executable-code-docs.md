# Executable Code in Documentation

This guide explains how meteaudata's documentation supports executable code blocks that run at build time and inject live outputs directly into the documentation.

## Overview

The meteaudata documentation uses the **markdown-exec** plugin to:

- **Run actual Python code** during documentation build
- **Capture real outputs** including print statements, plots, and data
- **Fail the build** if code examples have errors (ensuring docs stay up-to-date)
- **Maintain context** across multiple code blocks using sessions
- **Show both code and output** for realistic examples

## Basic Usage

### Standard Code Blocks vs Executable Blocks

**Standard code block (static):**
```python
# This code is just displayed, not executed
signal = Signal(data, "Temperature", provenance, "°C")
print(f"Created signal: {signal.name}")
```

**Executable code block:**
```python exec="1" result="console"
# This code actually runs during build and shows real output
import pandas as pd
data = pd.Series([20, 21, 22, 23, 24])
print(f"Created series with {len(data)} values")
print(f"Mean: {data.mean():.1f}")
```

The `exec="1" result="console"` option (or `exec="true"`, `exec="on"`) tells markdown-exec to execute the code block.

### Using Sessions for State Persistence

Sessions allow you to share variables between code blocks:

```python exec="1" result="console" session="demo"
# First block sets up variables
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance

provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Demo",
    location="Demo Site",
    equipment="Temperature Sensor",
    parameter="Temperature",
    purpose="Documentation example"
)

timestamps = pd.date_range('2024-01-01', periods=100, freq='1h')
data = pd.Series(np.random.randn(100) * 5 + 20, index=timestamps, name="RAW")

signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)

print(f"Created signal: {signal.name}")
print(f"Time series: {list(signal.time_series.keys())}")
```

```python exec="1" result="console" session="demo"
# Continues from previous block - signal variable is available
from meteaudata import resample

signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")
print(f"After resampling: {list(signal.time_series.keys())}")
```

## Output Display Options

### Show Code Above Output (default)

```python exec="1" result="console" source="above"
print("This is the output")
print("Code appears above this")
```

### Show Code Below Output

```python exec="1" result="console" source="below"
print("This is the output")
print("Code appears below this")
```

### Tabbed Display (Material theme)

```python exec="1" result="console" source="tabbed-left"
print("Code and output in tabs")
print("Code tab on the left")
```

## Error Handling

When code fails, markdown-exec will:
1. **Show the error** in the rendered documentation
2. **Log a warning** during build with the full traceback
3. **Display the error inline** so readers can see what went wrong

The `id` option helps identify which code block failed in build logs:

```markdown
# Use id to identify problematic blocks in your markdown files
​```python exec="1" result="console" id="my-example"
# Your code here
​```
```

## Best Practices

### Writing Executable Examples

**DO:**

- Use sessions to build up complex examples step-by-step
- Include meaningful print statements to show outputs
- Test examples work before committing
- Use `id` option for easier debugging

**DON'T:**

- Rely on external files or network resources
- Use overly complex examples that obscure the main point
- Forget to specify session names when continuing contexts
- Mix unrelated concepts in single code blocks

### Session Organization

For documentation pages with multiple examples:

1. **Use unique session names** for each independent example
2. **Reuse session names** when continuing the same example
3. **Don't share sessions across pages** (each page starts fresh)

Example:

```python exec="1" result="console" session="example1"
# First example
x = 10
print(f"Example 1: x = {x}")
```

```python exec="1" result="console" session="example1"
# Continuing first example
print(f"Still example 1: x = {x}")
```

```python exec="1" result="console" session="example2"
# Separate example - x is not defined here
y = 20
print(f"Example 2: y = {y}")
```

## Integration with MkDocs

The markdown-exec plugin:

1. **Runs during `mkdocs build` or `mkdocs serve`**
2. **Works with Material theme** for tabbed display
3. **Integrates with other plugins** like mkdocstrings
4. **Provides build-time validation** of code examples

## Migration from Bespoke System

Our previous system used custom scripts with `exec="context_name"` syntax. The new system uses:

- **`exec="1" result="console"`** instead of `exec="simple_signal"`
- **`session="name"`** instead of `exec="continue"`
- **No need for `_template.md` files** - edit markdown files directly
- **Cleaner integration** with standard MkDocs workflow

## Advanced Features

### Setting Working Directory

```python exec="1" result="console" workdir="src"
# Runs with src/ as working directory
import os
print(f"Current directory: {os.getcwd()}")
```

### HTML Output

```python exec="1" result="console" html="1"
# Output raw HTML directly
print("<h3>This is HTML output</h3>")
print("<p>Useful for rich displays</p>")
```

### Custom Result Formatting

```python exec="1" result="console" result="json"
# Format output as JSON code block
print('{"key": "value", "number": 42}')
```

## Reference

For complete documentation of markdown-exec features, see:
- [markdown-exec documentation](https://pawamoy.github.io/markdown-exec/)
- [GitHub repository](https://github.com/pawamoy/markdown-exec)
