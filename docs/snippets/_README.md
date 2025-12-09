# Documentation Code Snippets

This directory contains reusable Python setup code for documentation examples. These snippets are executed in documentation markdown files to create consistent, reproducible examples.

## Available Setups

### setup_simple_signal.py

**Use for:** Basic signal examples, introductory tutorials

**Provides:**
- `signal` - Temperature signal with 100 hourly data points
- `provenance` - DataProvenance for the signal
- `data` - Raw temperature pandas Series
- `timestamps` - DatetimeIndex

**Example usage in markdown:**
```markdown
​```python exec="1" result="console" source="tabbed-right" session="signals" id="setup"
exec(open('docs/snippets/setup_simple_signal.py').read())
​```

​```python exec="1" result="console" session="signals"
print(f"Signal: {signal.name}")
​```
```

### setup_dataset.py

**Use for:** Multi-signal examples, dataset operations

**Provides:**
- `dataset` - Dataset with temperature, pH, and DO signals
- `temperature_signal`, `ph_signal`, `do_signal` - Individual signals
- `temp_data`, `ph_data`, `do_data` - Raw data Series
- `temp_provenance`, `ph_provenance`, `do_provenance` - Provenances
- `timestamps` - Shared DatetimeIndex
- `signals` - Dictionary of signals

**Example usage in markdown:**
```markdown
​```python exec="1" result="console" source="tabbed-right" session="dataset" id="setup"
exec(open('docs/snippets/setup_dataset.py').read())
​```

​```python exec="1" result="console" session="dataset"
print(f"Dataset has {len(dataset.signals)} signals")
​```
```

### setup_processing.py

**Use for:** Processing examples, quality control demos

**Provides:**
- `signal` - Temperature signal with gaps and outliers
- `processing_provenance` - DataProvenance for the signal
- `problematic_data` - pandas Series with quality issues
- `timestamps` - DatetimeIndex

**Example usage in markdown:**
```markdown
​```python exec="1" result="console" source="tabbed-right" session="processing" id="setup"
exec(open('docs/snippets/setup_processing.py').read())
​```

​```python exec="1" result="console" session="processing"
raw_data = signal.time_series["Temperature#1_RAW#1"].series
print(f"Missing values: {raw_data.isnull().sum()}")
​```
```

## Usage Guidelines

1. **Session names**: Use unique session names per page (e.g., `session="signals"` for signals.md)
2. **Setup visibility**: Use `source="tabbed-right"` to show setup in a collapsible tab
3. **ID attribute**: Add `id="setup"` to setup blocks for easier debugging
4. **Continuing examples**: Use the same session name to continue with the setup variables

## Maintaining Snippets

- Keep snippets **focused and minimal** - only what's needed for examples
- Use **reproducible random seeds** (`np.random.seed(42)`)
- **Document** what each snippet provides in its docstring
- **Test** snippets can be executed standalone before committing

## See Also

- [Executable Code Docs](../development/executable-code-docs.md) - Full documentation system guide
- [markdown-exec](https://pawamoy.github.io/markdown-exec/) - Plugin documentation
