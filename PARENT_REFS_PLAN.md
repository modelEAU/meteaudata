# Plan: Add Parent References to meteaudata Object Hierarchy

## Context

meteaudata objects form a strict parent-child hierarchy (Dataset → Signal → TimeSeries → ProcessingStep),
but children have no reference back to their parent. This means a Signal cannot navigate to its parent
Dataset, and a TimeSeries cannot navigate to its parent Signal. This is a design flaw that prevents
cross-signal dependency graph traversal and generally makes navigation brittle.

The fix: add weak back-references via `PrivateAttr` (Pydantic's mechanism for non-serialized runtime
state — already used throughout for `_backend`, `_auto_save`, etc.).

## Scope of Changes

**Primary file**: `src/meteaudata/types.py`

**Secondary**: `src/meteaudata/graph_display.py` — update `render_dependency_graph_html` and
`Signal.build_dependency_graph` to use parent refs for cross-signal lookup once they exist.

---

## Step 1: Add `_parent_signal` PrivateAttr to `TimeSeries`

In the `TimeSeries` class (line ~765), add alongside the existing `_backend` and `_storage_key`:

```python
_parent_signal: Optional["Signal"] = PrivateAttr(default=None)
```

No changes to serialization — PrivateAttr is never included in `model_dump()` or JSON output.

---

## Step 2: Upgrade `Signal._parent_dataset_name` to a real reference

`Signal` already has:

```python
_parent_dataset_name: Optional[str] = PrivateAttr(default=None)
```

(line ~1288, used only for storage key construction in `Signal.save_all()`)

Add a proper reference alongside it:

```python
_parent_dataset: Optional["Dataset"] = PrivateAttr(default=None)
```

Update `Signal.save_all()` and any other code that currently reads `_parent_dataset_name` to also
work via `self._parent_dataset.name` when the name attr is None — keep `_parent_dataset_name` for
backward compat with any external code that may set it directly.

---

## Step 3: Add `_propagate_parent_refs()` helpers

Add a private method to **Signal**:

```python
def _propagate_parent_refs(self) -> None:
    """Set _parent_signal on every TimeSeries owned by this Signal."""
    for ts in self.time_series.values():
        ts._parent_signal = self
```

Add a private method to **Dataset**:

```python
def _propagate_parent_refs(self) -> None:
    """Set _parent_dataset on every Signal, and propagate down to TimeSeries."""
    for signal in self.signals.values():
        signal._parent_dataset = self
        if signal._parent_dataset_name is None:
            signal._parent_dataset_name = self.name
        signal._propagate_parent_refs()
```

---

## Step 4: Call `_propagate_parent_refs()` at every attachment point

### Signal — call `self._propagate_parent_refs()` at end of:

- `Signal.__init__()` — after all `time_series` dict population is complete (after line ~1360)
- `Signal.add(ts)` — after `self.time_series[new_name] = ts` (line ~1451)
- `Signal.process()` — after adding each new TimeSeries to `self.time_series` (after line ~1694)
- `Signal._save_data()` — if it adds a new TimeSeries

### Dataset — call `self._propagate_parent_refs()` at end of:

- `Dataset.__init__()` — after all `signals` dict population (after line ~2524)
- `Dataset.add(signal)` — after `self.signals[new_name] = signal` (line ~2783)
- `Dataset.process()` — after adding each new Signal to `self.signals` (after line ~3089)
- `Dataset.load()` static method — after reconstruction from disk/DB

### `__setattr__` overrides

Both `Signal.__setattr__` and `Dataset.__setattr__` exist (to track `last_updated`). These need to
re-propagate when `time_series` (on Signal) or `signals` (on Dataset) is reassigned directly:

```python
# In Signal.__setattr__
if name == "time_series":
    super().__setattr__(name, value)
    self._propagate_parent_refs()
    return
```

---

## Step 5: Handle `model_copy(deep=True)` in `Dataset.process()`

`Dataset.process()` (line ~2994-3009) deep-copies input Signals before passing them to the transform
function. These copies will have stale or None parent refs. That is acceptable for the copies (they
are inputs to the transform, not stored). The OUTPUT signals returned by the transform are new objects
— parent refs get set correctly when they are added to `self.signals` in step 4 above.

No special handling needed beyond step 4.

---

## Step 6: Update `build_dependency_graph` for cross-signal awareness

Once parent refs exist, refactor the recursive dependency traversal.

### Extract a module-level helper in `types.py`:

```python
def _collect_ts_lookup(signal: "Signal") -> dict[str, "TimeSeries"]:
    """
    Build a flat {ts_name: TimeSeries} lookup spanning all signals in the parent
    dataset (if available), or just this signal's time_series otherwise.
    """
    if signal._parent_dataset is not None:
        lookup = {}
        for s in signal._parent_dataset.signals.values():
            lookup.update(s.time_series)
        return lookup
    return dict(signal.time_series)
```

### Extract the recursive core as a module-level function:

```python
def _build_dependency_graph_recursive(
    ts_name: str, lookup: dict[str, "TimeSeries"]
) -> list[dict]:
    if ts_name not in lookup:
        return []  # external/unresolvable — stop gracefully (no error)
    ts = lookup[ts_name]
    if not ts.processing_steps:
        return []
    last_step = ts.processing_steps[-1]
    deps = []
    for input_name in last_step.input_series_names:
        deps.append({
            "step": last_step.function_info.name,
            "type": last_step.type,
            "origin": input_name,
            "destination": ts_name,
        })
        deps.extend(_build_dependency_graph_recursive(input_name, lookup))
    return deps
```

### Update `Signal.build_dependency_graph(ts_name)`:

```python
def build_dependency_graph(self, ts_name: str) -> list[dict]:
    if ts_name not in self.time_series:
        raise ValueError(f"Time series {ts_name} not found in the signal.")
    lookup = _collect_ts_lookup(self)
    return _build_dependency_graph_recursive(ts_name, lookup)
```

### Add `Dataset.build_dependency_graph(ts_name)`:

```python
def build_dependency_graph(self, ts_name: str) -> list[dict]:
    lookup = {}
    for s in self.signals.values():
        lookup.update(s.time_series)
    if ts_name not in lookup:
        raise ValueError(f"Time series {ts_name} not found in the dataset.")
    return _build_dependency_graph_recursive(ts_name, lookup)
```

### Add `Dataset.build_full_dependency_graph()`:

```python
def build_full_dependency_graph(self) -> list[dict]:
    """Build dependency graph for ALL time series in the dataset."""
    lookup = {}
    for s in self.signals.values():
        lookup.update(s.time_series)
    all_deps = []
    seen_edges = set()
    for ts_name in lookup:
        for dep in _build_dependency_graph_recursive(ts_name, lookup):
            key = (dep["origin"], dep["destination"])
            if key not in seen_edges:
                seen_edges.add(key)
                all_deps.append(dep)
    return all_deps
```

---

## Step 7: Update `render_dependency_graph_html` and `plot_dependency_graph`

### `graph_display.py` — `render_dependency_graph_html`

Change signature to accept pre-built dependencies and a lookup dict instead of a signal object:

```python
def render_dependency_graph_html(
    dependencies: list[dict],
    ts_lookup: dict,        # {ts_name: TimeSeries} for node detail enrichment
    title: str = None,
) -> str:
```

Update all callers accordingly.

### Add `Dataset.plot_dependency_graph(ts_name=None)`

```python
def plot_dependency_graph(self, ts_name: str = None) -> Optional[str]:
    """
    Plot dependency graph for a specific ts (if ts_name given) or the full
    dataset dependency graph (if ts_name is None).
    """
```

### `Signal.plot_dependency_graph(ts_name)`

Already calls `render_dependency_graph_html`; just needs to pass the cross-signal
`_collect_ts_lookup(self)` result through. No signature change visible to callers.

---

## Files to Modify

- `src/meteaudata/types.py` — all changes above
- `src/meteaudata/graph_display.py` — update `render_dependency_graph_html` signature and callers

## Files NOT to Modify

- `src/meteaudata/templates/dependency_graph_template.html` — no change needed
- `src/meteaudata/displayable.py` — no change needed

---

## Verification

1. **Unit tests**: `pytest tests/test_metEAUdata.py tests/test_graph_display.py -v`
   - All existing tests must pass
   - Update `test_plots` assertions if needed (return type stays `Optional[str]`)

2. **New test**: Add a test that:
   - Creates a Dataset with 2+ signals
   - Runs `Dataset.process()` with a multivariate step (e.g., `average_signals`) to create a cross-signal dependency
   - Calls `dataset.build_dependency_graph(merged_ts_name)` and asserts the result includes ts names from both source signals
   - Calls `dataset.plot_dependency_graph()` (no args) and asserts it returns a non-None HTML path

3. **Serialization round-trip**: Verify that `dataset.model_dump()` contains no parent refs
   (PrivateAttr excluded), and that `Dataset.load()` after save correctly re-propagates parent refs.

4. **Manual demo**: Run a two-signal average demo and confirm the browser shows the full cross-signal graph.
