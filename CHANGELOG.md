# Changelog

## 0.2.0

- First "feature-complete" release of `metEAUdata`. Contains functionality to transform both univariate and multivariate data.

## 0.2.1

- Added the `step_distance` attribute to `ProcessingStep`. `step_distance`indicates how many steps forward (or backward) a time series with this processing step is from its input time series.
- Removed `calibration_info` from `ProcessingStep` because that info is already found in `parameters`.

## 0.3.0

- `TimeSeries`, `Signal` and `Dataset` objects now have a `plot` method to generate basic visualizations. This adds `plotly` as a dependency. Other backends (namely, `matplotlib`) may be developed in the future.

## 0.3.1

- Added `input_series_names` to `ProcessingStep` so that step sequences and dependencies can be traced.

## 0.4.0

- Modified the naming convention of time series inside signals and of the signals themselves. Now, new time series are assigned a numbered name. The number is separated from the rest of the name with a `#` symbol. This allows multiple runs of a processing function to not overwrite the preceding versions. Signals are now also numbered, meaning that dataset processing will also not overwrite existing signals.

## 0.4.1

- Signals will be assigned a number if they are unnumbered at creation.

## 0.4.3

- Bug in dataset numbering
- Added tests
- Slight adjustments to plots

## 0.4.4

- Now updating the time series names in the processing steps

## 0.5.0

- Added a plottling method to `Signal.plot_dependency_graph` to visually represent the process that went into creating a time series from that signal.

## 0.5.1

- Added a field to `FunctionInfo` that replicated the entire processing function's code

## 0.5.2

- Fixed a bug where plots would not render if the frequency of the data was 1 second, 1 minute, 1 hour, .... Reason is that the "1" would be omitted in the series' `frequency`, which would throw off the plot function when trying to calculate the x coordinates of the plot.
- Fixed a bug that overwrote the `FunctionInfo.source_code` field when deserializing a serialized ProcessingStep.

## 0.6.0

- Fixed a typo in the processing functions where the reference pointed to the wrong GitHub repository.
- Added a processing function (`replace_ranges`) that lets users replace values in a time series with a filler value (eg, NaN). Can be used to filter out manually-identified invalid data.
- Added `__str__` method to processing steps so they print nicely

## 0.7.0

- Added a new univariate processing function `subset` that lets you slice time series to a desired length or index values

## 0.7.1

- Added a `dataset.remove(signal_name)` and `signal.remove(ts_name)` to facilitate rapid creation and deletion of time series / signals

## 0.7.2

- Added methods to Parameters so it can handle numpy arrays and tuples.

## 0.7.3

- Added support for series with strings in them.

## 0.7.4

- Added start and end parameters to plotting functions.

## 0.8.0

- Added the ability to visually explore metadata using the `Dataset.show_graph_in_browser()` and `Signal.showgraph_in_browser()` methods.

## 0.9.0

- The project now contains a documentation website!

## 0.9.2

- Fixed an issue where the HTML representation of meteaudata objects would not render properly. Updated documentation.

## 0.9.3

- Fixed an issue where datetime objects were not represented adequately in YAML exports.
- Fixed an issue where the test suite would open the web browser to render HTML.

## 0.9.4

- Fixed a bug where processing steps for TimeSeries objects were not appearing in the browser SVG graph visualization. The issue was caused by a mismatch between the identifier format expected by the SVG template ("Processing Steps") and the identifier returned by the CollectionContainer ("processing_steps").

## 0.10.0

This release adds export customization features to improve CSV compatibility with different computing environments and locales.

### Customizable CSV Export (Signal.save() and Dataset.save())

- **separator**: Choose any CSV separator character (comma, semicolon, tab, etc.)
  - Default: "," (backward compatible)
  - Example: `signal.save(path, separator=";")`  # For European Excel
- **index_name**: Set custom column name for time index in CSV exports
  - Default: None (uses pandas default)
  - Example: `dataset.save(path, index_name="timestamp")`

### Custom Output Naming in Process Functions

- **Signal.process()**: New `output_names` parameter for user-friendly time series names
  - Replaces operation suffixes (e.g., "RESAMPLED") with custom names
  - Example: `signal.process(["A#1_RAW#1"], resample, "1D", output_names=["daily"])`
  - Creates "A#1_daily#1" instead of "A#1_RESAMPLED#1"
  - Note: Underscores not allowed in custom names (reserved character)

- **Dataset.process()**: New `output_signal_names` and `output_ts_names` parameters
  - `output_signal_names`: Custom names for output signals
  - `output_ts_names`: Custom names for time series within those signals
  - Example: `dataset.process(inputs, average_signals, output_signal_names=["siteaverage"])`

### Overwrite Mode

- **overwrite** parameter in both Signal.process() and Dataset.process()
  - When True: Keeps existing hash number (e.g., stays at #1)
  - When False (default): Increments hash number (e.g., #1 → #2)
  - Useful for re-running processing without creating new versions

### Implementation Notes

- All new parameters are optional with backward-compatible defaults
- Added helper methods: `Signal.replace_operation_suffix()`, `Dataset.replace_signal_base_name()`
- Validation ensures custom names don't contain reserved characters
- Comprehensive test suite with 20 new tests covering all features

### Internal Improvements

- All processing functions now use `Signal.extract_ts_base_and_number()` for consistent signal name extraction instead of string splitting on underscores

## 0.11.0

### Backend Storage System

Added pluggable storage backend system to enable processing of larger-than-memory datasets:

- **Simple API**: Configure storage with one method call
  - `dataset.use_disk_storage("./data")` - Store as Parquet files
  - `dataset.use_sql_storage("sqlite:///db.db")` - Store in SQL database
  - `dataset.use_memory_storage()` - Switch back to in-memory
  - Works at all levels: `TimeSeries`, `Signal`, and `Dataset`
  - Auto-save enabled by default for disk/SQL backends

- **Pandas Disk Backend**: Store data as Parquet files with YAML metadata
  - Memory-efficient processing of large datasets
  - Human-readable metadata files
  - Configurable storage directory

- **SQL Backend**: Store data in SQL databases (SQLite, PostgreSQL, MySQL)
  - Multi-user access to shared data
  - Automatic table creation (`time_series_data`, `time_series_metadata`)
  - Compatible with any SQLAlchemy database (via SQLAlchemy)

- **Transparent Processing**: Processing code remains identical regardless of backend
  - Auto-save mode persists data after each processing step
  - Manual control with `save_all()` and `load_all()`
  - Switch backends mid-session

- **Advanced API**: Low-level control still available for power users
  - `StorageConfig.for_pandas_disk()` / `StorageConfig.for_sql()`
  - `create_backend(config)`
  - `Dataset.set_backend(backend, auto_save=True/False)`

See the [Backend Storage documentation](https://modeleau.github.io/meteaudata/user-guide/backend-storage/) for details

### Documentation System Improvements

- **Migrated to markdown-exec**: Replaced bespoke code execution system with the standard [markdown-exec](https://github.com/pawamoy/markdown-exec) plugin
  - Code blocks now execute during documentation build with `exec="1"` syntax
  - Reusable setup code in `docs/snippets/` eliminates duplication
  - Setup code shown in collapsible tabs with `source="tabbed-right"`
  - Sessions replace the old context system: use `session="name"` to share state between code blocks
  - Build warns when code examples have errors, ensuring docs stay synchronized with code
  - Removed custom scripts: `exec_processor.py`, `exec_contexts.py`, `process_templates.py`, `mkdocs_exec_plugin/`
  - No more `_template.md` files needed - edit markdown files directly
  - Better integration with standard MkDocs workflow and plugins
  - Converted 13 documentation files (user-guide, getting-started, examples)

### Breaking Changes (Documentation Only)

- Documentation contributors need to use new syntax for executable code blocks
- Setup code now uses snippet files: `exec(open('docs/snippets/setup_simple_signal.py').read())`
- Old: `exec="simple_signal"` → New: `exec="1" source="tabbed-right" session="signals" id="setup"`
- Old: `exec="continue"` → New: `exec="1" session="same_name"`
