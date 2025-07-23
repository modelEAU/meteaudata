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
