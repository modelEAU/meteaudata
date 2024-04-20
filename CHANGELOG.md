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

- Signals will be assigned a number if they are unnembered at creation.
