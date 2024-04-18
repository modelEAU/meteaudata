# Changelog

## 0.2.0

- First "feature-complete" release of `metEAUdata`. Contains functionality to transform both univariate and multivariate data.

## 0.2.1

- Added the `step_distance` attribute to `ProcessingStep`. `step_distance`indicates how many steps forward (or backward) a time series with this processing step is from its input time series.
- Removed `calibration_info` from `ProcessingStep` because that info is already found in `parameters`.

## 0.3.0

- `TimeSeries`, `Signal` and `Dataset` objects now have a `plot` method to generate basic visualizations. This adds `plotly` as a dependency. Other backends (namely, `metplotlib`) may be developed in the future.

