# Metadata Dictionary

This section provides the official definitions for all metadata attributes used in metEAUdata.
Each page documents the fields, types, and validation rules for the core data structures.

## Core Metadata Classes

- **[DataProvenance](data-provenance.md)** - Information about data sources and context
- **[ProcessingStep](processing-step.md)** - Documentation of data processing operations
- **[FunctionInfo](function-info.md)** - Metadata about processing functions
- **[Parameters](parameters.md)** - Storage for processing function parameters
- **[IndexMetadata](index-metadata.md)** - Time series index information

## Data Container Classes

- **[TimeSeries](time-series.md)** - Individual time series with processing history
- **[Signal](signal.md)** - Collection of related time series
- **[Dataset](dataset.md)** - Collection of signals with project metadata

## Enumerations

- **[ProcessingType](processing-type.md)** - Standardized processing step categories

## Protocols

- **[SignalTransformFunctionProtocol](signal-transform-protocol.md)** - Interface for Signal-level processing functions
- **[DatasetTransformFunctionProtocol](dataset-transform-protocol.md)** - Interface for Dataset-level processing functions

## Standards and Conventions

### Naming Conventions

- **Signal Names**: Use descriptive names followed by `#N` numbering (e.g., `temperature#1`)
- **Time Series Names**: Format as `{signal_name}_{processing_suffix}#{number}` (e.g., `temperature#1_SMOOTH#1`)
- **Processing Suffixes**: Use 3-4 letter abbreviations describing the operation (e.g., `SMOOTH`, `FILT`, `RESAMP`)

### Required vs Optional Fields

- ✓ indicates a required field that must be provided
- ✗ indicates an optional field with a default value

### Type Annotations

All type annotations follow Python type hint standards:

- `str` - Text string
- `int` - Integer number
- `float` - Decimal number
- `bool` - True/False value
- `datetime.datetime` - Date and time
- `Optional[T]` - Field can be type T or None
- `list[T]` - List containing items of type T
- `dict[K, V]` - Dictionary with keys of type K and values of type V
