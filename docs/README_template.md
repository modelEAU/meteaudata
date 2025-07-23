# met*EAU*data

`meteaudata` is a Python library designed for the comprehensive management and processing of time series data, particularly focusing on environmental data analytics. It provides tools for detailed metadata handling, data transformations, and serialization of processing steps to ensure reproducibility and clarity in data manipulation workflows.

## Features

- Detailed metadata management for time series data.
- Built-in support for serialization and deserialization of data and metadata.
- Customizable processing steps for various data transformations such as interpolation, resampling, and averaging.

## Installation

### To contribute to this project

1. Fork this repository to your GitHub account
1. Clone your fork to your computer:

```bash
git clone https://github.com/your-username/meteaudata.git
cd meteaudata
poetry install
```

### To use `meteaudata` as a dependency for another project

If your project uses poetry:

```bash
poetry add git+https://github.com/your-repo/meteaudata.git
```

If your project uses pip, the following should work:

```bash
pip install git+https://github.com/your-repo/meteaudata.git
```

## Usage

Below are a few examples of how to use `meteaudata`:

### Creating and Manipulating a Signal

<!-- INSERT CODE: create_signal.py -->

### Creating and Manipulating a Dataset

<!-- INSERT CODE: create_dataset.py -->


## Create your own transformation functions

If you already have a data processing pipeline, it can be easily adapted to work with metEAUdata. All that is needed is to create a wrapper function that adheres the either the SignalTransformationProtocol or the DatasetTransformationProtocol.

### Signal Tranformations

As long as your transformation is univariate (works on data from a single signal at a time), it can be adapted to match the SignalTranformationProtocol in the following way:

Create a function that:

1. Accepts the following arguments:
    1. A list of pandas Series the functions will use as inputs
    1. Any arguments you need to pass to *your* function for it to work.
    1. Any keyword arguments you need to pass to *your* function for it to work.
1. Returns a list of outputs. Each item in the list is a tuple (group of two objects). For every tuple, these objects are:
    1. One of the pandas Series that was produced by *your* function.
    1. A list of ProcessingStep objects. These objects represent each transformation your transformation applied to the output time series to obtain it from the input time series.

### Dataset Transformations

If your transformation function involves multiple signals (multivariate transformations), it should conform to the DatasetTransformFunctionProtocol in this manner:

Create a function that:

1. Accepts the following arguments:
    1. A list of Signal objects that the function will use as inputs.
    1. A list of time series names, where each name corresponds to the specific time series within the input signals that your function will process.
    1. Any additional arguments that are necessary for your transformation function.
    1. Any keyword arguments that are necessary for your transformation function.
1. Returns a list of outputs. Each item in the list is a Signal object:
    1. Each Signal object contains one or more transformed time series resulting from the applied transformation.
    1. Each time series within the Signal should be associated with a list of ProcessingStep objects. These objects document each transformation step applied to the time series to transform it from its original state in the input signals.

### Describing your Processing Steps with the `ProcessingStep`object

The `ProcessingStep` object represents a single step in the transformation of a time series, documenting the specifics of the transformation applied. Use this object to ensure traceability and reproducibility in data processing.

Attributes:

- `type`: An instance of ProcessingType that categorizes the transformation (e.g., smoothing, filtering, resampling). The list of accepted categories can be found in the ProcessingType object in the `meteaudata.types` module.
- `description`: A brief description of what the transformation step does.
- `function_info`: An instance of FunctionInfo providing details such as the name of the function, version, author, and reference URL.
- `run_datetime`: The date and time when the transformation was applied.
- `requires_calibration`: A boolean indicating whether the transformation requires calibration data.
- `parameters`: Optional. An instance of Parameters storing any parameters used in the transformation.
- `suffix`: A string appended to the name of the output series to denote this specific transformation step. By convention, suffixes are made of 3 or 4-letter words or abbreviations that briefly designate the applied transformation. The suffix should NEVER contain an underscore ("_"), as underscores are used to distinguish important parts of the time series name. Instead, if the suffix contains several words, link them using a dash "-".

Usage:

Include the `ProcessingStep` object(s) in the tuple returned by your transformation function, paired with the transformed time series. This linkage ensures that each transformation step’s metadata is directly associated with the resulting data.

### The `FunctionInfo` object

The `FunctionInfo` object provides essential metadata about the specific function used in a transformation step to ensure repeatability and traceability of data processing workflows.

Attributes:

- `name`: The name of the function, which should be descriptive enough to identify the purpose of the transformation.
- `version`: The version number of the function, helping to manage updates or changes over time.
- `author`: The name of the individual or organization that developed or implemented the function.
- `reference`: A URL or a citation to detailed documentation or the source code repository, providing deeper insights into the function's implementation and usage.

Usage:

Simply include a `FunctionInfo` object as part of each `ProcessingStep` to document the specific details of the function used for that step.

### Putting it all together (signal version)

A custom Signal transformation would therefore look like the following

<!-- INSERT CODE: all_together_signal.py -->

Explanation:

- Function Definition: We define a function that implements the SignalTransformFunctionProtocol (without explicitly depending on it). This ensures our transformation adheres to the expected interface.
- Transformation Logic: In the function, we iterate over the input series, applying a simple transformation to double each value. This example can be replaced with any logic specific to your needs.

- `FunctionInfo`: We create a `FunctionInfo` object to document who created the function, its version, and where more information can be found.
- `ProcessingStep`s: For each transformation, we instantiate a `ProcessingStep` that describes what the transformation does, when it was run, and other metadata like whether it requires calibration.

- Output: The transformed series is paired with its corresponding `ProcessingStep`(s) in a tuple, which is then collected into a list of such tuples.

### Putting it all together (dataset version)

A custom Dataset transformation would therefore look like the following:

<!-- INSERT CODE: all_together_dataset.py -->

Explanation:

Function Definition:

- `input_signals`: A list of `Signal` objects that are the input to the function.
- `input_series_names`: A list of strings that specifies which time series within each signal should be processed.
- `final_provenance`: Optional. An instance of `DataProvenance` to apply to the new Signal created as a result of this function. If not provided, the function will use the provenance from the first input signal.
- `*args` and `**kwargs`: These allow the function to accept additional positional and keyword arguments for flexibility.

Documentation and Metadata:

- `func_info`: An instance of `FunctionInfo` that documents critical information about the function, such as its name, version, author, and reference.

- `processing_step`: Defines a `ProcessingStep` object that records the specifics of the transformation applied— summing the series in this case. This step includes the type of transformation, a description, the date and time it was run, whether calibration was required, and a suffix to append to the new series' name for identification.

Transformation Logic:

Before the transformation, there may be checks (not fully implemented in the sample) to ensure that all input series have compatible data types and units and that each specified time series name exists within its corresponding signal.
The transformation itself involves concatenating the selected series horizontally (axis=1) and computing their sum across the rows (axis=1), resulting in a new series where each point is the sum of the corresponding points in the input series.

Creation of New Signal:

- `signals_prefix`: Constructs a descriptive name for the new signal by concatenating the names of the input signals, separated by a plus sign.
- `new_signal_name`: Appends "-SUM" to the signals_prefix to indicate that this signal represents the sum of the input signals.
- `summed_time_series`: A new `TimeSeries` object that wraps the summed series along with the processing steps detailing how it was created.
- `new_signal`: Constructs a new `Signal` object using the newly created time series, specifying its name, units, and provenance.

Output:

The function returns a list containing the newly created `Signal` object. This output format aligns with the expectations for dataset transformations, allowing the new signal to be integrated back into a dataset or further processed.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests to the main branch. For major changes, please open an issue first to discuss what you would like to change.

Types of accepted pull requests:

- Bug fixes.
- New transformation functions that conform to the provided protocols.
- Addition of metadata attributes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Jean-David Therrien

## Contact

For any queries, you can reach me at <jeandavidt@gmail.com>.
