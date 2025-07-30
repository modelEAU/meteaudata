#!/usr/bin/env python3
"""
Script to automatically generate metadata dictionary documentation
from Pydantic models in meteaudata.

This script is run by mkdocs-gen-files during documentation build.
"""

import inspect
import os
import datetime
from pathlib import Path
from typing import get_type_hints, get_origin, get_args
import mkdocs_gen_files

# Handle imports gracefully for documentation builds
try:
    from pydantic import BaseModel
    print("DEBUG: Successfully imported BaseModel")
    from meteaudata.types import (
        DataProvenance, ProcessingStep, FunctionInfo, Parameters, 
        IndexMetadata, TimeSeries, Signal, Dataset, ProcessingType
    )
    print("DEBUG: Successfully imported meteaudata types")
except ImportError as e:
    print(f"ERROR: Could not import meteaudata types: {e}")
    print("Make sure meteaudata is installed: uv pip install -e .")
    exit(1)



def format_type_hint(type_hint):
    """Format type hints for documentation."""
    if hasattr(type_hint, '__name__'):
        return f"`{type_hint.__name__}`"
    elif hasattr(type_hint, '_name'):
        return f"`{type_hint._name}`"
    elif get_origin(type_hint) is not None:
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        if origin is list:
            return f"`list[{format_type_hint(args[0]) if args else 'Any'}]`"
        elif origin is dict:
            key_type = format_type_hint(args[0]) if args else 'Any'
            value_type = format_type_hint(args[1]) if len(args) > 1 else 'Any'
            return f"`dict[{key_type}, {value_type}]`"
        elif origin is type(None):
            return "`None`"
        else:
            return f"`{origin.__name__}`"
    else:
        return f"`{str(type_hint)}`"



def get_field_info(model_class: type[BaseModel], field_name: str):
    """Extract comprehensive field information from a Pydantic model."""
    model_fields = model_class.model_fields
    field_info = model_fields.get(field_name)
    
    if not field_info:
        return None
    
    # Get type hints
    type_hints = get_type_hints(model_class)
    field_type = type_hints.get(field_name, "Unknown")
    
    # Extract field properties
    default_value = None
    default_description = None
    
    # Import PydanticUndefined for proper comparison
    try:
        from pydantic_core import PydanticUndefined
        undefined_marker = PydanticUndefined
    except ImportError:
        # Fallback for older Pydantic versions
        undefined_marker = ...
    
    if field_info.default is not undefined_marker:
        # Has a direct default value
        default_value = field_info.default
        # Check if it's a datetime that was computed at class definition time
        if isinstance(default_value, datetime.datetime):
            default_description = "Current timestamp (computed at startup)"
        else:
            default_description = str(default_value)
    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
        # Has a default factory
        try:
            # Try to get a meaningful description of the factory
            factory = field_info.default_factory
            
            # Handle built-in types
            if factory == dict:
                default_description = "Empty dictionary ({})"
            elif factory == list:
                default_description = "Empty list ([])"
            elif factory == set:
                default_description = "Empty set"
            elif hasattr(factory, '__name__'):
                if factory.__name__ == '<lambda>':
                    # For lambda functions, try to call it and see what we get
                    try:
                        sample_value = factory()
                        if hasattr(sample_value, '__class__'):
                            class_name = sample_value.__class__.__name__
                            if hasattr(sample_value, '__dict__'):
                                # For objects, show some key attributes
                                attrs = sample_value.__dict__
                                if len(attrs) <= 3 and all(isinstance(v, (str, int, float, bool)) for v in attrs.values()):
                                    attr_strs = [f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in list(attrs.items())[:3]]
                                    default_description = f"Factory: {class_name}({', '.join(attr_strs)})"
                                else:
                                    default_description = f"Factory: {class_name}(...)"
                            else:
                                default_description = f"Factory: {class_name}()"
                        else:
                            default_description = f"Factory returns: {str(sample_value)}"
                    except:
                        default_description = "Factory function (lambda)"
                elif factory.__name__ == 'datetime.datetime.now':
                    default_description = "Current timestamp (datetime.now())"
                elif 'datetime' in factory.__name__:
                    default_description = f"Runtime computed ({factory.__name__})"
                else:
                    default_description = f"Factory: {factory.__name__}()"
            else:
                default_description = "Factory function"
        except:
            default_description = "Factory function"
    
    info = {
        'name': field_name,
        'type': format_type_hint(field_type),
        'required': field_info.is_required(),
        'default': default_value,
        'default_description': default_description,
        'description': field_info.description or "No description provided",
        'constraints': {}
    }
    
    # Add validation constraints if any
    if hasattr(field_info, 'constraints'):
        for constraint_name, constraint_value in field_info.constraints.items():
            if constraint_value is not None:
                info['constraints'][constraint_name] = constraint_value
    
    return info

def get_user_defined_fields(model_class: type[BaseModel]):
    """Get only user-defined fields, excluding Pydantic's built-in fields."""
    user_fields = {}
    
    # Get annotations from the class itself (not inherited)
    class_annotations = getattr(model_class, '__annotations__', {})
    
    # Only include fields that are both in model_fields AND have class annotations
    for field_name, field_info in model_class.model_fields.items():
        if field_name in class_annotations:
            user_fields[field_name] = field_info
    
    return user_fields


def generate_model_documentation(model_class: type[BaseModel], filename: str):
    """Generate documentation for a single Pydantic model."""
    
    # Get the class docstring directly from the class, not inherited
    class_doc = None
    if hasattr(model_class, '__doc__') and model_class.__doc__:
        class_doc = model_class.__doc__.strip()
    
    # Check if this is just the BaseModel docstring by comparing
    if not class_doc or "A base class for creating Pydantic models" in class_doc:
        class_doc = f"Documentation for {model_class.__name__}"
    
    # Start building the markdown content
    content = [
        f"# {model_class.__name__}",
        "",
        class_doc,
        "",
        "## Field Definitions",
        "",
        "| Field | Type | Required | Default | Description |",
        "|-------|------|----------|---------|-------------|"
    ]
    
    # Get only user-defined fields
    user_fields = get_user_defined_fields(model_class)
    
    for field_name in user_fields:
        field_info = get_field_info(model_class, field_name)
        if field_info:
            required_text = "✓" if field_info['required'] else "✗"
            if field_info['default_description'] is not None:
                default_text = field_info['default_description']
                if len(default_text) > 50:
                    default_text = default_text[:47] + "..."
            else:
                default_text = "—"
            
            content.append(
                f"| `{field_info['name']}` | {field_info['type']} | {required_text} | `{default_text}` | {field_info['description']} |"
            )
    
    content.extend([
        "",
        "## Detailed Field Descriptions",
        ""
    ])
    
    # Add detailed descriptions for each user-defined field
    for field_name in user_fields:
        field_info = get_field_info(model_class, field_name)
        if field_info:
            content.extend([
                f"### {field_info['name']}",
                "",
                f"**Type:** {field_info['type']}",
                f"**Required:** {'Yes' if field_info['required'] else 'No'}",
            ])
            
            if field_info['default_description'] is not None:
                content.append(f"**Default:** {field_info['default_description']}")
            
            content.extend([
                "",
                field_info['description'],
                ""
            ])
            
            # Add constraints if any
            if field_info['constraints']:
                content.extend([
                    "**Validation Constraints:**",
                    ""
                ])
                for constraint, value in field_info['constraints'].items():
                    content.append(f"- `{constraint}`: {value}")
                content.append("")
    
    # Add usage example - improved version
    if model_class.__name__ in ['DataProvenance', 'Signal', 'Dataset', 'TimeSeries', 'ProcessingStep', 'FunctionInfo']:
        content.extend([
            "## Usage Example",
            "",
            "```python",
            f"from meteaudata.types import {model_class.__name__}",
            "",
            f"# Create a {model_class.__name__} instance",
        ])
        
        # Generate model-specific examples
        if model_class.__name__ == 'DataProvenance':
            content.extend([
                "provenance = DataProvenance(",
                '    source_repository="station_database",',
                '    project="water_quality_monitoring",',
                '    location="river_site_A",',
                '    equipment="multiparameter_probe",',
                '    parameter="dissolved_oxygen",',
                '    purpose="compliance_monitoring"',
                ")"
            ])
        elif model_class.__name__ == 'FunctionInfo':
            content.extend([
                "func_info = FunctionInfo(",
                '    name="moving_average_smooth",',
                '    version="1.2.0",',
                '    author="Data Processing Team",',
                '    reference="https://docs.example.com/smoothing"',
                ")"
            ])
        elif model_class.__name__ == 'ProcessingStep':
            content.extend([
                "from datetime import datetime",
                "",
                "step = ProcessingStep(",
                "    type=ProcessingType.SMOOTHING,",
                '    description="Applied moving average smoothing",',
                "    run_datetime=datetime.now(),",
                "    requires_calibration=False,",
                "    function_info=func_info,",
                '    suffix="SMOOTH",',
                '    input_series_names=["temperature#1_RAW#1"]',
                ")"
            ])
        elif model_class.__name__ == 'TimeSeries':
            content.extend([
                "import pandas as pd",
                "",
                "# Create with pandas Series",
                "data = pd.Series([20, 21, 22, 23], name='temperature')",
                "ts = TimeSeries(series=data)",
                "",
                "# Or load from files",
                "ts = TimeSeries.load(",
                '    data_file_path="data.csv",',
                '    metadata_file_path="metadata.yaml"',
                ")"
            ])
        elif model_class.__name__ == 'Signal':
            content.extend([
                "signal = Signal(",
                "    input_data=temperature_series,",
                '    name="temperature",',
                '    units="°C",',
                "    provenance=provenance",
                ")"
            ])
        elif model_class.__name__ == 'Dataset':
            content.extend([
                "dataset = Dataset(",
                '    name="river_monitoring_2024",',
                '    description="Continuous water quality monitoring",',
                '    owner="Environmental Team",',
                "    signals={",
                '        "temperature": temp_signal,',
                '        "dissolved_oxygen": do_signal',
                "    },",
                '    project="water_quality_assessment"',
                ")"
            ])
        else:
            # Generic example for other models
            required_fields = [name for name, field in user_fields.items() if field.is_required()]
            content.append(f"instance = {model_class.__name__}(")
            if required_fields:
                for i, field_name in enumerate(required_fields[:3]):  # Show max 3 required fields
                    field_info = get_field_info(model_class, field_name)
                    if field_info:
                        example_value = get_example_value(field_info['type'])
                        ending = "," if i < len(required_fields[:3]) - 1 else ""
                        content.append(f"    {field_name}={example_value}{ending}")
            content.append(")")
        
        content.extend([
            "```",
            ""
        ])
    
    # Write the file
    with mkdocs_gen_files.open(filename, "w") as f:
        f.write("\n".join(content))
        
def get_example_value(type_str: str) -> str:
    """Generate example values based on type."""
    type_lower = type_str.lower()
    if 'str' in type_lower:
        return '"example_value"'
    elif 'int' in type_lower:
        return '42'
    elif 'float' in type_lower:
        return '3.14'
    elif 'bool' in type_lower:
        return 'True'
    elif 'datetime' in type_lower:
        return 'datetime.datetime.now()'
    elif 'list' in type_lower:
        return '[]'
    elif 'dict' in type_lower:
        return '{}'
    else:
        return 'None'


def generate_enum_documentation(enum_class, filename: str):
    """Generate documentation for enum classes."""
    
    # Get the enum docstring directly from the class
    class_doc = None
    if hasattr(enum_class, '__doc__') and enum_class.__doc__:
        class_doc = enum_class.__doc__.strip()
    
    if not class_doc:
        class_doc = f"Enumeration defining the available {enum_class.__name__.lower()} values."
    
    content = [
        f"# {enum_class.__name__}",
        "",
        class_doc,
        "",
        "## Available Values",
        "",
        "| Value | Enum Key | Description |",
        "|-------|----------|-------------|"
    ]
    
    # Try to get field descriptions from source code comments
    enum_descriptions = {}
    try:
        import inspect
        source_lines = inspect.getsourcelines(enum_class)[0]
        for line in source_lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                # This is an enum field definition
                field_name = line.split('=')[0].strip()
                # Check if there's a comment on the same line
                if '#' in line:
                    comment_part = line.split('#', 1)[1].strip()
                    # Handle format: FIELD = "value"  # "Description"
                    if comment_part.startswith('"') and comment_part.endswith('"'):
                        enum_descriptions[field_name] = comment_part[1:-1]
    except Exception:
        # If we can't parse source, just continue without descriptions
        pass
    
    for item in enum_class:
        description = enum_descriptions.get(item.name, "No description available")
        content.append(f"| `{item.value}` | `{item.name}` | {description} |")
    
    content.extend([
        "",
        "## Usage Example",
        "",
        "```python",
        f"from meteaudata.types import {enum_class.__name__}",
        "",
        f"# Use in a ProcessingStep",
        f"step_type = {enum_class.__name__}.{list(enum_class)[0].name}",
        "```",
        ""
    ])
    
    with mkdocs_gen_files.open(filename, "w") as f:
        f.write("\n".join(content))

def generate_protocol_documentation(protocol_class, filename: str):
    """Generate documentation for Protocol classes."""
    
    # Get the protocol docstring directly from the class
    class_doc = None
    if hasattr(protocol_class, '__doc__') and protocol_class.__doc__:
        class_doc = protocol_class.__doc__.strip()
    
    if not class_doc:
        class_doc = f"Protocol defining the interface for {protocol_class.__name__}."
    
    content = [
        f"# {protocol_class.__name__}",
        "",
        class_doc,
        "",
    ]
    
    # Try to extract method signatures from the protocol
    try:
        import inspect
        
        # Get the __call__ method if it exists
        if hasattr(protocol_class, '__call__'):
            call_method = getattr(protocol_class, '__call__')
            if hasattr(call_method, '__annotations__'):
                content.extend([
                    "## Method Signature",
                    "",
                    "```python",
                    f"def __call__({', '.join(inspect.signature(call_method).parameters.keys())})",
                    "```",
                    ""
                ])
        
        # Get all methods defined in the protocol
        for name, method in inspect.getmembers(protocol_class, predicate=inspect.isfunction):
            if not name.startswith('_') or name == '__call__':
                method_doc = inspect.getdoc(method)
                if method_doc:
                    content.extend([
                        f"## {name}",
                        "",
                        method_doc,
                        ""
                    ])
    
    except Exception as e:
        # If we can't parse the protocol, just add a basic note
        content.extend([
            "## Protocol Definition",
            "",
            "This protocol defines the interface that implementing functions must follow.",
            "See the class docstring above for detailed usage information.",
            ""
        ])
    
    with mkdocs_gen_files.open(filename, "w") as f:
        f.write("\n".join(content))

# Update your imports section:
try:
    from pydantic import BaseModel
    from meteaudata.types import (
        DataProvenance, ProcessingStep, FunctionInfo, Parameters, 
        IndexMetadata, TimeSeries, Signal, Dataset, ProcessingType,
        SignalTransformFunctionProtocol, DatasetTransformFunctionProtocol
    )
except ImportError as e:
    print(f"Warning: Could not import meteaudata types: {e}")
    print("Make sure meteaudata is installed: uv pip install -e .")
    exit(1)

# Updated main function:
def main():
    """Generate all metadata dictionary documentation."""
    
    print("=== STARTING METADATA DICTIONARY GENERATION ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Models to document
    models = [
        (DataProvenance, "metadata-dictionary/data-provenance.md"),
        (ProcessingStep, "metadata-dictionary/processing-step.md"),
        (FunctionInfo, "metadata-dictionary/function-info.md"),
        (Parameters, "metadata-dictionary/parameters.md"),
        (IndexMetadata, "metadata-dictionary/index-metadata.md"),
        (TimeSeries, "metadata-dictionary/time-series.md"),
        (Signal, "metadata-dictionary/signal.md"),
        (Dataset, "metadata-dictionary/dataset.md"),
    ]
    
    # Generate documentation for each model
    for model_class, filename in models:
        print(f"Generating documentation for {model_class.__name__} -> {filename}")
        generate_model_documentation(model_class, filename)
        print(f"✓ Generated {filename}")
    
    # Generate enum documentation
    print("Generating documentation for ProcessingType -> metadata-dictionary/processing-type.md")
    generate_enum_documentation(ProcessingType, "metadata-dictionary/processing-type.md")
    print("✓ Generated metadata-dictionary/processing-type.md")
    
    print("=== COMPLETED METADATA DICTIONARY GENERATION ===")
    
    # Generate protocol documentation
    protocols = [
        (SignalTransformFunctionProtocol, "metadata-dictionary/signal-transform-protocol.md"),
        (DatasetTransformFunctionProtocol, "metadata-dictionary/dataset-transform-protocol.md"),
    ]
    
    for protocol_class, filename in protocols:
        generate_protocol_documentation(protocol_class, filename)
    
    # Generate the main metadata dictionary index
    index_content = [
        "# Metadata Dictionary",
        "",
        "This section provides the official definitions for all metadata attributes used in metEAUdata.",
        "Each page documents the fields, types, and validation rules for the core data structures.",
        "",
        "## Core Metadata Classes",
        "",
        "- **[DataProvenance](data-provenance.md)** - Information about data sources and context",
        "- **[ProcessingStep](processing-step.md)** - Documentation of data processing operations",
        "- **[FunctionInfo](function-info.md)** - Metadata about processing functions",
        "- **[Parameters](parameters.md)** - Storage for processing function parameters",
        "- **[IndexMetadata](index-metadata.md)** - Time series index information",
        "",
        "## Data Container Classes",
        "",
        "- **[TimeSeries](time-series.md)** - Individual time series with processing history",
        "- **[Signal](signal.md)** - Collection of related time series",
        "- **[Dataset](dataset.md)** - Collection of signals with project metadata",
        "",
        "## Enumerations",
        "",
        "- **[ProcessingType](processing-type.md)** - Standardized processing step categories",
        "",
        "## Protocols",
        "",
        "- **[SignalTransformFunctionProtocol](signal-transform-protocol.md)** - Interface for Signal-level processing functions",
        "- **[DatasetTransformFunctionProtocol](dataset-transform-protocol.md)** - Interface for Dataset-level processing functions",
        "",
        "## Standards and Conventions",
        "",
        "### Naming Conventions",
        "",
        "- **Signal Names**: Use descriptive names followed by `#N` numbering (e.g., `temperature#1`)",
        "- **Time Series Names**: Format as `{signal_name}_{processing_suffix}#{number}` (e.g., `temperature#1_SMOOTH#1`)",
        "- **Processing Suffixes**: Use 3-4 letter abbreviations describing the operation (e.g., `SMOOTH`, `FILT`, `RESAMP`)",
        "",
        "### Required vs Optional Fields",
        "",
        "- ✓ indicates a required field that must be provided",
        "- ✗ indicates an optional field with a default value",
        "",
        "### Type Annotations",
        "",
        "All type annotations follow Python type hint standards:",
        "",
        "- `str` - Text string",
        "- `int` - Integer number", 
        "- `float` - Decimal number",
        "- `bool` - True/False value",
        "- `datetime.datetime` - Date and time",
        "- `Optional[T]` - Field can be type T or None",
        "- `list[T]` - List containing items of type T",
        "- `dict[K, V]` - Dictionary with keys of type K and values of type V",
        ""
    ]
    
    with mkdocs_gen_files.open("metadata-dictionary/index.md", "w") as f:
        f.write("\n".join(index_content))


main()