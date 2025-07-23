#!/usr/bin/env python3
"""
Script to automatically generate visualization API documentation
from meteaudata classes.

This script is run by mkdocs-gen-files during documentation build.
"""

import inspect
import os
from pathlib import Path
from typing import get_type_hints, get_origin, get_args
import mkdocs_gen_files

# Handle imports gracefully for documentation builds
try:
    from meteaudata.types import TimeSeries, Signal, Dataset
    from meteaudata.displayable import DisplayableBase
    print("DEBUG: Successfully imported meteaudata visualization classes")
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


def parse_docstring_params(docstring):
    """Parse parameter descriptions from Google-style docstring."""
    if not docstring:
        return {}
    
    param_descriptions = {}
    lines = docstring.split('\n')
    in_args_section = False
    current_param = None
    
    for line in lines:
        line = line.strip()
        
        # Look for Args section
        if line.lower().startswith('args:'):
            in_args_section = True
            continue
        
        # Stop at next section
        if in_args_section and line.endswith(':') and not line.startswith(' '):
            break
            
        # Parse parameter lines
        if in_args_section and ':' in line and not line.startswith(' ' * 8):  # Not a continuation
            param_parts = line.split(':', 1)
            if len(param_parts) == 2:
                param_name = param_parts[0].strip()
                param_desc = param_parts[1].strip()
                param_descriptions[param_name] = param_desc
                current_param = param_name
        elif in_args_section and current_param and line.startswith(' '):
            # Continuation of previous parameter description
            param_descriptions[current_param] += ' ' + line.strip()
    
    return param_descriptions


def extract_method_info(cls, method_name):
    """Extract comprehensive method information."""
    method = getattr(cls, method_name)
    
    # Get method signature
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        sig = None
    
    # Get docstring
    doc = inspect.getdoc(method)
    
    # Parse parameter descriptions from docstring
    param_descriptions = parse_docstring_params(doc)
    
    # Get type hints
    try:
        type_hints = get_type_hints(method)
    except (NameError, AttributeError):
        type_hints = {}
    
    # Extract parameters
    parameters = []
    if sig:
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'name': param_name,
                'type': format_type_hint(type_hints.get(param_name, param.annotation)) if param.annotation != inspect.Parameter.empty else "`Any`",
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty,
                'description': param_descriptions.get(param_name, "No description")
            }
            parameters.append(param_info)
    
    # Get return type
    return_type = None
    if sig and sig.return_annotation != inspect.Parameter.empty:
        return_type = format_type_hint(type_hints.get('return', sig.return_annotation))
    elif 'return' in type_hints:
        return_type = format_type_hint(type_hints['return'])
    
    return {
        'name': method_name,
        'signature': str(sig) if sig else None,
        'docstring': doc,
        'parameters': parameters,
        'return_type': return_type or "`Any`"
    }


def generate_class_visualization_docs(cls, methods, filename):
    """Generate visualization documentation for a class."""
    
    class_doc = inspect.getdoc(cls) or f"Visualization methods for {cls.__name__}"
    
    content = [
        f"# {cls.__name__} Visualization API",
        "",
        class_doc,
        "",
        "## Methods",
        ""
    ]
    
    for method_name in methods:
        if not hasattr(cls, method_name):
            continue
            
        method_info = extract_method_info(cls, method_name)
        
        content.extend([
            f"### {method_name}",
            "",
        ])
        
        if method_info['signature']:
            content.extend([
                "**Signature:**",
                "",
                "```python",
                f"def {method_name}{method_info['signature'][method_info['signature'].find('('):]}",
                "```",
                ""
            ])
        
        if method_info['docstring']:
            content.extend([
                "**Description:**",
                "",
                method_info['docstring'],
                ""
            ])
        
        if method_info['parameters']:
            content.extend([
                "**Parameters:**",
                "",
                "| Parameter | Type | Required | Default | Description |",
                "|-----------|------|----------|---------|-------------|"
            ])
            
            for param in method_info['parameters']:
                required_text = "✓" if param['required'] else "✗"
                default_text = str(param['default']) if param['default'] is not None else "—"
                if len(default_text) > 30:
                    default_text = default_text[:27] + "..."
                
                content.append(
                    f"| `{param['name']}` | {param['type']} | {required_text} | `{default_text}` | {param['description']} |"
                )
            
            content.append("")
        
        content.extend([
            f"**Returns:** {method_info['return_type']}",
            "",
            "---",
            ""
        ])
    
    # Write the file
    with mkdocs_gen_files.open(filename, "w") as f:
        f.write("\n".join(content))


def generate_display_system_docs():
    """Generate documentation for DisplayableBase methods."""
    
    display_methods = [
        'display',
        'show_summary', 
        'show_details',
        'show_graph',
        'show_graph_in_browser'
    ]
    
    content = [
        "# Display System API",
        "",
        "Complete API reference for the meteaudata display system methods.",
        "All meteaudata objects inherit from `DisplayableBase` and provide rich visualization capabilities.",
        "",
        "## Overview",
        "",
        "The display system provides multiple output formats:",
        "",
        "- **Text display** - Simple text representation",
        "- **HTML display** - Rich HTML with expandable sections (Jupyter notebooks)",
        "- **Graph display** - Interactive SVG graphs of metadata structure",
        "- **Browser display** - Full-page interactive visualization",
        "",
        "## Methods",
        ""
    ]
    
    for method_name in display_methods:
        if not hasattr(DisplayableBase, method_name):
            continue
            
        method_info = extract_method_info(DisplayableBase, method_name)
        
        content.extend([
            f"### {method_name}",
            "",
        ])
        
        if method_info['signature']:
            content.extend([
                "**Signature:**",
                "",
                "```python",
                f"def {method_name}{method_info['signature'][method_info['signature'].find('('):]}",
                "```",
                ""
            ])
        
        if method_info['docstring']:
            content.extend([
                "**Description:**",
                "",
                method_info['docstring'],
                ""
            ])
        
        if method_info['parameters']:
            content.extend([
                "**Parameters:**",
                "",
                "| Parameter | Type | Required | Default | Description |",
                "|-----------|------|----------|---------|-------------|"
            ])
            
            for param in method_info['parameters']:
                required_text = "✓" if param['required'] else "✗"
                default_text = str(param['default']) if param['default'] is not None else "—"
                if len(default_text) > 30:
                    default_text = default_text[:27] + "..."
                
                content.append(
                    f"| `{param['name']}` | {param['type']} | {required_text} | `{default_text}` | {param['description']} |"
                )
            
            content.append("")
        
        content.extend([
            f"**Returns:** {method_info['return_type']}",
            "",
            "---",
            ""
        ])
    
    # Add usage examples
    content.extend([
        "## Common Usage Patterns",
        "",
        "### Quick Display Methods",
        "",
        "```python",
        "# Quick text summary",
        "signal.show_summary()",
        "",
        "# Rich HTML display (Jupyter)",
        "signal.show_details()",
        "",
        "# Interactive graph",
        "signal.show_graph()",
        "```",
        "",
        "### Customized Display",
        "",
        "```python",
        "# Custom text display",
        "signal.display(format='text', depth=3)",
        "",
        "# Custom HTML display",
        "signal.display(format='html', depth=4)",
        "",
        "# Custom graph display",
        "signal.display(format='graph', max_depth=5, width=1400, height=900)",
        "```",
        "",
        "### Browser Visualization",
        "",
        "```python",
        "# Open in browser with custom settings",
        "html_path = signal.show_graph_in_browser(",
        "    max_depth=4,",
        "    width=1600,",
        "    height=1000,",
        "    title='Custom Visualization'",
        ")",
        "```",
        ""
    ])
    
    with mkdocs_gen_files.open("api-reference/visualization/display-system.md", "w") as f:
        f.write("\n".join(content))


def main():
    """Generate all visualization API documentation."""
    
    print("=== STARTING VISUALIZATION API GENERATION ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Generate documentation for plotting methods
    plotting_classes = [
        (TimeSeries, ['plot'], "api-reference/visualization/timeseries-plotting.md"),
        (Signal, ['plot', 'plot_dependency_graph'], "api-reference/visualization/signal-plotting.md"),
        (Dataset, ['plot'], "api-reference/visualization/dataset-plotting.md"),
    ]
    
    for cls, methods, filename in plotting_classes:
        print(f"Generating plotting documentation for {cls.__name__} -> {filename}")
        generate_class_visualization_docs(cls, methods, filename)
        print(f"✓ Generated {filename}")
    
    # Generate display system documentation
    print("Generating display system documentation")
    generate_display_system_docs()
    print("✓ Generated display system documentation")
    
    # Generate main visualization API index
    index_content = [
        "# Visualization API Reference",
        "",
        "Complete API reference for meteaudata's visualization capabilities.",
        "",
        "## Plotting Methods",
        "",
        "Interactive Plotly-based plotting for time series data:",
        "",
        "- **[TimeSeries Plotting](timeseries-plotting.md)** - Individual time series visualization",
        "- **[Signal Plotting](signal-plotting.md)** - Multi-time series and dependency graphs", 
        "- **[Dataset Plotting](dataset-plotting.md)** - Multi-signal subplot visualization",
        "",
        "## Display System",
        "",
        "Rich metadata exploration and visualization:",
        "",
        "- **[Display System](display-system.md)** - Text, HTML, and interactive graph display methods",
        "",
        "## Overview",
        "",
        "### Plotting System",
        "",
        "meteaudata provides three main plotting classes:",
        "",
        "1. **TimeSeries.plot()** - Plot individual time series with automatic styling based on processing type",
        "2. **Signal.plot()** - Plot multiple time series from a signal, with dependency graph visualization",
        "3. **Dataset.plot()** - Plot multiple signals using subplots for comparison",
        "",
        "All plotting methods return Plotly Figure objects that can be customized further.",
        "",
        "### Display System",
        "",
        "All meteaudata objects inherit rich display capabilities:",
        "",
        "- **Text Display** - Simple text representation with configurable depth",
        "- **HTML Display** - Rich HTML with collapsible sections (Jupyter notebooks)",
        "- **Graph Display** - Interactive SVG visualization of metadata structure",
        "- **Browser Display** - Full-page interactive exploration",
        "",
        "### Key Features",
        "",
        "**Automatic Styling:**",
        "- Processing type-specific markers and modes",
        "- Temporal shifting for prediction data",
        "- Color cycling for multiple series",
        "",
        "**Interactivity:**",
        "- Plotly-based interactive charts",
        "- Zoom, pan, and hover capabilities",
        "- Exportable to HTML, PNG, PDF",
        "",
        "**Metadata Integration:**",
        "- Processing history visualization",
        "- Dependency graph generation",
        "- Complete audit trail display",
        "",
        "## Common Parameters",
        "",
        "### Plotting Parameters",
        "",
        "Most plotting methods accept these common parameters:",
        "",
        "| Parameter | Type | Description |",
        "|-----------|------|-------------|",
        "| `title` | `str` | Plot title |",
        "| `x_axis` | `str` | X-axis label |", 
        "| `y_axis` | `str` | Y-axis label |",
        "| `start` | `str` | Start date for filtering |",
        "| `end` | `str` | End date for filtering |",
        "",
        "### Display Parameters",
        "",
        "Display methods commonly accept:",
        "",
        "| Parameter | Type | Description |",
        "|-----------|------|-------------|",
        "| `format` | `str` | Output format: 'text', 'html', 'graph' |",
        "| `depth` | `int` | Display depth for text/HTML |",
        "| `max_depth` | `int` | Maximum depth for graph display |",
        "| `width` | `int` | Graph width in pixels |",
        "| `height` | `int` | Graph height in pixels |",
        ""
    ]
    
    with mkdocs_gen_files.open("api-reference/visualization/index.md", "w") as f:
        f.write("\n".join(index_content))
    
    print("✓ Generated visualization API index")
    print("=== COMPLETED VISUALIZATION API GENERATION ===")


if __name__ == "__main__":
    main()