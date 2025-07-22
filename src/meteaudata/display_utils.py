from typing import Any, Tuple
from datetime import datetime


def _is_notebook_environment() -> bool:
    """
    Check if we're running in a notebook environment (Jupyter or similar).
    
    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _is_complex_object(obj: Any) -> bool:
    """
    Check if an object is complex enough to warrant nested display.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object should be displayed with expansion/nesting
    """
    # Check if it's a meteaudata displayable object
    if hasattr(obj, '_get_display_attributes') and hasattr(obj, '_get_identifier'):
        return True
    
    # Check for complex built-in types
    if isinstance(obj, dict):
        # Complex if it has multiple keys or nested structures
        if len(obj) > 1:
            return True
        # Or if it contains complex values
        return any(_is_complex_object(v) for v in obj.values())
    
    if isinstance(obj, (list, tuple)):
        # Complex if it's long or contains complex items
        if len(obj) > 3:
            return True
        return any(_is_complex_object(item) for item in obj[:3])
    
    # Check for objects with custom attributes (but not basic types)
    if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, datetime)):
        return True
    
    return False


def _format_simple_value(value: Any, max_length: int = 50) -> str:
    """
    Format simple values for display in HTML/text.
    
    Args:
        value: Value to format
        max_length: Maximum length before truncation
        
    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"
    
    if isinstance(value, str):
        if len(value) > max_length:
            return f"'{value[:max_length-3]}...'"
        return f"'{value}'"
    
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return f"{type(value).__name__}[]"
        elif len(value) <= 3:
            # Show first few items
            items_str = ", ".join([_format_simple_value(item, 20) for item in value])
            return f"{type(value).__name__}[{items_str}]"
        else:
            # Show count for long lists
            return f"{type(value).__name__}[{len(value)} items]"
    
    if isinstance(value, dict):
        if len(value) == 0:
            return "dict{}"
        elif len(value) <= 2:
            # Show small dicts
            items = []
            for k, v in list(value.items())[:2]:
                key_str = str(k)[:15]
                val_str = _format_simple_value(v, 15)
                items.append(f"{key_str}: {val_str}")
            return f"dict{{{', '.join(items)}}}"
        else:
            return f"dict[{len(value)} items]"
    
    # Handle numpy arrays if present
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    
    # For other objects, use string representation but truncate if needed
    str_repr = str(value)
    if len(str_repr) > max_length:
        return f"{str_repr[:max_length-3]}..."
    
    return str_repr


def _get_object_color_and_symbol(obj_type: str) -> Tuple[str, str]:
    """
    Get color and symbol for an object type for graph visualization.
    
    Args:
        obj_type: Name of the object type
        
    Returns:
        Tuple of (color_hex, symbol_name)
    """
    colors = {
        'Dataset': '#1f77b4',      # Blue
        'Signal': '#ff7f0e',       # Orange  
        'TimeSeries': '#2ca02c',   # Green
        'ProcessingStep': '#d62728', # Red
        'Parameters': '#9467bd',   # Purple
        'ParameterValue': '#8c564b', # Brown
        'FunctionInfo': '#e377c2', # Pink
        'DataProvenance': '#7f7f7f', # Gray
        'IndexMetadata': '#bcbd22', # Olive
        'Container': '#17becf',    # Cyan for container nodes
    }
    
    symbols = {
        'Dataset': 'circle',
        'Signal': 'square',
        'TimeSeries': 'diamond',
        'ProcessingStep': 'triangle-up',
        'Parameters': 'hexagon',
        'ParameterValue': 'pentagon',
        'FunctionInfo': 'star',
        'DataProvenance': 'cross',
        'IndexMetadata': 'x',
        'Container': 'octagon',
    }
    
    color = colors.get(obj_type, '#999999')
    symbol = symbols.get(obj_type, 'circle')
    
    return color, symbol