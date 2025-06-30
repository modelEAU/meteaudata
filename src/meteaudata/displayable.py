"""
Display extensions for meteaudata objects.
Refactored to use inheritance with minimal code duplication.
"""

from typing import Any, Dict
from datetime import datetime
from abc import ABC, abstractmethod


def _is_jupyter_environment() -> bool:
    """Check if we're running in a Jupyter environment."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _import_widgets():
    """Safely import ipywidgets."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
        return widgets, display
    except ImportError:
        return None, None


def _is_complex_object(obj: Any) -> bool:
    """Check if an object is complex and should be expandable."""
    return hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, datetime))


def _format_simple_value(value: Any) -> str:
    """Format simple values for display."""
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(value, (list, tuple)) and len(value) > 3:
        return f"{type(value).__name__}[{len(value)} items]"
    else:
        return str(value)


# HTML style constants
HTML_STYLE = """
<style>
.meteaudata-display {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #24292f;
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 16px;
    margin: 8px 0;
}
.meteaudata-header {
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 12px;
    color: #0969da;
}
.meteaudata-attr {
    margin: 4px 0;
    padding: 2px 0;
}
.meteaudata-attr-name {
    font-weight: 600;
    color: #656d76;
    display: inline-block;
    min-width: 120px;
}
.meteaudata-attr-value {
    color: #24292f;
}
.meteaudata-nested {
    margin-left: 20px;
    border-left: 2px solid #f6f8fa;
    padding-left: 12px;
    margin-top: 8px;
}
details.meteaudata-collapsible {
    margin: 4px 0;
}
summary.meteaudata-summary {
    cursor: pointer;
    font-weight: 600;
    color: #656d76;
    padding: 4px 0;
}
summary.meteaudata-summary:hover {
    color: #0969da;
}
</style>
"""


class DisplayableBase(ABC):
    """
    Base class for all meteaudata objects that need rich display functionality.
    Can be mixed with pydantic BaseModel or used standalone.
    """
    
    @abstractmethod
    def _get_display_attributes(self) -> Dict[str, Any]:
        """Get attributes to display. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_identifier(self) -> str:
        """Get the key identifier for this object. Must be implemented by subclasses."""
        pass
    
    def __str__(self) -> str:
        """Short description: Object type + key identifier."""
        obj_type = self.__class__.__name__
        identifier = self._get_identifier()
        return f"{obj_type}({identifier})"
    
    def display(self, format: str = "text", depth: int = 2, use_widgets: bool = True) -> None:
        """
        Display rich representation of the object.
        
        Args:
            format: "text" or "html"
            depth: how many levels deep to expand complex objects
            use_widgets: if True and in Jupyter, use ipywidgets
        """
        if format == "text":
            print(self._render_text(depth))
        elif format == "html":
            if use_widgets and _is_jupyter_environment():
                self._render_widget(depth)
            else:
                self._render_html(depth)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _render_text(self, depth: int, indent: int = 0) -> str:
        """Render text representation."""
        lines = []
        prefix = "  " * indent
        
        # Object header
        lines.append(f"{prefix}{self.__class__.__name__}:")
        
        # Attributes
        for attr_name, attr_value in self._get_display_attributes().items():
            if depth <= 0:
                if hasattr(attr_value, '_render_text'):
                    value_str = str(attr_value)
                else:
                    value_str = f"{type(attr_value).__name__}(...)"
            elif _is_complex_object(attr_value):
                if hasattr(attr_value, '_render_text'):
                    value_str = "\n" + attr_value._render_text(depth - 1, indent + 1)
                else:
                    value_str = str(attr_value)
            else:
                value_str = _format_simple_value(attr_value)
            
            lines.append(f"{prefix}  {attr_name}: {value_str}")
        
        return "\n".join(lines)
    
    def _render_html(self, depth: int) -> None:
        """Render HTML representation and display it."""
        try:
            from IPython.display import HTML, display
            html_content = f"{HTML_STYLE}<div class='meteaudata-display'>{self._build_html_content(depth)}</div>"
            display(HTML(html_content))
        except ImportError:
            print(self._render_text(depth))
    
    def _render_widget(self, depth: int) -> None:
        """Render widget representation."""
        widgets, display_func = _import_widgets()
        if widgets is None:
            self._render_html(depth)
            return
        
        widget = self._build_widget(depth, widgets)
        display_func(widget)
    
    def _build_html_content(self, depth: int) -> str:
        """Build HTML content for the object."""
        lines = []
        
        # Header
        lines.append(f"<div class='meteaudata-header'>{self.__class__.__name__}</div>")
        
        # Attributes
        for attr_name, attr_value in self._get_display_attributes().items():
            if depth <= 0:
                if hasattr(attr_value, '_build_html_content'):
                    value_str = str(attr_value)
                else:
                    value_str = f"{type(attr_value).__name__}(...)"
                lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{value_str}</span></div>")
            elif _is_complex_object(attr_value):
                if hasattr(attr_value, '_build_html_content'):
                    nested_content = attr_value._build_html_content(depth - 1)
                    lines.append(f"""
                    <details class='meteaudata-collapsible'>
                        <summary class='meteaudata-summary'>{attr_name}: {type(attr_value).__name__}</summary>
                        <div class='meteaudata-nested'>{nested_content}</div>
                    </details>
                    """)
                else:
                    lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{str(attr_value)}</span></div>")
            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                # Handle collections that might contain displayable objects
                if hasattr(attr_value[0], '_build_html_content'):
                    # This is a list of displayable objects
                    nested_items = []
                    for i, item in enumerate(attr_value):
                        if i >= 10:  # Limit to first 10 items to avoid overwhelming display
                            nested_items.append(f"<div class='meteaudata-attr'>... and {len(attr_value) - 10} more items</div>")
                            break
                        item_content = item._build_html_content(depth - 1)
                        nested_items.append(f"<div class='meteaudata-nested'>{item_content}</div>")
                    
                    nested_content = "\n".join(nested_items)
                    lines.append(f"""
                    <details class='meteaudata-collapsible'>
                        <summary class='meteaudata-summary'>{attr_name}: {type(attr_value).__name__}[{len(attr_value)} items]</summary>
                        <div class='meteaudata-nested'>{nested_content}</div>
                    </details>
                    """)
                else:
                    # Regular list of simple values
                    value_str = _format_simple_value(attr_value)
                    lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{value_str}</span></div>")
            elif isinstance(attr_value, dict):
                # Handle dictionaries that might contain displayable objects
                if any(hasattr(v, '_build_html_content') for v in attr_value.values()):
                    # This dictionary contains displayable objects
                    nested_items = []
                    for key, value in list(attr_value.items())[:10]:  # Limit to first 10 items
                        if hasattr(value, '_build_html_content'):
                            item_content = value._build_html_content(depth - 1)
                            nested_items.append(f"<div class='meteaudata-nested'><strong>{key}:</strong><br>{item_content}</div>")
                        else:
                            nested_items.append(f"<div class='meteaudata-attr'><strong>{key}:</strong> {_format_simple_value(value)}</div>")
                    
                    if len(attr_value) > 10:
                        nested_items.append(f"<div class='meteaudata-attr'>... and {len(attr_value) - 10} more items</div>")
                    
                    nested_content = "\n".join(nested_items)
                    lines.append(f"""
                    <details class='meteaudata-collapsible'>
                        <summary class='meteaudata-summary'>{attr_name}: dict[{len(attr_value)} items]</summary>
                        <div class='meteaudata-nested'>{nested_content}</div>
                    </details>
                    """)
                else:
                    # Regular dictionary
                    value_str = _format_simple_value(attr_value)
                    lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{value_str}</span></div>")
            else:
                value_str = _format_simple_value(attr_value)
                lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{value_str}</span></div>")
        
        return "\n".join(lines)
    
    def _build_widget(self, depth: int, widgets) -> Any:
        """Build widget representation."""
        children = []
        
        # Header
        header = widgets.HTML(f"<h3 style='margin:0; color:#0969da; font-family: system-ui;'>{self.__class__.__name__}</h3>")
        children.append(header)
        
        # Attributes
        for attr_name, attr_value in self._get_display_attributes().items():
            if depth <= 0:
                if hasattr(attr_value, '_build_widget'):
                    value_str = str(attr_value)
                else:
                    value_str = f"{type(attr_value).__name__}(...)"
                children.append(widgets.HTML(f"<div style='font-family: system-ui; margin: 2px 0;'><strong>{attr_name}:</strong> {value_str}</div>"))
            elif _is_complex_object(attr_value):
                if hasattr(attr_value, '_build_widget'):
                    nested_widget = attr_value._build_widget(depth - 1, widgets)
                    accordion = widgets.Accordion(children=[nested_widget])
                    accordion.set_title(0, f"{attr_name}: {type(attr_value).__name__}")
                    accordion.selected_index = None  # Start collapsed
                    children.append(accordion)
                else:
                    children.append(widgets.HTML(f"<div style='font-family: system-ui; margin: 2px 0;'><strong>{attr_name}:</strong> {str(attr_value)}</div>"))
            else:
                value_str = _format_simple_value(attr_value)
                children.append(widgets.HTML(f"<div style='font-family: system-ui; margin: 2px 0;'><strong>{attr_name}:</strong> {value_str}</div>"))
        
        return widgets.VBox(children, layout=widgets.Layout(
            border='1px solid #d0d7de',
            border_radius='6px',
            padding='16px',
            margin='8px 0'
        ))
