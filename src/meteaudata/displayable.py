"""
Display extensions for meteaudata objects.
Refactored to use inheritance with minimal code duplication.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from meteaudata.display_utils import (
    _is_notebook_environment,
    _is_complex_object,
    _format_simple_value
)

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
    Enhanced base class for meteaudata objects with SVG graph visualization.
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
        """Render HTML representation with better style injection."""
        try:
            from IPython.display import HTML, display
            
            # Extract CSS content from HTML_STYLE constant
            # Remove <style> and </style> tags and any surrounding whitespace
            css_content = HTML_STYLE.replace('<style>', '').replace('</style>', '').strip()
            
            # Create JavaScript to inject styles
            style_injection = f"""
            <script>
            (function() {{
                var styleId = 'meteaudata-styles';
                if (!document.getElementById(styleId)) {{
                    var style = document.createElement('style');
                    style.id = styleId;
                    style.textContent = `{css_content}`;
                    document.head.appendChild(style);
                }}
            }})();
            </script>
            """
            
            html_content = f"{style_injection}<div class='meteaudata-display'>{self._build_html_content(depth)}</div>"
            display(HTML(html_content))
        except ImportError:
            print(self._render_text(depth))
    
    def _build_html_content(self, depth: int) -> str:
        """Build HTML content for the object."""
        lines = []
        
        # Header
        lines.append(f"<div class='meteaudata-header'>{self.__class__.__name__}</div>")
        
        # Attributes
        attrs = self._get_display_attributes()
        
        # Group signal_* and timeseries_* attributes
        signal_attrs = {}
        timeseries_attrs = {}
        regular_attrs = {}
        
        for attr_name, attr_value in attrs.items():
            if attr_name.startswith('signal_'):
                signal_attrs[attr_name] = attr_value
            elif attr_name.startswith('timeseries_'):
                timeseries_attrs[attr_name] = attr_value
            else:
                regular_attrs[attr_name] = attr_value
        
        # Render regular attributes first
        for attr_name, attr_value in regular_attrs.items():
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
                        if i >= 10:  # Limit to first 10 items
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
            else:
                value_str = _format_simple_value(attr_value)
                lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{value_str}</span></div>")
        
        # Render grouped signals if any
        if signal_attrs and depth > 0:
            nested_items = []
            for signal_name, signal_data in signal_attrs.items():
                clean_name = signal_name.replace('signal_', '')
                if isinstance(signal_data, dict):
                    # Build HTML for signal attributes
                    signal_html_parts = []
                    signal_html_parts.append(f"<div class='meteaudata-header'>Signal: {clean_name}</div>")
                    
                    # Separate regular attributes from timeseries attributes
                    regular_attrs = {}
                    timeseries_attrs_for_signal = {}
                    
                    for key, value in signal_data.items():
                        if key.startswith('timeseries_'):
                            timeseries_attrs_for_signal[key] = value
                        else:
                            regular_attrs[key] = value
                    
                    # Add regular signal attributes
                    for key, value in regular_attrs.items():
                        formatted_value = _format_simple_value(value)
                        signal_html_parts.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{key}:</span> <span class='meteaudata-attr-value'>{formatted_value}</span></div>")
                    
                    # Add timeseries sections within this signal if depth allows
                    if timeseries_attrs_for_signal and depth > 1:
                        ts_items = []
                        for ts_name, ts_data in timeseries_attrs_for_signal.items():
                            clean_ts_name = ts_name.replace('timeseries_', '')
                            if isinstance(ts_data, dict):
                                # Build HTML for time series attributes
                                ts_html_parts = []
                                ts_html_parts.append(f"<div class='meteaudata-header'>TimeSeries: {clean_ts_name}</div>")
                                for key, value in ts_data.items():
                                    formatted_value = _format_simple_value(value)
                                    ts_html_parts.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{key}:</span> <span class='meteaudata-attr-value'>{formatted_value}</span></div>")
                                ts_items.append(f"<div class='meteaudata-nested'>{''.join(ts_html_parts)}</div>")
                        
                        if ts_items:
                            ts_content = "\n".join(ts_items)
                            signal_html_parts.append(f"""
                            <details class='meteaudata-collapsible'>
                                <summary class='meteaudata-summary'>Time Series: [{len(timeseries_attrs_for_signal)} series]</summary>
                                <div class='meteaudata-nested'>{ts_content}</div>
                            </details>
                            """)
                    
                    nested_items.append(f"<div class='meteaudata-nested'>{''.join(signal_html_parts)}</div>")
            
            if nested_items:
                nested_content = "\n".join(nested_items)
                lines.append(f"""
                <details class='meteaudata-collapsible'>
                    <summary class='meteaudata-summary'>Signals: [{len(signal_attrs)} signals]</summary>
                    <div class='meteaudata-nested'>{nested_content}</div>
                </details>
                """)
        
        # Render grouped time series if any
        if timeseries_attrs and depth > 0:
            nested_items = []
            for ts_name, ts_data in timeseries_attrs.items():
                clean_name = ts_name.replace('timeseries_', '')
                if isinstance(ts_data, dict):
                    # Build HTML for time series attributes
                    ts_html_parts = []
                    ts_html_parts.append(f"<div class='meteaudata-header'>TimeSeries: {clean_name}</div>")
                    for key, value in ts_data.items():
                        formatted_value = _format_simple_value(value)
                        ts_html_parts.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{key}:</span> <span class='meteaudata-attr-value'>{formatted_value}</span></div>")
                    nested_items.append(f"<div class='meteaudata-nested'>{''.join(ts_html_parts)}</div>")
            
            if nested_items:
                nested_content = "\n".join(nested_items)
                lines.append(f"""
                <details class='meteaudata-collapsible'>
                    <summary class='meteaudata-summary'>Time Series: [{len(timeseries_attrs)} series]</summary>
                    <div class='meteaudata-nested'>{nested_content}</div>
                </details>
                """)
        
        return "\n".join(lines)
    
    def render_svg_graph(self, max_depth: int = 4, width: int = 1200, 
                        height: int = 800, title: Optional[str] = None) -> str:
        """
        Render as interactive SVG nested box graph and return HTML string.
        
        Args:
            max_depth: Maximum depth to traverse in object hierarchy
            width: Graph width in pixels
            height: Graph height in pixels
            title: Page title (auto-generated if None)
            
        Returns:
            HTML string with embedded interactive SVG graph
        """
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            if title is None:
                title = f"Interactive {self.__class__.__name__} Hierarchy"
            
            renderer = SVGNestedBoxGraphRenderer()
            return renderer.render_to_html(self, max_depth, width, height, title)
        except ImportError:
            raise ImportError(
                "SVG graph rendering requires the svg_nested_boxes module. "
                "Please ensure meteaudata is properly installed."
            )
    
    def show_graph_in_browser(self, max_depth: int = 4, width: int = 1200, 
                             height: int = 800, title: Optional[str] = None) -> str:
        """
        Render SVG graph and open in browser.
        
        Args:
            max_depth: Maximum depth to traverse in object hierarchy
            width: Graph width in pixels
            height: Graph height in pixels
            title: Page title (auto-generated if None)
            
        Returns:
            Path to the generated HTML file
        """
        try:
            from meteaudata.graph_display import open_meteaudata_graph_in_browser
            
            if title is None:
                title = f"Interactive {self.__class__.__name__} Hierarchy"
            
            return open_meteaudata_graph_in_browser(self, max_depth, width, height, title)
        except ImportError:
            raise ImportError(
                "Browser functionality requires additional modules. "
                "Please ensure meteaudata is properly installed."
            )
    
    def display(self, format: str = "html", depth: int = 2, 
            max_depth: int = 4, width: int = 1200, height: int = 800) -> None:
        """
        Display method with support for text, HTML, and interactive graph formats.
        """
        if format == "text":
            print(self._render_text(depth))
        elif format == "html":
            self._render_html(depth)
        elif format == "graph":
            if _is_notebook_environment():
                try:
                    from IPython.display import HTML, display
                    # Check if the imported objects are actually usable (not None)
                    if HTML is None or display is None:
                        raise ImportError("IPython.display components are None")
                    html_content = self.render_svg_graph(max_depth, width, height)
                    display(HTML(html_content))
                except (ImportError, AttributeError, TypeError):
                    print("Notebook environment detected but IPython not available.")
                    print("Use show_graph_in_browser() to open in browser instead.")
            else:
                self.show_graph_in_browser(max_depth, width, height)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'text', 'html', or 'graph'")
    
    # Convenience methods for quick access to different display modes
    def show_details(self, depth: int = 3) -> None:
        """
        Convenience method to show detailed HTML view.
        
        Args:
            depth: How deep to expand nested objects
        """
        self.display(format="html", depth=depth)
    
    def show_summary(self) -> None:
        """
        Convenience method to show a text summary.
        """
        self.display(format="text", depth=1)
    
    def show_graph(self, max_depth: int = 4, width: int = 1200, height: int = 800) -> None:
        """
        Convenience method to show the interactive graph.
        
        Args:
            max_depth: Maximum depth to traverse in object hierarchy
            width: Graph width in pixels  
            height: Graph height in pixels
        """
        self.display(format="graph", max_depth=max_depth, width=width, height=height)