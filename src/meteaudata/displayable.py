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
        """Render HTML representation."""
        try:
            from IPython.display import HTML, display
            html_content = f"{HTML_STYLE}<div class='meteaudata-display'>{self._build_html_content(depth)}</div>"
            display(HTML(html_content))
        except ImportError:
            print(self._render_text(depth))
    
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
    
    def render_svg_graph(self, max_depth: int = 4, width: int = 1200, 
                        height: int = 800, title: str = None) -> str:
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
                             height: int = 800, title: str = None) -> str:
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
        
        Args:
            format: Display format - 'text', 'html', or 'graph' 
            depth: Depth for text/html displays
            max_depth: Maximum depth for graph traversal
            width: Graph width in pixels
            height: Graph height in pixels
        """
        if format == "text":
            print(self._render_text(depth))
        elif format == "html":
            self._render_html(depth)
        elif format == "graph":
            # For notebooks, display the HTML directly
            if _is_notebook_environment():
                try:
                    from IPython.display import HTML, display
                    html_content = self.render_svg_graph(max_depth, width, height)
                    display(HTML(html_content))
                except ImportError:
                    print("Notebook environment detected but IPython not available.")
                    print("Use show_graph_in_browser() to open in browser instead.")
            else:
                # For non-notebook environments, open in browser
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