"""
Display extensions for meteaudata objects.
Refactored to use inheritance with minimal code duplication.
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from meteaudata.display_utils import (
    _is_notebook_environment,
    _is_complex_object,
    _format_simple_value
)
from meteaudata.tree_builder import TreeBuilder

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
        """Build HTML content for the object using TreeBuilder."""
        # Use TreeBuilder to get the structure
        tree_builder = TreeBuilder()
        tree = tree_builder.build_tree(self, max_depth=depth)
        
        # Render the tree as HTML
        return self._render_tree_node_html(tree, depth)
    
    def _render_tree_node_html(self, node, remaining_depth: int) -> str:
        """Render a TreeNode as HTML."""
        lines = []
        
        # Header
        lines.append(f"<div class='meteaudata-header'>{node.node_type}</div>")
        
        # Simple attributes
        for attr_name, attr_value in node.attributes.items():
            lines.append(f"<div class='meteaudata-attr'><span class='meteaudata-attr-name'>{attr_name}:</span> <span class='meteaudata-attr-value'>{attr_value}</span></div>")
        
        # Child nodes
        if remaining_depth > 0 and node.children:
            for child in node.children:
                if child.is_collection():
                    # Render collection with collapsible details
                    nested_items = []
                    for item_child in child.children:
                        item_content = self._render_tree_node_html(item_child, remaining_depth - 1)
                        nested_items.append(f"<div class='meteaudata-nested'>{item_content}</div>")
                    
                    if nested_items:
                        nested_content = "\n".join(nested_items)
                        lines.append(f"""
                        <details class='meteaudata-collapsible'>
                            <summary class='meteaudata-summary'>{child.name}</summary>
                            <div class='meteaudata-nested'>{nested_content}</div>
                        </details>
                        """)
                else:
                    # Render single object with collapsible details
                    child_content = self._render_tree_node_html(child, remaining_depth - 1)
                    lines.append(f"""
                    <details class='meteaudata-collapsible'>
                        <summary class='meteaudata-summary'>{child.relationship}: {child.node_type}</summary>
                        <div class='meteaudata-nested'>{child_content}</div>
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