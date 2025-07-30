"""
SVG-based interactive nested box graph implementation for meteaudata objects.
Uses D3.js with SVG rendering for better performance and interaction capabilities.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
import uuid
from meteaudata.tree_builder import TreeBuilder, TreeNode


class SVGGraphNode:
    """Node representation optimized for SVG rendering."""
    
    def __init__(self, obj: Any, node_id: str, parent_id: Optional[str] = None, 
                 relationship: str = "contains"):
        self.obj = obj
        self.node_id = node_id
        self.parent_id = parent_id
        self.relationship = relationship
        self.obj_type = obj.__class__.__name__
        self.identifier = obj._get_identifier() if hasattr(obj, '_get_identifier') else str(obj)
        
        # Extract key attributes for details panel
        self.attributes = self._extract_attributes()
    
    def _format_parameters(self, params_obj: Any) -> str:
        """Format Parameters object for display in details panel."""
        if not hasattr(params_obj, '_get_display_attributes'):
            return str(params_obj)
        
        # Use the Parameters object's own display method which knows how to access stored values
        display_attrs = params_obj._get_display_attributes()
        
        # Extract the actual parameter values, excluding metadata fields
        param_values = {}
        metadata_fields = {'parameter_count'}
        
        for key, value in display_attrs.items():
            # Skip metadata fields and the param_ prefixed fields from the Parameters class
            if key not in metadata_fields:
                # For param_ prefixed fields, we want to show them but clean up the key
                if key.startswith('param_'):
                    clean_key = key[6:]  # Remove 'param_' prefix
                    param_values[clean_key] = value
                else:
                    param_values[key] = value
        
        if not param_values:
            return "No parameters"
        
        # Format as a readable string
        formatted_params = []
        for key, value in param_values.items():
            # Handle special value types
            if isinstance(value, dict) and "__numpy_array__" in value:
                formatted_value = f"array(shape={value['shape']}, dtype={value['dtype']})"
            elif isinstance(value, (list, tuple)) and len(value) > 3:
                formatted_value = f"{type(value).__name__}[{len(value)} items]"
            elif isinstance(value, dict) and len(value) > 3:
                formatted_value = f"dict[{len(value)} items]"
            else:
                formatted_value = str(value)
                if len(formatted_value) > 50:
                    formatted_value = formatted_value[:47] + "..."
            
            formatted_params.append(f"{key}: {formatted_value}")
        
        # Return concise summary for small number of params, or count for many
        if len(formatted_params) <= 3:
            return " | ".join(formatted_params)
        else:
            return f"{len(formatted_params)} parameters: " + ", ".join(formatted_params[:2]) + f", ... and {len(formatted_params)-2} more"

    
    def _extract_attributes(self) -> Dict[str, Any]:
        """Extract and format attributes for the details panel."""
        if not hasattr(self.obj, '_get_display_attributes'):
            return {'Type': self.obj_type, 'Identifier': self.identifier}
        
        attrs = self.obj._get_display_attributes()
        formatted_attrs = {}
        
        # Priority attributes to show first
        priority_keys = [
            'name', 'series_name', 'signal_name', 'identifier',
            'description', 'units', 'length', 'count', 'size',
            'created_on', 'last_updated', 'date_range',
            'values_dtype', 'frequency', 'time_zone'
        ]
        
        # Add priority attributes first
        for key in priority_keys:
            if key in attrs:
                formatted_attrs[self._format_key(key)] = self._format_value(attrs[key])
        
        # Add ALL other attributes for the details panel (don't exclude Parameters/FunctionInfo/DataProvenance here)
        structural_keys = ['signals', 'time_series', 'processing_steps', 'index_metadata']
        for key, value in attrs.items():
            if key not in priority_keys and key not in structural_keys:
                # For Parameters objects, show full parameter details
                if hasattr(value, '__class__') and 'Parameters' in value.__class__.__name__:
                    formatted_attrs[self._format_key(key)] = self._format_parameters(value)
                # For FunctionInfo objects, show full function details  
                elif hasattr(value, '__class__') and 'FunctionInfo' in value.__class__.__name__:
                    formatted_attrs[self._format_key(key)] = self._format_function_info(value)
                # For DataProvenance objects, show full provenance details
                elif hasattr(value, '__class__') and 'DataProvenance' in value.__class__.__name__:
                    formatted_attrs[self._format_key(key)] = self._format_data_provenance(value)
                elif not self._is_complex_object(value):
                    formatted_attrs[self._format_key(key)] = self._format_value(value)
        
        return formatted_attrs

    def _format_function_info(self, func_info: Any) -> str:
        """Format FunctionInfo object for display in details panel."""
        if not hasattr(func_info, '_get_display_attributes'):
            return str(func_info)
        
        display_attrs = func_info._get_display_attributes()
        
        # Format function info details
        formatted_parts = []
        
        # Show key function information
        for key, value in display_attrs.items():
            if key in ['name', 'version', 'author', 'reference']:
                formatted_parts.append(f"{key}: {value}")
            elif key == 'has_source_code' and value:
                formatted_parts.append("Source code: Available")
            elif key == 'source_code_lines':
                formatted_parts.append(f"Code lines: {value}")
        
        return " | ".join(formatted_parts) if formatted_parts else "Function information available"
    
    def _format_key(self, key: str) -> str:
        """Format attribute key for display."""
        return key.replace('_', ' ').title()
    
    def _format_value(self, value: Any) -> str:
        """Format attribute value for display."""
        if value is None:
            return "None"
        
        if hasattr(value, 'strftime'):  # datetime
            return value.strftime("%Y-%m-%d %H:%M:%S")
        
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return f"Empty {type(value).__name__}"
            elif len(value) <= 3:
                return f"[{', '.join(str(v) for v in value)}]"
            else:
                return f"{type(value).__name__} with {len(value)} items"
        
        if isinstance(value, dict):
            if len(value) == 0:
                return "Empty dictionary"
            else:
                return f"Dictionary with {len(value)} items"
        
        # Convert to string and truncate if too long
        str_val = str(value)
        if len(str_val) > 100:
            return str_val[:97] + "..."
        
        return str_val
    
    def _format_data_provenance(self, provenance: Any) -> str:
        """Format DataProvenance object for display in details panel."""
        if not hasattr(provenance, '_get_display_attributes'):
            return str(provenance)
        
        display_attrs = provenance._get_display_attributes()
        
        # Format provenance details
        formatted_parts = []
        
        # Show key provenance information in logical order
        key_order = ['source_repository', 'project', 'location', 'equipment', 'parameter', 'purpose', 'metadata_id']
        
        for key in key_order:
            if key in display_attrs and display_attrs[key] is not None:
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {display_attrs[key]}")
        
        # Add any other attributes not in the key order
        for key, value in display_attrs.items():
            if key not in key_order and value is not None:
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return " | ".join(formatted_parts) if formatted_parts else "Data provenance information available"
    
    def _is_complex_object(self, obj: Any) -> bool:
        """Check if object is too complex for simple display."""
        return (hasattr(obj, '_get_display_attributes') or 
                isinstance(obj, dict) and len(obj) > 5 or
                isinstance(obj, (list, tuple)) and len(obj) > 10)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.node_id,
            'type': self.obj_type,
            'name': self._get_display_name(),
            'identifier': self.identifier,
            'attributes': self.attributes,
            'relationship': self.relationship
        }
    
    def _get_display_name(self) -> str:
        """Get a clean display name for the box label."""
        # For Container objects (collections), use the CollectionContainer display name
        if self.obj_type == 'Container' and hasattr(self.obj, '_get_display_name'):
            return self.obj._get_display_name()
        
        # For TimeSeries objects, show "TimeSeries: <name>" format
        if self.obj_type == 'TimeSeries':
            series_name = getattr(self.obj.series, 'name', 'unnamed')
            if '_' in series_name:
                # Split on underscore and take the part after signal name
                parts = series_name.split('_', 1)  # Split into max 2 parts
                if len(parts) > 1:
                    name = parts[1]  # Take everything after first underscore
                    # Remove numbering if present
                    if '#' in name:
                        name = name.split('#')[0]
                    return f"TimeSeries: {name}"
            
            # Fallback to full name if no underscore pattern
            name = series_name
            if '#' in name:
                name = name.split('#')[0]
            return f"TimeSeries: {name}"
        
        # For DataProvenance, extract clean parameter name
        if self.obj_type == 'DataProvenance':
            if hasattr(self.obj, 'parameter') and self.obj.parameter:
                return self.obj.parameter
            # Fallback to extracting from identifier
            if '=' in self.identifier:
                name = self.identifier.split('=', 1)[1].strip().strip("'\"")
                if name:
                    return name
        
        # For other object types, use existing logic
        if hasattr(self.obj, 'name') and self.obj.name:
            name = str(self.obj.name)
        elif hasattr(self.obj, 'series') and hasattr(self.obj.series, 'name') and self.obj.series.name:
            name = str(self.obj.series.name)
        else:
            name = self.identifier
        
        # Clean up the name
        if '#' in name:
            name = name.split('#')[0]  # Remove numbering
        if '_' in name and len(name.split('_')) > 1:
            parts = name.split('_')
            if len(parts) == 2:
                name = parts[1]  # Remove signal prefix for time series
        
        return name


class SVGGraphBuilder:
    """Builds graph data optimized for SVG rendering using shared TreeBuilder."""
    
    def __init__(self):
        self.nodes: Dict[str, SVGGraphNode] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.tree_builder = TreeBuilder()
    
    def build_graph(self, root_obj: Any, max_depth: int = 4) -> Dict[str, Any]:
        """Build graph data for SVG rendering using TreeBuilder."""
        self.nodes.clear()
        self.edges.clear()
        
        # Use TreeBuilder to get the object structure
        tree = self.tree_builder.build_tree(root_obj, max_depth)
        
        # Convert TreeNode structure to SVG graph structure
        root_id = str(uuid.uuid4())
        self._convert_tree_to_graph(tree, root_id, None)
        
        # Build hierarchy structure
        hierarchy = self._build_hierarchy_structure()
        
        return {
            'hierarchy': hierarchy,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'layout_type': 'svg_nested_boxes'
        }
    
    def _convert_tree_to_graph(self, tree_node: TreeNode, node_id: str, parent_id: Optional[str]):
        """Convert TreeNode structure to SVG graph nodes and edges."""
        # Create SVG node from TreeNode - but override the type for collection nodes
        svg_node = SVGGraphNode(tree_node.obj, node_id, parent_id, tree_node.relationship)
        
        # For collection nodes, use 'Container' as the type (for backwards compatibility with tests)
        if tree_node.node_type == 'collection':
            svg_node.obj_type = 'Container'
        
        self.nodes[node_id] = svg_node
        
        # Add edge to parent if exists
        if parent_id:
            self.edges.append((parent_id, node_id, tree_node.relationship))
        
        # Process children
        for child in tree_node.children:
            child_id = str(uuid.uuid4())
            self._convert_tree_to_graph(child, child_id, node_id)
    

    
    def _build_hierarchy_structure(self) -> Dict[str, Any]:
        """Build nested hierarchy structure for the frontend."""
        # Find root nodes (no parents)
        children_map = {}
        parent_map = {}
        
        for source_id, target_id, relationship in self.edges:
            if source_id not in children_map:
                children_map[source_id] = []
            children_map[source_id].append(target_id)
            parent_map[target_id] = source_id
        
        roots = [node_id for node_id in self.nodes.keys() if node_id not in parent_map]
        
        def build_tree(node_id):
            node = self.nodes[node_id]
            node_dict = node.to_dict()
            
            # Add children
            if node_id in children_map:
                node_dict['children'] = [build_tree(child_id) for child_id in children_map[node_id]]
            else:
                node_dict['children'] = []
            
            return node_dict
        
        # Return single root or create virtual root for multiple roots
        if len(roots) == 1:
            return build_tree(roots[0])
        else:
            return {
                'id': 'virtual_root',
                'type': 'Container',
                'name': 'Multiple Objects',
                'identifier': 'virtual_root',
                'attributes': {'Objects': len(roots)},
                'children': [build_tree(root_id) for root_id in roots]
            }


class SVGNestedBoxGraphRenderer:
    """Renders meteaudata objects as interactive SVG nested box graphs."""
    
    def __init__(self):
        self.builder = SVGGraphBuilder()
    
    def render_to_html(self, root_obj: Any, max_depth: int = 4, 
                  width: int = 1200, height: int = 800,
                  title: str = "Interactive Object Hierarchy") -> str:
        """
        Render object hierarchy as interactive HTML with SVG graph.
        """
        # Build graph data
        graph_data = self.builder.build_graph(root_obj, max_depth)
        
        # Get the base HTML template
        html_template = self._get_html_template()
        
        # Custom JSON serializer for complex objects
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # custom objects
                return str(obj)
            else:
                return str(obj)
        
        # Ensure the JSON is properly serialized
        graph_data_json = json.dumps(
            graph_data["hierarchy"], 
            indent=2, 
            default=json_serializer,
            ensure_ascii=False
        )
        
        # Escape quotes in title for JavaScript
        escaped_title = title.replace('"', '\\"')
        
        # Inject the data and configuration
        injection_code = f'''
const graphData = {graph_data_json};
const graphConfig = {{
    width: {width},
    height: {height},
    title: "{escaped_title}"
}};

// Initialize graph with real data
graph.loadData(graphData);
document.title = "{escaped_title}";
'''
        
        # Replace the injection placeholder
        html_content = html_template.replace('// INJECT_DATA_HERE', injection_code)
        
        return html_content
    
    def save_to_file(self, root_obj: Any, filepath: str, max_depth: int = 4,
                    width: int = 1200, height: int = 800,
                    title: str = "Interactive Object Hierarchy") -> None:
        """Save rendered graph to HTML file."""
        html_content = self.render_to_html(root_obj, max_depth, width, height, title)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_html_template(self) -> str:
        """Load HTML template from package resources."""
        try:
            # Try modern importlib.resources (Python 3.9+)
            try:
                from importlib.resources import files
                template_file = files('meteaudata.templates') / 'svg_graph_template.html'
                return template_file.read_text(encoding='utf-8')
            except (ImportError, AttributeError):
                # Fallback for older Python versions
                from importlib.resources import read_text
                return read_text('meteaudata.templates', 'svg_graph_template.html')
        except (ImportError, FileNotFoundError, ModuleNotFoundError):
            # Fallback: try to find template relative to this file
            from pathlib import Path
            template_path = Path(__file__).parent / 'templates' / 'svg_graph_template.html'
            try:
                return template_path.read_text(encoding='utf-8')
            except FileNotFoundError:
                raise FileNotFoundError(
                    "SVG graph template not found. Please ensure meteaudata is properly installed "
                    "with template files. Expected location: meteaudata/templates/svg_graph_template.html"
                )


# Clean convenience functions for external usage
def render_meteaudata_graph_html(obj: Any, max_depth: int = 4, width: int = 1200, 
                                 height: int = 800, title: Optional[str] = None) -> str:
    """
    Render any meteaudata object as HTML string with interactive SVG graph.
    
    Args:
        obj: Any meteaudata object (Dataset, Signal, TimeSeries, etc.)
        max_depth: Maximum depth to traverse in object hierarchy
        width: Graph width in pixels
        height: Graph height in pixels
        title: Page title (auto-generated if None)
        
    Returns:
        HTML string with embedded interactive graph
    """
    renderer = SVGNestedBoxGraphRenderer()
    
    if title is None:
        title = f"Interactive {obj.__class__.__name__} Hierarchy"
    
    return renderer.render_to_html(obj, max_depth, width, height, title)


def save_meteaudata_graph(obj, filepath: str, max_depth: int = 4, 
                         width: int = 1200, height: int = 800, title: str = None) -> None:
    """
    Save meteaudata object graph to HTML file.
    
    Args:
        obj: Any meteaudata object (Dataset, Signal, TimeSeries, etc.)
        filepath: Path where to save the HTML file
        max_depth: Maximum depth to traverse in object hierarchy
        width: Graph width in pixels
        height: Graph height in pixels
        title: Page title (auto-generated if None)
    """
    renderer = SVGNestedBoxGraphRenderer()
    
    if title is None:
        title = f"Interactive {obj.__class__.__name__} Hierarchy"
    
    renderer.save_to_file(obj, filepath, max_depth, width, height, title)


def open_meteaudata_graph_in_browser(obj, max_depth: int = 4, width: int = 1200, 
                                    height: int = 800, title: str = None) -> str:
    """
    Render meteaudata object graph and open in browser.
    
    Args:
        obj: Any meteaudata object (Dataset, Signal, TimeSeries, etc.)
        max_depth: Maximum depth to traverse in object hierarchy
        width: Graph width in pixels
        height: Graph height in pixels
        title: Page title (auto-generated if None)
        
    Returns:
        Path to the generated HTML file
    """
    import tempfile
    import webbrowser
    
    if title is None:
        title = f"Interactive {obj.__class__.__name__} Hierarchy"
    
    # Generate HTML
    html_content = render_meteaudata_graph_html(obj, max_depth, width, height, title)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
    temp_file.write(html_content)
    temp_file.close()
    
    # Open in browser
    webbrowser.open(f'file://{temp_file.name}')
    
    return temp_file.name