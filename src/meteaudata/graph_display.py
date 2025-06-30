"""
Graph display implementation for meteaudata objects.
Creates interactive network visualizations showing object hierarchies.
"""

from meteaudata.displayable import _is_jupyter_environment
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC
import uuid
import json

# Color scheme for different object types
OBJECT_COLORS = {
    'Dataset': '#1f77b4',      # Blue
    'Signal': '#ff7f0e',       # Orange  
    'TimeSeries': '#2ca02c',   # Green
    'ProcessingStep': '#d62728', # Red
    'Parameters': '#9467bd',   # Purple
    'ParameterValue': '#8c564b', # Brown
    'FunctionInfo': '#e377c2', # Pink
    'DataProvenance': '#7f7f7f', # Gray
    'IndexMetadata': '#bcbd22', # Olive
}

OBJECT_SHAPES = {
    'Dataset': 'circle',
    'Signal': 'square',
    'TimeSeries': 'diamond',
    'ProcessingStep': 'triangle-up',
    'Parameters': 'hexagon',
    'ParameterValue': 'pentagon',
    'FunctionInfo': 'star',
    'DataProvenance': 'cross',
    'IndexMetadata': 'x',
}

class GraphNode:
    """Represents a node in the object hierarchy graph."""
    
    def __init__(self, obj: Any, node_id: str, parent_id: Optional[str] = None, 
                 relationship: str = "contains"):
        self.obj = obj
        self.node_id = node_id
        self.parent_id = parent_id
        self.relationship = relationship
        self.obj_type = obj.__class__.__name__
        self.identifier = obj._get_identifier() if hasattr(obj, '_get_identifier') else str(obj)
        
        # Generate HTML content for hover and click
        if hasattr(obj, '_build_html_content'):
            self.html_content = obj._build_html_content(depth=2)
        else:
            self.html_content = f"<div><h3>{self.obj_type}</h3><p>{self.identifier}</p></div>"

class GraphBuilder:
    """Builds graph representations of meteaudata object hierarchies."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (source, target, relationship)
        self.positions: Dict[str, Tuple[float, float]] = {}
        
    def build_graph(self, root_obj: Any, max_depth: int = 4) -> Dict[str, Any]:
        """Build a complete graph from a root meteaudata object."""
        self.nodes.clear()
        self.edges.clear()
        self.positions.clear()
        
        # Create root node
        root_id = str(uuid.uuid4())
        self._add_object_recursive(root_obj, root_id, None, max_depth)
        
        # Calculate layout
        self._calculate_layout()
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'positions': self.positions
        }
    
    def _add_object_recursive(self, obj: Any, node_id: str, parent_id: Optional[str], 
                            remaining_depth: int, relationship: str = "contains"):
        """Recursively add objects and their children to the graph."""
        if remaining_depth <= 0:
            return
            
        # Add current node
        self.nodes[node_id] = GraphNode(obj, node_id, parent_id, relationship)
        
        # Add edge from parent if exists
        if parent_id:
            self.edges.append((parent_id, node_id, relationship))
        
        # Get display attributes to find child objects
        if hasattr(obj, '_get_display_attributes'):
            attrs = obj._get_display_attributes()
            
            for attr_name, attr_value in attrs.items():
                if self._is_displayable_meteaudata_object(attr_value):
                    child_id = str(uuid.uuid4())
                    self._add_object_recursive(
                        attr_value, child_id, node_id, 
                        remaining_depth - 1, attr_name
                    )
                elif isinstance(attr_value, dict):
                    # Handle dictionaries that might contain meteaudata objects
                    for key, value in attr_value.items():
                        if self._is_displayable_meteaudata_object(value):
                            child_id = str(uuid.uuid4())
                            self._add_object_recursive(
                                value, child_id, node_id,
                                remaining_depth - 1, f"{attr_name}.{key}"
                            )
    
    def _is_displayable_meteaudata_object(self, obj: Any) -> bool:
        """Check if an object is a meteaudata object with display capabilities."""
        return hasattr(obj, '_get_display_attributes') and hasattr(obj, '_get_identifier')
    
    def _calculate_layout(self):
        """Calculate node positions using a hierarchical layout."""
        if not self.nodes:
            return
            
        # Create NetworkX graph for layout calculation
        G = nx.DiGraph()
        
        # Add nodes
        for node_id in self.nodes.keys():
            G.add_node(node_id)
        
        # Add edges
        for source, target, _ in self.edges:
            G.add_edge(source, target)
        
        # Use hierarchical layout if graph is connected, otherwise spring layout
        try:
            if nx.is_weakly_connected(G):
                # Hierarchical layout
                pos = self._hierarchical_layout(G)
            else:
                # Spring layout for disconnected components
                pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            # Fallback to random layout
            pos = {node_id: (np.random.random(), np.random.random()) 
                  for node_id in self.nodes.keys()}
        
        self.positions = pos
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout with root at top."""
        # Find root nodes (no incoming edges)
        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        if not root_nodes:
            return nx.spring_layout(G)
        
        pos = {}
        levels = {}
        
        # BFS to assign levels
        queue = [(root, 0) for root in root_nodes]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
                
            visited.add(node)
            levels[node] = level
            
            for child in G.successors(node):
                if child not in visited:
                    queue.append((child, level + 1))
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Assign positions
        max_level = max(level_groups.keys()) if level_groups else 0
        
        for level, nodes in level_groups.items():
            y = 1.0 - (level / max_level) if max_level > 0 else 0.5
            
            for i, node in enumerate(nodes):
                if len(nodes) > 1:
                    x = i / (len(nodes) - 1)
                else:
                    x = 0.5
                pos[node] = (x, y)
        
        return pos

def add_graph_display_to_displayable_base():
    """Add graph display method to DisplayableBase class."""
    
    def display_graph(self, max_depth: int = 4, show_labels: bool = True, 
                     width: int = 800, height: int = 600) -> go.Figure:
        """
        Create an interactive graph visualization of the object hierarchy.
        
        Args:
            max_depth: Maximum depth to traverse in the object hierarchy
            show_labels: Whether to show node labels
            width: Figure width in pixels
            height: Figure height in pixels
            
        Returns:
            Plotly Figure object with interactive graph
        """
        builder = GraphBuilder()
        graph_data = builder.build_graph(self, max_depth)
        
        return create_plotly_graph(graph_data, show_labels, width, height)
    
    # This would be added to DisplayableBase class
    return display_graph

def create_plotly_graph(graph_data: Dict[str, Any], show_labels: bool = True,
                       width: int = 800, height: int = 600) -> go.Figure:
    """Create a Plotly figure from graph data."""
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    positions = graph_data['positions']
    
    if not nodes:
        # Empty graph
        fig = go.Figure()
        fig.add_annotation(text="No objects to display", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for source_id, target_id, relationship in edges:
        if source_id in positions and target_id in positions:
            x0, y0 = positions[source_id]
            x1, y1 = positions[target_id]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{relationship}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Prepare node traces (group by object type for legend)
    node_traces = {}
    
    for node_id, node in nodes.items():
        if node_id not in positions:
            continue
            
        obj_type = node.obj_type
        x, y = positions[node_id]
        
        if obj_type not in node_traces:
            node_traces[obj_type] = {
                'x': [], 'y': [], 'text': [], 'html': [], 'ids': []
            }
        
        node_traces[obj_type]['x'].append(x)
        node_traces[obj_type]['y'].append(y)
        node_traces[obj_type]['text'].append(node.identifier)
        node_traces[obj_type]['html'].append(node.html_content)
        node_traces[obj_type]['ids'].append(node_id)
    
    # Create Plotly traces for each object type
    traces = [edge_trace]
    
    for obj_type, data in node_traces.items():
        color = OBJECT_COLORS.get(obj_type, '#999999')
        
        hover_text = [f"<b>{obj_type}</b><br>{text}<br><i>Click for details</i>" 
                     for text in data['text']]
        
        trace = go.Scatter(
            x=data['x'], y=data['y'],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=20,
                color=color,
                symbol=OBJECT_SHAPES.get(obj_type, 'circle'),
                line=dict(width=2, color='white')
            ),
            text=data['text'] if show_labels else None,
            textposition="bottom center",
            hovertemplate='<b>%{text}</b><br><i>Click for details</i><extra></extra>',
            name=obj_type,
            customdata=data['html']  # Store HTML for click events
        )
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=f"Object Hierarchy: {list(nodes.values())[0].obj_type}",
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ 
            dict(
                text="Drag to pan • Scroll to zoom • Click nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height,
        plot_bgcolor='white'
    )
    
    return fig

