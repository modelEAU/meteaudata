"""
Shared tree builder for meteaudata objects.

This module provides a generic tree structure that represents the hierarchical
relationships within meteaudata objects. Both HTML display and SVG graph 
visualization use this same tree structure.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class CollectionContainer:
    """A container object for collections that provides displayable interface."""
    
    def __init__(self, items: List[Any], collection_type: str, collection_name: str):
        self.items = items
        self.collection_type = collection_type
        self.collection_name = collection_name
    
    def _get_identifier(self) -> str:
        """Get identifier for this collection container."""
        return f"{self.collection_type}"
    
    def _get_display_name(self) -> str:
        """Get display name for this collection container."""
        # Map collection types to better display names
        display_names = {
            'signals': 'Signals',
            'time_series': 'Time Series', 
            'processing_steps': 'Processing Steps'
        }
        return display_names.get(self.collection_type, self.collection_type.replace('_', ' ').title())
    
    def _get_display_attributes(self) -> Dict[str, Any]:
        """Get display attributes for this collection container."""
        return {
            'Container Type': self.collection_type.replace('_', ' ').title(),
            'Item Count': len(self.items),
            'Items': ', '.join([getattr(item, 'name', str(item)) for item in self.items[:3]]) + 
                    (f' and {len(self.items)-3} more' if len(self.items) > 3 else '')
        }


@dataclass
class TreeNode:
    """A node in the object tree structure."""
    obj: Any                                    # The actual object
    name: str                                   # Display name for this node
    node_type: str                             # Type of node (object class name or 'collection')
    attributes: Dict[str, Any]                 # Simple attributes for display
    children: List['TreeNode']                 # Child nodes
    relationship: str = "contains"             # Relationship to parent
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_collection(self) -> bool:
        """Check if this node represents a collection."""
        return self.node_type == 'collection'


class TreeBuilder:
    """
    Builds a hierarchical tree structure from meteaudata objects.
    
    This class recursively traverses objects using their _get_display_attributes()
    methods to build a unified tree structure. It doesn't make decisions about 
    what should be displayed - it faithfully represents whatever structure the
    _get_display_attributes() methods define.
    
    The tree can be used by both HTML display and SVG graph visualization.
    """
    
    def build_tree(self, root_obj: Any, max_depth: int = 4) -> TreeNode:
        """
        Build a tree structure from a meteaudata object.
        
        Args:
            root_obj: The root object to build tree from
            max_depth: Maximum depth to traverse
            
        Returns:
            TreeNode representing the root of the tree
        """
        return self._build_node_recursive(
            obj=root_obj,
            name=self._get_display_name(root_obj),
            remaining_depth=max_depth
        )
    
    def _build_node_recursive(self, obj: Any, name: str, remaining_depth: int) -> TreeNode:
        """
        Recursively build a tree node and its children.
        
        Args:
            obj: The object to build a node for
            name: Display name for this node
            remaining_depth: Remaining traversal depth
            
        Returns:
            TreeNode with children populated
        """
        # Create the node
        node = TreeNode(
            obj=obj,
            name=name,
            node_type=obj.__class__.__name__,
            attributes=self._extract_simple_attributes(obj),
            children=[]
        )
        
        # Stop if we've reached max depth
        if remaining_depth <= 0:
            return node
        
        # Get displayable attributes from the object
        if not hasattr(obj, '_get_display_attributes'):
            return node
            
        attrs = obj._get_display_attributes()
        
        # Group attributes by prefix patterns (for backwards compatibility with old container system)
        grouped_attrs = self._group_attributes_by_pattern(attrs)
        
        # Process grouped attributes
        for group_name, group_attrs in grouped_attrs.items():
            if group_name in ['signals', 'time_series'] and group_attrs:
                # Always create containers for signals and time_series (for backwards compatibility)
                collection_children = []
                for attr_name, attr_value in group_attrs.items():
                    child_nodes = self._process_attribute(attr_name, attr_value, remaining_depth - 1)
                    collection_children.extend(child_nodes)
                
                if collection_children:
                    # Create collection container object
                    container_obj = CollectionContainer(
                        items=list(group_attrs.values()),
                        collection_type=group_name,
                        collection_name=f"{group_name} ({len(collection_children)} items)"
                    )
                    
                    # Create collection node
                    collection_node = TreeNode(
                        obj=container_obj,
                        name=f"{group_name} ({len(collection_children)} items)",
                        node_type='collection',
                        attributes={'count': len(collection_children), 'type': group_name},
                        children=collection_children,
                        relationship=group_name
                    )
                    # Update children relationships to point to collection
                    for child in collection_children:
                        child.relationship = child.relationship.split('_', 1)[-1] if '_' in child.relationship else child.relationship
                    
                    node.children.append(collection_node)
            else:
                # Process other attributes normally (don't create containers)
                for attr_name, attr_value in group_attrs.items():
                    child_nodes = self._process_attribute(attr_name, attr_value, remaining_depth - 1)
                    node.children.extend(child_nodes)
        
        return node
    
    def _group_attributes_by_pattern(self, attrs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Group attributes by prefix patterns for container creation.
        
        Groups signal_* attributes under 'signals' and timeseries_* under 'time_series'.
        All other attributes go under 'other'.
        
        Args:
            attrs: Dictionary of attributes from _get_display_attributes()
            
        Returns:
            Dictionary mapping group names to attribute dictionaries
        """
        groups = {'signals': {}, 'time_series': {}, 'other': {}}
        
        for attr_name, attr_value in attrs.items():
            if attr_name.startswith('signal_'):
                groups['signals'][attr_name] = attr_value
            elif attr_name.startswith('timeseries_'):
                groups['time_series'][attr_name] = attr_value
            else:
                groups['other'][attr_name] = attr_value
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _process_attribute(self, attr_name: str, attr_value: Any, remaining_depth: int) -> List[TreeNode]:
        """
        Process a single attribute and return any child nodes it generates.
        
        Args:
            attr_name: Name of the attribute
            attr_value: Value of the attribute
            remaining_depth: Remaining traversal depth
            
        Returns:
            List of TreeNode objects (empty if no children)
        """
        if remaining_depth <= 0:
            return []
        
        # Handle lists that might contain displayable objects
        if isinstance(attr_value, (list, tuple)):
            return self._process_collection(attr_name, attr_value, remaining_depth)
        
        # Handle dictionaries that might contain displayable objects
        elif isinstance(attr_value, dict):
            return self._process_dict(attr_name, attr_value, remaining_depth)
        
        # Handle single displayable objects
        elif self._is_displayable_object(attr_value):
            child_name = self._get_display_name(attr_value)
            child_node = self._build_node_recursive(attr_value, child_name, remaining_depth)
            child_node.relationship = attr_name
            return [child_node]
        
        # Non-displayable attributes are already captured in _extract_simple_attributes
        return []
    
    def _process_collection(self, attr_name: str, collection: Union[List, tuple], remaining_depth: int) -> List[TreeNode]:
        """
        Process a collection that might contain displayable objects.
        
        Args:
            attr_name: Name of the collection attribute
            collection: The list or tuple to process
            remaining_depth: Remaining traversal depth
            
        Returns:
            List of TreeNode objects
        """
        if not collection:
            return []
        
        # Check if any items in the collection are displayable
        displayable_items = [item for item in collection if self._is_displayable_object(item)]
        
        if not displayable_items:
            return []
        
        # If we have displayable items, create a collection node
        # For known collection types, create a CollectionContainer to get proper display names
        if attr_name in ['processing_steps', 'signals', 'time_series']:
            container_obj = CollectionContainer(
                items=displayable_items,
                collection_type=attr_name,
                collection_name=f"{attr_name} ({len(displayable_items)} items)"
            )
            display_name = container_obj._get_display_name()
        else:
            container_obj = collection
            display_name = f"{attr_name} ({len(displayable_items)} items)"
        
        collection_node = TreeNode(
            obj=container_obj,
            name=display_name,
            node_type='collection',
            attributes={'count': len(displayable_items), 'type': attr_name},
            children=[],
            relationship=attr_name
        )
        
        # Add individual items as children
        for i, item in enumerate(displayable_items):
            item_name = self._get_display_name(item)
            item_node = self._build_node_recursive(item, item_name, remaining_depth)
            item_node.relationship = f"item_{i}"
            collection_node.children.append(item_node)
        
        return [collection_node]
    
    def _process_dict(self, attr_name: str, dict_value: Dict[str, Any], remaining_depth: int) -> List[TreeNode]:
        """
        Process a dictionary that might contain displayable objects.
        
        Args:
            attr_name: Name of the dictionary attribute
            dict_value: The dictionary to process
            remaining_depth: Remaining traversal depth
            
        Returns:
            List of TreeNode objects
        """
        child_nodes = []
        
        for key, value in dict_value.items():
            if self._is_displayable_object(value):
                child_name = self._get_display_name(value)
                child_node = self._build_node_recursive(value, child_name, remaining_depth)
                child_node.relationship = key
                child_nodes.append(child_node)
        
        return child_nodes
    
    def _is_displayable_object(self, obj: Any) -> bool:
        """
        Check if an object is displayable (has the required methods).
        
        Args:
            obj: Object to check
            
        Returns:
            True if object is displayable
        """
        return (hasattr(obj, '_get_display_attributes') and 
                hasattr(obj, '_get_identifier'))
    
    def _get_display_name(self, obj: Any) -> str:
        """
        Get a display name for an object.
        
        Args:
            obj: Object to get name for
            
        Returns:
            Display name string
        """
        # Try to get identifier first
        if hasattr(obj, '_get_identifier'):
            try:
                identifier = obj._get_identifier()
                # Clean up common identifier patterns
                if '=' in identifier:
                    # Extract value after = and remove quotes
                    name = identifier.split('=', 1)[1].strip().strip("'\"")
                    if name:
                        return name
            except:
                pass
        
        # Fallback to common name attributes
        for attr in ['name', 'series_name']:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if value:
                    return str(value)
        
        # Final fallback to class name
        return obj.__class__.__name__
    
    def _extract_simple_attributes(self, obj: Any) -> Dict[str, Any]:
        """
        Extract simple (non-object) attributes for display.
        
        Args:
            obj: Object to extract attributes from
            
        Returns:
            Dictionary of simple attributes
        """
        if not hasattr(obj, '_get_display_attributes'):
            return {'type': obj.__class__.__name__}
        
        attrs = obj._get_display_attributes()
        simple_attrs = {}
        
        for key, value in attrs.items():
            # Only include simple values, not complex objects or collections
            if not self._is_complex_value(value):
                simple_attrs[key] = self._format_simple_value(value)
        
        return simple_attrs
    
    def _is_complex_value(self, value: Any) -> bool:
        """
        Check if a value is too complex for simple display.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is complex
        """
        return (self._is_displayable_object(value) or
                isinstance(value, (list, tuple)) and any(self._is_displayable_object(item) for item in value) or
                isinstance(value, dict) and any(self._is_displayable_object(v) for v in value.values()))
    
    def _format_simple_value(self, value: Any) -> str:
        """
        Format a simple value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string representation
        """
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
                return f"{type(value).__name__}[{len(value)} items]"
        
        if isinstance(value, dict):
            if len(value) == 0:
                return "Empty dictionary"
            else:
                return f"Dictionary[{len(value)} items]"
        
        # Convert to string and truncate if too long
        str_val = str(value)
        if len(str_val) > 100:
            return str_val[:97] + "..."
        
        return str_val