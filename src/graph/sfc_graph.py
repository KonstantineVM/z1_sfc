#!/usr/bin/env python3
"""
PLACEMENT: src/graph/sfc_graph.py

Core SFC Graph classes and data structures.
This is the foundation for the graph-based constraint system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the SFC graph."""
    SECTOR = "sector"
    INSTRUMENT = "instrument"
    SERIES = "series"
    BILATERAL = "bilateral"
    AGGREGATE = "aggregate"
    IDENTITY = "identity"


class EdgeType(Enum):
    """Types of edges in the SFC graph."""
    HOLDS = "holds"                      # Sector holds instrument
    ISSUES = "issues"                    # Sector issues instrument
    STOCK_FLOW = "stock_flow"           # FU→FL relationship
    AGGREGATES_TO = "aggregates_to"     # Component→Total
    REPRESENTS = "represents"           # Series represents position
    LAG = "lag"                         # Temporal relationship
    BILATERAL_ASSET = "bilateral_asset"  # Bilateral asset position
    BILATERAL_LIABILITY = "bilateral_liability"  # Bilateral liability


@dataclass
class SFCNode:
    """
    Represents a node in the SFC graph.
    
    Attributes:
    -----------
    id : str
        Unique identifier for the node
    node_type : NodeType
        Type of node (sector, instrument, etc.)
    metadata : Dict
        Additional properties of the node
    """
    id: str
    node_type: NodeType
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, SFCNode):
            return self.id == other.id
        return self.id == other
    
    def __repr__(self):
        return f"SFCNode({self.id}, {self.node_type.value})"


@dataclass
class SFCEdge:
    """
    Represents an edge in the SFC graph.
    
    Attributes:
    -----------
    source : str
        Source node ID
    target : str
        Target node ID
    edge_type : EdgeType
        Type of relationship
    weight : float
        Strength/amount of relationship
    metadata : Dict
        Additional properties
    """
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"SFCEdge({self.source}→{self.target}, {self.edge_type.value}, w={self.weight})"


class SFCGraph:
    """
    Main SFC Graph structure for constraint generation.
    
    This class manages the complete financial network including:
    - Sectors (who holds/issues)
    - Instruments (what is held/issued)
    - Series (Z1 time series)
    - Bilateral positions (FWTW data)
    """
    
    def __init__(self):
        """Initialize empty SFC graph."""
        self.G = nx.MultiDiGraph()  # Allows multiple edges between nodes
        self._node_index = {}  # Fast lookup by ID
        self._series_index = {}  # Map series codes to nodes
        self._bilateral_index = {}  # Map (holder, issuer, instrument) to nodes
        
    def add_node(self, node: SFCNode) -> None:
        """
        Add a node to the graph.
        
        Parameters:
        -----------
        node : SFCNode
            Node to add
        """
        self.G.add_node(
            node.id,
            node_type=node.node_type.value,
            **node.metadata
        )
        self._node_index[node.id] = node
        
        # Update specialized indices
        if node.node_type == NodeType.SERIES:
            series_code = node.metadata.get('code')
            if series_code:
                self._series_index[series_code] = node.id
        elif node.node_type == NodeType.BILATERAL:
            key = (
                node.metadata.get('holder'),
                node.metadata.get('issuer'),
                node.metadata.get('instrument')
            )
            if all(key):
                self._bilateral_index[key] = node.id
    
    def add_edge(self, edge: SFCEdge) -> None:
        """
        Add an edge to the graph.
        
        Parameters:
        -----------
        edge : SFCEdge
            Edge to add
        """
        self.G.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            **edge.metadata
        )
    
    def get_node(self, node_id: str) -> Optional[SFCNode]:
        """Get node by ID."""
        return self._node_index.get(node_id)
    
    def get_series_node(self, series_code: str) -> Optional[str]:
        """Get node ID for a series code."""
        return self._series_index.get(series_code)
    
    def get_bilateral_node(self, holder: str, issuer: str, 
                          instrument: str) -> Optional[str]:
        """Get node ID for a bilateral position."""
        return self._bilateral_index.get((holder, issuer, instrument))
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Get all node IDs of a specific type."""
        return [
            node_id for node_id, data in self.G.nodes(data=True)
            if data.get('node_type') == node_type.value
        ]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Tuple[str, str, Dict]]:
        """Get all edges of a specific type."""
        edges = []
        for u, v, data in self.G.edges(data=True):
            if data.get('edge_type') == edge_type.value:
                edges.append((u, v, data))
        return edges
    
    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None,
                      direction: str = 'out') -> List[str]:
        """
        Get neighbors of a node.
        
        Parameters:
        -----------
        node_id : str
            Node to get neighbors for
        edge_type : EdgeType, optional
            Filter by edge type
        direction : str
            'out' for successors, 'in' for predecessors, 'both' for all
        
        Returns:
        --------
        List[str]
            List of neighbor node IDs
        """
        neighbors = []
        
        if direction in ['out', 'both']:
            for neighbor in self.G.successors(node_id):
                if edge_type is None:
                    neighbors.append(neighbor)
                else:
                    for _, _, data in self.G.edges(node_id, neighbor, data=True):
                        if data.get('edge_type') == edge_type.value:
                            neighbors.append(neighbor)
                            break
        
        if direction in ['in', 'both']:
            for neighbor in self.G.predecessors(node_id):
                if edge_type is None:
                    neighbors.append(neighbor)
                else:
                    for _, _, data in self.G.edges(neighbor, node_id, data=True):
                        if data.get('edge_type') == edge_type.value:
                            neighbors.append(neighbor)
                            break
        
        return list(set(neighbors))  # Remove duplicates
    
    def find_stock_flow_pairs(self) -> List[Dict[str, str]]:
        """
        Find all stock-flow relationships in the graph.
        
        Returns:
        --------
        List[Dict[str, str]]
            List of stock-flow pairs with FL, FU, FR, FV series
        """
        pairs = []
        
        # Find all FL series nodes
        for node_id in self.get_nodes_by_type(NodeType.SERIES):
            node = self.get_node(node_id)
            if node.metadata.get('prefix') == 'FL':
                fl_code = node.metadata.get('code')
                sector = node.metadata.get('sector')
                instrument = node.metadata.get('instrument')
                
                # Look for corresponding flow series
                pair = {'FL': fl_code}
                
                for prefix in ['FU', 'FR', 'FV']:
                    flow_code = f"{prefix}{sector}{instrument}005.Q"
                    if self.get_series_node(flow_code):
                        pair[prefix] = flow_code
                
                if len(pair) > 1:  # Has at least one flow series
                    pairs.append(pair)
        
        return pairs
    
    def find_aggregation_relationships(self) -> List[Dict[str, Any]]:
        """
        Find all aggregation relationships (parent = sum of children).
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of aggregation relationships
        """
        aggregations = []
        
        for parent_id in self.get_nodes_by_type(NodeType.AGGREGATE):
            children = self.get_neighbors(parent_id, EdgeType.AGGREGATES_TO, 'in')
            if children:
                aggregations.append({
                    'parent': parent_id,
                    'children': children,
                    'parent_node': self.get_node(parent_id),
                    'child_nodes': [self.get_node(c) for c in children]
                })
        
        return aggregations
    
    def find_bilateral_constraints(self) -> List[Dict[str, Any]]:
        """
        Find all bilateral position constraints.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of bilateral constraints
        """
        constraints = []
        
        for node_id in self.get_nodes_by_type(NodeType.BILATERAL):
            node = self.get_node(node_id)
            
            # Get connected series
            asset_series = []
            liability_series = []
            
            for neighbor in self.get_neighbors(node_id, direction='both'):
                neighbor_node = self.get_node(neighbor)
                if neighbor_node and neighbor_node.node_type == NodeType.SERIES:
                    # Check edge type to determine if asset or liability
                    for u, v, data in self.G.edges(data=True):
                        if u == neighbor and v == node_id:
                            if data.get('edge_type') == EdgeType.BILATERAL_ASSET.value:
                                asset_series.append(neighbor_node.metadata.get('code'))
                        elif u == node_id and v == neighbor:
                            if data.get('edge_type') == EdgeType.BILATERAL_LIABILITY.value:
                                liability_series.append(neighbor_node.metadata.get('code'))
            
            if asset_series or liability_series:
                constraints.append({
                    'bilateral_node': node_id,
                    'holder': node.metadata.get('holder'),
                    'issuer': node.metadata.get('issuer'),
                    'instrument': node.metadata.get('instrument'),
                    'asset_series': asset_series,
                    'liability_series': liability_series,
                    'level': node.metadata.get('level', 0)
                })
        
        return constraints
    
    def get_subgraph(self, node_ids: List[str], 
                    include_edges: bool = True) -> 'SFCGraph':
        """
        Extract a subgraph containing specified nodes.
        
        Parameters:
        -----------
        node_ids : List[str]
            Nodes to include
        include_edges : bool
            Whether to include edges between nodes
        
        Returns:
        --------
        SFCGraph
            New graph with subset of nodes
        """
        subgraph = SFCGraph()
        
        # Add nodes
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)
        
        # Add edges if requested
        if include_edges:
            for u, v, data in self.G.edges(data=True):
                if u in node_ids and v in node_ids:
                    edge = SFCEdge(
                        source=u,
                        target=v,
                        edge_type=EdgeType(data.get('edge_type', 'holds')),
                        weight=data.get('weight', 1.0),
                        metadata={k: v for k, v in data.items() 
                                if k not in ['edge_type', 'weight']}
                    )
                    subgraph.add_edge(edge)
        
        return subgraph
    
    def validate_structure(self) -> Dict[str, Any]:
        """
        Validate graph structure for consistency.
        
        Returns:
        --------
        Dict[str, Any]
            Validation report with any issues found
        """
        issues = []
        
        # Check for orphaned nodes
        orphans = [
            node for node in self.G.nodes()
            if self.G.degree(node) == 0
        ]
        if orphans:
            issues.append({
                'type': 'orphaned_nodes',
                'count': len(orphans),
                'nodes': orphans[:10]  # First 10
            })
        
        # Check for missing flow series
        for pair in self.find_stock_flow_pairs():
            if 'FU' not in pair and 'FR' not in pair and 'FV' not in pair:
                issues.append({
                    'type': 'incomplete_stock_flow',
                    'stock': pair['FL'],
                    'missing': ['FU', 'FR', 'FV']
                })
        
        # Check bilateral consistency
        for constraint in self.find_bilateral_constraints():
            if not constraint['asset_series'] and not constraint['liability_series']:
                issues.append({
                    'type': 'unmapped_bilateral',
                    'bilateral': constraint['bilateral_node']
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'statistics': {
                'nodes': self.G.number_of_nodes(),
                'edges': self.G.number_of_edges(),
                'sectors': len(self.get_nodes_by_type(NodeType.SECTOR)),
                'instruments': len(self.get_nodes_by_type(NodeType.INSTRUMENT)),
                'series': len(self.get_nodes_by_type(NodeType.SERIES)),
                'bilateral': len(self.get_nodes_by_type(NodeType.BILATERAL)),
                'orphans': len(orphans)
            }
        }
    
    def to_networkx(self) -> nx.MultiDiGraph:
        """Return the underlying NetworkX graph."""
        return self.G
    
    def __repr__(self):
        return (f"SFCGraph(nodes={self.G.number_of_nodes()}, "
                f"edges={self.G.number_of_edges()})")
