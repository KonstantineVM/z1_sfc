#!/usr/bin/env python3
"""
PLACEMENT: src/graph/constraint_extractor.py

Extracts constraint matrices (A, b) from the SFC graph structure.
Handles stock-flow, aggregation, market clearing, and bilateral constraints.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import logging
from dataclasses import dataclass
from enum import Enum

from .sfc_graph import SFCGraph, NodeType, EdgeType
from .state_index import StateIndex

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints in the SFC system."""
    STOCK_FLOW = "stock_flow"
    AGGREGATION = "aggregation"
    MARKET_CLEARING = "market_clearing"
    BILATERAL = "bilateral"
    FORMULA = "formula"
    IDENTITY = "identity"


@dataclass
class ConstraintMetadata:
    """Metadata for a constraint row."""
    row_index: int
    constraint_type: ConstraintType
    description: str
    series_involved: List[str]
    tolerance: float = 1e-10
    weight: float = 1.0
    metadata: Dict[str, Any] = None


class ConstraintExtractor:
    """
    Extracts constraint matrices from SFC graph.
    
    This class converts the graph structure into sparse constraint matrices
    A and b such that Ax = b represents all SFC constraints.
    """
    
    def __init__(self, graph: SFCGraph, state_index: StateIndex, 
                 config: Optional[Dict] = None):
        """
        Initialize constraint extractor.
        
        Parameters:
        -----------
        graph : SFCGraph
            The SFC graph structure
        state_index : StateIndex
            Mapping from series to state indices
        config : Optional[Dict]
            Configuration for constraint extraction
        """
        self.graph = graph
        self.state_index = state_index
        self.config = config or {}
        
        # Get constraint settings
        self.enforce_sfc = self.config.get('enforce_sfc', True)
        self.enforce_aggregation = self.config.get('enforce_aggregation', True)
        self.enforce_market_clearing = self.config.get('enforce_market_clearing', True)
        self.enforce_bilateral = self.config.get('enforce_bilateral', False)
        
        # Get weights
        weights = self.config.get('constraints', {}).get('weights', {})
        self.weight_stock_flow = weights.get('stock_flow', 1.0)
        self.weight_aggregation = weights.get('aggregation', 1.0)
        self.weight_market_clearing = weights.get('market_clearing', 0.8)
        self.weight_bilateral = weights.get('bilateral', 0.1)
        
        # Get tolerances
        tolerances = self.config.get('constraints', {}).get('tolerances', {})
        self.tol_stock_flow = tolerances.get('stock_flow', 1e-10)
        self.tol_aggregation = tolerances.get('aggregation', 1e-10)
        self.tol_market_clearing = tolerances.get('market_clearing', 1e-6)
        self.tol_bilateral = tolerances.get('bilateral', 1e-4)
        
        logger.info(f"ConstraintExtractor initialized with {len(graph.G.nodes)} nodes")
    
    def extract_at_time(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract all constraints at time t.
        
        Parameters:
        -----------
        t : int
            Time period
            
        Returns:
        --------
        A : sparse.csr_matrix
            Constraint matrix
        b : np.ndarray
            Right-hand side vector
        metadata : List[ConstraintMetadata]
            Metadata for each constraint
        """
        constraints = []
        
        # Stock-flow constraints
        if self.enforce_sfc:
            A_sf, b_sf, meta_sf = self.extract_stock_flow(t)
            if A_sf.shape[0] > 0:
                constraints.append((A_sf, b_sf, meta_sf))
        
        # Aggregation constraints
        if self.enforce_aggregation:
            A_agg, b_agg, meta_agg = self.extract_aggregation(t)
            if A_agg.shape[0] > 0:
                constraints.append((A_agg, b_agg, meta_agg))
        
        # Market clearing constraints
        if self.enforce_market_clearing:
            A_mc, b_mc, meta_mc = self.extract_market_clearing(t)
            if A_mc.shape[0] > 0:
                constraints.append((A_mc, b_mc, meta_mc))
        
        # Bilateral constraints
        if self.enforce_bilateral:
            A_bi, b_bi, meta_bi = self.extract_bilateral(t)
            if A_bi.shape[0] > 0:
                constraints.append((A_bi, b_bi, meta_bi))
        
        # Combine all constraints
        if constraints:
            A_list, b_list, meta_list = zip(*constraints)
            A = vstack(A_list, format='csr')
            b = np.hstack(b_list)
            metadata = []
            for meta_group in meta_list:
                metadata.extend(meta_group)
        else:
            # No constraints
            A = sparse.csr_matrix((0, self.state_index.size))
            b = np.array([])
            metadata = []
        
        logger.debug(f"Extracted {A.shape[0]} constraints at t={t}")
        return A, b, metadata
    
    def extract_stock_flow(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract stock-flow constraints.
        FL[t] - FL[t-1] - FU[t] - FR[t] - FV[t] = 0
        
        Only for base series (leaf nodes), not computed/aggregated series.
        """
        rows = []
        cols = []
        data = []
        metadata = []
        row_idx = 0
        
        # Find all FL series nodes
        fl_nodes = [
            (node_id, attrs) for node_id, attrs in self.graph.G.nodes(data=True)
            if attrs.get('node_type') == NodeType.SERIES.value
            and (attrs.get('code', node_id).startswith('FL') or 
                 attrs.get('prefix') == 'FL')
        ]
        
        for node_id, attrs in fl_nodes:
            series_code = attrs.get('code', node_id)
            
            # Check if this is a leaf FL that can have flows
            if not self.state_index.is_flow_allowed(series_code):
                logger.debug(f"Skipping flows for computed series {series_code}")
                continue
            
            # FL[t] coefficient
            try:
                idx = self.state_index.get(series_code, 0)
                cols.append(idx)
                data.append(1.0)
                rows.append(row_idx)
            except KeyError:
                logger.warning(f"Series {series_code} not in state index")
                continue
            
            # FL[t-1] coefficient (if t > 0)
            if t > 0 and self.state_index.has(series_code, -1):
                idx = self.state_index.get(series_code, -1)
                cols.append(idx)
                data.append(-1.0)
                rows.append(row_idx)
            
            # Find flow components (FU, FR, FV)
            series_involved = [series_code]
            
            # Look for incoming stock-flow edges
            for src, dst, edge_data in self.graph.G.in_edges(node_id, data=True):
                if edge_data.get('edge_type') == EdgeType.STOCK_FLOW.value:
                    src_attrs = self.graph.G.nodes[src]
                    src_code = src_attrs.get('code', src)
                    
                    # Add flow with negative coefficient
                    if self.state_index.has(src_code, 0):
                        idx = self.state_index.get(src_code, 0)
                        weight = float(edge_data.get('weight', 1.0))
                        cols.append(idx)
                        data.append(-weight)
                        rows.append(row_idx)
                        series_involved.append(src_code)
            
            # Create metadata
            meta = ConstraintMetadata(
                row_index=row_idx,
                constraint_type=ConstraintType.STOCK_FLOW,
                description=f"Stock-flow for {series_code} at t={t}",
                series_involved=series_involved,
                tolerance=self.tol_stock_flow,
                weight=self.weight_stock_flow,
                metadata={'time': t, 'series': series_code}
            )
            metadata.append(meta)
            row_idx += 1
        
        # Build sparse matrix
        if rows:
            A = coo_matrix((data, (rows, cols)), 
                          shape=(row_idx, self.state_index.size)).tocsr()
            b = np.zeros(row_idx)
        else:
            A = sparse.csr_matrix((0, self.state_index.size))
            b = np.array([])
        
        logger.debug(f"Extracted {row_idx} stock-flow constraints")
        return A, b, metadata
    
    def extract_aggregation(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract aggregation constraints from formulas.
        Parent = Sum(Children * weights)
        """
        rows = []
        cols = []
        data = []
        metadata = []
        row_idx = 0
        
        # Find aggregation edges
        for src, dst, edge_data in self.graph.G.edges(data=True):
            if edge_data.get('edge_type') != EdgeType.AGGREGATES_TO.value:
                continue
            
            # Get destination (parent) series
            dst_attrs = self.graph.G.nodes[dst]
            parent_code = dst_attrs.get('code', dst)
            
            if not self.state_index.has(parent_code, 0):
                continue
            
            # Collect all components for this parent
            components = []
            for s, d, ed in self.graph.G.in_edges(dst, data=True):
                if ed.get('edge_type') == EdgeType.AGGREGATES_TO.value:
                    src_attrs = self.graph.G.nodes[s]
                    comp_code = src_attrs.get('code', s)
                    weight = float(ed.get('weight', 1.0))
                    components.append((comp_code, weight))
            
            if not components:
                continue
            
            # Build constraint: parent - sum(components) = 0
            series_involved = [parent_code]
            
            # Parent coefficient
            idx = self.state_index.get(parent_code, 0)
            cols.append(idx)
            data.append(1.0)
            rows.append(row_idx)
            
            # Component coefficients
            for comp_code, weight in components:
                if self.state_index.has(comp_code, 0):
                    idx = self.state_index.get(comp_code, 0)
                    cols.append(idx)
                    data.append(-weight)
                    rows.append(row_idx)
                    series_involved.append(comp_code)
            
            # Create metadata
            meta = ConstraintMetadata(
                row_index=row_idx,
                constraint_type=ConstraintType.AGGREGATION,
                description=f"Aggregation for {parent_code}",
                series_involved=series_involved,
                tolerance=self.tol_aggregation,
                weight=self.weight_aggregation,
                metadata={'parent': parent_code, 'n_components': len(components)}
            )
            metadata.append(meta)
            row_idx += 1
        
        # Build sparse matrix
        if rows:
            A = coo_matrix((data, (rows, cols)), 
                          shape=(row_idx, self.state_index.size)).tocsr()
            b = np.zeros(row_idx)
        else:
            A = sparse.csr_matrix((0, self.state_index.size))
            b = np.array([])
        
        logger.debug(f"Extracted {row_idx} aggregation constraints")
        return A, b, metadata
    
    def extract_market_clearing(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract market clearing constraints.
        For each instrument: Sum(Assets) - Sum(Liabilities) = 0
        """
        rows = []
        cols = []
        data = []
        metadata = []
        row_idx = 0
        
        # Get instrument nodes
        instrument_nodes = [
            (node_id, attrs) for node_id, attrs in self.graph.G.nodes(data=True)
            if attrs.get('node_type') == NodeType.INSTRUMENT.value
        ]
        
        for inst_node_id, inst_attrs in instrument_nodes:
            instrument_code = inst_attrs.get('code', inst_node_id)
            
            # Find all series for this instrument
            assets = []
            liabilities = []
            
            # Look through series nodes
            for node_id, attrs in self.graph.G.nodes(data=True):
                if attrs.get('node_type') != NodeType.SERIES.value:
                    continue
                
                series_code = attrs.get('code', node_id)
                series_instrument = attrs.get('instrument')
                
                if series_instrument == instrument_code:
                    # Determine if asset or liability based on prefix or metadata
                    if attrs.get('is_liability'):
                        liabilities.append(series_code)
                    else:
                        assets.append(series_code)
            
            if not assets and not liabilities:
                continue
            
            # Build constraint: Sum(Assets) - Sum(Liabilities) = 0
            series_involved = []
            
            # Asset coefficients (+1)
            for asset_code in assets:
                if self.state_index.has(asset_code, 0):
                    idx = self.state_index.get(asset_code, 0)
                    cols.append(idx)
                    data.append(1.0)
                    rows.append(row_idx)
                    series_involved.append(asset_code)
            
            # Liability coefficients (-1)
            for liab_code in liabilities:
                if self.state_index.has(liab_code, 0):
                    idx = self.state_index.get(liab_code, 0)
                    cols.append(idx)
                    data.append(-1.0)
                    rows.append(row_idx)
                    series_involved.append(liab_code)
            
            if series_involved:
                # Create metadata
                meta = ConstraintMetadata(
                    row_index=row_idx,
                    constraint_type=ConstraintType.MARKET_CLEARING,
                    description=f"Market clearing for instrument {instrument_code}",
                    series_involved=series_involved,
                    tolerance=self.tol_market_clearing,
                    weight=self.weight_market_clearing,
                    metadata={
                        'instrument': instrument_code,
                        'n_assets': len(assets),
                        'n_liabilities': len(liabilities)
                    }
                )
                metadata.append(meta)
                row_idx += 1
        
        # Build sparse matrix
        if rows:
            A = coo_matrix((data, (rows, cols)), 
                          shape=(row_idx, self.state_index.size)).tocsr()
            b = np.zeros(row_idx)
        else:
            A = sparse.csr_matrix((0, self.state_index.size))
            b = np.array([])
        
        logger.debug(f"Extracted {row_idx} market clearing constraints")
        return A, b, metadata
    
    def extract_bilateral(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract bilateral consistency constraints.
        FWTW positions should sum to Z1 aggregates.
        """
        # Placeholder for bilateral constraints
        # This would be implemented based on FWTW data structure
        A = sparse.csr_matrix((0, self.state_index.size))
        b = np.array([])
        metadata = []
        
        logger.debug("Bilateral constraints not yet implemented")
        return A, b, metadata
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary statistics about constraints."""
        summary = {
            'n_nodes': len(self.graph.G.nodes),
            'n_edges': len(self.graph.G.edges),
            'state_size': self.state_index.size,
            'n_series': self.state_index.n_series,
            'max_lag': self.state_index.max_lag,
            'enforcement': {
                'stock_flow': self.enforce_sfc,
                'aggregation': self.enforce_aggregation,
                'market_clearing': self.enforce_market_clearing,
                'bilateral': self.enforce_bilateral
            },
            'weights': {
                'stock_flow': self.weight_stock_flow,
                'aggregation': self.weight_aggregation,
                'market_clearing': self.weight_market_clearing,
                'bilateral': self.weight_bilateral
            }
        }
        
        # Count constraint types
        n_fl_base = len([
            s for s in self.state_index.series_names
            if s.startswith('FL') and self.state_index.is_flow_allowed(s)
        ])
        n_fl_computed = len([
            s for s in self.state_index.series_names
            if s.startswith('FL') and not self.state_index.is_flow_allowed(s)
        ])
        
        summary['constraint_counts'] = {
            'potential_stock_flow': n_fl_base,
            'potential_aggregation': n_fl_computed,
            'potential_market_clearing': len([
                n for n, d in self.graph.G.nodes(data=True)
                if d.get('node_type') == NodeType.INSTRUMENT.value
            ])
        }
        
        return summary
