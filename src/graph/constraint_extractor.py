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
    STOCK_FLOW = "stock_flow"              # FL[t] - FL[t-1] - FU - FR - FV = 0
    AGGREGATION = "aggregation"            # Parent = Sum(Children)
    MARKET_CLEARING = "market_clearing"    # Sum(Assets) = Sum(Liabilities)
    BILATERAL = "bilateral"                # FWTW position consistency
    FORMULA = "formula"                    # User-defined formulas
    IDENTITY = "identity"                  # Accounting identities


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
    
    def __init__(self, graph: SFCGraph, state_index: StateIndex):
        """
        Initialize constraint extractor.
        
        Parameters:
        -----------
        graph : SFCGraph
            The SFC graph containing all relationships
        state_index : StateIndex
            Mapping from series/lag to state vector indices
        """
        self.graph = graph
        self.state_index = state_index
        self.constraint_metadata = []
        
        # Tolerance levels for different constraint types
        self.tolerances = {
            ConstraintType.STOCK_FLOW: 1e-10,
            ConstraintType.AGGREGATION: 1e-8,
            ConstraintType.MARKET_CLEARING: 1e-6,
            ConstraintType.BILATERAL: 1e-4,
            ConstraintType.FORMULA: 1e-8,
            ConstraintType.IDENTITY: 1e-10
        }
        
        # Weights for soft constraints
        self.weights = {
            ConstraintType.STOCK_FLOW: 1.0,       # Hard constraint
            ConstraintType.AGGREGATION: 1.0,      # Hard constraint
            ConstraintType.MARKET_CLEARING: 0.8,  # Slightly soft
            ConstraintType.BILATERAL: 0.5,        # Soft initially
            ConstraintType.FORMULA: 0.9,          # Mostly hard
            ConstraintType.IDENTITY: 1.0          # Hard constraint
        }
    
    def extract_at_time(self, t: int) -> Tuple[csr_matrix, np.ndarray, List[ConstraintMetadata]]:
        """
        Extract all constraints at time t.
        
        Parameters:
        -----------
        t : int
            Time period
        
        Returns:
        --------
        A : csr_matrix
            Constraint matrix (n_constraints x state_size)
        b : np.ndarray
            Constraint values (n_constraints,)
        metadata : List[ConstraintMetadata]
            Metadata for each constraint row
        """
        logger.info(f"Extracting constraints at t={t}")
        
        # Clear metadata
        self.constraint_metadata = []
        
        # Extract each type of constraint
        constraints = []
        
        # Stock-flow constraints
        A_sf, b_sf, meta_sf = self._extract_stock_flow_constraints(t)
        if A_sf is not None:
            constraints.append((A_sf, b_sf, meta_sf))
            logger.info(f"  Stock-flow: {A_sf.shape[0]} constraints")
        
        # Aggregation constraints
        A_agg, b_agg, meta_agg = self._extract_aggregation_constraints(t)
        if A_agg is not None:
            constraints.append((A_agg, b_agg, meta_agg))
            logger.info(f"  Aggregation: {A_agg.shape[0]} constraints")
        
        # Market clearing constraints
        A_mc, b_mc, meta_mc = self._extract_market_clearing_constraints(t)
        if A_mc is not None:
            constraints.append((A_mc, b_mc, meta_mc))
            logger.info(f"  Market clearing: {A_mc.shape[0]} constraints")
        
        # Bilateral constraints
        A_bil, b_bil, meta_bil = self._extract_bilateral_constraints(t)
        if A_bil is not None:
            constraints.append((A_bil, b_bil, meta_bil))
            logger.info(f"  Bilateral: {A_bil.shape[0]} constraints")
        
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
            A = csr_matrix((0, self.state_index.size))
            b = np.array([])
            metadata = []
        
        logger.info(f"Total constraints: {A.shape[0]} x {A.shape[1]}")
        logger.info(f"Sparsity: {A.nnz / (A.shape[0] * A.shape[1]) if A.shape[0] > 0 else 0:.2%}")
        
        return A, b, metadata
    
    def _extract_stock_flow_constraints(self, t: int) -> Tuple[Optional[csr_matrix], 
                                                               Optional[np.ndarray], 
                                                               List[ConstraintMetadata]]:
        """
        Extract stock-flow constraints: FL[t] - FL[t-1] - FU - FR - FV = 0
        """
        pairs = self.graph.find_stock_flow_pairs()
        if not pairs:
            return None, None, []
        
        rows, cols, data = [], [], []
        rhs = []
        metadata = []
        row_idx = len(self.constraint_metadata)
        
        for pair in pairs:
            fl_code = pair.get('FL')
            if not fl_code or not self.state_index.has(fl_code, 0):
                continue
            
            # Build constraint coefficients
            coefficients = {}
            
            # FL[t] term
            coefficients[(fl_code, 0)] = 1.0
            
            # FL[t-1] term (if t > 0)
            if t > 0 and self.state_index.has(fl_code, -1):
                coefficients[(fl_code, -1)] = -1.0
            
            # Flow terms
            for prefix in ['FU', 'FR', 'FV']:
                flow_code = pair.get(prefix)
                if flow_code and self.state_index.has(flow_code, 0):
                    coefficients[(flow_code, 0)] = -1.0
            
            # Build sparse row
            for (series, lag), coef in coefficients.items():
                idx = self.state_index.get(series, lag)
                rows.append(row_idx)
                cols.append(idx)
                data.append(coef)
            
            rhs.append(0.0)
            
            # Add metadata
            meta = ConstraintMetadata(
                row_index=row_idx,
                constraint_type=ConstraintType.STOCK_FLOW,
                description=f"Stock-flow: {fl_code}",
                series_involved=list(pair.values()),
                tolerance=self.tolerances[ConstraintType.STOCK_FLOW],
                weight=self.weights[ConstraintType.STOCK_FLOW],
                metadata={'pair': pair, 'time': t}
            )
            metadata.append(meta)
            self.constraint_metadata.append(meta)
            row_idx += 1
        
        if not rows:
            return None, None, []
        
        A = coo_matrix((data, (rows, cols)), 
                      shape=(row_idx - len(self.constraint_metadata) + len(metadata), 
                            self.state_index.size)).tocsr()
        b = np.array(rhs)
        
        return A, b, metadata
    
    def _extract_aggregation_constraints(self, t: int) -> Tuple[Optional[csr_matrix],
                                                                Optional[np.ndarray],
                                                                List[ConstraintMetadata]]:
        """
        Extract aggregation constraints: Parent = Sum(Children)
        """
        aggregations = self.graph.find_aggregation_relationships()
        if not aggregations:
            return None, None, []
        
        rows, cols, data = [], [], []
        rhs = []
        metadata = []
        row_idx = len(self.constraint_metadata)
        
        for agg in aggregations:
            parent_node = agg['parent_node']
            if parent_node.node_type != NodeType.SERIES:
                continue
            
            parent_code = parent_node.metadata.get('code')
            if not parent_code or not self.state_index.has(parent_code, 0):
                continue
            
            # Parent coefficient
            parent_idx = self.state_index.get(parent_code, 0)
            rows.append(row_idx)
            cols.append(parent_idx)
            data.append(1.0)
            
            # Children coefficients
            children_codes = []
            for child_node in agg['child_nodes']:
                if child_node.node_type == NodeType.SERIES:
                    child_code = child_node.metadata.get('code')
                    if child_code and self.state_index.has(child_code, 0):
                        child_idx = self.state_index.get(child_code, 0)
                        rows.append(row_idx)
                        cols.append(child_idx)
                        data.append(-1.0)
                        children_codes.append(child_code)
            
            if children_codes:
                rhs.append(0.0)
                
                # Add metadata
                meta = ConstraintMetadata(
                    row_index=row_idx,
                    constraint_type=ConstraintType.AGGREGATION,
                    description=f"Aggregation: {parent_code} = Sum({len(children_codes)} children)",
                    series_involved=[parent_code] + children_codes,
                    tolerance=self.tolerances[ConstraintType.AGGREGATION],
                    weight=self.weights[ConstraintType.AGGREGATION],
                    metadata={'parent': parent_code, 'children': children_codes, 'time': t}
                )
                metadata.append(meta)
                self.constraint_metadata.append(meta)
                row_idx += 1
        
        if not rows:
            return None, None, []
        
        A = coo_matrix((data, (rows, cols)),
                      shape=(len(metadata), self.state_index.size)).tocsr()
        b = np.array(rhs)
        
        return A, b, metadata
    
    def _extract_market_clearing_constraints(self, t: int) -> Tuple[Optional[csr_matrix],
                                                                   Optional[np.ndarray],
                                                                   List[ConstraintMetadata]]:
        """
        Extract market clearing constraints: Sum(Assets) = Sum(Liabilities) per instrument
        """
        # Group series by instrument
        instrument_groups = {}
        
        for node_id in self.graph.get_nodes_by_type(NodeType.SERIES):
            node = self.graph.get_node(node_id)
            instrument = node.metadata.get('instrument')
            prefix = node.metadata.get('prefix')
            code = node.metadata.get('code')
            
            if instrument and prefix == 'FL' and code:
                if instrument not in instrument_groups:
                    instrument_groups[instrument] = {'assets': [], 'liabilities': []}
                
                # Determine if asset or liability based on instrument code
                inst_code = int(instrument)
                if inst_code < 31000:  # Asset codes
                    instrument_groups[instrument]['assets'].append(code)
                else:  # Liability codes
                    instrument_groups[instrument]['liabilities'].append(code)
        
        if not instrument_groups:
            return None, None, []
        
        rows, cols, data = [], [], []
        rhs = []
        metadata = []
        row_idx = len(self.constraint_metadata)
        
        for instrument, groups in instrument_groups.items():
            if not groups['assets'] or not groups['liabilities']:
                continue
            
            # Assets (positive coefficients)
            for asset_code in groups['assets']:
                if self.state_index.has(asset_code, 0):
                    idx = self.state_index.get(asset_code, 0)
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(1.0)
            
            # Liabilities (negative coefficients)
            for liability_code in groups['liabilities']:
                if self.state_index.has(liability_code, 0):
                    idx = self.state_index.get(liability_code, 0)
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(-1.0)
            
            rhs.append(0.0)
            
            # Add metadata
            meta = ConstraintMetadata(
                row_index=row_idx,
                constraint_type=ConstraintType.MARKET_CLEARING,
                description=f"Market clearing: Instrument {instrument}",
                series_involved=groups['assets'] + groups['liabilities'],
                tolerance=self.tolerances[ConstraintType.MARKET_CLEARING],
                weight=self.weights[ConstraintType.MARKET_CLEARING],
                metadata={'instrument': instrument, 'time': t}
            )
            metadata.append(meta)
            self.constraint_metadata.append(meta)
            row_idx += 1
        
        if not rows:
            return None, None, []
        
        A = coo_matrix((data, (rows, cols)),
                      shape=(len(metadata), self.state_index.size)).tocsr()
        b = np.array(rhs)
        
        return A, b, metadata
    
    def _extract_bilateral_constraints(self, t: int) -> Tuple[Optional[csr_matrix],
                                                             Optional[np.ndarray],
                                                             List[ConstraintMetadata]]:
        """
        Extract bilateral constraints from FWTW positions
        """
        bilateral_constraints = self.graph.find_bilateral_constraints()
        if not bilateral_constraints:
            return None, None, []
        
        rows, cols, data = [], [], []
        rhs = []
        metadata = []
        row_idx = len(self.constraint_metadata)
        
        for constraint in bilateral_constraints:
            # For each bilateral position, asset series should match liability series
            asset_series = constraint.get('asset_series', [])
            liability_series = constraint.get('liability_series', [])
            
            if not asset_series or not liability_series:
                continue
            
            # Create constraint: Sum(assets) - Sum(liabilities) = 0
            has_terms = False
            
            for asset_code in asset_series:
                if self.state_index.has(asset_code, 0):
                    idx = self.state_index.get(asset_code, 0)
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(1.0)
                    has_terms = True
            
            for liability_code in liability_series:
                if self.state_index.has(liability_code, 0):
                    idx = self.state_index.get(liability_code, 0)
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(1.0)  # Note: positive because liability series already negative
                    has_terms = True
            
            if has_terms:
                rhs.append(0.0)
                
                # Add metadata
                meta = ConstraintMetadata(
                    row_index=row_idx,
                    constraint_type=ConstraintType.BILATERAL,
                    description=f"Bilateral: {constraint['holder']}→{constraint['issuer']} {constraint['instrument']}",
                    series_involved=asset_series + liability_series,
                    tolerance=self.tolerances[ConstraintType.BILATERAL],
                    weight=self.weights[ConstraintType.BILATERAL],
                    metadata=constraint
                )
                metadata.append(meta)
                self.constraint_metadata.append(meta)
                row_idx += 1
        
        if not rows:
            return None, None, []
        
        A = coo_matrix((data, (rows, cols)),
                      shape=(len(metadata), self.state_index.size)).tocsr()
        b = np.array(rhs)
        
        return A, b, metadata
    
    def build_weight_matrix(self, metadata: List[ConstraintMetadata]) -> csr_matrix:
        """
        Build diagonal weight matrix for weighted constraints.
        
        Parameters:
        -----------
        metadata : List[ConstraintMetadata]
            Constraint metadata with weights
        
        Returns:
        --------
        W : csr_matrix
            Diagonal weight matrix
        """
        n = len(metadata)
        weights = np.array([meta.weight for meta in metadata])
        W = sparse.diags(weights, format='csr')
        return W
    
    def get_constraint_summary(self) -> Dict[str, int]:
        """Get summary of constraints by type."""
        summary = {}
        for constraint_type in ConstraintType:
            count = sum(1 for meta in self.constraint_metadata 
                       if meta.constraint_type == constraint_type)
            if count > 0:
                summary[constraint_type.value] = count
        return summary
    
    def validate_constraints(self, x: np.ndarray, 
                           A: csr_matrix, 
                           b: np.ndarray,
                           metadata: List[ConstraintMetadata]) -> Dict[str, Any]:
        """
        Validate constraint satisfaction.
        
        Parameters:
        -----------
        x : np.ndarray
            State vector
        A : csr_matrix
            Constraint matrix
        b : np.ndarray
            Constraint values
        metadata : List[ConstraintMetadata]
            Constraint metadata
        
        Returns:
        --------
        Dict[str, Any]
            Validation report
        """
        residuals = A @ x - b
        
        violations = []
        for i, (res, meta) in enumerate(zip(residuals, metadata)):
            if abs(res) > meta.tolerance:
                violations.append({
                    'index': i,
                    'type': meta.constraint_type.value,
                    'description': meta.description,
                    'residual': float(res),
                    'tolerance': meta.tolerance,
                    'violation_ratio': abs(res) / meta.tolerance
                })
        
        # Summary by constraint type
        type_summary = {}
        for constraint_type in ConstraintType:
            type_residuals = [
                abs(res) for res, meta in zip(residuals, metadata)
                if meta.constraint_type == constraint_type
            ]
            if type_residuals:
                type_summary[constraint_type.value] = {
                    'count': len(type_residuals),
                    'max_violation': max(type_residuals),
                    'mean_violation': np.mean(type_residuals),
                    'satisfied': sum(1 for r in type_residuals if r < self.tolerances[constraint_type])
                }
        
        return {
            'valid': len(violations) == 0,
            'n_constraints': len(metadata),
            'n_violations': len(violations),
            'violations': violations[:10],  # First 10 violations
            'max_violation': float(np.max(np.abs(residuals))) if len(residuals) > 0 else 0.0,
            'mean_violation': float(np.mean(np.abs(residuals))) if len(residuals) > 0 else 0.0,
            'type_summary': type_summary
        }