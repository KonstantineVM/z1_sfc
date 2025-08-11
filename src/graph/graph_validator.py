#!/usr/bin/env python3
"""
PLACEMENT: src/graph/graph_validator.py

Validate graph-based constraints against legacy implementation.
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GraphValidator:
    """Validate graph constraints against legacy implementation."""
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize validator.
        
        Parameters:
        -----------
        tolerance : float
            Tolerance for numerical comparison
        """
        self.tolerance = tolerance
    
    def compare_constraints(self, 
                           A_legacy: sparse.spmatrix,
                           b_legacy: np.ndarray,
                           A_graph: sparse.spmatrix, 
                           b_graph: np.ndarray) -> Dict[str, Any]:
        """
        Compare legacy and graph-based constraints.
        
        Parameters:
        -----------
        A_legacy, b_legacy : Constraint matrices from legacy implementation
        A_graph, b_graph : Constraint matrices from graph implementation
        
        Returns:
        --------
        dict
            Comparison report with differences
        """
        report = {
            'shape_match': A_legacy.shape == A_graph.shape,
            'legacy_shape': A_legacy.shape,
            'graph_shape': A_graph.shape,
            'nnz_legacy': A_legacy.nnz,
            'nnz_graph': A_graph.nnz,
            'b_length_match': len(b_legacy) == len(b_graph),
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'mismatched_rows': []
        }
        
        # Check shapes
        if not report['shape_match']:
            logger.warning(f"Shape mismatch: legacy {A_legacy.shape} vs graph {A_graph.shape}")
            return report
        
        # Compare A matrices
        A_diff = A_legacy - A_graph
        if sparse.issparse(A_diff):
            A_diff_dense = A_diff.toarray()
        else:
            A_diff_dense = A_diff
        
        report['max_diff'] = np.max(np.abs(A_diff_dense))
        report['mean_diff'] = np.mean(np.abs(A_diff_dense))
        
        # Find mismatched rows
        row_diffs = np.max(np.abs(A_diff_dense), axis=1)
        mismatched = np.where(row_diffs > self.tolerance)[0]
        
        for row_idx in mismatched[:10]:  # Limit to first 10
            report['mismatched_rows'].append({
                'row': int(row_idx),
                'max_diff': float(row_diffs[row_idx]),
                'legacy_nnz': int((A_legacy[row_idx] != 0).sum()),
                'graph_nnz': int((A_graph[row_idx] != 0).sum())
            })
        
        # Compare b vectors
        b_diff = np.abs(b_legacy - b_graph)
        report['b_max_diff'] = float(np.max(b_diff)) if len(b_diff) > 0 else 0.0
        report['b_mean_diff'] = float(np.mean(b_diff)) if len(b_diff) > 0 else 0.0
        
        # Overall pass/fail
        report['passed'] = (
            report['max_diff'] < self.tolerance and
            report['b_max_diff'] < self.tolerance
        )
        
        return report
    
    def validate_constraint_properties(self,
                                      A: sparse.spmatrix,
                                      b: np.ndarray,
                                      metadata: List[Any]) -> Dict[str, Any]:
        """
        Validate properties of constraint matrices.
        
        Parameters:
        -----------
        A : Constraint matrix
        b : Right-hand side
        metadata : Constraint metadata
        
        Returns:
        --------
        dict
            Validation report
        """
        report = {
            'n_constraints': A.shape[0],
            'n_variables': A.shape[1],
            'sparsity': 1.0 - (A.nnz / (A.shape[0] * A.shape[1])),
            'rank': None,
            'condition_number': None,
            'has_zero_rows': False,
            'has_duplicate_rows': False,
            'constraint_types': {}
        }
        
        # Check for zero rows
        row_sums = np.abs(A).sum(axis=1).A1 if sparse.issparse(A) else np.abs(A).sum(axis=1)
        report['has_zero_rows'] = bool(np.any(row_sums == 0))
        
        # Count constraint types from metadata
        if metadata:
            for meta in metadata:
                ctype = str(meta.constraint_type.value) if hasattr(meta, 'constraint_type') else 'unknown'
                report['constraint_types'][ctype] = report['constraint_types'].get(ctype, 0) + 1
        
        # Compute rank and condition number for small matrices
        if A.shape[0] < 1000 and A.shape[1] < 1000:
            try:
                A_dense = A.toarray() if sparse.issparse(A) else A
                report['rank'] = int(np.linalg.matrix_rank(A_dense))
                
                # Condition number of A'A
                AtA = A.T @ A
                if sparse.issparse(AtA):
                    AtA = AtA.toarray()
                if AtA.shape[0] > 0:
                    eigenvalues = np.linalg.eigvalsh(AtA)
                    if len(eigenvalues) > 0 and eigenvalues[-1] > 0:
                        report['condition_number'] = float(
                            np.sqrt(eigenvalues[-1] / eigenvalues[0])
                        )
            except Exception as e:
                logger.debug(f"Could not compute rank/condition: {e}")
        
        return report
    
    def check_constraint_consistency(self,
                                    x: np.ndarray,
                                    A: sparse.spmatrix,
                                    b: np.ndarray) -> Dict[str, Any]:
        """
        Check how well a solution satisfies constraints.
        
        Parameters:
        -----------
        x : Solution vector
        A : Constraint matrix
        b : Right-hand side
        
        Returns:
        --------
        dict
            Consistency report
        """
        # Compute residual
        residual = A @ x - b
        
        report = {
            'max_violation': float(np.max(np.abs(residual))),
            'mean_violation': float(np.mean(np.abs(residual))),
            'std_violation': float(np.std(np.abs(residual))),
            'n_violations_1e-6': int(np.sum(np.abs(residual) > 1e-6)),
            'n_violations_1e-10': int(np.sum(np.abs(residual) > 1e-10)),
            'satisfied': bool(np.max(np.abs(residual)) < 1e-10)
        }
        
        return report
