#!/usr/bin/env python3
"""
PLACEMENT: src/graph/state_index.py

Maps series names and lags to state vector indices.
Critical for building constraint matrices that align with Kalman filter states.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StateIndex:
    """
    Maps (series_name, lag) pairs to state vector indices.
    
    This class maintains the mapping between:
    - Series names (e.g., 'FL154090005.Q')
    - Temporal lags (0 for current, -1 for t-1, etc.)
    - State vector positions in the Kalman filter
    
    The state vector is organized as:
    [x1[t], x2[t], ..., xn[t], x1[t-1], x2[t-1], ..., xn[t-1], ...]
    """
    
    def __init__(self, series_names: List[str], max_lag: int = 2):
        """
        Initialize state index mapping.
        
        Parameters:
        -----------
        series_names : List[str]
            List of series names in the data
        max_lag : int
            Maximum lag to include in state (default 2)
        """
        self.series_names = list(series_names)  # Preserve order
        self.max_lag = max_lag
        self.index: Dict[Tuple[str, int], int] = {}
        self.reverse_index: Dict[int, Tuple[str, int]] = {}
        
        # Build the index
        self._build_index()
        
        # Store dimensions
        self.n_series = len(self.series_names)
        self.size = len(self.index)
        
        logger.info(f"StateIndex created: {self.n_series} series, "
                   f"max_lag={max_lag}, state_size={self.size}")
    
    def _build_index(self) -> None:
        """Build the index mapping."""
        idx = 0
        
        # Current values (lag 0)
        for name in self.series_names:
            key = (name, 0)
            self.index[key] = idx
            self.reverse_index[idx] = key
            idx += 1
        
        # Lagged values
        for lag in range(1, self.max_lag + 1):
            for name in self.series_names:
                key = (name, -lag)
                self.index[key] = idx
                self.reverse_index[idx] = key
                idx += 1
    
    def get(self, name: str, lag: int = 0) -> int:
        """
        Get state index for a series at a specific lag.
        
        Parameters:
        -----------
        name : str
            Series name
        lag : int
            Lag (0 for current, -1 for t-1, etc.)
        
        Returns:
        --------
        int
            State vector index
        
        Raises:
        -------
        KeyError
            If series/lag combination not found
        """
        key = (name, lag)
        if key not in self.index:
            raise KeyError(f"State not found: {name} at lag {lag}")
        return self.index[key]
    
    def get_safe(self, name: str, lag: int = 0, 
                 default: Optional[int] = None) -> Optional[int]:
        """
        Safely get state index, returning default if not found.
        
        Parameters:
        -----------
        name : str
            Series name
        lag : int
            Lag value
        default : Optional[int]
            Default value if not found
        
        Returns:
        --------
        Optional[int]
            State index or default
        """
        try:
            return self.get(name, lag)
        except KeyError:
            return default
    
    def has(self, name: str, lag: int = 0) -> bool:
        """Check if series/lag combination exists in index."""
        return (name, lag) in self.index
    
    def get_series_at_index(self, idx: int) -> Tuple[str, int]:
        """
        Get series name and lag for a state index.
        
        Parameters:
        -----------
        idx : int
            State vector index
        
        Returns:
        --------
        Tuple[str, int]
            (series_name, lag)
        """
        if idx not in self.reverse_index:
            raise KeyError(f"Index {idx} not found")
        return self.reverse_index[idx]
    
    def get_current_indices(self) -> Dict[str, int]:
        """Get indices for all current (lag 0) series."""
        return {name: self.get(name, 0) for name in self.series_names}
    
    def get_lagged_indices(self, lag: int) -> Dict[str, int]:
        """
        Get indices for all series at a specific lag.
        
        Parameters:
        -----------
        lag : int
            Lag value (should be negative)
        
        Returns:
        --------
        Dict[str, int]
            Mapping of series names to indices
        """
        if lag > 0:
            raise ValueError("Lag should be 0 or negative")
        
        result = {}
        for name in self.series_names:
            if self.has(name, lag):
                result[name] = self.get(name, lag)
        return result
    
    def get_series_indices(self, series_name: str) -> Dict[int, int]:
        """
        Get all state indices for a specific series.
        
        Parameters:
        -----------
        series_name : str
            Series name
        
        Returns:
        --------
        Dict[int, int]
            Mapping of lag to state index
        """
        indices = {}
        
        # Current
        if self.has(series_name, 0):
            indices[0] = self.get(series_name, 0)
        
        # Lags
        for lag in range(1, self.max_lag + 1):
            if self.has(series_name, -lag):
                indices[-lag] = self.get(series_name, -lag)
        
        return indices
    
    def build_selection_vector(self, selections: List[Tuple[str, int]],
                              value: float = 1.0) -> np.ndarray:
        """
        Build a selection vector for specific series/lag combinations.
        
        Parameters:
        -----------
        selections : List[Tuple[str, int]]
            List of (series_name, lag) pairs to select
        value : float
            Value to place at selected indices
        
        Returns:
        --------
        np.ndarray
            Selection vector of size self.size
        """
        vector = np.zeros(self.size)
        
        for name, lag in selections:
            idx = self.get_safe(name, lag)
            if idx is not None:
                vector[idx] = value
        
        return vector
    
    def build_constraint_row(self, 
                            coefficients: Dict[Tuple[str, int], float]) -> np.ndarray:
        """
        Build a constraint matrix row from coefficients.
        
        Parameters:
        -----------
        coefficients : Dict[Tuple[str, int], float]
            Mapping of (series_name, lag) to coefficient
        
        Returns:
        --------
        np.ndarray
            Constraint row of size self.size
        """
        row = np.zeros(self.size)
        
        for (name, lag), coef in coefficients.items():
            idx = self.get_safe(name, lag)
            if idx is not None:
                row[idx] = coef
        
        return row
    
    def get_series_block(self, lag: int = 0) -> Tuple[int, int]:
        """
        Get the start and end indices for a lag block.
        
        Parameters:
        -----------
        lag : int
            Lag value (0, -1, -2, etc.)
        
        Returns:
        --------
        Tuple[int, int]
            (start_index, end_index) for the block
        """
        if lag == 0:
            return (0, self.n_series)
        elif abs(lag) <= self.max_lag:
            block_start = self.n_series * abs(lag)
            block_end = self.n_series * (abs(lag) + 1)
            return (block_start, block_end)
        else:
            raise ValueError(f"Lag {lag} exceeds max_lag {self.max_lag}")
    
    def create_lag_transition_matrix(self) -> np.ndarray:
        """
        Create transition matrix for lag structure.
        
        This creates a matrix that shifts lagged values:
        x[t-1] becomes x[t-2], etc.
        
        Returns:
        --------
        np.ndarray
            Transition matrix of shape (size, size)
        """
        T = np.zeros((self.size, self.size))
        
        # Identity for current values (they get updated by Kalman filter)
        T[:self.n_series, :self.n_series] = np.eye(self.n_series)
        
        # Shift lags
        for lag in range(1, self.max_lag):
            # x[t-lag] becomes x[t-(lag+1)]
            source_start, source_end = self.get_series_block(-lag)
            target_start, target_end = self.get_series_block(-(lag + 1))
            
            # Copy block
            T[target_start:target_end, source_start:source_end] = np.eye(self.n_series)
        
        # Current values become lag-1
        if self.max_lag >= 1:
            lag1_start, lag1_end = self.get_series_block(-1)
            T[lag1_start:lag1_end, :self.n_series] = np.eye(self.n_series)
        
        return T
    
    def validate_series(self, series_list: List[str]) -> Dict[str, bool]:
        """
        Validate which series are in the index.
        
        Parameters:
        -----------
        series_list : List[str]
            Series to check
        
        Returns:
        --------
        Dict[str, bool]
            Mapping of series name to whether it exists
        """
        return {
            series: series in self.series_names
            for series in series_list
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the state index."""
        return {
            'n_series': self.n_series,
            'max_lag': self.max_lag,
            'state_size': self.size,
            'current_block_size': self.n_series,
            'lag_blocks': self.max_lag,
            'total_entries': len(self.index)
        }
    
    def __repr__(self) -> str:
        return (f"StateIndex(n_series={self.n_series}, "
                f"max_lag={self.max_lag}, size={self.size})")
    
    def __len__(self) -> int:
        return self.size


class HierarchicalStateIndex(StateIndex):
    """
    Extended state index that handles hierarchical relationships.
    
    This class extends StateIndex to handle:
    - Level and trend components per series
    - Hierarchical aggregation relationships
    - Multi-frequency data
    """
    
    def __init__(self, series_names: List[str], 
                 max_lag: int = 2,
                 components: List[str] = ['level', 'trend']):
        """
        Initialize hierarchical state index.
        
        Parameters:
        -----------
        series_names : List[str]
            List of series names
        max_lag : int
            Maximum lag
        components : List[str]
            State components per series (e.g., ['level', 'trend'])
        """
        self.components = components
        self.n_components = len(components)
        
        # Expand series names to include components
        expanded_names = []
        for name in series_names:
            for component in components:
                expanded_names.append(f"{name}_{component}")
        
        # Initialize parent class with expanded names
        super().__init__(expanded_names, max_lag)
        
        # Store original series names
        self.original_series = series_names
        
    def get_component(self, name: str, component: str, lag: int = 0) -> int:
        """
        Get state index for a specific component of a series.
        
        Parameters:
        -----------
        name : str
            Original series name
        component : str
            Component name ('level', 'trend', etc.)
        lag : int
            Lag value
        
        Returns:
        --------
        int
            State index
        """
        expanded_name = f"{name}_{component}"
        return self.get(expanded_name, lag)
    
    def get_series_components(self, name: str, lag: int = 0) -> Dict[str, int]:
        """
        Get all component indices for a series.
        
        Parameters:
        -----------
        name : str
            Series name
        lag : int
            Lag value
        
        Returns:
        --------
        Dict[str, int]
            Mapping of component name to state index
        """
        indices = {}
        for component in self.components:
            try:
                indices[component] = self.get_component(name, component, lag)
            except KeyError:
                pass
        return indices
