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
        
        # Leaf-only flow policy support
        self.allowed_flow_bases: Set[str] = set()
        
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
    
    def set_allowed_flow_bases(self, base_series: List[str]) -> None:
        """
        Set whitelist of FL series that can have FU/FR/FV attached.
        
        This is critical for the Z1_SFC correctness: only base series
        (digit 9 = 0 or 3) should have flow constraints attached.
        Computed/aggregated series (digit 9 = 5) get their consistency
        through aggregation formulas, not direct flow attachment.
        
        Parameters:
        -----------
        base_series : List[str]
            List of base FL series codes that should have flows
        """
        self.allowed_flow_bases = set(base_series)
        logger.info(f"Set {len(self.allowed_flow_bases)} allowed flow bases")
        
        # Log some examples for verification
        if self.allowed_flow_bases:
            examples = list(self.allowed_flow_bases)[:5]
            logger.debug(f"Example allowed flow bases: {examples}")
    
    def is_flow_allowed(self, fl_series: str) -> bool:
        """
        Check if a FL series is allowed to have FU/FR/FV flows attached.
        
        Parameters:
        -----------
        fl_series : str
            FL series code
            
        Returns:
        --------
        bool
            True if flows can be attached (base series), False otherwise
        """
        if not self.allowed_flow_bases:
            # If no whitelist set, allow all (backward compatibility)
            logger.warning("No flow base whitelist set - allowing all series")
            return True
        return fl_series in self.allowed_flow_bases
    
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
            Name of the series
        
        Returns:
        --------
        Dict[int, int]
            Mapping of lag to state index
        """
        indices = {}
        for lag in range(0, -(self.max_lag + 1), -1):
            if self.has(series_name, lag):
                indices[lag] = self.get(series_name, lag)
        return indices
    
    def get_series_block(self, lag: int = 0) -> Tuple[int, int]:
        """
        Get start and end indices for all series at a specific lag.
        
        This is useful for block operations on the state vector.
        
        Parameters:
        -----------
        lag : int
            Lag value (0 for current, negative for lags)
        
        Returns:
        --------
        Tuple[int, int]
            (start_index, end_index) for the block
        """
        if lag > 0:
            raise ValueError("Lag should be 0 or negative")
        
        if lag == 0:
            return 0, self.n_series
        else:
            abs_lag = abs(lag)
            if abs_lag > self.max_lag:
                raise ValueError(f"Lag {lag} exceeds max_lag {self.max_lag}")
            start = self.n_series * abs_lag
            end = self.n_series * (abs_lag + 1)
            return start, end
    
    def build_lag_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix for lagged values.
        
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
            'total_entries': len(self.index),
            'n_flow_allowed': len(self.allowed_flow_bases)
        }
    
    def __repr__(self) -> str:
        return (f"StateIndex(n_series={self.n_series}, "
                f"max_lag={self.max_lag}, size={self.size}, "
                f"flow_allowed={len(self.allowed_flow_bases)})")
    
    def __len__(self) -> int:
        return self.size

    def apply_allowed_flow_bases(self, strict: bool = False):
        """
        Apply the allowed flow bases policy.
        
        Parameters:
        -----------
        strict : bool
            If True, raise error if non-base series would have flows
        """
        if not self.allowed_flow_bases:
            logger.warning("No allowed flow bases set - all flows will be allowed")
            return
        
        if strict:
            # Check that only allowed bases are used for flows
            for series in self.series_names:
                if series.startswith('FL') and series not in self.allowed_flow_bases:
                    # This is a computed FL - ensure no flows are attached
                    for flow_prefix in ['FU', 'FR', 'FV']:
                        flow_series = series.replace('FL', flow_prefix)
                        if flow_series in self.series_names:
                            raise ValueError(
                                f"Computed series {series} has flow {flow_series} - "
                                f"only base series should have flows"
                            )
        
        logger.info(f"Applied flow bases policy: {len(self.allowed_flow_bases)} bases allowed")

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
    
    def build_component_selection_matrix(self, component: str) -> np.ndarray:
        """
        Build selection matrix for a specific component.
        
        Parameters:
        -----------
        component : str
            Component name
        
        Returns:
        --------
        np.ndarray
            Selection matrix
        """
        if component not in self.components:
            raise ValueError(f"Component {component} not in {self.components}")
        
        comp_idx = self.components.index(component)
        n_original = len(self.original_series)
        
        # Selection matrix to extract component
        S = np.zeros((n_original, self.n_series))
        for i, series in enumerate(self.original_series):
            # Map to expanded index
            expanded_idx = i * self.n_components + comp_idx
            S[i, expanded_idx] = 1.0
        
        return S
    
    def build_component_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix for component model.
        
        Handles level-trend decomposition:
        - Level[t] = Level[t-1] + Trend[t-1]
        - Trend[t] = Trend[t-1]
        
        Returns:
        --------
        np.ndarray
            Transition matrix
        """
        T = np.zeros((self.size, self.size))
        
        # Current period dynamics
        for i, series in enumerate(self.original_series):
            if 'level' in self.components and 'trend' in self.components:
                level_idx = self.get_component(series, 'level', 0)
                trend_idx = self.get_component(series, 'trend', 0)
                
                # Level[t] = Level[t-1] + Trend[t-1]
                if self.max_lag >= 1:
                    level_lag_idx = self.get_component(series, 'level', -1)
                    trend_lag_idx = self.get_component(series, 'trend', -1)
                    T[level_idx, level_lag_idx] = 1.0
                    T[level_idx, trend_lag_idx] = 1.0
                
                # Trend[t] = Trend[t-1]
                if self.max_lag >= 1:
                    trend_lag_idx = self.get_component(series, 'trend', -1)
                    T[trend_idx, trend_lag_idx] = 1.0
        
        # Lag shifting (similar to parent class)
        if self.max_lag > 1:
            for lag in range(1, self.max_lag):
                source_start, source_end = self.get_series_block(-lag)
                target_start, target_end = self.get_series_block(-(lag + 1))
                T[target_start:target_end, source_start:source_end] = np.eye(
                    source_end - source_start
                )
        
        return T
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """Get statistics about component structure."""
        stats = self.get_statistics()
        stats.update({
            'n_original_series': len(self.original_series),
            'n_components': self.n_components,
            'components': self.components,
            'expanded_series': self.n_series // (self.max_lag + 1)
        })
        return stats


class AugmentedStateIndex(StateIndex):
    """
    State index with additional augmentation for auxiliary variables.
    
    This supports adding auxiliary state variables like:
    - Latent factors
    - Unobserved components
    - Auxiliary AR processes
    """
    
    def __init__(self, series_names: List[str], 
                 max_lag: int = 2,
                 n_factors: int = 0,
                 n_auxiliary: int = 0):
        """
        Initialize augmented state index.
        
        Parameters:
        -----------
        series_names : List[str]
            Observable series names
        max_lag : int
            Maximum lag for observables
        n_factors : int
            Number of latent factors
        n_auxiliary : int
            Number of auxiliary state variables
        """
        # Store augmentation dimensions
        self.n_factors = n_factors
        self.n_auxiliary = n_auxiliary
        
        # Create names for augmented variables
        augmented_names = list(series_names)
        
        # Add factor names
        for i in range(n_factors):
            augmented_names.append(f"__factor_{i}")
        
        # Add auxiliary names
        for i in range(n_auxiliary):
            augmented_names.append(f"__aux_{i}")
        
        # Initialize with augmented series list
        super().__init__(augmented_names, max_lag)
        
        # Store original series info
        self.n_observable = len(series_names)
        self.observable_names = series_names
        
    def get_factor_indices(self, lag: int = 0) -> List[int]:
        """
        Get indices for factor variables.
        
        Parameters:
        -----------
        lag : int
            Lag value
        
        Returns:
        --------
        List[int]
            Factor state indices
        """
        indices = []
        for i in range(self.n_factors):
            factor_name = f"__factor_{i}"
            if self.has(factor_name, lag):
                indices.append(self.get(factor_name, lag))
        return indices
    
    def get_auxiliary_indices(self, lag: int = 0) -> List[int]:
        """
        Get indices for auxiliary variables.
        
        Parameters:
        -----------
        lag : int
            Lag value
        
        Returns:
        --------
        List[int]
            Auxiliary state indices
        """
        indices = []
        for i in range(self.n_auxiliary):
            aux_name = f"__aux_{i}"
            if self.has(aux_name, lag):
                indices.append(self.get(aux_name, lag))
        return indices
    
    def build_factor_loading_matrix(self, loadings: np.ndarray) -> np.ndarray:
        """
        Build loading matrix for factors.
        
        Parameters:
        -----------
        loadings : np.ndarray
            Factor loadings of shape (n_observable, n_factors)
        
        Returns:
        --------
        np.ndarray
            Full loading matrix for state vector
        """
        if loadings.shape != (self.n_observable, self.n_factors):
            raise ValueError(f"Loadings shape {loadings.shape} doesn't match "
                           f"expected ({self.n_observable}, {self.n_factors})")
        
        # Full loading matrix
        Lambda = np.zeros((self.n_observable, self.n_series))
        
        # Identity for observables
        Lambda[:self.n_observable, :self.n_observable] = np.eye(self.n_observable)
        
        # Factor loadings
        factor_start = self.n_observable
        factor_end = factor_start + self.n_factors
        Lambda[:, factor_start:factor_end] = loadings
        
        return Lambda
    
    def get_augmented_statistics(self) -> Dict[str, Any]:
        """Get statistics about augmented structure."""
        stats = self.get_statistics()
        stats.update({
            'n_observable': self.n_observable,
            'n_factors': self.n_factors,
            'n_auxiliary': self.n_auxiliary,
            'total_augmented': self.n_factors + self.n_auxiliary
        })
        return stats
