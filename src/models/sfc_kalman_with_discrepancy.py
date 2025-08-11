#!/usr/bin/env python3
"""
PLACEMENT: src/models/sfc_kalman_with_discrepancy.py

SFC Kalman Filter that explicitly models sector discrepancies as unobserved states.
This acknowledges that Godley identities don't hold exactly in measured data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import sparse

from .sfc_kalman_proper import ProperSFCKalmanFilter
from ..graph.state_index import StateIndex

logger = logging.getLogger(__name__)


class SFCKalmanWithDiscrepancy(ProperSFCKalmanFilter):
    """
    Extended SFC Kalman Filter that models sector discrepancies.
    
    Key insight: The Godley identity Saving = Net Lending doesn't hold
    in the data due to measurement errors. We model this explicitly.
    """
    
    def __init__(self, data: pd.DataFrame, 
                 sectors: List[str],
                 include_discrepancy: bool = True,
                 discrepancy_variance_ratio: float = 0.1,
                 **kwargs):
        """
        Initialize SFC model with discrepancy states.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Z1 data including FA157005005.Q (sector discrepancy)
        sectors : List[str]
            Sector codes to model discrepancies for
        include_discrepancy : bool
            Whether to include discrepancy states
        discrepancy_variance_ratio : float
            Variance of discrepancy relative to other states
        """
        self.sectors = sectors
        self.include_discrepancy = include_discrepancy
        self.discrepancy_variance_ratio = discrepancy_variance_ratio
        
        # Identify discrepancy series in data
        self.discrepancy_series = {}
        for sector in sectors:
            # Z1 discrepancy series format: FA{sector}7005005.Q
            disc_series = f"FA{sector}7005005.Q"
            if disc_series in data.columns:
                self.discrepancy_series[sector] = disc_series
                logger.info(f"Found discrepancy series for sector {sector}: {disc_series}")
        
        # Initialize parent class
        super().__init__(data, **kwargs)
    
    def _build_extended_state_space(self):
        """
        Extend state space to include discrepancy states.
        
        State vector becomes:
        [Original States | Discrepancy States | Lags]
        
        Where discrepancy states track the unobserved measurement error
        that causes Saving ≠ Net Lending.
        """
        # Get base state space
        base_states = self.state_index.series_names
        n_base = len(base_states)
        
        if not self.include_discrepancy:
            return base_states
        
        # Add discrepancy states
        extended_states = base_states.copy()
        self.discrepancy_state_indices = {}
        
        for sector, disc_series in self.discrepancy_series.items():
            # Add unobserved discrepancy state
            disc_state_name = f"UNOBSERVED_DISC_{sector}"
            extended_states.append(disc_state_name)
            self.discrepancy_state_indices[sector] = len(extended_states) - 1
            logger.info(f"Added discrepancy state for sector {sector} at index {self.discrepancy_state_indices[sector]}")
        
        # Rebuild state index with extended states
        self.state_index = StateIndex(extended_states, max_lag=self.max_lag)
        
        return extended_states
    
    def _build_observation_matrix(self):
        """
        Build observation matrix that links observed series to states.
        
        Key: Observed discrepancy = Unobserved true discrepancy + measurement error
        """
        n_obs = len(self.data.columns)
        n_states = self.state_index.size
        
        # Start with identity mapping for regular series
        Z = np.zeros((n_obs, n_states))
        
        obs_idx = 0
        for series in self.data.columns:
            if series in self.state_index.series_names:
                state_idx = self.state_index.get(series, lag=0)
                Z[obs_idx, state_idx] = 1.0
            
            # Link observed discrepancy to unobserved state
            if series in self.discrepancy_series.values():
                sector = [s for s, d in self.discrepancy_series.items() if d == series][0]
                if sector in self.discrepancy_state_indices:
                    disc_state_idx = self.discrepancy_state_indices[sector]
                    Z[obs_idx, disc_state_idx] = 1.0
            
            obs_idx += 1
        
        return sparse.csr_matrix(Z)
    
    def _build_transition_matrix(self):
        """
        Build state transition matrix with discrepancy dynamics.
        
        Discrepancy states follow AR(1) process:
        disc[t] = ρ * disc[t-1] + ε
        """
        n_states = self.state_index.size
        T = np.eye(n_states)
        
        # Set persistence for discrepancy states
        discrepancy_persistence = 0.8  # Discrepancies are persistent
        
        for sector, disc_idx in self.discrepancy_state_indices.items():
            T[disc_idx, disc_idx] = discrepancy_persistence
        
        # Handle lags for regular states
        if self.max_lag > 0:
            # Shift lag blocks
            n_current = self.state_index.n_series
            for lag in range(1, self.max_lag + 1):
                start_prev = (lag - 1) * n_current
                end_prev = lag * n_current
                start_curr = lag * n_current
                end_curr = (lag + 1) * n_current
                
                if end_curr <= n_states:
                    T[start_curr:end_curr, start_prev:end_prev] = np.eye(n_current)
        
        return sparse.csr_matrix(T)
    
    def _build_state_covariance(self):
        """
        Build state innovation covariance with discrepancy variance.
        """
        n_states = self.state_index.size
        Q = np.eye(n_states) * self.state_variance
        
        # Set variance for discrepancy states
        # Higher variance = more variable discrepancies
        disc_variance = self.state_variance * self.discrepancy_variance_ratio
        
        for sector, disc_idx in self.discrepancy_state_indices.items():
            Q[disc_idx, disc_idx] = disc_variance
            logger.debug(f"Discrepancy variance for {sector}: {disc_variance}")
        
        return sparse.csr_matrix(Q)
    
    def build_godley_constraints(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Godley identity constraints that include discrepancy.
        
        For each sector:
        Saving - Investment = Net Lending + Discrepancy
        
        Or equivalently:
        Income - Expenditure - ΔFinancial_Assets + ΔLiabilities = Discrepancy
        """
        constraints = []
        rhs = []
        
        for sector in self.sectors:
            constraint = {}
            
            # Income side (if available)
            income_series = f"FA{sector}6010001.Q"  # Personal income
            if income_series in self.data.columns:
                idx = self.state_index.get(income_series, 0)
                constraint[idx] = 1.0
            
            # Expenditure side
            expenditure_series = f"FA{sector}6900005.Q"  # Personal outlays
            if expenditure_series in self.data.columns:
                idx = self.state_index.get(expenditure_series, 0)
                constraint[idx] = -1.0
            
            # Financial assets
            assets_series = f"FA{sector}4090005.Q"  # Total financial assets
            if assets_series in self.data.columns:
                idx = self.state_index.get(assets_series, 0)
                # Current minus lagged = change
                constraint[idx] = -1.0
                if t > 0:
                    idx_lag = self.state_index.get(assets_series, -1)
                    constraint[idx_lag] = 1.0
            
            # Liabilities
            liab_series = f"FA{sector}4190005.Q"  # Total liabilities
            if liab_series in self.data.columns:
                idx = self.state_index.get(liab_series, 0)
                constraint[idx] = 1.0
                if t > 0:
                    idx_lag = self.state_index.get(liab_series, -1)
                    constraint[idx_lag] = -1.0
            
            # Discrepancy state
            if sector in self.discrepancy_state_indices:
                disc_idx = self.discrepancy_state_indices[sector]
                constraint[disc_idx] = -1.0  # Absorbs the imbalance
            
            if constraint:
                constraints.append(constraint)
                rhs.append(0.0)
        
        # Convert to sparse matrix efficiently
        if constraints:
            rows, cols, data = [], [], []
            
            for i, constraint in enumerate(constraints):
                for idx, coef in constraint.items():
                    rows.append(i)
                    cols.append(idx)
                    data.append(coef)
            
            n_constraints = len(constraints)
            n_states = self.state_index.size
            
            A = sparse.coo_matrix(
                (np.array(data), (np.array(rows), np.array(cols))),
                shape=(n_constraints, n_states)
            ).tocsr()
            
            return A, np.array(rhs)
        
        return sparse.csr_matrix((0, self.state_index.size)), np.array([])
    
    def extract_discrepancy_estimates(self, filtered_states: np.ndarray) -> pd.DataFrame:
        """
        Extract estimated discrepancies from filtered states.
        
        Returns:
        --------
        pd.DataFrame
            Estimated true discrepancies for each sector over time
        """
        discrepancies = {}
        
        for sector, disc_idx in self.discrepancy_state_indices.items():
            discrepancies[f"Discrepancy_{sector}"] = filtered_states[disc_idx, :]
        
        return pd.DataFrame(discrepancies, index=self.data.index)
    
    def validate_godley_identities(self, filtered_states: np.ndarray) -> Dict:
        """
        Check how well Godley identities hold with estimated discrepancies.
        """
        validation = {}
        
        for sector in self.sectors:
            # Get relevant series
            income = self._get_series_values(f"FA{sector}6010001.Q", filtered_states)
            expenditure = self._get_series_values(f"FA{sector}6900005.Q", filtered_states)
            assets = self._get_series_values(f"FA{sector}4090005.Q", filtered_states)
            liabilities = self._get_series_values(f"FA{sector}4190005.Q", filtered_states)
            
            # Calculate components
            if income is not None and expenditure is not None:
                saving = income - expenditure
            else:
                saving = None
            
            if assets is not None and liabilities is not None:
                net_lending = np.diff(assets) - np.diff(liabilities)
            else:
                net_lending = None
            
            # Get discrepancy
            if sector in self.discrepancy_state_indices:
                disc_idx = self.discrepancy_state_indices[sector]
                discrepancy = filtered_states[disc_idx, 1:]  # Skip first period for diff
            else:
                discrepancy = None
            
            # Check identity: Saving = Net Lending + Discrepancy
            if saving is not None and net_lending is not None and discrepancy is not None:
                identity_error = saving[1:] - net_lending - discrepancy
                validation[sector] = {
                    'mean_error': np.mean(identity_error),
                    'std_error': np.std(identity_error),
                    'max_error': np.max(np.abs(identity_error)),
                    'identity_satisfied': np.max(np.abs(identity_error)) < 1e-6
                }
            else:
                validation[sector] = {
                    'mean_error': np.nan,
                    'std_error': np.nan,
                    'max_error': np.nan,
                    'identity_satisfied': False
                }
        
        return validation
    
    def _get_series_values(self, series: str, filtered_states: np.ndarray) -> Optional[np.ndarray]:
        """Helper to extract series values from filtered states."""
        if series in self.data.columns and series in self.state_index.series_names:
            idx = self.state_index.get(series, 0)
            return filtered_states[idx, :]
        return None
