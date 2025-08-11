# FILE: src/models/godley_flow_derivation.py

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GodleyFlowDerivation:
    """
    Derive Godley transaction flow matrices from FWTW stock differences.
    
    Key idea: FWTW stock changes reveal actual bilateral flows between sectors.
    This gives us the full "who-transacted-with-whom" matrix, not just net positions.
    """
    
    def __init__(self, 
                 z1_data: pd.DataFrame,
                 fwtw_stocks: pd.DataFrame,
                 sectors: List[str],
                 instruments: Dict[str, dict]):
        """
        Initialize flow derivation.
        
        Parameters:
        -----------
        z1_data : pd.DataFrame
            Z1 time series including FU (transaction) series
        fwtw_stocks : pd.DataFrame
            FWTW bilateral stock positions over time
        sectors : List[str]
            Sector codes
        instruments : Dict[str, dict]
            Instrument configuration with revaluation flags
        """
        self.z1_data = z1_data
        self.fwtw_stocks = fwtw_stocks
        self.sectors = sectors
        self.instruments = instruments
        
        # Classify instruments by revaluation risk
        self._classify_instruments()
    
    def _classify_instruments(self):
        """Classify instruments by whether they have significant revaluations."""
        self.low_reval_instruments = []  # Deposits, loans - no price changes
        self.high_reval_instruments = []  # Equities, bonds - significant price changes
        
        for inst_code, inst_info in self.instruments.items():
            inst_type = inst_info.get('type', '')
            
            if inst_type in ['deposit', 'loan', 'currency']:
                self.low_reval_instruments.append(inst_code)
            elif inst_type in ['equity', 'bond', 'mutual_fund']:
                self.high_reval_instruments.append(inst_code)
            else:
                # Default: assume some revaluation
                self.high_reval_instruments.append(inst_code)
        
        logger.info(f"Low revaluation instruments: {len(self.low_reval_instruments)}")
        logger.info(f"High revaluation instruments: {len(self.high_reval_instruments)}")
    
    def derive_bilateral_flows(self, t: int) -> pd.DataFrame:
        """
        Derive bilateral transaction flows from FWTW stock differences.
        
        Returns:
        --------
        pd.DataFrame
            Bilateral flow matrix with MultiIndex (holder, issuer, instrument)
        """
        flows = []
        
        for col in self.fwtw_stocks.columns:
            if not col.startswith('FB'):
                continue
            
            # Parse bilateral position code
            holder = col[2:4]
            issuer = col[4:6]
            instrument = col[6:11]
            
            if t > 0:
                # Calculate stock change
                stock_t = self.fwtw_stocks.iloc[t][col]
                stock_prev = self.fwtw_stocks.iloc[t-1][col]
                
                if pd.notna(stock_t) and pd.notna(stock_prev):
                    delta_stock = stock_t - stock_prev
                    
                    # Estimate transaction component
                    if instrument in self.low_reval_instruments:
                        # For deposits/loans: Δstock ≈ transaction
                        transaction_flow = delta_stock
                        confidence = 0.95  # High confidence
                    else:
                        # For securities: need to separate transaction from revaluation
                        transaction_flow = self._separate_transaction_from_reval(
                            delta_stock, holder, issuer, instrument, t
                        )
                        confidence = 0.70  # Lower confidence
                    
                    flows.append({
                        'holder': holder,
                        'issuer': issuer,
                        'instrument': instrument,
                        'flow': transaction_flow,
                        'confidence': confidence,
                        'period': self.fwtw_stocks.index[t]
                    })
        
        return pd.DataFrame(flows)
    
    def _separate_transaction_from_reval(self, 
                                        delta_stock: float,
                                        holder: str,
                                        issuer: str,
                                        instrument: str,
                                        t: int) -> float:
        """
        Separate transaction flow from revaluation in stock change.
        
        Uses multiple approaches:
        1. If FU series available: use as prior
        2. If price data available: calculate revaluation, remainder is transaction
        3. Statistical: use historical flow/reval ratio
        """
        
        # Approach 1: Use FU series if available (holder's view)
        fu_series = f"FU{holder}{instrument}05.Q"
        if fu_series in self.z1_data.columns:
            fu_value = self.z1_data.iloc[t][fu_series]
            if pd.notna(fu_value):
                # FU gives us aggregate transactions, distribute proportionally
                return self._distribute_fu_to_bilateral(
                    fu_value, holder, issuer, instrument, t
                )
        
        # Approach 2: Use revaluation series if available
        fr_series = f"FR{holder}{instrument}05.Q"
        if fr_series in self.z1_data.columns:
            fr_value = self.z1_data.iloc[t][fr_series]
            if pd.notna(fr_value):
                # Distribute revaluation proportionally to holdings
                bilateral_reval = self._distribute_fr_to_bilateral(
                    fr_value, holder, issuer, instrument, t
                )
                # Transaction = Total change - Revaluation
                return delta_stock - bilateral_reval
        
        # Approach 3: Statistical decomposition
        # Assume revaluation is proportional to previous period stock
        if instrument in self.high_reval_instruments:
            # Estimate revaluation as % of previous stock
            avg_return = self._get_instrument_return(instrument, t)
            prev_stock = self.fwtw_stocks.iloc[t-1][f"FB{holder}{issuer}{instrument}05.Q"]
            estimated_reval = prev_stock * avg_return
            return delta_stock - estimated_reval
        
        # Default: assume all is transaction (for low-reval instruments)
        return delta_stock
    
    def build_godley_flow_matrix(self, t: int) -> pd.DataFrame:
        """
        Build complete Godley transaction flow matrix.
        
        This is the actual "who-transacted-with-whom" matrix that populates
        the transaction part of Godley tables.
        
        Returns:
        --------
        pd.DataFrame
            Matrix with sectors as rows/columns, instruments as layers
        """
        # Get bilateral flows
        bilateral_flows = self.derive_bilateral_flows(t)
        
        # Initialize 3D matrix: [instrument][holder_sector][issuer_sector]
        flow_matrices = {}
        
        for inst_code in self.instruments.keys():
            # Create sector x sector matrix for this instrument
            matrix = pd.DataFrame(
                0.0,
                index=self.sectors,
                columns=self.sectors
            )
            
            # Fill with bilateral flows
            inst_flows = bilateral_flows[bilateral_flows['instrument'] == inst_code]
            
            for _, flow in inst_flows.iterrows():
                if flow['holder'] in self.sectors and flow['issuer'] in self.sectors:
                    # Flow from issuer to holder (positive = holder acquires)
                    matrix.loc[flow['holder'], flow['issuer']] = flow['flow']
            
            flow_matrices[inst_code] = matrix
        
        return flow_matrices
    
    def validate_against_aggregates(self, t: int) -> Dict:
        """
        Validate derived bilateral flows against Z1 aggregate FU series.
        
        Returns:
        --------
        Dict
            Validation metrics for each sector-instrument
        """
        validation = {}
        bilateral_flows = self.derive_bilateral_flows(t)
        
        for sector in self.sectors:
            for inst_code in self.instruments.keys():
                # Sum bilateral flows for this sector-instrument
                # As holder (acquisitions)
                holder_flows = bilateral_flows[
                    (bilateral_flows['holder'] == sector) & 
                    (bilateral_flows['instrument'] == inst_code)
                ]['flow'].sum()
                
                # As issuer (issuance)
                issuer_flows = bilateral_flows[
                    (bilateral_flows['issuer'] == sector) & 
                    (bilateral_flows['instrument'] == inst_code)
                ]['flow'].sum()
                
                # Net flow from bilateral
                net_bilateral = holder_flows - issuer_flows
                
                # Compare to Z1 FU aggregate
                fu_series = f"FU{sector}{inst_code}05.Q"
                if fu_series in self.z1_data.columns:
                    fu_aggregate = self.z1_data.iloc[t][fu_series]
                    
                    if pd.notna(fu_aggregate):
                        discrepancy = net_bilateral - fu_aggregate
                        relative_error = abs(discrepancy / fu_aggregate) if fu_aggregate != 0 else np.inf
                        
                        validation[f"{sector}_{inst_code}"] = {
                            'bilateral_sum': net_bilateral,
                            'z1_aggregate': fu_aggregate,
                            'discrepancy': discrepancy,
                            'relative_error': relative_error,
                            'validation_passed': relative_error < 0.1  # 10% threshold
                        }
        
        return validation
    
    def enforce_consistency_constraints(self, 
                                       bilateral_flows: pd.DataFrame,
                                       t: int) -> pd.DataFrame:
        """
        Adjust bilateral flows to ensure consistency with Z1 aggregates.
        
        Uses optimization to minimally adjust bilateral flows while ensuring:
        1. Sum of bilateral flows = Z1 FU aggregates
        2. Market clearing across sectors
        3. Minimal deviation from FWTW-derived flows
        """
        from scipy.optimize import minimize
        
        # Set up optimization problem
        # Decision variables: adjusted bilateral flows
        # Objective: minimize adjustment from original flows
        # Constraints: match Z1 aggregates, market clearing
        
        n_flows = len(bilateral_flows)
        original_flows = bilateral_flows['flow'].values
        
        def objective(x):
            """Minimize squared adjustments, weighted by confidence."""
            adjustments = x - original_flows
            weights = bilateral_flows['confidence'].values
            return np.sum(weights * adjustments**2)
        
        constraints = []
        
        # Add Z1 aggregate constraints
        for sector in self.sectors:
            for inst_code in self.instruments.keys():
                fu_series = f"FU{sector}{inst_code}05.Q"
                if fu_series in self.z1_data.columns:
                    fu_target = self.z1_data.iloc[t][fu_series]
                    
                    if pd.notna(fu_target):
                        # Indices of relevant bilateral flows
                        holder_idx = bilateral_flows[
                            (bilateral_flows['holder'] == sector) & 
                            (bilateral_flows['instrument'] == inst_code)
                        ].index
                        
                        issuer_idx = bilateral_flows[
                            (bilateral_flows['issuer'] == sector) & 
                            (bilateral_flows['instrument'] == inst_code)
                        ].index
                        
                        def constraint_func(x, h_idx=holder_idx, i_idx=issuer_idx, target=fu_target):
                            return x[h_idx].sum() - x[i_idx].sum() - target
                        
                        constraints.append({
                            'type': 'eq',
                            'fun': constraint_func
                        })
        
        # Add market clearing constraints
        for inst_code in self.instruments.keys():
            inst_flows = bilateral_flows[bilateral_flows['instrument'] == inst_code].index
            
            def market_clearing(x, idx=inst_flows):
                return x[idx].sum()  # Should sum to zero
            
            constraints.append({
                'type': 'eq',
                'fun': market_clearing
            })
        
        # Solve optimization
        result = minimize(
            objective,
            original_flows,
            method='SLSQP',
            constraints=constraints
        )
        
        if result.success:
            bilateral_flows['adjusted_flow'] = result.x
            bilateral_flows['adjustment'] = result.x - original_flows
            logger.info(f"Successfully enforced consistency. Mean adjustment: {np.mean(np.abs(bilateral_flows['adjustment'])):.2f}")
        else:
            logger.warning("Could not fully enforce consistency")
            bilateral_flows['adjusted_flow'] = original_flows
            bilateral_flows['adjustment'] = 0
        
        return bilateral_flows
    
    def create_full_godley_accounts(self, t: int) -> Dict:
        """
        Create complete Godley accounting matrices with derived flows.
        
        Returns:
        --------
        Dict containing:
        - balance_sheet: Stock matrix
        - transactions: Flow matrix (from FWTW)
        - revaluations: Revaluation matrix
        - validation: Consistency metrics
        """
        
        # Get bilateral flows
        bilateral_flows = self.derive_bilateral_flows(t)
        
        # Enforce consistency with Z1
        adjusted_flows = self.enforce_consistency_constraints(bilateral_flows, t)
        
        # Build matrices
        flow_matrices = self.build_godley_flow_matrix(t)
        
        # Build balance sheet (stocks)
        balance_sheet = self._build_balance_sheet_matrix(t)
        
        # Separate revaluations
        reval_matrix = self._build_revaluation_matrix(t)
        
        # Validate stock-flow consistency
        validation = self._validate_stock_flow_consistency(
            balance_sheet, flow_matrices, reval_matrix, t
        )
        
        return {
            'balance_sheet': balance_sheet,
            'transactions': flow_matrices,
            'revaluations': reval_matrix,
            'bilateral_flows': adjusted_flows,
            'validation': validation
        }
