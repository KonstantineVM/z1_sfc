#!/usr/bin/env python3
"""
PLACEMENT: src/models/sfc_kalman_fwtw_discrepancy.py

Complete SFC model integrating Z1, FWTW bilateral data, and sector discrepancies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import sparse
from dataclasses import dataclass

from .sfc_kalman_with_discrepancy import SFCKalmanWithDiscrepancy

from .godley_flow_derivation import GodleyFlowDerivation

logger = logging.getLogger(__name__)


@dataclass
class BilateralPosition:
    """Represents a FWTW bilateral position."""
    holder: str
    issuer: str
    instrument: str
    series_code: str  # Constructed Z1-style code
    value: float
    period: str


class SFCKalmanFWTWDiscrepancy(SFCKalmanWithDiscrepancy):
    """
    Complete SFC model with Z1 + FWTW + Discrepancy handling.
    
    This model:
    1. Uses Z1 aggregates as primary observations
    2. Adds FWTW bilateral positions as additional observations
    3. Models discrepancies to handle Godley identity violations
    4. Enforces bilateral consistency constraints
    """
    
    def __init__(self, 
                 z1_data: pd.DataFrame,
                 fwtw_data: pd.DataFrame,
                 sectors: List[str],
                 instruments: Dict[str, str],
                 bilateral_weight: float = 0.8,
                 **kwargs):
        """
        Initialize complete SFC model.
        
        Parameters:
        -----------
        z1_data : pd.DataFrame
            Z1 time series data
        fwtw_data : pd.DataFrame
            FWTW bilateral positions with columns:
            - Holder Code, Issuer Code, Instrument Code
            - Date, Level
        sectors : List[str]
            Sector codes to model
        instruments : Dict[str, str]
            Instrument code to name mapping
        bilateral_weight : float
            Weight for bilateral consistency constraints (0-1)
        """
        self.fwtw_data = fwtw_data
        self.instruments = instruments
        self.bilateral_weight = bilateral_weight
        
        # Parse FWTW data into bilateral positions
        self.bilateral_positions = self._parse_fwtw_data(fwtw_data)
        
        # Create extended dataset combining Z1 and FWTW
        self.combined_data = self._combine_z1_fwtw(z1_data, self.bilateral_positions)
        
        # Build issuer/holder cache from FWTW data
        self._build_issuer_cache()
        
        # Validate issuer-holder relationships
        validation = self.validate_issuer_holder_relationships()
        if validation['errors']:
            logger.error(f"Issuer-holder validation errors: {validation['errors']}")
        if validation['warnings']:
            logger.warning(f"Issuer-holder validation warnings: {validation['warnings'][:5]}")
        
        logger.info(f"Identified {len(self._issuer_cache)} issuer relationships from FWTW data")

        # Validate series names
        if not self._validate_series_names():
            logger.warning("Series name validation failed - some constraints may be incomplete")

        # Initialize parent with combined data
        super().__init__(
            data=self.combined_data,
            sectors=sectors,
            **kwargs
        )
    
    def _parse_fwtw_data(self, fwtw_df: pd.DataFrame) -> Dict[str, List[BilateralPosition]]:
        """Parse FWTW data into structured bilateral positions."""
        positions = {}
        
        for _, row in fwtw_df.iterrows():
            # Create Z1-style series code for this bilateral position
            # Format: FB{holder}{issuer}{instrument}05.Q (digit 8=0, digit 9=5)
            series_code = f"FB{row['Holder Code']}{row['Issuer Code']}{row['Instrument Code']}05.Q"
            
            pos = BilateralPosition(
                holder=row['Holder Code'],
                issuer=row['Issuer Code'],
                instrument=row['Instrument Code'],
                series_code=series_code,
                value=row['Level'],
                period=row['Date']
            )
            
            if row['Date'] not in positions:
                positions[row['Date']] = []
            positions[row['Date']].append(pos)
        
        logger.info(f"Parsed {len(fwtw_df)} FWTW records into bilateral positions")
        return positions
    
    def _combine_z1_fwtw(self, z1_data: pd.DataFrame, 
                         bilateral_positions: Dict) -> pd.DataFrame:
        """
        Combine Z1 and FWTW data into unified dataset.
        """
        combined = z1_data.copy()
        
        # Create time series for each unique bilateral position
        bilateral_series = {}
        
        for period, positions in bilateral_positions.items():
            for pos in positions:
                if pos.series_code not in bilateral_series:
                    bilateral_series[pos.series_code] = pd.Series(
                        index=z1_data.index,
                        dtype=float
                    )
                
                # Convert period to match index type
                if isinstance(z1_data.index, pd.PeriodIndex):
                    period_key = pd.Period(period, freq='Q')
                else:
                    period_key = pd.to_datetime(period)
                
                # Find matching time period
                if period_key in z1_data.index:
                    bilateral_series[pos.series_code].loc[period_key] = pos.value
        
        # Add bilateral series to combined dataset
        for series_code, series_data in bilateral_series.items():
            combined[series_code] = series_data
        
        logger.info(f"Combined dataset: {len(combined.columns)} series "
                   f"({len(z1_data.columns)} Z1 + {len(bilateral_series)} FWTW)")
        
        return combined
    
    def _validate_series_names(self):
        """
        Validate that all series in combined_data can be found in StateIndex.
        Log warnings for any mismatches.
        """
        missing_series = []
        
        for series in self.combined_data.columns:
            if series not in self.state_index.series_names:
                missing_series.append(series)
        
        if missing_series:
            logger.warning(f"Found {len(missing_series)} series not in StateIndex:")
            for series in missing_series[:10]:  # Show first 10
                logger.warning(f"  - {series}")
            
            # Check for format issues
            format_issues = []
            for series in missing_series:
                if series.endswith('5.Q'):  # Wrong format
                    correct = series.replace('5.Q', '05.Q')
                    if correct in self.state_index.series_names:
                        format_issues.append((series, correct))
            
            if format_issues:
                logger.error("Series format issues detected:")
                for wrong, correct in format_issues[:5]:
                    logger.error(f"  {wrong} should be {correct}")
        
        return len(missing_series) == 0

    def build_bilateral_consistency_constraints(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build constraints ensuring Z1 aggregates = Sum(FWTW bilaterals).
        Both use FL (stock) series since FWTW positions are stocks.
        
        For each sector-instrument pair:
        - As holder: FL{sector}{instrument}05.Q = Σ FB{sector}{issuer}{instrument}05.Q
        - As issuer: FL{sector}{instrument}05.Q = Σ FB{holder}{sector}{instrument}05.Q
        """
        rows, cols, data = [], [], []
        rhs = []
        row = 0
        n_states = self.state_index.size
        
        # Build constraints for each sector-instrument combination
        for sector in self.sectors:
            for inst_code in self.instruments.keys():
                
                # --- HOLDER SIDE CONSTRAINT ---
                # FL aggregate for this sector holding this instrument
                fl_holder_series = f"FL{sector}{inst_code}05.Q"
                
                if fl_holder_series in self.state_index.series_names:
                    # Add FL series to constraint (LHS)
                    fl_idx = self.state_index.get(fl_holder_series, 0)
                    rows.append(row)
                    cols.append(fl_idx)
                    data.append(1.0)
                    
                    # Find all FB positions where this sector is holder
                    fb_found = False
                    for col in self.combined_data.columns:
                        if not col.startswith('FB') or len(col) < 16:
                            continue
                        
                        # Parse FB series: FB{holder}{issuer}{instrument}05.Q
                        holder = col[2:4]
                        issuer = col[4:6]
                        instrument = col[6:11]
                        
                        # Check if this FB contributes to our holder constraint
                        if holder == sector and instrument == inst_code:
                            if col in self.state_index.series_names:
                                fb_idx = self.state_index.get(col, 0)
                                rows.append(row)
                                cols.append(fb_idx)
                                data.append(-1.0)  # Subtract FB from FL
                                fb_found = True
                    
                    # Only keep constraint if we found FB positions
                    if fb_found:
                        rhs.append(0.0)
                        row += 1
                    else:
                        # Remove the FL entry we added
                        rows.pop()
                        cols.pop()
                        data.pop()
                
                # --- ISSUER SIDE CONSTRAINT ---
                # FL aggregate for this sector issuing this instrument (liability)
                # Note: Some sectors can't issue certain instruments
                if self._can_issue(sector, inst_code):
                    fl_issuer_series = f"FL{sector}{inst_code}05.Q"
                    
                    if fl_issuer_series in self.state_index.series_names:
                        # Add FL series to constraint (LHS)
                        fl_idx = self.state_index.get(fl_issuer_series, 0)
                        rows.append(row)
                        cols.append(fl_idx)
                        data.append(1.0)
                        
                        # Find all FB positions where this sector is issuer
                        fb_found = False
                        for col in self.combined_data.columns:
                            if not col.startswith('FB') or len(col) < 16:
                                continue
                            
                            holder = col[2:4]
                            issuer = col[4:6]
                            instrument = col[6:11]
                            
                            # Check if this FB contributes to our issuer constraint
                            if issuer == sector and instrument == inst_code:
                                if col in self.state_index.series_names:
                                    fb_idx = self.state_index.get(col, 0)
                                    rows.append(row)
                                    cols.append(fb_idx)
                                    data.append(-1.0)  # Subtract FB from FL
                                    fb_found = True
                        
                        # Only keep constraint if we found FB positions
                        if fb_found:
                            rhs.append(0.0)
                            row += 1
                        else:
                            # Remove the FL entry we added
                            rows.pop()
                            cols.pop()
                            data.pop()
        
        # Convert to sparse matrix
        if row == 0:
            return sparse.csr_matrix((0, n_states)), np.array([])
        
        # Build sparse matrix efficiently
        A = sparse.coo_matrix(
            (np.array(data), (np.array(rows), np.array(cols))),
            shape=(row, n_states)
        ).tocsr()
        
        # Apply bilateral weight if specified
        if self.bilateral_weight not in (None, 1.0):
            A = A.multiply(self.bilateral_weight)
        
        logger.debug(f"Built {row} bilateral consistency constraints")
        return A, np.array(rhs)

    def _can_issue(self, sector: str, instrument: str) -> bool:
        """
        Check if a sector can issue a given instrument based on FWTW data.
        
        A sector is an issuer if it appears as issuer in any FB position for this instrument.
        """
        # Check if we've already built the issuer cache
        if not hasattr(self, '_issuer_cache'):
            self._build_issuer_cache()
        
        # Look up in cache
        key = f"{sector}_{instrument}"
        return self._issuer_cache.get(key, False)

    def _build_issuer_cache(self):
        """
        Build cache of which sectors issue which instruments based on FWTW data.
        """
        self._issuer_cache = {}
        self._holder_cache = {}
        
        # Scan all FB series in the data
        for col in self.combined_data.columns:
            if col.startswith('FB') and len(col) >= 16:
                # Parse FB series: FB{holder}{issuer}{instrument}05.Q
                holder = col[2:4]
                issuer = col[4:6]
                instrument = col[6:11]
                
                # Check if there's actual data (non-zero positions)
                if self.combined_data[col].abs().sum() > 0:
                    # Record that this issuer issues this instrument
                    issuer_key = f"{issuer}_{instrument}"
                    self._issuer_cache[issuer_key] = True
                    
                    # Record that this holder holds this instrument
                    holder_key = f"{holder}_{instrument}"
                    self._holder_cache[holder_key] = True
        
        # Log what we found
        logger.info("Built issuer/holder cache from FWTW data:")
        
        # Count issuers per instrument
        issuer_counts = {}
        for key in self._issuer_cache.keys():
            sector, inst = key.split('_')
            if inst not in issuer_counts:
                issuer_counts[inst] = []
            issuer_counts[inst].append(sector)
        
        for inst, issuers in issuer_counts.items():
            logger.debug(f"  Instrument {inst}: issued by sectors {issuers}")

    def _is_asset(self, sector: str, instrument: str) -> bool:
        """
        Determine if a position is an asset for the given sector based on FWTW data.
        
        Logic:
        - If sector appears only as holder (never issuer) for this instrument -> Asset
        - If sector appears only as issuer (never holder) for this instrument -> Liability
        - If sector appears as both -> Need to check net position or use convention
        """
        # Build cache if needed
        if not hasattr(self, '_issuer_cache'):
            self._build_issuer_cache()
        
        is_issuer = self._issuer_cache.get(f"{sector}_{instrument}", False)
        is_holder = self._holder_cache.get(f"{sector}_{instrument}", False)
        
        if is_holder and not is_issuer:
            # Only holds, never issues -> Asset
            return True
        elif is_issuer and not is_holder:
            # Only issues, never holds -> Liability
            return False
        elif is_holder and is_issuer:
            # Both holds and issues (e.g., banks with interbank loans)
            # Need to check net position
            return self._check_net_position(sector, instrument)
        else:
            # Not in FWTW data at all - check Z1 data
            return self._check_z1_position(sector, instrument)

    def _check_net_position(self, sector: str, instrument: str) -> bool:
        """
        For sectors that both hold and issue an instrument, check net position.
        """
        # Sum all FB positions where sector is holder
        holder_sum = 0
        for col in self.combined_data.columns:
            if col.startswith('FB') and len(col) >= 16:
                holder = col[2:4]
                issuer = col[4:6]
                inst = col[6:11]
                
                if holder == sector and inst == instrument:
                    # Sector holds this
                    holder_sum += self.combined_data[col].abs().mean()
        
        # Sum all FB positions where sector is issuer
        issuer_sum = 0
        for col in self.combined_data.columns:
            if col.startswith('FB') and len(col) >= 16:
                holder = col[2:4]
                issuer = col[4:6]
                inst = col[6:11]
                
                if issuer == sector and inst == instrument:
                    # Sector issued this
                    issuer_sum += self.combined_data[col].abs().mean()
        
        # If holds more than issues -> net asset
        return holder_sum > issuer_sum

    def _check_z1_position(self, sector: str, instrument: str) -> bool:
        """
        For positions not in FWTW, check Z1 FL series to determine asset/liability.
        """
        fl_series = f"FL{sector}{instrument}05.Q"
        
        if fl_series in self.combined_data.columns:
            # Check if typically positive (asset) or negative (liability)
            mean_value = self.combined_data[fl_series].mean()
            
            # In Z1, assets are typically positive values
            return mean_value > 0
        
        # Default: assume asset if no information
        return True

    def get_issuers_for_instrument(self, instrument: str) -> List[str]:
        """
        Get list of sectors that issue a given instrument based on FWTW data.
        """
        if not hasattr(self, '_issuer_cache'):
            self._build_issuer_cache()
        
        issuers = []
        for sector in self.sectors:
            if self._issuer_cache.get(f"{sector}_{instrument}", False):
                issuers.append(sector)
        
        return issuers

    def get_holders_for_instrument(self, instrument: str) -> List[str]:
        """
        Get list of sectors that hold a given instrument based on FWTW data.
        """
        if not hasattr(self, '_issuer_cache'):
            self._build_issuer_cache()
        
        holders = []
        for sector in self.sectors:
            if self._holder_cache.get(f"{sector}_{instrument}", False):
                holders.append(sector)
        
        return holders

    def validate_issuer_holder_relationships(self) -> Dict:
        """
        Validate that issuer-holder relationships make economic sense.
        """
        if not hasattr(self, '_issuer_cache'):
            self._build_issuer_cache()
        
        validation = {
            'valid': [],
            'warnings': [],
            'errors': []
        }
        
        for instrument in self.instruments.keys():
            issuers = self.get_issuers_for_instrument(instrument)
            holders = self.get_holders_for_instrument(instrument)
            
            # Check: Every instrument should have at least one issuer
            if len(issuers) == 0:
                validation['warnings'].append(f"Instrument {instrument} has no issuers in FWTW data")
            
            # Check: Every instrument should have at least one holder
            if len(holders) == 0:
                validation['warnings'].append(f"Instrument {instrument} has no holders in FWTW data")
            
            # Check: For each issued instrument, there should be holders
            if len(issuers) > 0 and len(holders) == 0:
                validation['errors'].append(f"Instrument {instrument} is issued but has no holders!")
            
            # Special checks for known instrument types
            if instrument == '31611':  # Treasuries
                if '31' not in issuers:
                    validation['warnings'].append("Federal government (31) not issuing Treasuries in FWTW")
            
            if instrument in ['30000', '30200']:  # Deposits
                if '70' not in issuers:
                    validation['warnings'].append(f"Banks (70) not issuing deposits ({instrument}) in FWTW")
            
            validation['valid'].append({
                'instrument': instrument,
                'issuers': issuers,
                'holders': holders
            })
        
        return validation
    
    def build_market_clearing_from_aggregates(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build market clearing constraints from Z1 aggregates.
        
        For each instrument:
        Σ(FL assets across sectors) = Σ(FL liabilities across sectors)
        
        This uses FL (stock) series for market clearing.
        """
        rows, cols, data = [], [], []
        rhs = []
        row = 0
        n_states = self.state_index.size
        
        for inst_code in self.instruments.keys():
            constraint_has_data = False
            
            for sector in self.sectors:
                # FL series for this sector-instrument
                fl_series = f"FL{sector}{inst_code}05.Q"
                
                if fl_series in self.state_index.series_names:
                    fl_idx = self.state_index.get(fl_series, 0)
                    
                    # Determine if this is asset or liability
                    if self._is_asset(sector, inst_code):
                        # Asset: positive contribution
                        rows.append(row)
                        cols.append(fl_idx)
                        data.append(1.0)
                        constraint_has_data = True
                    elif self._can_issue(sector, inst_code):
                        # Liability: negative contribution
                        rows.append(row)
                        cols.append(fl_idx)
                        data.append(-1.0)
                        constraint_has_data = True
            
            # Only add constraint if we have data
            if constraint_has_data:
                rhs.append(0.0)  # Market clearing: assets - liabilities = 0
                row += 1
        
        # Convert to sparse matrix
        if row == 0:
            return sparse.csr_matrix((0, n_states)), np.array([])
        
        A = sparse.coo_matrix(
            (np.array(data), (np.array(rows), np.array(cols))),
            shape=(row, n_states)
        ).tocsr()
        
        logger.debug(f"Built {row} market clearing constraints")
        return A, np.array(rhs)

    def build_godley_matrix(self, t: int) -> pd.DataFrame:
        """
        Build full Godley matrix using both Z1 and FWTW data.
        
        Returns a proper balance sheet matrix with:
        - Rows: Instruments (with bilateral detail)
        - Columns: Sectors
        - Values: Positions (+ for assets, - for liabilities)
        """
        # Initialize matrix
        matrix_data = {}
        
        for sector in self.sectors:
            sector_data = {}
            
            for inst_code, inst_name in self.instruments.items():
                # Get Z1 aggregate
                z1_series = f"FL{sector}{inst_code}5.Q"
                if z1_series in self.combined_data.columns:
                    z1_value = self.combined_data.loc[
                        self.combined_data.index[t], z1_series
                    ]
                else:
                    z1_value = 0
                
                # Get FWTW bilateral breakdown
                bilateral_breakdown = {}
                
                for col in self.combined_data.columns:
                    if col.startswith('FB'):
                        holder = col[2:4]
                        issuer = col[4:6]
                        instrument = col[6:11]
                        
                        if instrument == inst_code:
                            if holder == sector:
                                # Asset position
                                counterparty = issuer
                                value = self.combined_data.loc[
                                    self.combined_data.index[t], col
                                ]
                                bilateral_breakdown[f"from_{counterparty}"] = value
                            elif issuer == sector:
                                # Liability position
                                counterparty = holder
                                value = -self.combined_data.loc[
                                    self.combined_data.index[t], col
                                ]
                                bilateral_breakdown[f"to_{counterparty}"] = value
                
                # Store both aggregate and bilateral
                sector_data[inst_name] = {
                    'z1_aggregate': z1_value,
                    'fwtw_bilateral': bilateral_breakdown,
                    'discrepancy': z1_value - sum(bilateral_breakdown.values())
                }
            
            matrix_data[f"Sector_{sector}"] = sector_data
        
        return pd.DataFrame(matrix_data)
    
    def validate_bilateral_consistency(self, filtered_states: np.ndarray) -> Dict:
        """
        Validate that Z1 aggregates = Sum(FWTW bilaterals) after filtering.
        """
        validation = {}
        
        for sector in self.sectors:
            sector_validation = {}
            
            for inst_code in self.instruments:
                # Get filtered Z1 aggregate
                z1_series = f"FL{sector}{inst_code}5.Q"
                if z1_series in self.state_index.series_names:
                    z1_idx = self.state_index.get(z1_series, 0)
                    z1_values = filtered_states[z1_idx, :]
                else:
                    continue
                
                # Sum filtered FWTW bilaterals
                bilateral_sum = np.zeros_like(z1_values)
                n_bilaterals = 0
                
                for col in self.combined_data.columns:
                    if col.startswith('FB') and inst_code in col:
                        holder = col[2:4]
                        issuer = col[4:6]
                        
                        if (holder == sector or issuer == sector) and col in self.state_index.series_names:
                            idx = self.state_index.get(col, 0)
                            bilateral_sum += filtered_states[idx, :]
                            n_bilaterals += 1
                
                if n_bilaterals > 0:
                    discrepancy = z1_values - bilateral_sum
                    sector_validation[inst_code] = {
                        'mean_discrepancy': np.mean(discrepancy),
                        'std_discrepancy': np.std(discrepancy),
                        'max_discrepancy': np.max(np.abs(discrepancy)),
                        'n_bilaterals': n_bilaterals
                    }
            
            validation[sector] = sector_validation
        
        return validation
    
    def extract_all_constraints(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray, Dict]:
        """
        Extract all constraints: SFC, Godley, Bilateral, Market Clearing.
        """
        all_A = []
        all_b = []
        metadata = {}
        
        # 1. Stock-flow constraints (from parent)
        A_sf, b_sf = self.extract_stock_flow_constraints(t)
        if A_sf.shape[0] > 0:
            all_A.append(A_sf)
            all_b.append(b_sf)
            metadata['stock_flow'] = A_sf.shape[0]
        
        # 2. Godley constraints with discrepancy
        A_god, b_god = self.build_godley_constraints(t)
        if A_god.shape[0] > 0:
            all_A.append(A_god)
            all_b.append(b_god)
            metadata['godley'] = A_god.shape[0]
        
        # 3. Bilateral consistency (Z1 = Sum(FWTW))
        A_bil, b_bil = self.build_bilateral_consistency_constraints(t)
        if A_bil.shape[0] > 0:
            all_A.append(A_bil)
            all_b.append(b_bil)
            metadata['bilateral'] = A_bil.shape[0]
        
        # 4. Market clearing from aggregates
        A_mc, b_mc = self.build_market_clearing_from_aggregates(t)
        if A_mc.shape[0] > 0:
            all_A.append(A_mc)
            all_b.append(b_mc)
            metadata['market_clearing'] = A_mc.shape[0]
        
        # 5. Bilateral flow constraints
        if self.config.get('constraints', {}).get('use_bilateral_flows', False):
            A_flow, b_flow = self.add_bilateral_flow_constraints(t)
            if A_flow.shape[0] > 0:
                all_A.append(A_flow)
                all_b.append(b_flow)
                metadata['bilateral_flows'] = A_flow.shape[0]
                logger.info(f"Added {A_flow.shape[0]} bilateral flow constraints")

        # Combine all constraints
        if all_A:
            A = sparse.vstack(all_A)
            b = np.hstack(all_b)
        else:
            A = sparse.csr_matrix((0, self.state_index.size))
            b = np.array([])
        
        logger.info(f"Total constraints at t={t}: {A.shape[0]}")
        for ctype, count in metadata.items():
            logger.info(f"  {ctype}: {count}")
        
        return A, b, metadata
        
    def derive_bilateral_flows(self, t: int) -> pd.DataFrame:
        """
        Derive bilateral transaction flows from FWTW stock changes.
        """
        if not hasattr(self, 'flow_deriver'):
            self.flow_deriver = GodleyFlowDerivation(
                z1_data=self.combined_data,
                fwtw_stocks=self._extract_fwtw_stocks(),
                sectors=self.sectors,
                instruments=self.instruments
            )
        
        return self.flow_deriver.derive_bilateral_flows(t)

    def _extract_fwtw_stocks(self) -> pd.DataFrame:
        """Extract FWTW stock series from combined data."""
        fwtw_cols = [col for col in self.combined_data.columns if col.startswith('FB')]
        return self.combined_data[fwtw_cols]

    def add_bilateral_flow_constraints(self, t: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Add constraints based on derived bilateral flows.
        
        These enforce that FWTW stock changes match derived transaction flows.
        """
        bilateral_flows = self.derive_bilateral_flows(t)
        
        rows, cols, data = [], [], []
        rhs = []
        row = 0
        n_states = self.state_index.size
        
        for _, flow in bilateral_flows.iterrows():
            if flow['confidence'] > 0.8:  # Only high-confidence flows
                fb_series = f"FB{flow['holder']}{flow['issuer']}{flow['instrument']}05.Q"
                
                if fb_series in self.state_index.series_names and t > 0:
                    # FB[t] - FB[t-1] = derived_flow
                    idx_t = self.state_index.get(fb_series, 0)
                    idx_prev = self.state_index.get(fb_series, -1)
                    
                    rows.extend([row, row])
                    cols.extend([idx_t, idx_prev])
                    data.extend([1.0, -1.0])
                    rhs.append(flow['flow'])
                    row += 1
        
        if row == 0:
            return sparse.csr_matrix((0, n_states)), np.array([])
        
        A = sparse.coo_matrix(
            (np.array(data), (np.array(rows), np.array(cols))),
            shape=(row, n_states)
        ).tocsr()
        
        # Apply confidence-based weighting
        A = A.multiply(self.bilateral_weight * 0.8)  # Additional factor for flow constraints
        
        return A, np.array(rhs)        
