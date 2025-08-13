#!/usr/bin/env python3
"""
SFC Integrated Analyzer - Complete Pipeline
===========================================
Combines matrix generation (from sfc_core.py) with analysis (from sfc_analyzer_final.py)
in a single efficient process without subprocess calls.

- Part 1: Load Z1 data once, generate all SFC matrices (exact sfc_core.py logic)
- Part 2: Analyze generated matrices with crisis indicators (exact sfc_analyzer_final.py logic)

Usage:
    python sfc_integrated_analyzer.py --start 1965 --end 2024 --workers 8
    python sfc_integrated_analyzer.py --start 2020 --end 2024 --roles-map mappings/instrument_roles.yaml

Part 1: Matrix Generation (EXACT sfc_core.py logic)

Uses the exact same parsing patterns
Uses the exact same sfc_sign() function
Uses the exact same matrix building functions
Uses the exact same macro flow addition logic
Uses the exact same reconciliation calculation
Saves with the exact same format (instrument padding, etc.)

Part 2: Analysis (EXACT sfc_analyzer_final.py logic)

Uses the exact same role indexing system
Uses the exact same indicator calculations
Uses the exact same composite score weighting
Uses the exact same normalization approach

Key Efficiency:

Loads Z1 data ONCE at the start
Generates all matrices using in-memory data
Then analyzes all generated matrices
No subprocess calls, no repeated data loading

Usage:
bash# Basic run
python sfc_integrated_analyzer.py --start 2020 --end 2024

# With roles mapping
python sfc_integrated_analyzer.py --roles-map mappings/instrument_roles.yaml --start 1990 --end 2024

# Full historical with parallel processing
python sfc_integrated_analyzer.py --start 1965 --end 2024 --workers 8 --force

# Force regeneration
python sfc_integrated_analyzer.py --start 2020 --end 2024 --force
    
"""

import argparse
import json
import yaml
import logging
import time
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== PART 1: MATRIX GENERATION (from sfc_core.py) ==========

# Configuration (from sfc_core.py)
@dataclass
class SFCConfig:
    """Configuration container aligned with sfc_core.py."""
    base_dir: str
    cache_dir: str
    output_dir: str
    instrument_map_path: str
    flow_map_path: str
    date: str = None  # Will be overridden per quarter
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SFCConfig':
        """Load config from YAML file (exact same as sfc_core.py)."""
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f) or {}
        
        if "sfc" not in raw:
            raise ValueError(f"{yaml_path} must contain an 'sfc' section.")
        
        sfc = raw["sfc"]
        req = ["base_dir", "cache_dir", "output_dir", "instrument_map", "flow_map"]
        missing = [k for k in req if not sfc.get(k)]
        if missing:
            raise ValueError(f"sfc config missing: {missing}")
        
        return cls(
            base_dir=sfc['base_dir'],
            cache_dir=sfc['cache_dir'],
            output_dir=sfc['output_dir'],
            instrument_map_path=sfc['instrument_map'],
            flow_map_path=sfc['flow_map'],
            date=sfc.get('date')
        )


class Z1MatrixGenerator:
    """
    Generates SFC matrices using EXACT logic from sfc_core.py.
    Loads data once, processes many quarters.
    """
    
    def __init__(self, config: SFCConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load maps (same as sfc_core.py)
        self.load_maps()
        
        # Will hold Z1 data
        self.tidy_data = None
        
        # Regex patterns from sfc_core.py
        self._SERIES_RE = re.compile(
            r'^(?P<prefix>[A-Z]{2})'      # Prefix (FA, FL, FU, etc.)
            r'(?P<sector>\d{2})'           # Sector (2 digits)
            r'(?P<instrument>\d{5})'       # Instrument (5 digits)
            r'(?P<digit8>\d)'              # Digit 8 (always 0)
            r'(?P<calc_type>\d)'           # Digit 9 (calculation type)
            r'\.(?P<freq>[AQ])$'           # Frequency (.A or .Q)
        )
        
        self._SERIES_RE_NO_DOT = re.compile(
            r'^(?P<prefix>[A-Z]{2})'
            r'(?P<sector>\d{2})'
            r'(?P<instrument>\d{5})'
            r'(?P<digit8>\d)'
            r'(?P<calc_type>\d)'
            r'(?P<freq>[AQ])$'
        )
    
    def load_maps(self):
        """Load instrument and flow maps (exact from sfc_core.py)."""
        with open(self.config.instrument_map_path, 'r', encoding='utf-8') as f:
            self.imap = json.load(f)
        
        with open(self.config.flow_map_path, 'r', encoding='utf-8') as f:
            self.fmap = json.load(f)
    
    def load_z1_panel(self):
        """Load and parse Z1 data (exact logic from sfc_core.py load_z1_panel)."""
        logger.info("="*60)
        logger.info("LOADING Z1 DATA")
        logger.info("="*60)
        
        # Import and load data (EXACT from sfc_core.py)
        from src.data.cached_fed_data_loader import CachedFedDataLoader
        from src.data.data_processor import DataProcessor
        
        loader = CachedFedDataLoader(
            base_directory=self.config.base_dir,
            cache_directory=self.config.cache_dir,
            start_year=1959,
            end_year=2025,  # Extended to get latest data
            cache_expiry_days=30
        )
        
        z1_raw = loader.load_single_source('Z1')
        if z1_raw is None:
            raise ValueError("Failed to load Z1 data from cache/source")
        
        logger.info(f"✓ Raw Z1 shape: {z1_raw.shape}")
        
        z1 = DataProcessor().process_fed_data(z1_raw, 'Z1')
        logger.info(f"✓ Processed Z1 shape: {z1.shape}")
        
        # Ensure index is named for melt
        if z1.index.name is None:
            z1.index.name = "date"
        
        # Wide -> long format (EXACT from sfc_core.py)
        df = z1.reset_index().melt(id_vars=["date"], var_name="series", value_name="value")
        df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"✓ Long format: {df.shape[0]:,} rows")
        logger.info(f"✓ Unique series: {df['series'].nunique():,}")
        
        # Parse all series (using parse_z1_series from sfc_core.py)
        parsed_results = df["series"].apply(self.parse_z1_series)
        
        # Count successful parses
        success_mask = parsed_results.notnull()
        logger.info(f"  ✓ Successful: {success_mask.sum():,} rows ({100*success_mask.mean():.1f}%)")
        
        if (~success_mask).sum() > 0:
            logger.info(f"  ✗ Failed: {(~success_mask).sum():,} rows")
        
        # Expand parsed tuples
        expanded = parsed_results.apply(
            lambda x: x if x is not None else (pd.NA, pd.NA, pd.NA)
        ).apply(pd.Series)
        expanded.columns = ["kind", "sector", "instrument"]
        
        # Create tidy dataframe
        tidy = pd.concat([df, expanded], axis=1)
        self.tidy_data = tidy.dropna(subset=["kind", "sector", "instrument"])
        
        # Ensure instrument is string and 5 digits
        self.tidy_data['instrument'] = self.tidy_data['instrument'].astype(str).str.zfill(5)
        
        # Summary
        logger.info(f"\nSeries breakdown:")
        for kind in sorted(self.tidy_data["kind"].unique()):
            count = (self.tidy_data["kind"] == kind).sum()
            logger.info(f"  {kind}: {count:,} rows")
        
        # Report available dates
        dates = self.tidy_data['date'].unique()
        logger.info(f"Date range: {min(dates).date()} to {max(dates).date()}")
        logger.info(f"Quarters available: {len(dates)}")
        
        self.tidy_data = self.tidy_data[["date", "series", "value", "kind", "sector", "instrument"]]
    
    def parse_z1_series(self, series_str: str):
        """Parse Z1 series mnemonic (exact from sfc_core.py)."""
        series_str = str(series_str).strip().upper()
        
        # Try main pattern first
        m = self._SERIES_RE.match(series_str)
        if m:
            prefix = m.group('prefix')
            if prefix in {"FA","FL","FU","FV","FR","FC","FG","FI","FS","LA","LM","PC"}:
                # Ensure instrument is 5 digits
                instrument = m.group('instrument')
                if len(instrument) < 5:
                    instrument = instrument.zfill(5)
                return prefix, m.group('sector'), instrument
        
        # Try without dot
        m = self._SERIES_RE_NO_DOT.match(series_str)
        if m:
            prefix = m.group('prefix')
            if prefix in {"FA","FL","FU","FV","FR","FC","FG","FI","FS","LA","LM","PC"}:
                # Ensure instrument is 5 digits
                instrument = m.group('instrument')
                if len(instrument) < 5:
                    instrument = instrument.zfill(5)
                return prefix, m.group('sector'), instrument
        
        return None
    
    def sfc_sign(self, instr: str) -> int:
        """Get sign for instrument (exact from sfc_core.py)."""
        side = self.imap.get(instr, {}).get('side', 'asset')
        
        if instr not in self.imap:
            return -1
        
        if side == 'asset':
            return -1
        elif side == 'liability':
            return +1
        else:  # 'macro' or other
            return 0
    
    def get_all_sectors(self, qdate: pd.Timestamp) -> List[str]:
        """Get all sectors for date (from sfc_core.py)."""
        date_data = self.tidy_data[self.tidy_data['date'] == qdate]
        if len(date_data) == 0:
            return []
        
        sectors = sorted(date_data['sector'].unique())
        return sectors
    
    def build_balance_sheet(self, qdate: pd.Timestamp, all_sectors: List[str]) -> pd.DataFrame:
        """Build balance sheet matrix (exact from sfc_core.py)."""
        z = self.tidy_data[(self.tidy_data['kind'] == 'FL') & 
                          (self.tidy_data['date'] == qdate)].copy()
        
        if len(z) == 0:
            return pd.DataFrame()
        
        mat = z.pivot_table(
            index='instrument', 
            columns='sector', 
            values='value', 
            aggfunc='sum'
        ).fillna(0.0)
        
        mat.insert(0, 'label', mat.index.map(lambda k: self.imap.get(k, {}).get('label', '')))
        
        # Ensure all sectors present
        if all_sectors:
            for s in all_sectors:
                if s not in mat.columns:
                    mat[s] = 0.0
        
        sector_cols = [c for c in mat.columns if c != 'label']
        mat['Total'] = mat[sector_cols].sum(axis=1)
        
        return mat.sort_index()
    
    def build_transactions_financial(self, qdate: pd.Timestamp, all_sectors: List[str]) -> pd.DataFrame:
        """Build transaction matrix (exact from sfc_core.py)."""
        z = self.tidy_data[(self.tidy_data['kind'] == 'FU') & 
                          (self.tidy_data['date'] == qdate)].copy()
        
        if len(z) == 0:
            return pd.DataFrame()
        
        z['signed_value'] = z.apply(
            lambda row: self.sfc_sign(row['instrument']) * row['value'],
            axis=1
        )
        
        mat = z.pivot_table(
            index='instrument',
            columns='sector',
            values='signed_value',
            aggfunc='sum'
        ).fillna(0.0)
        
        mat.insert(0, 'label', mat.index.map(lambda k: self.imap.get(k, {}).get('label', '')))
        
        # Ensure all sectors present
        if all_sectors:
            for s in all_sectors:
                if s not in mat.columns:
                    mat[s] = 0.0
        
        sector_cols = [c for c in mat.columns if c != 'label']
        mat['Total'] = mat[sector_cols].sum(axis=1)
        
        return mat
    
    def add_macro_flows(self, trans_mat: pd.DataFrame, qdate: pd.Timestamp, 
                       all_sectors: List[str]) -> pd.DataFrame:
        """Add macro flows (exact from sfc_core.py)."""
        if trans_mat.empty:
            return trans_mat
        
        if all_sectors:
            sector_cols = all_sectors
            # Ensure all sectors in trans_mat
            for s in all_sectors:
                if s not in trans_mat.columns:
                    trans_mat[s] = 0.0
        else:
            sector_cols = [c for c in trans_mat.columns if c not in ('label', 'Total')]
        
        rows = []
        
        for code, meta in self.fmap.items():
            blk = self.tidy_data[(self.tidy_data['date'] == qdate) & 
                                (self.tidy_data['instrument'] == code)]
            if blk.empty:
                continue
            
            val = blk['value'].sum()
            row = {c: 0.0 for c in sector_cols}
            
            sbs = meta.get('sign_by_sector', {})
            if sbs:
                for sec, sign in sbs.items():
                    if sec in row:
                        row[sec] += sign * val
            else:
                tos = [s for s in meta.get('to', []) if s in row]
                frs = [s for s in meta.get('from', []) if s in row]
                
                n_to = len(tos) if tos else 1
                n_fr = len(frs) if frs else 1
                
                for sec in tos:
                    row[sec] += val / n_to
                for sec in frs:
                    row[sec] -= val / n_fr
            
            row['label'] = meta.get('label', f'Flow {code}')
            rows.append((code, row))
        
        if not rows:
            return trans_mat
        
        add_df = pd.DataFrame.from_dict({k: v for k, v in rows}, orient='index')
        
        for c in sector_cols:
            if c not in add_df.columns:
                add_df[c] = 0.0
        
        ordered_cols = ['label'] + sorted(sector_cols)
        add_df = add_df[ordered_cols]
        add_df['Total'] = add_df[sector_cols].sum(axis=1)
        
        result = pd.concat([trans_mat, add_df], axis=0)
        
        return result
    
    def recon_stock_flow(self, qdate: pd.Timestamp) -> pd.DataFrame:
        """Reconcile stocks and flows (exact from sfc_core.py)."""
        t = self.tidy_data[self.tidy_data['date'] == qdate]
        prev_dates = self.tidy_data[self.tidy_data['date'] < qdate]['date'].unique()
        
        if len(prev_dates) == 0:
            return pd.DataFrame()
        
        prev_date = max(prev_dates)
        prev = self.tidy_data[self.tidy_data['date'] == prev_date]
        
        recon = []
        for (sector, instrument) in t[['sector', 'instrument']].drop_duplicates().values:
            fl_curr = t[(t['kind'] == 'FL') & (t['sector'] == sector) & 
                       (t['instrument'] == instrument)]['value'].sum()
            fl_prev = prev[(prev['kind'] == 'FL') & (prev['sector'] == sector) & 
                          (prev['instrument'] == instrument)]['value'].sum()
            
            fu = t[(t['kind'] == 'FU') & (t['sector'] == sector) & 
                  (t['instrument'] == instrument)]['value'].sum()
            fr = t[(t['kind'] == 'FR') & (t['sector'] == sector) & 
                  (t['instrument'] == instrument)]['value'].sum()
            fv = t[(t['kind'] == 'FV') & (t['sector'] == sector) & 
                  (t['instrument'] == instrument)]['value'].sum()
            
            dfl = fl_curr - fl_prev
            gap = dfl - (fu + fr + fv)
            
            if abs(dfl) > 1e-6 or abs(fu) > 1e-6:
                recon.append({
                    'sector': sector,
                    'instrument': instrument,
                    'dFL': dfl,
                    'FU': fu,
                    'FR': fr,
                    'FV': fv,
                    'Gap': gap
                })
        
        return pd.DataFrame(recon)
    
    def generate_quarter_matrices(self, qdate: pd.Timestamp) -> Dict[str, Any]:
        """Generate all matrices for a single quarter."""
        date_str = qdate.strftime('%Y-%m-%d')
        result = {'date': date_str, 'success': False}
        
        try:
            # Get all sectors
            all_sectors = self.get_all_sectors(qdate)
            
            if not all_sectors:
                result['error'] = f"No data for {date_str}"
                return result
            
            # Build matrices
            bs = self.build_balance_sheet(qdate, all_sectors)
            tf_fin = self.build_transactions_financial(qdate, all_sectors)
            
            if not tf_fin.empty:
                tf_full = self.add_macro_flows(tf_fin, qdate, all_sectors)
            else:
                tf_full = tf_fin
            
            rc = self.recon_stock_flow(qdate)
            
            # Save outputs
            tag = date_str
            
            if not bs.empty:
                bs_file = self.output_dir / f'sfc_balance_sheet_{tag}.csv'
                # The index IS the instrument column
                bs_out = bs.copy()
                bs_out.index = bs_out.index.astype(str).str.zfill(5)
                bs_out.to_csv(bs_file)
                result['balance_sheet'] = bs_file.name
            
            if not tf_full.empty:
                tf_file = self.output_dir / f'sfc_transactions_{tag}.csv'
                # The index IS the instrument column
                tf_out = tf_full.copy()
                tf_out.index = tf_out.index.astype(str).str.zfill(5)
                tf_out.to_csv(tf_file)
                result['transactions'] = tf_file.name
            
            if not rc.empty:
                rc_file = self.output_dir / f'sfc_recon_{tag}.csv'
                if 'instrument' in rc.columns:
                    rc['instrument'] = rc['instrument'].astype(str).str.zfill(5)
                rc.to_csv(rc_file, index=False)
                result['reconciliation'] = rc_file.name
            
            result['success'] = True
            
        except Exception as e:
            import traceback
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Error generating {date_str}: {e}")
            logger.error(f"Traceback:\n{result['traceback']}")
        
        return result


# ========== PART 2: ANALYSIS (from sfc_analyzer_final.py) ==========

@dataclass
class QuarterData:
    """Container for single quarter's data."""
    date: str
    balance_sheet: Optional[pd.DataFrame] = None
    transactions: Optional[pd.DataFrame] = None
    reconciliation: Optional[pd.DataFrame] = None
    indicators: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class SFCIndicatorCalculator:
    """
    Calculate indicators using EXACT logic from sfc_analyzer_final.py.
    Fully data-driven with role-based classification.
    """
    
    def __init__(self, instrument_map: dict, flow_map: dict,
                 class_roles: dict = None, historical_data: pd.DataFrame = None):
        self.instrument_map = instrument_map
        self.flow_map = flow_map
        self.class_roles = class_roles or {}
        self.historical_data = historical_data
        
        # Index roles (from sfc_analyzer_final.py)
        self._index_roles()
        self._compute_normalization_params()
    
    def _index_roles(self):
        """Build instrument role sets (exact from sfc_analyzer_final.py)."""
        self.roles_by_instr = {}
        
        for code, meta in self.instrument_map.items():
            roles = set()
            
            # 1) Explicit per-instrument roles
            r = meta.get('roles')
            if isinstance(r, list):
                roles.update(str(x).lower() for x in r)
            
            # 2) Class-based roles
            klass = str(meta.get('class', '')).lower()
            if klass and klass in self.class_roles:
                class_role_list = self.class_roles[klass]
                if isinstance(class_role_list, list):
                    roles.update(str(x).lower() for x in class_role_list)
            
            # 3) Minimal heuristic ONLY if nothing assigned
            if not roles:
                side = str(meta.get('side', '')).lower()
                label = str(meta.get('label', '')).lower()
                
                if side == 'liability' or any(k in klass for k in ['debt', 'bond', 'note', 'security', 'loan', 'mortgage', 'paper']):
                    roles.add('debt')
                
                if any(k in klass for k in ['loan', 'mortgage', 'consumer', 'credit']):
                    roles.add('loan')
                
                if (str(meta.get('liquidity', '')).lower() == 'high' or 
                    any(k in klass for k in ['currency', 'deposit', 'mmf', 'reserves', 'repo', 'cash']) or
                    any(k in label for k in ['currency', 'deposit', 'cash'])):
                    roles.add('liquid')
                
                if any(k in klass for k in ['payable', 'short-term', 'cp', 'repo-liab', 'taxes']):
                    roles.add('st_liability')
                
                if any(k in klass for k in ['equity', 'stock', 'share']):
                    roles.add('equity')
                
                if any(k in klass for k in ['derivative', 'option', 'future', 'swap']):
                    roles.add('derivative')
            
            self.roles_by_instr[code] = roles
        
        # Materialized role sets
        self.debt_instruments = {k for k, r in self.roles_by_instr.items() if 'debt' in r}
        self.loan_instruments = {k for k, r in self.roles_by_instr.items() if 'loan' in r}
        self.liquid_instruments = {k for k, r in self.roles_by_instr.items() if 'liquid' in r}
        self.st_liab_instruments = {k for k, r in self.roles_by_instr.items() if 'st_liability' in r}
        self.equity_instruments = {k for k, r in self.roles_by_instr.items() if 'equity' in r}
        self.derivative_instruments = {k for k, r in self.roles_by_instr.items() if 'derivative' in r}
    
    def _compute_normalization_params(self):
        """Compute percentile normalization (from sfc_analyzer_final.py)."""
        self.norm_params = {}
        
        if self.historical_data is not None and not self.historical_data.empty:
            for col in self.historical_data.columns:
                if col != 'date':
                    values = self.historical_data[col].dropna()
                    if len(values) > 0:
                        self.norm_params[col] = {
                            'p1': np.percentile(values, 1),
                            'p25': np.percentile(values, 25),
                            'p50': np.percentile(values, 50),
                            'p75': np.percentile(values, 75),
                            'p99': np.percentile(values, 99),
                            'mean': values.mean(),
                            'std': values.std()
                        }
    
    def normalize_value(self, value: float, indicator: str) -> float:
        """Normalize value (from sfc_analyzer_final.py)."""
        if indicator in self.norm_params:
            params = self.norm_params[indicator]
            p1, p99 = params['p1'], params['p99']
            if p99 > p1:
                normalized = (value - p1) / (p99 - p1)
                return np.clip(normalized, 0, 1)
        
        return min(max(value, 0), 1)
    
    def calculate_quarter_indicators(self, quarter_data: QuarterData) -> Dict[str, float]:
        """Calculate all indicators (exact from sfc_analyzer_final.py)."""
        try:
            bs = quarter_data.balance_sheet
            tf = quarter_data.transactions
            
            if bs is None or tf is None:
                return {}
            
            # Get sectors
            sectors = [c for c in bs.columns if c not in ['label', 'Total', 'instrument']]
            
            indicators = {}
            
            # All indicator calculations from sfc_analyzer_final.py
            indicators.update(self._calculate_leverage(bs, sectors))
            indicators.update(self._calculate_liquidity(bs, sectors))
            indicators.update(self._calculate_flows(tf, sectors))
            indicators.update(self._calculate_network(bs, tf, sectors))
            
            if self.equity_instruments:
                indicators.update(self._calculate_equity_metrics(bs, sectors))
            if self.derivative_instruments:
                indicators.update(self._calculate_derivative_exposure(bs, sectors))
            
            # Composite score
            indicators['composite_score'] = self._calculate_composite(indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {quarter_data.date}: {e}")
            return {'error': str(e)}
    
    # Include all the calculation methods from sfc_analyzer_final.py
    def _calculate_leverage(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate leverage indicators (from sfc_analyzer_final.py)."""
        results = {}
        
        total_debt = 0
        total_assets = 0
        
        sector_groups = {
            'household': ['15'],
            'corporate': ['10', '11'],
            'financial': ['70', '71', '66', '40', '41', '42'],
            'government': ['20', '21', '22']
        }
        
        for group_name, group_sectors in sector_groups.items():
            group_debt = 0
            group_assets = 0
            
            for sector in sectors:
                if sector in group_sectors:
                    sector_debt = sum(
                        abs(bs.loc[inst, sector]) 
                        for inst in self.debt_instruments 
                        if inst in bs.index
                    )
                    sector_assets = abs(bs[sector].sum())
                    
                    group_debt += sector_debt
                    group_assets += sector_assets
            
            if group_assets > 0:
                results[f'{group_name}_leverage'] = group_debt / group_assets
            
            total_debt += group_debt
            total_assets += group_assets
        
        results['system_leverage'] = total_debt / total_assets if total_assets > 0 else 0
        
        return results
    
    def _calculate_liquidity(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate liquidity indicators (from sfc_analyzer_final.py)."""
        results = {}
        
        total_liquid = 0
        total_assets = 0
        total_st_coverage = 0
        coverage_count = 0
        
        for sector in sectors:
            sector_liquid = sum(
                abs(bs.loc[inst, sector])
                for inst in self.liquid_instruments
                if inst in bs.index
            )
            
            sector_st_liab = sum(
                abs(bs.loc[inst, sector])
                for inst in self.st_liab_instruments
                if inst in bs.index
            )
            
            sector_assets = abs(bs[sector].sum())
            
            total_liquid += sector_liquid
            total_assets += sector_assets
            
            if sector_st_liab > 0:
                coverage = sector_liquid / sector_st_liab
                total_st_coverage += coverage
                coverage_count += 1
        
        results['system_liquidity_ratio'] = total_liquid / total_assets if total_assets > 0 else 0
        results['avg_liquidity_coverage'] = total_st_coverage / coverage_count if coverage_count > 0 else 0
        
        return results
    
    def _calculate_flows(self, tf: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate flow indicators (from sfc_analyzer_final.py)."""
        results = {}
        
        total_imbalance = 0
        for sector in sectors:
            if sector in tf.columns:
                inflows = tf[sector][tf[sector] > 0].sum()
                outflows = abs(tf[sector][tf[sector] < 0].sum())
                total = inflows + outflows
                
                if total > 0:
                    imbalance = abs(inflows - outflows) / total
                    total_imbalance += imbalance
        
        results['avg_flow_imbalance'] = total_imbalance / len(sectors) if sectors else 0
        
        credit_flow = 0
        for inst in self.loan_instruments:
            if inst in tf.index:
                credit_flow += tf.loc[inst, sectors].clip(lower=0).sum()
        
        results['credit_flow'] = credit_flow
        
        debt_issuance = 0
        for inst in self.debt_instruments:
            if inst in tf.index:
                debt_issuance += tf.loc[inst, sectors].sum()
        
        results['net_debt_issuance'] = debt_issuance
        
        return results
    
    def _calculate_network(self, bs: pd.DataFrame, tf: pd.DataFrame, 
                          sectors: List[str]) -> Dict[str, float]:
        """Calculate network metrics (from sfc_analyzer_final.py)."""
        results = {}
        
        bs_totals = bs[sectors].sum(axis=0)
        bs_sum = bs_totals.sum()
        if bs_sum > 0:
            bs_shares = bs_totals / bs_sum
            results['bs_herfindahl'] = (bs_shares ** 2).sum()
        else:
            results['bs_herfindahl'] = 0
        
        tf_abs = tf[sectors].abs()
        tf_totals = tf_abs.sum(axis=0)
        tf_sum = tf_totals.sum()
        if tf_sum > 0:
            tf_shares = tf_totals / tf_sum
            results['tf_herfindahl'] = (tf_shares ** 2).sum()
        else:
            results['tf_herfindahl'] = 0
        
        total_positions = len(bs.index) * len(sectors)
        if total_positions > 0:
            non_zero = (bs[sectors] != 0).sum().sum()
            results['network_density'] = non_zero / total_positions
        else:
            results['network_density'] = 0
        
        if '70' in sectors and '15' in sectors:
            bank_household = sum(
                abs(bs.loc[inst, '70']) 
                for inst in self.loan_instruments 
                if inst in bs.index
            )
            results['bank_household_exposure'] = bank_household
        
        return results
    
    def _calculate_equity_metrics(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate equity metrics (from sfc_analyzer_final.py)."""
        results = {}
        
        total_equity = 0
        for inst in self.equity_instruments:
            if inst in bs.index:
                total_equity += bs.loc[inst, sectors].sum()
        
        results['total_equity_value'] = total_equity
        
        return results
    
    def _calculate_derivative_exposure(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate derivative exposure (from sfc_analyzer_final.py)."""
        results = {}
        
        total_derivatives = 0
        for inst in self.derivative_instruments:
            if inst in bs.index:
                total_derivatives += bs.loc[inst, sectors].abs().sum()
        
        results['derivative_exposure'] = total_derivatives
        
        return results
    
    def _calculate_composite(self, indicators: Dict[str, float]) -> float:
        """Calculate weighted composite score (from sfc_analyzer_final.py)."""
        weights = {
            'system_leverage': 0.20,
            'household_leverage': 0.10,
            'corporate_leverage': 0.10,
            'financial_leverage': 0.15,
            'avg_flow_imbalance': 0.15,
            'system_liquidity_ratio': -0.10,
            'avg_liquidity_coverage': -0.05,
            'credit_flow': 0.10,
            'bs_herfindahl': 0.05,
            'network_density': 0.05,
            'bank_household_exposure': 0.05,
        }
        
        score = 0
        weight_sum = 0
        
        for key, weight in weights.items():
            if key in indicators:
                value = indicators[key]
                if not pd.isna(value) and not np.isinf(value):
                    norm_value = self.normalize_value(value, key)
                    score += abs(weight) * norm_value * (1 if weight > 0 else -1)
                    weight_sum += abs(weight)
        
        return score / weight_sum if weight_sum > 0 else 0


# ========== INTEGRATED PROCESSOR ==========

class IntegratedSFCProcessor:
    """
    Combines matrix generation and analysis in a single efficient pipeline.
    """
    
    def __init__(self, config: SFCConfig, workers: int = None, roles_map_path: str = None):
        self.config = config
        self.workers = workers or min(multiprocessing.cpu_count(), 8)
        self.roles_map_path = roles_map_path
        
        # Initialize components
        self.generator = Z1MatrixGenerator(config)
        
        # Load roles map if provided
        self.class_roles = {}
        if roles_map_path and Path(roles_map_path).exists():
            self._load_roles_map(roles_map_path)
        
        # Will be initialized after data load
        self.calculator = None
    
    def _load_roles_map(self, path: str):
        """Load roles mapping file."""
        p = Path(path)
        logger.info(f"Loading roles map from {p}")
        with open(p, 'r', encoding='utf-8') as f:
            if p.suffix.lower() in ('.yml', '.yaml'):
                self.class_roles = yaml.safe_load(f) or {}
            else:
                self.class_roles = json.load(f)
        logger.info(f"Loaded {len(self.class_roles)} class-role mappings")
    
    def run_complete_analysis(self, start_year: int, end_year: int, 
                             force_regenerate: bool = False):
        """Run complete pipeline: generate matrices then analyze."""
        
        logger.info("="*70)
        logger.info("INTEGRATED SFC ANALYSIS PIPELINE")
        logger.info("="*70)
        logger.info(f"Period: {start_year}-{end_year}")
        logger.info(f"Workers: {self.workers}")
        
        # STEP 1: Load Z1 data once
        self.generator.load_z1_panel()
        
        # STEP 2: Generate all matrices
        matrices_df = self._generate_all_matrices(start_year, end_year, force_regenerate)
        
        # STEP 3: Initialize calculator with loaded data
        self.calculator = SFCIndicatorCalculator(
            instrument_map=self.generator.imap,
            flow_map=self.generator.fmap,
            class_roles=self.class_roles
        )
        
        # Log role coverage
        total_instruments = len(self.generator.imap)
        mapped = len([k for k, r in self.calculator.roles_by_instr.items() if r])
        logger.info(f"Instrument role coverage: {mapped}/{total_instruments}")
        
        # STEP 4: Analyze all generated matrices
        indicators_df = self._analyze_all_matrices(matrices_df)
        
        # STEP 5: Generate summary and visualizations
        self._generate_summary(indicators_df)
        
        return indicators_df
    
    def _generate_all_matrices(self, start_year: int, end_year: int, 
                              force_regenerate: bool) -> pd.DataFrame:
        """Generate all SFC matrices."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: MATRIX GENERATION")
        logger.info("="*60)
        
        # Get all quarter dates
        quarters = []
        for year in range(start_year, end_year + 1):
            for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
                quarters.append(pd.Timestamp(year, month, day))
        
        # Filter to available dates
        available_dates = set(self.generator.tidy_data['date'].unique())
        quarters = [q for q in quarters if q in available_dates]
        
        logger.info(f"Quarters to process: {len(quarters)}")
        
        # Check existing if not forcing
        if not force_regenerate:
            to_generate = []
            for q in quarters:
                date_str = q.strftime('%Y-%m-%d')
                bs_file = self.generator.output_dir / f'sfc_balance_sheet_{date_str}.csv'
                tf_file = self.generator.output_dir / f'sfc_transactions_{date_str}.csv'
                if not (bs_file.exists() and tf_file.exists()):
                    to_generate.append(q)
            
            logger.info(f"Already exist: {len(quarters) - len(to_generate)}")
            logger.info(f"Need to generate: {len(to_generate)}")
            quarters = to_generate
        
        if len(quarters) == 0:
            logger.info("All matrices already generated!")
            return pd.DataFrame()
        
        # Generate matrices
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.generator.generate_quarter_matrices, q): q
                for q in quarters
            }
            
            for future in tqdm(as_completed(futures), total=len(quarters),
                             desc="Generating matrices"):
                result = future.result()
                results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Generated successfully: {successful}/{len(results)}")
        
        return pd.DataFrame(results)
    
    def _analyze_all_matrices(self, matrices_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all generated matrices."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: CRISIS INDICATOR ANALYSIS")
        logger.info("="*60)
        
        # Get all available quarters
        all_files = list(self.generator.output_dir.glob("sfc_balance_sheet_*.csv"))
        quarters = []
        for f in all_files:
            date_str = f.name.replace('sfc_balance_sheet_', '').replace('.csv', '')
            quarters.append(date_str)
        
        quarters = sorted(quarters)
        logger.info(f"Quarters to analyze: {len(quarters)}")
        
        # Process each quarter
        indicator_results = []
        
        for date_str in tqdm(quarters, desc="Calculating indicators"):
            # Load matrices
            quarter_data = QuarterData(date=date_str)
            
            bs_file = self.generator.output_dir / f'sfc_balance_sheet_{date_str}.csv'
            tf_file = self.generator.output_dir / f'sfc_transactions_{date_str}.csv'
            
            if bs_file.exists():
                quarter_data.balance_sheet = pd.read_csv(bs_file, index_col=0)
            if tf_file.exists():
                quarter_data.transactions = pd.read_csv(tf_file, index_col=0)
            
            # Calculate indicators
            indicators = self.calculator.calculate_quarter_indicators(quarter_data)
            indicators['date'] = date_str
            indicator_results.append(indicators)
        
        # Create DataFrame
        indicators_df = pd.DataFrame(indicator_results)
        indicators_df['date'] = pd.to_datetime(indicators_df['date'])
        indicators_df = indicators_df.set_index('date').sort_index()
        
        # Save results
        output_file = self.generator.output_dir / 'integrated_indicators.csv'
        indicators_df.to_csv(output_file)
        logger.info(f"Saved indicators to {output_file}")
        
        return indicators_df
    
    def _generate_summary(self, indicators_df: pd.DataFrame):
        """Generate summary statistics and visualizations."""
        if indicators_df.empty:
            return
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Quarters analyzed: {len(indicators_df)}")
        logger.info(f"Date range: {indicators_df.index[0].date()} to {indicators_df.index[-1].date()}")
        
        # Latest indicators
        latest = indicators_df.iloc[-1]
        logger.info(f"\nLatest indicators ({indicators_df.index[-1].date()}):")
        logger.info(f"  Composite score: {latest['composite_score']:.3f}")
        logger.info(f"  System leverage: {latest.get('system_leverage', 0):.3f}")
        logger.info(f"  System liquidity: {latest.get('system_liquidity_ratio', 0):.3f}")
        logger.info(f"  Flow imbalance: {latest.get('avg_flow_imbalance', 0):.3f}")
        
        # Risk assessment
        if latest['composite_score'] > 0.6:
            logger.warning("⚠️ HIGH RISK - Crisis indicators elevated")
        elif latest['composite_score'] > 0.4:
            logger.info("⚠️ MODERATE RISK - Enhanced monitoring recommended")
        else:
            logger.info("✅ LOW RISK - System appears stable")


# ========== MAIN ==========

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Integrated SFC Analyzer - Complete Pipeline"
    )
    
    parser.add_argument('--config', default='config/proper_sfc_config.yaml',
                       help='Configuration file')
    parser.add_argument('--roles-map', default=None,
                       help='Optional roles mapping file')
    parser.add_argument('--start', type=int, default=2020,
                       help='Start year')
    parser.add_argument('--end', type=int, default=2024,
                       help='End year')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of existing matrices')
    
    args = parser.parse_args()
    
    # Load configuration
    config = SFCConfig.from_yaml(args.config)
    
    # Initialize processor
    processor = IntegratedSFCProcessor(
        config=config,
        workers=args.workers,
        roles_map_path=args.roles_map
    )
    
    # Run complete analysis
    indicators = processor.run_complete_analysis(
        args.start,
        args.end,
        force_regenerate=args.force
    )
    
    return indicators


if __name__ == "__main__":
    indicators = main()
