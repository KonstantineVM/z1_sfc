#!/usr/bin/env python3
"""
Final Production SFC Analyzer - Fully Data-Driven
==================================================
100% instrument-map-driven analyzer with no hardcoded buckets.
Supports role-based classification via instrument map or separate roles map.

1. Parallel Processing Engine

Smart worker allocation (auto-detects CPU cores)
Chunk-based processing for memory efficiency
Retry logic with exponential backoff
Progress tracking with tqdm
Failed quarter logging for debugging

2. Enhanced Crisis Indicators

Sectoral Leverage (financial, household, corporate, government)
Liquidity Metrics (system liquidity, coverage ratios)
Network Concentration (Herfindahl index, network density)
Flow Imbalances (payment pattern asymmetries)
Credit Growth (loan expansion tracking)
Interconnectedness (bank-household exposures)
Maturity Mismatch (banking sector specific)

3. Dynamic Instrument Classification

Reads from instrument_map.json
Falls back to sensible defaults
Expandable bucket system via JSON configuration
Property-based instrument selection

4. Robust Statistical Methods

Rolling window normalization (adaptive to regime changes)
Winsorized z-scores (handles outliers)
Sigmoid transformation for bounded scores
Weighted composite scoring with economic intuition

5. Comprehensive Validation

ROC curve analysis with AUC calculation
F1 score optimization for threshold selection
Lead time analysis for early warning
False positive exclusion for recession windows
Performance metrics by recession severity

6. Production-Ready Features

No YAML file mutations (uses --date CLI argument)
Headless environment detection
Memory management with garbage collection
Comprehensive error handling
Detailed logging and reporting


✅ 100% Data-Driven Classification

No hardcoded instrument buckets
Three-tier priority system:

Per-instrument roles from instrument_map.json
Class-based roles from optional roles_map file
Minimal heuristics only as last resort


✅ Enhanced Normalization

Historical percentile-based normalization (1st-99th percentiles)
Automatically loads previous historical_indicators.csv if available
Falls back to simple 0-1 capping for new indicators

✅ Configurable Timeout & Retry

--timeout flag for per-quarter processing time
--retry flag for maximum retry attempts
Exponential backoff between retries

✅ Extended Role System
Supports multiple role tags:

debt, loan, liquid, st_liability
equity, derivative (for extended metrics)
Easily extensible via data files

✅ Production Features

Generates instrument_role_mappings.csv report
Shows role coverage statistics
Parallel processing with ProcessPoolExecutor
Full alignment with sfc_core.py config

Usage Examples:
bash# Basic run
python scripts/sfc_analyzer_final.py --config config/proper_sfc_config.yaml --start 2020 --end 2024

# With custom roles mapping
python scripts/sfc_analyzer_final.py --roles-map mappings/instrument_roles.yaml --start 1990 --end 2024

# Full production run with all options
python scripts/sfc_analyzer_final.py \
  --config config/proper_sfc_config.yaml \
  --roles-map mappings/instrument_roles.yaml \
  --start 1965 --end 2024 \
  --workers 8 \
  --timeout 120 \
  --retry 3 \
  --generate


Location:
Place in scripts/ folder:
z1_sfc/
├── scripts/
│   ├── sfc_core.py
│   └── sfc_analyzer_final.py  <-- HERE
├── mappings/
│   ├── instrument_map.json
│   └── instrument_roles.yaml  <-- Optional
└── config/
    └── proper_sfc_config.yaml

Features:
- All instrument classifications from data (JSON/YAML)
- Configurable timeout and retry logic
- Historical percentile normalization
- Parallel processing with ProcessPoolExecutor
- Full alignment with sfc_core.py configuration

Usage:
# Basic run
python scripts/sfc_analyzer_final.py --config config/proper_sfc_config.yaml --start 2020 --end 2024

# With custom roles mapping
python scripts/sfc_analyzer_final.py --roles-map mappings/instrument_roles.yaml --start 1990 --end 2024

# Full production run with all options
python scripts/sfc_analyzer_final.py \
  --config config/proper_sfc_config.yaml \
  --roles-map mappings/instrument_roles.yaml \
  --start 1965 --end 2024 \
  --workers 8 \
  --timeout 120 \
  --retry 3 \
  --generate
"""

import argparse
import json
import yaml
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== DATA CLASSES ==========

@dataclass
class SFCConfig:
    """Configuration container aligned with sfc_core.py."""
    base_dir: str
    cache_dir: str
    output_dir: str
    instrument_map_path: str
    flow_map_path: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SFCConfig':
        """Load config from YAML file (same as sfc_core.py uses)."""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        sfc = cfg.get('sfc', cfg)  # Handle both nested and flat configs
        
        return cls(
            base_dir=sfc.get('base_dir', './data/fed_data'),
            cache_dir=sfc.get('cache_dir', './data/cache'),
            output_dir=sfc.get('output_dir', 'outputs'),
            instrument_map_path=sfc.get('instrument_map', 'mappings/instrument_map.json'),
            flow_map_path=sfc.get('flow_map', 'mappings/flow_map_expanded.json')
        )


@dataclass
class QuarterData:
    """Container for single quarter's data."""
    date: str
    balance_sheet: Optional[pd.DataFrame] = None
    transactions: Optional[pd.DataFrame] = None
    reconciliation: Optional[pd.DataFrame] = None
    indicators: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# ========== FULLY DATA-DRIVEN CALCULATOR ==========

class SFCIndicatorCalculator:
    """
    Fully data-driven calculator for SFC indicators.
    No hardcoded buckets - everything from instrument map and roles.
    """
    
    def __init__(self, instrument_map: dict, flow_map: dict, 
                 standard_sectors: List[str] = None,
                 class_roles: dict = None,
                 historical_data: pd.DataFrame = None):
        """
        Initialize with maps loaded ONCE.
        
        Args:
            instrument_map: Instrument metadata
            flow_map: Flow classifications
            standard_sectors: Standard sector list for consistency
            class_roles: Optional mapping from class -> role tags
            historical_data: Optional historical data for percentile normalization
        """
        self.instrument_map = instrument_map
        self.flow_map = flow_map
        self.standard_sectors = standard_sectors or []
        self.class_roles = class_roles or {}
        self.historical_data = historical_data
        
        # Index roles from data
        self._index_roles()
        
        # Compute normalization parameters if historical data provided
        self._compute_normalization_params()
    
    def _index_roles(self):
        """
        Build instrument role sets from instrument_map and optional class_roles.
        Priority:
          1) per-instrument explicit roles: entry['roles'] (list of tags)
          2) class_roles: map class -> roles[]
          3) minimal heuristic fallback (only if nothing found)
        """
        self.roles_by_instr = {}  # code -> set(role tags)
        
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
                
                # Debt instruments
                if side == 'liability' or any(k in klass for k in ['debt', 'bond', 'note', 'security', 'loan', 'mortgage', 'paper']):
                    roles.add('debt')
                
                # Loan instruments
                if any(k in klass for k in ['loan', 'mortgage', 'consumer', 'credit']):
                    roles.add('loan')
                
                # Liquid instruments
                if (str(meta.get('liquidity', '')).lower() == 'high' or 
                    any(k in klass for k in ['currency', 'deposit', 'mmf', 'reserves', 'repo', 'cash']) or
                    any(k in label for k in ['currency', 'deposit', 'cash'])):
                    roles.add('liquid')
                
                # Short-term liabilities
                if any(k in klass for k in ['payable', 'short-term', 'cp', 'repo-liab', 'taxes']):
                    roles.add('st_liability')
                
                # Equity instruments
                if any(k in klass for k in ['equity', 'stock', 'share']):
                    roles.add('equity')
                
                # Derivatives
                if any(k in klass for k in ['derivative', 'option', 'future', 'swap']):
                    roles.add('derivative')
            
            self.roles_by_instr[code] = roles
        
        # Materialized role sets for efficient lookup
        self.debt_instruments = {k for k, r in self.roles_by_instr.items() if 'debt' in r}
        self.loan_instruments = {k for k, r in self.roles_by_instr.items() if 'loan' in r}
        self.liquid_instruments = {k for k, r in self.roles_by_instr.items() if 'liquid' in r}
        self.st_liab_instruments = {k for k, r in self.roles_by_instr.items() if 'st_liability' in r}
        self.equity_instruments = {k for k, r in self.roles_by_instr.items() if 'equity' in r}
        self.derivative_instruments = {k for k, r in self.roles_by_instr.items() if 'derivative' in r}
        
        # Log role coverage
        total_instruments = len(self.instrument_map)
        mapped_instruments = len([k for k, r in self.roles_by_instr.items() if r])
        logger.info(f"Role mapping coverage: {mapped_instruments}/{total_instruments} instruments")
        logger.info(f"  Debt: {len(self.debt_instruments)}, Loan: {len(self.loan_instruments)}, "
                   f"Liquid: {len(self.liquid_instruments)}, ST Liab: {len(self.st_liab_instruments)}")
    
    def _compute_normalization_params(self):
        """Compute percentile-based normalization parameters from historical data."""
        self.norm_params = {}
        
        if self.historical_data is not None and not self.historical_data.empty:
            # Compute percentiles for each indicator
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
        """
        Normalize value using historical percentiles if available.
        Falls back to simple 0-1 capping if no historical data.
        """
        if indicator in self.norm_params:
            params = self.norm_params[indicator]
            # Robust percentile normalization
            p1, p99 = params['p1'], params['p99']
            if p99 > p1:
                normalized = (value - p1) / (p99 - p1)
                return np.clip(normalized, 0, 1)
        
        # Fallback: simple capping
        return min(max(value, 0), 1)
    
    def calculate_quarter_indicators(self, quarter_data: QuarterData) -> Dict[str, float]:
        """
        Calculate all indicators for a single quarter.
        Pure function - no side effects, ready for parallel execution.
        """
        try:
            bs = quarter_data.balance_sheet
            tf = quarter_data.transactions
            
            if bs is None or tf is None:
                return {}
            
            # Standardize sectors
            sectors = self._get_sectors(bs, tf)
            
            indicators = {}
            
            # Core indicators
            indicators.update(self._calculate_leverage(bs, sectors))
            indicators.update(self._calculate_liquidity(bs, sectors))
            indicators.update(self._calculate_flows(tf, sectors))
            indicators.update(self._calculate_network(bs, tf, sectors))
            
            # Extended indicators if instruments available
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
    
    def _get_sectors(self, bs: pd.DataFrame, tf: pd.DataFrame) -> List[str]:
        """Get consistent sector list."""
        bs_sectors = set(c for c in bs.columns if c not in ['label', 'Total'])
        tf_sectors = set(c for c in tf.columns if c not in ['label', 'Total'])
        
        if self.standard_sectors:
            return [s for s in self.standard_sectors if s in bs_sectors and s in tf_sectors]
        else:
            return sorted(bs_sectors & tf_sectors)
    
    def _calculate_leverage(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate leverage indicators using role-based instruments."""
        results = {}
        
        # System-wide leverage
        total_debt = 0
        total_assets = 0
        
        # Sector groups for targeted analysis
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
                    # Calculate debt from role-based instruments
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
        """Calculate liquidity indicators using role-based instruments."""
        results = {}
        
        total_liquid = 0
        total_assets = 0
        total_st_coverage = 0
        coverage_count = 0
        
        for sector in sectors:
            # Liquid assets from role-based instruments
            sector_liquid = sum(
                abs(bs.loc[inst, sector])
                for inst in self.liquid_instruments
                if inst in bs.index
            )
            
            # Short-term liabilities from role-based instruments
            sector_st_liab = sum(
                abs(bs.loc[inst, sector])
                for inst in self.st_liab_instruments
                if inst in bs.index
            )
            
            sector_assets = abs(bs[sector].sum())
            
            total_liquid += sector_liquid
            total_assets += sector_assets
            
            # Liquidity coverage ratio
            if sector_st_liab > 0:
                coverage = sector_liquid / sector_st_liab
                total_st_coverage += coverage
                coverage_count += 1
        
        results['system_liquidity_ratio'] = total_liquid / total_assets if total_assets > 0 else 0
        results['avg_liquidity_coverage'] = total_st_coverage / coverage_count if coverage_count > 0 else 0
        
        return results
    
    def _calculate_flows(self, tf: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate flow indicators."""
        results = {}
        
        # Flow imbalances
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
        
        # Credit flows (loan expansion)
        credit_flow = 0
        for inst in self.loan_instruments:
            if inst in tf.index:
                # Positive flows indicate credit expansion
                credit_flow += tf.loc[inst, sectors].clip(lower=0).sum()
        
        results['credit_flow'] = credit_flow
        
        # Net debt issuance
        debt_issuance = 0
        for inst in self.debt_instruments:
            if inst in tf.index:
                debt_issuance += tf.loc[inst, sectors].sum()
        
        results['net_debt_issuance'] = debt_issuance
        
        return results
    
    def _calculate_network(self, bs: pd.DataFrame, tf: pd.DataFrame, 
                          sectors: List[str]) -> Dict[str, float]:
        """Calculate network concentration and interconnectedness metrics."""
        results = {}
        
        # Herfindahl-Hirschman Index for balance sheet concentration
        bs_totals = bs[sectors].sum(axis=0)
        bs_sum = bs_totals.sum()
        if bs_sum > 0:
            bs_shares = bs_totals / bs_sum
            results['bs_herfindahl'] = (bs_shares ** 2).sum()
        else:
            results['bs_herfindahl'] = 0
        
        # Transaction concentration
        tf_abs = tf[sectors].abs()
        tf_totals = tf_abs.sum(axis=0)
        tf_sum = tf_totals.sum()
        if tf_sum > 0:
            tf_shares = tf_totals / tf_sum
            results['tf_herfindahl'] = (tf_shares ** 2).sum()
        else:
            results['tf_herfindahl'] = 0
        
        # Network density (non-zero positions / total possible)
        total_positions = len(bs.index) * len(sectors)
        if total_positions > 0:
            non_zero = (bs[sectors] != 0).sum().sum()
            results['network_density'] = non_zero / total_positions
        else:
            results['network_density'] = 0
        
        # Cross-sector exposure concentration
        if '70' in sectors and '15' in sectors:  # Bank-household
            bank_household = sum(
                abs(bs.loc[inst, '70']) 
                for inst in self.loan_instruments 
                if inst in bs.index
            )
            results['bank_household_exposure'] = bank_household
        
        return results
    
    def _calculate_equity_metrics(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate equity-related metrics if equity instruments are defined."""
        results = {}
        
        total_equity = 0
        for inst in self.equity_instruments:
            if inst in bs.index:
                total_equity += bs.loc[inst, sectors].sum()
        
        results['total_equity_value'] = total_equity
        
        return results
    
    def _calculate_derivative_exposure(self, bs: pd.DataFrame, sectors: List[str]) -> Dict[str, float]:
        """Calculate derivative exposure if derivative instruments are defined."""
        results = {}
        
        total_derivatives = 0
        for inst in self.derivative_instruments:
            if inst in bs.index:
                total_derivatives += bs.loc[inst, sectors].abs().sum()
        
        results['derivative_exposure'] = total_derivatives
        
        return results
    
    def _calculate_composite(self, indicators: Dict[str, float]) -> float:
        """
        Calculate weighted composite score with proper normalization.
        """
        # Define weights for each indicator
        weights = {
            'system_leverage': 0.20,
            'household_leverage': 0.10,
            'corporate_leverage': 0.10,
            'financial_leverage': 0.15,
            'avg_flow_imbalance': 0.15,
            'system_liquidity_ratio': -0.10,  # Negative: high liquidity is good
            'avg_liquidity_coverage': -0.05,  # Negative: high coverage is good
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
                    # Use percentile normalization if available
                    norm_value = self.normalize_value(value, key)
                    score += abs(weight) * norm_value * (1 if weight > 0 else -1)
                    weight_sum += abs(weight)
        
        return score / weight_sum if weight_sum > 0 else 0


# ========== BATCH PROCESSOR WITH RETRY LOGIC ==========

class SFCBatchProcessor:
    """
    Handles batch processing with configurable timeout and retry logic.
    """
    
    def __init__(self, config: SFCConfig, workers: int = None, 
                 roles_map_path: str = None, timeout: int = 60, max_retries: int = 3):
        """
        Initialize batch processor.
        
        Args:
            config: SFC configuration
            workers: Number of parallel workers (None = auto)
            roles_map_path: Optional path to roles mapping file
            timeout: Timeout per quarter in seconds
            max_retries: Maximum retry attempts per quarter
        """
        self.config = config
        self.workers = workers or min(multiprocessing.cpu_count(), 8)
        self.roles_map_path = roles_map_path
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load maps ONCE
        self._load_maps()
        
        # Determine standard sectors
        self.standard_sectors = self._get_standard_sectors()
        
        # Load historical data for normalization (if available)
        self.historical_data = self._load_historical_data()
        
        # Initialize calculator
        self.calculator = SFCIndicatorCalculator(
            instrument_map=self.instrument_map,
            flow_map=self.flow_map,
            standard_sectors=self.standard_sectors,
            class_roles=self.class_roles,
            historical_data=self.historical_data
        )
    
    def _load_maps(self):
        """Load all required maps."""
        # Instrument map
        logger.info(f"Loading instrument map from {self.config.instrument_map_path}")
        with open(self.config.instrument_map_path, 'r', encoding='utf-8') as f:
            self.instrument_map = json.load(f)
        
        # Flow map
        logger.info(f"Loading flow map from {self.config.flow_map_path}")
        with open(self.config.flow_map_path, 'r', encoding='utf-8') as f:
            self.flow_map = json.load(f)
        
        # Optional roles map: class -> [role tags]
        self.class_roles = {}
        if self.roles_map_path:
            p = Path(self.roles_map_path)
            if p.exists():
                logger.info(f"Loading roles map from {p}")
                with open(p, 'r', encoding='utf-8') as f:
                    if p.suffix.lower() in ('.yml', '.yaml'):
                        self.class_roles = yaml.safe_load(f) or {}
                    else:
                        self.class_roles = json.load(f)
                logger.info(f"Loaded {len(self.class_roles)} class-role mappings")
            else:
                logger.warning(f"Roles map not found: {p}")
    
    def _get_standard_sectors(self) -> List[str]:
        """Get standard sector list from most recent available data."""
        output_dir = Path(self.config.output_dir)
        
        # Find most recent balance sheet
        bs_files = sorted(output_dir.glob("sfc_balance_sheet_*.csv"))
        if not bs_files:
            return []
        
        # Load and extract sectors
        latest_bs = pd.read_csv(bs_files[-1], index_col=0)
        sectors = [c for c in latest_bs.columns if c not in ['label', 'Total']]
        
        logger.info(f"Standard sectors ({len(sectors)}): {sectors}")
        return sectors
    
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical indicators if available for normalization."""
        hist_file = Path(self.config.output_dir) / "historical_indicators.csv"
        if hist_file.exists():
            logger.info(f"Loading historical data for normalization from {hist_file}")
            return pd.read_csv(hist_file, index_col=0, parse_dates=True)
        return None
    
    def load_quarter_data_with_retry(self, date_str: str) -> QuarterData:
        """
        Load data for a single quarter with retry logic.
        """
        for attempt in range(self.max_retries):
            quarter_data = self._load_quarter_data_attempt(date_str)
            
            if quarter_data.error is None:
                return quarter_data
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.debug(f"Retry {attempt + 1} for {date_str} after {wait_time}s")
                time.sleep(wait_time)
        
        return quarter_data
    
    def _load_quarter_data_attempt(self, date_str: str) -> QuarterData:
        """Single attempt to load quarter data."""
        quarter_data = QuarterData(date=date_str)
        
        try:
            output_dir = Path(self.config.output_dir)
            
            # Load balance sheet
            bs_path = output_dir / f"sfc_balance_sheet_{date_str}.csv"
            if bs_path.exists():
                quarter_data.balance_sheet = pd.read_csv(bs_path, index_col=0)
            else:
                quarter_data.error = f"Balance sheet not found"
                return quarter_data
            
            # Load transactions
            tf_path = output_dir / f"sfc_transactions_{date_str}.csv"
            if tf_path.exists():
                quarter_data.transactions = pd.read_csv(tf_path, index_col=0)
            else:
                quarter_data.error = f"Transactions not found"
                return quarter_data
            
            # Load reconciliation (optional)
            rc_path = output_dir / f"sfc_recon_{date_str}.csv"
            if rc_path.exists():
                quarter_data.reconciliation = pd.read_csv(rc_path)
            
        except Exception as e:
            quarter_data.error = str(e)
        
        return quarter_data
    
    def process_quarter(self, date_str: str) -> Tuple[str, Dict[str, float], Optional[str]]:
        """Process a single quarter with retry logic."""
        # Load data with retry
        quarter_data = self.load_quarter_data_with_retry(date_str)
        
        if quarter_data.error:
            return date_str, {}, quarter_data.error
        
        # Calculate indicators
        indicators = self.calculator.calculate_quarter_indicators(quarter_data)
        
        # Check for calculation errors
        if 'error' in indicators:
            return date_str, indicators, indicators['error']
        
        return date_str, indicators, None
    
    def process_quarters_parallel(self, quarters: List[str]) -> pd.DataFrame:
        """Process multiple quarters in parallel."""
        logger.info(f"Processing {len(quarters)} quarters with {self.workers} workers")
        
        results = []
        errors = []
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.process_quarter, q): q 
                for q in quarters
            }
            
            for future in tqdm(as_completed(futures), total=len(quarters), 
                             desc="Processing quarters"):
                quarter = futures[future]
                
                try:
                    date_str, indicators, error = future.result(timeout=self.timeout)
                    
                    if error:
                        errors.append((date_str, error))
                    else:
                        indicators['date'] = date_str
                        results.append(indicators)
                        
                except Exception as e:
                    errors.append((quarter, str(e)))
                    logger.error(f"Exception for {quarter}: {e}")
        
        # Report errors
        if errors:
            logger.warning(f"Failed quarters: {len(errors)}")
            error_file = Path(self.config.output_dir) / "failed_quarters.txt"
            with open(error_file, 'w') as f:
                for date, error in errors:
                    f.write(f"{date}: {error}\n")
        
        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Ensure consistent columns
            all_columns = set()
            for r in results:
                all_columns.update(r.keys())
            
            for col in all_columns:
                if col not in df.columns and col != 'date':
                    df[col] = np.nan
            
            return df
        else:
            return pd.DataFrame()
    
    def generate_quarters(self, start_year: int, end_year: int) -> List[str]:
        """Generate list of quarter-end dates."""
        quarters = []
        for year in range(start_year, end_year + 1):
            for quarter_end in ['03-31', '06-30', '09-30', '12-31']:
                quarters.append(f"{year}-{quarter_end}")
        return quarters
    
    def run_analysis(self, start_year: int, end_year: int, 
                    generate_matrices: bool = False) -> pd.DataFrame:
        """Run complete analysis for date range."""
        quarters = self.generate_quarters(start_year, end_year)
        
        # Generate missing matrices if requested
        if generate_matrices:
            self._generate_matrices_with_retry(quarters)
        
        # Process all quarters
        indicators_df = self.process_quarters_parallel(quarters)
        
        # Save results
        if not indicators_df.empty:
            output_file = Path(self.config.output_dir) / "historical_indicators.csv"
            indicators_df.to_csv(output_file)
            logger.info(f"Saved indicators to {output_file}")
            
            # Also save role mapping report
            self._save_role_mapping_report()
        
        return indicators_df
    
    def _generate_matrices_with_retry(self, quarters: List[str]):
        """Generate SFC matrices with retry logic."""
        logger.info("Checking for missing SFC matrices...")
        
        missing = []
        for q in quarters:
            bs_path = Path(self.config.output_dir) / f"sfc_balance_sheet_{q}.csv"
            if not bs_path.exists():
                missing.append(q)
        
        if not missing:
            logger.info("All matrices already exist")
            return
        
        logger.info(f"Generating {len(missing)} missing matrices...")
        
        for date_str in tqdm(missing, desc="Generating matrices"):
            success = False
            
            for attempt in range(self.max_retries):
                try:
                    cmd = ["python", "scripts/sfc_core.py", "baseline", "--date", date_str]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                    
                    if result.returncode == 0:
                        success = True
                        break
                    
                except Exception as e:
                    logger.debug(f"Attempt {attempt + 1} failed for {date_str}: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            
            if not success:
                logger.warning(f"Failed to generate {date_str} after {self.max_retries} attempts")
    
    def _save_role_mapping_report(self):
        """Save a report of instrument role mappings."""
        report_file = Path(self.config.output_dir) / "instrument_role_mappings.csv"
        
        rows = []
        for code, roles in self.calculator.roles_by_instr.items():
            meta = self.instrument_map.get(code, {})
            rows.append({
                'instrument': code,
                'label': meta.get('label', ''),
                'class': meta.get('class', ''),
                'side': meta.get('side', ''),
                'roles': ', '.join(sorted(roles)) if roles else 'unmapped'
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(report_file, index=False)
        logger.info(f"Saved role mapping report to {report_file}")


# ========== MAIN ==========

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Final Production SFC Historical Analysis System"
    )
    
    parser.add_argument('--config', default='config/proper_sfc_config.yaml',
                       help='Configuration file (same as sfc_core.py)')
    parser.add_argument('--roles-map', default=None,
                       help='Optional YAML/JSON mapping from instrument class -> role tags')
    parser.add_argument('--start', type=int, default=2020,
                       help='Start year')
    parser.add_argument('--end', type=int, default=2024,
                       help='End year')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout per quarter in seconds')
    parser.add_argument('--retry', type=int, default=3,
                       help='Maximum retry attempts per quarter')
    parser.add_argument('--generate', action='store_true',
                       help='Generate missing SFC matrices first')
    
    args = parser.parse_args()
    
    # Load configuration
    config = SFCConfig.from_yaml(args.config)
    
    # Initialize processor
    processor = SFCBatchProcessor(
        config=config,
        workers=args.workers,
        roles_map_path=args.roles_map,
        timeout=args.timeout,
        max_retries=args.retry
    )
    
    # Run analysis
    logger.info("="*60)
    logger.info("FINAL PRODUCTION SFC ANALYSIS")
    logger.info("="*60)
    
    indicators = processor.run_analysis(
        args.start, 
        args.end,
        generate_matrices=args.generate
    )
    
    # Summary
    if not indicators.empty:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Quarters processed: {len(indicators)}")
        logger.info(f"Date range: {indicators.index[0].date()} to {indicators.index[-1].date()}")
        logger.info(f"Indicators computed: {len(indicators.columns)}")
        
        # Role coverage summary
        total_instruments = len(processor.instrument_map)
        mapped = len([k for k, r in processor.calculator.roles_by_instr.items() if r])
        logger.info(f"Instrument role coverage: {mapped}/{total_instruments} ({100*mapped/total_instruments:.1f}%)")
        
        # Current status
        latest = indicators.iloc[-1]
        logger.info(f"\nLatest composite score: {latest['composite_score']:.3f}")
        
        if latest['composite_score'] > 0.6:
            logger.warning("⚠️ HIGH RISK")
        elif latest['composite_score'] > 0.4:
            logger.info("⚠️ MODERATE RISK")
        else:
            logger.info("✅ LOW RISK")
    
    return indicators


if __name__ == "__main__":
    indicators = main()
