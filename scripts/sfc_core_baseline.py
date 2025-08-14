#!/usr/bin/env python3
"""
SFC core runner with one-switch modes.
Usage examples (no long CLI):
  python scripts/sfc_core.py baseline
  python scripts/sfc_core.py fwtw
  python scripts/sfc_core.py graph
  python scripts/sfc_core.py refined
Optional: --config to point to a different YAML (defaults to config/proper_sfc_config.yaml)

Config file must contain an 'sfc' block with:
  base_dir, cache_dir, output_dir, date (YYYY-MM-DD),
  instrument_map, flow_map,
  fwtw (for fwtw mode),
  graph_spec (for graph mode),
  instrument_group_map & graph_spec_refined (for refined mode).

All heavy lifting follows Z1/examples/run_proper_sfc.py:
  CachedFedDataLoader(base_directory=..., cache_directory=...)
  load_single_source('Z1')
  DataProcessor.process_fed_data(z1_raw, 'Z1')


Fixed SFC core runner with correct Z1 series parsing based on Fed documentation.

Z1 Series Structure:
FA 15 30611 0 5 . Q
│  │  │     │ │ │ └── Frequency (Q=Quarterly, A=Annual)
│  │  │     │ │ └──── Calculation type (0,1,3=input; 5,6=calculated)
│  │  │     │ └────── Always 0 (digit 8)
│  │  │     └──────── Instrument code (5 digits, positions 3-7)
│  │  └─────────────── Sector code (2 digits, positions 1-2)
└───────────────────── Prefix (FL=Level, FU=Flow, FR=Revaluation, etc.)
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import re

from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor
from src.utils.z1_series_interpreter import Z1Series
from src.alloc.graph_flow_allocator import allocate_flows_from_graph

# ========== CONFIG LOADING ==========

def load_cfg(path: str) -> dict:
    """Load configuration from YAML file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "sfc" not in raw:
        raise ValueError(f"{cfg_path} must contain an 'sfc' section.")
    sfc = raw["sfc"]
    req = ["base_dir","cache_dir","output_dir","date","instrument_map","flow_map"]
    missing = [k for k in req if not sfc.get(k)]
    if missing:
        raise ValueError(f"sfc config missing: {missing}")
    return sfc

def load_json(path: str):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========== SERIES PARSING ==========

# Correct regex pattern based on Fed documentation
_SERIES_RE = re.compile(
    r'^(?P<prefix>[A-Z]{2})'      # Prefix (FA, FL, FU, etc.)
    r'(?P<sector>\d{2})'           # Sector (2 digits)
    r'(?P<instrument>\d{5})'       # Instrument (5 digits)
    r'(?P<digit8>\d)'              # Digit 8 (always 0)
    r'(?P<calc_type>\d)'           # Digit 9 (calculation type)
    r'\.(?P<freq>[AQ])$'           # Frequency (.A or .Q)
)

# Alternative patterns for variations
_SERIES_RE_NO_DOT = re.compile(
    r'^(?P<prefix>[A-Z]{2})'
    r'(?P<sector>\d{2})'
    r'(?P<instrument>\d{5})'
    r'(?P<digit8>\d)'
    r'(?P<calc_type>\d)'
    r'(?P<freq>[AQ])$'
)

_SERIES_RE_SHORT = re.compile(
    r'^(?P<prefix>[A-Z]{2})'
    r'(?P<sector>\d{2})'
    r'(?P<instrument>\d+)'        # Variable length instrument
    r'\.(?P<freq>[AQ])$'
)

_ALLOWED_PREFIXES = {"FA","FL","FU","FV","FR","FC","FG","FI","FS","LA","LM","PC"}

def parse_z1_series(series_str: str):
    """Parse Z1 series mnemonic according to Fed documentation."""
    series_str = str(series_str).strip().upper()
    
    # Try main pattern first
    m = _SERIES_RE.match(series_str)
    if m:
        prefix = m.group('prefix')
        if prefix in _ALLOWED_PREFIXES:
            return prefix, m.group('sector'), m.group('instrument')
    
    # Try without dot
    m = _SERIES_RE_NO_DOT.match(series_str)
    if m:
        prefix = m.group('prefix')
        if prefix in _ALLOWED_PREFIXES:
            return prefix, m.group('sector'), m.group('instrument')
    
    # Try shorter format
    m = _SERIES_RE_SHORT.match(series_str)
    if m:
        prefix = m.group('prefix')
        if prefix in _ALLOWED_PREFIXES:
            instrument = m.group('instrument')
            # Pad or truncate to 5 digits
            if len(instrument) > 5:
                instrument = instrument[:5]
            elif len(instrument) < 5:
                instrument = instrument.zfill(5)
            return prefix, m.group('sector'), instrument
    
    return None

# ========== DATA LOADING ==========

def load_z1_panel(sfc: dict) -> pd.DataFrame:
    """Load and parse Z1 data."""
    print("\n" + "="*60)
    print("LOADING Z1 DATA")
    print("="*60)
    
    loader = CachedFedDataLoader(
        base_directory=sfc["base_dir"],
        cache_directory=sfc["cache_dir"],
        start_year=1959,
        end_year=2025,  # Extended to get latest data
        cache_expiry_days=30
    )
    
    z1_raw = loader.load_single_source('Z1')
    if z1_raw is None:
        raise ValueError("Failed to load Z1 data from cache/source")
    
    print(f"✓ Raw Z1 shape: {z1_raw.shape}")
    
    z1 = DataProcessor().process_fed_data(z1_raw, 'Z1')
    print(f"✓ Processed Z1 shape: {z1.shape}")
    
    # Ensure index is named for melt
    if z1.index.name is None:
        z1.index.name = "date"
    
    # Wide -> long format
    df = z1.reset_index().melt(id_vars=["date"], var_name="series", value_name="value")
    df["date"] = pd.to_datetime(df["date"])
    
    print(f"✓ Long format: {df.shape[0]:,} rows")
    print(f"✓ Unique series: {df['series'].nunique():,}")
    
    # Parse all series
    print("\nParsing series codes...")
    parsed_results = df["series"].apply(parse_z1_series)
    
    # Count successful parses
    success_mask = parsed_results.notnull()
    print(f"  ✓ Successful: {success_mask.sum():,} rows ({100*success_mask.mean():.1f}%)")
    
    if (~success_mask).sum() > 0:
        print(f"  ✗ Failed: {(~success_mask).sum():,} rows")
    
    # Expand parsed tuples
    expanded = parsed_results.apply(
        lambda x: x if x is not None else (pd.NA, pd.NA, pd.NA)
    ).apply(pd.Series)
    expanded.columns = ["kind", "sector", "instrument"]
    
    # Create tidy dataframe
    tidy = pd.concat([df, expanded], axis=1)
    tidy = tidy.dropna(subset=["kind", "sector", "instrument"])
    
    # Summary
    print(f"\nSeries breakdown:")
    for kind in sorted(tidy["kind"].unique()):
        count = (tidy["kind"] == kind).sum()
        print(f"  {kind}: {count:,} rows")
    
    return tidy[["date", "series", "value", "kind", "sector", "instrument"]]

# ========== HELPER FUNCTIONS ==========

def get_all_sectors(tidy: pd.DataFrame, qdate: pd.Timestamp) -> list:
    """Get all unique sectors for the specific date and relevant series types."""
    mask = (tidy['date'] == qdate) & (tidy['kind'].isin(['FL', 'FU']))
    sectors = sorted(tidy.loc[mask, 'sector'].unique().tolist())
    return sectors

def ensure_all_sectors(matrix: pd.DataFrame, all_sectors: list) -> pd.DataFrame:
    """
    Ensure matrix has columns for all sectors, filling missing with 0.
    ALWAYS computes the Total column.
    """
    special_cols = ['label', 'Total']
    current_special = [col for col in special_cols if col in matrix.columns]
    current_sectors = [col for col in matrix.columns if col not in special_cols]
    
    # Add missing sector columns
    added = [s for s in all_sectors if s not in current_sectors]
    if added:
        print(f"    Adding sectors: {', '.join(sorted(added)[:5])}...")
    
    for sector in all_sectors:
        if sector not in matrix.columns:
            matrix[sector] = 0.0
    
    # Build column order
    new_cols = []
    if 'label' in current_special:
        new_cols.append('label')
    
    # Add all sectors in sorted order
    new_cols.extend(sorted(all_sectors))
    
    # ALWAYS compute and add Total
    matrix['Total'] = matrix[all_sectors].sum(axis=1)
    new_cols.append('Total')
    
    return matrix[new_cols]

def sfc_sign(instr: str, imap: dict) -> int:
    """Get sign for instrument based on side classification."""
    side = imap.get(instr, {}).get('side', 'macro')
    if side == 'asset':
        return -1
    elif side == 'liability':
        return +1
    else:  # 'macro' or unknown
        return 0

def prep_save_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Prepare matrix for saving to CSV with proper formatting."""
    out = mat.copy()
    
    if out.index.name != 'instrument':
        out.index.name = 'instrument'
    
    out = out.reset_index()
    out['instrument'] = out['instrument'].astype(str).str.zfill(5)
    
    return out

# ========== MATRIX BUILDERS ==========

def build_balance_sheet(tidy: pd.DataFrame, imap: dict, qdate: pd.Timestamp, all_sectors: list = None) -> pd.DataFrame:
    """Build balance sheet matrix from FL (level) series."""
    z = tidy[(tidy['kind'] == 'FL') & (tidy['date'] == qdate)].copy()
    
    print(f"\nBuilding Balance Sheet:")
    print(f"  FL entries: {len(z):,}")
    
    if len(z) == 0:
        print("  ⚠ No FL data found!")
        return pd.DataFrame()
    
    mat = z.pivot_table(
        index='instrument', 
        columns='sector', 
        values='value', 
        aggfunc='sum'
    ).fillna(0.0)
    
    mat.insert(0, 'label', mat.index.map(lambda k: imap.get(k, {}).get('label', '')))
    
    if all_sectors:
        mat = ensure_all_sectors(mat, all_sectors)
    else:
        sector_cols = [c for c in mat.columns if c != 'label']
        mat['Total'] = mat[sector_cols].sum(axis=1)
    
    print(f"  Matrix shape: {mat.shape}")
    return mat.sort_index()

def build_transactions_financial(tidy: pd.DataFrame, imap: dict, qdate: pd.Timestamp, all_sectors: list = None) -> pd.DataFrame:
    """Build financial transactions matrix from FU (flow) series."""
    z = tidy[(tidy['kind'] == 'FU') & (tidy['date'] == qdate)].copy()
    
    print(f"\nBuilding Transaction Matrix:")
    print(f"  FU entries: {len(z):,}")
    
    if len(z) == 0:
        print("  ⚠ No FU data found!")
        return pd.DataFrame()
    
    z['signed_value'] = z.apply(
        lambda row: sfc_sign(row['instrument'], imap) * row['value'],
        axis=1
    )
    
    non_zero = (z['signed_value'] != 0).sum()
    zeros = (z['signed_value'] == 0).sum()
    print(f"  Non-zero values: {non_zero:,}, Zero values: {zeros:,}")
    
    mat = z.pivot_table(
        index='instrument',
        columns='sector',
        values='signed_value',
        aggfunc='sum'
    ).fillna(0.0)
    
    mat.insert(0, 'label', mat.index.map(lambda k: imap.get(k, {}).get('label', '')))
    
    if all_sectors:
        mat = ensure_all_sectors(mat, all_sectors)
    else:
        sector_cols = [c for c in mat.columns if c != 'label']
        mat['Total'] = mat[sector_cols].sum(axis=1)
    
    print(f"  Matrix shape: {mat.shape}")
    return mat

def add_macro_flows(trans_mat: pd.DataFrame, tidy: pd.DataFrame, fmap: dict, qdate: pd.Timestamp, all_sectors: list = None) -> pd.DataFrame:
    """Add macro flows to transaction matrix."""
    if trans_mat.empty:
        print("\n⚠ Cannot add macro flows to empty transaction matrix")
        return trans_mat
    
    print(f"\nAdding Macro Flows:")
    
    if all_sectors:
        sector_cols = all_sectors
        trans_mat = ensure_all_sectors(trans_mat, all_sectors)
    else:
        sector_cols = [c for c in trans_mat.columns if c not in ('label', 'Total')]
    
    rows = []
    added_count = 0
    
    for code, meta in fmap.items():
        blk = tidy[(tidy['date'] == qdate) & (tidy['instrument'] == code)]
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
        added_count += 1
    
    print(f"  Added {added_count} macro flows")
    
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
    print(f"  Final shape: {result.shape}")
    
    return result

def recon_stock_flow(tidy: pd.DataFrame, qdate: pd.Timestamp) -> pd.DataFrame:
    """Reconcile stocks and flows."""
    print(f"\nStock-Flow Reconciliation:")
    
    t = tidy[tidy['date'] == qdate]
    prev_dates = tidy[tidy['date'] < qdate]['date'].unique()
    
    if len(prev_dates) == 0:
        print("  ⚠ No prior period for reconciliation")
        return pd.DataFrame()
    
    prev_date = max(prev_dates)
    prev = tidy[tidy['date'] == prev_date]
    
    print(f"  Period: {prev_date.date()} → {qdate.date()}")
    
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
    
    result = pd.DataFrame(recon)
    print(f"  Entries: {len(result):,}")
    
    return result

# ========== EXECUTION MODES ==========

def run_baseline(sfc: dict, tidy: pd.DataFrame):
    """Run baseline mode with consistent sector columns and proper formatting."""
    print("\n" + "="*60)
    print("BASELINE MODE EXECUTION")
    print("="*60)
    
    imap = load_json(sfc["instrument_map"])
    fmap = load_json(sfc["flow_map"])
    qdate = pd.to_datetime(sfc["date"])
    outdir = Path(sfc["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Analysis date: {qdate.date()}")
    print(f"  Instruments: {len(imap)}")
    print(f"  Flow mappings: {len(fmap)}")
    
    # Get all sectors for this date
    all_sectors = get_all_sectors(tidy, qdate)
    
    if not all_sectors:
        print("\n⚠ ERROR: No sectors found for this date!")
        return
    
    print(f"  Sectors: {len(all_sectors)} total")
    
    # Build matrices with consistent sectors
    bs = build_balance_sheet(tidy, imap, qdate, all_sectors)
    tf_fin = build_transactions_financial(tidy, imap, qdate, all_sectors)
    
    if not tf_fin.empty:
        tf_full = add_macro_flows(tf_fin, tidy, fmap, qdate, all_sectors)
    else:
        tf_full = tf_fin
        print("\n⚠ Transaction matrix is empty")
    
    rc = recon_stock_flow(tidy, qdate)
    
    # Verify consistency
    if not bs.empty and not tf_full.empty:
        bs_sectors = set(c for c in bs.columns if c not in ['label', 'Total'])
        tf_sectors = set(c for c in tf_full.columns if c not in ['label', 'Total'])
        
        if bs_sectors == tf_sectors:
            print(f"\n✓ Sectors match: {len(bs_sectors)} sectors in both matrices")
        else:
            print(f"\n⚠ Sector mismatch!")
    
    # Save outputs
    tag = qdate.date().isoformat()
    
    print(f"\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    if not bs.empty:
        bs_file = outdir / f'sfc_balance_sheet_{tag}.csv'
        bs_out = prep_save_matrix(bs)
        bs_out.to_csv(bs_file, index=False)
        print(f"  ✓ Balance sheet: {bs_file.name}")
    
    if not tf_full.empty:
        tf_file = outdir / f'sfc_transactions_{tag}.csv'
        tf_out = prep_save_matrix(tf_full)
        tf_out.to_csv(tf_file, index=False)
        print(f"  ✓ Transactions: {tf_file.name}")
    
    if not rc.empty:
        rc_file = outdir / f'sfc_recon_{tag}.csv'
        if 'instrument' in rc.columns:
            rc['instrument'] = rc['instrument'].astype(str).str.zfill(5)
        rc.to_csv(rc_file, index=False)
        print(f"  ✓ Reconciliation: {rc_file.name}")
    
    print(f"\n✅ Complete!")

def run_fwtw(sfc: dict, tidy: pd.DataFrame):
    """FWTW mode - placeholder."""
    print("\nFWTW mode not yet implemented")
    print("This will integrate From-Whom-To-Whom bilateral data")

def run_graph(sfc: dict, tidy: pd.DataFrame):
    """Graph mode - placeholder."""
    print("\nGraph mode not yet implemented")
    print("This will use graph-based allocation methods")

def run_refined(sfc: dict, tidy: pd.DataFrame):
    """Refined mode - placeholder."""
    print("\nRefined mode not yet implemented")
    print("This will apply temporal priors and instrument subgroups")

# ========== MAIN ==========

def main():
    ap = argparse.ArgumentParser(description="SFC core runner")
    ap.add_argument("mode", choices=["baseline", "fwtw", "graph", "refined"])
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    args = ap.parse_args()

    try:
        sfc = load_cfg(args.config)
        tidy = load_z1_panel(sfc)
        
        if args.mode == "baseline":
            run_baseline(sfc, tidy)
        elif args.mode == "fwtw":
            run_fwtw(sfc, tidy)
        elif args.mode == "graph":
            run_graph(sfc, tidy)
        elif args.mode == "refined":
            run_refined(sfc, tidy)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
