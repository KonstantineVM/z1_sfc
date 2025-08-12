#!/usr/bin/env python3
"""
Upgraded SFC driver: builds balance-sheet, transaction-flows, bilateral flows,
using instrument_map.json (all instruments) and flow_map.json (macro/behavioral flows).
"""
from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import os

from src.utils.config_manager import ConfigManager
from src.utils.helpers import ensure_dir
from src.utils.z1_series_interpreter import Z1Series
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import pivot_series


# --- flexible config loader (supports top-level and data.*) ---
import yaml
def load_paths_from_yaml(cfg_path:str):
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f) or {}
    # Try nested schema
    data = cfg.get('data', {})
    base = data.get('input_path') or data.get('base_dir')
    cache = data.get('cache_dir')
    out = data.get('output_dir')
    # Fall back to top-level schema
    base = base or cfg.get('base_dir')
    cache = cache or cfg.get('cache_dir')
    out = out or cfg.get('output_dir')
    if not base:
        raise ValueError("Config must define base_dir (top-level) or data.input_path/data.base_dir")
    return base, cache, out


def parse_series(series: str):
    p = Z1Series.parse(series)
    if p is None:
        return None
    return p.prefix, p.sector, p.instrument

def tidy_panel(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df['series'].apply(parse_series)
    mask = parsed.notnull()
    out = df.loc[mask].copy()
    out[['kind','sector','instrument']] = pd.DataFrame(parsed[mask].tolist(), index=out.index)
    return out

def build_balance_sheet(tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp) -> pd.DataFrame:
    z = tidy_df[(tidy_df['kind']=='FL') & (tidy_df['date']==date)].copy()
    mat = z.pivot_table(index='instrument', columns='sector', values='value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat.sort_index()

def sfc_sign(instr: str, imap: dict) -> int:
    side = imap.get(instr,{}).get('side','macro')
    if side=='asset': return -1
    if side=='liability': return +1
    return 0

def build_transactions_financial(tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp) -> pd.DataFrame:
    z = tidy_df[(tidy_df['kind']=='FU') & (tidy_df['date']==date)].copy()
    z['signed_value'] = z['instrument'].map(lambda k: sfc_sign(k,imap)) * z['value']
    mat = z.pivot_table(index='instrument', columns='sector', values='signed_value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat

def add_macro_flows(trans_mat: pd.DataFrame, tidy_df: pd.DataFrame, flow_map: dict, date: pd.Timestamp) -> pd.DataFrame:
    """Insert macro flows (consumption, wages, taxes, transfers, etc.) as additional rows.
       Each flow is identified by its instrument code key in flow_map; we take the value from the panel by
       summing that instrument's FU at date across sectors (Z.1 often stores such flows under sector 08/01 etc.).
       Then we distribute into sector columns using 'sign_by_sector'. If the map specifies 'from'/'to' only,
       we place signs accordingly (+ for recipients, - for payers)."""
    # Current columns except label/Total
    sector_cols = [c for c in trans_mat.columns if c not in ('label','Total')]
    rows = []
    for code, meta in flow_map.items():
        # Pull the magnitude. Prefer FU; if not present, try any kind and take value.
        block = tidy_df[(tidy_df['date']==date) & (tidy_df['instrument']==code)]
        if block.empty:
            continue
        val = block['value'].sum()
        row = {c: 0.0 for c in sector_cols}
        sbs = meta.get('sign_by_sector', {})
        if sbs:
            for sec, s in sbs.items():
                if sec in row:
                    row[sec] += s * val
        else:
            # fallback: split equally among 'to' with + and 'from' with -
            tos = [s for s in meta.get('to',[]) if s in row]
            frs = [s for s in meta.get('from',[]) if s in row]
            n_to = len(tos) or 1
            n_fr = len(frs) or 1
            for sec in tos: row[sec] += val/n_to
            for sec in frs: row[sec] -= val/n_fr
        row['label'] = meta.get('label', f'Flow {code}')
        rows.append((code,row))
    if not rows:
        return trans_mat
    # Build DF
    add_df = pd.DataFrame.from_dict({k:v for k,v in rows}, orient='index')
    # Ensure all sector cols exist
    for c in sector_cols:
        if c not in add_df.columns:
            add_df[c]=0.0
    add_df = add_df[ ['label'] + sector_cols ]
    add_df['Total'] = add_df[sector_cols].sum(axis=1)
    # Append to existing matrix (keep numeric index as code strings)
    out = pd.concat([trans_mat, add_df], axis=0)
    return out

def recon_stock_flow(tidy_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    t = tidy_df[tidy_df['date']==date]
    prev = tidy_df[tidy_df['date']<date]
    if prev.empty:
        raise ValueError('No prior quarter for ΔFL.')
    prev_date = prev['date'].max()
    fl_t = t[t['kind']=='FL'].set_index(['sector','instrument'])['value']
    fl_tm1 = tidy_df[(tidy_df['kind']=='FL') & (tidy_df['date']==prev_date)].set_index(['sector','instrument'])['value']
    fu = t[t['kind']=='FU'].set_index(['sector','instrument'])['value']
    fr = t[t['kind']=='FR'].set_index(['sector','instrument'])['value']
    fv = t[t['kind']=='FV'].set_index(['sector','instrument'])['value']
    idx = fl_t.index.union(fl_tm1.index).union(fu.index).union(fr.index).union(fv.index)
    df = pd.DataFrame(index=idx)
    df['dFL'] = fl_t.reindex(idx,fill_value=0)-fl_tm1.reindex(idx,fill_value=0)
    df['FU']  = fu.reindex(idx,fill_value=0)
    df['FR']  = fr.reindex(idx,fill_value=0)
    df['FV']  = fv.reindex(idx,fill_value=0)
    df['Gap'] = df['dFL']-(df['FU']+df['FR']+df['FV'])
    return df.reset_index().sort_values(['sector','instrument'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/proper_sfc_config.yaml')
    ap.add_argument('--date', required=True, help='Quarter end date, YYYY-MM-DD')
    ap.add_argument('--instrument_map', default='instrument_map.json')
    ap.add_argument('--flow_map', default='flow_map.json')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    base_dir, cache_dir, outdir_cfg = load_paths_from_yaml(args.config)

    # Try to read a base directory from the config; fall back to a sensible default.
    base_dir = getattr(cfg, "base_directory", None)
    if isinstance(base_dir, dict):
        base_dir = base_dir.get("base_directory")
    if not isinstance(base_dir, (str, bytes)) or not base_dir:
        base_dir = getattr(cfg, "data_dir", None) or getattr(cfg, "cache_dir", None) or "data_cache"

    os.makedirs(base_dir, exist_ok=True)
    loader = CachedFedDataLoader(base_directory=base_dir, cache_directory=cache_dir)
    # Load all relevant datasets from Fed
    # If you need both Z1 and FWTW for full SFC construction, include them here
    data_dict = loader.load_multiple_sources(['Z1', 'FWTW'], force_download=False)

    # The Z1 dataset is the authoritative source for sector/instrument data
    if 'Z1' not in data_dict:
        raise RuntimeError("Z1 dataset could not be loaded by CachedFedDataLoader.")

    # Convert Z1 dataset to tidy long format for SFC processing
    panel = pivot_series(data_dict['Z1'])

    # Optionally: keep FWTW panel if needed later for flow approximations
    fwtw_panel = None
    if 'FWTW' in data_dict:
        fwtw_panel = pivot_series(data_dict['FWTW'])

    panel['date'] = pd.to_datetime(panel['date'])
    tidy = tidy_panel(panel)

    with open(args.instrument_map,'r',encoding='utf-8') as f:
        imap = json.load(f)
    with open(args.flow_map,'r',encoding='utf-8') as f:
        fmap = json.load(f)

    qdate = pd.to_datetime(args.date)
    bs = build_balance_sheet(tidy, imap, qdate)
    tf_fin = build_transactions_financial(tidy, imap, qdate)
    tf_full = add_macro_flows(tf_fin, tidy, fmap, qdate)
    rc = recon_stock_flow(tidy, qdate)

    outdir = Path(args.outdir or (outdir_cfg or 'outputs'))
    tag = qdate.date().isoformat()
    bs.to_csv(outdir / f'sfc_balance_sheet_{tag}.csv')
    tf_full.to_csv(outdir / f'sfc_transactions_{tag}.csv')
    rc.to_csv(outdir / f'sfc_recon_{tag}.csv', index=False)

    print('Saved:')
    print(' ', outdir / f'sfc_balance_sheet_{tag}.csv')
    print(' ', outdir / f'sfc_transactions_{tag}.csv')
    print(' ', outdir / f'sfc_recon_{tag}.csv')

if __name__=='__main__':
    main()
