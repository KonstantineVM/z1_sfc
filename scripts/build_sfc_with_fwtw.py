#!/usr/bin/env python3
"""
SFC builder with FWTW integration (stocks -> flows) and reconciliation to FU totals.

Inputs:
  - instrument_map.json          (full instrument metadata)
  - flow_map.json                (macro/behavioral flows)
  - FWTW tidy panel (CSV/Parquet): ['date','instrument_group','holder_sector','issuer_sector','level']

Outputs (per --outdir):
  - sfc_balance_sheet_YYYY-MM-DD.csv            (FL matrix)
  - sfc_transactions_YYYY-MM-DD.csv             (FU + macro flows matrix)
  - sfc_recon_YYYY-MM-DD.csv                    (ΔFL vs FU+FR+FV)
  - w2w_flows_LONG_YYYY-MM-DD.csv               (bilateral flows, stacked)
  - w2w_flows_{group}_YYYY-MM-DD.csv            (pivoted matrices by group)

Reconciliation:
  - For each instrument_group, we compute ΔW2W := level_t - level_{t-1}.
  - Row sums (by holder) are compared to FU(asset) for that group; column sums (by issuer) to FU(liability).
  - Optional RAS scaling adjusts ΔW2W to match both marginals.
"""

from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

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

def sfc_sign(instr: str, imap: dict) -> int:
    side = imap.get(instr,{}).get('side','macro')
    if side=='asset': return -1
    if side=='liability': return +1
    return 0

def build_balance_sheet(tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp) -> pd.DataFrame:
    z = tidy_df[(tidy_df['kind']=='FL') & (tidy_df['date']==date)].copy()
    mat = z.pivot_table(index='instrument', columns='sector', values='value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat.sort_index()

def build_transactions_financial(tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp) -> pd.DataFrame:
    z = tidy_df[(tidy_df['kind']=='FU') & (tidy_df['date']==date)].copy()
    z['signed_value'] = z['instrument'].map(lambda k: sfc_sign(k,imap)) * z['value']
    mat = z.pivot_table(index='instrument', columns='sector', values='signed_value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat

def add_macro_flows(trans_mat: pd.DataFrame, tidy_df: pd.DataFrame, flow_map: dict, date: pd.Timestamp) -> pd.DataFrame:
    sector_cols = [c for c in trans_mat.columns if c not in ('label','Total')]
    rows = []
    for code, meta in flow_map.items():
        blk = tidy_df[(tidy_df['date']==date) & (tidy_df['instrument']==code)]
        if blk.empty: 
            continue
        val = blk['value'].sum()
        row = {c: 0.0 for c in sector_cols}
        sbs = meta.get('sign_by_sector', {})
        if sbs:
            for sec, s in sbs.items():
                if sec in row:
                    row[sec] += s * val
        else:
            tos = [s for s in meta.get('to',[]) if s in row]
            frs = [s for s in meta.get('from',[]) if s in row]
            n_to = len(tos) or 1; n_fr = len(frs) or 1
            for sec in tos: row[sec] += val/n_to
            for sec in frs: row[sec] -= val/n_fr
        row['label'] = meta.get('label', f'Flow {code}')
        rows.append((code,row))
    if not rows: return trans_mat
    add_df = pd.DataFrame.from_dict({k:v for k,v in rows}, orient='index')
    for c in sector_cols:
        if c not in add_df.columns: add_df[c]=0.0
    add_df = add_df[ ['label'] + sector_cols ]
    add_df['Total'] = add_df[sector_cols].sum(axis=1)
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

# ---------- FWTW integration ----------

def ras_balance(matrix: pd.DataFrame, row_targets: pd.Series, col_targets: pd.Series, tol=1e-8, max_iter=1000):
    """RAS/bi-proportional scaling to match row and column sums (nonnegative)."""
    A = matrix.values.astype(float)
    r = row_targets.reindex(matrix.index).fillna(0).values
    c = col_targets.reindex(matrix.columns).fillna(0).values
    # Avoid negatives for RAS; clip
    A = np.maximum(A, 0.0)
    for _ in range(max_iter):
        # Row scaling
        row_sums = A.sum(axis=1); row_sums[row_sums==0] = 1.0
        A *= (r/row_sums)[:,None]
        # Column scaling
        col_sums = A.sum(axis=0); col_sums[col_sums==0] = 1.0
        A *= (c/col_sums)[None,:]
        # Check convergence
        if np.allclose(A.sum(axis=1), r, atol=tol) and np.allclose(A.sum(axis=0), c, atol=tol):
            break
    out = pd.DataFrame(A, index=matrix.index, columns=matrix.columns)
    return out

def build_w2w_flows(fwtw_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Compute Δ levels by instrument_group, holder, issuer -> bilateral flows (long form)."""
    fwtw_df = fwtw_df.copy()
    fwtw_df['date'] = pd.to_datetime(fwtw_df['date'])
    t = fwtw_df[fwtw_df['date']==date]
    prev = fwtw_df[fwtw_df['date'] < date]
    if prev.empty:
        raise ValueError('No prior FWTW quarter for Δ computation.')
    prev_date = prev['date'].max()
    L_t = t.set_index(['instrument_group','holder_sector','issuer_sector'])['level']
    L_tm1 = fwtw_df[fwtw_df['date']==prev_date].set_index(['instrument_group','holder_sector','issuer_sector'])['level']
    idx = L_t.index.union(L_tm1.index)
    d = L_t.reindex(idx, fill_value=0) - L_tm1.reindex(idx, fill_value=0)
    out = d.rename('flow').reset_index()
    return out

def reconcile_w2w_to_fu(w2w_long: pd.DataFrame, tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp, do_ras=True):
    """For each instrument_group, balance bilateral flows to FU marginals by sector (assets/liabilities)."""
    # Build FU marginals by sector and by instrument group using instrument_map classes
    t = tidy_df[(tidy_df['kind']=='FU') & (tidy_df['date']==date)].copy()
    t['side'] = t['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    t['class'] = t['instrument'].map(lambda k: imap.get(k,{}).get('class','Other'))
    # Aggregate FU to 'instrument_group' ~ here we use 'class' as proxy; you can refine mapping
    # Asset acquisitions (uses, negative); liability incurrence (sources, positive)
    fu_assets = t[t['side']=='asset'].groupby(['class','sector'])['value'].sum().rename('FU_assets')
    fu_liabs  = t[t['side']=='liability'].groupby(['class','sector'])['value'].sum().rename('FU_liabs')
    results = []
    for grp, gdf in w2w_long.groupby('instrument_group'):
        mat = gdf.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        # Targets (magnitudes) from FU by class=grp (you may need a dedicated mapping)
        rows_target = fu_assets.xs(grp, level='class', drop_level=False).droplevel('class', axis=0) if grp in fu_assets.index.get_level_values(0) else pd.Series(dtype=float)
        cols_target = fu_liabs.xs(grp, level='class', drop_level=False).droplevel('class', axis=0) if grp in fu_liabs.index.get_level_values(0) else pd.Series(dtype=float)
        # Align index/cols
        rows_target = rows_target.reindex(mat.index).fillna(0).abs()  # uses magnitude
        cols_target = cols_target.reindex(mat.columns).fillna(0).abs()
        # Optional RAS
        mat_bal = ras_balance(mat.clip(lower=0.0), rows_target, cols_target) if do_ras else mat
        # Return long
        long_bal = mat_bal.stack().rename('flow').reset_index()
        long_bal['instrument_group'] = grp
        results.append(long_bal[['instrument_group','holder_sector','issuer_sector','flow']])
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=['instrument_group','holder_sector','issuer_sector','flow'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/proper_sfc_config.yaml')
    ap.add_argument('--date', required=True)
    ap.add_argument('--instrument_map', required=True)
    ap.add_argument('--flow_map', required=True)
    ap.add_argument('--fwtw', required=True, help='Path to tidy FWTW levels (CSV/Parquet)')
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--no_ras', action='store_true', help='Disable RAS balancing')
    args = ap.parse_args()

    base_dir, cache_dir, outdir_cfg = load_paths_from_yaml(args.config)
    loader = CachedFedDataLoader(base_directory=base_dir, cache_directory=cache_dir)
    panel = pivot_series(loader.load_all_series_panel())
    panel['date'] = pd.to_datetime(panel['date'])
    tidy = tidy_panel(panel)

    with open(args.instrument_map,'r',encoding='utf-8') as f:
        imap = json.load(f)
    with open(args.flow_map,'r',encoding='utf-8') as f:
        fmap = json.load(f)

    qdate = pd.to_datetime(args.date)

    # Core matrices (as before)
    bs = build_balance_sheet(tidy, imap, qdate)
    tf_fin = build_transactions_financial(tidy, imap, qdate)
    tf_full = add_macro_flows(tf_fin, tidy, fmap, qdate)
    rc = recon_stock_flow(tidy, qdate)

    # FWTW: read and compute Δ
    fwtw_path = Path(args.fwtw)
    if fwtw_path.suffix.lower() in ('.parquet','.pq','.pqt'):
        fwtw = pd.read_parquet(fwtw_path)
    else:
        fwtw = pd.read_csv(fwtw_path)
    # standardize expected columns
    rename = {'value':'level'}
    for k,v in rename.items():
        if k in fwtw.columns and v not in fwtw.columns:
            fwtw[v] = fwtw[k]
    required = {'date','instrument_group','holder_sector','issuer_sector','level'}
    missing = required - set(fwtw.columns)
    if missing:
        raise ValueError(f'FWTW file missing columns: {missing}')
    w2w_long = build_w2w_flows(fwtw, qdate)

    # Reconcile to FU marginals by instrument class (proxy for group)
    w2w_bal = reconcile_w2w_to_fu(w2w_long, tidy, imap, qdate, do_ras=(not args.no_ras))

    # Save outputs
    outdir = Path(args.outdir or (outdir_cfg or 'outputs'))
    tag = qdate.date().isoformat()
    bs.to_csv(outdir / f'sfc_balance_sheet_{tag}.csv')
    tf_full.to_csv(outdir / f'sfc_transactions_{tag}.csv')
    rc.to_csv(outdir / f'sfc_recon_{tag}.csv', index=False)
    # long
    w2w_bal.to_csv(outdir / f'w2w_flows_LONG_{tag}.csv', index=False)
    # matrices per group
    for grp, g in w2w_bal.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        mat.to_csv(outdir / f'w2w_flows_{grp}_{tag}.csv')

    print('Saved:')
    print(' ', outdir / f'sfc_balance_sheet_{tag}.csv')
    print(' ', outdir / f'sfc_transactions_{tag}.csv')
    print(' ', outdir / f'sfc_recon_{tag}.csv')
    print(' ', outdir / f'w2w_flows_LONG_{tag}.csv')
    print('   + per-group CSVs in', outdir)

if __name__=='__main__':
    main()
