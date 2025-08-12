#!/usr/bin/env python3
"""
Graph-only W2W reconstruction from Z.1 marginals using masked entropic OT (Sinkhorn).
Requires:
  - instrument_map.json  (maps instrument code -> class & side)
  - graph_adjacency_spec.json  (holders, issuers, allowed edges per instrument class)

Outputs:
  - w2w_graph_LONG_YYYY-MM-DD.csv  (bilateral flows by instrument class)
  - w2w_graph_{class}_YYYY-MM-DD.csv (matrix per class)
  - plus standard SFC matrices (balance sheet, transactions, recon)
"""
from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.config_manager import ConfigManager
from src.utils.helpers import ensure_dir
from src.utils.z1_series_interpreter import Z1Series
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import pivot_series

from graph_flow_allocator import allocate_flows_from_graph


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
    if p is None: return None
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

def build_marginals_by_class(tidy_df: pd.DataFrame, imap: dict, date: pd.Timestamp):
    t = tidy_df[(tidy_df['kind']=='FU') & (tidy_df['date']==date)].copy()
    t['side']  = t['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    t['class'] = t['instrument'].map(lambda k: imap.get(k,{}).get('class','Other'))
    assets = t[t['side']=='asset'].groupby(['class','sector'])['value'].sum().rename('uses').reset_index()
    liabs  = t[t['side']=='liability'].groupby(['class','sector'])['value'].sum().rename('sources').reset_index()
    return assets, liabs

def to_matrix(r_index, c_index, edges, cost_hints, default_tau, default_penalty):
    import numpy as np
    A = np.zeros((len(r_index), len(c_index)))
    C = np.zeros_like(A)
    # allowed edges
    for (i,j) in edges:
        if i in r_index and j in c_index:
            A[r_index.index(i), c_index.index(j)] = 1.0
    # costs: small penalty for disallowed handled by adjacency; here only set relative prefs
    prefer = cost_hints.get('prefer', []) if cost_hints else []
    penal  = cost_hints.get('penalize', []) if cost_hints else []
    for (i,j) in prefer:
        if i in r_index and j in c_index:
            C[r_index.index(i), c_index.index(j)] -= default_penalty  # cheaper
    for (i,j) in penal:
        if i in r_index and j in c_index:
            C[r_index.index(i), c_index.index(j)] += default_penalty  # more expensive
    return A, C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/proper_sfc_config.yaml')
    ap.add_argument('--date', required=True)
    ap.add_argument('--instrument_map', required=True)
    ap.add_argument('--graph_spec', required=True)
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    base_dir, cache_dir, outdir_cfg = load_paths_from_yaml(args.config)
    loader = CachedFedDataLoader(base_directory=base_dir, cache_directory=cache_dir)
    panel = pivot_series(loader.load_all_series_panel())
    panel['date'] = pd.to_datetime(panel['date'])
    tidy = tidy_panel(panel)

    with open(args.instrument_map,'r',encoding='utf-8') as f:
        imap = json.load(f)
    with open(args.graph_spec,'r',encoding='utf-8') as f:
        spec = json.load(f)

    qdate = pd.to_datetime(args.date)
    # Build standard tables (optional, useful for checks)
    bs = build_balance_sheet(tidy, imap, qdate)
    tf_fin = build_transactions_financial(tidy, imap, qdate)
    rc = recon_stock_flow(tidy, qdate)

    # Marginals by class
    assets, liabs = build_marginals_by_class(tidy, imap, qdate)

    # Graph nodes
    holders = spec['nodes']['holders']
    issuers = spec['nodes']['issuers']

    # Collect graph allocations per class
    long_rows = []
    default_tau = spec.get('default',{}).get('tau', 2.0)
    default_penalty = spec.get('default',{}).get('penalty', 2.0)

    classes = list(spec['groups'].keys())
    for g in classes:
        # targets
        r = assets[assets['class']==g].set_index('sector')['uses'].reindex(holders).fillna(0.0).values
        c = liabs[liabs['class']==g].set_index('sector')['sources'].reindex(issuers).fillna(0.0).values

        # adjacency & cost
        edges = spec['groups'][g]['edges']
        cost_hints = spec['groups'][g].get('cost_hints', {})
        tau = spec['groups'][g].get('tau', default_tau)
        A, C = to_matrix(holders, issuers, edges, cost_hints, default_tau, default_penalty)

        # allocate
        X, info = allocate_flows_from_graph(r, c, A, cost=C, tau=tau)
        # build long
        for i,h in enumerate(holders):
            for j,iss in enumerate(issuers):
                val = float(X[i,j])
                if val!=0.0:
                    long_rows.append({"instrument_group": g, "holder_sector": h, "issuer_sector": iss, "flow": val})

    long_df = pd.DataFrame(long_rows)
    # Save outputs
    outdir = Path(args.outdir or (outdir_cfg or 'outputs'))
    tag = qdate.date().isoformat()
    bs.to_csv(outdir / f'sfc_balance_sheet_{tag}.csv')
    tf_fin.to_csv(outdir / f'sfc_transactions_{tag}.csv')
    rc.to_csv(outdir / f'sfc_recon_{tag}.csv', index=False)
    long_df.to_csv(outdir / f'w2w_graph_LONG_{tag}.csv', index=False)
    for grp, g in long_df.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        mat.to_csv(outdir / f'w2w_graph_{grp}_{tag}.csv')
    print('Saved graph-based bilateral flows and SFC matrices into', outdir)

if __name__=='__main__':
    main()
