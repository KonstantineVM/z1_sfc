#!/usr/bin/env python3
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

def load_maps(instr_map_path: str, instr_group_map_path: str):
    with open(instr_map_path,'r',encoding='utf-8') as f:
        imap = json.load(f)
    with open(instr_group_map_path,'r',encoding='utf-8') as f:
        gmap = json.load(f)
    return imap, {k:v.get('subgroup','Other:Misc') for k,v in gmap.items()}

def sfc_sign(instr: str, imap: dict) -> int:
    side = imap.get(instr,{}).get('side','macro')
    if side=='asset': return -1
    if side=='liability': return +1
    return 0

def build_marginals_by_subgroup(tidy_df: pd.DataFrame, imap: dict, subgroup_map: dict, date: pd.Timestamp):
    t = tidy_df[(tidy_df['kind']=='FU') & (tidy_df['date']==date)].copy()
    t['side']  = t['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    t['subgroup'] = t['instrument'].map(lambda k: subgroup_map.get(k, 'Other:Misc'))
    assets = t[t['side']=='asset'].groupby(['subgroup','sector'])['value'].sum().rename('uses').reset_index()
    liabs  = t[t['side']=='liability'].groupby(['subgroup','sector'])['value'].sum().rename('sources').reset_index()
    return assets, liabs

def build_prior_matrix(prior_file: Path, holders, issuers):
    if not prior_file.exists():
        return None
    df = pd.read_csv(prior_file)
    if set(['holder_sector','issuer_sector','flow']) - set(df.columns):
        return None
    mat = df.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').reindex(index=holders, columns=issuers).fillna(0.0)
    # Convert to shares to use as prior weights
    total = mat.values.sum()
    if total <= 0: return None
    P = mat.values / max(total, 1e-30)
    return P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/proper_sfc_config.yaml')
    ap.add_argument('--date', required=True)
    ap.add_argument('--instrument_map', required=True)
    ap.add_argument('--instrument_group_map', required=True)
    ap.add_argument('--graph_spec', required=True)
    ap.add_argument('--prior_dir', default=None, help='Directory with prior w2w_graph_{group}_{prevdate}.csv files')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    base_dir, cache_dir, outdir_cfg = load_paths_from_yaml(args.config)
    loader = CachedFedDataLoader(base_directory=base_dir, cache_directory=cache_dir)
    panel = pivot_series(loader.load_all_series_panel())
    panel['date'] = pd.to_datetime(panel['date'])
    tidy = tidy_panel(panel)

    imap, subgroup_map = load_maps(args.instrument_map, args.instrument_group_map)
    with open(args.graph_spec,'r',encoding='utf-8') as f:
        spec = json.load(f)

    qdate = pd.to_datetime(args.date)
    prev_date = panel.loc[panel['date'] < qdate, 'date'].max() if (panel['date'] < qdate).any() else None

    # Marginals by subgroup
    assets, liabs = build_marginals_by_subgroup(tidy, imap, subgroup_map, qdate)

    holders = spec['nodes']['holders']
    issuers = spec['nodes']['issuers']

    default_tau = spec.get('default',{}).get('tau', 2.0)
    default_penalty = spec.get('default',{}).get('penalty', 2.0)

    long_rows = []
    for group, params in spec['groups'].items():
        r = assets[assets['subgroup']==group].set_index('sector')['uses'].reindex(holders).fillna(0.0).values
        c = liabs[liabs['subgroup']==group].set_index('sector')['sources'].reindex(issuers).fillna(0.0).values

        # Skip if no mass
        if (r.sum() <= 0) and (c.sum() <= 0):
            continue

        # Build adjacency and cost
        edges = params.get('edges', [])
        prefer = params.get('cost_hints',{}).get('prefer',[])
        penal  = params.get('cost_hints',{}).get('penalize',[])
        tau    = params.get('tau', default_tau)

        A = np.zeros((len(holders), len(issuers)))
        C = np.zeros_like(A)
        for (i,j) in edges:
            if i in holders and j in issuers:
                A[holders.index(i), issuers.index(j)] = 1.0
        for (i,j) in prefer:
            if i in holders and j in issuers:
                C[holders.index(i), issuers.index(j)] -= default_penalty
        for (i,j) in penal:
            if i in holders and j in issuers:
                C[holders.index(i), issuers.index(j)] += default_penalty

        # Temporal prior from previous output (if available)
        prior = None
        if args.prior_dir and prev_date is not None:
            prior_file = Path(args.prior_dir) / f"w2w_graph_{group}_{prev_date.date().isoformat()}.csv"
            P = build_prior_matrix(prior_file, holders, issuers)
            if P is not None:
                prior = P

        X, info = allocate_flows_from_graph(r, c, A, cost=C, prior=prior, tau=tau)
        for i,h in enumerate(holders):
            for j,iss in enumerate(issuers):
                val = float(X[i,j])
                if val != 0.0:
                    long_rows.append({"instrument_group": group, "holder_sector": h, "issuer_sector": iss, "flow": val})

    long_df = pd.DataFrame(long_rows)
    outdir = Path(args.outdir or (outdir_cfg or 'outputs'))
    tag = qdate.date().isoformat()
    long_df.to_csv(outdir / f"w2w_graph_LONG_{tag}.csv", index=False)
    # Save per group
    for grp, g in long_df.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        gpath = outdir / f"w2w_graph_{grp}_{tag}.csv"
        mat.to_csv(gpath)

    print("Saved graph-based bilateral flows into", outdir)

if __name__ == "__main__":
    main()
