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
"""

import argparse, json, sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import re

from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor
from src.utils.z1_series_interpreter import Z1Series
from src.alloc.graph_flow_allocator import allocate_flows_from_graph

# ---------- config & IO ----------

def load_cfg(path: str) -> dict:
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


_SERIES_RE = re.compile(
    r'^(?P<kind>[A-Z]{2})'
    r'(?P<sector>\d{2})'
    r'(?P<instr5>\d{5})'
    r'(?P<rest>\d*)'
    r'\.(?P<freq>[A-Z])$'
)
_ALLOWED_KINDS = {"FA","FL","FU","FV","FR","LM"}  # include LM

def parse_series_loose(s: str):
    """
    Loose Z.1 mnemonic parser.
    - Accepts LM in addition to FA/FL/FU/FR/FV.
    - Takes first 5 digits after sector as the instrument (canonical key).
    - Ignores trailing digits ('rest'), but logs them for debug.
    Returns (kind, sector, instrument5, rest) or None if no match.
    """
    m = _SERIES_RE.match(s)
    if not m: 
        return None
    kind = m.group('kind')
    if kind not in _ALLOWED_KINDS:
        return None
    return kind, m.group('sector'), m.group('instr5'), m.group('rest')


def load_z1_panel(sfc: dict) -> pd.DataFrame:
    loader = CachedFedDataLoader(
        base_directory=sfc["base_dir"],
        cache_directory=sfc["cache_dir"],
    )
    z1_raw = loader.load_single_source('Z1')
    z1 = DataProcessor().process_fed_data(z1_raw, 'Z1')

    # 1) Ensure index is named for melt
    if z1.index.name is None:
        z1.index.name = "date"

    # 2) Wide -> long
    df = z1.reset_index().melt(id_vars=["date"], var_name="series", value_name="value")
    df["date"] = pd.to_datetime(df["date"])

    # 3) DEBUG: inspect parsing before expanding
    #    We record which series failed to parse and write them to outputs/debug_bad_series_*.csv
    outdir = Path(sfc["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    def _safe_parse(s):
        try:
            return Z1Series.parse(s)
        except Exception:
            return None

    # --- DEBUG + robust parse ---
    # 1) try loose parser on all series
    parsed = df["series"].apply(parse_series_loose)

    # 2) debug: log non-matching series
    bad_mask = parsed.isnull()
    if bad_mask.any():
        dbg = (df.loc[bad_mask, ["series","value"]]
                 .groupby("series", as_index=False)
                 .agg(bad_rows=("series","size"), example_value=("value","first"))
                 .sort_values("bad_rows", ascending=False))
        dbg_path = Path(sfc["output_dir"]) / f"debug_bad_series_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
        dbg.to_csv(dbg_path, index=False)
        print(f"[DEBUG] Unparsed mnemonics: {bad_mask.sum()} rows. Details: {dbg_path}")

    # 3) expand parsed tuples safely
    exp = parsed.apply(lambda t: t if t is not None else (pd.NA, pd.NA, pd.NA, pd.NA)).apply(pd.Series)
    exp.columns = ["kind","sector","instrument","rest"]

    # 4) keep only rows that parsed and drop the debug-only 'rest' column
    tidy = pd.concat([df, exp], axis=1).dropna(subset=["kind","sector","instrument"]).drop(columns=["rest"])
    return tidy[["date","series","value","kind","sector","instrument"]]



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- SFC primitives ----------

def sfc_sign(instr: str, imap: dict) -> int:
    side = imap.get(instr,{}).get('side','macro')
    if side=='asset': return -1
    if side=='liability': return +1
    return 0

def build_balance_sheet(tidy: pd.DataFrame, imap: dict, qdate: pd.Timestamp) -> pd.DataFrame:
    z = tidy[(tidy['kind']=='FL') & (tidy['date']==qdate)].copy()
    mat = z.pivot_table(index='instrument', columns='sector', values='value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat.sort_index()

def build_transactions_financial(tidy: pd.DataFrame, imap: dict, qdate: pd.Timestamp) -> pd.DataFrame:
    z = tidy[(tidy['kind']=='FU') & (tidy['date']==qdate)].copy()
    z['signed_value'] = z['instrument'].map(lambda k: sfc_sign(k,imap)) * z['value']
    mat = z.pivot_table(index='instrument', columns='sector', values='signed_value', aggfunc='sum').fillna(0.0)
    mat.insert(0,'label', mat.index.map(lambda k: imap.get(k,{}).get('label','')))
    mat['Total'] = mat.drop(columns=['label']).sum(axis=1)
    return mat

def recon_stock_flow(tidy: pd.DataFrame, qdate: pd.Timestamp) -> pd.DataFrame:
    t = tidy[tidy['date']==qdate]
    prev = tidy[tidy['date']<qdate]
    if prev.empty:
        raise ValueError('No prior quarter for ΔFL.')
    prev_date = prev['date'].max()
    fl_t = t[t['kind']=='FL'].set_index(['sector','instrument'])['value']
    fl_tm1 = tidy[(tidy['kind']=='FL') & (tidy['date']==prev_date)].set_index(['sector','instrument'])['value']
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

def add_macro_flows(trans_mat: pd.DataFrame, tidy: pd.DataFrame, fmap: dict, qdate: pd.Timestamp) -> pd.DataFrame:
    sector_cols = [c for c in trans_mat.columns if c not in ('label','Total')]
    rows = []
    for code, meta in fmap.items():
        blk = tidy[(tidy['date']==qdate) & (tidy['instrument']==code)]
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
    return pd.concat([trans_mat, add_df], axis=0)

def ras_balance(matrix: pd.DataFrame, row_targets: pd.Series, col_targets: pd.Series, tol=1e-8, max_iter=1000):
    A = matrix.values.astype(float)
    r = row_targets.reindex(matrix.index).fillna(0).values
    c = col_targets.reindex(matrix.columns).fillna(0).values
    A = np.maximum(A, 0.0)
    for _ in range(max_iter):
        row_sums = A.sum(axis=1); row_sums[row_sums==0] = 1.0
        A *= (r/row_sums)[:,None]
        col_sums = A.sum(axis=0); col_sums[col_sums==0] = 1.0
        A *= (c/col_sums)[None,:]
        if np.allclose(A.sum(axis=1), r, atol=tol) and np.allclose(A.sum(axis=0), c, atol=tol):
            break
    return pd.DataFrame(A, index=matrix.index, columns=matrix.columns)

# ---------- modes ----------

def run_baseline(sfc: dict, tidy: pd.DataFrame):
    imap = load_json(sfc["instrument_map"]); fmap = load_json(sfc["flow_map"])
    qdate = pd.to_datetime(sfc["date"])
    outdir = Path(sfc["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    bs = build_balance_sheet(tidy, imap, qdate)
    tf_fin = build_transactions_financial(tidy, imap, qdate)
    tf_full = add_macro_flows(tf_fin, tidy, fmap, qdate)
    rc = recon_stock_flow(tidy, qdate)
    tag = qdate.date().isoformat()
    bs.to_csv(outdir / f'sfc_balance_sheet_{tag}.csv')
    tf_full.to_csv(outdir / f'sfc_transactions_{tag}.csv')
    rc.to_csv(outdir / f'sfc_recon_{tag}.csv', index=False)
    print("Baseline outputs in", outdir)

def run_fwtw(sfc: dict, tidy: pd.DataFrame):
    for key in ["fwtw"]:
        if not sfc.get(key): raise ValueError(f"sfc.{key} is required for fwtw mode.")
    imap = load_json(sfc["instrument_map"])
    qdate = pd.to_datetime(sfc["date"]); outdir = Path(sfc["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    fwtw_path = Path(sfc["fwtw"])
    if fwtw_path.suffix.lower() in ('.parquet','.pq','.pqt'):
        fwtw = pd.read_parquet(fwtw_path)
    else:
        fwtw = pd.read_csv(fwtw_path)
    if 'value' in fwtw.columns and 'level' not in fwtw.columns:
        fwtw['level'] = fwtw['value']
    required = {'date','instrument_group','holder_sector','issuer_sector','level'}
    missing = required - set(fwtw.columns)
    if missing:
        raise ValueError(f'FWTW file missing columns: {missing}')
    fwtw['date'] = pd.to_datetime(fwtw['date'])
    t = fwtw[fwtw['date']==qdate]
    prev = fwtw[fwtw['date']<qdate]
    if prev.empty: raise ValueError('No prior FWTW quarter for Δ computation.')
    prev_date = prev['date'].max()
    L_t = t.set_index(['instrument_group','holder_sector','issuer_sector'])['level']
    L_tm1 = fwtw[fwtw['date']==prev_date].set_index(['instrument_group','holder_sector','issuer_sector'])['level']
    idx = L_t.index.union(L_tm1.index)
    d = L_t.reindex(idx, fill_value=0) - L_tm1.reindex(idx, fill_value=0)
    w2w = d.rename('flow').reset_index()

    tt = tidy[(tidy['kind']=='FU') & (tidy['date']==qdate)].copy()
    tt['side'] = tt['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    tt['class'] = tt['instrument'].map(lambda k: imap.get(k,{}).get('class','Other'))
    fu_assets = tt[tt['side']=='asset'].groupby(['class','sector'])['value'].sum()
    fu_liabs  = tt[tt['side']=='liability'].groupby(['class','sector'])['value'].sum()

    out_long = []
    for grp, g in w2w.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        rows_t = fu_assets.xs(grp, level='class', drop_level=False).droplevel('class', axis=0) if grp in fu_assets.index.get_level_values(0) else pd.Series(dtype=float)
        cols_t = fu_liabs.xs(grp, level='class', drop_level=False).droplevel('class', axis=0) if grp in fu_liabs.index.get_level_values(0) else pd.Series(dtype=float)
        rows_t = rows_t.reindex(mat.index).fillna(0).abs(); cols_t = cols_t.reindex(mat.columns).fillna(0).abs()
        mat_bal = ras_balance(mat.clip(lower=0.0), rows_t, cols_t)
        long_bal = mat_bal.stack().rename('flow').reset_index(); long_bal['instrument_group'] = grp
        out_long.append(long_bal[['instrument_group','holder_sector','issuer_sector','flow']])
    w2w_bal = pd.concat(out_long, ignore_index=True) if out_long else pd.DataFrame(columns=['instrument_group','holder_sector','issuer_sector','flow'])

    tag = qdate.date().isoformat()
    w2w_bal.to_csv(outdir / f'w2w_flows_LONG_{tag}.csv', index=False)
    for grp, g in w2w_bal.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        mat.to_csv(outdir / f'w2w_flows_{grp}_{tag}.csv')
    print("FWTW outputs in", outdir)

def run_graph(sfc: dict, tidy: pd.DataFrame):
    for key in ["graph_spec"]:
        if not sfc.get(key): raise ValueError(f"sfc.{key} is required for graph mode.")
    imap = load_json(sfc["instrument_map"])
    spec = load_json(sfc["graph_spec"])
    qdate = pd.to_datetime(sfc["date"]); outdir = Path(sfc["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    holders = spec['nodes']['holders']; issuers = spec['nodes']['issuers']
    default_tau = spec.get('default',{}).get('tau', 2.0); default_penalty = spec.get('default',{}).get('penalty', 2.0)

    t = tidy[(tidy['kind']=='FU') & (tidy['date']==qdate)].copy()
    t['side']  = t['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    t['class'] = t['instrument'].map(lambda k: imap.get(k,{}).get('class','Other'))
    assets = t[t['side']=='asset'].groupby(['class','sector'])['value'].sum().rename('uses').reset_index()
    liabs  = t[t['side']=='liability'].groupby(['class','sector'])['value'].sum().rename('sources').reset_index()

    long_rows = []
    for group, params in spec['groups'].items():
        r = assets[assets['class']==group].set_index('sector')['uses'].reindex(holders).fillna(0.0).values
        c = liabs[liabs['class']==group].set_index('sector')['sources'].reindex(issuers).fillna(0.0).values
        edges = params.get('edges', []); prefer = params.get('cost_hints',{}).get('prefer',[]); penal  = params.get('cost_hints',{}).get('penalize',[])
        tau    = params.get('tau', default_tau)
        A = np.zeros((len(holders), len(issuers))); C = np.zeros_like(A)
        for (i,j) in edges:
            if i in holders and j in issuers: A[holders.index(i), issuers.index(j)] = 1.0
        for (i,j) in prefer:
            if i in holders and j in issuers: C[holders.index(i), issuers.index(j)] -= default_penalty
        for (i,j) in penal:
            if i in holders and j in issuers: C[holders.index(i), issuers.index(j)] += default_penalty
        X, info = allocate_flows_from_graph(r, c, A, cost=C, tau=tau)
        for i,h in enumerate(holders):
            for j,iss in enumerate(issuers):
                val = float(X[i,j])
                if val != 0.0:
                    long_rows.append({"instrument_group": group, "holder_sector": h, "issuer_sector": iss, "flow": val})
    long_df = pd.DataFrame(long_rows)
    tag = qdate.date().isoformat()
    long_df.to_csv(outdir / f"w2w_graph_LONG_{tag}.csv", index=False)
    for grp, g in long_df.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        mat.to_csv(outdir / f"w2w_graph_{grp}_{tag}.csv")
    print("Graph outputs in", outdir)

def run_refined(sfc: dict, tidy: pd.DataFrame):
    for key in ["graph_spec_refined","instrument_group_map"]:
        if not sfc.get(key): raise ValueError(f"sfc.{key} is required for refined mode.")
    prior_dir = Path(sfc.get("prior_dir") or sfc["output_dir"])
    imap = load_json(sfc["instrument_map"]); spec = load_json(sfc["graph_spec_refined"]); gmap = load_json(sfc["instrument_group_map"])
    subgroup = {k:v.get('subgroup','Other:Misc') for k,v in gmap.items()}
    qdate = pd.to_datetime(sfc["date"]); outdir = Path(sfc["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    holders = spec['nodes']['holders']; issuers = spec['nodes']['issuers']
    default_tau = spec.get('default',{}).get('tau', 2.0); default_penalty = spec.get('default',{}).get('penalty', 2.0)

    t = tidy[(tidy['kind']=='FU') & (tidy['date']==qdate)].copy()
    t['side']  = t['instrument'].map(lambda k: imap.get(k,{}).get('side','macro'))
    t['subgroup'] = t['instrument'].map(lambda k: subgroup.get(k, 'Other:Misc'))
    assets = t[t['side']=='asset'].groupby(['subgroup','sector'])['value'].sum().rename('uses').reset_index()
    liabs  = t[t['side']=='liability'].groupby(['subgroup','sector'])['value'].sum().rename('sources').reset_index()

    def prior_matrix(file: Path, holders, issuers):
        if not file.exists(): return None
        df = pd.read_csv(file)
        if set(['holder_sector','issuer_sector','flow']) - set(df.columns): return None
        M = df.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').reindex(index=holders, columns=issuers).fillna(0.0)
        tot = M.values.sum()
        if tot <= 0: return None
        return M.values / max(tot, 1e-30)

    long_rows = []
    prev_date = tidy.loc[tidy['date'] < qdate, 'date'].max() if (tidy['date'] < qdate).any() else None
    for group, params in spec['groups'].items():
        r = assets[assets['subgroup']==group].set_index('sector')['uses'].reindex(holders).fillna(0.0).values
        c = liabs[liabs['subgroup']==group].set_index('sector')['sources'].reindex(issuers).fillna(0.0).values
        if (r.sum() <= 0) and (c.sum() <= 0): continue
        edges = params.get('edges', []); prefer = params.get('cost_hints',{}).get('prefer',[]); penal  = params.get('cost_hints',{}).get('penalize',[])
        tau    = params.get('tau', default_tau)
        A = np.zeros((len(holders), len(issuers))); C = np.zeros_like(A)
        for (i,j) in edges:
            if i in holders and j in issuers: A[holders.index(i), issuers.index(j)] = 1.0
        for (i,j) in prefer:
            if i in holders and j in issuers: C[holders.index(i), issuers.index(j)] -= default_penalty
        for (i,j) in penal:
            if i in holders and j in issuers: C[holders.index(i), issuers.index(j)] += default_penalty
        prior = None
        if prev_date is not None:
            Pfile = prior_dir / f"w2w_graph_{group}_{prev_date.date().isoformat()}.csv"
            prior = prior_matrix(Pfile, holders, issuers)
        X, info = allocate_flows_from_graph(r, c, A, cost=C, prior=prior, tau=tau)
        for i,h in enumerate(holders):
            for j,iss in enumerate(issuers):
                val = float(X[i,j])
                if val != 0.0:
                    long_rows.append({"instrument_group": group, "holder_sector": h, "issuer_sector": iss, "flow": val})
    long_df = pd.DataFrame(long_rows)
    tag = qdate.date().isoformat()
    long_df.to_csv(outdir / f"w2w_graph_LONG_{tag}.csv", index=False)
    for grp, g in long_df.groupby('instrument_group'):
        mat = g.pivot_table(index='holder_sector', columns='issuer_sector', values='flow', aggfunc='sum').fillna(0.0)
        mat.to_csv(outdir / f"w2w_graph_{grp}_{tag}.csv")
    print("Refined graph outputs in", outdir)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="SFC core runner (one switch per mode)")
    ap.add_argument("mode", choices=["baseline","fwtw","graph","refined"])
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    args = ap.parse_args()

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
    else:
        raise ValueError("Unknown mode.")

if __name__ == "__main__":
    main()
