#!/usr/bin/env python3
"""
Integrated run including FWTW network:
- Loads FWTW, normalizes schema
- Ensures leaf-only FU generation from Δlevels (holder-side only)
- Builds the same SFC constraints and runs the model
"""

import argparse, os
from examples.run_proper_sfc import (
    build_graph_from_formulas, collect_allowed_flow_bases
)
from src.utils.config_manager import ConfigManager
from src.utils.helpers import ensure_dir
from src.utils.results_manager import ResultsManager

from src.network.fwtw_loader import load_fwtw
from src.network.fwtw_loader import normalize_fwtw_schema
from src.network.fwtw_z1_mapper import compute_bilateral_flows_from_levels

from src.graph.sfc_graph import SFCGraph, NodeType
from src.graph.state_index import StateIndex
from src.graph.constraint_extractor import ConstraintExtractor
from src.models.sfc_kalman_proper import ProperSFCKalmanFilter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    ap.add_argument("--formulas", default="data/fof_formulas_extracted.json")
    ap.add_argument("--fwtw_path", default="data/cache/fwtw/fwtw_processed.parquet")
    ap.add_argument("--output_root", default="output/proper_test")
    args = ap.parse_args()

    cm = ConfigManager(root="config")
    cfg = cm.load(os.path.basename(args.config))

    # 1) Graph & allowed bases
    graph = build_graph_from_formulas(args.formulas)
    offenders = graph.find_computed_with_flows()
    if offenders:
        raise RuntimeError(f"Computed FL have flows attached: {offenders[:10]}")
    allowed_bases = set(collect_allowed_flow_bases(graph))

    # 2) FWTW loading
    df_levels = load_fwtw(args.fwtw_path)
    df_levels = normalize_fwtw_schema(df_levels)
    df_fu = compute_bilateral_flows_from_levels(df_levels, allowed_bases)

    # 3) Build series universe (levels + flows present in data)
    series_names = sorted(set(
        list(df_levels.apply(lambda r: f"FL{r['holder_sector']}{r['instrument_code']}005.Q", axis=1)) +
        list(df_levels.apply(lambda r: f"FL{r['issuer_sector']}{r['instrument_code']}005.Q", axis=1)) +
        list(df_fu["fu_series"])
    ))

    # 4) State index with enforcement
    sidx = StateIndex(series_names=series_names, max_lag=cfg.get("max_lag", 2))
    sidx.set_allowed_flow_bases(allowed_bases)
    sidx.apply_allowed_flow_bases(strict=True)

    # 5) Constraints at last date present in FWTW
    extractor = ConstraintExtractor(graph=graph, state_index=sidx, config=cfg)
    t = 0  # you can map dates→t; set to 0 here to build structure only
    A, b, meta = extractor.extract_at_time(t=t)
    W = extractor.build_weight_matrix(meta)

    # 6) Run model (assuming you have measurements aligned; if not, this is just constraint projection)
    model = ProperSFCKalmanFilter(
        state_index=sidx, config=cfg, constraint_matrix=A, constraint_rhs=b, constraint_weights=W
    )
    # Placeholder for actual measurement data: you’ll use your Z1 or integrated df_wide here
    # res = model.fit(df_wide)  # same as in run_proper_sfc

    print("Integrated pipeline wired. Use your measurement matrix to run filtering.")

if __name__ == "__main__":
    main()

