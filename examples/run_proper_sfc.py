#!/usr/bin/env python3
"""
Run a Proper SFC Kalman pass with:
- leaf-only stock–flow identities (FU/FR/FV only for source FL)
- aggregation + bilateral constraints
- Z1-style config presets (run_modes.yaml + proper_sfc_config.yaml)
"""

import argparse, os, sys, json, time
import numpy as np
import pandas as pd

# Local imports
from src.utils.config_manager import ConfigManager
from src.utils.results_manager import ResultsManager
from src.utils.helpers import ensure_dir
from src.utils.z1_series_interpreter import parse_series_code

# Data
from src.data.cache_manager import CacheManager
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import pivot_series

# Graph/constraints
from src.graph.sfc_graph import SFCGraph, NodeType
from src.graph.state_index import StateIndex
from src.graph.constraint_extractor import ConstraintExtractor

# Models
from src.models.sfc_kalman_proper import ProperSFCKalmanFilter
from src.models.sfc_projection import project_state  # if used explicitly

RUN_DEFAULT_MODE = "production"

def build_graph_from_formulas(formulas_path: str) -> SFCGraph:
    g = SFCGraph()
    # You already have a loader in your codebase; if not, minimally:
    # Expect a JSON with edges parent <- children for FL series
    with open(formulas_path, "r") as f:
        formulas = json.load(f)
    # Add SERIES and AGGREGATES_TO edges
    for parent, children in formulas.items():
        # Parent node
        pfx = parent[:2]
        md = {"code": parent, "prefix": pfx}
        try:
            md.update(parse_series_code(parent))
        except Exception:
            pass
        g.add_series_node(parent, metadata=md)
        for ch in children:
            mdch = {"code": ch, "prefix": ch[:2]}
            try:
                mdch.update(parse_series_code(ch))
            except Exception:
                pass
            g.add_series_node(ch, metadata=mdch)
            g.add_edge(g.get_series_node(ch), g.get_series_node(parent), edge_type="AGGREGATES_TO")
    return g

def collect_allowed_flow_bases(graph: SFCGraph) -> list[str]:
    graph.tag_source_vs_computed()
    bases = []
    for nid in graph.get_nodes_by_type(NodeType.SERIES):
        node = graph.get_node(nid)
        md = node.metadata or {}
        if md.get("prefix") == "FL" and md.get("is_source") is True:
            sector = md.get("sector")
            inst = md.get("instrument")
            if sector and inst:
                bases.append(f"{sector}{inst}005.Q")
    return bases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=RUN_DEFAULT_MODE, help="run mode from config/run_modes.yaml")
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    ap.add_argument("--formulas", default="data/fof_formulas_extracted.json")
    ap.add_argument("--output_root", default="output/proper_test")
    ap.add_argument("--max_series", type=int, default=None, help="optional limit for speed")
    args = ap.parse_args()

    # Load config with mode overlay
    cm = ConfigManager(root="config")
    base_cfg = cm.load(os.path.basename(args.config))
    mode_cfg = cm.load("run_modes.yaml").get(args.mode, {})
    cfg = cm.merge(base_cfg, mode_cfg)

    # Prepare results manager
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    rm = ResultsManager(output_root=args.output_root, run_id=run_id)
    ensure_dir(rm.paths["root"])

    # Load Z1 values & metadata (from cache if available)
    # (Your CachedFedDataLoader already set up in Batch 4)
    data_paths = cfg.get("data_paths", {})
    z1_root = data_paths.get("z1_root", "data/fed_data/FRB_Z1")
    files_cfg = cfg.get("z1_files", {
        "data_xml": "Z1_data.xml",
        "struct_xml": "Z1_struct.xml",
        "schema_xsd": "Z1_Z1.xsd",
        "common_xsd": "frb_common.xsd",
    })
    cache = CacheManager(cfg.get("data_paths", {}).get("cache_root", "data/cache"))
    loader = CachedFedDataLoader(cache=cache, z1_root=z1_root, files_cfg=files_cfg)

    df_vals = loader.load_or_extract_values()
    df_meta = loader.load_or_extract_metadata()
    df_wide = pivot_series(df_vals)  # columns = Z1 series codes

    # Optional: reduce series for faster iteration
    series_names = list(df_wide.columns)
    if args.max_series:
        series_names = series_names[:args.max_series]
        df_wide = df_wide[series_names]

    # Build graph from formulas (or use your existing graph builder)
    graph = build_graph_from_formulas(args.formulas)

    # Guardrail: no flows on computed FL
    offenders = graph.find_computed_with_flows()
    if offenders:
        raise RuntimeError(f"Computed FL have flows attached: {offenders[:10]}")

    # Collect allowed flow bases (leaf-only)
    allowed_bases = set(collect_allowed_flow_bases(graph))

    # Build StateIndex and enforce allowed flows
    sidx = StateIndex(series_names=series_names, max_lag=cfg.get("max_lag", 2))
    sidx.set_allowed_flow_bases(allowed_bases)
    sidx.apply_allowed_flow_bases(strict=True)

    # Constraint extraction for a single time t (or loop over t)
    extractor = ConstraintExtractor(graph=graph, state_index=sidx, config=cfg)
    # Example: last available quarter index
    t = len(df_wide) - 1
    A, b, meta = extractor.extract_at_time(t=t)
    W = extractor.build_weight_matrix(meta)

    # Prepare model inputs (Z, H, T, R, Q) as your ProperSFCKalmanFilter expects
    # The model in your repo already wires these; here we just call it.
    model = ProperSFCKalmanFilter(
        state_index=sidx,
        config=cfg,
        constraint_matrix=A,
        constraint_rhs=b,
        constraint_weights=W,
    )

    # Fit / filter
    res = model.fit(df_wide)

    # Save outputs using your ResultsManager
    rm.save_model_artifacts(model, extra={"n_constraints": int(A.shape[0])})
    rm.save_filtered_arrays(
        filtered=res.get("filtered"), smoothed=res.get("smoothed"),
        state_index_mapping=sidx.index
    )
    rm.save_constraints_report(A, b, meta)

    print(f"Done. Outputs → {rm.paths['root']}")

if __name__ == "__main__":
    main()

