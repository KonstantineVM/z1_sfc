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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local imports
from src.utils.config_manager import ConfigManager
from src.utils.results_manager import ResultsManager
from src.utils.helpers import ensure_dir
from src.utils.z1_series_interpreter import Z1Series

# Data
from src.data.cache_manager import CacheManager
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import pivot_series

# Graph/constraints
from src.graph.sfc_graph import SFCGraph, NodeType, EdgeType
from src.graph.state_index import StateIndex
from src.graph.constraint_extractor import ConstraintExtractor

# Models
from src.models.sfc_kalman_proper import ProperSFCKalmanFilter
from src.models.sfc_projection import project_state  # if used explicitly

RUN_DEFAULT_MODE = "production"

def parse_series_code(code):
    """Parse Z1 series code using existing interpreter."""
    parsed = Z1Series.parse(code)
    if parsed:
        return {
            'code': code,
            'prefix': parsed.prefix,
            'sector': parsed.sector,
            'instrument': parsed.instrument,
            'calculation': parsed.calculation,  # This is digit 9
            'is_base': parsed.calculation in ['0', '3'],
            'is_computed': parsed.calculation == '5'
        }
    return {'code': code}

def build_graph_from_formulas(formulas_path: str, series_names: list) -> SFCGraph:
    """Build graph from formulas and add all series nodes."""
    g = SFCGraph()
    
    # First add all series as nodes
    for series in series_names:
        md = parse_series_code(series)
        g.add_series_node(series, metadata=md)
    
    # Then add aggregation edges from formulas if file exists
    if os.path.exists(formulas_path):
        with open(formulas_path, "r") as f:
            formulas = json.load(f)
        
        # Add aggregation edges
        for parent, formula_info in formulas.items():
            # Ensure parent exists
            if parent not in g._series_index:
                md = parse_series_code(parent)
                g.add_series_node(parent, metadata=md)
            
            # Handle different formula structures
            children = []
            if isinstance(formula_info, list):
                children = formula_info
            elif isinstance(formula_info, dict):
                if 'derived_from' in formula_info:
                    children = [c.get('code') for c in formula_info['derived_from'] if c.get('code')]
                elif 'components' in formula_info:
                    children = formula_info['components']
            
            for ch in children:
                if ch:
                    # Ensure child exists
                    if ch not in g._series_index:
                        mdch = parse_series_code(ch)
                        g.add_series_node(ch, metadata=mdch)
                    # Add aggregation edge
                    g.add_edge(
                        g.get_series_node(ch), 
                        g.get_series_node(parent), 
                        edge_type=EdgeType.AGGREGATES_TO.value
                    )
    else:
        logger.warning(f"Formula file not found: {formulas_path}")
    
    return g

def add_stock_flow_edges(graph: SFCGraph, series_names: list):
    """Add stock-flow edges for base FL series."""
    stock_flow_pairs = []
    
    for series in series_names:
        if series.startswith('FL'):
            parsed = parse_series_code(series)
            # Only base series (digit 9 = 0 or 3) should have flows
            if parsed and parsed.get('is_base'):
                # Look for corresponding flow series
                base_code = series[2:]  # Remove 'FL' prefix
                
                flows_found = {}
                for flow_prefix in ['FU', 'FR', 'FV']:
                    flow_series = flow_prefix + base_code
                    if flow_series in series_names:
                        # Ensure flow series exists in graph
                        if flow_series not in graph._series_index:
                            flow_md = parse_series_code(flow_series)
                            graph.add_series_node(flow_series, metadata=flow_md)
                        
                        # Add stock-flow edge: flow → stock
                        graph.add_edge(
                            graph.get_series_node(flow_series),
                            graph.get_series_node(series),
                            edge_type=EdgeType.STOCK_FLOW.value,
                            weight=1.0
                        )
                        flows_found[flow_prefix] = flow_series
                
                if flows_found:
                    stock_flow_pairs.append({
                        'stock': series,
                        'flows': flows_found
                    })
    
    logger.info(f"Added stock-flow edges for {len(stock_flow_pairs)} base FL series")
    return stock_flow_pairs

def tag_source_vs_computed_nodes(graph: SFCGraph):
    """Tag nodes as source or computed based on aggregation edges."""
    for node_id in graph.get_nodes_by_type(NodeType.SERIES):
        node = graph.get_node(node_id)
        if node and node.metadata:
            # Check if has incoming aggregation edges
            has_incoming_agg = False
            for _, _, edge_data in graph.G.in_edges(node_id, data=True):
                if edge_data.get('edge_type') == EdgeType.AGGREGATES_TO.value:
                    has_incoming_agg = True
                    break
            
            # Also check calculation digit
            is_base = node.metadata.get('is_base', False)
            
            # Source if base series OR no incoming aggregation
            node.metadata['is_source'] = is_base or not has_incoming_agg
            node.metadata['is_computed'] = not node.metadata['is_source']

def collect_allowed_flow_bases(graph: SFCGraph, series_names: list) -> list:
    """Collect FL series that are allowed to have flows (base series only)."""
    bases = []
    
    for series in series_names:
        if series.startswith('FL'):
            parsed = parse_series_code(series)
            # Only base series (digit 9 = 0 or 3) can have flows
            if parsed and parsed.get('is_base'):
                bases.append(series)
    
    logger.info(f"Identified {len(bases)} base FL series for flow attachment")
    return bases

def find_computed_with_flows(graph: SFCGraph) -> list:
    """Find computed FL series that incorrectly have flow edges."""
    offenders = []
    
    for node_id in graph.get_nodes_by_type(NodeType.SERIES):
        node = graph.get_node(node_id)
        if node and node.metadata:
            code = node.metadata.get('code', node_id)
            
            # Check if this is a computed FL
            if (code.startswith('FL') and 
                (node.metadata.get('is_computed') or 
                 node.metadata.get('calculation') == '5')):
                
                # Check for incoming stock-flow edges
                for _, _, edge_data in graph.G.in_edges(node_id, data=True):
                    if edge_data.get('edge_type') == EdgeType.STOCK_FLOW.value:
                        offenders.append(code)
                        break
    
    return offenders

def validate_leaf_only_flows(graph, series_names):
    """
    Ensure only leaf FL series (digit 9 = 0 or 3) have FU/FR/FV flows.
    """
    violations = []
    
    for series in series_names:
        if series.startswith('FL') and len(series) >= 10:
            digit_9 = series[8]  # The 9th position (0-indexed)
            
            if digit_9 == '5':  # Calculated series
                # Check if FU/FR/FV exist for this series
                for prefix in ['FU', 'FR', 'FV']:
                    flow_series = series.replace('FL', prefix)
                    if flow_series in series_names:
                        violations.append({
                            'stock': series,
                            'flow': flow_series,
                            'issue': f"Calculated series (digit 9={digit_9}) should not have flows"
                        })
    
    if violations:
        logger.error(f"Found {len(violations)} leaf-only flow violations:")
        for v in violations[:10]:
            logger.error(f"  {v['stock']} has {v['flow']}: {v['issue']}")
        raise ValueError("Leaf-only flow constraint violated")
    
    return True

# Call it after loading data
validate_leaf_only_flows(graph, series_names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=RUN_DEFAULT_MODE, help="run mode from config/run_modes.yaml")
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    ap.add_argument("--formulas", default="data/fof_formulas_extracted.json")
    ap.add_argument("--fwtw", default="data/fwtw_data.parquet", help="path to FWTW bilateral data")
    ap.add_argument("--output_root", default="output/proper_test")
    ap.add_argument("--max_series", type=int, default=None, help="optional limit for speed")
    ap.add_argument("--use_discrepancy", action="store_true", help="model sector discrepancies")
    args = ap.parse_args()

    logger.info("="*70)
    logger.info("PROPER SFC KALMAN FILTER WITH GRAPH CONSTRAINTS")
    logger.info(f"Mode: {args.mode}")
    logger.info("="*70)

    # Load config with mode overlay
    cm = ConfigManager(root="config")
    base_cfg = cm.load(os.path.basename(args.config))
    mode_cfg = cm.load("run_modes.yaml").get('modes', {}).get(args.mode, {})
    cfg = cm.merge(base_cfg, mode_cfg)

    # Prepare results manager
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    rm = ResultsManager(output_root=args.output_root, run_id=run_id)
    ensure_dir(rm.paths["root"])

    # Load Z1 values & metadata
    logger.info("Loading Z1 data...")
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

    # Get series names
    series_names = list(df_wide.columns)
    logger.info(f"Loaded {len(series_names)} series, {len(df_wide)} periods")
    
    validate_leaf_only_flows(graph, series_names)    
    
    # Optional: reduce series for faster iteration
    if args.max_series:
        series_names = series_names[:args.max_series]
        df_wide = df_wide[series_names]
        logger.info(f"Limited to {len(series_names)} series for testing")

    # Load FWTW data if available and requested
    fwtw_data = None
    if os.path.exists(args.fwtw) and cfg.get("use_fwtw", False):
        logger.info("Loading FWTW bilateral data...")
        try:
            fwtw_data = pd.read_parquet(args.fwtw)
            logger.info(f"Loaded {len(fwtw_data)} FWTW bilateral positions")
        except Exception as e:
            logger.warning(f"Could not load FWTW data: {e}")

    # Build graph from formulas and series
    logger.info("Building graph...")
    graph = build_graph_from_formulas(args.formulas, series_names)
    
    # Add stock-flow edges for base series
    stock_flow_pairs = add_stock_flow_edges(graph, series_names)
    
    # Tag nodes as source vs computed
    tag_source_vs_computed_nodes(graph)
    
    logger.info(f"Graph has {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges")

    # Validate: no flows on computed FL (leaf-only validation)
    offenders = find_computed_with_flows(graph)
    if offenders:
        logger.error(f"ERROR: Computed FL series have flows attached: {offenders[:10]}")
        logger.error("This violates the core SFC principle - only base series should have flows")
        if cfg.get("constraints", {}).get("validation", {}).get("enforce_leaf_only", True):
            raise RuntimeError(f"Found {len(offenders)} computed FL with flows")
        else:
            logger.warning("Continuing despite violations (enforce_leaf_only=False)")

    # Collect allowed flow bases (base FL series only)
    allowed_bases = collect_allowed_flow_bases(graph, series_names)
    logger.info(f"Identified {len(allowed_bases)} base FL series that can have flows")

    # Build StateIndex with flow enforcement
    logger.info("Building state index...")
    sidx = StateIndex(series_names=series_names, max_lag=cfg.get("kalman", {}).get("max_lag", 2))
    sidx.set_allowed_flow_bases(allowed_bases)
    logger.info(f"State index: {sidx.n_series} series, max_lag={sidx.max_lag}, size={sidx.size}")

    # Load sectors and instruments configuration
    sectors = list(cfg.get("sectors", {}).keys())
    instruments = cfg.get("instruments", {})
    
    # Choose model based on data availability and configuration
    if fwtw_data is not None and args.use_discrepancy:
        # Full model with Z1 + FWTW + Discrepancy
        logger.info("Using complete model with Z1 + FWTW + Discrepancy...")
        from src.models.sfc_kalman_fwtw_discrepancy import SFCKalmanFWTWDiscrepancy
        
        model = SFCKalmanFWTWDiscrepancy(
            z1_data=df_wide,
            fwtw_data=fwtw_data,
            sectors=sectors,
            instruments=instruments,
            bilateral_weight=cfg.get("constraints", {}).get("weights", {}).get("bilateral", 0.5),
            include_discrepancy=True,
            discrepancy_variance_ratio=cfg.get("discrepancy", {}).get("variance_ratio", 0.1),
            state_index=sidx,
            graph=graph,
            config=cfg
        )
        
        # Build all constraints including bilateral
        t = len(df_wide) - 1
        A, b, metadata = model.extract_all_constraints(t)
        
        logger.info(f"Extracted {A.shape[0]} total constraints at t={t}")
        for ctype, count in metadata.items():
            logger.info(f"  {ctype}: {count} constraints")
            
    elif args.use_discrepancy:
        # Model with discrepancy but no FWTW
        logger.info("Using model with discrepancy (no FWTW)...")
        from src.models.sfc_kalman_with_discrepancy import SFCKalmanWithDiscrepancy
        
        model = SFCKalmanWithDiscrepancy(
            data=df_wide,
            sectors=sectors,
            include_discrepancy=True,
            discrepancy_variance_ratio=cfg.get("discrepancy", {}).get("variance_ratio", 0.1),
            state_index=sidx,
            graph=graph,
            config=cfg
        )
        
        # Extract constraints
        extractor = ConstraintExtractor(graph=graph, state_index=sidx, config=cfg)
        t = len(df_wide) - 1
        A, b, meta = extractor.extract_at_time(t=t)
        
        # Add Godley constraints with discrepancy
        A_godley, b_godley = model.build_godley_constraints(t)
        if A_godley.shape[0] > 0:
            A = sparse.vstack([A, A_godley])
            b = np.hstack([b, b_godley])
            logger.info(f"Added {A_godley.shape[0]} Godley constraints with discrepancy")
            
    else:
        # Basic model without FWTW or discrepancy
        logger.info("Using basic SFC model...")
        
        # Constraint extraction
        logger.info("Extracting constraints...")
        extractor = ConstraintExtractor(graph=graph, state_index=sidx, config=cfg)
        
        # Extract for last available time period
        t = len(df_wide) - 1
        A, b, meta = extractor.extract_at_time(t=t)
        
        logger.info(f"Extracted {A.shape[0]} constraints at t={t}")
        logger.info(f"  Matrix shape: {A.shape}, non-zeros: {A.nnz}")
        
        # Report constraint types
        constraint_types = {}
        for m in meta:
            ctype = m.constraint_type.value if hasattr(m, 'constraint_type') else 'unknown'
            constraint_types[ctype] = constraint_types.get(ctype, 0) + 1
        for ctype, count in constraint_types.items():
            logger.info(f"  {ctype}: {count} constraints")

        # Build weight matrix if needed
        W = None
        if hasattr(extractor, 'build_weight_matrix'):
            W = extractor.build_weight_matrix(meta)

        # Initialize basic model
        model = ProperSFCKalmanFilter(
            data=df_wide,
            state_index=sidx,
            graph=graph,
            constraint_extractor=extractor,
            config=cfg
        )

    # Fit/filter
    logger.info("Running Kalman filter with SFC constraints...")
    res = model.fit()

    # After model initialization with FWTW
    if fwtw_data is not None:
        # Report issuer-holder relationships found in FWTW
        logger.info("\nIssuer-Holder relationships from FWTW data:")
        
        for inst_code in list(instruments.keys())[:5]:  # Show first 5 instruments
            issuers = model.get_issuers_for_instrument(inst_code)
            holders = model.get_holders_for_instrument(inst_code)
            
            logger.info(f"  Instrument {inst_code}:")
            logger.info(f"    Issuers: {issuers}")
            logger.info(f"    Holders: {holders}")
        
        # Validate relationships
        validation = model.validate_issuer_holder_relationships()
        if validation['errors']:
            logger.error(f"Found {len(validation['errors'])} critical issues in issuer-holder relationships")
        else:
            logger.info("  ✓ All issuer-holder relationships validated successfully")

    # After model initialization, add flow derivation section
    if fwtw_data is not None and cfg.get('godley', {}).get('derive_flows', False):
        logger.info("Deriving bilateral flows from FWTW...")
        
        # Import flow derivation and visualization
        from src.models.godley_flow_derivation import GodleyFlowDerivation
        from src.visualization.godley_matrix_viz import GodleyMatrixVisualizer
        
        # Derive flows for last period
        t = len(df_wide) - 1
        godley_accounts = model.flow_deriver.create_full_godley_accounts(t)
        
        # Validate against Z1
        validation = model.flow_deriver.validate_against_aggregates(t)
        
        logger.info("Bilateral flow validation:")
        n_valid = sum(1 for v in validation.values() if v.get('validation_passed', False))
        logger.info(f"  {n_valid}/{len(validation)} sector-instruments pass validation")
        
        # Save Godley matrices
        if cfg.get('output', {}).get('save_godley_matrix', False):
            # Create visualizer
            viz = GodleyMatrixVisualizer(output_dir=rm.paths["root"])
            
            # Plot flow matrices
            viz.plot_godley_flow_matrix(
                godley_accounts['transactions'],
                period=str(df_wide.index[t])
            )
            
            # Plot network for major instruments
            for inst in ['31611', '31630', '40000']:  # Treasuries, Equities, Loans
                if inst in instruments:
                    viz.plot_flow_network(
                        godley_accounts['bilateral_flows'],
                        instrument=inst,
                        threshold=100
                    )
            
            # Create HTML report
            viz.create_godley_report(godley_accounts, str(df_wide.index[t]))
            
            logger.info(f"  Saved Godley visualizations to {rm.paths['root']}")

    # Validation for models with discrepancy
    if args.use_discrepancy and hasattr(model, 'validate_godley_identities'):
        logger.info("\nValidating Godley identities...")
        validation = model.validate_godley_identities(res.get('filtered'))
        
        for sector, metrics in validation.items():
            if metrics['identity_satisfied']:
                logger.info(f"  Sector {sector}: ✓ Identity satisfied (max error: {metrics['max_error']:.2e})")
            else:
                logger.warning(f"  Sector {sector}: ✗ Identity violated (max error: {metrics['max_error']:.2e})")
        
        # Extract and save discrepancy estimates
        if hasattr(model, 'extract_discrepancy_estimates'):
            discrepancies = model.extract_discrepancy_estimates(res.get('filtered'))
            disc_path = os.path.join(rm.paths["root"], "estimated_discrepancies.csv")
            discrepancies.to_csv(disc_path)
            logger.info(f"  Saved discrepancy estimates to {disc_path}")

    # Validation for FWTW models
    if fwtw_data is not None and hasattr(model, 'validate_bilateral_consistency'):
        logger.info("\nValidating bilateral consistency...")
        bilateral_validation = model.validate_bilateral_consistency(res.get('filtered'))
        
        for sector, instruments in bilateral_validation.items():
            logger.info(f"  Sector {sector}:")
            for inst, metrics in instruments.items():
                logger.info(f"    {inst}: mean discrepancy = {metrics['mean_discrepancy']:.2f}, "
                          f"n_bilateral = {metrics['n_bilaterals']}")

    # Save outputs
    logger.info(f"\nSaving results to {rm.paths['root']}...")
    
    # Save model artifacts
    if hasattr(rm, 'save_model_artifacts'):
        rm.save_model_artifacts(model, extra={
            "n_constraints": int(A.shape[0]),
            "n_series": len(series_names),
            "n_periods": len(df_wide),
            "has_fwtw": fwtw_data is not None,
            "has_discrepancy": args.use_discrepancy
        })
    
    # Save filtered states
    if hasattr(rm, 'save_filtered_arrays'):
        rm.save_filtered_arrays(
            filtered=res.get("filtered"),
            smoothed=res.get("smoothed"),
            state_index_mapping=sidx.index
        )
    
    # Save constraint report
    if hasattr(rm, 'save_constraints_report'):
        rm.save_constraints_report(A, b, meta if 'meta' in locals() else metadata if 'metadata' in locals() else None)
    
    # Save graph if needed
    if cfg.get("output", {}).get("save_graphml", False):
        graph_path = os.path.join(rm.paths["root"], "sfc_graph.graphml")
        if hasattr(graph, 'export_to_graphml'):
            graph.export_to_graphml(graph_path)
            logger.info(f"  Saved graph to {graph_path}")
    
    # Generate diagnostics report
    if cfg.get("output", {}).get("generate_diagnostics", True):
        logger.info("\nGenerating diagnostics...")
        diagnostics = {
            "n_series": len(series_names),
            "n_base_fl": len(allowed_bases),
            "n_computed_fl": len([s for s in series_names if s.startswith('FL') and s[8] == '5']),
            "n_constraints": int(A.shape[0]),
            "constraint_sparsity": 1.0 - (A.nnz / (A.shape[0] * A.shape[1])) if A.shape[0] > 0 else 0,
            "has_fwtw": fwtw_data is not None,
            "has_discrepancy": args.use_discrepancy,
            "mode": args.mode
        }
        
        diag_path = os.path.join(rm.paths["root"], "diagnostics.json")
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        logger.info(f"  Saved diagnostics to {diag_path}")

    logger.info("="*70)
    logger.info("✓ ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("="*70)

if __name__ == "__main__":
    main()
