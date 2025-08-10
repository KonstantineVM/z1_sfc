#!/usr/bin/env python3
"""
Test script for graph-based constraint generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import SFCGraphBuilder, StateIndex, SFCConstraintExtractor
from src.data import Z1Loader
import numpy as np

def main():
    """Test graph-based constraint extraction."""
    
    # Load sample data
    print("Loading Z1 data...")
    z1_loader = Z1Loader()
    z1_data = z1_loader.load_cached("Z1")
    
    # Build graph from series
    print("Building graph from series...")
    builder = SFCGraphBuilder()
    graph = builder.build_from_series(z1_data.columns[:100])  # Start with 100 series
    
    # Add stock-flow relationships
    print("Identifying stock-flow pairs...")
    pairs = builder.identify_stock_flow_pairs(z1_data.columns[:100])
    builder.add_stock_flow_relationships(graph, pairs)
    
    print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")
    
    # Create state mapping
    print("Creating state index...")
    state_index = StateIndex(z1_data.columns[:100], max_lag=2)
    print(f"State space dimension: {state_index.size}")
    
    # Extract constraints
    print("Extracting constraints at t=1...")
    extractor = SFCConstraintExtractor(graph, state_index)
    A, b, metadata = extractor.extract_at_time(t=1)
    
    print(f"Constraint matrix shape: {A.shape}")
    print(f"Number of constraints: {A.shape[0]}")
    print(f"Sparsity: {A.nnz / (A.shape[0] * A.shape[1]):.4%}")
    
    # Validate constraints
    print("\nConstraint types:")
    from collections import Counter
    types = Counter(m['type'] for m in metadata)
    for ctype, count in types.items():
        print(f"  {ctype}: {count}")
    
    # Export graph
    print("\nExporting graph to GraphML...")
    import networkx as nx
    nx.write_graphml(graph.G, "output/graphs/test_graph.graphml")
    print("✓ Graph exported to output/graphs/test_graph.graphml")

if __name__ == "__main__":
    main()
