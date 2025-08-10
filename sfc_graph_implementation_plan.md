# Graph-Based Stock-Flow Consistent Kalman Filter

## Implementation Plan for Full-Scale Z1 Analysis

### Version 2.0 - Incremental Shadow Implementation

---

## 1. Executive Summary

### Objective

Transform the current Z1 SFC Kalman filter project into a graph-based architecture capable of processing all 19,000+ Federal Reserve Z1 series with complete stock-flow consistency, market clearing, and bilateral constraint enforcement.

### Current State

- Successfully enforces SFC constraints for 200-500 series via projection
- Achieves machine-precision constraint satisfaction (1e-12)
- Limited FWTW integration (0.0% overlap)
- Manual constraint generation becoming bottleneck

### Target State

- Graph-driven constraint generation for 19,000+ series
- Automatic discovery of constraint networks
- Full FWTW bilateral integration
- Scalable sparse matrix operations
- Visual debugging via graph exports

### Approach

Incremental shadow implementation - build graph-based system alongside current constraint generation, validate equivalence at each step, then switch over when proven.

---

## 2. Architecture Overview

### Core Components

```
Z1 Data (19,000 series) → Graph Construction → Constraint Generation → Projection → Validated Results
                              ↑                      ↓
                         FWTW Bilateral          Sparse Matrices
                           Formulas              (A, b constraints)
```

### Graph Structure

**Nodes (≈20,000)**

- Series nodes: Each Z1 series (FL, FU, FR, FV, FA, LA)
- Sector nodes: \~100 sectors
- Instrument nodes: \~300 instruments
- Identity nodes: Market clearing points
- Aggregate nodes: Formula targets

**Edges (≈100,000+)**

- Stock-flow edges: FU→FL, FR→FL, FV→FL
- Aggregation edges: Components→Formula target
- Bilateral edges: FWTW positions
- Market clearing edges: Sector-instrument→Identity

**Properties**

- Edge weights: Coefficients (typically 1.0, -1.0)
- Node metadata: Sector, instrument, frequency, type
- Edge types: Constraint classification

---

### 2.1 Reusing the Hierarchical Kalman Filter (HKF)

**Keep the HKF core; let the graph drive constraints and indexing.** The graph layer does not replace the filter—only how we generate constraint matrices and map names → state indices.

#### What stays the same

- `HierarchicalKalmanFilter` state layout, lag augmentation, `update/transform_params`, filtering and smoothing.
- Block-diagonal Q/H by sector/instrument and the hierarchical variance structure.

#### What the graph adds

- A static topology of series/aggregates/lags/bilateral links.
- A `StateIndex` mapping `(series, lag)` → HKF state column (immutable across time).
- A `GraphConstraintExtractor` that emits sparse `(A_t, b_t)` each period based on the graph + `StateIndex`.

#### Integration flow per period

1. Run HKF filter/smoother as usual to get `x_t`, `P_t`.
2. Build `(A_t, b_t)` from the graph.
3. Project `x_t, P_t` onto `A_t x = b_t` via the existing projection step.

#### Minimal class wiring (sketch)

```python
class GraphConstraintProvider:
    def __init__(self, sfc_graph, state_index): ...
    def at_time(self, t):  # -> (A_t, b_t, meta)
        return A_t, b_t, meta

class ProperSFCKalmanGraph(HierarchicalKalmanFilter):
    def __init__(self, *, sfc_graph, state_index, projector, **kwargs):
        super().__init__(**kwargs)
        self.constraints = GraphConstraintProvider(sfc_graph, state_index)
        self.projector = projector

    def filter(self, params=None, **kw):
        res = super().filter(params, **kw)
        X, P = res.filtered_state, res.filtered_state_cov
        for t in range(X.shape[1]):
            A, b, _ = self.constraints.at_time(t)
            X[:, t], P[:, :, t] = self.projector.project(X[:, t], P[:, :, t], A, b)
        return res

    def smooth(self, params=None, **kw):
        res = super().smooth(params, **kw)
        X, P = res.smoothed_state, res.smoothed_state_cov
        for t in range(X.shape[1]):
            A, b, _ = self.constraints.at_time(t)
            X[:, t], P[:, :, t] = self.projector.project(X[:, t], P[:, :, t], A, b)
        return res
```

#### Pitfalls to avoid

- Don’t both **observe** an aggregate and enforce the exact same formula as a hard constraint—choose one (or soften one) to avoid redundancy.
- Keep `StateIndex` fixed across time; only values change, not indices.
- Represent lags only in the state; `A_t` must not couple across time indices.

---

## 3. Technical Implementation

### 3.1 Graph Layer (`src/graph/`)

**Core Classes**

```python
SFCNode: id, kind, metadata (sector, instrument, prefix, suffix)
SFCEdge: source, destination, type, weight, lag (temporal offset)
SFCGraph: NetworkX DiGraph wrapper, STATIC topology
StateIndex: Centralized name→state_index mapping, handles lags
SFCConstraintExtractor: Generates A_t, b_t from static graph + time t
ConstraintMetadata: Row-level tracking {type, node_id, edge_ids, t}
```

**Key Methods**

- `build_static_graph()`: One-time topology construction
- `extract_constraints_at_t()`: Graph + t → A\_t, b\_t
- `add_lag_edges()`: FL[t-1]→FL[t] with lag=-1 attribute
- `diff_constraints()`: Compare legacy vs graph matrices
- `validate_equivalence()`: Ensure ||A\_legacy - A\_graph|| < 1e-12

### 3.2 Shadow Architecture

**Dual Pipeline**

```python
# Configuration flag
constraints_mode: "legacy" | "graph" | "both"

# Runtime selection
if mode == "both":
    A_legacy, b_legacy = legacy_builder.build()
    A_graph, b_graph = graph_extractor.extract(t)
    diff_report = compare_constraints(A_legacy, A_graph)
    if max_diff > 1e-12:
        logger.warning(f"Constraint mismatch: {diff_report}")
    return A_legacy, b_legacy  # Use legacy until validated
```

**Validation Pipeline**

1. Build constraints both ways
2. Compare matrix norms, row counts, sparsity patterns
3. Log differences with row-level detail
4. Use legacy for projection, graph for monitoring
5. Switch when diff < 1e-12 for all test cases

### 3.3 Constraint Generation Strategy

**Time-Aware Extraction**

```python
# Graph is static topology, StateIndex manages augmented state mapping
class StateIndex:
    """Maps series names to state vector indices with lag support"""
    def __init__(self, series_names, max_lag=2):
        self.index = {}
        idx = 0
        # Current values
        for name in series_names:
            self.index[(name, 0)] = idx
            idx += 1
        # Lagged values (already in augmented state)
        for lag in range(1, max_lag + 1):
            for name in series_names:
                self.index[(name, -lag)] = idx
                idx += 1
        self.size = idx
    
    def get(self, name: str, lag: int = 0) -> int:
        """Get state index for series at lag. Raises if not found."""
        key = (name, lag)
        if key not in self.index:
            raise KeyError(f"State not found: {name} at lag {lag}")
        return self.index[key]

def extract_constraints_at_t(graph, state_index, t):
    """Extract A_t, b_t from static graph at time t"""
    rows, cols, data = [], [], []
    metadata = []
    
    # Stock-flow: FL[t] - FL[t-1] - (FU[t] + FR[t] + FV[t]) = 0
    for fl_node in (n for n,d in graph.G.nodes(data=True)
                    if d.get('prefix') == 'FL'):
        row = len(metadata)
        
        # FL[t] coefficient
        cols.append(state_index.get(fl_node, lag=0))
        data.append(1.0)
        
        # FL[t-1] coefficient (if t > 0)
        if t > 0:
            cols.append(state_index.get(fl_node, lag=-1))
            data.append(-1.0)
        
        # Flow coefficients
        for flow, edge_data in graph.G.in_edges(fl_node, data=True):
            if edge_data.get('etype') == 'stock_flow':
                cols.append(state_index.get(flow, lag=0))
                data.append(-edge_data.get('weight', 1.0))
        
        # Same row for all coefficients
        rows.extend([row] * len(data))
        
        metadata.append({
            'type': 'stock_flow',
            'node': fl_node,
            'time': t,
            'weight': 1.0,
            'component_ids': [fl_node] + list(graph.G.predecessors(fl_node))
        })
    
    n_rows = len(metadata)
    A = coo_matrix((data, (rows, cols)),
                   shape=(n_rows, state_index.size)).tocsr()
    b = np.zeros(n_rows)
    
    return A, b, metadata
```

**Sparse Construction Rules**

- Never materialize dense matrices
- Use COO for construction, CSR for algebra
- Vectorize by constraint type
- Pre-allocate with estimated nnz

---

## 4. Data Integration

### 4.1 Z1 Series Parsing

**Pattern Recognition**

```
FL152064105.Q → {
    prefix: "FL",      # Level/stock
    sector: "15",      # Households
    instrument: "20641", # Money market funds
    suffix: "05",      # Calculation type
    frequency: "Q"     # Quarterly
}
```

**Completeness Discovery**

- For each FL, find matching FU, FR, FV
- Log orphaned series
- Prioritize complete quartets

### 4.2 FWTW Bilateral Mapping

**Current Issues**

- 0.0% overlap with Z1 series
- Different coding systems
- Aggregation level mismatch

**Solution Approach**

- Build FWTW graph separately
- Graph matching algorithm
- Fuzzy matching on sector/instrument
- Manual mapping table for critical relationships

### 4.3 Formula Integration

**Expected Format**

```json
{
    "formula_id": "AGG_001",
    "lhs": "FL892090005.Q",
    "rhs": ["FL152090005.Q", "FL102090005.Q"],
    "weights": [1.0, 1.0],
    "lag_structure": [0, 0]
}
```

**Complex Formulas**

- Handle lagged terms via augmented state
- Nonlinear relationships via auxiliary nodes
- Conditional constraints via subgraphs

---

## 5. Scalability Design

### 5.1 Graph Partitioning

**By Instrument**

- Each instrument forms natural partition
- Parallel processing possible
- Market clearing at boundaries

**By Sector**

- Sector-level subgraphs
- Inter-sector flows as cut edges
- Hierarchical aggregation

### 5.2 Numerical Stability

**At 19,000 Series Scale**

- Condition number monitoring
- Iterative refinement for projection
- Regularization for near-singular systems
- Pivoting strategies for sparse solvers

### 5.3 Performance Targets

**Benchmarks**

- Graph construction: < 30 seconds for 19,000 series
- Constraint generation: < 10 seconds
- Projection: < 1 minute per time period
- Full run: < 30 minutes for 100 quarters

**Memory Usage**

- Graph: \~2-3 GB in memory
- Sparse matrices: \~500 MB
- State vectors: \~1 GB
- Total: < 8 GB RAM

---

## 6. Validation Framework

### 6.1 Constraint Satisfaction

**Metrics**

- Maximum violation per constraint type
- Mean absolute violation
- Percentage constraints satisfied
- Violation heatmap by sector/instrument

**Tolerances**

```python
TOLERANCES = {
    'stock_flow': 1e-10,      # Accounting identity
    'aggregation': 1e-8,      # Formula precision
    'market_clearing': 1e-6,  # Allow small discrepancies
    'bilateral': 1e-4         # FWTW measurement error
}
```

### 6.2 Constraint Weighting Strategy

**Soft vs Hard Constraints**

```python
class ConstraintWeight:
    """Manage constraint weights for soft/hard enforcement"""
    
    WEIGHTS = {
        'stock_flow': 1.0,      # Hard - accounting identity
        'aggregation': 1.0,     # Hard - formula exact
        'market_clearing': 0.8, # Slightly soft - measurement error
        'bilateral': 0.1        # Very soft initially - mapping uncertain
    }
    
    @classmethod
    def get_weight_matrix(cls, metadata_list):
        """Build diagonal weight matrix W for weighted projection"""
        n = len(metadata_list)
        W = np.ones(n)
        for i, meta in enumerate(metadata_list):
            W[i] = cls.WEIGHTS.get(meta['type'], 1.0)
        return sparse.diags(W)
    
    @classmethod
    def promote_constraint(cls, ctype: str, new_weight: float):
        """Gradually increase weight as confidence improves"""
        old = cls.WEIGHTS[ctype]
        cls.WEIGHTS[ctype] = min(1.0, new_weight)
        logger.info(f"Promoted {ctype}: {old:.2f} → {new_weight:.2f}")
```

**Weighted Projection**

```python
def weighted_project(A, b, x, P, W):
    """min ||x - x_hat||_P subject to W^(1/2)(Ax - b) = 0"""
    Aw = W @ A  # Weight the constraints
    bw = W @ b
    return project(Aw, bw, x, P)
```

### 6.3 Numerical Stability Enhancements

**Regularization Strategy**

```python
def project_with_regularization(A, b, x, P, lambda_ridge=1e-12):
    """
    Solve: min ||x - x_hat||_P subject to Ax = b
    With ridge regularization for rank-deficient systems
    """
    APA = A @ P @ A.T
    
    # Monitor condition number
    cond = np.linalg.cond(APA)
    if cond > 1e12:
        logger.warning(f"Poor conditioning: {cond:.2e}")
        APA += lambda_ridge * sparse.eye(APA.shape[0])
    
    # Solve with iterative refinement
    try:
        factor = splu(APA)
        lambda_opt = factor.solve(A @ x - b)
    except:
        # Fallback to iterative solver
        lambda_opt, info = cg(APA, A @ x - b, tol=1e-10)
        if info != 0:
            logger.error(f"CG failed with code {info}")
    
    return x - P @ A.T @ lambda_opt
```

**Constraint Prioritization**

```python
CONSTRAINT_WEIGHTS = {
    'stock_flow': 1.0,      # Hard constraint
    'aggregation': 1.0,     # Hard constraint
    'market_clearing': 0.8, # Slightly soft
    'bilateral': 0.1        # Very soft initially
}
```

---

## 7. Visualization & Debugging

### 7.1 Graph Exports

**GraphML Format**

- Full graph with all metadata
- Importable to Gephi/Cytoscape
- Subgraph extraction tools

**Interactive Visualization**

- Web-based viewer (D3.js/Sigma.js)
- Sector/instrument filtering
- Constraint violation overlay
- Time animation

### 7.2 Diagnostic Outputs

**Reports**

- Constraint violation report by type
- Missing series relationships
- FWTW mapping success rate
- Network statistics (density, clustering)

**Debug Mode**

- Constraint-by-constraint traceback
- Series contributing to violations
- Numerical conditioning warnings
- Memory usage profiling

---

## 8. Implementation Phases

### Phase 0: Infrastructure & Testing Framework (Days 1-2)

1. Create `src/graph/` module skeleton
2. Add `--constraints=legacy|graph|both` CLI flag
3. Implement series parser with comprehensive tests
4. Set up golden test suite (10-node toy SFC)
5. Build diff tool for constraint comparison

### Phase 1: Stock-Flow Subgraph (Days 3-4)

1. Build graph for current 50-series subset
2. Add FL nodes and FU/FR/FV→FL edges
3. Add FL[t-1]→FL[t] lag edges
4. Implement basic extractor (stock-flow only)
5. Validate: ||A\_legacy - A\_graph|| < 1e-12

**Acceptance Criteria**

- Graph path reproduces legacy stock-flow constraints
- Same violation metrics (\~1e-12)
- Identical row counts and sparsity pattern

### Phase 2: Aggregation Formulas (Days 5-6)

1. Parse formula file into graph edges
2. Add aggregation nodes and edges
3. Extend extractor for aggregation constraints
4. Validate against legacy with formulas

### Phase 3: Single Instrument Test (Days 7-8)

1. Select Treasury securities (instrument 31305)
2. Build subgraph for all sectors holding Treasuries
3. Add market clearing constraints
4. Test at 500-series scale

### Phase 4: FWTW Soft Constraints (Days 9-10)

1. Fix FWTW mapping with fuzzy matching
2. Add bilateral edges as soft constraints
3. Weight bilateral constraints (start at 0.1)
4. Gradually increase weight as mapping improves

### Phase 5: Full Scale Testing (Days 11-15)

1. Load all 19,000 series
2. Build complete graph (with progress bar)
3. Test constraint generation performance
4. Memory profiling and optimization
5. Parallel processing for sectors

### Phase 6: Production Cutover (Days 16-20)

1. Run both pipelines in parallel for 1 week
2. Monitor difference metrics
3. Performance comparison
4. Switch default to graph when stable
5. Deprecate legacy builder

---

## 9. Critical Questions to Answer

### Data Questions

1. **Formula Completeness**: What percentage of Z1 series have formula definitions?
2. **FWTW Coverage**: Why is current overlap 0.0%? Different coding system?
3. **Series Selection**: How to prioritize which series to include when memory-limited?
4. **Temporal Consistency**: How to handle series that start/end at different dates?

### Technical Questions

5. **State Augmentation**: Should lagged terms be in state vector or handled separately?
6. **Sparse Solver**: Which solver for (APA')^-1? Direct vs iterative?
7. **Parallel Processing**: Process sectors in parallel or maintain global consistency?
8. **Numerical Precision**: Float64 sufficient or need higher precision?

### Algorithmic Questions

9. **Graph Matching**: How to align FWTW bilateral graph with Z1 structure?
10. **Constraint Priority**: When constraints conflict, which take precedence?
11. **Missing Data**: How to handle constraints when some series have missing values?
12. **Convergence**: What if projection doesn't converge in reasonable time?

### Scale Questions

13. **Component Processing**: Process giant component only or all components?
14. **Memory Limits**: If graph exceeds RAM, use graph database or streaming?
15. **Incremental Updates**: How to update graph when new data arrives?
16. **Visualization Limits**: How to visualize 19,000 node graph meaningfully?

### Validation Questions

17. **Ground Truth**: What validates our constraints are economically correct?
18. **Benchmarking**: Compare results with Fed's internal consistency checks?
19. **Sensitivity**: How sensitive are results to constraint tolerance levels?
20. **Robustness**: How to detect and handle ill-conditioned constraint systems?

---

## 10. Success Criteria & Deliverables

### Functional

- Process all 19,000+ Z1 series
- Generate constraints from graph topology (stock‑flow, aggregation, market‑clearing, bilateral)
- Maintain machine‑precision SFC (≤1e‑10) on hard constraints
- Export full graph (GraphML) and per‑subgraph slices

### Performance

- Graph build < 30s (full Z1)
- Constraint extraction < 10s per run
- Projection < 60s per 100 quarters (per partition)
- Total end‑to‑end run < 30 minutes on 8–16 core box, 8–16 GB RAM

### Quality

- Parity with legacy constraints on 50‑series suite (max diff < 1e‑12)
- ≥95% aggregation constraints satisfied within tolerance
- ≥90% market‑clearing satisfied initially, trending upward as mapping improves
- Deterministic, reproducible runs (seeded)

### Deliverables

- Graph layer (`SFCGraph`, `StateIndex`, `SFCConstraintExtractor`)
- Dual‑mode constraints (legacy/graph) with validator & diff report
- Projection module with weighting & regularization
- GraphML exports and validation report (HTML)
- Unit/integration test suite (toy SFC, 50‑series, 1k‑series)

---

## 11. First Implementation Slice (Days 1-2)

### Day 1: Foundation

**Morning: Module Setup**

```bash
# Create structure
mkdir -p src/graph tests/graph
touch src/graph/__init__.py
touch src/graph/sfc_graph.py
touch src/graph/series_parser.py
touch tests/graph/test_parser.py
```

**File:** `src/graph/series_parser.py`

```python
def parse_z1_code(code: str) -> dict:
    """
    FL152064105.Q → {
        'prefix': 'FL', 'sector': '15',
        'instrument': '20641', 'suffix': '05', 'freq': 'Q'
    }
    """
    import re
    s = str(code)
    freq = s[-1] if s.endswith(('.Q', '.A')) else None
    base = s[:-2] if freq else s
    
    match = re.match(r'^([A-Z]{2})(\d{2})(\d{5})(\d{3})', base)
    if not match:
        return {'prefix': None, 'sector': None,
                'instrument': None, 'suffix': None, 'freq': freq}
    
    return {
        'prefix': match.group(1),
        'sector': match.group(2),
        'instrument': match.group(3),
        'suffix': match.group(4),
        'freq': freq
    }
```

**Afternoon: Core Graph Classes**

```python
# src/graph/sfc_graph.py
from dataclasses import dataclass
from typing import Optional
import networkx as nx

@dataclass(frozen=True)
class SFCNode:
    id: str
    kind: str  # 'series', 'aggregate', 'identity'
    prefix: Optional[str] = None
    sector: Optional[str] = None
    instrument: Optional[str] = None
    
@dataclass(frozen=True)  
class SFCEdge:
    src: str
    dst: str
    etype: str  # 'stock_flow', 'lag', 'aggregation'
    weight: float = 1.0
    lag: int = 0  # Temporal offset

class SFCGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self._node_index = {}
        
    def add_node(self, node: SFCNode):
        self.G.add_node(node.id,
                       kind=node.kind,
                       prefix=node.prefix,
                       sector=node.sector,
                       instrument=node.instrument)
        self._node_index[node.id] = node
        
    def add_edge(self, edge: SFCEdge):
        self.G.add_edge(edge.src, edge.dst,
                       etype=edge.etype,
                       weight=edge.weight,
                       lag=edge.lag)
```

### Testing Framework

**Golden Mini-SFC Test**

```python
# tests/graph/test_golden.py
def test_golden_mini_sfc():
    """Hand-derived constraints for 10-node system"""
    # 3 series: FL10, FU10, FR10 for sector 10, instrument X
    graph = build_mini_graph()
    state_index = StateIndex(['FL10', 'FU10', 'FR10'], max_lag=1)
    
    # At t=1, states are [FL10[t], FU10[t], FR10[t], FL10[t-1]]
    # Constraint: FL10[t] - FL10[t-1] - FU10[t] - FR10[t] = 0
    # Row 0: [1, -1, -1, -1]
    
    A, b, meta = extract_constraints_at_t(graph, state_index, t=1)
    
    expected_A = np.array([[1, -1, -1, -1]])
    expected_b = np.array([0])
    
    assert A.shape == (1, 4), f"Shape mismatch: {A.shape}"
    assert np.allclose(A.toarray(), expected_A), "A matrix mismatch"
    assert np.allclose(b, expected_b), "b vector mismatch"
    assert meta[0]['type'] == 'stock_flow'

def test_50_series_parity():
    """Exact match with legacy builder"""
    # Load current 50-series test data
    data = load_test_data()
    
    # Legacy path
    legacy_builder = SFCConstraintBuilder(...)
    A_legacy, b_legacy = legacy_builder.build_constraints(t=1)
    
    # Graph path
    graph = build_graph_from_data(data)
    state_index = StateIndex(data.columns, max_lag=2)
    A_graph, b_graph, _ = extract_constraints_at_t(graph, state_index, t=1)
    
    # Compare
    assert A_legacy.shape == A_graph.shape, f"Shape: {A_legacy.shape} vs {A_graph.shape}"
    assert A_legacy.nnz == A_graph.nnz, f"NNZ: {A_legacy.nnz} vs {A_graph.nnz}"
    
    max_diff = np.max(np.abs((A_legacy - A_graph).data)) if A_legacy.nnz > 0 else 0
    assert max_diff < 1e-12, f"Max difference: {max_diff}"

def test_missing_lag_error():
    """Ensure clear error for missing lagged state"""
    state_index = StateIndex(['FL10', 'FU10'], max_lag=0)  # No lags
    
    with pytest.raises(KeyError, match="State not found: FL10 at lag -1"):
        state_index.get('FL10', lag=-1)
```

**Switching Logic in Model**

```python
# src/models/sfc_kalman_proper.py
def _setup_constraints(self):
    """Set up constraint system with mode selection"""
    mode = self.config.get('constraints_mode', 'legacy')
    self.logger.info(f"Constraint mode: {mode}")
    
    if mode in ['graph', 'both']:
        # Build graph
        self.sfc_graph = build_graph_from_series(self.series_names,
                                                 self.stock_flow_pairs,
                                                 self.formulas)
        self.state_index = StateIndex(self.series_names, max_lag=2)
        self.graph_extractor = SFCConstraintExtractor(self.sfc_graph,
                                                      self.state_index)
        self.logger.info(f"  Graph: {self.sfc_graph.G.number_of_nodes()} nodes, "
                        f"{self.sfc_graph.G.number_of_edges()} edges")
    
    if mode in ['legacy', 'both']:
        # Legacy builder
        self.legacy_builder = SFCConstraintBuilder(...)
        self.logger.info(f"  Legacy builder initialized")
    
    if mode == 'both':
        self.constraint_validator = ConstraintValidator()

def build_constraints(self, t: int):
    """Build constraints with mode-aware logic"""
    mode = self.config.get('constraints_mode', 'legacy')
    
    if mode == 'legacy':
        return self.legacy_builder.build_constraints(t)
    
    elif mode == 'graph':
        A, b, meta = self.graph_extractor.extract_constraints_at_t(t)
        return A, b
    
    elif mode == 'both':
        # Build both
        A_legacy, b_legacy = self.legacy_builder.build_constraints(t)
        A_graph, b_graph, meta = self.graph_extractor.extract_constraints_at_t(t)
        
        # Compare and log
        diff_report = self.constraint_validator.compare(A_legacy, b_legacy,
                                                        A_graph, b_graph)
        self.logger.info(f"  Constraint diff at t={t}: max={diff_report['max_diff']:.2e}, "
                        f"rows={len(diff_report['mismatched_rows'])}")
        
        if diff_report['max_diff'] > 1e-10:
            self.logger.warning(f"  Constraint mismatch: {diff_report}")
        
        # Use legacy for projection (safe)
        return A_legacy, b_legacy
```

**Performance Monitoring**

```python
# Add to extractor
import time
from tqdm import tqdm

def extract_with_progress(self, t):
    """Extract with timing and progress"""
    start = time.time()
    
    with tqdm(total=3, desc=f"Constraints at t={t}") as pbar:
        # Stock-flow
        A_sf, b_sf, meta_sf = self._extract_stock_flow(t)
        pbar.update(1)
        
        # Aggregation  
        A_agg, b_agg, meta_agg = self._extract_aggregation(t)
        pbar.update(1)
        
        # Combine
        A = sparse.vstack([A_sf, A_agg])
        b = np.hstack([b_sf, b_agg])
        meta = meta_sf + meta_agg
        pbar.update(1)
    
    elapsed = time.time() - start
    self.logger.info(f"  Extracted {A.shape[0]} constraints in {elapsed:.3f}s")
    
    # Log condition number
    if A.shape[0] > 0:
        AAt = A @ A.T
        cond = np.linalg.cond(AAt.toarray()) if AAt.shape[0] < 100 else 0
        if cond > 1e12:
            self.logger.warning(f"  Poor conditioning: {cond:.2e}")
    
    return A, b, meta
```

---

## Appendix A: File Structure

```
Z1_Graph/
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── sfc_graph.py          # Core graph classes
│   │   ├── graph_builder.py      # Construction from data
│   │   ├── constraint_extractor.py # Graph → Constraints
│   │   ├── graph_validator.py    # Structural checks
│   │   └── graph_visualizer.py   # Export/rendering
│   ├── models/
│   │   ├── sfc_kalman_graph.py   # Graph-based Kalman filter
│   │   └── graph_projection.py   # Optimized projection
│   └── utils/
│       ├── series_parser.py      # Enhanced Z1 parsing
│       └── graph_analytics.py    # Network statistics
├── tests/
│   └── test_graph/
│       ├── test_construction.py
│       ├── test_constraints.py
│       └── test_scale.py
└── output/
    ├── graphs/
    │   ├── sfc_graph.graphml
    │   └── subgraphs/
    └── reports/
        └── constraint_validation.html
```

## Appendix B: Configuration Schema

```yaml
graph:
  construction:
    include_all_series: true
    min_series_length: 20
    handle_missing: "forward_fill"
  
  constraints:
    stock_flow:
      enforce: true
      tolerance: 1e-10
    market_clearing:
      enforce: true
      tolerance: 1e-6
    bilateral:
      enforce: true
      tolerance: 1e-4
      min_position_size: 1000
    aggregation:
      enforce: true
      tolerance: 1e-8
  
  performance:
    use_parallel: true
    n_workers: -1
    chunk_size: 1000
    sparse_format: "csr"
    solver: "splu"  # or "iterative"
  
  visualization:
    export_graphml: true
    generate_png: false  # Too large
    web_viewer: true
    subgraph_threshold: 500
```

## Appendix C: Risk Mitigation

| Risk                       | Impact | Mitigation                             |
| -------------------------- | ------ | -------------------------------------- |
| Graph too large for memory | High   | Streaming construction, graph database |
| Constraint conflicts       | High   | Priority system, soft constraints      |
| FWTW mapping failure       | Medium | Manual mapping table, fuzzy matching   |
| Numerical instability      | High   | Regularization, condition monitoring   |
| Performance degradation    | Medium | Profiling, parallel processing         |
| Visualization unusable     | Low    | Subgraph extraction, aggregation       |

---

*This plan provides the complete blueprint for implementing a graph-based SFC system capable of handling the full Z1 dataset. The incremental shadow implementation approach systematically builds confidence while maintaining the proven projection-based enforcement mechanism.*



---

## Change Log for V2 (this edit)

- Added **2.1 Reusing the Hierarchical Kalman Filter (HKF)** and wiring sketch
- Restored **10. Success Criteria & Deliverables** section (was in V1)
- Fixed missing filename label in the parser snippet
- No other content removed; structure otherwise unchanged



## 12. Actionable Work Plan (so we can start coding now)

> Goal for Sprint 1 (10 working days): land the graph layer, dual‑mode constraints, HKF wiring, and parity tests for the current 50‑series subset. Ship a runnable CLI flag `--constraints=legacy|graph|both` with reports and GraphML export.

### 12.1 Branching, environment, and repeatable commands

**Branching**

- Create feature branch: `git checkout -b feature/graph-constraints-v2`

**Python env**

```bash
pyenv local 3.12 || true
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -e .[dev]  # expects extras: dev = pytest, ruff, mypy, tqdm, networkx, scipy, numpy, pyarrow
pre-commit install
```

**Makefile (add at project root)**

```makefile
.PHONY: setup test lint type bench graphs run50 run_full

setup:
	pip install -e .[dev]
	pre-commit install

test:
	pytest -q

lint:
	ruff check src tests

type:
	mypy src

bench:
	python -m benchmarks.graph_extractor_bench --n_series 1000 --n_quarters 252

graphs:
	python -m tools.export_graph --out output/graphs/sfc_graph.graphml

run50:
	python examples/run_proper_sfc.py test --constraints=both --create_visualizations

run_full:
	python examples/run_proper_sfc.py full --constraints=graph --export_graphml
```

### 12.2 Files to create (exact paths) + minimal stubs

Create these files with the provided scaffolds. They compile and are unit‑testable as‑is.

``

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict
import networkx as nx
from .sfc_graph import SFCGraph, SFCNode, SFCEdge

@dataclass
class GraphBuilderConfig:
    include_all_series: bool = True
    min_series_length: int = 20

class SFCGraphBuilder:
    def __init__(self, config: GraphBuilderConfig | None = None):
        self.config = config or GraphBuilderConfig()

    def build_from_series(self, series_names: Iterable[str]) -> SFCGraph:
        G = SFCGraph()
        for name in series_names:
            meta = self._parse(name)
            node = SFCNode(id=name, kind='series', prefix=meta['prefix'],
                           sector=meta['sector'], instrument=meta['instrument'])
            G.add_node(node)
        # Stock‑flow edges filled later via add_stock_flow_relationships
        return G

    def add_stock_flow_relationships(self, graph: SFCGraph, pairs: Dict[str, Dict[str, str]]):
        # pairs example: { 'FLxxxx': {'FU': 'FUxxxx', 'FR': 'FRxxxx', 'FV':'FVxxxx'} }
        for fl, flows in pairs.items():
            for k in ('FU','FR','FV'):
                src = flows.get(k)
                if src and graph.G.has_node(src) and graph.G.has_node(fl):
                    graph.add_edge(SFCEdge(src=src, dst=fl, etype='stock_flow', weight=1.0))

    def _parse(self, code: str):
        from .series_parser import parse_z1_code
        return parse_z1_code(code)
```

``

```python
from __future__ import annotations
from typing import Tuple, List, Dict, Any
import numpy as np
from scipy import sparse
from .sfc_graph import SFCGraph

class StateIndex:
    def __init__(self, series_names: List[str], max_lag: int = 1):
        self.index: Dict[tuple[str,int], int] = {}
        idx = 0
        for name in series_names:
            self.index[(name, 0)] = idx; idx += 1
        for lag in range(1, max_lag+1):
            for name in series_names:
                self.index[(name, -lag)] = idx; idx += 1
        self.size = idx
    def get(self, name: str, lag: int = 0) -> int:
        key = (name, lag)
        if key not in self.index:
            raise KeyError(f"State not found: {name} at lag {lag}")
        return self.index[key]

class SFCConstraintExtractor:
    def __init__(self, graph: SFCGraph, state_index: StateIndex):
        self.graph = graph
        self.state_index = state_index

    def extract_stock_flow(self, t: int):
        rows: List[int] = []; cols: List[int] = []; data: List[float] = []
        meta: List[Dict[str, Any]] = []
        row = 0
        for n, d in self.graph.G.nodes(data=True):
            if d.get('prefix') != 'FL':
                continue
            # FL[t]
            cols.append(self.state_index.get(n, 0)); data.append(1.0); rows.append(row)
            # FL[t-1]
            if t > 0:
                cols.append(self.state_index.get(n, -1)); data.append(-1.0); rows.append(row)
            # Flows
            for src, _, ed in self.graph.G.in_edges(n, data=True):
                if ed.get('etype') == 'stock_flow':
                    cols.append(self.state_index.get(src, 0)); data.append(-ed.get('weight',1.0)); rows.append(row)
            meta.append({'type':'stock_flow','node':n,'time':t})
            row += 1
        A = sparse.coo_matrix((data, (rows, cols)), shape=(row, self.state_index.size)).tocsr()
        b = np.zeros(row)
        return A, b, meta

    def extract_at_time(self, t: int):
        A1, b1, m1 = self.extract_stock_flow(t)
        # TODO: add aggregation / market clearing; for now return stock‑flow only
        return A1, b1, m1
```

``

```python
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import sparse

def compare_constraints(A_legacy: sparse.spmatrix, b_legacy, A_graph: sparse.spmatrix, b_graph) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        'shape_match': A_legacy.shape == A_graph.shape,
        'nnz_legacy': A_legacy.nnz,
        'nnz_graph': A_graph.nnz,
        'max_diff': 0.0,
        'rows': int(min(A_legacy.shape[0], A_graph.shape[0])),
        'mismatched_rows': []
    }
    rows = report['rows']
    for i in range(rows):
        r1 = A_legacy.getrow(i).toarray(); r2 = A_graph.getrow(i).toarray()
        diff = float(np.linalg.norm(r1 - r2))
        if diff > 1e-12:
            report['mismatched_rows'].append({'row': i, 'diff': diff, 'nnz_legacy': int(np.count_nonzero(r1)), 'nnz_graph': int(np.count_nonzero(r2))})
    if report['mismatched_rows']:
        report['max_diff'] = max(r['diff'] for r in report['mismatched_rows'])
    return report
```

``

```python
from __future__ import annotations
from typing import Any, Tuple
import numpy as np
from .hierarchical_kalman_filter import HierarchicalKalmanFilter
from ..graph.constraint_extractor import SFCConstraintExtractor

class ProperSFCKalmanGraph(HierarchicalKalmanFilter):
    def __init__(self, *args, graph_extractor: SFCConstraintExtractor, projector, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_extractor = graph_extractor
        self.projector = projector

    def _project_t(self, x_t: np.ndarray, P_t: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        A_t, b_t, _ = self.graph_extractor.extract_at_time(t)
        return self.projector.project(x_t, P_t, A_t, b_t)

    def filter(self, params=None, **kwargs):
        res = super().filter(params=params, **kwargs)
        X, P = res.filtered_state, res.filtered_state_cov
        for t in range(X.shape[1]):
            X[:, t], P[:, :, t] = self._project_t(X[:, t], P[:, :, t], t)
        return res

    def smooth(self, params=None, **kwargs):
        res = super().smooth(params=params, **kwargs)
        X, P = res.smoothed_state, res.smoothed_state_cov
        for t in range(X.shape[1]):
            X[:, t], P[:, :, t] = self._project_t(X[:, t], P[:, :, t], t)
        return res
```

``

```python
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, cg

class Projector:
    def __init__(self, ridge: float = 1e-12):
        self.ridge = ridge

    def project(self, x: np.ndarray, P: np.ndarray, A: sparse.spmatrix, b: np.ndarray):
        if A.shape[0] == 0:
            return x, P
        # Solve min ||x - xhat||_P s.t. A xhat = b
        # Lambda = argmin (A P A^T) Lambda = A x - b
        APA = A @ P @ A.T
        APA = APA.tocsc() if sparse.issparse(APA) else sparse.csc_matrix(APA)
        n = APA.shape[0]
        APA = APA + self.ridge * sparse.eye(n, format='csc')
        rhs = (A @ x) - b
        try:
            lu = splu(APA)
            lam = lu.solve(rhs)
        except Exception:
            lam, info = cg(APA, rhs, tol=1e-10)
            if info != 0:
                raise RuntimeError(f"CG failed: info={info}")
        xhat = x - (P @ A.T @ lam)
        # Joseph form update for covariance projection
        At = A.todense() if sparse.issparse(A) and A.shape[0] < 200 else A
        K = P @ A.T @ self._solve(APA, np.eye(APA.shape[0]))
        P_new = P - K @ A @ P
        return xhat, P_new

    def _solve(self, APA, B):
        try:
            lu = splu(APA)
            return lu.solve(B)
        except Exception:
            # Dense fallback (small systems)
            return np.linalg.solve(APA.toarray(), B)
```

`` (sanity)

```python
from src.graph.series_parser import parse_z1_code

def test_parse_example():
    m = parse_z1_code('FL152064105.Q')
    assert m['prefix']=='FL' and m['sector']=='15' and m['instrument']=='20641' and m['freq']=='Q'
```

### 12.3 CLI wiring (exact lines to touch)

**File:** `examples/run_proper_sfc.py`

- Find the top‑level `argparse` block; add:

```python
parser.add_argument('--constraints', choices=['legacy','graph','both'], default='legacy')
parser.add_argument('--export_graphml', action='store_true')
```

- After model initialization (right after `Initialized Proper SFC Kalman Filter:` log), insert:

```python
self.config['constraints_mode'] = args.constraints
self.config['export_graphml'] = args.export_graphml
```

- In the method that sets up constraints (e.g., `_setup_constraints`), keep legacy init and add graph path using `SFCGraphBuilder`, `StateIndex`, `SFCConstraintExtractor`.
- In the save phase, if `export_graphml`, write: `nx.write_graphml(self.sfc_graph.G, 'output/graphs/sfc_graph.graphml')`.

### 12.4 Acceptance criteria per deliverable (Sprint 1)

- **Parser**: passes `tests/graph/test_parser.py`; handles `.Q` and `.A`; returns `None` fields for non‑matching codes.
- **GraphBuilder**: given 50 series and provided stock‑flow pairs, graph has ≥50 nodes, ≥15 stock‑flow edges; no duplicates.
- **Extractor**: `A_graph, b_graph` match legacy on 50‑series suite with `max_diff < 1e-12` and identical `nnz`.
- **HKF wiring**: running `make run50` writes filtered/smoothed states and logs `✓ All constraints satisfied within tolerance`.
- **Graph export**: `output/graphs/sfc_graph.graphml` exists and loads in Gephi.

### 12.5 Benchmarks and telemetry

- Add timing logs: `Constraint extraction took X.XXXs (rows=R, nnz=K)`
- Add condition number warning if `cond(A P A^T) > 1e12` (already in projector; keep it).
- Benchmark script `benchmarks/graph_extractor_bench.py` (optional in Sprint 1) to generate synthetic graphs and measure scaling.

### 12.6 Definition of Done (DoD)

- CI green: `make lint type test` passes.
- Reproducible run: `make run50` finishes in < 2 min on dev box.
- Parity report printed when `--constraints=both` with `max_diff < 1e-12`.
- Code documented with module docstrings and `README-graph.md` quickstart.

### 12.7 First three PRs (scope and titles)

1. **PR #1 – Graph scaffolding & parser**

   - Files: `series_parser.py`, `sfc_graph.py`, `graph_builder.py`, test\_parser
   - No model wiring; just structures. Acceptance: tests pass.

2. **PR #2 – Constraint extractor + validator + CLI flag**

   - Files: `constraint_extractor.py`, `graph_validator.py`, argparse changes, parity diff tool
   - Acceptance: parity on 50‑series suite.

3. **PR #3 – HKF projection wiring (graph mode) + GraphML export**

   - Files: `sfc_kalman_graph.py`, `graph_projection.py`, integration in `run_proper_sfc.py`
   - Acceptance: constraints satisfied; export written; runtime OK.

---

## Change Log for V2 (this edit #2)

- Added **12. Actionable Work Plan** with concrete file paths, code stubs, CLI wiring, DoD, and PR plan
- Included Makefile targets and exact argparse insert locations
- Provided minimal unit tests to gate progress

