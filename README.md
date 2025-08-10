# Z1_SFC: Stock-Flow Consistent Kalman Filter for Federal Reserve Z.1 Data

## Project Overview

Z1_SFC is a sophisticated econometric framework that implements a **Stock-Flow Consistent (SFC) Kalman Filter** for analyzing Federal Reserve Z.1 (Flow of Funds) data with From-Whom-To-Whom (FWTW) bilateral positions. The system enforces macroeconomic accounting identities while filtering noisy financial time series data, providing a complete, internally consistent view of the U.S. financial system.

### Key Innovation

This project solves a fundamental challenge in macroeconomic analysis: maintaining exact accounting consistency while performing statistical filtering. Unlike traditional approaches that either ignore accounting constraints or enforce them post-hoc, Z1_SFC integrates constraints directly into the Kalman filter through optimal projection methods.

## Project Structure

```
Z1_SFC/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .env.example                      # Environment variables template
│
├── config/
│   ├── sfc_config.yaml              # Main SFC configuration
│   ├── sectors.yaml                 # Sector definitions and codes
│   ├── instruments.yaml             # Instrument definitions and codes
│   ├── run_modes.yaml               # Execution mode configurations
│   └── matrix_structure.yaml        # SFC matrix specifications
│
├── data/
│   ├── raw/                         # Raw data storage
│   │   ├── z1/                      # Federal Reserve Z.1 XML files
│   │   └── fwtw/                    # FWTW CSV files
│   ├── cache/                       # Cached processed data
│   ├── formulas/                    # Z.1 formula definitions
│   │   ├── z1_formulas.json         # Hierarchical formula structure
│   │   └── formula_tree.json        # Aggregation hierarchy
│   └── mappings/
│       ├── sector_mappings.json     # Sector code mappings
│       ├── instrument_mappings.json # Instrument code mappings
│       └── bilateral_mappings.json  # FWTW to Z1 mappings
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                        # Core SFC functionality
│   │   ├── __init__.py
│   │   ├── sfc_matrix.py           # SFC matrix structure
│   │   ├── sfc_constraints.py      # Constraint builder
│   │   ├── sfc_projection.py       # Optimal projection implementation
│   │   └── sfc_validation.py       # Constraint validation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── z1_loader.py            # Z.1 data loader with caching
│   │   ├── z1_processor.py         # Z.1 series processor
│   │   ├── fwtw_loader.py          # FWTW data loader
│   │   ├── fwtw_processor.py       # FWTW bilateral processor
│   │   └── data_aggregator.py      # Hierarchical aggregation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_kalman.py          # Base Kalman filter
│   │   ├── hierarchical_kalman.py  # Hierarchical state-space model
│   │   ├── sfc_kalman.py           # SFC-constrained Kalman filter
│   │   └── network_kalman.py       # Network-aware extensions
│   │
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── series_mapper.py        # Z.1 series code parser
│   │   ├── fwtw_z1_mapper.py       # FWTW to Z.1 mapping
│   │   ├── hierarchy_builder.py    # Build aggregation hierarchy
│   │   └── matrix_mapper.py        # Map to SFC matrix structure
│   │
│   ├── constraints/
│   │   ├── __init__.py
│   │   ├── stock_flow.py           # Stock-flow identities
│   │   ├── market_clearing.py      # Market clearing constraints
│   │   ├── bilateral.py            # Bilateral consistency
│   │   ├── hierarchical.py         # Hierarchical aggregation
│   │   └── formula_based.py        # Formula-based constraints
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── network_analysis.py     # Financial network metrics
│   │   ├── systemic_risk.py        # Systemic risk indicators
│   │   ├── flow_analysis.py        # Flow decomposition
│   │   └── forecast.py             # Constrained forecasting
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── matrix_viz.py           # SFC matrix visualization
│   │   ├── network_viz.py          # Network graph visualization
│   │   ├── constraint_viz.py       # Constraint satisfaction plots
│   │   └── dashboard.py            # Interactive dashboard
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       ├── validators.py           # Data validators
│       ├── sparse_utils.py         # Sparse matrix utilities
│       └── performance.py          # Performance monitoring
│
├── scripts/
│   ├── run_sfc_analysis.py         # Main execution script
│   ├── build_hierarchy.py          # Build formula hierarchy
│   ├── validate_constraints.py     # Validate SFC consistency
│   ├── generate_matrix.py          # Generate SFC matrices
│   └── benchmark.py                # Performance benchmarking
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore Z.1 and FWTW data
│   ├── 02_sfc_matrix_demo.ipynb    # SFC matrix construction
│   ├── 03_kalman_filtering.ipynb   # Kalman filter application
│   ├── 04_network_analysis.ipynb   # Network visualization
│   └── 05_case_studies.ipynb       # Economic case studies
│
├── tests/
│   ├── unit/
│   │   ├── test_sfc_matrix.py
│   │   ├── test_constraints.py
│   │   ├── test_projection.py
│   │   └── test_mapping.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_fwtw_integration.py
│   │   └── test_hierarchical.py
│   └── fixtures/
│       ├── sample_z1_data.py
│       └── sample_fwtw_data.py
│
├── docs/
│   ├── theory/
│   │   ├── sfc_theory.md           # SFC theoretical foundation
│   │   ├── kalman_filter.md        # Kalman filter mathematics
│   │   └── projection_method.md    # Constraint projection theory
│   ├── implementation/
│   │   ├── architecture.md         # System architecture
│   │   ├── data_flow.md           # Data processing pipeline
│   │   └── constraint_system.md    # Constraint implementation
│   └── api/
│       └── reference.md            # API documentation
│
└── output/
    ├── matrices/                    # Generated SFC matrices
    ├── filtered/                    # Filtered time series
    ├── reports/                     # Analysis reports
    └── visualizations/              # Generated plots

```

## Core Concepts

### 1. Z.1 Series Structure

Each Z.1 series follows a precise coding structure:

```
FL 15 30641 0 5 .Q
│  │  │     │ │ │
│  │  │     │ │ └── Frequency (Q=Quarterly, A=Annual)
│  │  │     │ └──── Calculation type (5=computed, 0=input)
│  │  │     └────── Always 0 (digit 8)
│  │  └──────────── Instrument code (5 digits)
│  └─────────────── Sector code (2 digits)
└────────────────── Prefix (FL=Level, FU=Flow, FR=Revaluation)
```

**Key Prefixes:**
- **FL**: Level/Stock (not seasonally adjusted)
- **FU**: Flow/Transaction (not seasonally adjusted)
- **FR**: Revaluation (price changes)
- **FV**: Other volume changes
- **FA**: Flow at seasonally adjusted annual rate (SAAR)

**Fundamental Identity:**
```
FL[t] = FL[t-1] + FU[t] + FR[t] + FV[t]
```

### 2. Hierarchical Structure

Z.1 data has a tree-like aggregation structure:

```
FL102000005 (Total nonfinancial assets)
├── FL102010005 (Nonfinancial assets)
│   ├── FL105035005 (Real estate)
│   │   ├── FL105035023 (Residential)
│   │   ├── FL105035033 (Nonresidential)
│   │   └── FL105010103 (Land)
│   └── FL105015205 (Equipment)
└── FL104090005 (Financial assets)
    ├── FL103030005 (Deposits)
    └── FL103064100 (Securities)
```

**Critical Insight**: Only leaf nodes have complete {FL, FU, FR, FV} quartets. Aggregates must be built bottom-up while preserving SFC identities.

### 3. SFC Matrix Structure

The system maps Z.1 series to a structured matrix format:

```
                 │ House │ Corps │ Banks │ Govt  │ Fed  │ RoW  │ Sum
─────────────────┼───────┼───────┼───────┼───────┼──────┼──────┼─────
Deposits         │ +500  │ +200  │ -1200 │ +100  │ +300 │ +100 │  0
Treasuries       │ +300  │ +100  │ +800  │ -2000 │ +500 │ +300 │  0
Corp Equity      │ +2000 │ -3000 │ +200  │   0   │  0   │ +800 │  0
Mortgages        │ -1500 │ -300  │ +1600 │   0   │  0   │ +200 │  0
─────────────────┼───────┼───────┼───────┼───────┼──────┼──────┼─────
Net Worth        │ +3000 │ +1500 │ +200  │ -2000 │ +100 │ -800 │  0
```

**Properties:**
- Every row sums to zero (market clearing)
- Every column sums to zero (balance sheet identity)
- Every financial asset has a corresponding liability

### 4. FWTW Integration

From-Whom-To-Whom data provides bilateral positions:

```
Date: 2023-Q4
Holder: 15 (Households)
Issuer: 10 (NonFin Corps)
Instrument: 30641 (Equity)
Level: $2,500,000 million
```

Maps to Z.1 series:
- Asset: `FL1530641005` (Household equity holdings)
- Liability: `FL1030641005` (Corporate equity outstanding)

### 5. Constraint System

#### Stock-Flow Constraints
```python
FL[t] - FL[t-1] - FU[t] - FR[t] - FV[t] = 0
```

#### Market Clearing Constraints
```python
Σ(Assets in instrument i) - Σ(Liabilities in instrument i) = 0
```

#### Bilateral Aggregation Constraints
```python
Aggregate_position = Σ(All bilateral positions)
```

#### Hierarchical Constraints
```python
Parent_series = Σ(Child_series)
```

### 6. Kalman Filter Architecture

The system uses a three-layer architecture:

1. **Base Layer** (`HierarchicalKalmanFilter`):
   - Standard state-space model
   - State: `[Level, Trend, Lags...]`
   - Handles missing data naturally

2. **SFC Extension** (`SFCKalmanFilter`):
   - Extends state space (optional)
   - Adds flow and revaluation states
   - Preserves shock structure

3. **Constraint Projection** (`ConstraintProjector`):
   - Optimal projection: `x* = x̂ - P·A'(APA')⁻¹(Ax̂ - b)`
   - Updates covariance consistently
   - Handles sparse systems efficiently

## Installation

### Prerequisites

- Python 3.8+
- 16GB RAM minimum (32GB recommended for full dataset)
- 10GB disk space for data cache

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Z1_SFC.git
cd Z1_SFC
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Download initial data:
```bash
python scripts/download_data.py
```

## Usage

### Quick Start

```python
from src.core import SFCMatrix
from src.models import SFCKalmanFilter
from src.data import Z1Loader, FWTWLoader

# Load data
z1_data = Z1Loader().load_cached('Z1')
fwtw_data = FWTWLoader().load_cached()

# Build SFC matrix
matrix = SFCMatrix()
matrix.build_from_z1(z1_data, period='2023-Q4')

# Run Kalman filter with constraints
filter = SFCKalmanFilter(
    data=z1_data,
    fwtw_data=fwtw_data,
    enforce_sfc=True,
    enforce_bilateral=True
)
results = filter.fit()

# Validate constraints
validation = filter.validate_constraints()
print(f"SFC violations: {validation['max_violation']:.2e}")
```

### Running Full Analysis

```bash
# Test mode (small subset)
python scripts/run_sfc_analysis.py --mode test

# Development mode (medium dataset)
python scripts/run_sfc_analysis.py --mode development

# Production mode (full dataset)
python scripts/run_sfc_analysis.py --mode production

# With specific configuration
python scripts/run_sfc_analysis.py --config config/custom.yaml
```

### Configuration Options

Edit `config/sfc_config.yaml`:

```yaml
sfc:
  # Data selection
  max_series: 500
  priority_sectors: [15, 10, 31, 70, 71]  # Key sectors
  
  # Constraints
  enforce_sfc: true
  enforce_market_clearing: true
  enforce_bilateral: true
  enforce_hierarchical: true
  
  # Numerical settings
  use_sparse: true
  tolerance: 1e-10
  max_iterations: 100
  
  # Variance ratios
  error_variance_ratio: 0.01
  revaluation_variance_ratio: 0.1
  bilateral_variance_ratio: 0.05
```

## Key Algorithms

### 1. Optimal Projection

The constraint projection minimizes information loss while enforcing constraints exactly:

```python
def project_onto_constraints(x, P, A, b):
    """
    Project state x onto constraint manifold Ax = b
    
    Minimizes: ||x* - x||²_P⁻¹
    Subject to: Ax* = b
    
    Solution: x* = x - P·A'·(A·P·A')⁻¹·(A·x - b)
    """
    residual = A @ x - b
    
    if norm(residual) < tolerance:
        return x, P  # Already satisfied
    
    # Compute gain matrix
    APA = A @ P @ A.T
    gain = P @ A.T @ inv(APA)
    
    # Project state
    x_proj = x - gain @ residual
    
    # Update covariance (Joseph form)
    I_GA = I - gain @ A
    P_proj = I_GA @ P @ I_GA.T + gain @ R @ gain.T
    
    return x_proj, P_proj
```

### 2. Hierarchical Aggregation

Build constraint system respecting the tree structure:

```python
def build_hierarchical_constraints(tree, t):
    """
    Build constraints from formula tree
    """
    constraints = []
    
    # Level 1: Leaf-level SFC
    for leaf in tree.get_leaves():
        fl, fu, fr, fv = leaf.get_components()
        # FL[t] - FL[t-1] - FU - FR - FV = 0
        constraints.append(build_sfc_constraint(fl, fu, fr, fv, t))
    
    # Level 2: Aggregation up tree
    for level in tree.get_levels(bottom_up=True):
        for parent, children in level.items():
            # Parent = Σ(Children)
            constraints.append(build_sum_constraint(parent, children))
    
    # Level 3: Cross-sectional balance
    for instrument in tree.get_instruments():
        # Σ(Assets) = Σ(Liabilities)
        constraints.append(build_market_clearing(instrument))
    
    return stack_constraints(constraints)
```

### 3. Bilateral Flow Calculation

Calculate flows from FWTW stock changes:

```python
def calculate_bilateral_flows(fwtw_positions):
    """
    Flow[t] = Stock[t] - Stock[t-1]
    """
    flows = []
    
    for (holder, issuer, instrument), time_series in fwtw_positions.groupby(
        ['holder', 'issuer', 'instrument']
    ):
        for t in range(1, len(time_series)):
            flow = {
                'date': time_series.iloc[t]['date'],
                'holder': holder,
                'issuer': issuer,
                'instrument': instrument,
                'flow': time_series.iloc[t]['level'] - time_series.iloc[t-1]['level'],
                'stock_t': time_series.iloc[t]['level'],
                'stock_t_1': time_series.iloc[t-1]['level']
            }
            flows.append(flow)
    
    return pd.DataFrame(flows)
```

## Performance Considerations

### Memory Management

For large datasets (2000+ series):

1. **Use sparse matrices**:
```python
config['use_sparse'] = True
```

2. **Limit series selection**:
```python
config['max_series'] = 500
config['priority_sectors'] = [15, 10, 31, 70, 71]
```

3. **Process in chunks**:
```python
for chunk in data.chunk_by_date(chunk_size='1Y'):
    results = filter.process_chunk(chunk)
```

### Computational Optimization

1. **Parallel constraint building**:
```python
from joblib import Parallel, delayed

constraints = Parallel(n_jobs=-1)(
    delayed(build_constraint)(series) 
    for series in series_list
)
```

2. **Cached matrix decompositions**:
```python
# LU decomposition for repeated solves
lu = splu(A @ P @ A.T)
for iteration in range(max_iter):
    gain = lu.solve(A @ P)
```

3. **Selective constraint enforcement**:
```python
# Enforce only binding constraints
active_set = identify_binding_constraints(x, A, b, tolerance)
A_active = A[active_set]
b_active = b[active_set]
```

## Validation and Testing

### Constraint Validation

The system provides comprehensive validation:

```python
validation = filter.validate_constraints()

print("Stock-Flow Identity:")
print(f"  Max violation: {validation['sfc']['max_violation']:.2e}")
print(f"  Mean violation: {validation['sfc']['mean_violation']:.2e}")
print(f"  Violated series: {validation['sfc']['violated_count']}")

print("\nMarket Clearing:")
print(f"  Max imbalance: {validation['market']['max_imbalance']:.2e}")
print(f"  Instruments cleared: {validation['market']['cleared_count']}/
      {validation['market']['total_count']}")

print("\nBilateral Consistency:")
print(f"  Max discrepancy: {validation['bilateral']['max_discrepancy']:.2e}")
print(f"  Consistent pairs: {validation['bilateral']['consistent_count']}")
```

### Unit Tests

Run test suite:

```bash
# All tests
pytest

# Specific module
pytest tests/unit/test_sfc_matrix.py

# With coverage
pytest --cov=src --cov-report=html

# Integration tests only
pytest tests/integration/
```

### Benchmarking

Benchmark performance:

```bash
python scripts/benchmark.py --series [100, 500, 1000, 2000]
```

Expected performance:
- 100 series: ~5 seconds
- 500 series: ~30 seconds
- 1000 series: ~2 minutes
- 2000 series: ~10 minutes

## Case Studies

### 1. 2008 Financial Crisis

Analyze contagion through bilateral exposures:

```python
# Load crisis period data
crisis_data = z1_loader.load_period('2007-Q1', '2009-Q4')

# Focus on critical sectors
sectors = ['70', '66', '65']  # Banks, Brokers, Funds

# Build network
network = NetworkBuilder.from_fwtw(fwtw_data, sectors)

# Analyze contagion
contagion = analyze_contagion(
    network,
    shock={'sector': '66', 'magnitude': -0.3}  # Lehman shock
)
```

### 2. COVID-19 Market Disruption

Track Fed intervention effects:

```python
# Load COVID period
covid_data = z1_loader.load_period('2019-Q4', '2021-Q4')

# Focus on Fed balance sheet
fed_series = filter_series(covid_data, sector='71')

# Analyze QE impact
qe_impact = analyze_qe(
    fed_series,
    market_series=filter_series(covid_data, instrument='30611')  # Treasuries
)
```

### 3. Household Wealth Distribution

Decompose wealth changes:

```python
# Load household data
household = filter_series(z1_data, sector='15')

# Decompose wealth changes
decomposition = decompose_wealth_changes(
    household,
    components=['savings', 'capital_gains', 'transfers']
)

# Analyze distribution
distribution = analyze_distribution(
    decomposition,
    percentiles=[50, 90, 99, 99.9]
)
```

## Mathematical Foundation

### State-Space Model

The hierarchical Kalman filter uses:

**State Equation:**
```
x[t] = T·x[t-1] + R·η[t]
```

**Observation Equation:**
```
y[t] = Z·x[t] + ε[t]
```

Where:
- `x[t]` = State vector (levels, trends, lags)
- `y[t]` = Observed Z.1 series
- `T` = Transition matrix
- `Z` = Observation matrix
- `η[t]` ~ N(0, Q) = State shocks
- `ε[t]` ~ N(0, H) = Measurement errors

### Constraint Projection

The optimal projection solves:

```
minimize   (x - x̂)'P⁻¹(x - x̂)
subject to  Ax = b
```

Using Lagrangian method:
```
L = (x - x̂)'P⁻¹(x - x̂) + λ'(Ax - b)
```

First-order conditions yield:
```
x* = x̂ - P·A'·(A·P·A')⁻¹·(A·x̂ - b)
```

### SFC Accounting

The fundamental identity:
```
ΔWealth = Income - Consumption + Capital_Gains
```

In matrix form:
```
ΔNW = Y - C + ΔA·p
```

Where:
- `NW` = Net worth
- `Y` = Income flows
- `C` = Consumption flows
- `A` = Asset quantities
- `p` = Asset prices

## Troubleshooting

### Common Issues

1. **Memory Error**
   - Reduce `max_series` in config
   - Enable `use_sparse: true`
   - Increase system swap space

2. **Constraint Violations**
   - Check data quality
   - Increase `tolerance` slightly
   - Verify formula definitions

3. **FWTW Mapping Failures**
   - Ensure consistent sector codes
   - Check instrument mapping file
   - Verify .Q suffix handling

4. **Slow Performance**
   - Use development mode first
   - Enable sparse matrices
   - Reduce constraint set

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
mypy src/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{z1_sfc,
  title = {Z1_SFC: Stock-Flow Consistent Kalman Filter for Federal Reserve Z.1 Data},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Z1_SFC}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Federal Reserve Board for Z.1 data
- Board of Governors for FWTW data
- Anthropic for technical consultation
- Open source contributors

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/Z1_SFC/issues)
- Email: your.email@example.com
- Documentation: [https://z1-sfc.readthedocs.io](https://z1-sfc.readthedocs.io)

---

**Note**: This project implements cutting-edge econometric methods. Users should have a solid understanding of:
- Stock-Flow Consistent modeling
- Kalman filtering
- National accounts
- Matrix algebra
- Python programming

For theoretical background, see the `/docs/theory/` directory.