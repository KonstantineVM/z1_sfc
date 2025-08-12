# Stock-Flow Consistent Graph Allocation for Z.1 and FWTW Data

## 1. Overview

This subproject implements **bilateral who-to-whom flow reconstruction** for the Federal Reserve’s Z.1 *Financial Accounts of the United States* (Flow of Funds), optionally using **FWTW** (Financial Whom-To-Whom) stocks when available.  
It produces empirically grounded **Stock-Flow Consistent (SFC)** matrices:

- **Balance Sheet** (stocks by sector × instrument)
- **Transaction-Flow Matrix** (flows by sector × instrument)
- **Bilateral Flows** (who-to-whom transaction flows per instrument or subgroup)

The core innovation is a **graph-based entropic optimal transport allocator**:
- Uses **row marginals** = holders’ asset acquisitions (FU, uses)  
- Uses **column marginals** = issuers’ liability incurrence (FU, sources)  
- Constrains flows to **allowed edges** in an economic connectivity graph  
- Optionally biases toward plausible flows via **cost hints** or **temporal priors**

This allows:
- Running **without** unreliable FWTW stock levels (graph-only mode)
- Using **FWTW as priors** when partial/reliable data exist
- Ensuring **stock-flow consistency** across instruments, sectors, and time

---

## 2. Key Components

### 2.1 Metadata and Mappings
| File | Purpose |
|------|---------|
| `mappings/instrument_map.json` | Maps each Z.1 series to **side** (asset/liability), **instrument class**, and **sign** for flows |
| `mappings/instrument_group_map.json` | Maps 5-digit instrument codes to **refined subgroups** (e.g., `Debt:Treasuries`, `Loans:Mortgages`) |
| `mappings/flow_map_expanded.json` | Behavioral flows for SFC macro rows (consumption, wages, taxes, transfers) |
| `mappings/sectors.yaml` / `mappings/sectors.csv` | Sector code dictionary |

### 2.2 Graph Specifications
| File | Purpose |
|------|---------|
| `graphs/graph_adjacency_spec.json` | **Coarse** instrument group connectivity and costs |
| `graphs/graph_adjacency_spec_refined.json` | **Fine-grained** subgroup connectivity and costs |

Each graph spec defines:
- `edges`: allowed sector → sector connections
- `cost_hints`: `prefer` or `penalize` certain flows
- `tau`: temperature (lower = sharper allocations, higher = smoother)

### 2.3 Allocator
| File | Purpose |
|------|---------|
| `src/alloc/graph_flow_allocator.py` | Masked entropic OT (Sinkhorn/IPF/RAS) to allocate flows from marginals + graph |

---

## 3. ASCII Diagrams

### 3.1 Sector Graph (Simplified)

```
[Households] --Loans--> [Banks] --Credit--> [Nonfinancial Corporations]
     |                                    ^
     v                                    |
 [Govt Bonds] <--Taxes-- [Federal Govt] --Spending--> [Households]
     |
     v
 [Rest of World] <--Trade--> [Nonfinancial Corporations]
```

**Legend:**  
- Arrows indicate allowed transaction directions in the **graph adjacency spec**.  
- Costs and preferences in `graph_adjacency_spec_refined.json` adjust the weight of these connections.

### 3.2 Instrument Subgroups

```
Debt
 ├─ Treasuries
 ├─ AgenciesGSE
 ├─ Munis
 ├─ Corporates
 ├─ ABS_MBS
 ├─ CommercialPaper

Loans
 ├─ Mortgages
 ├─ ConsumerCredit
 ├─ CommercialIndustrial
 ├─ Syndicated
 ├─ GSELoans
 ├─ GovernmentLoans
 └─ PolicyLoans

Equity
 ├─ CorporateEquity
 └─ MutualFundShares
```

---

## 4. Drivers

| Script | Mode | Uses FWTW? | Subgroups? | Temporal Priors? |
|--------|------|-----------|------------|------------------|
| `scripts/build_sfc_matrices_v2.py` | Stocks & flows only | No | No | No |
| `scripts/build_sfc_with_fwtw.py` | FU + ΔFWTW → bilateral flows | Yes (levels) | No | No |
| `scripts/build_sfc_graph_only.py` | FU marginals + coarse graph | No | No | No |
| `scripts/build_sfc_graph_refined.py` | FU marginals + refined graph | No / optional priors | Yes | Yes |

---

## 5. Installation and Setup

1. Clone the main `z1_sfc` repository (this subproject lives inside it).
2. Place the provided files:

```
z1_sfc/
├─ configs/
│   └─ proper_sfc_config.yaml
├─ mappings/
│   ├─ instrument_map.json
│   ├─ instrument_group_map.json
│   ├─ flow_map_expanded.json
│   ├─ sectors.yaml
│   └─ sectors.csv
├─ graphs/
│   ├─ graph_adjacency_spec.json
│   └─ graph_adjacency_spec_refined.json
├─ src/
│   └─ alloc/
│       └─ graph_flow_allocator.py
├─ scripts/
│   ├─ build_sfc_matrices_v2.py
│   ├─ build_sfc_with_fwtw.py
│   ├─ build_sfc_graph_only.py
│   └─ build_sfc_graph_refined.py
└─ outputs/
```

3. Add to PYTHONPATH:
```bash
export PYTHONPATH=src:.
```

---
# 6. Running the Unified SFC Driver

The four separate build scripts have been merged into a single core runner: `sfc_core.py`.  
You can now execute **all configurations** with a single command and one switch.

---

## 6.1 Baseline FL/FU Matrices (no bilateral)
```bash
python scripts/sfc_core.py baseline
```

---

## 6.2 Using FWTW Levels
```bash
python scripts/sfc_core.py fwtw
```

---

## 6.3 Graph-Only (Coarse)
```bash
python scripts/sfc_core.py graph
```

---

## 6.4 Graph-Refined + Temporal Priors
```bash
python scripts/sfc_core.py refined
```

---

## 6.5 Configuration

The driver uses `config/proper_sfc_config.yaml`.  
Add the following section (or ensure these keys exist):

```yaml
sfc:
  base_dir: ./data/fed_data
  cache_dir: ./data/cache
  output_dir: ./outputs
  date: 2024-12-31

  instrument_map: mappings/instrument_map.json
  flow_map: mappings/flow_map_expanded.json

  # Mode-specific
  fwtw: data/fwtw_levels.csv
  graph_spec: graphs/graph_adjacency_spec.json
  instrument_group_map: mappings/instrument_group_map.json
  graph_spec_refined: graphs/graph_adjacency_spec_refined.json
```

No more long CLI lines—just pick the mode you need.
```

---

## 7. Output Files

- `sfc_balance_sheet_YYYY-MM-DD.csv` — sector × instrument stocks (FL)
- `sfc_transactions_YYYY-MM-DD.csv` — sector × instrument flows (FU)
- `sfc_recon_YYYY-MM-DD.csv` — ΔFL − (FU+FR+FV)
- `w2w_graph_LONG_YYYY-MM-DD.csv` — stacked bilateral flows (holder, issuer, instrument/subgroup, value)
- `w2w_graph_{subgroup}_{YYYY-MM-DD}.csv` — matrix per subgroup

---

## 8. Methodology: Graph OT Allocation

Given:
- **Row marginals** r_i — FU assets by sector (uses)
- **Column marginals** c_j — FU liabilities by sector (sources)
- **Adjacency mask** A_{ij} ∈ {0,1} — allowed edges
- **Cost matrix** C_{ij} ≥ 0 — economic plausibility penalties
- **Prior matrix** P_{ij} ≥ 0 — last quarter’s flows or external guess

We solve:
```
min_X Σ_ij C_ij X_ij - τ H(X)
s.t. X 1 = r, 1^T X = c, X_ij = 0 if A_ij = 0
```
where H(X) = -Σ_ij X_ij log X_ij.

Solution via **masked Sinkhorn iterations** converges to an allocation that:
- Matches marginals exactly
- Respects connectivity graph
- Minimizes cost subject to entropy smoothing

---

## 9. Next Steps

- **Subgroup refinement**: e.g., split `Mortgages` into `1–4 family` vs `commercial`
- **Interest allocation**: yield × lagged stock, RAS to interest marginals
- **Mixed mode**: use ΔFWTW where reliable, graph OT elsewhere
