#!/usr/bin/env python3
"""
PLACEMENT: src/graph/fwtw_graph_builder.py

Builds SFC graph from FWTW bilateral data and Z1 series.
This is the main builder that creates the complete graph structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging
import re

from .sfc_graph import SFCGraph, SFCNode, SFCEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class FWTWGraphBuilder:
    """
    Builds SFC graph from FWTW bilateral positions and Z1 series.
    
    This builder:
    1. Loads FWTW bilateral positions
    2. Creates sector and instrument nodes
    3. Maps bilateral positions to Z1 series
    4. Identifies stock-flow relationships
    5. Builds complete constraint graph
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = SFCGraph()
        self.bilateral_positions = {}  # (holder, issuer, instrument, date) -> level
        self.z1_series_map = {}  # series_code -> parsed components
        self.stock_flow_pairs = []  # List of identified stock-flow pairs
        
        # Statistics
        self.stats = {
            'n_fwtw_records': 0,
            'n_bilateral_positions': 0,
            'n_z1_series': 0,
            'n_stock_flow_pairs': 0,
            'n_unmapped_bilateral': 0,
            'n_unmapped_series': 0
        }
    
    def build_from_fwtw(self, fwtw_df: pd.DataFrame) -> SFCGraph:
        """
        Build graph from FWTW data.
        
        Parameters:
        -----------
        fwtw_df : pd.DataFrame
            FWTW data with columns:
            - Instrument Code
            - Holder Code
            - Issuer Code
            - Date
            - Level
            - Instrument Name (optional)
            - Holder Name (optional)
            - Issuer Name (optional)
        
        Returns:
        --------
        SFCGraph
            Complete graph structure
        """
        logger.info(f"Building graph from {len(fwtw_df)} FWTW records")
        self.stats['n_fwtw_records'] = len(fwtw_df)
        
        # Step 1: Add sector nodes
        self._add_sector_nodes(fwtw_df)
        
        # Step 2: Add instrument nodes
        self._add_instrument_nodes(fwtw_df)
        
        # Step 3: Add bilateral position nodes
        self._add_bilateral_positions(fwtw_df)
        
        logger.info(f"Graph built with {self.graph.G.number_of_nodes()} nodes, "
                   f"{self.graph.G.number_of_edges()} edges")
        
        return self.graph
    
    def _add_sector_nodes(self, fwtw_df: pd.DataFrame) -> None:
        """Add sector nodes from FWTW data."""
        # Get unique holders and issuers
        holders = fwtw_df[['Holder Code', 'Holder Name']].drop_duplicates()
        issuers = fwtw_df[['Issuer Code', 'Issuer Name']].drop_duplicates()
        
        # Combine and deduplicate
        sectors = pd.concat([
            holders.rename(columns={'Holder Code': 'Code', 'Holder Name': 'Name'}),
            issuers.rename(columns={'Issuer Code': 'Code', 'Issuer Name': 'Name'})
        ]).drop_duplicates()
        
        for _, row in sectors.iterrows():
            node = SFCNode(
                id=f"SECTOR_{row['Code']}",
                node_type=NodeType.SECTOR,
                metadata={
                    'code': row['Code'],
                    'name': row.get('Name', f"Sector {row['Code']}")
                }
            )
            self.graph.add_node(node)
            logger.debug(f"Added sector node: {node.id}")
    
    def _add_instrument_nodes(self, fwtw_df: pd.DataFrame) -> None:
        """Add instrument nodes from FWTW data."""
        instruments = fwtw_df[['Instrument Code', 'Instrument Name']].drop_duplicates()
        
        for _, row in instruments.iterrows():
            node = SFCNode(
                id=f"INST_{row['Instrument Code']}",
                node_type=NodeType.INSTRUMENT,
                metadata={
                    'code': row['Instrument Code'],
                    'name': row.get('Instrument Name', f"Instrument {row['Instrument Code']}")
                }
            )
            self.graph.add_node(node)
            logger.debug(f"Added instrument node: {node.id}")
    
    def _add_bilateral_positions(self, fwtw_df: pd.DataFrame) -> None:
        """Add bilateral position nodes and edges."""
        # Group by unique bilateral positions
        position_groups = fwtw_df.groupby(['Holder Code', 'Issuer Code', 'Instrument Code'])
        
        for (holder, issuer, instrument), group in position_groups:
            # Create bilateral position node
            bilateral_id = f"BILATERAL_{holder}_{issuer}_{instrument}"
            node = SFCNode(
                id=bilateral_id,
                node_type=NodeType.BILATERAL,
                metadata={
                    'holder': holder,
                    'issuer': issuer,
                    'instrument': instrument,
                    'dates': group['Date'].unique().tolist(),
                    'levels': group.set_index('Date')['Level'].to_dict()
                }
            )
            self.graph.add_node(node)
            
            # Store bilateral position data
            for _, row in group.iterrows():
                key = (holder, issuer, instrument, row['Date'])
                self.bilateral_positions[key] = row['Level']
            
            # Add edges: Holder -> Bilateral (asset)
            holder_edge = SFCEdge(
                source=f"SECTOR_{holder}",
                target=bilateral_id,
                edge_type=EdgeType.BILATERAL_ASSET,
                metadata={'instrument': instrument}
            )
            self.graph.add_edge(holder_edge)
            
            # Add edges: Bilateral -> Issuer (liability)
            issuer_edge = SFCEdge(
                source=bilateral_id,
                target=f"SECTOR_{issuer}",
                edge_type=EdgeType.BILATERAL_LIABILITY,
                metadata={'instrument': instrument}
            )
            self.graph.add_edge(issuer_edge)
            
            # Add edge: Bilateral -> Instrument
            inst_edge = SFCEdge(
                source=bilateral_id,
                target=f"INST_{instrument}",
                edge_type=EdgeType.REPRESENTS,
                metadata={'holder': holder, 'issuer': issuer}
            )
            self.graph.add_edge(inst_edge)
        
        self.stats['n_bilateral_positions'] = len(self.bilateral_positions)
        logger.info(f"Added {len(position_groups)} bilateral position nodes")
    
    def add_z1_series(self, z1_series: List[str]) -> None:
        """
        Add Z1 series to the graph and map to FWTW positions.
        
        Parameters:
        -----------
        z1_series : List[str]
            List of Z1 series codes
        """
        logger.info(f"Adding {len(z1_series)} Z1 series to graph")
        self.stats['n_z1_series'] = len(z1_series)
        
        for series_code in z1_series:
            # Parse series code
            parsed = self._parse_z1_series(series_code)
            if not parsed:
                logger.warning(f"Could not parse series: {series_code}")
                continue
            
            self.z1_series_map[series_code] = parsed
            
            # Add series node
            node = SFCNode(
                id=series_code,
                node_type=NodeType.SERIES,
                metadata=parsed
            )
            self.graph.add_node(node)
            
            # Try to map to FWTW bilateral position
            self._map_series_to_bilateral(series_code, parsed)
    
    def _parse_z1_series(self, series_code: str) -> Optional[Dict]:
        """
        Parse Z1 series code into components.
        
        Z1 Format: PPSSIIIIICCC.F
        PP = Prefix (FL, FU, FR, FV, FA, LA)
        SS = Sector (2 digits)
        IIIII = Instrument (5 digits)
        CCC = Calculation suffix (3 digits, digit 9 is key)
        F = Frequency (Q or A)
        """
        # Remove frequency suffix
        if series_code.endswith('.Q'):
            base_code = series_code[:-2]
            freq = 'Q'
        elif series_code.endswith('.A'):
            base_code = series_code[:-2]
            freq = 'A'
        else:
            base_code = series_code
            freq = None
        
        # Match pattern
        pattern = r'^([A-Z]{2})(\d{2})(\d{5})(\d{3})$'
        match = re.match(pattern, base_code)
        
        if not match:
            return None
        
        prefix, sector, instrument, suffix = match.groups()
        
        # Determine series type
        series_type = self._get_series_type(prefix)
        
        # Check if it's a base series (digit 9 = 0 or 3)
        is_base = suffix[0] in ['0', '3']
        
        return {
            'prefix': prefix,
            'sector': sector,
            'instrument': instrument,
            'suffix': suffix,
            'frequency': freq,
            'series_type': series_type,
            'is_base': is_base,
            'digit_9': suffix[0] if len(suffix) > 0 else None
        }
    
    def _get_series_type(self, prefix: str) -> str:
        """Get series type from prefix."""
        type_map = {
            'FL': 'level',           # Stock/Level
            'FU': 'transaction',     # Flow/Transaction
            'FR': 'revaluation',     # Revaluation
            'FV': 'other_change',    # Other volume change
            'FA': 'flow_saar',       # Flow, seasonally adjusted
            'LA': 'level_sa',        # Level, seasonally adjusted
            'FC': 'change',          # Change in level
            'FG': 'growth_rate'      # Growth rate
        }
        return type_map.get(prefix, 'unknown')
    
    def _map_series_to_bilateral(self, series_code: str, parsed: Dict) -> None:
        """Map Z1 series to FWTW bilateral positions."""
        sector = parsed['sector']
        instrument = parsed['instrument']
        
        # Find matching bilateral positions
        # This is where holder = sector for assets
        # or issuer = sector for liabilities
        
        matches = []
        for (h, i, inst, date), level in self.bilateral_positions.items():
            # Check if this series could represent this bilateral position
            if inst == instrument:
                if h == sector:  # This sector holds the instrument
                    matches.append(('asset', h, i, inst))
                elif i == sector:  # This sector issued the instrument
                    matches.append(('liability', h, i, inst))
        
        if matches:
            # Create mapping edges
            for position_type, holder, issuer, inst in matches:
                bilateral_id = f"BILATERAL_{holder}_{issuer}_{inst}"
                if bilateral_id in self.graph._node_index:
                    edge = SFCEdge(
                        source=series_code,
                        target=bilateral_id,
                        edge_type=EdgeType.REPRESENTS,
                        metadata={
                            'position_type': position_type,
                            'mapping_confidence': 0.8  # Could be improved with better matching
                        }
                    )
                    self.graph.add_edge(edge)
                    logger.debug(f"Mapped {series_code} to {bilateral_id}")
        else:
            self.stats['n_unmapped_series'] += 1
    
    def identify_stock_flow_pairs(self) -> List[Dict]:
        """
        Identify stock-flow pairs in the graph.
        Only works for base series (digit 9 = 0 or 3).
        """
        logger.info("Identifying stock-flow pairs")
        pairs = []
        
        # Find all FL (stock) series that are base series
        fl_series = [
            code for code, parsed in self.z1_series_map.items()
            if parsed['prefix'] == 'FL' and parsed['is_base']
        ]
        
        for fl in fl_series:
            parsed = self.z1_series_map[fl]
            base = f"{parsed['prefix']}{parsed['sector']}{parsed['instrument']}{parsed['suffix']}"
            
            # Look for corresponding flows
            fu = base.replace('FL', 'FU') + (f".{parsed['frequency']}" if parsed['frequency'] else "")
            fr = base.replace('FL', 'FR') + (f".{parsed['frequency']}" if parsed['frequency'] else "")
            fv = base.replace('FL', 'FV') + (f".{parsed['frequency']}" if parsed['frequency'] else "")
            
            # Check which exist
            existing_flows = {}
            for flow_type, flow_code in [('FU', fu), ('FR', fr), ('FV', fv)]:
                if flow_code in self.z1_series_map:
                    existing_flows[flow_type] = flow_code
                    
                    # Add stock-flow edge
                    edge = SFCEdge(
                        source=flow_code,
                        target=fl,
                        edge_type=EdgeType.STOCK_FLOW,
                        metadata={'flow_type': flow_type}
                    )
                    self.graph.add_edge(edge)
            
            if existing_flows:
                pair = {
                    'stock': fl,
                    'flows': existing_flows,
                    'sector': parsed['sector'],
                    'instrument': parsed['instrument']
                }
                pairs.append(pair)
                self.stock_flow_pairs.append(pair)
        
        self.stats['n_stock_flow_pairs'] = len(pairs)
        logger.info(f"Found {len(pairs)} stock-flow pairs")
        return pairs
    
    def add_formula_relationships(self, formulas: Dict) -> None:
        """
        Add formula-based aggregation relationships.
        
        Parameters:
        -----------
        formulas : Dict
            Dictionary of formulas from Z1 documentation
        """
        logger.info(f"Adding formula relationships from {len(formulas)} formulas")
        
        for target_series, formula in formulas.items():
            if 'derived_from' not in formula:
                continue
            
            # Ensure target series exists
            if target_series not in self.graph._series_index:
                parsed = self._parse_z1_series(target_series)
                if parsed:
                    node = SFCNode(
                        id=target_series,
                        node_type=NodeType.AGGREGATE,
                        metadata=parsed
                    )
                    self.graph.add_node(node)
            
            # Add edges from components to target
            for component in formula['derived_from']:
                comp_series = component.get('code', '')
                operator = component.get('operator', '+')
                weight = 1.0 if operator == '+' else -1.0
                
                if comp_series:
                    # Ensure component exists
                    if comp_series not in self.graph._series_index:
                        parsed = self._parse_z1_series(comp_series)
                        if parsed:
                            node = SFCNode(
                                id=comp_series,
                                node_type=NodeType.SERIES,
                                metadata=parsed
                            )
                            self.graph.add_node(node)
                    
                    # Add aggregation edge
                    edge = SFCEdge(
                        source=comp_series,
                        target=target_series,
                        edge_type=EdgeType.AGGREGATES_TO,
                        weight=weight,
                        metadata={'formula_id': formula.get('id')}
                    )
                    self.graph.add_edge(edge)
    
    def build_market_clearing_constraints(self) -> List[Dict]:
        """
        Build market clearing constraints from bilateral positions.
        For each instrument: Σ(Assets) = Σ(Liabilities)
        """
        constraints = []
        
        # Group bilateral positions by instrument
        instrument_positions = defaultdict(list)
        for (holder, issuer, instrument, date), level in self.bilateral_positions.items():
            instrument_positions[instrument].append({
                'holder': holder,
                'issuer': issuer,
                'date': date,
                'level': level
            })
        
        for instrument, positions in instrument_positions.items():
            # For each date, assets should equal liabilities
            dates = set(p['date'] for p in positions)
            
            for date in dates:
                constraint = {
                    'type': 'market_clearing',
                    'instrument': instrument,
                    'date': date,
                    'positions': []
                }
                
                for pos in positions:
                    if pos['date'] == date:
                        # This represents an asset for holder, liability for issuer
                        constraint['positions'].append({
                            'asset_holder': pos['holder'],
                            'liability_issuer': pos['issuer'],
                            'level': pos['level']
                        })
                
                constraints.append(constraint)
        
        logger.info(f"Built {len(constraints)} market clearing constraints")
        return constraints
    
    def export_to_graphml(self, filepath: str) -> None:
        """Export graph to GraphML format for visualization."""
        import networkx as nx
        
        # Convert to format suitable for export
        export_graph = nx.MultiDiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.graph._node_index.items():
            attrs = {
                'node_type': node.node_type.value,
                **node.metadata
            }
            export_graph.add_node(node_id, **attrs)
        
        # Add edges with attributes
        for u, v, data in self.graph.G.edges(data=True):
            attrs = {
                'edge_type': data.get('edge_type', 'unknown'),
                'weight': data.get('weight', 1.0),
                **data.get('metadata', {})
            }
            export_graph.add_edge(u, v, **attrs)
        
        # Write to file
        nx.write_graphml(export_graph, filepath)
        logger.info(f"Exported graph to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get graph building statistics."""
        self.stats.update({
            'n_nodes': self.graph.G.number_of_nodes(),
            'n_edges': self.graph.G.number_of_edges(),
            'n_sectors': len(self.graph.get_nodes_by_type(NodeType.SECTOR)),
            'n_instruments': len(self.graph.get_nodes_by_type(NodeType.INSTRUMENT)),
            'n_bilateral': len(self.graph.get_nodes_by_type(NodeType.BILATERAL)),
            'n_series': len(self.graph.get_nodes_by_type(NodeType.SERIES))
        })
        return self.stats