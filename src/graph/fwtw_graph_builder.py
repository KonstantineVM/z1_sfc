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
        # Get unique sectors
        holder_sectors = fwtw_df[['Holder Code', 'Holder Name']].drop_duplicates()
        issuer_sectors = fwtw_df[['Issuer Code', 'Issuer Name']].drop_duplicates()
        
        # Combine and deduplicate
        all_sectors = {}
        
        for _, row in holder_sectors.iterrows():
            code = str(row['Holder Code']).zfill(2)
            name = row.get('Holder Name', f'Sector_{code}')
            all_sectors[code] = name
        
        for _, row in issuer_sectors.iterrows():
            code = str(row['Issuer Code']).zfill(2)
            name = row.get('Issuer Name', f'Sector_{code}')
            all_sectors[code] = name
        
        # Add sector nodes
        for code, name in all_sectors.items():
            node = SFCNode(
                id=f"SECTOR_{code}",
                node_type=NodeType.SECTOR,
                metadata={
                    'code': code,
                    'name': name,
                    'sector_type': self._classify_sector(code)
                }
            )
            self.graph.add_node(node)
        
        logger.info(f"Added {len(all_sectors)} sector nodes")
    
    def _add_instrument_nodes(self, fwtw_df: pd.DataFrame) -> None:
        """Add instrument nodes from FWTW data."""
        # Get unique instruments
        instruments = fwtw_df[['Instrument Code', 'Instrument Name']].drop_duplicates()
        
        for _, row in instruments.iterrows():
            code = str(row['Instrument Code']).zfill(5)
            name = row.get('Instrument Name', f'Instrument_{code}')
            
            node = SFCNode(
                id=f"INST_{code}",
                node_type=NodeType.INSTRUMENT,
                metadata={
                    'code': code,
                    'name': name,
                    'instrument_type': self._classify_instrument(code)
                }
            )
            self.graph.add_node(node)
        
        logger.info(f"Added {len(instruments)} instrument nodes")
    
    def _add_bilateral_positions(self, fwtw_df: pd.DataFrame) -> None:
        """Add bilateral position nodes and edges."""
        for _, row in fwtw_df.iterrows():
            holder = str(row['Holder Code']).zfill(2)
            issuer = str(row['Issuer Code']).zfill(2)
            instrument = str(row['Instrument Code']).zfill(5)
            date = row['Date']
            level = float(row['Level'])
            
            # Create bilateral position node
            bilateral_id = f"BILATERAL_{holder}_{issuer}_{instrument}_{date}"
            
            node = SFCNode(
                id=bilateral_id,
                node_type=NodeType.BILATERAL,
                metadata={
                    'holder': holder,
                    'issuer': issuer,
                    'instrument': instrument,
                    'date': date,
                    'level': level,
                    'holder_name': row.get('Holder Name', ''),
                    'issuer_name': row.get('Issuer Name', ''),
                    'instrument_name': row.get('Instrument Name', '')
                }
            )
            self.graph.add_node(node)
            
            # Store for quick lookup
            self.bilateral_positions[(holder, issuer, instrument, date)] = level
            
            # Create edges
            # Holder holds this position
            self.graph.add_edge(SFCEdge(
                source=f"SECTOR_{holder}",
                target=bilateral_id,
                edge_type=EdgeType.HOLDS,
                weight=level,
                metadata={'instrument': instrument, 'date': date}
            ))
            
            # Issuer issued this position
            self.graph.add_edge(SFCEdge(
                source=f"SECTOR_{issuer}",
                target=bilateral_id,
                edge_type=EdgeType.ISSUES,
                weight=-level,  # Negative for liability
                metadata={'instrument': instrument, 'date': date}
            ))
            
            # Link to instrument
            self.graph.add_edge(SFCEdge(
                source=bilateral_id,
                target=f"INST_{instrument}",
                edge_type=EdgeType.REPRESENTS,
                weight=level,
                metadata={'date': date}
            ))
        
        self.stats['n_bilateral_positions'] = len(self.bilateral_positions)
        logger.info(f"Added {len(self.bilateral_positions)} bilateral positions")
    
    def add_z1_series(self, series_codes: List[str]) -> None:
        """
        Add Z1 series nodes and map to bilateral positions.
        
        Parameters:
        -----------
        series_codes : List[str]
            List of Z1 series codes (e.g., 'FL154090005.Q')
        """
        logger.info(f"Adding {len(series_codes)} Z1 series")
        self.stats['n_z1_series'] = len(series_codes)
        
        for code in series_codes:
            parsed = self._parse_z1_code(code)
            if not parsed:
                self.stats['n_unmapped_series'] += 1
                continue
            
            # Store parsed info
            self.z1_series_map[code] = parsed
            
            # Add series node
            node = SFCNode(
                id=f"SERIES_{code}",
                node_type=NodeType.SERIES,
                metadata=parsed
            )
            self.graph.add_node(node)
            
            # Map to bilateral positions
            self._map_series_to_bilateral(code, parsed)
        
        # Identify stock-flow pairs
        self._identify_stock_flow_pairs()
        
        logger.info(f"Mapped {len(self.z1_series_map)} series, "
                   f"found {len(self.stock_flow_pairs)} stock-flow pairs")
    
    def _parse_z1_code(self, code: str) -> Optional[Dict]:
        """Parse Z1 series code into components."""
        # Remove .Q or .A suffix
        base = code[:-2] if code.endswith(('.Q', '.A')) else code
        freq = code[-1] if code.endswith(('.Q', '.A')) else None
        
        # Parse: FL1530641005 -> FL 15 30641 005
        match = re.match(r'^([A-Z]{2})(\d{2})(\d{5})(\d{3})', base)
        if not match:
            return None
        
        return {
            'code': code,
            'prefix': match.group(1),
            'sector': match.group(2),
            'instrument': match.group(3),
            'suffix': match.group(4),
            'frequency': freq,
            'is_level': match.group(1) == 'FL',
            'is_flow': match.group(1) == 'FU',
            'is_reval': match.group(1) == 'FR',
            'is_other': match.group(1) == 'FV'
        }
    
    def _map_series_to_bilateral(self, series_code: str, parsed: Dict) -> None:
        """Map Z1 series to corresponding bilateral positions."""
        sector = parsed['sector']
        instrument = parsed['instrument']
        prefix = parsed['prefix']
        
        mapped_count = 0
        
        # Find matching bilateral positions
        for (h, i, inst, date), level in self.bilateral_positions.items():
            if inst != instrument:
                continue
            
            bilateral_id = f"BILATERAL_{h}_{i}_{inst}_{date}"
            
            # Map based on series type and sector match
            if prefix == 'FL':  # Level series
                if h == sector:  # Holder's asset position
                    self.graph.add_edge(SFCEdge(
                        source=f"SERIES_{series_code}",
                        target=bilateral_id,
                        edge_type=EdgeType.BILATERAL_ASSET,
                        weight=1.0,
                        metadata={'date': date}
                    ))
                    mapped_count += 1
                    
                elif i == sector:  # Issuer's liability position
                    self.graph.add_edge(SFCEdge(
