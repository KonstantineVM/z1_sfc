#!/usr/bin/env python3
"""
PLACEMENT: src/graph/series_parser.py

Parse Z1 series codes and integrate with z1_series_interpreter.
"""

import re
from typing import Dict, Optional
from src.utils.z1_series_interpreter import Z1Series

def parse_z1_code(code: str) -> Dict:
    """
    Parse Z1 series code into components.
    
    Parameters:
    -----------
    code : str
        Z1 series code (e.g., 'FL154090005.Q')
    
    Returns:
    --------
    dict
        Parsed components with keys:
        - prefix: 2-letter prefix (FL, FU, FR, FV, etc.)
        - sector: 2-digit sector code
        - instrument: 5-digit instrument code  
        - suffix: 1-digit calculation type (digit 9)
        - freq: Frequency (Q or A)
        - is_base: True if base series (digit 9 = 0 or 3)
    """
    # Use the existing Z1Series parser
    parsed = Z1Series.parse(code)
    
    if parsed:
        return {
            'prefix': parsed.prefix,
            'sector': parsed.sector,
            'instrument': parsed.instrument,
            'suffix': parsed.calculation,  # This is digit 9
            'freq': parsed.frequency,
            'is_base': parsed.calculation in ['0', '3'],
            'is_computed': parsed.calculation == '5',
            'series_type': parsed.series_type
        }
    
    # Fallback pattern matching if Z1Series fails
    s = str(code)
    freq = s[-1] if s.endswith(('.Q', '.A')) else None
    base = s[:-2] if freq else s
    
    # Z1 pattern: 2 letters + 2 digits (sector) + 5 digits (instrument) + 1 digit (calculation)
    match = re.match(r'^([A-Z]{2})(\d{2})(\d{5})(\d)$', base)
    
    if not match:
        return {
            'prefix': None,
            'sector': None, 
            'instrument': None,
            'suffix': None,
            'freq': freq,
            'is_base': False,
            'is_computed': False,
            'series_type': 'unknown'
        }
    
    prefix, sector, instrument, calculation = match.groups()
    
    return {
        'prefix': prefix,
        'sector': sector,
        'instrument': instrument,
        'suffix': calculation,
        'freq': freq,
        'is_base': calculation in ['0', '3'],
        'is_computed': calculation == '5',
        'series_type': get_series_type(prefix)
    }

def get_series_type(prefix: str) -> str:
    """Get series type from prefix."""
    type_map = {
        'FL': 'level',
        'FU': 'transaction',
        'FR': 'revaluation',
        'FV': 'other_change',
        'FA': 'flow_saar',
        'LA': 'level_sa',
        'FC': 'change',
        'FG': 'growth_rate'
    }
    return type_map.get(prefix, 'unknown')

def identify_base_fl_series(series_list: list) -> list:
    """
    Identify base FL series that should have FU/FR/FV flows.
    
    Parameters:
    -----------
    series_list : list
        List of all series codes
        
    Returns:
    --------
    list
        List of base FL series codes (digit 9 = 0 or 3)
    """
    base_fl = []
    
    for series in series_list:
        parsed = parse_z1_code(series)
        if (parsed and 
            parsed['prefix'] == 'FL' and 
            parsed['is_base']):
            base_fl.append(series)
    
    return base_fl
