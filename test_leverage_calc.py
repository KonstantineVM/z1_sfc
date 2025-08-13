#!/usr/bin/env python3
"""
Test leverage calculation to debug why it's returning 0
"""

import pandas as pd
import json
import yaml
import numpy as np
from pathlib import Path

def test_leverage_calculation():
    """Test the leverage calculation step by step."""
    
    print("="*70)
    print("TESTING LEVERAGE CALCULATION")
    print("="*70)
    
    # Load balance sheet
    bs_file = "outputs/sfc_balance_sheet_2025-03-31.csv"
    bs = pd.read_csv(bs_file, index_col=0)
    print(f"\n1. Loaded balance sheet: {bs.shape}")
    print(f"   Index dtype: {bs.index.dtype}")
    print(f"   Sample indices: {bs.index[:5].tolist()}")
    
    # Load instrument map
    with open("mappings/instrument_map.json", 'r') as f:
        imap = json.load(f)
    
    # Load roles map
    with open("mappings/instrument_roles.yaml", 'r') as f:
        class_roles = yaml.safe_load(f) or {}
    
    # Build role sets (same logic as in analyzer)
    print("\n2. Building role sets...")
    roles_by_instr = {}
    
    for code, meta in imap.items():
        roles = set()
        
        # Direct roles
        r = meta.get('roles')
        if isinstance(r, list):
            roles.update(str(x).lower() for x in r)
        
        # Class-based roles
        klass = str(meta.get('class', '')).lower()
        if klass and klass in class_roles:
            class_role_list = class_roles[klass]
            if isinstance(class_role_list, list):
                roles.update(str(x).lower() for x in class_role_list)
        
        # Heuristic roles
        if not roles:
            side = str(meta.get('side', '')).lower()
            label = str(meta.get('label', '')).lower()
            
            if side == 'liability' or any(k in klass for k in ['debt', 'bond', 'note', 'security', 'loan', 'mortgage', 'paper']):
                roles.add('debt')
            
            if any(k in klass for k in ['loan', 'mortgage', 'consumer', 'credit']):
                roles.add('loan')
        
        roles_by_instr[code] = roles
    
    debt_instruments = {k for k, r in roles_by_instr.items() if 'debt' in r}
    print(f"   Debt instruments identified: {len(debt_instruments)}")
    print(f"   Sample debt instruments: {list(debt_instruments)[:5]}")
    
    # Get sectors
    sectors = [c for c in bs.columns if c not in ['label', 'Total']]
    print(f"\n3. Sectors in balance sheet: {sectors}")
    
    # Test debt calculation
    print("\n4. Testing debt calculation...")
    
    # Method 1: String comparison (what diagnostic shows works)
    bs_index_str = set(bs.index.astype(str))
    debt_in_bs = debt_instruments & bs_index_str
    print(f"   Debt instruments in BS (string match): {len(debt_in_bs)}")
    
    if debt_in_bs:
        # Get indices that match
        debt_indices = [x for x in bs.index if str(x) in debt_in_bs]
        print(f"   Debt indices found: {len(debt_indices)}")
        print(f"   Sample debt indices: {debt_indices[:5]}")
        
        # Calculate debt
        debt_values = bs.loc[debt_indices, sectors].abs().sum(axis=1)
        print(f"   Sample debt values by instrument:")
        for i, (idx, val) in enumerate(debt_values.head().items()):
            print(f"     {idx}: {val:,.0f}")
        
        total_debt = bs.loc[debt_indices, sectors].abs().sum().sum()
        print(f"   Total debt: {total_debt:,.0f}")
    
    # Method 2: Direct index.isin() (what analyzer might be doing)
    print("\n5. Testing index.isin() method...")
    
    # Try different approaches
    debt_mask_str = bs.index.astype(str).isin(debt_instruments)
    print(f"   Mask with str conversion: {debt_mask_str.sum()} matches")
    
    debt_mask_direct = bs.index.isin(debt_instruments)
    print(f"   Direct mask (no conversion): {debt_mask_direct.sum()} matches")
    
    # Try with integer conversion
    debt_instruments_int = set()
    for inst in debt_instruments:
        try:
            debt_instruments_int.add(int(inst))
        except:
            pass
    
    debt_mask_int = bs.index.isin(debt_instruments_int)
    print(f"   Mask with int instruments: {debt_mask_int.sum()} matches")
    
    # Calculate leverage
    print("\n6. Calculating leverage...")
    
    sector_groups = {
        'household': ['15'],
        'corporate': ['10', '11'],
        'financial': ['70', '71', '66', '40', '41', '42'],
        'government': ['20', '21', '22']
    }
    
    for group_name, group_sectors in sector_groups.items():
        group_cols = [s for s in group_sectors if s in sectors]
        if group_cols:
            print(f"\n   {group_name.capitalize()} sectors: {group_cols}")
            
            if debt_mask_str.any():
                group_debt = bs.loc[debt_mask_str, group_cols].abs().sum().sum()
            else:
                group_debt = 0
            
            group_assets = bs[group_cols].abs().sum().sum()
            
            if group_assets > 0:
                leverage = group_debt / group_assets
                print(f"     Debt: {group_debt:,.0f}")
                print(f"     Assets: {group_assets:,.0f}")
                print(f"     Leverage: {leverage:.4f}")
            else:
                print(f"     No assets in this group")
    
    # System-wide
    print("\n   System-wide:")
    if debt_mask_str.any():
        total_debt = bs.loc[debt_mask_str, sectors].abs().sum().sum()
    else:
        total_debt = 0
    total_assets = bs[sectors].abs().sum().sum()
    
    if total_assets > 0:
        system_leverage = total_debt / total_assets
        print(f"     Total debt: {total_debt:,.0f}")
        print(f"     Total assets: {total_assets:,.0f}")
        print(f"     System leverage: {system_leverage:.4f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_leverage_calculation()
