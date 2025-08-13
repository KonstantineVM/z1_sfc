#!/usr/bin/env python3
"""
SFC Diagnostic Script - Check Role Mappings and Data
=====================================================
Diagnoses why leverage and other indicators are zero.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
import sys

def diagnose_role_mappings(
    balance_sheet_file: str,
    instrument_map_file: str = "mappings/instruments.json",
    roles_map_file: str = "mappings/instrument_roles.yaml"
):
    """Diagnose role mapping issues."""
    
    print("="*70)
    print("SFC ROLE MAPPING DIAGNOSTIC")
    print("="*70)
    
    # Load balance sheet to see actual instruments
    print("\n1. LOADING BALANCE SHEET DATA")
    print("-"*40)
    bs = pd.read_csv(balance_sheet_file, index_col=0)
    bs_instruments = set(bs.index.astype(str))
    print(f"Balance sheet file: {balance_sheet_file}")
    print(f"Number of instruments in data: {len(bs_instruments)}")
    print(f"Sample instruments: {list(bs_instruments)[:10]}")
    
    # Load instrument map
    print("\n2. LOADING INSTRUMENT MAP")
    print("-"*40)
    with open(instrument_map_file, 'r') as f:
        imap = json.load(f)
    imap_instruments = set(imap.keys())
    print(f"Instrument map file: {instrument_map_file}")
    print(f"Number of instruments in map: {len(imap_instruments)}")
    print(f"Sample instruments: {list(imap_instruments)[:10]}")
    
    # Check instrument format consistency
    print("\n3. CHECKING INSTRUMENT FORMAT")
    print("-"*40)
    
    # Check if instruments match between data and map
    matching = bs_instruments & imap_instruments
    data_only = bs_instruments - imap_instruments
    map_only = imap_instruments - bs_instruments
    
    print(f"Matching instruments: {len(matching)}")
    print(f"In data but not in map: {len(data_only)}")
    print(f"In map but not in data: {len(map_only)}")
    
    if data_only:
        print(f"  Sample data-only: {list(data_only)[:5]}")
    if map_only:
        print(f"  Sample map-only: {list(map_only)[:5]}")
    
    # Load roles map
    print("\n4. LOADING ROLES MAP")
    print("-"*40)
    class_roles = {}
    if Path(roles_map_file).exists():
        with open(roles_map_file, 'r') as f:
            class_roles = yaml.safe_load(f) or {}
        print(f"Roles map file: {roles_map_file}")
        print(f"Number of classes with roles: {len(class_roles)}")
        print(f"Classes: {list(class_roles.keys())}")
    else:
        print(f"⚠️ Roles map not found: {roles_map_file}")
    
    # Analyze role assignments
    print("\n5. ANALYZING ROLE ASSIGNMENTS")
    print("-"*40)
    
    roles_by_instr = {}
    role_counts = {}
    
    for code in matching:  # Only check instruments that exist in both
        meta = imap.get(code, {})
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
        
        # Heuristic roles (if no explicit roles)
        if not roles:
            side = str(meta.get('side', '')).lower()
            label = str(meta.get('label', '')).lower()
            
            if side == 'liability' or any(k in klass for k in ['debt', 'bond', 'note', 'security', 'loan', 'mortgage', 'paper']):
                roles.add('debt')
            
            if any(k in klass for k in ['loan', 'mortgage', 'consumer', 'credit']):
                roles.add('loan')
            
            if (str(meta.get('liquidity', '')).lower() == 'high' or 
                any(k in klass for k in ['currency', 'deposit', 'mmf', 'reserves', 'repo', 'cash']) or
                any(k in label for k in ['currency', 'deposit', 'cash'])):
                roles.add('liquid')
            
            if any(k in klass for k in ['payable', 'short-term', 'cp', 'repo-liab', 'taxes']):
                roles.add('st_liability')
            
            if any(k in klass for k in ['equity', 'stock', 'share']):
                roles.add('equity')
            
            if any(k in klass for k in ['derivative', 'option', 'future', 'swap']):
                roles.add('derivative')
        
        roles_by_instr[code] = roles
        
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    
    print("Role distribution:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count} instruments")
    
    # Check specific role sets
    print("\n6. CHECKING KEY ROLE SETS")
    print("-"*40)
    
    debt_instruments = {k for k, r in roles_by_instr.items() if 'debt' in r}
    loan_instruments = {k for k, r in roles_by_instr.items() if 'loan' in r}
    liquid_instruments = {k for k, r in roles_by_instr.items() if 'liquid' in r}
    equity_instruments = {k for k, r in roles_by_instr.items() if 'equity' in r}
    
    print(f"Debt instruments: {len(debt_instruments)}")
    print(f"Loan instruments: {len(loan_instruments)}")
    print(f"Liquid instruments: {len(liquid_instruments)}")
    print(f"Equity instruments: {len(equity_instruments)}")
    
    # Check if these instruments have non-zero values in balance sheet
    print("\n7. CHECKING VALUES IN BALANCE SHEET")
    print("-"*40)
    
    sectors = [c for c in bs.columns if c not in ['label', 'Total']]
    
    # Convert balance sheet index to strings for comparison
    bs_index_str = set(bs.index.astype(str))
    
    if debt_instruments:
        debt_in_bs = debt_instruments & bs_index_str
        if debt_in_bs:
            # Convert back to the original index type for lookup
            debt_indices = [x for x in bs.index if str(x) in debt_in_bs]
            debt_values = bs.loc[debt_indices, sectors].abs().sum().sum()
            print(f"Debt instruments in BS: {len(debt_in_bs)}, Total value: {debt_values:,.0f}")
            sample_debt = debt_indices[:3]
            for inst in sample_debt:
                print(f"  {inst}: {bs.loc[inst, 'label'] if 'label' in bs.columns else 'N/A'}")
        else:
            print("⚠️ No debt instruments found in balance sheet!")
            print(f"   Sample debt instruments: {list(debt_instruments)[:5]}")
            print(f"   Sample BS instruments: {list(bs_index_str)[:5]}")
    else:
        print("⚠️ No debt instruments identified!")
    
    if liquid_instruments:
        liquid_in_bs = liquid_instruments & bs_index_str
        if liquid_in_bs:
            liquid_indices = [x for x in bs.index if str(x) in liquid_in_bs]
            liquid_values = bs.loc[liquid_indices, sectors].abs().sum().sum()
            print(f"Liquid instruments in BS: {len(liquid_in_bs)}, Total value: {liquid_values:,.0f}")
            sample_liquid = liquid_indices[:3]
            for inst in sample_liquid:
                print(f"  {inst}: {bs.loc[inst, 'label'] if 'label' in bs.columns else 'N/A'}")
        else:
            print("⚠️ No liquid instruments found in balance sheet!")
    else:
        print("⚠️ No liquid instruments identified!")
    
    if loan_instruments:
        loan_in_bs = loan_instruments & bs_index_str
        if loan_in_bs:
            loan_indices = [x for x in bs.index if str(x) in loan_in_bs]
            loan_values = bs.loc[loan_indices, sectors].abs().sum().sum()
            print(f"Loan instruments in BS: {len(loan_in_bs)}, Total value: {loan_values:,.0f}")
        else:
            print("⚠️ No loan instruments found in balance sheet!")
    
    if equity_instruments:
        equity_in_bs = equity_instruments & bs_index_str
        if equity_in_bs:
            equity_indices = [x for x in bs.index if str(x) in equity_in_bs]
            equity_values = bs.loc[equity_indices, sectors].abs().sum().sum()
            print(f"Equity instruments in BS: {len(equity_in_bs)}, Total value: {equity_values:,.0f}")
        else:
            print("⚠️ No equity instruments found in balance sheet!")
    
    # Sample instrument analysis
    print("\n8. SAMPLE INSTRUMENT ANALYSIS")
    print("-"*40)
    
    # Pick a few instruments from balance sheet with non-zero values
    non_zero_instruments = []
    for inst in bs.index[:50]:  # Check first 50
        if bs.loc[inst, sectors].abs().sum() > 0:
            non_zero_instruments.append(inst)
        if len(non_zero_instruments) >= 5:
            break
    
    print("Sample instruments with values:")
    for inst in non_zero_instruments:
        inst_str = str(inst)
        meta = imap.get(inst_str, {})
        roles = roles_by_instr.get(inst_str, set())
        print(f"\n  Instrument: {inst_str}")
        print(f"    Label: {bs.loc[inst, 'label'] if 'label' in bs.columns else 'N/A'}")
        print(f"    Class: {meta.get('class', 'N/A')}")
        print(f"    Side: {meta.get('side', 'N/A')}")
        print(f"    Roles: {roles if roles else 'NONE'}")
        print(f"    Total value: {bs.loc[inst, sectors].abs().sum():,.0f}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if len(matching) < len(bs_instruments) * 0.5:
        print("⚠️ Less than 50% of instruments match between data and map!")
        print("   Check instrument code format (padding, type conversion)")
    
    if not debt_instruments:
        print("⚠️ No debt instruments identified!")
        print("   Check role mappings and instrument classes")
    
    if debt_instruments and not (debt_instruments & set(bs.index.astype(str))):
        print("⚠️ Debt instruments don't match balance sheet indices!")
        print("   Check index format consistency")
    
    return {
        'bs_instruments': bs_instruments,
        'imap_instruments': imap_instruments,
        'matching': matching,
        'roles_by_instr': roles_by_instr,
        'role_counts': role_counts
    }


if __name__ == "__main__":
    # Use most recent balance sheet file
    import glob
    
    bs_files = sorted(glob.glob("outputs/sfc_balance_sheet_*.csv"))
    if not bs_files:
        print("No balance sheet files found in outputs/")
        sys.exit(1)
    
    latest_bs = bs_files[-1]
    print(f"Using latest balance sheet: {latest_bs}")
    
    # Check for different possible paths for instrument map
    possible_instrument_maps = [
        "mappings/instrument_map.json",
        "mappings/instruments.json",
        "config/instrument_map.json",
        "data/instrument_map.json"
    ]
    
    instrument_map_file = None
    for path in possible_instrument_maps:
        if Path(path).exists():
            instrument_map_file = path
            break
    
    if not instrument_map_file:
        print("\nCould not find instrument map file. Trying to load from config...")
        # Try to load from config file
        config_files = ["config/proper_sfc_config.yaml", "config/sfc_config.yaml"]
        for cf in config_files:
            if Path(cf).exists():
                with open(cf, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'sfc' in config and 'instrument_map' in config['sfc']:
                        instrument_map_file = config['sfc']['instrument_map']
                        print(f"Found instrument map path in config: {instrument_map_file}")
                        break
    
    if not instrument_map_file or not Path(instrument_map_file).exists():
        print("ERROR: Could not find instrument map file!")
        print("Please specify the correct path to your instrument_map.json file")
        sys.exit(1)
    
    # Run diagnostic
    results = diagnose_role_mappings(
        balance_sheet_file=latest_bs,
        instrument_map_file=instrument_map_file,
        roles_map_file="mappings/instrument_roles.yaml"
    )
