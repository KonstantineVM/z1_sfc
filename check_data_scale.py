#!/usr/bin/env python3
"""
Check the scale and units of the Z1 data
"""

import pandas as pd
import numpy as np

def check_data_scale():
    """Check the scale of values in the matrices."""
    
    print("="*70)
    print("CHECKING DATA SCALE AND UNITS")
    print("="*70)
    
    # Load a balance sheet and transaction matrix
    bs = pd.read_csv("outputs/sfc_balance_sheet_2025-03-31.csv", index_col=0)
    tf = pd.read_csv("outputs/sfc_transactions_2025-03-31.csv", index_col=0)
    
    # Get sectors (excluding label and Total)
    sectors = [c for c in bs.columns if c not in ['label', 'Total']]
    
    print("\n1. BALANCE SHEET SCALE")
    print("-"*40)
    
    # Total assets across all sectors
    total_assets = bs[sectors].abs().sum().sum()
    print(f"Total assets (absolute sum): ${total_assets:,.0f}")
    print(f"  In billions: ${total_assets/1e9:,.1f}B")
    print(f"  In trillions: ${total_assets/1e12:,.2f}T")
    
    # Sample some major items
    print("\nSample balance sheet items:")
    major_items = bs[bs['Total'].abs() > 1e6].head(10)
    for idx in major_items.index:
        value = bs.loc[idx, 'Total']
        label = bs.loc[idx, 'label'] if 'label' in bs.columns else ''
        print(f"  {idx}: {label[:40]:<40} ${value:>15,.0f}")
    
    print("\n2. TRANSACTION FLOW SCALE")
    print("-"*40)
    
    # Total flows
    total_flows = tf[sectors].abs().sum().sum()
    print(f"Total flows (absolute sum): ${total_flows:,.0f}")
    print(f"  In billions: ${total_flows/1e9:,.1f}B")
    print(f"  In millions: ${total_flows/1e6:,.1f}M")
    
    # Sample some major flows
    print("\nSample transaction items:")
    major_flows = tf[tf['Total'].abs() > 1e3].head(10)
    for idx in major_flows.index:
        value = tf.loc[idx, 'Total']
        label = tf.loc[idx, 'label'] if 'label' in tf.columns else ''
        print(f"  {idx}: {label[:40]:<40} ${value:>15,.0f}")
    
    print("\n3. CHECKING LOAN INSTRUMENTS")
    print("-"*40)
    
    # Load instrument map to identify loan instruments
    import json
    import yaml
    
    with open("mappings/instrument_map.json", 'r') as f:
        imap = json.load(f)
    
    with open("mappings/instrument_roles.yaml", 'r') as f:
        class_roles = yaml.safe_load(f) or {}
    
    # Identify loan instruments
    loan_instruments = set()
    for code, meta in imap.items():
        roles = set()
        
        # Check for loan role
        r = meta.get('roles')
        if isinstance(r, list):
            roles.update(str(x).lower() for x in r)
        
        klass = str(meta.get('class', '')).lower()
        if klass and klass in class_roles:
            class_role_list = class_roles[klass]
            if isinstance(class_role_list, list):
                roles.update(str(x).lower() for x in class_role_list)
        
        if not roles:
            if any(k in klass for k in ['loan', 'mortgage', 'consumer', 'credit']):
                roles.add('loan')
        
        if 'loan' in roles:
            loan_instruments.add(code)
    
    print(f"Number of loan instruments: {len(loan_instruments)}")
    
    # Check loan values in transactions
    loan_flows = []
    for inst in loan_instruments:
        try:
            inst_int = int(inst)
            if inst_int in tf.index:
                flow_value = tf.loc[inst_int, sectors].clip(lower=0).sum()
                if flow_value > 0:
                    loan_flows.append({
                        'instrument': inst_int,
                        'label': tf.loc[inst_int, 'label'] if 'label' in tf.columns else '',
                        'flow': flow_value
                    })
        except:
            pass
    
    loan_flows = sorted(loan_flows, key=lambda x: -x['flow'])
    
    print(f"\nTop loan flows (positive only):")
    total_loan_flow = sum(x['flow'] for x in loan_flows)
    print(f"Total credit flow: ${total_loan_flow:,.0f}")
    print(f"  In billions: ${total_loan_flow/1e9:,.3f}B")
    print(f"  In millions: ${total_loan_flow/1e6:,.1f}M")
    
    print("\nTop 10 loan instruments by flow:")
    for item in loan_flows[:10]:
        print(f"  {item['instrument']}: {item['label'][:35]:<35} ${item['flow']:>12,.0f}")
    
    print("\n4. DATA UNIT ANALYSIS")
    print("-"*40)
    
    # Check if values might be in different units
    max_bs_value = bs[sectors].abs().max().max()
    min_nonzero_bs = bs[sectors].replace(0, np.nan).abs().min().min()
    
    print(f"Balance sheet value range:")
    print(f"  Maximum: ${max_bs_value:,.0f}")
    print(f"  Minimum non-zero: ${min_nonzero_bs:,.6f}")
    
    if max_bs_value < 1e6:
        print("  ⚠️ Values seem too small - might be in millions or billions?")
    elif max_bs_value > 1e15:
        print("  ⚠️ Values seem very large - might be in smaller units?")
    else:
        print("  ✓ Values appear to be in reasonable range")
    
    # Check Federal Reserve Z.1 documentation
    print("\n5. Z.1 DATA UNITS (from Federal Reserve documentation)")
    print("-"*40)
    print("Federal Reserve Z.1 data is typically reported in:")
    print("  - Millions of dollars for flow data")
    print("  - Millions of dollars for level (stock) data")
    print("  - Seasonally adjusted annual rates (SAAR) for flows")
    print("\nIf data is in millions, then:")
    print(f"  Total assets: ${total_assets:,.0f}M = ${total_assets/1e3:,.1f}B")
    print(f"  Credit flow: ${total_loan_flow:,.0f}M = ${total_loan_flow/1e3:,.1f}B")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    check_data_scale()
