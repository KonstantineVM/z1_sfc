# PLACEMENT: src/visualization/fwtw_godley_matrix.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_godley_matrix_with_bilateral(model, t: int):
    """
    Plot Godley matrix showing Z1 aggregates and FWTW bilateral detail.
    """
    matrix = model.build_godley_matrix(t)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Z1 Aggregates Matrix
    ax = axes[0, 0]
    z1_matrix = pd.DataFrame()
    for sector in model.sectors:
        sector_data = []
        for inst in model.instruments:
            z1_series = f"FL{sector}{inst}5.Q"
            if z1_series in model.combined_data.columns:
                value = model.combined_data.iloc[t][z1_series]
            else:
                value = 0
            sector_data.append(value)
        z1_matrix[sector] = sector_data
    
    sns.heatmap(z1_matrix, annot=True, fmt='.0f', 
                cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Z1 Aggregate Positions')
    ax.set_ylabel('Instruments')
    ax.set_xlabel('Sectors')
    
    # Plot 2: FWTW Bilateral Network
    ax = axes[0, 1]
    # Create network visualization of bilateral positions
    # (Simplified - would use networkx for full implementation)
    bilateral_flows = []
    for col in model.combined_data.columns:
        if col.startswith('FB'):
            holder = col[2:4]
            issuer = col[4:6]
            value = model.combined_data.iloc[t][col]
            if not pd.isna(value) and abs(value) > 100:  # Filter small positions
                bilateral_flows.append({
                    'from': issuer,
                    'to': holder,
                    'value': value
                })
    
    # Plot as heatmap of bilateral flows
    flow_matrix = pd.DataFrame(0, 
                              index=model.sectors,
                              columns=model.sectors)
    for flow in bilateral_flows:
        if flow['from'] in model.sectors and flow['to'] in model.sectors:
            flow_matrix.loc[flow['from'], flow['to']] += flow['value']
    
    sns.heatmap(flow_matrix, annot=True, fmt='.0f',
                cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('FWTW Bilateral Flows')
    
    # Plot 3: Z1 vs FWTW Discrepancy
    ax = axes[1, 0]
    discrepancies = []
    labels = []
    
    for sector in model.sectors[:4]:  # Limit to first 4 sectors
        for inst in list(model.instruments.keys())[:3]:  # First 3 instruments
            z1_series = f"FL{sector}{inst}5.Q"
            if z1_series in model.combined_data.columns:
                z1_val = model.combined_data.iloc[t][z1_series]
                
                # Sum FWTW bilaterals
                fwtw_sum = 0
                for col in model.combined_data.columns:
                    if col.startswith('FB') and inst in col:
                        if sector in col[2:6]:  # Holder or issuer
                            fwtw_sum += model.combined_data.iloc[t][col]
                
                discrepancies.append(z1_val - fwtw_sum)
                labels.append(f"{sector}-{inst}")
    
    ax.bar(labels, discrepancies)
    ax.set_title('Z1 - FWTW Discrepancies by Sector-Instrument')
    ax.set_ylabel('Discrepancy (Billions $)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Godley Identity Check
    ax = axes[1, 1]
    for i, sector in enumerate(model.sectors[:4]):
        # Get income, expenditure, financial changes
        income = model.combined_data[f"FA{sector}6010001.Q"].iloc[t] if f"FA{sector}6010001.Q" in model.combined_data.columns else 0
        expenditure = model.combined_data[f"FA{sector}6900005.Q"].iloc[t] if f"FA{sector}6900005.Q" in model.combined_data.columns else 0
        net_lending = model.combined_data[f"FA{sector}5000005.Q"].iloc[t] if f"FA{sector}5000005.Q" in model.combined_data.columns else 0
        discrepancy = model.combined_data[f"FA{sector}7005005.Q"].iloc[t] if f"FA{sector}7005005.Q" in model.combined_data.columns else 0
        
        saving = income - expenditure
        identity_check = saving - net_lending - discrepancy
        
        x = [f"S{sector}", f"NL{sector}", f"D{sector}", f"Check{sector}"]
        y = [saving, net_lending, discrepancy, identity_check]
        
        ax.bar([xi + i*0.2 for xi in range(len(x))], y, width=0.2, label=f"Sector {sector}")
    
    ax.set_title('Godley Identity Components')
    ax.set_ylabel('Billions $')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Saving', 'Net Lending', 'Discrepancy', 'Identity Check'])
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig
