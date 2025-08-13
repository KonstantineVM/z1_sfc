#!/usr/bin/env python3
"""
Corrected Sector Analysis - Real Economic Sectors Only
========================================================
Excludes aggregate sectors and focuses on actual economic entities.
Properly calculates leverage using correct instrument identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# REAL Economic Sectors (excluding aggregates)
REAL_SECTORS = {
    # Government
    '01': 'Federal Government (Combined)',
    '07': 'State and Local Governments',
    '20': 'Federal Government (Defense)',
    '21': 'Federal Government (Nondefense)',
    '22': 'State and Local Governments (General)',
    
    # Households and Business
    '10': 'Nonfinancial Corporate Business',
    '11': 'Nonfinancial Noncorporate Business',
    '15': 'Households and Nonprofit Organizations',
    '16': 'Nonprofit Organizations',
    
    # Banking and Depository
    '14': 'Private Depository Institutions',
    '40': 'Commercial Banking',
    '41': 'Savings Institutions',
    '42': 'Credit Unions',
    '70': 'U.S.-Chartered Depository Institutions',
    '71': 'Foreign Banking Offices in U.S.',
    '26': 'Foreign Banking Offices in U.S.',
    
    # Insurance and Pension
    '31': 'Private Pension Funds',
    '43': 'Property-Casualty Insurance',
    '47': 'Life Insurance Companies',
    '48': 'Private Pension Funds (DB)',
    '49': 'Private Pension Funds (DC)',
    '79': 'Insurance Companies',
    
    # GSEs and Mortgage
    '17': 'Government-Sponsored Enterprises',
    '18': 'Agency- and GSE-backed Mortgage Pools',
    '50': 'Government-Sponsored Enterprises (GSEs)',
    '51': 'Agency- and GSE-backed Mortgage Pools',
    
    # Shadow Banking
    '52': 'Issuers of Asset-Backed Securities',
    '53': 'Finance Companies',
    '54': 'Real Estate Investment Trusts (REITs)',
    '55': 'Security Brokers and Dealers',
    '57': 'Holding Companies',
    '58': 'Funding Corporations',
    
    # Funds
    '59': 'Private Equity Funds',
    '60': 'Hedge Funds',
    '61': 'Money Market Funds',
    '65': 'Mutual Funds',
    '66': 'Exchange-Traded Funds',
    
    # Foreign
    '08': 'Foreign Sector',
    '13': 'Rest of the World',
    
    # Other Financial
    '09': 'Monetary Authority (Fed)',
    '69': 'Other Financial Business',
    '77': 'Brokers and Dealers',
}

# Aggregate sectors to EXCLUDE
AGGREGATE_SECTORS = ['86', '87', '88', '89', '90', '91']


class CorrectedSectorAnalyzer:
    """
    Analyzes real economic sectors with proper leverage calculations.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.output_dir = Path("outputs/corrected_sector_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find most recent files
        bs_files = sorted(Path("outputs").glob("sfc_balance_sheet_*.csv"))
        tf_files = sorted(Path("outputs").glob("sfc_transactions_*.csv"))
        
        if not bs_files or not tf_files:
            raise FileNotFoundError("No SFC data files found")
        
        self.bs_file = str(bs_files[-1])
        self.tf_file = str(tf_files[-1])
        self.date = Path(self.bs_file).stem.replace('sfc_balance_sheet_', '')
        
        print(f"Analyzing data from: {self.date}")
        
        # Load data
        self.load_and_clean_data()
        
    def load_and_clean_data(self):
        """Load data and filter to real sectors only."""
        # Load raw data
        self.bs_raw = pd.read_csv(self.bs_file, index_col=0)
        self.tf_raw = pd.read_csv(self.tf_file, index_col=0)
        
        # Filter to real sectors only
        real_sector_codes = list(REAL_SECTORS.keys())
        available_real_sectors = [s for s in self.bs_raw.columns 
                                 if s in real_sector_codes]
        
        # Create filtered dataframes
        self.bs = self.bs_raw[['label'] + available_real_sectors].copy() if 'label' in self.bs_raw.columns else self.bs_raw[available_real_sectors].copy()
        self.tf = self.tf_raw[['label'] + available_real_sectors].copy() if 'label' in self.tf_raw.columns else self.tf_raw[available_real_sectors].copy()
        
        self.sectors = available_real_sectors
        
        print(f"Found {len(self.sectors)} real economic sectors")
        
    def analyze_real_sector_sizes(self):
        """Analyze sizes of real economic sectors."""
        print("\n" + "="*70)
        print("REAL SECTOR SIZE ANALYSIS")
        print("="*70)
        
        sector_data = []
        
        for sector in self.sectors:
            # Assets (negative values)
            assets = self.bs[self.bs[sector] < 0][sector].abs().sum()
            
            # Liabilities (positive values)
            liabilities = self.bs[self.bs[sector] > 0][sector].abs().sum()
            
            # Total size
            total_size = assets + liabilities
            
            # Net position
            net_position = self.bs[sector].sum()
            
            # Transaction volume
            transaction_volume = self.tf[sector].abs().sum() if sector in self.tf.columns else 0
            
            sector_data.append({
                'sector_code': sector,
                'sector_name': REAL_SECTORS.get(sector, f'Sector {sector}'),
                'total_assets': assets,
                'total_liabilities': liabilities,
                'total_size': total_size,
                'net_position': net_position,
                'transaction_volume': transaction_volume,
                'asset_liability_ratio': assets / liabilities if liabilities > 0 else 0
            })
        
        self.sector_df = pd.DataFrame(sector_data)
        self.sector_df = self.sector_df.sort_values('total_size', ascending=False)
        
        # Calculate percentage of real economy
        total_real_economy = self.sector_df['total_size'].sum()
        self.sector_df['pct_of_real_economy'] = (self.sector_df['total_size'] / total_real_economy * 100)
        
        # Save
        self.sector_df.to_csv(self.output_dir / f'real_sector_sizes_{self.date}.csv', index=False)
        
        # Print top 10 real sectors
        print("\nTop 10 Real Economic Sectors (in millions):")
        print("-"*70)
        
        for idx, row in self.sector_df.head(10).iterrows():
            print(f"\n{row['sector_code']}: {row['sector_name']}")
            print(f"  Total Size: ${row['total_size']:,.0f}M ({row['pct_of_real_economy']:.1f}% of real economy)")
            print(f"  Assets: ${row['total_assets']:,.0f}M")
            print(f"  Liabilities: ${row['total_liabilities']:,.0f}M")
            print(f"  Net Position: ${row['net_position']:,.0f}M")
            print(f"  Transaction Volume: ${row['transaction_volume']:,.0f}M")
        
        return self.sector_df
    
    def calculate_proper_leverage(self):
        """Calculate leverage using proper instrument identification."""
        print("\n" + "="*70)
        print("PROPER LEVERAGE CALCULATION")
        print("="*70)
        
        # Load instrument map
        try:
            with open("mappings/instrument_map.json", 'r') as f:
                imap = json.load(f)
        except:
            print("Warning: Could not load instrument map")
            imap = {}
        
        # More comprehensive debt identification
        debt_keywords = ['debt', 'bond', 'loan', 'mortgage', 'credit', 'paper', 
                        'note', 'security', 'liability', 'payable', 'borrowing']
        
        debt_instruments = set()
        for code, meta in imap.items():
            # Check class
            inst_class = str(meta.get('class', '')).lower()
            if any(keyword in inst_class for keyword in debt_keywords):
                debt_instruments.add(code)
            
            # Check label
            label = str(meta.get('label', '')).lower()
            if any(keyword in label for keyword in debt_keywords):
                debt_instruments.add(code)
            
            # Check side (liabilities are often debt)
            if meta.get('side') == 'liability':
                debt_instruments.add(code)
        
        print(f"Identified {len(debt_instruments)} debt instruments")
        
        # Calculate leverage for each real sector
        leverage_data = []
        
        for _, row in self.sector_df.iterrows():
            sector = row['sector_code']
            
            # Calculate debt for this sector
            sector_debt = 0
            debt_count = 0
            
            for inst_str in debt_instruments:
                try:
                    inst = int(inst_str)
                    if inst in self.bs.index and sector in self.bs.columns:
                        value = abs(self.bs.loc[inst, sector])
                        if value > 0:
                            sector_debt += value
                            debt_count += 1
                except:
                    continue
            
            # Use total liabilities as denominator (more reliable than assets)
            total_base = row['total_liabilities'] if row['total_liabilities'] > 0 else row['total_assets']
            
            # Calculate leverage ratio
            leverage = sector_debt / total_base if total_base > 0 else 0
            
            leverage_data.append({
                'sector_code': sector,
                'sector_name': row['sector_name'],
                'total_debt': sector_debt,
                'total_base': total_base,
                'leverage_ratio': leverage,
                'debt_instruments_count': debt_count,
                'size_pct': row['pct_of_real_economy']
            })
        
        self.leverage_df = pd.DataFrame(leverage_data)
        self.leverage_df = self.leverage_df.sort_values('leverage_ratio', ascending=False)
        
        print("\nLeverage Analysis for Real Sectors:")
        print("-"*70)
        
        for _, row in self.leverage_df.head(10).iterrows():
            if row['leverage_ratio'] > 0.5:
                risk = "🔴 CRITICAL"
            elif row['leverage_ratio'] > 0.3:
                risk = "🟠 HIGH"
            elif row['leverage_ratio'] > 0.2:
                risk = "🟡 MODERATE"
            else:
                risk = "🟢 LOW"
            
            print(f"\n{row['sector_code']}: {row['sector_name']} ({row['size_pct']:.1f}% of economy)")
            print(f"  Leverage Ratio: {row['leverage_ratio']:.1%} {risk}")
            print(f"  Total Debt: ${row['total_debt']:,.0f}M")
            print(f"  Base (Liabilities): ${row['total_base']:,.0f}M")
            print(f"  Debt Instruments: {row['debt_instruments_count']}")
        
        return self.leverage_df
    
    def identify_systemic_sectors(self):
        """Identify systemically important sectors."""
        print("\n" + "="*70)
        print("SYSTEMIC IMPORTANCE ANALYSIS")
        print("="*70)
        
        systemic_data = []
        
        for _, row in self.sector_df.iterrows():
            sector = row['sector_code']
            
            # Get leverage
            leverage = self.leverage_df[self.leverage_df['sector_code'] == sector]['leverage_ratio'].values
            leverage_ratio = leverage[0] if len(leverage) > 0 else 0
            
            # Size score (0-1)
            size_score = min(row['pct_of_real_economy'] / 20, 1)  # Cap at 20%
            
            # Leverage score (0-1)
            leverage_score = min(leverage_ratio * 2, 1)  # Cap at 50% leverage
            
            # Transaction activity score (0-1)
            max_transactions = self.sector_df['transaction_volume'].max()
            activity_score = row['transaction_volume'] / max_transactions if max_transactions > 0 else 0
            
            # Interconnection score (based on sector type)
            interconnection_score = 0
            if sector in ['70', '40', '41', '42', '14', '26', '71']:  # Banks
                interconnection_score = 0.9
            elif sector in ['65', '61', '79', '47', '43']:  # Funds and Insurance
                interconnection_score = 0.7
            elif sector in ['55', '77', '52', '53']:  # Shadow banking
                interconnection_score = 0.8
            elif sector in ['10', '11']:  # Corporate
                interconnection_score = 0.5
            elif sector in ['15']:  # Households
                interconnection_score = 0.3
            else:
                interconnection_score = 0.4
            
            # Composite systemic score
            systemic_score = (
                0.30 * size_score +
                0.30 * leverage_score +
                0.20 * activity_score +
                0.20 * interconnection_score
            )
            
            # Determine risk level
            if systemic_score > 0.6:
                risk_level = 'CRITICAL'
            elif systemic_score > 0.4:
                risk_level = 'HIGH'
            elif systemic_score > 0.25:
                risk_level = 'MODERATE'
            else:
                risk_level = 'LOW'
            
            systemic_data.append({
                'sector_code': sector,
                'sector_name': row['sector_name'],
                'systemic_score': systemic_score,
                'size_score': size_score,
                'leverage_score': leverage_score,
                'activity_score': activity_score,
                'interconnection_score': interconnection_score,
                'risk_level': risk_level,
                'size_pct': row['pct_of_real_economy'],
                'leverage_ratio': leverage_ratio
            })
        
        self.systemic_df = pd.DataFrame(systemic_data)
        self.systemic_df = self.systemic_df.sort_values('systemic_score', ascending=False)
        
        print("\nSystemically Important Sectors:")
        print("-"*70)
        
        for _, row in self.systemic_df.head(10).iterrows():
            emoji = "🔴" if row['risk_level'] == 'CRITICAL' else "🟠" if row['risk_level'] == 'HIGH' else "🟡" if row['risk_level'] == 'MODERATE' else "🟢"
            
            print(f"\n{emoji} {row['sector_code']}: {row['sector_name']}")
            print(f"  Systemic Score: {row['systemic_score']:.3f} ({row['risk_level']})")
            print(f"  Size: {row['size_pct']:.1f}% | Leverage: {row['leverage_ratio']:.1%}")
            print(f"  Scores - Size: {row['size_score']:.2f} | Leverage: {row['leverage_score']:.2f}")
            print(f"          Activity: {row['activity_score']:.2f} | Interconnection: {row['interconnection_score']:.2f}")
        
        return self.systemic_df
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive visualization dashboard."""
        print("\n" + "="*70)
        print("CREATING VISUALIZATION DASHBOARD")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Sector Size Bar Chart (Top 10)
        ax1 = plt.subplot(3, 3, 1)
        top10 = self.sector_df.head(10)
        colors = ['red' if s in ['70', '40', '14', '26'] else 'blue' if s == '15' else 'green' if s in ['10', '11'] else 'gray' 
                 for s in top10['sector_code']]
        
        ax1.barh(range(len(top10)), top10['pct_of_real_economy'], color=colors)
        ax1.set_yticks(range(len(top10)))
        ax1.set_yticklabels([f"{row['sector_code']}: {row['sector_name'][:25]}" 
                            for _, row in top10.iterrows()], fontsize=8)
        ax1.set_xlabel('% of Real Economy')
        ax1.set_title('Top 10 Sectors by Size', fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Leverage Ranking
        ax2 = plt.subplot(3, 3, 2)
        top_leverage = self.leverage_df.head(10)
        colors_lev = ['red' if x > 0.3 else 'orange' if x > 0.2 else 'green' 
                     for x in top_leverage['leverage_ratio']]
        
        ax2.barh(range(len(top_leverage)), top_leverage['leverage_ratio'], color=colors_lev)
        ax2.set_yticks(range(len(top_leverage)))
        ax2.set_yticklabels([f"{row['sector_code']}: {row['sector_name'][:25]}" 
                            for _, row in top_leverage.iterrows()], fontsize=8)
        ax2.set_xlabel('Leverage Ratio')
        ax2.set_title('Highest Leveraged Sectors', fontweight='bold')
        ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()
        
        # 3. Systemic Risk Ranking
        ax3 = plt.subplot(3, 3, 3)
        top_systemic = self.systemic_df.head(10)
        colors_sys = ['darkred' if row['risk_level'] == 'CRITICAL' else 
                     'red' if row['risk_level'] == 'HIGH' else 
                     'orange' if row['risk_level'] == 'MODERATE' else 'green'
                     for _, row in top_systemic.iterrows()]
        
        ax3.barh(range(len(top_systemic)), top_systemic['systemic_score'], color=colors_sys)
        ax3.set_yticks(range(len(top_systemic)))
        ax3.set_yticklabels([f"{row['sector_code']}: {row['sector_name'][:25]}" 
                            for _, row in top_systemic.iterrows()], fontsize=8)
        ax3.set_xlabel('Systemic Risk Score')
        ax3.set_title('Systemically Important Sectors', fontweight='bold')
        ax3.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5)
        ax3.axvline(x=0.6, color='red', linestyle='--', alpha=0.5)
        ax3.invert_yaxis()
        
        # 4. Size vs Leverage Scatter
        ax4 = plt.subplot(3, 3, 4)
        scatter_data = self.systemic_df.head(15)
        colors_scatter = ['red' if row['risk_level'] in ['CRITICAL', 'HIGH'] else 
                         'orange' if row['risk_level'] == 'MODERATE' else 'green'
                         for _, row in scatter_data.iterrows()]
        
        ax4.scatter(scatter_data['size_pct'], 
                   scatter_data['leverage_ratio'],
                   s=scatter_data['systemic_score']*500,
                   alpha=0.6,
                   c=colors_scatter)
        
        for _, row in scatter_data.iterrows():
            ax4.annotate(row['sector_code'], 
                        (row['size_pct'], row['leverage_ratio']),
                        fontsize=7)
        
        ax4.set_xlabel('Size (% of Economy)')
        ax4.set_ylabel('Leverage Ratio')
        ax4.set_title('Size vs Leverage\n(bubble = systemic risk)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
        ax4.axvline(x=10, color='blue', linestyle='--', alpha=0.3)
        
        # 5. Sector Type Distribution
        ax5 = plt.subplot(3, 3, 5)
        
        # Categorize sectors
        categories = {
            'Banking': 0,
            'Insurance/Pension': 0,
            'Shadow Banking': 0,
            'Households': 0,
            'Corporate': 0,
            'Government': 0,
            'Foreign': 0
        }
        
        for _, row in self.sector_df.iterrows():
            code = row['sector_code']
            size = row['pct_of_real_economy']
            
            if code in ['70', '40', '41', '42', '14', '26', '71']:
                categories['Banking'] += size
            elif code in ['79', '47', '43', '31', '48', '49']:
                categories['Insurance/Pension'] += size
            elif code in ['55', '77', '52', '53', '54', '57', '58']:
                categories['Shadow Banking'] += size
            elif code in ['15', '16']:
                categories['Households'] += size
            elif code in ['10', '11']:
                categories['Corporate'] += size
            elif code in ['01', '07', '20', '21', '22']:
                categories['Government'] += size
            elif code in ['08', '13']:
                categories['Foreign'] += size
        
        colors_cat = ['red', 'orange', 'darkred', 'blue', 'green', 'purple', 'gray']
        wedges, texts, autotexts = ax5.pie(categories.values(), 
                                           labels=categories.keys(),
                                           colors=colors_cat,
                                           autopct='%1.1f%%',
                                           startangle=90)
        
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        ax5.set_title('Economy by Sector Type', fontweight='bold')
        
        # 6. Risk Matrix Heatmap
        ax6 = plt.subplot(3, 3, 6)
        
        # Create risk matrix
        risk_matrix = pd.DataFrame(index=['Size', 'Leverage', 'Activity', 'Interconnection'],
                                  columns=[row['sector_code'] for _, row in self.systemic_df.head(8).iterrows()])
        
        for col, row in self.systemic_df.head(8).iterrows():
            risk_matrix.loc['Size', row['sector_code']] = row['size_score']
            risk_matrix.loc['Leverage', row['sector_code']] = row['leverage_score']
            risk_matrix.loc['Activity', row['sector_code']] = row['activity_score']
            risk_matrix.loc['Interconnection', row['sector_code']] = row['interconnection_score']
        
        sns.heatmap(risk_matrix.astype(float), annot=True, fmt='.2f', 
                   cmap='RdYlGn_r', vmin=0, vmax=1, ax=ax6,
                   cbar_kws={'label': 'Risk Score'})
        ax6.set_title('Risk Component Matrix', fontweight='bold')
        ax6.set_xlabel('Sector Code')
        ax6.set_ylabel('Risk Component')
        
        # 7-9. Summary Statistics
        ax7 = plt.subplot(3, 1, 3)
        ax7.axis('off')
        
        # Create summary text
        summary = "RISK ASSESSMENT SUMMARY\n" + "="*80 + "\n\n"
        
        # Critical sectors
        critical = self.systemic_df[self.systemic_df['risk_level'] == 'CRITICAL']
        high = self.systemic_df[self.systemic_df['risk_level'] == 'HIGH']
        
        if len(critical) > 0:
            summary += "🔴 CRITICAL RISK SECTORS:\n"
            for _, row in critical.head(3).iterrows():
                summary += f"  • {row['sector_code']}: {row['sector_name'][:40]}\n"
                summary += f"    Score: {row['systemic_score']:.3f} | Size: {row['size_pct']:.1f}% | Leverage: {row['leverage_ratio']:.1%}\n"
        
        if len(high) > 0:
            summary += "\n🟠 HIGH RISK SECTORS:\n"
            for _, row in high.head(3).iterrows():
                summary += f"  • {row['sector_code']}: {row['sector_name'][:40]}\n"
                summary += f"    Score: {row['systemic_score']:.3f} | Size: {row['size_pct']:.1f}% | Leverage: {row['leverage_ratio']:.1%}\n"
        
        # System metrics
        summary += f"\n\nSYSTEM METRICS:\n"
        summary += f"  • Total Real Economy Size: ${self.sector_df['total_size'].sum():,.0f}M\n"
        summary += f"  • Average Leverage (Top 10): {self.leverage_df.head(10)['leverage_ratio'].mean():.1%}\n"
        summary += f"  • Banking Sector Size: {categories['Banking']:.1f}%\n"
        summary += f"  • Shadow Banking Size: {categories['Shadow Banking']:.1f}%\n"
        
        # Key vulnerabilities
        summary += f"\n\nKEY VULNERABILITIES:\n"
        if categories['Shadow Banking'] > 10:
            summary += f"  ⚠️ Large shadow banking sector ({categories['Shadow Banking']:.1f}%)\n"
        
        high_leverage_banks = self.leverage_df[
            (self.leverage_df['sector_code'].isin(['70', '40', '14', '26'])) & 
            (self.leverage_df['leverage_ratio'] > 0.3)
        ]
        if len(high_leverage_banks) > 0:
            summary += f"  ⚠️ High leverage in banking sector\n"
        
        concentration = self.sector_df.head(3)['pct_of_real_economy'].sum()
        if concentration > 40:
            summary += f"  ⚠️ High concentration (top 3 = {concentration:.1f}%)\n"
        
        ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Real Sector Analysis Dashboard - {self.date}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save
        output_path = self.output_dir / f'real_sector_dashboard_{self.date}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved dashboard to {output_path}")
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive report."""
        report_path = self.output_dir / f'real_sector_report_{self.date}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("REAL SECTOR ANALYSIS REPORT\n")
            f.write(f"Date: {self.date}\n")
            f.write("="*70 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            
            critical = self.systemic_df[self.systemic_df['risk_level'] == 'CRITICAL']
            high = self.systemic_df[self.systemic_df['risk_level'] == 'HIGH']
            
            f.write(f"Critical Risk Sectors: {len(critical)}\n")
            f.write(f"High Risk Sectors: {len(high)}\n")
            f.write(f"Average Leverage (Top 10): {self.leverage_df.head(10)['leverage_ratio'].mean():.1%}\n")
            
            # Detailed findings
            f.write("\n\nDETAILED FINDINGS\n")
            f.write("-"*40 + "\n")
            
            f.write("\n1. SYSTEMICALLY IMPORTANT SECTORS:\n")
            for _, row in self.systemic_df.head(5).iterrows():
                f.write(f"\n{row['sector_code']}: {row['sector_name']}\n")
                f.write(f"  Risk Level: {row['risk_level']}\n")
                f.write(f"  Systemic Score: {row['systemic_score']:.3f}\n")
                f.write(f"  Size: {row['size_pct']:.1f}% of economy\n")
                f.write(f"  Leverage: {row['leverage_ratio']:.1%}\n")
            
            f.write("\n2. HIGHEST LEVERAGE SECTORS:\n")
            for _, row in self.leverage_df.head(5).iterrows():
                if row['leverage_ratio'] > 0:
                    f.write(f"\n{row['sector_code']}: {row['sector_name']}\n")
                    f.write(f"  Leverage Ratio: {row['leverage_ratio']:.1%}\n")
                    f.write(f"  Total Debt: ${row['total_debt']:,.0f}M\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            if len(critical) > 0:
                f.write("\n• IMMEDIATE ACTION REQUIRED for critical sectors\n")
            if len(high) > 2:
                f.write("\n• Multiple high-risk sectors indicate systemic stress\n")
            
            avg_leverage = self.leverage_df.head(10)['leverage_ratio'].mean()
            if avg_leverage > 0.3:
                f.write("\n• System-wide deleveraging may be necessary\n")
        
        print(f"✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run complete corrected analysis."""
        print("\n" + "="*70)
        print("RUNNING CORRECTED SECTOR ANALYSIS")
        print("="*70)
        
        # Step 1: Analyze real sector sizes
        self.analyze_real_sector_sizes()
        
        # Step 2: Calculate proper leverage
        self.calculate_proper_leverage()
        
        # Step 3: Identify systemic sectors
        self.identify_systemic_sectors()
        
        # Step 4: Create dashboard
        self.create_comprehensive_dashboard()
        
        # Step 5: Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return {
            'sector_sizes': self.sector_df,
            'leverage': self.leverage_df,
            'systemic_risk': self.systemic_df
        }


def main():
    """Run corrected sector analysis."""
    
    analyzer = CorrectedSectorAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Executive Summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY - REAL SECTORS")
    print("="*70)
    
    # Find critical sectors
    critical = results['systemic_risk'][results['systemic_risk']['risk_level'] == 'CRITICAL']
    high = results['systemic_risk'][results['systemic_risk']['risk_level'] == 'HIGH']
    
    if len(critical) > 0:
        print("\n🔴 CRITICAL RISK SECTORS:")
        for _, row in critical.iterrows():
            print(f"  • {row['sector_code']}: {row['sector_name']}")
            print(f"    Systemic Score: {row['systemic_score']:.3f}")
            print(f"    Leverage: {row['leverage_ratio']:.1%}")
    
    if len(high) > 0:
        print("\n🟠 HIGH RISK SECTORS:")
        for _, row in high.iterrows():
            print(f"  • {row['sector_code']}: {row['sector_name']}")
            print(f"    Systemic Score: {row['systemic_score']:.3f}")
            print(f"    Leverage: {row['leverage_ratio']:.1%}")
    
    # Overall assessment
    avg_leverage = results['leverage']['leverage_ratio'].mean()
    print(f"\n📊 SYSTEM METRICS:")
    print(f"  Average Leverage: {avg_leverage:.1%}")
    print(f"  Critical Sectors: {len(critical)}")
    print(f"  High Risk Sectors: {len(high)}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
