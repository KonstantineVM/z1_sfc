#!/usr/bin/env python3
"""
SFC Granular Sector Decomposition Analysis
===========================================
Analyzes all Z1 sectors to identify the top 10 by size and their risk contributions.
Provides detailed sector-by-sector risk assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Z1 Sector Codes and Names
SECTOR_NAMES = {
    '01': 'Federal Government',
    '07': 'State and Local Governments',
    '08': 'Foreign Sector',
    '09': 'Monetary Authority',
    '10': 'Nonfinancial Corporate Business',
    '11': 'Nonfinancial Noncorporate Business',
    '12': 'Households (Old)',
    '13': 'Rest of the World',
    '14': 'Private Depository Institutions',
    '15': 'Households and Nonprofit Organizations',
    '16': 'Nonprofit Organizations',
    '17': 'Government-Sponsored Enterprises',
    '18': 'Agency- and GSE-backed Mortgage Pools',
    '19': 'Domestic Hedge Funds',
    '20': 'Federal Government (Defense)',
    '21': 'Federal Government (Nondefense)',
    '22': 'State and Local Governments (General)',
    '23': 'State and Local Government Employee Retirement Funds',
    '26': 'Foreign Banking Offices in U.S.',
    '31': 'Private Pension Funds',
    '34': 'Federal Government Employee Retirement Funds',
    '36': 'State and Local Government Employee DB Retirement Funds',
    '37': 'State and Local Government Employee DC Retirement Funds',
    '38': 'Federal Government DB Retirement Funds',
    '39': 'Federal Government DC Retirement Funds',
    '40': 'Private Depository Institutions (Commercial Banking)',
    '41': 'Private Depository Institutions (Savings Institutions)',
    '42': 'Credit Unions',
    '43': 'Property-Casualty Insurance Companies',
    '47': 'Life Insurance Companies',
    '48': 'Private Pension Funds (DB)',
    '49': 'Private Pension Funds (DC)',
    '50': 'Government-Sponsored Enterprises (GSEs)',
    '51': 'Agency- and GSE-backed Mortgage Pools',
    '52': 'Issuers of Asset-Backed Securities',
    '53': 'Finance Companies',
    '54': 'Real Estate Investment Trusts (REITs)',
    '55': 'Security Brokers and Dealers',
    '57': 'Holding Companies',
    '58': 'Funding Corporations',
    '59': 'Private Equity Funds',
    '60': 'Hedge Funds',
    '61': 'Money Market Funds',
    '65': 'Mutual Funds',
    '66': 'Exchange-Traded Funds',
    '68': 'Banks and Credit Unions in U.S.-Affiliated Areas',
    '69': 'Other Financial Business',
    '70': 'U.S.-Chartered Depository Institutions',
    '71': 'Foreign Banking Offices in U.S.',
    '73': 'Banks in U.S.-Affiliated Areas',
    '74': 'Savings and Loan Associations',
    '75': 'Finance Companies (captive)',
    '76': 'Finance Companies (independent)',
    '77': 'Brokers and Dealers',
    '78': 'Private Equity',
    '79': 'Insurance Companies',
    '80': 'Defined Benefit Pension Funds',
    '81': 'Defined Contribution Pension Funds',
    '82': 'State and Local Governments (excluding employee retirement)',
    '83': 'Federal Government (unified budget)',
    '84': 'Rest of the World (official)',
    '85': 'Rest of the World (private)',
    '86': 'Domestic Nonfinancial Sectors',
    '87': 'Domestic Financial Sectors',
    '88': 'All Domestic Sectors',
    '89': 'All Sectors',
    '90': 'Financial Business',
    '91': 'Nonfinancial Business'
}


class SectorDecompositionAnalyzer:
    """
    Analyzes sector composition and risk contributions in detail.
    """
    
    def __init__(self, balance_sheet_file: str = None, transaction_file: str = None):
        """Initialize with most recent data files."""
        self.output_dir = Path("outputs/sector_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find most recent files if not specified
        if balance_sheet_file is None:
            bs_files = sorted(Path("outputs").glob("sfc_balance_sheet_*.csv"))
            if bs_files:
                balance_sheet_file = str(bs_files[-1])
            else:
                raise FileNotFoundError("No balance sheet files found")
        
        if transaction_file is None:
            tf_files = sorted(Path("outputs").glob("sfc_transactions_*.csv"))
            if tf_files:
                transaction_file = str(tf_files[-1])
            else:
                raise FileNotFoundError("No transaction files found")
        
        self.bs_file = balance_sheet_file
        self.tf_file = transaction_file
        
        # Extract date from filename
        self.date = Path(balance_sheet_file).stem.replace('sfc_balance_sheet_', '')
        
        print(f"Analyzing data from: {self.date}")
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load balance sheet and transaction data."""
        self.bs = pd.read_csv(self.bs_file, index_col=0)
        self.tf = pd.read_csv(self.tf_file, index_col=0)
        
        # Get sectors (excluding label and Total columns)
        self.sectors = [c for c in self.bs.columns if c not in ['label', 'Total']]
        
        print(f"Loaded {len(self.sectors)} sectors")
        
    def analyze_sector_sizes(self):
        """Analyze sector sizes by total assets and liabilities."""
        print("\n" + "="*70)
        print("SECTOR SIZE ANALYSIS")
        print("="*70)
        
        sector_analysis = []
        
        for sector in self.sectors:
            if sector not in self.bs.columns:
                continue
            
            # Calculate total assets (negative values in Z1)
            assets = self.bs[self.bs[sector] < 0][sector].abs().sum()
            
            # Calculate total liabilities (positive values in Z1)
            liabilities = self.bs[self.bs[sector] > 0][sector].abs().sum()
            
            # Total size (both sides of balance sheet)
            total_size = assets + liabilities
            
            # Net position
            net_position = self.bs[sector].sum()
            
            # Transaction volume
            transaction_volume = self.tf[sector].abs().sum() if sector in self.tf.columns else 0
            
            sector_analysis.append({
                'sector_code': sector,
                'sector_name': SECTOR_NAMES.get(sector, f'Sector {sector}'),
                'total_assets': assets,
                'total_liabilities': liabilities,
                'total_size': total_size,
                'net_position': net_position,
                'transaction_volume': transaction_volume,
                'asset_liability_ratio': assets / liabilities if liabilities > 0 else 0
            })
        
        self.sector_df = pd.DataFrame(sector_analysis)
        
        # Sort by total size
        self.sector_df = self.sector_df.sort_values('total_size', ascending=False)
        
        # Add percentage of total
        total_system_size = self.sector_df['total_size'].sum()
        self.sector_df['pct_of_system'] = (self.sector_df['total_size'] / total_system_size * 100)
        
        # Save to file
        self.sector_df.to_csv(self.output_dir / f'sector_sizes_{self.date}.csv', index=False)
        
        # Print top 10
        print("\nTop 10 Sectors by Total Size (in millions):")
        print("-"*70)
        
        for idx, row in self.sector_df.head(10).iterrows():
            print(f"\n{row['sector_code']}: {row['sector_name']}")
            print(f"  Total Size: ${row['total_size']:,.0f}M ({row['pct_of_system']:.1f}% of system)")
            print(f"  Assets: ${row['total_assets']:,.0f}M")
            print(f"  Liabilities: ${row['total_liabilities']:,.0f}M")
            print(f"  A/L Ratio: {row['asset_liability_ratio']:.2f}")
            print(f"  Transaction Volume: ${row['transaction_volume']:,.0f}M")
        
        return self.sector_df
    
    def analyze_sector_leverage(self):
        """Analyze leverage by sector."""
        print("\n" + "="*70)
        print("SECTOR LEVERAGE ANALYSIS")
        print("="*70)
        
        # Load instrument map to identify debt instruments
        try:
            with open("mappings/instrument_map.json", 'r') as f:
                imap = json.load(f)
        except:
            print("Warning: Could not load instrument map")
            imap = {}
        
        # Identify debt instruments (simplified approach)
        debt_instruments = []
        for code, meta in imap.items():
            if any(term in str(meta.get('class', '')).lower() 
                   for term in ['debt', 'bond', 'loan', 'mortgage', 'credit', 'paper']):
                debt_instruments.append(code)
        
        # Convert to integers to match balance sheet index
        debt_instruments_int = []
        for inst in debt_instruments:
            try:
                debt_instruments_int.append(int(inst))
            except:
                pass
        
        print(f"Identified {len(debt_instruments_int)} debt instruments")
        
        # Calculate leverage for top 10 sectors
        leverage_analysis = []
        
        for _, row in self.sector_df.head(10).iterrows():
            sector = row['sector_code']
            
            if sector not in self.bs.columns:
                continue
            
            # Calculate debt
            sector_debt = 0
            for inst in debt_instruments_int:
                if inst in self.bs.index:
                    sector_debt += abs(self.bs.loc[inst, sector])
            
            # Total assets
            total_assets = row['total_assets']
            
            # Calculate leverage
            leverage = sector_debt / total_assets if total_assets > 0 else 0
            
            leverage_analysis.append({
                'sector_code': sector,
                'sector_name': row['sector_name'],
                'total_debt': sector_debt,
                'total_assets': total_assets,
                'leverage_ratio': leverage,
                'size_pct': row['pct_of_system']
            })
        
        self.leverage_df = pd.DataFrame(leverage_analysis)
        self.leverage_df = self.leverage_df.sort_values('leverage_ratio', ascending=False)
        
        print("\nTop 10 Sectors - Leverage Ratios:")
        print("-"*70)
        
        for _, row in self.leverage_df.iterrows():
            risk_level = "🔴 HIGH" if row['leverage_ratio'] > 0.4 else "🟡 MODERATE" if row['leverage_ratio'] > 0.2 else "🟢 LOW"
            print(f"\n{row['sector_code']}: {row['sector_name']} ({row['size_pct']:.1f}% of system)")
            print(f"  Leverage: {row['leverage_ratio']:.1%} {risk_level}")
            print(f"  Debt: ${row['total_debt']:,.0f}M")
            print(f"  Assets: ${row['total_assets']:,.0f}M")
        
        return self.leverage_df
    
    def analyze_interconnectedness(self):
        """Analyze interconnections between top sectors."""
        print("\n" + "="*70)
        print("SECTOR INTERCONNECTEDNESS ANALYSIS")
        print("="*70)
        
        # Get top 10 sectors
        top_10_sectors = self.sector_df.head(10)['sector_code'].tolist()
        
        # Create interconnection matrix
        interconnection_matrix = pd.DataFrame(index=top_10_sectors, columns=top_10_sectors, data=0.0)
        
        # Analyze cross-holdings
        for sector1 in top_10_sectors:
            for sector2 in top_10_sectors:
                if sector1 == sector2:
                    continue
                
                # Sum all instruments where sector1 has assets and sector2 has liabilities
                cross_exposure = 0
                for inst in self.bs.index:
                    val1 = self.bs.loc[inst, sector1] if sector1 in self.bs.columns else 0
                    val2 = self.bs.loc[inst, sector2] if sector2 in self.bs.columns else 0
                    
                    # If opposite signs, there's a cross-holding
                    if val1 * val2 < 0:
                        cross_exposure += min(abs(val1), abs(val2))
                
                interconnection_matrix.loc[sector1, sector2] = cross_exposure
        
        # Save matrix
        interconnection_matrix.to_csv(self.output_dir / f'interconnection_matrix_{self.date}.csv')
        
        # Find most interconnected pairs
        connections = []
        for i, sector1 in enumerate(top_10_sectors):
            for j, sector2 in enumerate(top_10_sectors):
                if i < j:  # Avoid duplicates
                    exposure = interconnection_matrix.loc[sector1, sector2] + interconnection_matrix.loc[sector2, sector1]
                    connections.append({
                        'sector1': sector1,
                        'sector2': sector2,
                        'name1': SECTOR_NAMES.get(sector1, sector1),
                        'name2': SECTOR_NAMES.get(sector2, sector2),
                        'total_exposure': exposure
                    })
        
        connections_df = pd.DataFrame(connections).sort_values('total_exposure', ascending=False)
        
        print("\nTop 10 Sector Interconnections (in millions):")
        print("-"*70)
        
        for _, row in connections_df.head(10).iterrows():
            print(f"\n{row['sector1']} ↔ {row['sector2']}")
            print(f"  {row['name1'][:30]} ↔ {row['name2'][:30]}")
            print(f"  Total Exposure: ${row['total_exposure']:,.0f}M")
        
        return connections_df
    
    def calculate_systemic_risk_contribution(self):
        """Calculate each sector's contribution to systemic risk."""
        print("\n" + "="*70)
        print("SYSTEMIC RISK CONTRIBUTION ANALYSIS")
        print("="*70)
        
        risk_scores = []
        
        for _, row in self.sector_df.head(10).iterrows():
            sector = row['sector_code']
            
            # Size risk (larger = more systemic)
            size_risk = row['pct_of_system'] / 100
            
            # Leverage risk
            leverage = self.leverage_df[self.leverage_df['sector_code'] == sector]['leverage_ratio'].values
            leverage_risk = leverage[0] if len(leverage) > 0 else 0
            
            # Transaction volume risk (more active = more systemic)
            transaction_risk = row['transaction_volume'] / self.sector_df['transaction_volume'].sum()
            
            # Calculate composite systemic risk score
            # Weighted: 40% size, 40% leverage, 20% transaction volume
            systemic_risk = (0.4 * size_risk + 0.4 * leverage_risk + 0.2 * transaction_risk)
            
            risk_scores.append({
                'sector_code': sector,
                'sector_name': row['sector_name'],
                'size_pct': row['pct_of_system'],
                'leverage_ratio': leverage_risk,
                'transaction_share': transaction_risk * 100,
                'systemic_risk_score': systemic_risk,
                'risk_level': 'CRITICAL' if systemic_risk > 0.3 else 'HIGH' if systemic_risk > 0.2 else 'MODERATE' if systemic_risk > 0.1 else 'LOW'
            })
        
        self.risk_df = pd.DataFrame(risk_scores).sort_values('systemic_risk_score', ascending=False)
        
        print("\nSystemic Risk Contribution by Sector:")
        print("-"*70)
        
        for _, row in self.risk_df.iterrows():
            emoji = "🔴" if row['risk_level'] == 'CRITICAL' else "🟠" if row['risk_level'] == 'HIGH' else "🟡" if row['risk_level'] == 'MODERATE' else "🟢"
            print(f"\n{emoji} {row['sector_code']}: {row['sector_name']}")
            print(f"  Systemic Risk Score: {row['systemic_risk_score']:.3f} ({row['risk_level']})")
            print(f"  Size: {row['size_pct']:.1f}% of system")
            print(f"  Leverage: {row['leverage_ratio']:.1%}")
            print(f"  Transaction Share: {row['transaction_share']:.1f}%")
        
        return self.risk_df
    
    def create_visualizations(self):
        """Create comprehensive sector analysis visualizations."""
        print("\n" + "="*70)
        print("CREATING SECTOR VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sector Size Distribution (Pie Chart)
        ax1 = plt.subplot(2, 3, 1)
        top_10 = self.sector_df.head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        wedges, texts, autotexts = ax1.pie(
            top_10['pct_of_system'], 
            labels=[f"{row['sector_code']}: {row['sector_name'][:15]}" for _, row in top_10.iterrows()],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Make percentage text smaller
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        ax1.set_title('Top 10 Sectors by Size\n(% of Total System)', fontweight='bold')
        
        # 2. Leverage Comparison (Bar Chart)
        ax2 = plt.subplot(2, 3, 2)
        leverage_sorted = self.leverage_df.sort_values('leverage_ratio', ascending=True)
        
        colors_leverage = ['red' if x > 0.4 else 'orange' if x > 0.2 else 'green' 
                          for x in leverage_sorted['leverage_ratio']]
        
        ax2.barh(range(len(leverage_sorted)), leverage_sorted['leverage_ratio'], color=colors_leverage)
        ax2.set_yticks(range(len(leverage_sorted)))
        ax2.set_yticklabels([f"{row['sector_code']}: {row['sector_name'][:20]}" 
                            for _, row in leverage_sorted.iterrows()], fontsize=8)
        ax2.set_xlabel('Leverage Ratio')
        ax2.set_title('Sector Leverage Ratios', fontweight='bold')
        ax2.axvline(x=0.4, color='red', linestyle='--', alpha=0.5, label='High Risk')
        ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk')
        ax2.legend(fontsize=8)
        
        # 3. Systemic Risk Scores (Bar Chart)
        ax3 = plt.subplot(2, 3, 3)
        risk_sorted = self.risk_df.sort_values('systemic_risk_score', ascending=True)
        
        colors_risk = ['darkred' if row['risk_level'] == 'CRITICAL' else 
                      'red' if row['risk_level'] == 'HIGH' else 
                      'orange' if row['risk_level'] == 'MODERATE' else 'green'
                      for _, row in risk_sorted.iterrows()]
        
        ax3.barh(range(len(risk_sorted)), risk_sorted['systemic_risk_score'], color=colors_risk)
        ax3.set_yticks(range(len(risk_sorted)))
        ax3.set_yticklabels([f"{row['sector_code']}: {row['sector_name'][:20]}" 
                            for _, row in risk_sorted.iterrows()], fontsize=8)
        ax3.set_xlabel('Systemic Risk Score')
        ax3.set_title('Systemic Risk Contribution', fontweight='bold')
        
        # 4. Size vs Leverage Scatter
        ax4 = plt.subplot(2, 3, 4)
        
        scatter_colors = ['red' if row['risk_level'] in ['CRITICAL', 'HIGH'] else 
                         'orange' if row['risk_level'] == 'MODERATE' else 'green'
                         for _, row in self.risk_df.iterrows()]
        
        ax4.scatter(self.risk_df['size_pct'], 
                   self.risk_df['leverage_ratio'], 
                   s=self.risk_df['systemic_risk_score']*1000,
                   alpha=0.6,
                   c=scatter_colors)
        
        for _, row in self.risk_df.iterrows():
            ax4.annotate(row['sector_code'], 
                        (row['size_pct'], row['leverage_ratio']),
                        fontsize=8)
        
        ax4.set_xlabel('Size (% of System)')
        ax4.set_ylabel('Leverage Ratio')
        ax4.set_title('Size vs Leverage\n(bubble size = systemic risk)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Transaction Flow Heatmap (Top 5x5)
        ax5 = plt.subplot(2, 3, 5)
        top_5_sectors = self.sector_df.head(5)['sector_code'].tolist()
        
        # Create flow matrix for top 5
        flow_matrix = pd.DataFrame(index=top_5_sectors, columns=top_5_sectors, data=0.0)
        
        for sector1 in top_5_sectors:
            for sector2 in top_5_sectors:
                if sector1 in self.tf.columns and sector2 in self.tf.columns:
                    # Use transaction flows
                    for inst in self.tf.index[:50]:  # Sample for speed
                        val1 = self.tf.loc[inst, sector1] if sector1 in self.tf.columns else 0
                        val2 = self.tf.loc[inst, sector2] if sector2 in self.tf.columns else 0
                        if val1 > 0 and val2 < 0:
                            flow_matrix.loc[sector1, sector2] += val1
        
        # Plot heatmap
        sns.heatmap(flow_matrix, annot=False, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Flow Volume'})
        ax5.set_title('Transaction Flows (Top 5 Sectors)', fontweight='bold')
        ax5.set_xlabel('To Sector')
        ax5.set_ylabel('From Sector')
        
        # 6. Risk Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary text
        summary_text = "RISK SUMMARY\n" + "="*40 + "\n\n"
        
        # Critical sectors
        critical = self.risk_df[self.risk_df['risk_level'] == 'CRITICAL']
        if len(critical) > 0:
            summary_text += "🔴 CRITICAL RISK SECTORS:\n"
            for _, row in critical.iterrows():
                summary_text += f"  • {row['sector_code']}: {row['sector_name'][:30]}\n"
                summary_text += f"    Score: {row['systemic_risk_score']:.3f}, Leverage: {row['leverage_ratio']:.1%}\n"
        
        # High risk sectors
        high = self.risk_df[self.risk_df['risk_level'] == 'HIGH']
        if len(high) > 0:
            summary_text += "\n🟠 HIGH RISK SECTORS:\n"
            for _, row in high.iterrows():
                summary_text += f"  • {row['sector_code']}: {row['sector_name'][:30]}\n"
                summary_text += f"    Score: {row['systemic_risk_score']:.3f}, Leverage: {row['leverage_ratio']:.1%}\n"
        
        # System metrics
        summary_text += f"\n\nSYSTEM METRICS:\n"
        summary_text += f"  Total System Size: ${self.sector_df['total_size'].sum():,.0f}M\n"
        summary_text += f"  Avg Leverage (Top 10): {self.leverage_df['leverage_ratio'].mean():.1%}\n"
        summary_text += f"  Concentration (Top 3): {self.sector_df.head(3)['pct_of_system'].sum():.1f}%\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Sector Decomposition Analysis - {self.date}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save
        output_path = self.output_dir / f'sector_analysis_{self.date}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive sector analysis report."""
        report_path = self.output_dir / f'sector_report_{self.date}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"SECTOR DECOMPOSITION ANALYSIS REPORT\n")
            f.write(f"Date: {self.date}\n")
            f.write("="*70 + "\n\n")
            
            # Top sectors by size
            f.write("TOP 10 SECTORS BY SIZE\n")
            f.write("-"*40 + "\n")
            for idx, row in self.sector_df.head(10).iterrows():
                f.write(f"\n{row['sector_code']}: {row['sector_name']}\n")
                f.write(f"  Size: ${row['total_size']:,.0f}M ({row['pct_of_system']:.1f}%)\n")
                f.write(f"  Assets: ${row['total_assets']:,.0f}M\n")
                f.write(f"  Liabilities: ${row['total_liabilities']:,.0f}M\n")
            
            # Risk assessment
            f.write("\n\nSYSTEMIC RISK ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            for _, row in self.risk_df.iterrows():
                f.write(f"\n{row['sector_code']}: {row['sector_name']}\n")
                f.write(f"  Risk Level: {row['risk_level']}\n")
                f.write(f"  Systemic Score: {row['systemic_risk_score']:.3f}\n")
                f.write(f"  Leverage: {row['leverage_ratio']:.1%}\n")
            
            # Key findings
            f.write("\n\nKEY FINDINGS\n")
            f.write("-"*40 + "\n")
            
            # Concentration risk
            top3_concentration = self.sector_df.head(3)['pct_of_system'].sum()
            f.write(f"\n• System Concentration: Top 3 sectors = {top3_concentration:.1f}%\n")
            if top3_concentration > 50:
                f.write("  ⚠️ HIGH CONCENTRATION RISK\n")
            
            # Leverage risk
            high_leverage = self.leverage_df[self.leverage_df['leverage_ratio'] > 0.4]
            if len(high_leverage) > 0:
                f.write(f"\n• High Leverage Sectors ({len(high_leverage)}):\n")
                for _, row in high_leverage.iterrows():
                    f.write(f"  - {row['sector_code']}: {row['leverage_ratio']:.1%}\n")
            
            # Systemic risk
            critical_risk = self.risk_df[self.risk_df['risk_level'] == 'CRITICAL']
            if len(critical_risk) > 0:
                f.write(f"\n• CRITICAL Risk Sectors ({len(critical_risk)}):\n")
                for _, row in critical_risk.iterrows():
                    f.write(f"  - {row['sector_code']}: {row['sector_name']}\n")
        
        print(f"✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run complete sector decomposition analysis."""
        print("\n" + "="*70)
        print("RUNNING COMPLETE SECTOR DECOMPOSITION")
        print("="*70)
        
        # Step 1: Analyze sector sizes
        self.analyze_sector_sizes()
        
        # Step 2: Analyze leverage
        self.analyze_sector_leverage()
        
        # Step 3: Analyze interconnectedness
        self.analyze_interconnectedness()
        
        # Step 4: Calculate systemic risk
        self.calculate_systemic_risk_contribution()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        # Step 6: Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'sector_sizes': self.sector_df,
            'leverage': self.leverage_df,
            'systemic_risk': self.risk_df
        }


def main():
    """Run sector decomposition analysis."""
    
    # Initialize analyzer
    analyzer = SectorDecompositionAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    # Identify highest risk sectors
    critical_sectors = results['systemic_risk'][results['systemic_risk']['risk_level'] == 'CRITICAL']
    high_risk_sectors = results['systemic_risk'][results['systemic_risk']['risk_level'] == 'HIGH']
    
    print(f"\n🔴 Critical Risk Sectors: {len(critical_sectors)}")
    for _, row in critical_sectors.iterrows():
        print(f"  • {row['sector_code']}: {row['sector_name']} (Score: {row['systemic_risk_score']:.3f})")
    
    print(f"\n🟠 High Risk Sectors: {len(high_risk_sectors)}")
    for _, row in high_risk_sectors.iterrows():
        print(f"  • {row['sector_code']}: {row['sector_name']} (Score: {row['systemic_risk_score']:.3f})")
    
    # System concentration
    top3_pct = results['sector_sizes'].head(3)['pct_of_system'].sum()
    print(f"\n📊 System Concentration:")
    print(f"  Top 3 sectors control {top3_pct:.1f}% of system")
    
    # Average leverage
    avg_leverage = results['leverage']['leverage_ratio'].mean()
    print(f"\n📈 Average Leverage (Top 10): {avg_leverage:.1%}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
