#!/usr/bin/env python3
"""
SFC Time Series Indicator Tracker
==================================
Tracks critical indicators over time to validate crisis assessment.
Analyzes trends in leverage, liquidity, and systemic risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Critical sectors to track
CRITICAL_SECTORS = {
    '10': 'Nonfinancial Corporate',
    '11': 'Nonfinancial Noncorporate',
    '15': 'Households',
    '70': 'U.S. Banks',
    '40': 'Commercial Banking',
    '14': 'Private Depository',
    '26': 'Foreign Banking',
    '79': 'Insurance',
    '31': 'Pension Funds',
    '58': 'Funding Corps',
    '52': 'Asset-Backed Securities',
    '65': 'Mutual Funds'
}

# Recession periods for reference
RECESSIONS = [
    ('2007-12-01', '2009-06-01', '2008 Financial Crisis'),
    ('2020-02-01', '2020-04-01', 'COVID Recession'),
]


class TimeSeriesTracker:
    """
    Tracks critical financial indicators over time.
    """
    
    def __init__(self, start_year=2000, end_year=2025):
        """Initialize tracker with date range."""
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = Path("outputs/time_series_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for time series
        self.leverage_series = {}
        self.size_series = {}
        self.liquidity_series = {}
        self.systemic_series = {}
        self.all_dates = []
        
        print(f"Tracking indicators from {start_year} to {end_year}")
        
    def collect_historical_data(self):
        """Collect all available historical data."""
        print("\n" + "="*70)
        print("COLLECTING HISTORICAL DATA")
        print("="*70)
        
        # Find all balance sheet and transaction files
        bs_files = sorted(Path("outputs").glob("sfc_balance_sheet_*.csv"))
        tf_files = sorted(Path("outputs").glob("sfc_transactions_*.csv"))
        
        # Also check for indicators file
        indicator_files = list(Path("outputs").glob("*indicators*.csv"))
        
        print(f"Found {len(bs_files)} balance sheet files")
        print(f"Found {len(tf_files)} transaction files")
        print(f"Found {len(indicator_files)} indicator files")
        
        # Process each quarter
        processed_quarters = []
        
        for bs_file in bs_files:
            date_str = bs_file.stem.replace('sfc_balance_sheet_', '')
            
            # Filter by year range
            year = int(date_str[:4])
            if year < self.start_year or year > self.end_year:
                continue
            
            # Find corresponding transaction file
            tf_file = Path("outputs") / f"sfc_transactions_{date_str}.csv"
            
            if not tf_file.exists():
                continue
            
            # Process this quarter
            result = self.analyze_quarter(date_str, bs_file, tf_file)
            if result:
                processed_quarters.append(result)
        
        # Sort by date
        processed_quarters.sort(key=lambda x: x['date'])
        self.all_dates = [q['date'] for q in processed_quarters]
        
        # Build time series
        for quarter in processed_quarters:
            date = quarter['date']
            
            # Store leverage by sector
            for sector, leverage in quarter.get('leverage_by_sector', {}).items():
                if sector not in self.leverage_series:
                    self.leverage_series[sector] = {}
                self.leverage_series[sector][date] = leverage
            
            # Store size by sector
            for sector, size in quarter.get('size_by_sector', {}).items():
                if sector not in self.size_series:
                    self.size_series[sector] = {}
                self.size_series[sector][date] = size
            
            # Store system metrics
            if 'system_metrics' in quarter:
                metrics = quarter['system_metrics']
                if 'liquidity' not in self.liquidity_series:
                    self.liquidity_series = {}
                self.liquidity_series[date] = metrics
        
        print(f"\n✓ Processed {len(processed_quarters)} quarters")
        
        # Load existing indicators if available
        if indicator_files:
            self.load_existing_indicators(indicator_files[0])
        
        return processed_quarters
    
    def analyze_quarter(self, date_str, bs_file, tf_file):
        """Analyze a single quarter using the WORKING leverage calculation method."""
        try:
            # Load data
            bs = pd.read_csv(bs_file, index_col=0)
            tf = pd.read_csv(tf_file, index_col=0)
            
            # Get real sectors only
            sectors = [s for s in bs.columns if s in CRITICAL_SECTORS.keys()]
            
            if not sectors:
                return None
            
            result = {
                'date': pd.to_datetime(date_str),
                'leverage_by_sector': {},
                'size_by_sector': {},
                'system_metrics': {}
            }
            
            # Calculate metrics for each sector using PROVEN METHOD
            total_system_size = 0
            total_system_liabilities = 0
            total_system_assets = 0
            leverage_values = []
            
            for sector in sectors:
                # Use the WORKING calculation method from diagnostic
                # This method correctly calculates leverage as positive/(positive+negative)
                sector_data = bs[sector]
                
                # Calculate components
                positive_sum = sector_data[sector_data > 0].sum()  # Liabilities
                negative_sum = abs(sector_data[sector_data < 0].sum())  # Assets
                total = positive_sum + negative_sum
                
                # Calculate leverage (positive/total method that WORKS)
                if total > 0:
                    leverage = positive_sum / total
                else:
                    leverage = 0
                
                # Store results
                result['leverage_by_sector'][sector] = leverage
                result['size_by_sector'][sector] = total
                
                # Accumulate for system metrics
                total_system_size += total
                total_system_liabilities += positive_sum
                total_system_assets += negative_sum
                
                if leverage > 0:  # Only include valid leverages
                    leverage_values.append(leverage)
            
            # System-wide metrics using WORKING calculations
            if total_system_size > 0:
                result['system_metrics']['avg_leverage'] = total_system_liabilities / total_system_size
            else:
                result['system_metrics']['avg_leverage'] = 0
            
            # Alternative: Use mean of sector leverages
            if leverage_values:
                result['system_metrics']['avg_leverage_alt'] = np.mean(leverage_values)
            
            result['system_metrics']['total_size'] = total_system_size
            result['system_metrics']['total_liabilities'] = total_system_liabilities
            result['system_metrics']['total_assets'] = total_system_assets
            
            # Calculate liquidity (simplified)
            liquid_instruments = [30200, 30250, 30300]  # Cash and deposits as integers
            system_liquid = 0
            for inst in liquid_instruments:
                if inst in bs.index:
                    for sector in sectors:
                        if sector in bs.columns:
                            system_liquid += abs(bs.loc[inst, sector])
            
            result['system_metrics']['liquidity'] = system_liquid / total_system_size if total_system_size > 0 else 0
            
            return result
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_existing_indicators(self, indicator_file):
        """Load existing calculated indicators."""
        print(f"\nLoading existing indicators from {indicator_file}")
        
        df = pd.read_csv(indicator_file, index_col=0, parse_dates=True)
        
        # Filter by date range
        df = df[(df.index.year >= self.start_year) & (df.index.year <= self.end_year)]
        
        # Store relevant series
        self.indicator_df = df
        
        print(f"Loaded {len(df)} quarters of indicators")
    
    def create_time_series_visualizations(self):
        """Create comprehensive time series visualizations."""
        print("\n" + "="*70)
        print("CREATING TIME SERIES VISUALIZATIONS")
        print("="*70)
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Leverage Evolution by Sector
        ax1 = plt.subplot(4, 2, 1)
        self.plot_leverage_evolution(ax1)
        
        # 2. System-wide Leverage Trend
        ax2 = plt.subplot(4, 2, 2)
        self.plot_system_leverage(ax2)
        
        # 3. Sector Size Evolution
        ax3 = plt.subplot(4, 2, 3)
        self.plot_sector_sizes(ax3)
        
        # 4. Liquidity Trend
        ax4 = plt.subplot(4, 2, 4)
        self.plot_liquidity_trend(ax4)
        
        # 5. Crisis Indicators (if available)
        ax5 = plt.subplot(4, 2, 5)
        self.plot_crisis_indicators(ax5)
        
        # 6. Leverage vs Time (Critical Sectors)
        ax6 = plt.subplot(4, 2, 6)
        self.plot_critical_sectors(ax6)
        
        # 7. Year-over-Year Changes
        ax7 = plt.subplot(4, 2, 7)
        self.plot_yoy_changes(ax7)
        
        # 8. Risk Dashboard
        ax8 = plt.subplot(4, 2, 8)
        self.plot_risk_dashboard(ax8)
        
        plt.suptitle('Financial System Time Series Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = self.output_dir / 'time_series_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
        
        return fig
    
    def plot_leverage_evolution(self, ax):
        """Plot leverage evolution for key sectors."""
        
        # Select key sectors to plot
        key_sectors = ['10', '15', '70', '79', '31']
        
        for sector in key_sectors:
            if sector in self.leverage_series:
                dates = sorted(self.leverage_series[sector].keys())
                values = [self.leverage_series[sector][d] for d in dates]
                
                label = CRITICAL_SECTORS.get(sector, sector)
                ax.plot(dates, values, marker='o', markersize=3, label=label, linewidth=2)
        
        # Add recession shading
        for start, end, name in RECESSIONS:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                      alpha=0.2, color='gray')
        
        # Add danger zone
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Danger Level')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Warning Level')
        
        ax.set_title('Leverage Evolution by Sector', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Leverage Ratio')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    def plot_system_leverage(self, ax):
        """Plot system-wide average leverage."""
        
        if self.liquidity_series:
            dates = sorted(self.liquidity_series.keys())
            avg_leverages = []
            
            for date in dates:
                if 'avg_leverage' in self.liquidity_series[date]:
                    avg_leverages.append(self.liquidity_series[date]['avg_leverage'])
                else:
                    avg_leverages.append(np.nan)
            
            # Remove NaN values
            valid_data = [(d, v) for d, v in zip(dates, avg_leverages) if not np.isnan(v)]
            if valid_data:
                dates, values = zip(*valid_data)
                
                ax.plot(dates, values, 'r-', linewidth=3, label='System Average')
                ax.fill_between(dates, values, alpha=0.3, color='red')
                
                # Add trend line
                if len(dates) > 1:
                    z = np.polyfit(mdates.date2num(dates), values, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(mdates.date2num(dates)), 'r--', alpha=0.5, label='Trend')
        
        # Add recession shading
        for start, end, name in RECESSIONS:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                      alpha=0.2, color='gray')
        
        ax.set_title('System-Wide Average Leverage', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Leverage Ratio')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add current value annotation
        if valid_data:
            current_value = values[-1]
            ax.annotate(f'Current: {current_value:.1%}',
                       xy=(dates[-1], current_value),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    def plot_sector_sizes(self, ax):
        """Plot evolution of sector sizes."""
        
        # Calculate total size over time
        total_sizes = {}
        for sector in self.size_series:
            for date, size in self.size_series[sector].items():
                if date not in total_sizes:
                    total_sizes[date] = 0
                total_sizes[date] += size
        
        # Calculate percentages for key sectors
        key_sectors = ['10', '15', '79', '70']
        
        for sector in key_sectors:
            if sector in self.size_series:
                dates = sorted(self.size_series[sector].keys())
                percentages = []
                
                for date in dates:
                    if date in total_sizes and total_sizes[date] > 0:
                        pct = self.size_series[sector][date] / total_sizes[date] * 100
                        percentages.append(pct)
                    else:
                        percentages.append(0)
                
                label = CRITICAL_SECTORS.get(sector, sector)
                ax.plot(dates, percentages, marker='o', markersize=3, label=label)
        
        ax.set_title('Sector Size Evolution (% of Economy)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('% of Total Economy')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_liquidity_trend(self, ax):
        """Plot system liquidity over time."""
        
        if hasattr(self, 'indicator_df') and 'system_liquidity_ratio' in self.indicator_df.columns:
            liquidity = self.indicator_df['system_liquidity_ratio'].dropna()
            
            ax.plot(liquidity.index, liquidity.values, 'b-', linewidth=2, label='System Liquidity')
            ax.fill_between(liquidity.index, liquidity.values, alpha=0.3, color='blue')
            
            # Add danger zones
            ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Low Liquidity')
            ax.axhline(y=0.03, color='red', linestyle='--', alpha=0.5, label='Critical')
            
            # Current value
            current = liquidity.iloc[-1]
            ax.annotate(f'Current: {current:.1%}',
                       xy=(liquidity.index[-1], current),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='red' if current < 0.05 else 'yellow', 
                                alpha=0.5))
        
        ax.set_title('System Liquidity Ratio', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Liquidity Ratio')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def plot_crisis_indicators(self, ax):
        """Plot composite crisis indicators if available."""
        
        if hasattr(self, 'indicator_df') and 'composite_score' in self.indicator_df.columns:
            scores = self.indicator_df['composite_score'].dropna()
            
            ax.plot(scores.index, scores.values, 'r-', linewidth=2)
            ax.fill_between(scores.index, scores.values, alpha=0.3, color='red')
            
            # Add thresholds
            ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Crisis Threshold')
            ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning Level')
            
            # Add recession shading
            for start, end, name in RECESSIONS:
                ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                          alpha=0.2, color='gray', label=name if start == RECESSIONS[0][0] else '')
            
            # Current value
            current = scores.iloc[-1]
            ax.annotate(f'Current: {current:.3f}',
                       xy=(scores.index[-1], current),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='red' if current > 0.6 else 'orange' if current > 0.4 else 'yellow', 
                                alpha=0.5))
        
        ax.set_title('Composite Crisis Score', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Crisis Score')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_critical_sectors(self, ax):
        """Focus on most critical sectors."""
        
        # Focus on banking and insurance
        critical = ['70', '79', '31']  # Banks, Insurance, Pension
        
        for sector in critical:
            if sector in self.leverage_series:
                dates = sorted(self.leverage_series[sector].keys())
                values = [self.leverage_series[sector][d] for d in dates]
                
                label = CRITICAL_SECTORS.get(sector, sector)
                ax.plot(dates, values, marker='o', markersize=4, 
                       label=label, linewidth=2.5)
        
        # Highlight current values
        if critical[0] in self.leverage_series and self.all_dates:
            latest_date = max(self.all_dates)
            for sector in critical:
                if sector in self.leverage_series and latest_date in self.leverage_series[sector]:
                    value = self.leverage_series[sector][latest_date]
                    ax.scatter(latest_date, value, s=100, zorder=5)
                    ax.annotate(f'{value:.1%}',
                              xy=(latest_date, value),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, fontweight='bold')
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Critical Sector Leverage (Banking/Insurance)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Leverage Ratio')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def plot_yoy_changes(self, ax):
        """Plot year-over-year changes."""
        
        if hasattr(self, 'indicator_df'):
            # Calculate YoY changes for key metrics
            metrics = ['system_leverage', 'financial_leverage', 'household_leverage']
            
            for metric in metrics:
                if metric in self.indicator_df.columns:
                    series = self.indicator_df[metric].dropna()
                    
                    # Calculate YoY change
                    yoy = series.pct_change(periods=4) * 100  # Quarterly data, 4 periods = 1 year
                    
                    if len(yoy) > 0:
                        ax.plot(yoy.index, yoy.values, label=metric.replace('_', ' ').title())
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-20, color='green', linestyle='--', alpha=0.5)
            
            ax.set_title('Year-over-Year Changes (%)', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('YoY Change (%)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def plot_risk_dashboard(self, ax):
        """Create risk assessment dashboard."""
        
        ax.axis('off')
        
        # Create risk assessment text
        assessment = "CURRENT RISK ASSESSMENT\n" + "="*40 + "\n\n"
        
        # Check latest values
        if self.all_dates:
            latest_date = max(self.all_dates)
            assessment += f"Date: {latest_date.strftime('%Y-%m-%d')}\n\n"
            
            # Get latest leverage values
            critical_leverages = []
            for sector in ['70', '79', '31', '10', '15']:
                if sector in self.leverage_series and latest_date in self.leverage_series[sector]:
                    leverage = self.leverage_series[sector][latest_date]
                    name = CRITICAL_SECTORS.get(sector, sector)
                    critical_leverages.append((name, leverage))
            
            if critical_leverages:
                assessment += "LEVERAGE BY SECTOR:\n"
                for name, leverage in critical_leverages:
                    emoji = "🔴" if leverage > 0.5 else "🟠" if leverage > 0.3 else "🟡" if leverage > 0.2 else "🟢"
                    assessment += f"  {emoji} {name}: {leverage:.1%}\n"
            
            # System metrics
            if latest_date in self.liquidity_series:
                metrics = self.liquidity_series[latest_date]
                
                assessment += f"\n\nSYSTEM METRICS:\n"
                if 'avg_leverage' in metrics:
                    avg_lev = metrics['avg_leverage']
                    emoji = "🔴" if avg_lev > 0.5 else "🟠" if avg_lev > 0.3 else "🟢"
                    assessment += f"  {emoji} Average Leverage: {avg_lev:.1%}\n"
                
                if 'liquidity' in metrics:
                    liq = metrics['liquidity']
                    emoji = "🔴" if liq < 0.05 else "🟠" if liq < 0.10 else "🟢"
                    assessment += f"  {emoji} System Liquidity: {liq:.1%}\n"
            
            # Trend assessment
            assessment += "\n\nTREND ANALYSIS:\n"
            
            # Check if leverage is increasing
            if len(self.all_dates) > 4:
                recent_dates = sorted(self.all_dates)[-4:]
                
                # Average leverage trend
                recent_avg_leverages = []
                for date in recent_dates:
                    if date in self.liquidity_series and 'avg_leverage' in self.liquidity_series[date]:
                        recent_avg_leverages.append(self.liquidity_series[date]['avg_leverage'])
                
                if len(recent_avg_leverages) >= 2:
                    if recent_avg_leverages[-1] > recent_avg_leverages[0]:
                        assessment += "  📈 Leverage: INCREASING\n"
                    else:
                        assessment += "  📉 Leverage: DECREASING\n"
            
            # Overall risk level
            assessment += "\n\nOVERALL RISK LEVEL:\n"
            
            # Determine risk based on multiple factors
            risk_score = 0
            risk_factors = []
            
            if critical_leverages:
                avg_critical_leverage = np.mean([lev for _, lev in critical_leverages])
                if avg_critical_leverage > 0.5:
                    risk_score += 3
                    risk_factors.append("Extreme leverage")
                elif avg_critical_leverage > 0.3:
                    risk_score += 2
                    risk_factors.append("High leverage")
            
            if latest_date in self.liquidity_series:
                if 'liquidity' in self.liquidity_series[latest_date]:
                    if self.liquidity_series[latest_date]['liquidity'] < 0.05:
                        risk_score += 2
                        risk_factors.append("Low liquidity")
            
            if risk_score >= 4:
                assessment += "  🔴🔴🔴 CRITICAL RISK\n"
            elif risk_score >= 2:
                assessment += "  🟠🟠 HIGH RISK\n"
            else:
                assessment += "  🟡 MODERATE RISK\n"
            
            if risk_factors:
                assessment += f"  Factors: {', '.join(risk_factors)}\n"
        
        ax.text(0.05, 0.95, assessment, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    def generate_report(self):
        """Generate comprehensive time series report."""
        report_path = self.output_dir / 'time_series_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TIME SERIES ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: {self.start_year} - {self.end_year}\n")
            f.write("="*70 + "\n\n")
            
            # Data coverage
            f.write("DATA COVERAGE\n")
            f.write("-"*40 + "\n")
            f.write(f"Quarters analyzed: {len(self.all_dates)}\n")
            if self.all_dates:
                f.write(f"First quarter: {min(self.all_dates).strftime('%Y-%m-%d')}\n")
                f.write(f"Last quarter: {max(self.all_dates).strftime('%Y-%m-%d')}\n")
            
            # Trend analysis
            f.write("\n\nTREND ANALYSIS\n")
            f.write("-"*40 + "\n")
            
            # Leverage trends
            for sector in ['10', '15', '70', '79']:
                if sector in self.leverage_series:
                    name = CRITICAL_SECTORS.get(sector, sector)
                    dates = sorted(self.leverage_series[sector].keys())
                    
                    if len(dates) > 1:
                        first_val = self.leverage_series[sector][dates[0]]
                        last_val = self.leverage_series[sector][dates[-1]]
                        change = (last_val - first_val) / first_val * 100 if first_val > 0 else 0
                        
                        f.write(f"\n{name}:\n")
                        f.write(f"  Start ({dates[0].strftime('%Y')}): {first_val:.1%}\n")
                        f.write(f"  End ({dates[-1].strftime('%Y')}): {last_val:.1%}\n")
                        f.write(f"  Change: {change:+.1f}%\n")
            
            # Current status
            if self.all_dates:
                latest_date = max(self.all_dates)
                
                f.write("\n\nCURRENT STATUS\n")
                f.write("-"*40 + "\n")
                f.write(f"Date: {latest_date.strftime('%Y-%m-%d')}\n")
                
                # Critical sectors
                f.write("\nCritical Sector Leverage:\n")
                for sector in ['70', '79', '31', '10', '15']:
                    if sector in self.leverage_series and latest_date in self.leverage_series[sector]:
                        name = CRITICAL_SECTORS.get(sector, sector)
                        leverage = self.leverage_series[sector][latest_date]
                        status = "CRITICAL" if leverage > 0.5 else "HIGH" if leverage > 0.3 else "MODERATE"
                        f.write(f"  {name}: {leverage:.1%} ({status})\n")
            
            # Risk assessment
            f.write("\n\nRISK ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            # Count high-risk sectors
            if self.all_dates and self.leverage_series:
                latest_date = max(self.all_dates)
                high_risk_count = 0
                critical_count = 0
                
                for sector in self.leverage_series:
                    if latest_date in self.leverage_series[sector]:
                        leverage = self.leverage_series[sector][latest_date]
                        if leverage > 0.5:
                            critical_count += 1
                        elif leverage > 0.3:
                            high_risk_count += 1
                
                f.write(f"Critical Risk Sectors (>50% leverage): {critical_count}\n")
                f.write(f"High Risk Sectors (>30% leverage): {high_risk_count}\n")
                
                if critical_count > 2:
                    f.write("\n⚠️ MULTIPLE CRITICAL SECTORS - SYSTEMIC CRISIS LIKELY\n")
                elif high_risk_count > 5:
                    f.write("\n⚠️ WIDESPREAD HIGH RISK - SYSTEM VULNERABLE\n")
        
        print(f"✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run complete time series analysis."""
        print("\n" + "="*70)
        print("RUNNING TIME SERIES ANALYSIS")
        print("="*70)
        
        # Step 1: Collect historical data
        quarters = self.collect_historical_data()
        
        # Step 2: Create visualizations
        self.create_time_series_visualizations()
        
        # Step 3: Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'quarters_analyzed': len(quarters),
            'date_range': (min(self.all_dates), max(self.all_dates)) if self.all_dates else None,
            'leverage_series': self.leverage_series,
            'liquidity_series': self.liquidity_series
        }


def main():
    """Run time series tracking analysis."""
    
    # Initialize tracker
    tracker = TimeSeriesTracker(start_year=2000, end_year=2025)
    
    # Run analysis
    results = tracker.run_complete_analysis()
    
    # Executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY - TIME SERIES ANALYSIS")
    print("="*70)
    
    if results['date_range']:
        start_date, end_date = results['date_range']
        print(f"\n📅 Period Analyzed: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Quarters: {results['quarters_analyzed']}")
        
        # Check latest leverage
        if tracker.leverage_series:
            latest_date = max(tracker.all_dates)
            
            print(f"\n🔍 Latest Leverage Readings ({latest_date.strftime('%Y-%m-%d')}):")
            
            critical_sectors = ['70', '79', '31', '10', '15']
            for sector in critical_sectors:
                if sector in tracker.leverage_series and latest_date in tracker.leverage_series[sector]:
                    name = CRITICAL_SECTORS.get(sector, sector)
                    leverage = tracker.leverage_series[sector][latest_date]
                    
                    if leverage > 0.5:
                        emoji = "🔴"
                        status = "CRITICAL"
                    elif leverage > 0.3:
                        emoji = "🟠"
                        status = "HIGH"
                    else:
                        emoji = "🟡"
                        status = "MODERATE"
                    
                    print(f"  {emoji} {name}: {leverage:.1%} ({status})")
            
            # Average leverage
            if latest_date in tracker.liquidity_series:
                metrics = tracker.liquidity_series[latest_date]
                if 'avg_leverage' in metrics:
                    avg_lev = metrics['avg_leverage']
                    print(f"\n📈 System Average Leverage: {avg_lev:.1%}")
                    
                    if avg_lev > 0.5:
                        print("  ⚠️ EXTREME LEVERAGE - CRISIS IMMINENT")
                    elif avg_lev > 0.3:
                        print("  ⚠️ HIGH LEVERAGE - SYSTEM AT RISK")
    
    return tracker


if __name__ == "__main__":
    tracker = main()
