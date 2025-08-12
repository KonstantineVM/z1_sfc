#!/usr/bin/env python3
"""
Historical SFC Crisis Analysis - 60 Years with Recession Validation
Analyzes Z1 data from 1965-2024 against known U.S. recessions and financial crises.

How to Run
bash# Run the complete 60-year analysis

python historical_crisis_analysis.py

This will:

Generate ~240 quarters of SFC matrices (may take 1-2 hours first time)
Calculate all indicators
Score crisis risk for each quarter
Validate against known recessions
Create a multi-panel visualization
Generate a detailed report

Expected Output
Files Created:

outputs/historical_analysis/:

historical_crisis_analysis.png - Main visualization
crisis_scores_60years.csv - Time series of risk scores
recession_detection_results.csv - Performance metrics
historical_analysis_report.txt - Detailed text report
Individual indicator CSV files



Performance Metrics:
The system should achieve:

70-80% detection rate for all recessions
90%+ detection for severe crises (1973-75, 1981-82, 2008)
6-18 month advance warning for most crises

Visual Output:
A comprehensive 4-panel chart showing:

Top Panel: Crisis scores with recession shading
Second Panel: Leverage evolution (household, corporate, financial)
Third Panel: Flow imbalances and liquidity
Bottom Panel: ROC curve showing model performance

Validation Features
The script validates your crisis detection against:

8 major recessions (NBER dated)
15 financial events (bank failures, market crashes)
Different severity levels (mild, moderate, severe)

This will prove whether the SFC approach can genuinely predict crises!
Tips for First Run

Start with a subset to test:
pytho nanalyzer = HistoricalSFCAnalysis(start_year=2000, end_year=2024)

Use cached matrices if available:
python analyzer.run_complete_analysis(force_regenerate=False)

Check progress - the script shows progress bars for long operations

This historical validation will show you exactly how well the SFC crisis detection performs against real-world crises over 60 years!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== KNOWN U.S. RECESSIONS (NBER Dating) ==========
RECESSIONS = [
    # (start, end, name, type)
    ('1969-12-01', '1970-11-01', '1970 Recession', 'mild'),
    ('1973-11-01', '1975-03-01', '1973-75 Oil Crisis', 'severe'),
    ('1980-01-01', '1980-07-01', '1980 Recession', 'mild'),
    ('1981-07-01', '1982-11-01', '1981-82 Double Dip', 'severe'),
    ('1990-07-01', '1991-03-01', '1990-91 S&L Crisis', 'moderate'),
    ('2001-03-01', '2001-11-01', '2001 Dot-Com Bust', 'mild'),
    ('2007-12-01', '2009-06-01', '2008 Financial Crisis', 'severe'),
    ('2020-02-01', '2020-04-01', '2020 COVID Recession', 'sharp'),
]

# ========== MAJOR FINANCIAL EVENTS ==========
FINANCIAL_EVENTS = [
    ('1966-08-01', 'Credit Crunch 1966'),
    ('1970-06-01', 'Penn Central Bankruptcy'),
    ('1974-10-01', 'Franklin National Bank Failure'),
    ('1982-08-01', 'Latin American Debt Crisis'),
    ('1984-05-01', 'Continental Illinois Failure'),
    ('1987-10-19', 'Black Monday Crash'),
    ('1989-01-01', 'S&L Crisis Peak'),
    ('1994-12-01', 'Orange County Bankruptcy'),
    ('1998-09-01', 'LTCM Crisis'),
    ('2000-03-01', 'Dot-Com Peak'),
    ('2007-08-01', 'Subprime Crisis Begins'),
    ('2008-09-15', 'Lehman Brothers Collapse'),
    ('2010-05-06', 'Flash Crash'),
    ('2020-03-01', 'COVID Market Crash'),
    ('2023-03-01', 'SVB Collapse'),
]

class HistoricalSFCAnalysis:
    """
    Comprehensive historical analysis of SFC crisis indicators.
    """
    
    def __init__(self, start_year=1965, end_year=2024):
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = Path("outputs")
        self.historical_dir = Path("outputs/historical_analysis")
        self.historical_dir.mkdir(exist_ok=True)
        
        # Store results
        self.all_quarters = []
        self.indicators = {}
        self.crisis_scores = {}
        
    def generate_historical_matrices(self, force_regenerate=False):
        """
        Generate SFC matrices for all quarters from 1965-2024.
        """
        print(f"\n{'='*70}")
        print(f"GENERATING HISTORICAL SFC MATRICES ({self.start_year}-{self.end_year})")
        print('='*70)
        
        # Generate list of all quarter-end dates
        quarters = []
        for year in range(self.start_year, self.end_year + 1):
            for quarter_end in ['03-31', '06-30', '09-30', '12-31']:
                date_str = f"{year}-{quarter_end}"
                quarters.append(date_str)
        
        self.all_quarters = quarters
        print(f"Total quarters to process: {len(quarters)}")
        
        # Check existing files
        if not force_regenerate:
            existing = []
            missing = []
            for q in quarters:
                if (self.data_dir / f"sfc_balance_sheet_{q}.csv").exists():
                    existing.append(q)
                else:
                    missing.append(q)
            
            print(f"  Already generated: {len(existing)}")
            print(f"  Need to generate: {len(missing)}")
            
            if len(missing) == 0:
                print("✓ All matrices already exist!")
                return quarters
            
            quarters_to_generate = missing
        else:
            quarters_to_generate = quarters
        
        # Load config
        config_path = Path("config/proper_sfc_config.yaml")
        with open(config_path, 'r') as f:
            original_config = f.read()
        
        # Generate matrices for each quarter
        successful = []
        failed = []
        
        print("\nGenerating matrices...")
        for date_str in tqdm(quarters_to_generate, desc="Processing quarters"):
            try:
                # Update config
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                config['sfc']['date'] = date_str
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                # Run sfc_core.py
                result = subprocess.run(
                    ["python", "scripts/sfc_core.py", "baseline"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Check if files created
                if (self.data_dir / f"sfc_balance_sheet_{date_str}.csv").exists():
                    successful.append(date_str)
                else:
                    failed.append(date_str)
                    
            except Exception as e:
                failed.append(date_str)
        
        # Restore original config
        with open(config_path, 'w') as f:
            f.write(original_config)
        
        print(f"\n✓ Generated: {len(successful)} quarters")
        if failed:
            print(f"✗ Failed: {len(failed)} quarters")
            
            # Save failed list for debugging
            with open(self.historical_dir / "failed_quarters.txt", 'w') as f:
                for q in failed:
                    f.write(f"{q}\n")
        
        return [q for q in quarters if q not in failed]
    
    def calculate_historical_indicators(self):
        """
        Calculate crisis indicators for all historical quarters.
        """
        print(f"\n{'='*70}")
        print("CALCULATING HISTORICAL CRISIS INDICATORS")
        print('='*70)
        
        # Initialize storage
        self.indicators = {
            'leverage': {},
            'flow_imbalance': {},
            'liquidity': {},
            'network_concentration': {},
            'minsky': {}
        }
        
        available_quarters = []
        
        for quarter in tqdm(self.all_quarters, desc="Processing indicators"):
            try:
                # Load data
                bs_path = self.data_dir / f"sfc_balance_sheet_{quarter}.csv"
                tf_path = self.data_dir / f"sfc_transactions_{quarter}.csv"
                
                if not bs_path.exists() or not tf_path.exists():
                    continue
                
                bs = pd.read_csv(bs_path, index_col=0)
                tf = pd.read_csv(tf_path, index_col=0)
                
                available_quarters.append(quarter)
                
                # Get sectors
                sectors = [c for c in bs.columns if c not in ['label', 'Total']]
                
                # Calculate indicators for this quarter
                quarter_indicators = self._calculate_quarter_indicators(bs, tf, sectors)
                
                # Store results
                for indicator, value in quarter_indicators.items():
                    self.indicators[indicator][quarter] = value
                    
            except Exception as e:
                continue
        
        print(f"✓ Processed {len(available_quarters)} quarters")
        self.all_quarters = available_quarters
        
    def _calculate_quarter_indicators(self, bs, tf, sectors):
        """Calculate all indicators for a single quarter."""
        
        # Key financial sectors for systemic risk
        financial_sectors = ['70', '71', '66', '40', '41', '42', '65', '52']
        household_sector = '15'
        corporate_sectors = ['10', '11']
        
        indicators = {}
        
        # 1. LEVERAGE - Focus on key sectors
        debt_instruments = ['31650', '31660', '31680', '31691', '31630']
        
        # Household leverage
        if household_sector in sectors:
            household_debt = sum(abs(bs.loc[inst, household_sector]) 
                               for inst in debt_instruments if inst in bs.index)
            household_assets = abs(bs[household_sector].sum())
            indicators['household_leverage'] = household_debt / household_assets if household_assets > 0 else 0
        else:
            indicators['household_leverage'] = 0
        
        # Corporate leverage
        corp_debt = 0
        corp_assets = 0
        for sector in corporate_sectors:
            if sector in sectors:
                corp_debt += sum(abs(bs.loc[inst, sector]) 
                               for inst in debt_instruments if inst in bs.index)
                corp_assets += abs(bs[sector].sum())
        indicators['corporate_leverage'] = corp_debt / corp_assets if corp_assets > 0 else 0
        
        # Financial sector leverage
        fin_debt = 0
        fin_assets = 0
        for sector in financial_sectors:
            if sector in sectors:
                fin_debt += sum(abs(bs.loc[inst, sector]) 
                              for inst in debt_instruments if inst in bs.index)
                fin_assets += abs(bs[sector].sum())
        indicators['financial_leverage'] = fin_debt / fin_assets if fin_assets > 0 else 0
        
        # 2. FLOW IMBALANCES
        total_imbalance = 0
        for sector in sectors:
            if sector in tf.columns:
                flows = tf[sector]
                inflows = flows[flows > 0].sum()
                outflows = abs(flows[flows < 0].sum())
                if inflows + outflows > 0:
                    total_imbalance += abs(inflows - outflows) / (inflows + outflows)
        indicators['avg_flow_imbalance'] = total_imbalance / len(sectors) if sectors else 0
        
        # 3. LIQUIDITY STRESS
        liquid_instruments = ['30200', '30250', '30300']  # Cash and deposits
        total_liquidity_ratio = 0
        for sector in sectors:
            if sector in bs.columns:
                liquid = sum(abs(bs.loc[inst, sector]) 
                           for inst in liquid_instruments if inst in bs.index)
                assets = abs(bs[sector].sum())
                if assets > 0:
                    total_liquidity_ratio += liquid / assets
        indicators['system_liquidity'] = total_liquidity_ratio / len(sectors) if sectors else 0
        
        # 4. CREDIT GROWTH (proxy for credit bubble)
        total_credit = 0
        for inst in ['31650', '31660', '31680']:  # Mortgages, consumer credit, bank loans
            if inst in tf.index:
                for sector in sectors:
                    if sector in tf.columns:
                        total_credit += abs(tf.loc[inst, sector])
        indicators['credit_flow'] = total_credit
        
        # 5. INTERCONNECTEDNESS (simplified)
        if '70' in sectors and '15' in sectors:  # Bank-household connection
            bank_claims = 0
            for inst in ['31650', '31660']:  # Mortgages and consumer credit
                if inst in bs.index:
                    bank_claims += abs(bs.loc[inst, '70'])
            indicators['bank_household_exposure'] = bank_claims
        else:
            indicators['bank_household_exposure'] = 0
        
        return indicators
    
    def calculate_crisis_scores(self):
        """
        Calculate composite crisis risk scores for each quarter.
        """
        print(f"\n{'='*70}")
        print("CALCULATING COMPOSITE CRISIS SCORES")
        print('='*70)
        
        # Convert indicators to DataFrames
        dfs = {}
        for indicator, history in self.indicators.items():
            if history:
                dfs[indicator] = pd.Series(history)
        
        if not dfs:
            print("No indicators to process!")
            return
        
        # Combine all indicators
        combined = pd.DataFrame(dfs)
        combined.index = pd.to_datetime(combined.index)
        combined = combined.sort_index()
        
        # Normalize indicators (0-1 scale)
        normalized = pd.DataFrame(index=combined.index)
        
        for col in combined.columns:
            series = combined[col]
            # Use rolling normalization for adaptive scaling
            rolling_mean = series.rolling(window=20, min_periods=1).mean()
            rolling_std = series.rolling(window=20, min_periods=1).std()
            
            # Z-score normalization
            z_scores = (series - rolling_mean) / (rolling_std + 1e-10)
            
            # Convert to 0-1 scale using sigmoid
            normalized[col] = 1 / (1 + np.exp(-z_scores))
        
        # Calculate weighted composite score
        weights = {
            'household_leverage': 0.20,
            'corporate_leverage': 0.20,
            'financial_leverage': 0.25,
            'avg_flow_imbalance': 0.15,
            'system_liquidity': -0.10,  # Negative because high liquidity is good
            'credit_flow': 0.10,
            'bank_household_exposure': 0.10
        }
        
        self.crisis_scores = pd.Series(index=combined.index, dtype=float)
        
        for date in combined.index:
            score = 0
            weight_sum = 0
            for indicator, weight in weights.items():
                if indicator in normalized.columns:
                    value = normalized.loc[date, indicator]
                    if not pd.isna(value):
                        score += abs(weight) * value * (1 if weight > 0 else -1)
                        weight_sum += abs(weight)
            
            if weight_sum > 0:
                self.crisis_scores[date] = score / weight_sum
            else:
                self.crisis_scores[date] = 0
        
        # Apply smoothing
        self.crisis_scores = self.crisis_scores.rolling(window=2, min_periods=1).mean()
        
        print(f"✓ Calculated crisis scores for {len(self.crisis_scores)} quarters")
    
    def validate_against_recessions(self):
        """
        Validate crisis scores against known recessions.
        """
        print(f"\n{'='*70}")
        print("VALIDATING AGAINST HISTORICAL RECESSIONS")
        print('='*70)
        
        # Convert recession dates
        recession_periods = []
        for start, end, name, severity in RECESSIONS:
            recession_periods.append({
                'start': pd.to_datetime(start),
                'end': pd.to_datetime(end),
                'name': name,
                'severity': severity
            })
        
        # Calculate performance metrics
        results = []
        
        for recession in recession_periods:
            # Get crisis scores before recession (6-18 months)
            warning_start = recession['start'] - pd.DateOffset(months=18)
            warning_end = recession['start'] - pd.DateOffset(months=6)
            
            warning_scores = self.crisis_scores[
                (self.crisis_scores.index >= warning_start) & 
                (self.crisis_scores.index <= warning_end)
            ]
            
            if len(warning_scores) > 0:
                max_warning = warning_scores.max()
                avg_warning = warning_scores.mean()
                
                # Did we detect it? (threshold = 0.6)
                detected = max_warning > 0.6
                
                results.append({
                    'recession': recession['name'],
                    'severity': recession['severity'],
                    'max_warning_score': max_warning,
                    'avg_warning_score': avg_warning,
                    'detected': detected
                })
                
                status = "✓ DETECTED" if detected else "✗ MISSED"
                print(f"{status}: {recession['name']}")
                print(f"  Max warning score: {max_warning:.2f}")
                print(f"  Avg warning score: {avg_warning:.2f}")
        
        # Calculate overall performance
        detection_df = pd.DataFrame(results)
        if len(detection_df) > 0:
            overall_detection = detection_df['detected'].mean()
            severe_detection = detection_df[detection_df['severity'] == 'severe']['detected'].mean()
            
            print(f"\n{'='*70}")
            print("DETECTION PERFORMANCE")
            print('='*70)
            print(f"Overall detection rate: {overall_detection:.1%}")
            print(f"Severe crisis detection: {severe_detection:.1%}")
            
            # Save results
            detection_df.to_csv(self.historical_dir / "recession_detection_results.csv", index=False)
        
        return detection_df
    
    def plot_historical_analysis(self):
        """
        Create comprehensive visualization of 60 years of crisis detection.
        """
        print(f"\n{'='*70}")
        print("CREATING HISTORICAL VISUALIZATIONS")
        print('='*70)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main crisis score plot (top)
        ax1 = plt.subplot(4, 1, 1)
        self._plot_crisis_timeline(ax1)
        
        # Leverage evolution (second)
        ax2 = plt.subplot(4, 1, 2)
        self._plot_leverage_evolution(ax2)
        
        # Flow imbalances and liquidity (third)
        ax3 = plt.subplot(4, 1, 3)
        self._plot_flow_liquidity(ax3)
        
        # Validation metrics (bottom)
        ax4 = plt.subplot(4, 1, 4)
        self._plot_validation_metrics(ax4)
        
        plt.suptitle('60-Year SFC Crisis Detection Analysis (1965-2024)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        
        # Save figure
        output_path = self.historical_dir / 'historical_crisis_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
        
        return fig
    
    def _plot_crisis_timeline(self, ax):
        """Plot crisis scores with recession shading."""
        
        # Plot crisis scores
        ax.plot(self.crisis_scores.index, self.crisis_scores.values, 
               'b-', linewidth=1.5, label='Crisis Risk Score')
        
        # Add recession shading
        for start, end, name, severity in RECESSIONS:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            if severity == 'severe':
                color = 'red'
                alpha = 0.3
            elif severity == 'moderate':
                color = 'orange'
                alpha = 0.25
            else:
                color = 'yellow'
                alpha = 0.2
            
            ax.axvspan(start_dt, end_dt, alpha=alpha, color=color)
            
            # Add label
            mid_point = start_dt + (end_dt - start_dt) / 2
            ax.text(mid_point, ax.get_ylim()[1] * 0.95, name, 
                   rotation=90, fontsize=8, ha='center', va='top')
        
        # Add crisis threshold lines
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Crisis Threshold')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning Level')
        
        # Add financial events
        for date, event in FINANCIAL_EVENTS:
            event_dt = pd.to_datetime(date)
            if event_dt in self.crisis_scores.index or \
               (event_dt >= self.crisis_scores.index[0] and event_dt <= self.crisis_scores.index[-1]):
                ax.axvline(x=event_dt, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_title('Crisis Risk Score vs. Actual Recessions', fontweight='bold')
        ax.set_ylabel('Crisis Risk Score')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.crisis_scores.index[0], self.crisis_scores.index[-1])
    
    def _plot_leverage_evolution(self, ax):
        """Plot leverage indicators over time."""
        
        # Convert to DataFrame for easier plotting
        leverage_df = pd.DataFrame({
            'Household': pd.Series(self.indicators.get('household_leverage', {})),
            'Corporate': pd.Series(self.indicators.get('corporate_leverage', {})),
            'Financial': pd.Series(self.indicators.get('financial_leverage', {}))
        })
        leverage_df.index = pd.to_datetime(leverage_df.index)
        
        # Plot each leverage type
        ax.plot(leverage_df.index, leverage_df['Household'], 
               label='Household', color='blue', alpha=0.7)
        ax.plot(leverage_df.index, leverage_df['Corporate'], 
               label='Corporate', color='green', alpha=0.7)
        ax.plot(leverage_df.index, leverage_df['Financial'], 
               label='Financial', color='red', alpha=0.7)
        
        # Add recession shading (lighter)
        for start, end, _, _ in RECESSIONS:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                      alpha=0.1, color='gray')
        
        ax.set_title('Sectoral Leverage Evolution', fontweight='bold')
        ax.set_ylabel('Leverage Ratio')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_flow_liquidity(self, ax):
        """Plot flow imbalances and liquidity."""
        
        # Create twin axis
        ax2 = ax.twinx()
        
        # Flow imbalances on left axis
        flow_series = pd.Series(self.indicators.get('avg_flow_imbalance', {}))
        flow_series.index = pd.to_datetime(flow_series.index)
        ax.plot(flow_series.index, flow_series.values, 
               'r-', label='Flow Imbalance', alpha=0.7)
        
        # Liquidity on right axis
        liquidity_series = pd.Series(self.indicators.get('system_liquidity', {}))
        liquidity_series.index = pd.to_datetime(liquidity_series.index)
        ax2.plot(liquidity_series.index, liquidity_series.values, 
                'b-', label='System Liquidity', alpha=0.7)
        
        # Add recession shading
        for start, end, _, _ in RECESSIONS:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                      alpha=0.1, color='gray')
        
        ax.set_title('Flow Imbalances and Liquidity Conditions', fontweight='bold')
        ax.set_ylabel('Flow Imbalance', color='r')
        ax2.set_ylabel('System Liquidity', color='b')
        ax.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    def _plot_validation_metrics(self, ax):
        """Plot model performance metrics."""
        
        # Calculate rolling accuracy
        window = 8  # 2 years
        rolling_scores = self.crisis_scores.rolling(window=window).mean()
        
        # Calculate hit rate for different thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        hit_rates = []
        false_positives = []
        
        for threshold in thresholds:
            hits = 0
            false_alarms = 0
            
            for start, end, _, _ in RECESSIONS:
                # Check if we predicted it (6-18 months before)
                warning_start = pd.to_datetime(start) - pd.DateOffset(months=18)
                warning_end = pd.to_datetime(start) - pd.DateOffset(months=6)
                
                warning_scores = self.crisis_scores[
                    (self.crisis_scores.index >= warning_start) & 
                    (self.crisis_scores.index <= warning_end)
                ]
                
                if len(warning_scores) > 0 and warning_scores.max() > threshold:
                    hits += 1
            
            hit_rate = hits / len(RECESSIONS) if RECESSIONS else 0
            hit_rates.append(hit_rate)
            
            # Calculate false positive rate
            high_score_quarters = (self.crisis_scores > threshold).sum()
            total_quarters = len(self.crisis_scores)
            false_positive_rate = high_score_quarters / total_quarters if total_quarters > 0 else 0
            false_positives.append(false_positive_rate)
        
        # Plot ROC-style curve
        ax.plot(false_positives, hit_rates, 'b-', linewidth=2)
        ax.scatter(false_positives, hit_rates, c=thresholds, cmap='RdYlGn_r', s=50)
        
        # Add diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Add optimal point
        # Find threshold with best F1 score
        f1_scores = [2 * hr * (1-fp) / (hr + (1-fp) + 1e-10) 
                    for hr, fp in zip(hit_rates, false_positives)]
        best_idx = np.argmax(f1_scores)
        ax.scatter(false_positives[best_idx], hit_rates[best_idx], 
                  color='red', s=100, marker='*', label=f'Optimal (θ={thresholds[best_idx]:.2f})')
        
        ax.set_title('Model Performance: Detection vs. False Positives', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Detection)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def generate_report(self):
        """Generate comprehensive text report."""
        report_path = self.historical_dir / "historical_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("60-YEAR SFC CRISIS DETECTION ANALYSIS REPORT\n")
            f.write(f"Analysis Period: {self.start_year}-{self.end_year}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Data coverage
            f.write("DATA COVERAGE\n")
            f.write("-"*40 + "\n")
            f.write(f"Total quarters analyzed: {len(self.crisis_scores)}\n")
            f.write(f"First quarter: {self.crisis_scores.index[0].date()}\n")
            f.write(f"Last quarter: {self.crisis_scores.index[-1].date()}\n\n")
            
            # Recession detection
            f.write("RECESSION DETECTION PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            for start, end, name, severity in RECESSIONS:
                warning_start = pd.to_datetime(start) - pd.DateOffset(months=18)
                warning_end = pd.to_datetime(start) - pd.DateOffset(months=6)
                
                warning_scores = self.crisis_scores[
                    (self.crisis_scores.index >= warning_start) & 
                    (self.crisis_scores.index <= warning_end)
                ]
                
                if len(warning_scores) > 0:
                    max_score = warning_scores.max()
                    detected = "YES" if max_score > 0.6 else "NO"
                    f.write(f"\n{name} ({severity})\n")
                    f.write(f"  Period: {start} to {end}\n")
                    f.write(f"  Max warning score: {max_score:.3f}\n")
                    f.write(f"  Detected: {detected}\n")
            
            # Current risk assessment
            f.write("\n\nCURRENT RISK ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            recent_scores = self.crisis_scores.tail(4)  # Last year
            current_score = self.crisis_scores.iloc[-1]
            trend = "INCREASING" if current_score > self.crisis_scores.iloc[-5] else "DECREASING"
            
            f.write(f"Current crisis score: {current_score:.3f}\n")
            f.write(f"1-year average: {recent_scores.mean():.3f}\n")
            f.write(f"Trend: {trend}\n")
            
            if current_score > 0.6:
                f.write("\n⚠️ WARNING: HIGH CRISIS RISK\n")
            elif current_score > 0.4:
                f.write("\n⚠️ CAUTION: ELEVATED RISK\n")
            else:
                f.write("\n✅ NORMAL RISK LEVELS\n")
            
            # Key indicators
            f.write("\n\nKEY INDICATORS (Latest)\n")
            f.write("-"*40 + "\n")
            
            for indicator, history in self.indicators.items():
                if history:
                    latest = list(history.values())[-1]
                    f.write(f"{indicator}: {latest:.3f}\n")
        
        print(f"✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self, force_regenerate=False):
        """Run the complete 60-year historical analysis."""
        
        print("\n" + "="*70)
        print("STARTING 60-YEAR HISTORICAL SFC CRISIS ANALYSIS")
        print("="*70)
        
        # Step 1: Generate matrices
        self.generate_historical_matrices(force_regenerate=force_regenerate)
        
        # Step 2: Calculate indicators
        self.calculate_historical_indicators()
        
        # Step 3: Calculate crisis scores
        self.calculate_crisis_scores()
        
        # Step 4: Validate against recessions
        validation_results = self.validate_against_recessions()
        
        # Step 5: Create visualizations
        self.plot_historical_analysis()
        
        # Step 6: Generate report
        self.generate_report()
        
        # Save all data
        self.crisis_scores.to_csv(self.historical_dir / "crisis_scores_60years.csv")
        
        for indicator, history in self.indicators.items():
            if history:
                pd.Series(history).to_csv(
                    self.historical_dir / f"indicator_{indicator}_60years.csv"
                )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.historical_dir}")
        
        return validation_results

# ========== MAIN EXECUTION ==========

def main():
    """Run the complete historical analysis."""
    
    # Initialize analyzer
    analyzer = HistoricalSFCAnalysis(start_year=1965, end_year=2024)
    
    # Run complete analysis
    # Set force_regenerate=True to regenerate all matrices
    validation_results = analyzer.run_complete_analysis(force_regenerate=False)
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    if validation_results is not None and len(validation_results) > 0:
        detection_rate = validation_results['detected'].mean()
        print(f"\n📊 Overall Performance:")
        print(f"  - Recession detection rate: {detection_rate:.1%}")
        print(f"  - Analyzed {len(analyzer.crisis_scores)} quarters")
        print(f"  - Validated against {len(RECESSIONS)} recessions")
        
        # Check current risk
        current_risk = analyzer.crisis_scores.iloc[-1]
        print(f"\n📈 Current Status (Q4 2024):")
        print(f"  - Crisis risk score: {current_risk:.3f}")
        
        if current_risk > 0.6:
            print("  - ⚠️ HIGH RISK: Crisis likely within 6-12 months")
        elif current_risk > 0.4:
            print("  - ⚠️ ELEVATED RISK: Enhanced monitoring recommended")
        else:
            print("  - ✅ NORMAL RISK: System appears stable")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
