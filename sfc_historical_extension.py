#!/usr/bin/env python3
"""
SFC Historical Validation Extension
====================================
Adds historical validation, comparison, and visualization to the optimized analyzer.
Works alongside the existing analyzer without modifying it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ========== KNOWN U.S. RECESSIONS (NBER Dating) ==========
RECESSIONS = [
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
    ('1987-10-19', 'Black Monday'),
    ('1989-01-01', 'S&L Crisis Peak'),
    ('1998-09-01', 'LTCM Crisis'),
    ('2000-03-01', 'Dot-Com Peak'),
    ('2007-08-01', 'Subprime Begins'),
    ('2008-09-15', 'Lehman Collapse'),
    ('2020-03-01', 'COVID Crash'),
    ('2023-03-01', 'SVB Collapse'),
]


class HistoricalValidation:
    """
    Adds historical validation capabilities to existing SFC analysis.
    """
    
    def __init__(self, indicators_file: str = "outputs/optimized_complete_indicators.csv"):
        """
        Initialize with existing indicators from optimized analyzer.
        
        Args:
            indicators_file: Path to CSV with calculated indicators
        """
        self.indicators_file = Path(indicators_file)
        self.output_dir = Path("outputs/historical_validation")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load existing indicators
        self.load_indicators()
        
        # Storage for analysis results
        self.validation_results = {}
        self.optimal_threshold = None
        self.historical_comparison = {}
        
    def load_indicators(self):
        """Load indicators from optimized analyzer output."""
        if not self.indicators_file.exists():
            raise FileNotFoundError(f"Indicators file not found: {self.indicators_file}")
        
        self.indicators_df = pd.read_csv(self.indicators_file, index_col=0, parse_dates=True)
        print(f"Loaded {len(self.indicators_df)} quarters of indicators")
        print(f"Date range: {self.indicators_df.index[0].date()} to {self.indicators_df.index[-1].date()}")
        
    def calculate_enhanced_crisis_score(self):
        """
        Calculate crisis score with rolling normalization and sigmoid transform.
        """
        print("\n" + "="*70)
        print("CALCULATING ENHANCED CRISIS SCORES")
        print("="*70)
        
        # Use rolling window for adaptive normalization
        window = 20  # 5 years
        
        # Normalize each indicator
        normalized = pd.DataFrame(index=self.indicators_df.index)
        
        for col in self.indicators_df.columns:
            if col == 'composite_score':
                continue
                
            series = self.indicators_df[col]
            
            # Rolling statistics
            rolling_mean = series.rolling(window=window, min_periods=4).mean()
            rolling_std = series.rolling(window=window, min_periods=4).std()
            
            # Z-score normalization
            z_scores = (series - rolling_mean) / (rolling_std + 1e-10)
            
            # Sigmoid transform to 0-1 scale
            normalized[col] = 1 / (1 + np.exp(-z_scores))
        
        # Enhanced weights based on historical importance
        weights = {
            'system_leverage': 0.25,
            'financial_leverage': 0.30,  # Most important historically
            'household_leverage': 0.15,
            'corporate_leverage': 0.15,
            'avg_flow_imbalance': 0.10,
            'system_liquidity_ratio': -0.15,  # Negative weight
            'credit_flow': 0.10,
            'net_debt_issuance': 0.05,
            'bank_household_exposure': 0.05,
        }
        
        # Calculate enhanced score
        self.enhanced_score = pd.Series(index=normalized.index, dtype=float)
        
        for date in normalized.index:
            score = 0
            weight_sum = 0
            
            for indicator, weight in weights.items():
                if indicator in normalized.columns:
                    value = normalized.loc[date, indicator]
                    if not pd.isna(value):
                        score += abs(weight) * value * (1 if weight > 0 else -1)
                        weight_sum += abs(weight)
            
            self.enhanced_score[date] = score / weight_sum if weight_sum > 0 else 0
        
        # Apply smoothing
        self.enhanced_score = self.enhanced_score.rolling(window=2, min_periods=1).mean()
        
        print(f"✓ Calculated enhanced scores for {len(self.enhanced_score)} quarters")
        
    def validate_against_recessions(self):
        """
        Validate model performance against historical recessions.
        """
        print("\n" + "="*70)
        print("VALIDATING AGAINST HISTORICAL RECESSIONS")
        print("="*70)
        
        results = []
        
        for start, end, name, severity in RECESSIONS:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            # Skip if outside our data range
            if start_dt < self.enhanced_score.index[0]:
                continue
            
            # Look for warning signals 6-18 months before
            warning_start = start_dt - pd.DateOffset(months=18)
            warning_end = start_dt - pd.DateOffset(months=6)
            
            # Get scores in warning period
            warning_scores = self.enhanced_score[
                (self.enhanced_score.index >= warning_start) & 
                (self.enhanced_score.index <= warning_end)
            ]
            
            if len(warning_scores) > 0:
                max_warning = warning_scores.max()
                avg_warning = warning_scores.mean()
                
                # Test different thresholds
                for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
                    detected = max_warning > threshold
                    
                    results.append({
                        'recession': name,
                        'severity': severity,
                        'start': start,
                        'max_warning_score': max_warning,
                        'avg_warning_score': avg_warning,
                        'threshold': threshold,
                        'detected': detected
                    })
                
                print(f"{name} ({severity})")
                print(f"  Max warning score: {max_warning:.3f}")
                print(f"  Avg warning score: {avg_warning:.3f}")
                print(f"  Detection (θ=0.6): {'✓' if max_warning > 0.6 else '✗'}")
        
        self.validation_results = pd.DataFrame(results)
        
        # Find optimal threshold
        self.find_optimal_threshold()
        
        # Save results
        self.validation_results.to_csv(self.output_dir / "recession_validation.csv", index=False)
        
        return self.validation_results
    
    def find_optimal_threshold(self):
        """
        Find optimal threshold for crisis detection.
        """
        if self.validation_results.empty:
            return
        
        # Calculate metrics for each threshold
        threshold_metrics = []
        
        for threshold in self.validation_results['threshold'].unique():
            subset = self.validation_results[self.validation_results['threshold'] == threshold]
            
            # Group by recession to avoid double counting
            grouped = subset.groupby('recession')['detected'].max()
            
            # Calculate metrics
            tp = grouped.sum()  # True positives
            fn = len(grouped) - tp  # False negatives
            
            # Calculate false positive rate (quarters above threshold)
            high_score_quarters = (self.enhanced_score > threshold).sum()
            total_quarters = len(self.enhanced_score)
            fp_rate = high_score_quarters / total_quarters if total_quarters > 0 else 0
            
            # Calculate F1 score
            precision = tp / (tp + fp_rate * total_quarters) if (tp + fp_rate * total_quarters) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_metrics.append({
                'threshold': threshold,
                'true_positives': tp,
                'false_negatives': fn,
                'false_positive_rate': fp_rate,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        metrics_df = pd.DataFrame(threshold_metrics)
        
        # Find optimal threshold (highest F1 score)
        self.optimal_threshold = metrics_df.loc[metrics_df['f1_score'].idxmax(), 'threshold']
        
        print(f"\n✓ Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"  Best F1 score: {metrics_df['f1_score'].max():.3f}")
        
        # Save metrics
        metrics_df.to_csv(self.output_dir / "threshold_metrics.csv", index=False)
        
        return self.optimal_threshold
    
    def compare_to_historical_crises(self):
        """
        Compare current conditions to pre-crisis periods.
        """
        print("\n" + "="*70)
        print("HISTORICAL CRISIS COMPARISON")
        print("="*70)
        
        current_date = self.enhanced_score.index[-1]
        current_score = self.enhanced_score.iloc[-1]
        current_indicators = self.indicators_df.iloc[-1]
        
        comparisons = []
        
        # Compare to each major crisis
        crisis_periods = [
            ('2007-12-01', '2008 Financial Crisis'),
            ('2001-03-01', '2001 Dot-Com Bust'),
            ('1990-07-01', '1990-91 S&L Crisis'),
        ]
        
        for crisis_date, crisis_name in crisis_periods:
            crisis_dt = pd.to_datetime(crisis_date)
            
            # Skip if outside our data
            if crisis_dt < self.indicators_df.index[0]:
                continue
            
            # Get indicators 12 months before crisis
            pre_crisis_dt = crisis_dt - pd.DateOffset(months=12)
            
            if pre_crisis_dt in self.indicators_df.index:
                pre_crisis = self.indicators_df.loc[pre_crisis_dt]
                
                comparison = {
                    'crisis': crisis_name,
                    'pre_crisis_date': pre_crisis_dt.date(),
                    'pre_crisis_score': self.enhanced_score.loc[pre_crisis_dt],
                    'current_score': current_score,
                    'score_ratio': current_score / self.enhanced_score.loc[pre_crisis_dt] if self.enhanced_score.loc[pre_crisis_dt] > 0 else 0
                }
                
                # Compare key indicators
                for indicator in ['system_leverage', 'financial_leverage', 'system_liquidity_ratio']:
                    if indicator in pre_crisis.index:
                        comparison[f'{indicator}_then'] = pre_crisis[indicator]
                        comparison[f'{indicator}_now'] = current_indicators[indicator]
                
                comparisons.append(comparison)
        
        self.historical_comparison = pd.DataFrame(comparisons)
        
        # Print comparison
        print(f"\nCurrent conditions vs. pre-crisis periods:")
        print(f"Current crisis score: {current_score:.3f}")
        print(f"Current date: {current_date.date()}")
        
        for _, row in self.historical_comparison.iterrows():
            print(f"\nvs. {row['crisis']} (12 months before):")
            print(f"  Crisis score then: {row['pre_crisis_score']:.3f}")
            print(f"  Score ratio (now/then): {row['score_ratio']:.2f}x")
            
            if 'system_leverage_then' in row:
                print(f"  System leverage - Then: {row['system_leverage_then']:.3f}, Now: {row['system_leverage_now']:.3f}")
            if 'financial_leverage_then' in row:
                print(f"  Financial leverage - Then: {row['financial_leverage_then']:.3f}, Now: {row['financial_leverage_now']:.3f}")
        
        # Save comparison
        self.historical_comparison.to_csv(self.output_dir / "historical_comparison.csv", index=False)
        
        return self.historical_comparison
    
    def create_comprehensive_visualization(self):
        """
        Create multi-panel visualization showing historical context.
        """
        print("\n" + "="*70)
        print("CREATING HISTORICAL VISUALIZATIONS")
        print("="*70)
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Crisis Score Timeline with Recessions
        ax1 = plt.subplot(3, 2, (1, 2))
        self._plot_crisis_timeline(ax1)
        
        # 2. Leverage Evolution
        ax2 = plt.subplot(3, 2, 3)
        self._plot_leverage_evolution(ax2)
        
        # 3. Model Performance (ROC-style)
        ax3 = plt.subplot(3, 2, 4)
        self._plot_model_performance(ax3)
        
        # 4. Historical Comparison
        ax4 = plt.subplot(3, 2, 5)
        self._plot_historical_comparison(ax4)
        
        # 5. Current Risk Dashboard
        ax5 = plt.subplot(3, 2, 6)
        self._plot_risk_dashboard(ax5)
        
        plt.suptitle('SFC Crisis Detection: Historical Validation & Current Assessment', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save
        output_path = self.output_dir / 'historical_validation_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
        
        return fig
    
    def _plot_crisis_timeline(self, ax):
        """Plot crisis scores with recession shading."""
        
        # Plot both original and enhanced scores
        ax.plot(self.indicators_df.index, self.indicators_df['composite_score'], 
               'b-', alpha=0.5, linewidth=1, label='Original Score')
        ax.plot(self.enhanced_score.index, self.enhanced_score.values, 
               'r-', linewidth=2, label='Enhanced Score')
        
        # Add recession shading
        for start, end, name, severity in RECESSIONS:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            # Skip if outside range
            if end_dt < self.enhanced_score.index[0] or start_dt > self.enhanced_score.index[-1]:
                continue
            
            color = 'red' if severity == 'severe' else 'orange' if severity == 'moderate' else 'yellow'
            alpha = 0.3 if severity == 'severe' else 0.2
            
            ax.axvspan(start_dt, end_dt, alpha=alpha, color=color)
            
            # Add label
            mid_point = start_dt + (end_dt - start_dt) / 2
            if mid_point >= self.enhanced_score.index[0] and mid_point <= self.enhanced_score.index[-1]:
                ax.text(mid_point, ax.get_ylim()[1] * 0.95, name, 
                       rotation=90, fontsize=7, ha='center', va='top')
        
        # Add threshold lines
        if self.optimal_threshold:
            ax.axhline(y=self.optimal_threshold, color='red', linestyle='--', 
                      alpha=0.7, label=f'Optimal Threshold ({self.optimal_threshold:.2f})')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Warning Level (0.6)')
        
        # Add financial events
        for date, event in FINANCIAL_EVENTS:
            event_dt = pd.to_datetime(date)
            if event_dt >= self.enhanced_score.index[0] and event_dt <= self.enhanced_score.index[-1]:
                ax.axvline(x=event_dt, color='gray', linestyle=':', alpha=0.3)
        
        # Highlight current position
        current_score = self.enhanced_score.iloc[-1]
        ax.scatter(self.enhanced_score.index[-1], current_score, 
                  color='red', s=100, marker='*', zorder=5)
        ax.text(self.enhanced_score.index[-1], current_score + 0.05, 
               f'Current\n{current_score:.3f}', ha='center', fontsize=8)
        
        ax.set_title('Crisis Risk Score vs. Historical Recessions', fontweight='bold')
        ax.set_ylabel('Crisis Risk Score')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.enhanced_score.index[0], self.enhanced_score.index[-1])
    
    def _plot_leverage_evolution(self, ax):
        """Plot leverage indicators evolution."""
        
        # Plot leverage types
        if 'household_leverage' in self.indicators_df.columns:
            ax.plot(self.indicators_df.index, self.indicators_df['household_leverage'], 
                   'b-', label='Household', alpha=0.7)
        if 'corporate_leverage' in self.indicators_df.columns:
            ax.plot(self.indicators_df.index, self.indicators_df['corporate_leverage'], 
                   'g-', label='Corporate', alpha=0.7)
        if 'financial_leverage' in self.indicators_df.columns:
            ax.plot(self.indicators_df.index, self.indicators_df['financial_leverage'], 
                   'r-', label='Financial', alpha=0.7, linewidth=2)
        if 'system_leverage' in self.indicators_df.columns:
            ax.plot(self.indicators_df.index, self.indicators_df['system_leverage'], 
                   'k--', label='System', alpha=0.7)
        
        # Add recession shading
        for start, end, _, severity in RECESSIONS:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            if end_dt < self.indicators_df.index[0] or start_dt > self.indicators_df.index[-1]:
                continue
            
            if severity == 'severe':
                ax.axvspan(start_dt, end_dt, alpha=0.1, color='red')
        
        ax.set_title('Leverage Evolution by Sector', fontweight='bold')
        ax.set_ylabel('Leverage Ratio')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_model_performance(self, ax):
        """Plot model performance metrics."""
        
        if self.validation_results.empty:
            ax.text(0.5, 0.5, 'No validation data available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Calculate ROC curve
        thresholds = sorted(self.validation_results['threshold'].unique())
        tpr_list = []  # True positive rate
        fpr_list = []  # False positive rate
        
        for threshold in thresholds:
            subset = self.validation_results[self.validation_results['threshold'] == threshold]
            grouped = subset.groupby('recession')['detected'].max()
            
            tpr = grouped.mean()  # Detection rate
            
            # False positive rate
            high_score_quarters = (self.enhanced_score > threshold).sum()
            total_quarters = len(self.enhanced_score)
            fpr = high_score_quarters / total_quarters if total_quarters > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Plot ROC curve
        ax.plot(fpr_list, tpr_list, 'b-', linewidth=2, label='Model Performance')
        ax.scatter(fpr_list, tpr_list, c=thresholds, cmap='RdYlGn_r', s=50)
        
        # Add diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
        
        # Highlight optimal point
        if self.optimal_threshold:
            opt_idx = thresholds.index(self.optimal_threshold)
            ax.scatter(fpr_list[opt_idx], tpr_list[opt_idx], 
                      color='red', s=100, marker='*', 
                      label=f'Optimal (θ={self.optimal_threshold:.2f})')
        
        ax.set_title('Model Performance: ROC Analysis', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Detection)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    
    def _plot_historical_comparison(self, ax):
        """Plot comparison to historical crises."""
        
        if self.historical_comparison.empty:
            ax.text(0.5, 0.5, 'No historical comparison data', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Bar chart comparing scores
        crises = self.historical_comparison['crisis'].tolist()
        pre_crisis_scores = self.historical_comparison['pre_crisis_score'].tolist()
        current_score = self.enhanced_score.iloc[-1]
        
        x = np.arange(len(crises))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pre_crisis_scores, width, 
                      label='12mo Before Crisis', color='orange', alpha=0.7)
        bars2 = ax.bar(x + width/2, [current_score] * len(crises), width, 
                      label='Current', color='red', alpha=0.7)
        
        # Add threshold line
        if self.optimal_threshold:
            ax.axhline(y=self.optimal_threshold, color='red', linestyle='--', 
                      alpha=0.5, label=f'Crisis Threshold')
        
        ax.set_title('Current vs. Pre-Crisis Conditions', fontweight='bold')
        ax.set_ylabel('Crisis Score')
        ax.set_xticks(x)
        ax.set_xticklabels(crises, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars1, pre_crisis_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{current_score:.2f}', ha='center', va='bottom', fontsize=7)
    
    def _plot_risk_dashboard(self, ax):
        """Plot current risk dashboard."""
        
        ax.axis('off')
        
        current_score = self.enhanced_score.iloc[-1]
        current_date = self.enhanced_score.index[-1]
        current_indicators = self.indicators_df.iloc[-1]
        
        # Determine risk level
        if current_score > 0.7:
            risk_level = "CRITICAL"
            risk_color = 'darkred'
        elif current_score > 0.6:
            risk_level = "HIGH"
            risk_color = 'red'
        elif current_score > 0.5:
            risk_level = "ELEVATED"
            risk_color = 'orange'
        elif current_score > 0.4:
            risk_level = "MODERATE"
            risk_color = 'gold'
        else:
            risk_level = "LOW"
            risk_color = 'green'
        
        # Title
        ax.text(0.5, 0.95, 'CURRENT RISK ASSESSMENT', 
               ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.88, f'{current_date.date()}', 
               ha='center', fontsize=10)
        
        # Risk level box
        rect = patches.Rectangle((0.2, 0.65), 0.6, 0.15, 
                                linewidth=2, edgecolor=risk_color, 
                                facecolor=risk_color, alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.5, 0.725, risk_level, ha='center', va='center',
               fontsize=20, fontweight='bold', color=risk_color)
        
        # Crisis score
        ax.text(0.5, 0.58, f'Crisis Score: {current_score:.3f}', 
               ha='center', fontsize=12)
        
        # Percentile
        percentile = (self.enhanced_score <= current_score).mean() * 100
        ax.text(0.5, 0.52, f'Historical Percentile: {percentile:.1f}%', 
               ha='center', fontsize=10)
        
        # Key metrics
        y_pos = 0.42
        metrics = [
            ('System Leverage', current_indicators.get('system_leverage', 0)),
            ('Financial Leverage', current_indicators.get('financial_leverage', 0)),
            ('System Liquidity', current_indicators.get('system_liquidity_ratio', 0)),
            ('Flow Imbalance', current_indicators.get('avg_flow_imbalance', 0))
        ]
        
        for metric, value in metrics:
            ax.text(0.3, y_pos, f'{metric}:', ha='right', fontsize=9)
            ax.text(0.32, y_pos, f'{value:.3f}', ha='left', fontsize=9, fontweight='bold')
            y_pos -= 0.06
        
        # Trend arrow
        if len(self.enhanced_score) > 4:
            prev_score = self.enhanced_score.iloc[-5]
            if current_score > prev_score * 1.05:
                arrow = '↑'
                arrow_color = 'red'
                trend = 'INCREASING'
            elif current_score < prev_score * 0.95:
                arrow = '↓'
                arrow_color = 'green'
                trend = 'DECREASING'
            else:
                arrow = '→'
                arrow_color = 'gray'
                trend = 'STABLE'
            
            ax.text(0.7, 0.25, arrow, fontsize=40, color=arrow_color, ha='center')
            ax.text(0.7, 0.15, trend, fontsize=10, color=arrow_color, ha='center')
        
        # Recommendation
        if current_score > 0.6:
            rec = "Immediate risk mitigation recommended"
        elif current_score > 0.5:
            rec = "Enhanced monitoring advised"
        else:
            rec = "Continue standard monitoring"
        
        ax.text(0.5, 0.05, rec, ha='center', fontsize=9, style='italic')
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        
        report_path = self.output_dir / "validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SFC CRISIS DETECTION - HISTORICAL VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Current status
            current_score = self.enhanced_score.iloc[-1]
            current_date = self.enhanced_score.index[-1]
            
            f.write("CURRENT STATUS\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {current_date.date()}\n")
            f.write(f"Enhanced Crisis Score: {current_score:.3f}\n")
            f.write(f"Original Composite Score: {self.indicators_df.iloc[-1]['composite_score']:.3f}\n")
            
            percentile = (self.enhanced_score <= current_score).mean() * 100
            f.write(f"Historical Percentile: {percentile:.1f}%\n")
            
            if self.optimal_threshold:
                if current_score > self.optimal_threshold:
                    f.write("\n⚠️ WARNING: Score exceeds optimal crisis threshold\n")
            
            # Validation performance
            f.write("\n\nMODEL VALIDATION PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            if not self.validation_results.empty:
                # Get performance at optimal threshold
                opt_results = self.validation_results[
                    self.validation_results['threshold'] == self.optimal_threshold
                ]
                
                if not opt_results.empty:
                    grouped = opt_results.groupby('recession')['detected'].max()
                    detection_rate = grouped.mean()
                    
                    f.write(f"Optimal Threshold: {self.optimal_threshold:.3f}\n")
                    f.write(f"Detection Rate: {detection_rate:.1%}\n")
                    f.write(f"Recessions Tested: {len(grouped)}\n")
                    
                    # List detection results
                    f.write("\nDetection Results:\n")
                    for recession, detected in grouped.items():
                        status = "✓" if detected else "✗"
                        f.write(f"  {status} {recession}\n")
            
            # Historical comparison
            f.write("\n\nHISTORICAL COMPARISON\n")
            f.write("-"*40 + "\n")
            
            if not self.historical_comparison.empty:
                for _, row in self.historical_comparison.iterrows():
                    f.write(f"\nvs. {row['crisis']}:\n")
                    f.write(f"  Pre-crisis score: {row['pre_crisis_score']:.3f}\n")
                    f.write(f"  Current score: {row['current_score']:.3f}\n")
                    f.write(f"  Ratio (now/then): {row['score_ratio']:.2f}x\n")
            
            # Risk assessment
            f.write("\n\nRISK ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            if current_score > 0.7:
                f.write("CRITICAL RISK - Crisis highly probable\n")
            elif current_score > 0.6:
                f.write("HIGH RISK - Significant stress detected\n")
            elif current_score > 0.5:
                f.write("ELEVATED RISK - Enhanced monitoring needed\n")
            elif current_score > 0.4:
                f.write("MODERATE RISK - Some stress indicators\n")
            else:
                f.write("LOW RISK - System appears stable\n")
        
        print(f"✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_validation(self):
        """Run complete historical validation analysis."""
        
        print("\n" + "="*70)
        print("RUNNING HISTORICAL VALIDATION ANALYSIS")
        print("="*70)
        
        # Step 1: Calculate enhanced crisis scores
        self.calculate_enhanced_crisis_score()
        
        # Step 2: Validate against recessions
        self.validate_against_recessions()
        
        # Step 3: Compare to historical crises
        self.compare_to_historical_crises()
        
        # Step 4: Create visualizations
        self.create_comprehensive_visualization()
        
        # Step 5: Generate report
        self.generate_report()
        
        # Save enhanced scores
        self.enhanced_score.to_csv(self.output_dir / "enhanced_crisis_scores.csv")
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'current_score': self.enhanced_score.iloc[-1],
            'optimal_threshold': self.optimal_threshold,
            'validation_results': self.validation_results,
            'historical_comparison': self.historical_comparison
        }


def main():
    """Run historical validation on existing indicators."""
    
    # Check if indicators file exists
    indicators_file = "outputs/optimized_complete_indicators.csv"
    if not Path(indicators_file).exists():
        print(f"ERROR: Indicators file not found: {indicators_file}")
        print("Please run the optimized analyzer first to generate indicators.")
        return None
    
    # Initialize validator
    validator = HistoricalValidation(indicators_file)
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    # Print executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    current_score = results['current_score']
    optimal_threshold = results['optimal_threshold']
    
    print(f"\n📊 Current Status:")
    print(f"  Enhanced Crisis Score: {current_score:.3f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    
    if current_score > optimal_threshold:
        print("\n⚠️ WARNING: System exceeds crisis threshold!")
        print("  High probability of financial stress within 6-18 months")
    elif current_score > optimal_threshold * 0.9:
        print("\n⚠️ CAUTION: Approaching crisis threshold")
        print("  Enhanced monitoring recommended")
    else:
        print("\n✅ System below crisis threshold")
        print("  Continue standard monitoring")
    
    # Model performance
    if not results['validation_results'].empty:
        opt_results = results['validation_results'][
            results['validation_results']['threshold'] == optimal_threshold
        ]
        if not opt_results.empty:
            detection_rate = opt_results.groupby('recession')['detected'].max().mean()
            print(f"\n📈 Model Performance:")
            print(f"  Historical Detection Rate: {detection_rate:.1%}")
    
    return validator


if __name__ == "__main__":
    validator = main()
