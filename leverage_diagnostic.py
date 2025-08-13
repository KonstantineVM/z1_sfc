#!/usr/bin/env python3
"""
Leverage Calculation Diagnostic Tool
=====================================
Diagnoses why leverage calculations are returning NaN values.
Tests different calculation methods to find what works.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class LeverageDiagnostic:
    """
    Diagnoses leverage calculation issues.
    """
    
    def __init__(self):
        """Initialize diagnostic tool."""
        self.output_dir = Path("outputs/leverage_diagnostic")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find a recent balance sheet file
        bs_files = sorted(Path("outputs").glob("sfc_balance_sheet_*.csv"))
        if not bs_files:
            raise FileNotFoundError("No balance sheet files found")
        
        # Use most recent file for testing
        self.test_file = bs_files[-1]
        self.date = self.test_file.stem.replace('sfc_balance_sheet_', '')
        
        print(f"Using test file: {self.test_file}")
        print(f"Date: {self.date}")
        
        # Load test data
        self.bs = pd.read_csv(self.test_file, index_col=0)
        
        # Load instrument map
        try:
            with open("mappings/instrument_map.json", 'r') as f:
                self.imap = json.load(f)
        except:
            print("Warning: Could not load instrument map")
            self.imap = {}
    
    def diagnose_data_structure(self):
        """Diagnose the structure of the balance sheet data."""
        print("\n" + "="*70)
        print("BALANCE SHEET DATA STRUCTURE")
        print("="*70)
        
        print(f"\nShape: {self.bs.shape}")
        print(f"Columns ({len(self.bs.columns)}): {list(self.bs.columns[:10])}...")
        print(f"Index ({len(self.bs.index)}): {list(self.bs.index[:10])}...")
        
        # Check index data type
        print(f"\nIndex dtype: {self.bs.index.dtype}")
        print(f"Index name: {self.bs.index.name}")
        
        # Check for specific sectors
        test_sectors = ['10', '15', '70', '79', '31']
        print("\nChecking for key sectors:")
        for sector in test_sectors:
            if sector in self.bs.columns:
                print(f"  ✓ Sector {sector} found")
                # Check non-zero values
                non_zero = (self.bs[sector] != 0).sum()
                print(f"    Non-zero entries: {non_zero}")
            else:
                print(f"  ✗ Sector {sector} NOT found")
        
        # Sample data
        print("\nSample data (first 5 rows, first 3 sectors):")
        if len(self.bs.columns) > 3:
            print(self.bs.iloc[:5, :3])
    
    def diagnose_instrument_mapping(self):
        """Diagnose instrument identification issues."""
        print("\n" + "="*70)
        print("INSTRUMENT MAPPING DIAGNOSTIC")
        print("="*70)
        
        print(f"\nTotal instruments in map: {len(self.imap)}")
        
        # Check instrument formats
        print("\nInstrument key formats in map:")
        sample_keys = list(self.imap.keys())[:10]
        for key in sample_keys:
            print(f"  {key} (type: {type(key).__name__}, len: {len(str(key))})")
        
        # Check balance sheet index format
        print("\nBalance sheet index formats:")
        for idx in self.bs.index[:10]:
            print(f"  {idx} (type: {type(idx).__name__})")
        
        # Try to match instruments
        print("\nMatching test:")
        matches = 0
        no_matches = 0
        
        for inst_str in list(self.imap.keys())[:50]:  # Test first 50
            # Try different matching approaches
            matched = False
            
            # Direct match
            if inst_str in self.bs.index:
                matched = True
            
            # Try as integer
            try:
                inst_int = int(inst_str)
                if inst_int in self.bs.index:
                    matched = True
            except:
                pass
            
            # Try with padding
            if inst_str.zfill(5) in self.bs.index:
                matched = True
            
            if matched:
                matches += 1
            else:
                no_matches += 1
        
        print(f"  Matches: {matches}")
        print(f"  No matches: {no_matches}")
    
    def test_leverage_calculations(self):
        """Test different leverage calculation methods."""
        print("\n" + "="*70)
        print("TESTING LEVERAGE CALCULATION METHODS")
        print("="*70)
        
        # Test sector
        test_sector = None
        for sector in ['10', '15', '70']:
            if sector in self.bs.columns:
                test_sector = sector
                break
        
        if not test_sector:
            print("No test sector found!")
            return
        
        print(f"\nTesting with sector: {test_sector}")
        
        # Method 1: Simple ratio of positive to total
        print("\n1. Simple Positive/Total Method:")
        sector_data = self.bs[test_sector]
        positive_sum = sector_data[sector_data > 0].sum()
        negative_sum = abs(sector_data[sector_data < 0].sum())
        total = positive_sum + negative_sum
        
        if total > 0:
            simple_leverage = positive_sum / total
            print(f"   Positive sum: {positive_sum:,.0f}")
            print(f"   Negative sum: {negative_sum:,.0f}")
            print(f"   Total: {total:,.0f}")
            print(f"   Leverage: {simple_leverage:.2%}")
        else:
            print("   Total is zero!")
        
        # Method 2: Using specific instruments
        print("\n2. Debt Instrument Method:")
        
        # Identify debt instruments
        debt_keywords = ['debt', 'bond', 'loan', 'mortgage', 'credit', 'liability']
        debt_instruments = []
        
        for code, meta in self.imap.items():
            # Check various fields
            if any(keyword in str(meta).lower() for keyword in debt_keywords):
                debt_instruments.append(code)
        
        print(f"   Found {len(debt_instruments)} debt instruments")
        
        # Calculate debt for sector
        sector_debt = 0
        matched_instruments = 0
        
        for inst_str in debt_instruments[:100]:  # Test first 100
            # Try different formats
            found = False
            value = 0
            
            # Try as string
            if inst_str in self.bs.index:
                value = abs(self.bs.loc[inst_str, test_sector])
                found = True
            
            # Try as integer
            if not found:
                try:
                    inst_int = int(inst_str)
                    if inst_int in self.bs.index:
                        value = abs(self.bs.loc[inst_int, test_sector])
                        found = True
                except:
                    pass
            
            if found and value > 0:
                sector_debt += value
                matched_instruments += 1
        
        print(f"   Matched instruments: {matched_instruments}")
        print(f"   Total debt: {sector_debt:,.0f}")
        
        # Calculate leverage
        if positive_sum > 0:
            debt_leverage = sector_debt / positive_sum
            print(f"   Leverage (debt/liabilities): {debt_leverage:.2%}")
        
        # Method 3: Top instruments by size
        print("\n3. Top Instruments Method:")
        
        # Get top 20 instruments by absolute value for this sector
        sector_abs = abs(sector_data)
        top_instruments = sector_abs.nlargest(20)
        
        print(f"   Top 20 instruments for sector {test_sector}:")
        for inst, value in top_instruments.items():
            # Get label if available
            label = self.bs.loc[inst, 'label'] if 'label' in self.bs.columns else 'N/A'
            
            # Check if it's in instrument map
            in_map = inst in self.imap or str(inst) in self.imap
            
            print(f"     {inst}: {value:,.0f} - {label[:30]} (in map: {in_map})")
        
        # Method 4: Liability-based calculation
        print("\n4. Liability Classification Method:")
        
        liability_count = 0
        asset_count = 0
        
        for inst in self.bs.index[:100]:  # Sample first 100
            value = self.bs.loc[inst, test_sector]
            if value > 0:
                liability_count += 1
            elif value < 0:
                asset_count += 1
        
        print(f"   Instruments with positive values (liabilities): {liability_count}")
        print(f"   Instruments with negative values (assets): {asset_count}")
        
        # Calculate leverage based on sign convention
        total_liabilities = self.bs[self.bs[test_sector] > 0][test_sector].sum()
        total_assets = abs(self.bs[self.bs[test_sector] < 0][test_sector].sum())
        
        if total_assets > 0:
            balance_sheet_leverage = total_liabilities / total_assets
            print(f"   Total liabilities: {total_liabilities:,.0f}")
            print(f"   Total assets: {total_assets:,.0f}")
            print(f"   Leverage (L/A): {balance_sheet_leverage:.2%}")
    
    def test_all_sectors(self):
        """Test leverage calculation for all available sectors."""
        print("\n" + "="*70)
        print("TESTING ALL SECTORS")
        print("="*70)
        
        results = []
        
        # Test sectors
        test_sectors = ['10', '11', '15', '70', '40', '14', '26', '79', '31', '58']
        
        for sector in test_sectors:
            if sector not in self.bs.columns:
                continue
            
            # Calculate using simple method
            sector_data = self.bs[sector]
            positive_sum = sector_data[sector_data > 0].sum()
            negative_sum = abs(sector_data[sector_data < 0].sum())
            total = positive_sum + negative_sum
            
            if total > 0:
                leverage = positive_sum / total
            else:
                leverage = np.nan
            
            # Also calculate L/A ratio
            if negative_sum > 0:
                la_ratio = positive_sum / negative_sum
            else:
                la_ratio = np.nan
            
            results.append({
                'sector': sector,
                'positive': positive_sum,
                'negative': negative_sum,
                'total': total,
                'leverage': leverage,
                'la_ratio': la_ratio
            })
        
        # Display results
        print("\nSector Leverage Summary:")
        print("-"*70)
        print(f"{'Sector':<10} {'Positive':>15} {'Negative':>15} {'Leverage':>10} {'L/A Ratio':>10}")
        print("-"*70)
        
        for r in results:
            print(f"{r['sector']:<10} {r['positive']:>15,.0f} {r['negative']:>15,.0f} "
                  f"{r['leverage']:>10.1%} {r['la_ratio']:>10.1%}")
        
        return results
    
    def create_diagnostic_report(self):
        """Create comprehensive diagnostic report."""
        report_path = self.output_dir / f'leverage_diagnostic_{self.date}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LEVERAGE CALCULATION DIAGNOSTIC REPORT\n")
            f.write(f"Date: {self.date}\n")
            f.write("="*70 + "\n\n")
            
            # Data structure
            f.write("DATA STRUCTURE\n")
            f.write("-"*40 + "\n")
            f.write(f"Balance sheet shape: {self.bs.shape}\n")
            f.write(f"Index type: {self.bs.index.dtype}\n")
            f.write(f"Number of instruments: {len(self.bs.index)}\n")
            f.write(f"Number of sectors: {len(self.bs.columns)}\n")
            
            # Instrument mapping
            f.write("\n\nINSTRUMENT MAPPING\n")
            f.write("-"*40 + "\n")
            f.write(f"Instruments in map: {len(self.imap)}\n")
            
            # Test results
            results = self.test_all_sectors()
            
            f.write("\n\nLEVERAGE CALCULATIONS\n")
            f.write("-"*40 + "\n")
            
            valid_count = sum(1 for r in results if not np.isnan(r['leverage']))
            f.write(f"Sectors with valid leverage: {valid_count}/{len(results)}\n")
            
            if valid_count > 0:
                avg_leverage = np.nanmean([r['leverage'] for r in results])
                max_leverage = np.nanmax([r['leverage'] for r in results])
                f.write(f"Average leverage: {avg_leverage:.1%}\n")
                f.write(f"Maximum leverage: {max_leverage:.1%}\n")
            
            f.write("\n\nDETAILED RESULTS\n")
            f.write("-"*40 + "\n")
            
            for r in results:
                f.write(f"\nSector {r['sector']}:\n")
                f.write(f"  Liabilities (positive): ${r['positive']:,.0f}\n")
                f.write(f"  Assets (negative): ${r['negative']:,.0f}\n")
                f.write(f"  Leverage: {r['leverage']:.1%}\n")
                f.write(f"  L/A Ratio: {r['la_ratio']:.1%}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            if valid_count == 0:
                f.write("⚠️ No valid leverage calculations - check data format\n")
            elif avg_leverage > 0.5:
                f.write("⚠️ CRITICAL: Average leverage exceeds 50%\n")
            elif avg_leverage > 0.3:
                f.write("⚠️ WARNING: High average leverage\n")
            else:
                f.write("✓ Leverage calculations working\n")
        
        print(f"\n✓ Report saved to {report_path}")
        return report_path
    
    def run_complete_diagnostic(self):
        """Run complete diagnostic analysis."""
        print("\n" + "="*70)
        print("RUNNING LEVERAGE DIAGNOSTIC")
        print("="*70)
        
        # Step 1: Diagnose data structure
        self.diagnose_data_structure()
        
        # Step 2: Diagnose instrument mapping
        self.diagnose_instrument_mapping()
        
        # Step 3: Test calculation methods
        self.test_leverage_calculations()
        
        # Step 4: Test all sectors
        results = self.test_all_sectors()
        
        # Step 5: Create report
        self.create_diagnostic_report()
        
        print("\n" + "="*70)
        print("DIAGNOSTIC COMPLETE")
        print("="*70)
        
        return results


def main():
    """Run leverage diagnostic."""
    
    diagnostic = LeverageDiagnostic()
    results = diagnostic.run_complete_diagnostic()
    
    # Executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    if results:
        valid_results = [r for r in results if not np.isnan(r['leverage'])]
        
        if valid_results:
            print(f"\n✓ Successfully calculated leverage for {len(valid_results)} sectors")
            
            # Show results
            print("\nLeverage Results:")
            for r in valid_results:
                leverage = r['leverage']
                if leverage > 0.5:
                    emoji = "🔴"
                    status = "CRITICAL"
                elif leverage > 0.3:
                    emoji = "🟠"
                    status = "HIGH"
                else:
                    emoji = "🟢"
                    status = "NORMAL"
                
                print(f"  {emoji} Sector {r['sector']}: {leverage:.1%} ({status})")
            
            # Average
            avg_leverage = np.mean([r['leverage'] for r in valid_results])
            print(f"\n📊 Average Leverage: {avg_leverage:.1%}")
            
            if avg_leverage > 0.5:
                print("  ⚠️ SYSTEM CRITICALLY OVERLEVERAGED")
            elif avg_leverage > 0.3:
                print("  ⚠️ SYSTEM HIGHLY LEVERAGED")
        else:
            print("\n⚠️ No valid leverage calculations - data format issue")
    
    return diagnostic


if __name__ == "__main__":
    diagnostic = main()
