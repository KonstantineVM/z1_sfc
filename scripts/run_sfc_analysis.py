#!/usr/bin/env python3
"""
Main script to run SFC Kalman Filter analysis on Z.1 data.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import SFCMatrix
from src.models import SFCKalmanFilter
from src.data import Z1Loader, FWTWLoader
from src.utils import setup_logging

def main():
    """Run SFC analysis."""
    parser = argparse.ArgumentParser(description="Run SFC Kalman Filter Analysis")
    parser.add_argument(
        "--mode", 
        choices=["test", "development", "production", "full"],
        default="development",
        help="Execution mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sfc_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=logging.INFO)
    logger.info(f"Starting SFC analysis in {args.mode} mode")
    
    try:
        # Load data
        logger.info("Loading Z.1 data...")
        z1_loader = Z1Loader()
        z1_data = z1_loader.load_cached("Z1")
        
        logger.info("Loading FWTW data...")
        fwtw_loader = FWTWLoader()
        fwtw_data = fwtw_loader.load_cached()
        
        # Initialize filter
        logger.info("Initializing SFC Kalman Filter...")
        filter = SFCKalmanFilter(
            data=z1_data,
            fwtw_data=fwtw_data,
            config_path=args.config,
            mode=args.mode
        )
        
        # Run filter
        logger.info("Running Kalman filter with SFC constraints...")
        results = filter.fit()
        
        # Validate
        logger.info("Validating constraints...")
        validation = filter.validate_constraints()
        
        # Save results
        logger.info(f"Saving results to {args.output}...")
        filter.save_results(args.output)
        
        logger.info("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
