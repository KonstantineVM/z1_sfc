#!/usr/bin/env python3
"""
Script to download Z.1 and FWTW data from Federal Reserve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import Z1Loader, FWTWLoader

def main():
    """Download all required data."""
    print("Downloading Z.1 data...")
    z1_loader = Z1Loader()
    z1_loader.download_latest()
    
    print("Downloading FWTW data...")
    fwtw_loader = FWTWLoader()
    fwtw_loader.download_latest()
    
    print("Data download completed!")

if __name__ == "__main__":
    main()
