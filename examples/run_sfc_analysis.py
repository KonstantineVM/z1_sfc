#!/usr/bin/env python3
"""
Run multiple modes defined in config/run_modes.yaml on the same dataset.
Useful for quick A/B tests (e.g., toggle bilateral, market-clearing).
"""

import argparse, os, time
from copy import deepcopy
from examples.run_proper_sfc import main as run_once
from src.utils.config_manager import ConfigManager

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["development","production"])
    ap.add_argument("--config", default="config/proper_sfc_config.yaml")
    args = ap.parse_args()

    cm = ConfigManager(root="config")
    base = cm.load(os.path.basename(args.config))
    modes = cm.load("run_modes.yaml")

    for mode in args.modes:
        print(f"\n=== Running mode: {mode} ===")
        # Reuse the same entry point via CLI:
        os.system(f"python -m examples.run_proper_sfc --mode {mode} --config {args.config}")

if __name__ == "__main__":
    main()

