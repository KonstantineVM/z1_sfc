# src/utils/helpers.py
import os

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path

