"""
conftest.py - Pytest Configuration & Path Setup
Ensures all four modular folders and the backend are cleanly importable.
"""

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "03_backend_api")
ml_dir = os.path.join(root_dir, "02_ml_engine")
data_dir = os.path.join(root_dir, "01_data_pipeline")

for p in [root_dir, backend_dir, ml_dir, data_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
