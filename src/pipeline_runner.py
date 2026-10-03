"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Pipeline Runner Orchestrator
Reference: LOG-ML-OPT-2026-T4
"""

import sys
import os

# Ensure src directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipeline import run_master_pipeline

if __name__ == "__main__":
    run_master_pipeline()
