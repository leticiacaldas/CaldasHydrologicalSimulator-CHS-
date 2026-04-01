#!/usr/bin/env python3
"""
HydroSim-RF: Hybrid Raster-Based Urban Flood Simulation Framework

This package provides a comprehensive computational framework for rapid 2-D flood 
inundation modeling over Digital Elevation Models (DEMs) combined with scikit-learn 
based machine learning flood probability estimation.

Documentation: https://github.com/leticia-caldas/hydrosim-rf
Publication: Environmental Modelling & Software
"""

import subprocess
import sys

# Quick start guide
if __name__ == "__main__":
    print(__doc__)
    print("\nQuick Start:")
    print("============\n")
    print("1. Launch interactive web interface:")
    print("   $ streamlit run run.py\n")
    
    print("2. Run tests:")
    print("   $ python -m pytest tests/ -v\n")
    
    print("3. Run batch simulation:")
    print("   $ python run.py --mode batch --config configs/default.json\n")
