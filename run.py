#!/usr/bin/env python3
"""
HydroSim-RF: Scientific Flood Simulation Framework

Main entry point for running flood simulations with optional machine learning
flood probability estimation and spatial mitigation analysis.

Usage:
    streamlit run run.py  (For interactive web interface)
    python run.py --config configs/default.json  (For batch simulation)

Author: Letícia Caldas
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hydrosim_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Banner
BANNER = """
╔════════════════════════════════════════════════════════════════╗
║                       HydroSim-RF v1.0.0                       ║
║     Hybrid Raster-Based Urban Flood Simulation Framework       ║
║          with Random Forest Inundation Probability             ║
║                                                                ║
║  A scientific tool for flood inundation modeling and           ║
║  machine learning-based probability estimation.               ║
║                                                                ║
║  For journal publication in:                                  ║
║  Environmental Modelling & Software                           ║
║                                                                ║
║  Author: Letícia Caldas                                       ║
║  License: MIT                                                 ║
╚════════════════════════════════════════════════════════════════╝
"""


def load_configuration(config_path: str) -> dict:
    """
    Load simulation configuration from JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration JSON file.
        
    Returns
    -------
    config : dict
        Configuration dictionary.
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON configuration: {e}")
        sys.exit(1)


def save_configuration(config: dict, output_dir: str) -> None:
    """
    Save current configuration to output directory for reproducibility.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_dir : str
        Output directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_out = os.path.join(output_dir, f"config_{timestamp}.json")
    
    with open(config_out, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_out}")


def run_streamlit_interface():
    """
    Launch interactive Streamlit web interface.
    """
    print(BANNER)
    logger.info("Launching Streamlit web interface...")
    
    try:
        import streamlit.cli  # type: ignore
        sys.argv = ["streamlit", "run", 
                   str(Path(__file__).parent / "hydrosim_rf.py")]
        streamlit.cli.main()
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to launch Streamlit: {e}")
        sys.exit(1)


def run_batch_simulation(config_path: str):
    """
    Run batch flood simulation from configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration JSON file.
    """
    print(BANNER)
    
    config = load_configuration(config_path)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration used for this run
    save_configuration(config, str(output_dir))
    
    # Import simulation modules
    from src.core.simulator import DiffusionWaveFloodModel
    from src.ml.flood_classifier import (
        train_flood_classifier,
        predict_probability,
    )
    
    # [Batch simulation logic would go here in full implementation]
    logger.info("Batch simulation mode not yet implemented in main entry point")
    logger.info("Use: streamlit run run.py for interactive mode")


def main():
    """
    Main entry point orchestrating simulation modes.
    """
    print(BANNER)
    
    parser = argparse.ArgumentParser(
        description="HydroSim-RF: Flood Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mode',
        choices=['web', 'batch'],
        default='web',
        help='Execution mode (default: web interface)',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.json',
        help='Configuration file for batch simulations',
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='HydroSim-RF v1.0.0',
    )
    
    args = parser.parse_args()
    
    logger.info(f"HydroSim-RF starting in {args.mode} mode")
    
    if args.mode == 'web':
        run_streamlit_interface()
    elif args.mode == 'batch':
        run_batch_simulation(args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
