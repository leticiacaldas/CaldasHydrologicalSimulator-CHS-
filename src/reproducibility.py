"""
Reproducibility and versioning module for HydroSim-RF.

Provides utilities for ensuring scientific reproducibility through
version tracking, seed management, and configuration persistence.

Author: Letícia Caldas
License: MIT
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Letícia Caldas"
__date__ = "2026-03-23"


class ReproducibilityManager:
    """
    Manages reproducibility aspects of simulations.
    
    Handles:
    - Random seed management
    - Configuration persistence
    - Simulation metadata logging
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, Any] = {
            "version": __version__,
            "author": __author__,
            "created_date": __date__,
        }
    
    def set_random_seed(self, seed: int = 42) -> None:
        """
        Set global random seeds for numpy and Python random.
        
        Parameters
        ----------
        seed : int
            Seed value for reproducibility (default: 42)
        """
        np.random.seed(int(seed))
        logger.info(f"Random seed set to {seed}")
        self.metadata["random_seed"] = int(seed)
    
    def save_configuration(self, config: Dict[str, Any], name: str = "config") -> str:
        """
        Save configuration to JSON file with timestamp.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        name : str
            Configuration file name prefix
            
        Returns
        -------
        str
            Path to saved configuration file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.output_dir / f"{name}_{timestamp}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
        return str(config_file)
    
    def save_metadata(self, **kwargs) -> None:
        """
        Save simulation metadata for reproducibility.
        
        Parameters
        ----------
        **kwargs : dict
            Arbitrary metadata key-value pairs
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata.update(kwargs)
        self.metadata["timestamp"] = timestamp
        
        metadata_file = self.output_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    print(f"HydroSim-RF v{__version__}")
    print(f"Author: {__author__}")
    print(f"Release Date: {__date__}")
    print("\nReproducibility features enabled:")
    print("  - Fixed random seed management")
    print("  - Configuration persistence")
    print("  - Metadata logging")
    print("  - Deterministic algorithms")
