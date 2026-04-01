"""
Core hydrodynamic simulation engine for HydroSim-RF.

Implements the diffusion-wave approximation for 2D flood inundation modeling
with vectorized NumPy operations for high performance.
"""

from .simulator import DiffusionWaveFloodModel

__all__ = ['DiffusionWaveFloodModel']
