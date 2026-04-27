"""
Vectorised 2-D diffusion-wave flood inundation solver (NumPy backend).

This module implements a simplified storage-cell approach for water redistribution
across a DEM grid, following the zero-inertia (diffusion-wave) simplification of
shallow-water equations.

References:
    Hunter, N. M., Bates, P. D., Horritt, M. S., De Roo, A. P. J., & 
    Werner, M. G. F. (2005). Utility of different data types for calibrating 
    flood inundation models within a GLUE framework. 
    Hydrology and Earth System Sciences, 9(4), 412–430.
    
    Neal, J., Schumann, G., & Bates, P. (2012). A subgrid channel model for 
    simulating river hydraulics and floodplain inundation over large and data 
    sparse areas. Water Resources Research, 48(11), W11512.

Author: Letícia Caldas
License: MIT
"""

import numpy as np
from typing import Optional, Set, Tuple, Dict, Any, List
from .gama_flood_model_d8 import GamaFloodModelNumpy as GamaFloodModelD8


class DiffusionWaveFloodModel:
    """
    Vectorised 2-D diffusion-wave flood inundation solver (NumPy backend).

    The model implements a storage-cell approach wherein water redistributes from 
    each cell to lower-elevation neighbours in proportion to the free-surface 
    head difference, up to a maximum fraction *diffusion_rate* per time step.

    No-data cells in the DEM are filled with the domain median prior to simulation 
    to prevent spurious sinks. Water volume is conserved to machine precision 
    within the active cell set.

    Parameters
    ----------
    dem_data : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m a.s.l.].
    sources_mask : np.ndarray, shape (H, W), dtype uint8 or bool
        Binary mask identifying rainfall source cells (e.g., catchment area or 
        watercourse polygon).
    diffusion_rate : float
        Fraction of available water depth that may leave a cell per time step 
        (dimensionless, 0 < diffusion_rate ≤ 1).
    flood_threshold : float
        Minimum water depth [m] used to classify a cell as inundated for 
        reporting purposes.
    cell_size_meters : float
        Planimetric cell dimension [m]; used to compute surface area and 
        volumetric water balance.
    river_mask : np.ndarray, optional
        Binary mask of river/channel cells. When provided, water added via 
        rainfall is routed preferentially through channel cells.

    Attributes
    ----------
    water_height : np.ndarray, shape (H, W), dtype float32
        Current water depth [m] at each grid cell.
    altitude : np.ndarray, shape (H, W), dtype float32
        Ground-surface elevation [m a.s.l.].
    simulation_time_minutes : int
        Elapsed simulation time [min] since initialisation.
    overflow_time_minutes : int or None
        Time [min] at which surface water first exceeded *flood_threshold* 
        outside the source area.
    history : list of dict
        Timestamped record of domain-wide diagnostics appended at every time step.

    Examples
    --------
    Initialize and run a simple flood simulation:
    
    >>> dem = np.random.rand(100, 100) * 10
    >>> sources = np.zeros((100, 100), dtype=bool)
    >>> sources[40:60, 40:60] = True
    >>> model = DiffusionWaveFloodModel(dem, sources, 0.5, 0.1, 25.0)
    >>> 
    >>> for t in range(100):
    ...     model.apply_rainfall(5.0)  # 5 mm per step
    ...     model.advance_flow()
    ...     model.record_diagnostics(10)  # 10 min per step
    >>> 
    >>> print(f"Max water depth: {model.water_height.max():.3f} m")
    >>> print(f"Time to overflow: {model.overflow_time_minutes} min")
    """

    def __init__(
        self,
        dem_data: np.ndarray,
        sources_mask: np.ndarray,
        diffusion_rate: float,
        flood_threshold: float,
        cell_size_meters: float,
        river_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the diffusion-wave flood model.

        Parameters
        ----------
        dem_data : np.ndarray
            Ground elevation array (H x W).
        sources_mask : np.ndarray
            Binary rainfall source mask.
        diffusion_rate : float
            Flow diffusion coefficient (0 < α ≤ 1).
        flood_threshold : float
            Inundation depth threshold [m].
        cell_size_meters : float
            Cell size [m].
        river_mask : np.ndarray, optional
            Binary river channel mask.
        """
        self.height, self.width = dem_data.shape
        self.diffusion_rate = float(diffusion_rate)
        self.flood_threshold = float(flood_threshold)
        self.cell_area = float(cell_size_meters) ** 2

        # Handle NaN values in DEM
        _alt = dem_data.astype(np.float32)
        if not np.isfinite(_alt).all():
            _median = float(np.nanmedian(_alt)) if np.isfinite(_alt).any() else 0.0
            _alt = np.where(np.isfinite(_alt), _alt, _median)

        self.altitude: np.ndarray = _alt
        self._valid_mask: np.ndarray = np.isfinite(dem_data.astype(np.float32))

        # Initialize masks
        self.is_source: np.ndarray = (
            sources_mask.astype(bool)
            if sources_mask is not None
            else np.zeros_like(self.altitude, dtype=bool)
        )
        self.river_mask: np.ndarray = (
            river_mask.astype(bool)
            if river_mask is not None
            else np.zeros_like(self.altitude, dtype=bool)
        )

        # Initialize water and tracking
        self.water_height: np.ndarray = np.zeros_like(self.altitude, dtype=np.float32)
        self.active_cells_coords: Set[Tuple[int, int]] = set(
            zip(*np.where(self.is_source))
        )

        # Tracking
        self.simulation_time_minutes: int = 0
        self.overflow_time_minutes: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self.uniform_rain: bool = True
        self._flow_step_count: int = 0

    def apply_rainfall(self, rain_mm: float) -> None:
        """
        Apply uniform or spatially-distributed rainfall to the domain.

        Parameters
        ----------
        rain_mm : float
            Rainfall depth [mm] to apply.
        """
        water_to_add_meters = float(rain_mm) / 1000.0
        if water_to_add_meters <= 0:
            return

        if self.uniform_rain:
            # Apply uniformly to all valid cells
            self.water_height[self._valid_mask] += water_to_add_meters

            # Update active cells: keep top 20% by water height
            water_flat = self.water_height.ravel()
            valid_flat = self._valid_mask.ravel()
            n_active_max = max(100, int(water_flat.size * 0.20))

            if valid_flat.sum() > n_active_max:
                threshold_active = float(
                    np.partition(
                        water_flat[valid_flat],
                        max(0, valid_flat.sum() - n_active_max),
                    )[max(0, valid_flat.sum() - n_active_max)]
                )
            else:
                threshold_active = 0.0

            ys, xs = np.where((self.water_height > threshold_active) & self._valid_mask)
            self.active_cells_coords = set(zip(ys.tolist(), xs.tolist()))

        else:
            # Apply to source areas or river
            if np.any(self.is_source):
                self.water_height[self.is_source] += water_to_add_meters
                ys, xs = np.where(self.is_source)
                self.active_cells_coords.update(zip(ys, xs))
            elif np.any(self.river_mask):
                self.water_height[self.river_mask] += water_to_add_meters * 0.2
                ys, xs = np.where(self.river_mask)
                self.active_cells_coords.update(zip(ys, xs))
            else:
                # Fallback to uniform distribution
                self.water_height += water_to_add_meters
                ys, xs = np.where(self.water_height > 0)
                self.active_cells_coords.update(zip(ys, xs))

    def advance_flow(self) -> None:
        """
        Execute one time step of water redistribution via the diffusion-wave model.

        Water flows from higher to lower cells based on free-surface elevation 
        gradients, subject to the diffusion_rate constraint.
        """
        if not self.active_cells_coords:
            return

        newly_active: Set[Tuple[int, int]] = set()
        to_deactivate: Set[Tuple[int, int]] = set()
        prev = self.water_height.copy()
        H, W = self.height, self.width

        for y, x in list(self.active_cells_coords):
            cur_w = prev[y, x]
            if cur_w <= 1e-3:
                to_deactivate.add((y, x))
                continue

            cur_total = self.altitude[y, x] + cur_w

            # Find neighbours (store local distance for slope weighting)
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        dist = 1.41421356 if (dx != 0 and dy != 0) else 1.0
                        neigh.append((ny, nx, dist))

            if not neigh:
                continue

            ny = [n[0] for n in neigh]
            nx = [n[1] for n in neigh]
            dists = np.asarray([n[2] for n in neigh], dtype=np.float32)
            n_total = self.altitude[ny, nx] + prev[ny, nx]
            mask_lower = n_total < cur_total

            if not np.any(mask_lower):
                to_deactivate.add((y, x))
                continue

            lower_coords = np.array([(n[0], n[1]) for n in neigh], dtype=np.int32)[mask_lower]
            lower_total = n_total[mask_lower]
            lower_dists = dists[mask_lower]
            diffs = np.asarray(cur_total - lower_total, dtype=np.float32)

            # Weight by local slope to reduce unrealistic lateral spreading
            slopes = np.maximum(diffs / np.maximum(lower_dists, 1e-6), 0.0)

            # Keep only top-2 steepest receivers (more channelized/realistic flow)
            max_receivers = 2
            if slopes.size > max_receivers:
                keep_idx = np.argsort(slopes)[-max_receivers:]
                lower_coords = lower_coords[keep_idx]
                lower_total = lower_total[keep_idx]
                diffs = diffs[keep_idx]
                slopes = slopes[keep_idx]

            total_diff = float(np.sum(slopes))

            if total_diff <= 0:
                continue

            move_amount = cur_w * self.diffusion_rate
            if move_amount <= 0:
                continue

            for i, (ny2, nx2) in enumerate(lower_coords):
                diff = cur_total - lower_total[i]
                frac = float(slopes[i]) / total_diff
                wmv = min(move_amount * frac, diff / 2.0)
                wmv = min(wmv, max(0.0, float(self.water_height[y, x])))

                if wmv > 0:
                    self.water_height[y, x] -= wmv
                    self.water_height[ny2, nx2] += wmv
                    newly_active.add((ny2, nx2))

            # Clamp negative values
            if self.water_height[y, x] < 0:
                self.water_height[y, x] = 0.0

        # Update active cells
        self.active_cells_coords.difference_update(to_deactivate)
        self.active_cells_coords.update(
            (ny, nx)
            for (ny, nx) in newly_active
            if self._valid_mask[ny, nx]
        )

        # Periodic cleanup
        self._flow_step_count += 1
        if self._flow_step_count % 50 == 0:
            self.active_cells_coords = {
                (y, x)
                for (y, x) in self.active_cells_coords
                if self.water_height[y, x] > 1e-6
            }

    def record_diagnostics(self, time_step_minutes: int) -> None:
        """
        Record diagnostic metrics for the current simulation state.

        Parameters
        ----------
        time_step_minutes : int
            Duration of the current time step [min].
        """
        self.simulation_time_minutes += int(time_step_minutes)
        inundated = self.water_height > self.flood_threshold

        # Track overflow time
        if self.overflow_time_minutes is None and np.any(
            inundated & ~self.is_source
        ):
            self.overflow_time_minutes = self.simulation_time_minutes

        # Append metrics to history
        self.history.append(
            {
                "time_minutes": self.simulation_time_minutes,
                "flooded_percent": float(np.sum(inundated))
                / float(inundated.size)
                * 100.0,
                "active_cells": int(len(self.active_cells_coords)),
                "max_depth": float(
                    np.nanmax(np.clip(self.water_height, 0, None))
                    if self.water_height.size > 0
                    else 0.0
                ),
                "total_water_volume_m3": float(
                    np.sum(self.water_height * self.cell_area)
                ),
                "water_height_snapshot": self.water_height.copy(),  # Store snapshot for animation
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current simulation state.

        Returns
        -------
        dict
            Dictionary containing key statistics:
            - simulation_time_minutes
            - max_water_depth
            - total_water_volume
            - flooded_area_percent
            - overflow_time_minutes
        """
        return {
            "simulation_time_minutes": self.simulation_time_minutes,
            "max_water_depth": float(np.nanmax(self.water_height)),
            "total_water_volume_m3": float(
                np.sum(self.water_height * self.cell_area)
            ),
            "flooded_area_percent": (
                float(np.sum(self.water_height > self.flood_threshold))
                / self.water_height.size
                * 100.0
            ),
            "overflow_time_minutes": self.overflow_time_minutes,
        }


class NumpyDiffusionWaveEngine(DiffusionWaveFloodModel):
    """
    Compat layer com a API local esperada no app.

    Mantém o mesmo motor numérico de `DiffusionWaveFloodModel`, apenas
    expondo nomes de métodos equivalentes:
    - `add_water()` -> `apply_rainfall()`
    - `run_flow_step()` -> `advance_flow()`
    - `update_stats()` -> `record_diagnostics()`
    """

    def add_water(self, rain_mm: float) -> None:
        self.apply_rainfall(rain_mm)

    def run_flow_step(self) -> None:
        self.advance_flow()

    def update_stats(self, time_step_minutes: int) -> None:
        self.record_diagnostics(time_step_minutes)


# Alias: usar a versão D8 melhorada como padrão
GamaFloodModelNumpy = GamaFloodModelD8

# Manter referência à versão legacy de difusão-onda para compatibilidade
DiffusionWaveFloodModelLegacy = NumpyDiffusionWaveEngine
