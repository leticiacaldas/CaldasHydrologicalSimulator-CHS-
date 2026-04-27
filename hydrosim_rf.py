from shapes import apply_custom_styles, create_header

# =============================================================================
# HydroSim-RF: A Hybrid Raster-Based Urban Flood Simulation Framework
# with Random Forest Inundation Probability Estimation
#
# Description:
#   Interactive web application for rapid 2-D flood inundation simulation
#   over Digital Elevation Models (DEMs). The hydrodynamic core implements
#   a diffusion-wave approximation via a vectorised NumPy solver
#   (DiffusionWaveFloodModel). A Random Forest classifier (scikit-learn) is
#   trained on DEM-derived topographic indices to estimate spatial flood
#   probability without requiring calibration data.
#
# Dependencies:
#   streamlit, numpy, pandas, matplotlib, rasterio, geopandas, contextily,
#   scipy, scikit-learn, Pillow, imageio-ffmpeg (optional, for MP4 export)
#
# Usage:
#   streamlit run hydrosim_rf.py
#
# Licence: MIT
# =============================================================================

# Standard library and third-party imports
import os
import io
import sys
import time
import shutil
import tempfile
import subprocess
import contextlib
from typing import Optional, Tuple, Any, Dict, cast
import zipfile

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.features import rasterize
import geopandas as gpd
import contextily as ctx
from scipy import ndimage
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.cluster import DBSCAN
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)

# Feature-flag para LISFLOOD (desativado por padrão)
ENABLE_LISFLOOD = str(os.environ.get("ENABLE_LISFLOOD", "")
                      ).strip().lower() in {"1", "true", "yes", "on"}


class DiffusionWaveFloodModel:
    """Vectorised 2-D diffusion-wave flood inundation solver (NumPy backend).

    The model implements a simplified storage-cell approach in which water
    redistributes from each cell to lower-elevation neighbours in proportion
    to the free-surface head difference, up to a maximum fraction
    *diffusion_rate* per time step. The formulation is analogous to the
    zero-inertia (diffusion-wave) simplification of the shallow-water
    equations (Hunter et al., 2005; Neal et al., 2012).

    No-data cells in the DEM are filled with the domain median prior to
    simulation to prevent spurious sinks. Water volume is conserved to
    machine precision within the active cell set.

    Parameters
    ----------
    dem_data : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m a.s.l.].
    sources_mask : np.ndarray, shape (H, W), dtype uint8 or bool
        Binary mask identifying rainfall source cells (e.g. catchment
        area or watercourse polygon).
    diffusion_rate : float
        Fraction of available water depth that may leave a cell per
        time step (dimensionless, 0 < diffusion_rate ≤ 1).
    flood_threshold : float
        Minimum water depth [m] used to classify a cell as inundated
        for reporting purposes.
    cell_size_meters : float
        Planimetric cell dimension [m]; used to compute surface area
        and volumetric water balance.
    river_mask : np.ndarray or None, optional
        Binary mask of river/channel cells. When provided, water added
        via rainfall is routed preferentially through channel cells.

    Attributes
    ----------
    water_height : np.ndarray, shape (H, W), dtype float32
        Current water depth [m] at each grid cell.
    simulation_time_minutes : int
        Elapsed simulation time [min] since initialisation.
    overflow_time_minutes : int or None
        Time [min] at which surface water first exceeded *flood_threshold*
        outside the source area; None if overflow has not yet occurred.
    history : list of dict
        Timestamped record of domain-wide diagnostics appended by
        :meth:`record_diagnostics` at every time step.

    References
    ----------
    Hunter, N. M., Bates, P. D., Horritt, M. S., De Roo, A. P. J., &
        Werner, M. G. F. (2005). Utility of different data types for
        calibrating flood inundation models within a GLUE framework.
        *Hydrology and Earth System Sciences*, 9(4), 412–430.
    Neal, J., Schumann, G., & Bates, P. (2012). A subgrid channel model
        for simulating river hydraulics and floodplain inundation over
        large and data sparse areas. *Water Resources Research*, 48(11).
    """

    def __init__(self, dem_data: np.ndarray, sources_mask: np.ndarray, diffusion_rate: float, flood_threshold: float, cell_size_meters: float, river_mask: Optional[np.ndarray] = None):
        self.height, self.width = dem_data.shape
        self.diffusion_rate = float(diffusion_rate)
        self.flood_threshold = float(flood_threshold)
        self.cell_area = float(cell_size_meters) * float(cell_size_meters)
        # Substituir NaN no DEM por mediana local para evitar propagação para water_height
        _alt = dem_data.astype(np.float32)
        if not np.isfinite(_alt).all():
            _median = float(np.nanmedian(_alt)) if np.isfinite(_alt).any() else 0.0
            _alt = np.where(np.isfinite(_alt), _alt, _median)
        self.altitude = _alt
        # Máscara de células válidas (não-NaN no DEM original)
        self._valid_mask = np.isfinite(dem_data.astype(np.float32))
        self.is_source = (sources_mask.astype(
            bool) if sources_mask is not None else np.zeros_like(self.altitude, dtype=bool))
        self.river_mask = (river_mask.astype(
            bool) if river_mask is not None else np.zeros_like(self.altitude, dtype=bool))
        self.water_height = np.zeros_like(self.altitude, dtype=np.float32)
        self.active_cells_coords = set(zip(*np.where(self.is_source)))
        self.simulation_time_minutes = 0
        self.overflow_time_minutes: Optional[int] = None
        self.history: list[dict] = []
        self.uniform_rain: bool = True

    def apply_rainfall(self, rain_mm: float):
        water_to_add_meters = float(rain_mm) / 1000.0
        if water_to_add_meters <= 0:
            return
        # Chuva uniforme: aplica apenas em células válidas do DEM
        if self.uniform_rain:
            self.water_height[self._valid_mask] += water_to_add_meters
            # Limitar active_cells ao top-20% por water_height para evitar O(H*W) no loop de fluxo
            # (no modo uniforme toda a grade tem água após o 1º ciclo)
            water_flat = self.water_height.ravel()
            valid_flat = self._valid_mask.ravel()
            n_active_max = max(100, int(water_flat.size * 0.20))
            threshold_active = float(np.partition(
                water_flat[valid_flat], max(0, valid_flat.sum() - n_active_max)
            )[max(0, valid_flat.sum() - n_active_max)]) if valid_flat.sum() > n_active_max else 0.0
            ys, xs = np.where((self.water_height > threshold_active) & self._valid_mask)
            self.active_cells_coords = set(zip(ys.tolist(), xs.tolist()))
            return

        # Caso não seja uniforme: aplicar nas fontes; se não houver fontes, usar rio como fallback; se ainda assim não houver, aplicar uniforme para evitar simulação "em branco"
        if np.any(self.is_source):
            self.water_height[self.is_source] += water_to_add_meters
            ys, xs = np.where(self.is_source)
            self.active_cells_coords.update(zip(ys, xs))
        elif np.any(self.river_mask):
            self.water_height[self.river_mask] += water_to_add_meters * 0.2
            ys, xs = np.where(self.river_mask)
            self.active_cells_coords.update(zip(ys, xs))
        else:
            # Fallback: sem fontes nem rio definidos, distribuir uniformemente
            self.water_height += water_to_add_meters
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))

    def advance_flow(self):
        if not self.active_cells_coords:
            return
        newly_active, to_deactivate = set(), set()
        prev = self.water_height.copy()
        H, W = self.height, self.width
        for y, x in list(self.active_cells_coords):
            cur_w = prev[y, x]
            if cur_w <= 1e-3:
                to_deactivate.add((y, x))
                continue
            cur_total = self.altitude[y, x] + cur_w
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        neigh.append((ny, nx))
            if not neigh:
                continue
            ny, nx = zip(*neigh)
            n_total = self.altitude[ny, nx] + prev[ny, nx]
            mask_lower = n_total < cur_total
            if not np.any(mask_lower):
                to_deactivate.add((y, x))
                continue
            lower_coords = np.array(neigh)[mask_lower]
            lower_total = n_total[mask_lower]
            total_diff = float(np.sum(cur_total - lower_total))
            if total_diff <= 0:
                continue
            move_amount = cur_w * self.diffusion_rate
            if move_amount <= 0:
                continue
            for i, (ny2, nx2) in enumerate(lower_coords):
                diff = cur_total - lower_total[i]
                frac = float(diff) / total_diff
                wmv = min(move_amount * frac, diff / 2.0)
                # Garantir que não drenamos mais água do que existe na célula
                wmv = min(wmv, max(0.0, float(self.water_height[y, x])))
                if wmv > 0:
                    self.water_height[y, x] -= wmv
                    self.water_height[ny2, nx2] += wmv
                    newly_active.add((ny2, nx2))
            # Clamp final: eliminar underflow de ponto flutuante
            if self.water_height[y, x] < 0:
                self.water_height[y, x] = 0.0
        self.active_cells_coords.difference_update(to_deactivate)
        # Filtrar newly_active para não incluir células inválidas (NaN no DEM original)
        self.active_cells_coords.update(
            (ny, nx) for (ny, nx) in newly_active
            if self._valid_mask[ny, nx]
        )
        # Limpeza periódica: remover células com água insignificante para evitar crescimento
        # ilimitado do conjunto ativo (a cada ~50 chamadas para não impactar performance)
        if not hasattr(self, '_flow_step_count'):
            self._flow_step_count = 0
        self._flow_step_count += 1
        if self._flow_step_count % 50 == 0:
            self.active_cells_coords = {
                (y, x) for (y, x) in self.active_cells_coords
                if self.water_height[y, x] > 1e-6
            }

    def record_diagnostics(self, time_step_minutes: int):
        self.simulation_time_minutes += int(time_step_minutes)
        inundated = self.water_height > self.flood_threshold
        if self.overflow_time_minutes is None and np.any(inundated & ~self.is_source):
            self.overflow_time_minutes = self.simulation_time_minutes
        self.history.append({
            "time_minutes": self.simulation_time_minutes,
            "flooded_percent": float(np.sum(inundated)) / float(inundated.size) * 100.0,
            "active_cells": int(len(self.active_cells_coords)),
            "max_depth": float(np.nanmax(np.clip(self.water_height, 0, None))) if self.water_height.size > 0 else 0.0,
            "total_water_volume_m3": float(np.sum(self.water_height * self.cell_area)),
        })


def _load_orthoimage(img_path: str, target_shape: Tuple[int, int], target_crs) -> Optional[np.ndarray]:
    """Carrega um raster de fundo (ex. DOM ortomosaico), reamostra para target_shape e retorna RGB float [0,1]."""
    try:
        H, W = target_shape
        with rio.open(img_path) as src:
            count = min(3, src.count or 1)
            bands: list[np.ndarray] = []
            for b in range(1, count + 1):
                band = src.read(
                    b,
                    out_shape=(H, W),
                    resampling=Resampling.bilinear,
                ).astype(float)
                bands.append(band)
            if len(bands) == 1:
                rgb = np.stack([bands[0]] * 3, axis=-1)
            else:
                rgb = np.stack(bands, axis=-1)
            # Normalização robusta para [0,1]
            if not np.issubdtype(rgb.dtype, np.floating):
                rgb = rgb.astype(float)
            finite_vals = rgb[np.isfinite(rgb)]
            if finite_vals.size > 0:
                p2, p98 = np.percentile(finite_vals, (2, 98))
                p2 = max(float(p2), 0.0)
                denom = (float(p98) - p2) or 1.0
                rgb = np.clip((rgb - p2) / denom, 0, 1)
            else:
                rgb = np.zeros_like(rgb, dtype=float)
            return rgb
    except (OSError, ValueError, RuntimeError) as e:
        st.warning(f"Falha ao preparar fundo (DOM): {e}")
        return None


def _save_input_files(dem_file, vector_files):
    """Salva DEM e arquivos vetoriais (primeiro .gpkg/.shp encontrado). Retorna (dem_path, vector_path, tmp_dir)."""
    if not dem_file:
        return None, None, None
    tmp = tempfile.mkdtemp(prefix="sim_numpy_")
    dem_path = os.path.join(tmp, dem_file.name)
    with open(dem_path, "wb") as f:
        f.write(dem_file.getbuffer())
    vector_path = None
    for f in (vector_files or []):
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        if f.name.lower().endswith((".gpkg", ".shp")) and vector_path is None:
            vector_path = p
    return dem_path, vector_path, tmp


def _prepare_spatial_domain(dem_path: str, vector_path: Optional[str], grid_reduction_factor: int, river_path: Optional[str] = None):
    """Lê DEM, reamostra grade, rasteriza fontes (e opcionalmente rio) e retorna (dem, sources_mask, transform, crs, cell_size_m, river_mask)."""
    with rio.open(dem_path) as dem_src:
        crs = dem_src.crs
        t = dem_src.transform
        factor = max(1, int(grid_reduction_factor))
        h2 = max(1, dem_src.height // factor)
        w2 = max(1, dem_src.width // factor)
        dem_data = dem_src.read(1, out_shape=(
            h2, w2), resampling=Resampling.bilinear)
        # Construir novo transform com base na origem e no tamanho de pixel reamostrado
        px = abs(float(t.a)) * factor
        py = abs(float(t.e)) * factor
        transform = from_origin(float(getattr(t, 'c', 0.0) or t.c), float(
            getattr(t, 'f', 0.0) or t.f), px, py)
        # Tamanho de célula (m) aproximado
        if crs and crs.is_geographic:
            cell_size = px * 111320.0
        else:
            cell_size = px

    # Rasterizar fontes (vetor)
    
    if vector_path:
        gdf = gpd.read_file(vector_path)
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
        elif crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        sources_mask = rasterize(
            gdf.geometry,
            out_shape=dem_data.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
    else:
        sources_mask = np.zeros_like(dem_data, dtype=np.uint8)

    # Rasterizar rio (opcional)
    river_mask = np.zeros_like(dem_data, dtype=bool)
    if river_path and os.path.exists(river_path):
        try:
            rgdf = gpd.read_file(river_path)
            if rgdf.crs is None and crs is not None:
                rgdf = rgdf.set_crs(crs)
            elif crs is not None and rgdf.crs != crs:
                rgdf = rgdf.to_crs(crs)
            river_mask_arr = rasterize(
                rgdf.geometry,
                out_shape=dem_data.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )
            river_mask = np.asarray(river_mask_arr, dtype=bool)
        except Exception:
            river_mask = np.zeros_like(dem_data, dtype=bool)

    return dem_data, sources_mask, transform, crs, cell_size, river_mask


def _init_animation_figure(dem_data: np.ndarray, transform, crs, background_rgb: Optional[np.ndarray], apply_hs: bool, hs_intensity: float, basemap_source: Optional[str] = 'CartoDB.Positron', show_dem_on_basemap: bool = False):
    """Prepara a figura principal e camadas (água, realce de poças e partículas de chuva)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    bounds = array_bounds(dem_data.shape[0], dem_data.shape[1], transform)

    # Fixar limites antes de adicionar tiles (necessário para o mapa base cobrir toda a área)
    x_min, y_min, x_max, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    use_basemap = bool(basemap_source) and str(
        basemap_source).strip().lower() != 'nenhum'

    # Mapa base online (por trás). Se o usuário quiser o “mapa grande do Esri”, priorize isso como fundo.
    if use_basemap:
        with contextlib.suppress(Exception):
            src = basemap_source
            # Evitar que o contextily altere o extent (isso costuma causar o overlay virar um “mapa pequeno” no canto)
            try:
                ctx.add_basemap(ax, crs=crs, source=src,
                                zorder=0, reset_extent=False)
            except TypeError:
                ctx.add_basemap(ax, crs=crs, source=src, zorder=0)
            # Refixar limites por segurança
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Fundo: DOM/Orto se fornecido. Caso contrário, DEM opcional.
    # Quando o mapa base estiver ativo, esconda o DEM por padrão para não parecer uma grade de “células”.
    if background_rgb is not None and background_rgb.shape[:2] == dem_data.shape:
        img = (np.clip(background_rgb, 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, extent=bounds,
                  alpha=1.0 if not use_basemap else 0.85, zorder=1)
    else:
        if (not use_basemap) or bool(show_dem_on_basemap):
            dem_b = dem_data.astype(float)
            vmin, vmax = np.nanpercentile(
                dem_b, (5, 95)) if np.isfinite(dem_b).any() else (0, 1)
            ax.imshow(
                dem_b,
                extent=bounds,
                cmap="terrain",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85 if not use_basemap else 0.25,
                zorder=1,
            )

    # Água: paleta mais clara e levemente “turquesa” para não confundir com o azul do mapa base (ex.: Esri)
    water_cmap = mcolors.LinearSegmentedColormap.from_list(
        "water_layer",
        [
            (0.78, 0.98, 1.00),
            (0.00, 0.92, 0.90),
            (0.00, 0.62, 1.00),
            (0.00, 0.30, 0.72),
        ],
        N=256,
    )
    water_cmap.set_under(cast(Any, (0, 0, 0, 0.0)))
    water_layer = ax.imshow(
        np.zeros_like(dem_data, dtype=float),
        extent=bounds,
        cmap=water_cmap,
        alpha=0.0,
        vmin=0.0,
        vmax=1.0,
        zorder=10,
        interpolation='nearest',
    )

    # Realce de poças/acúmulos: camada quente (amarelo/laranja) para destacar hotspots
    puddle_cmap = mcolors.LinearSegmentedColormap.from_list(
        "puddle_hotspots",
        [
            (1.00, 0.96, 0.55),
            (1.00, 0.78, 0.20),
            (0.98, 0.40, 0.12),
        ],
        N=256,
    )
    puddle_cmap.set_under(cast(Any, (0, 0, 0, 0.0)))
    puddle_layer = ax.imshow(
        np.zeros_like(dem_data, dtype=float),
        extent=bounds,
        cmap=puddle_cmap,
        alpha=0.0,
        vmin=0.0,
        vmax=1.0,
        zorder=12,
        interpolation='nearest',
    )
    rain_particles, = ax.plot(
        [], [], '.', color='royalblue', markersize=1.0, alpha=0.7)
    title = ax.set_title("Flood Inundation Simulation  |  t = 0 h 0 min", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig, ax, water_layer, puddle_layer, rain_particles, title, bounds


def _display_raster_layer(fp: str):
    with rio.open(fp) as src:
        data = src.read(1)
        bounds = array_bounds(src.height, src.width, src.transform)
        crs = src.crs
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(
        data, (5, 95)) if np.isfinite(data).any() else (0, 1)
    ax.imshow(data, extent=bounds, cmap="Blues",
              alpha=0.8, vmin=vmin, vmax=vmax)
    with contextlib.suppress(OSError, ValueError, RuntimeError):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')
    ax.set_title(os.path.basename(fp))
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)

# ── Spatial flood mitigation analysis ──────────────────────────────────


def _identify_intervention_zones(dem: np.ndarray, flood_prob: np.ndarray, river_mask: Optional[np.ndarray] = None,
                                    prob_threshold: float = 0.7, min_slope: float = 0.01, cell_size_m: float = 1.0) -> tuple[np.ndarray, dict]:
    """Identify spatial flood mitigation opportunities from terrain and risk data.

    Applies rule-based spatial analysis to a DEM and a flood probability
    surface to delineate candidate zones for four nature-based and
    engineering interventions:

    1. **Reforestation / green infrastructure** – flat, high-risk areas
       where vegetation can increase infiltration and reduce runoff
       (Bullock & Acreman, 2003).
    2. **Flood embankments / levees** – high-risk cells adjacent to the
       river channel where linear protection structures are feasible.
    3. **Urban drainage upgrade** – topographic depressions at medium
       flood risk where improved conveyance reduces ponding.
    4. **Terrain raising / land-fill** – low-lying critical cells where
       elevation gain reduces inundation frequency.

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m].
    flood_prob : np.ndarray, shape (H, W)
        Spatial flood probability field in [0, 1] (e.g. from Random
        Forest classifier or normalised water depth).
    river_mask : np.ndarray or None, shape (H, W)
        Binary mask of river/channel cells used to buffer levee zones.
    prob_threshold : float
        Probability threshold above which a cell is classified as
        high flood risk (default 0.7).
    min_slope : float
        Normalised slope threshold (relative to 98th-percentile slope)
        below which cells are eligible for reforestation (default 0.05).
    cell_size_m : float
        Grid cell size [m] used for area and volume estimates.

    Returns
    -------
    intervention_mask : np.ndarray, shape (H, W), dtype uint8
        Raster encoding intervention type per cell
        (0=none, 1=reforestation, 2=levee, 3=drainage, 4=terrain-raising).
    suggestions : dict
        Nested dictionary containing area/volume estimates and centroid
        coordinates for each intervention category.

    References
    ----------
    Bullock, A., & Acreman, M. (2003). The role of wetlands in the
        hydrological cycle. *Hydrology and Earth System Sciences*, 7(3),
        358–389.
    """
    H, W = dem.shape
    suggestions: Dict[str, Any] = {
        "florestamento": {"areas": [], "percentual": 0.0, "beneficio_estimado": 0.0},
        "diques": {"locais": [], "comprimento_estimado": 0.0, "areas_protegidas": []},
        "sistemas_drenagem": {"locais": [], "volume_estimado": 0.0},
        "aterro_terreno": {"areas": [], "volume_estimado": 0.0},
        "reservatorios": {"locais": [], "volume_estimado": 0.0}
    }

    intervention_mask = np.zeros_like(dem, dtype=np.uint8)
    try:
        dem_f = dem.astype(float)
        gy, gx = np.gradient(dem_f)
        # Converter gradiente de pixel para m/m: np.gradient retorna diferença por pixel,
        # dividir por cell_size_m converte para diferença por metro (adimensional, m/m)
        cs = max(1e-9, float(cell_size_m))
        slope = np.hypot(gx / cs, gy / cs)

        # Normalizar declividade para percentil 98 para tornar o threshold do slider robusto
        # independentemente da escala do DEM ou resolução da grade
        slope_p98 = float(np.nanpercentile(slope, 98)) if np.isfinite(slope).any() else 1.0
        slope_norm = slope / max(1e-9, slope_p98)  # 0..1, onde 1 = declive muito acentuado

        # Risco de inundação (booleano)
        prob_f = np.clip(flood_prob.astype(float), 0.0, 1.0)
        high_flood_risk = prob_f >= float(prob_threshold)
        # Médio risco: valores intermediários para alimentar drenagem
        medium_flood_risk = (prob_f >= 0.25) & (prob_f < float(prob_threshold))

        # 1) Florestamento — usar declividade normalizada para que min_slope seja comparável
        # entre DEMs de diferentes resoluções. min_slope=0.05 → apenas 5% dos declives mais suaves
        gentle_slope = slope_norm < float(min_slope)
        forest_candidates = high_flood_risk & gentle_slope
        # Fallback: se nenhuma área for encontrada, relaxar threshold de declividade
        if not np.any(forest_candidates) and np.any(high_flood_risk):
            gentle_slope = slope_norm < min(0.5, float(min_slope) * 5)
            forest_candidates = high_flood_risk & gentle_slope
        if np.any(forest_candidates):
            labeled_forest, num_forest = cast(
                Tuple[np.ndarray, int], ndimage.label(forest_candidates))
            for i in range(1, num_forest + 1):
                area_mask = labeled_forest == i
                area_size = int(np.sum(area_mask))
                if area_size > 10:
                    y_coords, x_coords = np.where(area_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["florestamento"]["areas"].append({
                        "centroide": (centroid_x, centroid_y),
                        "tamanho_pixels": area_size,
                        "beneficio_estimado": float(min(1.0, area_size / (H * W) * 10))
                    })
                    intervention_mask[area_mask] = 1
        suggestions["florestamento"]["percentual"] = float(
            np.sum(forest_candidates)) / float(H * W) * 100.0

        # 2) Diques próximos ao rio
        if river_mask is not None and np.any(river_mask):
            structure = np.ones((7, 7), dtype=bool)
            river_bool = river_mask.astype(bool)
            river_buffer = ndimage.binary_dilation(
                river_bool, structure=structure).astype(bool)
            high_risk_near_river = (
                high_flood_risk.astype(bool) & river_buffer & (~river_bool))
            if np.any(high_risk_near_river):
                labeled_dikes, num_dikes = cast(
                    Tuple[np.ndarray, int], ndimage.label(high_risk_near_river))
                total_length = 0.0
                for i in range(1, num_dikes + 1):
                    dike_mask = labeled_dikes == i
                    if int(np.sum(dike_mask)) > 5:
                        y_coords, x_coords = np.where(dike_mask)
                        if len(y_coords) > 1:
                            coords = np.column_stack([x_coords, y_coords])
                            clustering = DBSCAN(
                                eps=2, min_samples=3).fit(coords)
                            for cluster_id in set(clustering.labels_):
                                if cluster_id == -1:
                                    continue
                                cluster_coords = coords[clustering.labels_ == cluster_id]
                                if len(cluster_coords) > 2:
                                    length = float(len(cluster_coords)) * 0.5
                                    total_length += length
                                    centroid = np.mean(cluster_coords, axis=0)
                                    suggestions["diques"]["locais"].append({
                                        "centroide": (float(centroid[0]), float(centroid[1])),
                                        "comprimento_estimado": length,
                                        "areas_protegidas": int(len(cluster_coords))
                                    })
                                    intervention_mask[dike_mask] = 2
                suggestions["diques"]["comprimento_estimado"] = total_length

        # 3) Sistemas de drenagem
        local_min = ndimage.minimum_filter(dem_f, size=5) == dem_f
        drainage_candidates = local_min & medium_flood_risk
        if np.any(drainage_candidates):
            volume_total = 0.0
            labeled_drainage, num_drainage = cast(
                Tuple[np.ndarray, int], ndimage.label(drainage_candidates))
            for i in range(1, num_drainage + 1):
                drain_mask = labeled_drainage == i
                area_size = int(np.sum(drain_mask))
                if area_size > 3:
                    depression_depth = float(np.percentile(
                        dem_f[drain_mask], 25) - np.min(dem_f[drain_mask]))
                    volume = area_size * max(0.1, depression_depth)
                    volume_total += volume
                    y_coords, x_coords = np.where(drain_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["sistemas_drenagem"]["locais"].append({
                        "centroide": (centroid_x, centroid_y),
                        "volume_estimado": float(volume),
                        "area_pixels": area_size
                    })
                    intervention_mask[drain_mask] = 3
            suggestions["sistemas_drenagem"]["volume_estimado"] = float(
                volume_total)

        # 4) Aterro em áreas baixas críticas
        valid = dem_f[np.isfinite(dem_f)]
        p25 = float(np.percentile(valid, 25)) if valid.size > 0 else (
            float(np.nanmin(dem_f)) if np.isfinite(dem_f).any() else 0.0)
        low_areas = dem_f < p25
        critical_low_areas = low_areas & high_flood_risk
        if np.any(critical_low_areas):
            volume_total = 0.0
            labeled_fill, num_fill = cast(
                Tuple[np.ndarray, int], ndimage.label(critical_low_areas))
            for i in range(1, num_fill + 1):
                fill_mask = labeled_fill == i
                area_size = int(np.sum(fill_mask))
                if area_size > 5:
                    vals = dem_f[fill_mask]
                    fill_height = float(
                        np.percentile(vals, 75) - np.mean(vals))
                    volume = area_size * max(0.5, fill_height)
                    volume_total += volume
                    y_coords, x_coords = np.where(fill_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["aterro_terreno"]["areas"].append({
                        "centroide": (centroid_x, centroid_y),
                        "volume_necessario": float(volume),
                        "area_pixels": area_size,
                        "altura_media_aterro": float(fill_height)
                    })
                    intervention_mask[fill_mask] = 4
            suggestions["aterro_terreno"]["volume_estimado"] = float(
                volume_total)

        total_benefit = (
            suggestions["florestamento"]["percentual"] * 0.1 +
            len(suggestions["diques"]["locais"]) * 0.3 +
            len(suggestions["sistemas_drenagem"]["locais"]) * 0.2 +
            len(suggestions["aterro_terreno"]["areas"]) * 0.4
        )
        suggestions["beneficio_total_estimado"] = float(
            min(10.0, total_benefit))
    except Exception as e:
        print(f"Erro na análise de mitigação: {e}")
    return intervention_mask, suggestions


def _resolve_icon_paths(icon_dir: Optional[str] = None) -> Dict[int, Optional[str]]:
    """Descobre caminhos de ícones por classe usando pastas conhecidas e um diretório opcional.
    Classes: 1=florestamento, 2=diques, 3=drenagem, 4=aterro.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    if icon_dir:
        candidates.append(icon_dir)
    candidates.extend([
        os.path.join(base_dir, 'logos', 'icons'),
        os.path.join(base_dir, 'icons'),
        os.path.join(base_dir, 'logos'),
    ])

    def _find_icon(names: list[str]) -> Optional[str]:
        for d in candidates:
            for n in names:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    icon_map: Dict[int, Optional[str]] = {
        1: _find_icon(['tree.png', 'florestamento.png', 'arvore.png']),
        2: _find_icon(['dike.png', 'diques.png', 'barragem.png']),
        3: _find_icon(['drainage.png', 'drenagem.png', 'sistemaDrenagem.png']),
        4: _find_icon(['fill.png', 'aterro.png', 'aterroNoTerreno.png']),
    }
    return icon_map


def _plot_intervention_map(dem: np.ndarray, intervention_mask: np.ndarray, suggestions: dict,
                           transform, crs, background_rgb: Optional[np.ndarray] = None,
                           use_icons: bool = False, icon_dir: Optional[str] = None, icon_size: int = 24):
    """Cria mapa visual com sugestões de mitigação."""
    bounds = array_bounds(dem.shape[0], dem.shape[1], transform)
    fig, ax = plt.subplots(figsize=(12, 10))
    # Base
    if background_rgb is not None and background_rgb.shape[:2] == dem.shape:
        img = (np.clip(background_rgb, 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, extent=bounds, alpha=0.8)
    else:
        dem_f = np.asarray(dem, dtype=float)
        finite = np.isfinite(dem_f)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(dem_f[finite], (5, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(dem_f[finite])), float(
                    np.nanmax(dem_f[finite]) + 1e-6)
            ax.imshow(dem_f, extent=bounds, cmap="terrain",
                      vmin=vmin, vmax=vmax, alpha=0.8)
        else:
            # Fallback: gradiente simples para evitar figura em branco
            grad = np.linspace(
                0, 1, dem_f.size, dtype=float).reshape(dem_f.shape)
            ax.imshow(grad, extent=bounds, cmap="Greys",
                      vmin=0, vmax=1, alpha=0.6)

    # Paleta e rótulos (RGBA 0..1)
    colors = {1: (0, 0.7, 0, 0.45), 2: (0.8, 0.2, 0.2, 0.5),
              3: (0, 0.5, 1, 0.45), 4: (0.9, 0.7, 0, 0.45)}
    labels = {1: "Reforestation / Green Infrastructure", 2: "Flood Embankments / Levees",
              3: "Drainage Upgrade", 4: "Terrain Raising"}

    # Overlay de cores por classe (mais leve e limpo que scatter por pixel)
    any_points = np.any(intervention_mask > 0)
    if any_points:
        H, W = intervention_mask.shape
        overlay = np.zeros((H, W, 4), dtype=float)
        for cls in [1, 2, 3, 4]:
            mk = intervention_mask == cls
            if np.any(mk):
                r, g, b, a = colors[cls]
                overlay[mk, 0] = r
                overlay[mk, 1] = g
                overlay[mk, 2] = b
                overlay[mk, 3] = a
        ax.imshow(overlay, extent=bounds, alpha=1.0)

    # Ícones (opcional) nas posições de centróides das sugestões
    def _try_icon(path: str, size_px: int) -> Optional[OffsetImage]:
        try:
            im = Image.open(path).convert('RGBA')
            im_arr = np.asarray(im)
            return OffsetImage(im_arr, zoom=max(0.1, size_px/64.0))
        except Exception:
            return None

    if use_icons and any_points:
        icon_map = _resolve_icon_paths(icon_dir)

        # Preparar listas de centros por classe
        centers_by_cls: dict[int, list[tuple[float, float]]] = {
            1: [], 2: [], 3: [], 4: []}
        from rasterio.transform import xy as _xy
        # Florestamento
        for a in suggestions.get('florestamento', {}).get('areas', []):
            cx, cy = a.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[1].append((xw[0], yw[0]))
        # Diques
        for d in suggestions.get('diques', {}).get('locais', []):
            cx, cy = d.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[2].append((xw[0], yw[0]))
        # Drenagem
        for d in suggestions.get('sistemas_drenagem', {}).get('locais', []):
            cx, cy = d.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[3].append((xw[0], yw[0]))
        # Aterro
        for a in suggestions.get('aterro_terreno', {}).get('areas', []):
            cx, cy = a.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[4].append((xw[0], yw[0]))

        # Desenhar ícones
        for cls in [1, 2, 3, 4]:
            icon_path = icon_map.get(cls)
            if not icon_path:
                continue
            img = _try_icon(icon_path, int(icon_size))
            if img is None:
                continue
            for (xw, yw) in centers_by_cls.get(cls, []):
                ab = AnnotationBbox(
                    img, (xw, yw), frameon=False, pad=0.0, zorder=20)
                ax.add_artist(ab)

    with contextlib.suppress(Exception):
        # reset_extent=False evita que o basemap reposicione o mapa de intervenções
        try:
            ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron',
                            alpha=0.5, reset_extent=False)
        except TypeError:
            ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron', alpha=0.5)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    if any_points:
        # Legenda com patches de cor
        handles = [Patch(facecolor=colors[c][:3], alpha=colors[c]
                         [3], label=labels[c]) for c in [1, 2, 3, 4]]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
    else:
        # Mensagem informativa na figura quando não há intervenções
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        ax.text(cx, cy, "No eligible zones under current parameters",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    ax.set_title("Flood Mitigation Intervention Map",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    return fig


def _build_mitigation_report(suggestions: dict, cell_size: float) -> str:
    """Generate a structured flood mitigation assessment report with scientific citations.

    The report follows the structure recommended by the EU Floods Directive
    (2007/60/EC) and incorporates quantitative estimates derived from the
    spatial analysis performed by :func:`_identify_intervention_zones`.
    Each intervention category is annotated with peer-reviewed references to
    support evidence-based decision making.

    Parameters
    ----------
    suggestions : dict
        Output dictionary from :func:`_identify_intervention_zones`
        containing per-category area/volume estimates and centroid lists.
    cell_size : float
        Grid cell size [m] used to convert pixel counts to physical units.

    Returns
    -------
    str
        Plain-text Markdown report suitable for export as .txt or .md.
    """
    import time as _time

    R = []  # report lines

    # ------------------------------------------------------------------ header
    R += [
        "=" * 72,
        "FLOOD MITIGATION ASSESSMENT REPORT",
        "Generated by HydroSim-RF  |  " + _time.strftime("%Y-%m-%d %H:%M UTC", _time.gmtime()),
        "=" * 72,
        "",
        "DISCLAIMER",
        "-" * 40,
        "This report is generated automatically from a simplified rule-based",
        "spatial analysis and should be treated as a screening-level assessment",
        "only. Intervention feasibility, costs, and detailed design must be",
        "verified by qualified hydraulic and environmental engineers.",
        "",
    ]

    # ---------------------------------------------------------- overall score
    benefit_score = float(suggestions.get("beneficio_total_estimado", 0))
    if benefit_score > 7:
        rating, rating_note = "VERY HIGH", "Multiple high-priority interventions identified."
    elif benefit_score > 5:
        rating, rating_note = "HIGH", "Several significant interventions identified."
    elif benefit_score > 3:
        rating, rating_note = "MODERATE", "Targeted interventions may reduce peak flood risk."
    else:
        rating, rating_note = "LOW", "Limited intervention potential under current scenario."

    R += [
        "1. OVERALL RISK REDUCTION POTENTIAL",
        "-" * 40,
        f"   Score : {benefit_score:.2f} / 10  ({rating})",
        f"   Note  : {rating_note}",
        "",
        "   The composite score is a heuristic index combining the spatial",
        "   extent of reforestation candidates (weight 0.1), number of levee",
        "   segments (0.3), drainage nodes (0.2), and terrain-raising zones",
        "   (0.4), capped at 10. It is intended for inter-scenario comparison",
        "   only and does not represent a quantitative risk reduction estimate.",
        "",
    ]

    # helper: append a section only when content exists
    def _sec(title, lines):
        if lines:
            R.append(title)
            R.append("-" * 40)
            R.extend(lines)
            R.append("")

    # ----------------------------------------------- 1. reforestation / NbS
    forest = suggestions.get("florestamento", {})
    if forest.get("areas"):
        area_px  = sum(a.get("tamanho_pixels", 0) for a in forest["areas"])
        area_km2 = (float(area_px) * cell_size ** 2) / 1e6
        n_zones  = len(forest["areas"])
        _sec("2. REFORESTATION / GREEN INFRASTRUCTURE", [
            f"   Eligible zones          : {n_zones}",
            f"   Total area              : {area_km2:.4f} km²  "
            f"({area_km2 * 100:.2f} ha)",
            "",
            "   RATIONALE",
            "   Reforestation and restoration of riparian vegetation increase",
            "   soil infiltration capacity and interception storage, reducing",
            "   peak runoff volumes and delaying flood wave propagation.",
            "   Eligible zones are defined as high-risk cells (RF probability",
            "   >= threshold) with normalised terrain slope below the",
            "   user-specified reforestation criterion.",
            "",
            "   EXPECTED EFFECTS",
            "   - Runoff coefficient reduction: 10–30% (land-cover dependent)",
            "     (Rogger et al., 2017; O'Connell et al., 2007)",
            "   - Peak flow attenuation: 5–45% for small catchments",
            "     (Chandler et al., 2018)",
            "   - Co-benefits: carbon sequestration, biodiversity, water",
            "     quality improvement (Millennium Ecosystem Assessment, 2005)",
            "",
            "   KEY REFERENCES",
            "   Bullock, A., & Acreman, M. (2003). The role of wetlands in the",
            "     hydrological cycle. Hydrology and Earth System Sciences,",
            "     7(3), 358–389. https://doi.org/10.5194/hess-7-358-2003",
            "   O'Connell, P. E., et al. (2007). Review of impacts of rural",
            "     land use and management on flood generation. Advances in",
            "     Water Resources, 30(5), 1293–1313.",
            "     https://doi.org/10.1016/j.advwatres.2007.01.001",
            "   Rogger, M., et al. (2017). Land use change impacts on floods at",
            "     the catchment scale. Water Resources Research, 53(7),",
            "     5209–5219. https://doi.org/10.1002/2017WR020723",
        ])

    # --------------------------------------------------- 2. levees / dikes
    dikes = suggestions.get("diques", {})
    if dikes.get("locais"):
        total_length = (sum(d.get("comprimento_estimado", 0.0)
                            for d in dikes["locais"]) * cell_size)
        n_seg = len(dikes["locais"])
        _sec("3. FLOOD EMBANKMENTS / LEVEES", [
            f"   Candidate segments      : {n_seg}",
            f"   Total estimated length  : {total_length:.0f} m  "
            f"({total_length / 1000:.2f} km)",
            "",
            "   RATIONALE",
            "   Candidate levee segments are identified as contiguous clusters",
            "   of high-risk cells located within a 7-cell buffer of the river",
            "   mask. The estimated length is derived from DBSCAN cluster size",
            "   (eps=2 cells, min_samples=3) and should be treated as a lower",
            "   bound for design purposes.",
            "",
            "   EXPECTED EFFECTS",
            "   - Direct protection of floodplain areas against bankfull",
            "     and moderate return-period events",
            "   - Levee setback designs can restore natural floodplain storage",
            "     while maintaining protection levels (Opperman et al., 2009)",
            "",
            "   DESIGN CONSIDERATIONS",
            "   - Freeboard allowance: minimum 0.5 m above design flood level",
            "     (USACE, 2000; BSI PD 6697:2010)",
            "   - Residual risk from overtopping must be assessed",
            "   - Environmental impact assessment required under EU Habitats",
            "     Directive (92/43/EEC) for projects near protected areas",
            "",
            "   KEY REFERENCES",
            "   Opperman, J. J., et al. (2009). Sustainable floodplains through",
            "     large-scale reconnection to rivers. Science, 326(5959),",
            "     1487–1488. https://doi.org/10.1126/science.1178256",
            "   Scussolini, P., et al. (2016). FLOPROS: an evolving global",
            "     database of flood protection standards. Natural Hazards and",
            "     Earth System Sciences, 16, 1049–1061.",
            "     https://doi.org/10.5194/nhess-16-1049-2016",
        ])

    # ------------------------------------------- 3. drainage infrastructure
    drainage = suggestions.get("sistemas_drenagem", {})
    if drainage.get("locais"):
        total_vol = float(drainage.get("volume_estimado", 0.0)) * cell_size ** 2
        n_nodes   = len(drainage["locais"])
        _sec("4. URBAN DRAINAGE UPGRADE", [
            f"   Critical depression nodes : {n_nodes}",
            f"   Estimated drainage volume : {total_vol:.1f} m³",
            "",
            "   RATIONALE",
            "   Critical drainage nodes are local topographic minima (minimum",
            "   filter, 5-cell window) coinciding with medium flood risk",
            "   (0.25 <= RF probability < threshold). These depressions act as",
            "   preferential ponding locations during intense rainfall events.",
            "",
            "   EXPECTED EFFECTS",
            "   - Elimination or reduction of surface ponding at identified",
            "     nodes",
            "   - Integration into Sustainable Drainage Systems (SuDS) can",
            "     provide additional water quality and amenity benefits",
            "     (Woods-Ballard et al., 2015)",
            "",
            "   DESIGN GUIDANCE",
            "   - Sewer network capacity: design storm return period >= 1:30 yr",
            "     (EN 752:2017; BS EN 752-4)",
            "   - Retention/detention ponds should be sized for the 1:100 yr",
            "     critical storm duration",
            "   - Green infrastructure options (bioretention, permeable",
            "     pavements) preferred where space allows",
            "",
            "   KEY REFERENCES",
            "   Fletcher, T. D., et al. (2015). SUDS, LID, BMPs, WSUD and more",
            "     – The evolution and application of terminology surrounding",
            "     urban drainage. Urban Water Journal, 12(7), 525–542.",
            "     https://doi.org/10.1080/1573062X.2014.916314",
            "   Woods-Ballard, B., et al. (2015). The SUDS Manual (C753).",
            "     CIRIA, London.",
        ])

    # ------------------------------------------- 4. terrain raising
    fill = suggestions.get("aterro_terreno", {})
    if fill.get("areas"):
        total_vol = float(fill.get("volume_estimado", 0.0)) * cell_size ** 2
        n_zones   = len(fill["areas"])
        _sec("5. TERRAIN RAISING", [
            f"   Critical low-lying zones  : {n_zones}",
            f"   Total fill volume (est.)  : {total_vol:.1f} m³",
            "",
            "   RATIONALE",
            "   Terrain-raising candidates are cells in the lowest elevation",
            "   quartile of the domain that simultaneously exhibit high flood",
            "   risk (RF probability >= threshold). The estimated fill volume",
            "   is computed as (75th percentile elevation − mean elevation) ×",
            "   cell area, providing a conservative lower bound for earthwork",
            "   quantities.",
            "",
            "   EXPECTED EFFECTS",
            "   - Freeboard gain reduces inundation frequency of structures",
            "     located in identified zones",
            "   - Most effective for isolated low-lying areas not adjacent to",
            "     major watercourses",
            "",
            "   CONSTRAINTS",
            "   - Raising terrain in floodplains may displace flood storage",
            "     volume; compensatory storage is required under many national",
            "     planning frameworks (e.g. NPPF, England; Directive 2007/60/EC)",
            "   - Geotechnical assessment required for fill > 1 m",
            "",
            "   KEY REFERENCES",
            "   Merz, B., et al. (2010). Flood risk mapping at the local scale:",
            "     concepts and challenges. Flood Risk Management: Research and",
            "     Practice. Taylor & Francis, London.",
            "   FEMA (2021). Homeowner's Guide to Retrofitting: Six Ways to",
            "     Protect Your Home from Flooding (3rd ed.). FEMA P-312.",
        ])

    # ------------------------------------------- general recommendations
    R.append("6. GENERAL RECOMMENDATIONS")
    R.append("-" * 40)
    has_any = any([forest.get("areas"), dikes.get("locais"),
                   drainage.get("locais"), fill.get("areas")])
    if not has_any:
        R += [
            "   No critical intervention zones were identified under the current",
            "   parameter configuration. The domain appears resilient under the",
            "   simulated precipitation scenario.",
            "",
            "   Recommended preventive actions:",
            "   - Maintain existing vegetation cover in riparian zones",
            "   - Establish real-time flood early-warning systems",
            "   - Review results with higher-resolution DEM or increased",
            "     precipitation forcing",
        ]
    else:
        if forest.get("areas"):
            R.append("   [1] Prioritise reforestation on flat high-risk terrain:")
            R.append("       lowest unit cost; maximum co-benefit value (Rogger et al., 2017).")
        if dikes.get("locais"):
            R.append("   [2] Commission detailed hydraulic design of levee segments")
            R.append("       identified in Section 3 using a calibrated 1-D/2-D model")
            R.append("       (e.g. HEC-RAS 2D, LISFLOOD-FP, Delft3D FM).")
        if drainage.get("locais"):
            R.append("   [3] Upgrade drainage conveyance at critical nodes (Section 4);")
            R.append("       integrate SuDS where space permits (Woods-Ballard et al., 2015).")
        if fill.get("areas"):
            R.append("   [4] Evaluate terrain-raising feasibility at lowest-lying zones")
            R.append("       (Section 5); ensure compensatory floodplain storage is")
            R.append("       provided as required by applicable planning regulations.")
        R += [
            "",
            "   INTEGRATED FLOOD RISK MANAGEMENT NOTE",
            "   Single-measure interventions rarely achieve adequate risk reduction.",
            "   A portfolio approach combining nature-based solutions, structural",
            "   measures, and enhanced conveyance is recommended, consistent with",
            "   the EU Floods Directive (2007/60/EC) and IPCC AR6 Chapter 4",
            "   (Ranasinghe et al., 2021).",
        ]

    # --------------------------------------------------------- references
    R += [
        "",
        "=" * 72,
        "FULL REFERENCE LIST",
        "=" * 72,
        "",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
        "  https://doi.org/10.1023/A:1010933404324",
        "",
        "Bullock, A., & Acreman, M. (2003). The role of wetlands in the",
        "  hydrological cycle. Hydrology and Earth System Sciences, 7(3), 358–389.",
        "  https://doi.org/10.5194/hess-7-358-2003",
        "",
        "Chandler, K. R., et al. (2018). Individual tree-scale rainfall",
        "  partitioning in a temperate Atlantic oakwood. Hydrological Processes,",
        "  32(9), 1238–1253. https://doi.org/10.1002/hyp.11505",
        "",
        "European Commission (2007). Directive 2007/60/EC of the European",
        "  Parliament and of the Council on the assessment and management of",
        "  flood risks. Official Journal of the European Union, L 288/27.",
        "",
        "FEMA (2021). Homeowner's Guide to Retrofitting: Six Ways to Protect",
        "  Your Home from Flooding (3rd ed.). FEMA P-312. Federal Emergency",
        "  Management Agency, Washington, D.C.",
        "",
        "Fletcher, T. D., et al. (2015). SUDS, LID, BMPs, WSUD and more –",
        "  The evolution and application of terminology surrounding urban",
        "  drainage. Urban Water Journal, 12(7), 525–542.",
        "  https://doi.org/10.1080/1573062X.2014.916314",
        "",
        "Hunter, N. M., Bates, P. D., Horritt, M. S., De Roo, A. P. J., &",
        "  Werner, M. G. F. (2005). Utility of different data types for",
        "  calibrating flood inundation models within a GLUE framework.",
        "  Hydrology and Earth System Sciences, 9(4), 412–430.",
        "  https://doi.org/10.5194/hess-9-412-2005",
        "",
        "Millennium Ecosystem Assessment (2005). Ecosystems and Human",
        "  Well-being: Synthesis. Island Press, Washington, D.C.",
        "",
        "Neal, J., Schumann, G., & Bates, P. (2012). A subgrid channel model",
        "  for simulating river hydraulics and floodplain inundation over large",
        "  and data sparse areas. Water Resources Research, 48(11), W11506.",
        "  https://doi.org/10.1029/2012WR012514",
        "",
        "O'Connell, P. E., et al. (2007). Review of impacts of rural land use",
        "  and management on flood generation. Advances in Water Resources,",
        "  30(5), 1293–1313. https://doi.org/10.1016/j.advwatres.2007.01.001",
        "",
        "Opperman, J. J., et al. (2009). Sustainable floodplains through",
        "  large-scale reconnection to rivers. Science, 326(5959), 1487–1488.",
        "  https://doi.org/10.1126/science.1178256",
        "",
        "Ranasinghe, R., et al. (2021). Climate change information for",
        "  regional impact and for risk assessment. In: Climate Change 2021:",
        "  The Physical Science Basis. IPCC AR6, Chapter 12. Cambridge",
        "  University Press.",
        "",
        "Rogger, M., et al. (2017). Land use change impacts on floods at the",
        "  catchment scale: Challenges and opportunities for future research.",
        "  Water Resources Research, 53(7), 5209–5219.",
        "  https://doi.org/10.1002/2017WR020723",
        "",
        "Scussolini, P., et al. (2016). FLOPROS: an evolving global database",
        "  of flood protection standards. Natural Hazards and Earth System",
        "  Sciences, 16, 1049–1061. https://doi.org/10.5194/nhess-16-1049-2016",
        "",
        "Woods-Ballard, B., et al. (2015). The SUDS Manual (C753).",
        "  CIRIA, London. ISBN 978-0-86017-760-9",
        "",
        "=" * 72,
        "END OF REPORT",
        "=" * 72,
    ]

    return "\n".join(R)


# ── Random Forest flood probability estimator ──────────────────────────

def _compute_topographic_features(dem: np.ndarray) -> np.ndarray:
    """Compute topographic predictor variables from a DEM raster.

    Derives two normalised features used as inputs to the Random Forest
    classifier: (1) percentile-normalised elevation and (2) normalised
    slope magnitude estimated via first-order finite differences
    (np.gradient). Both features are clipped to [0, 1].

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation array [m].

    Returns
    -------
    X : np.ndarray, shape (H*W, 2)
        Feature matrix with columns [elevation_norm, slope_norm].
    """
    H, W = dem.shape
    dem = dem.astype(float)
    # Normalização robusta por percentis
    if np.isfinite(dem).any():
        p2, p98 = np.nanpercentile(dem, (2, 98))
        denom = (p98 - p2) if (p98 - p2) != 0 else 1.0
        dem_norm = np.clip((dem - p2) / denom, 0, 1)
    else:
        dem_norm = np.zeros_like(dem, dtype=float)
    # Declividade aproximada
    gy, gx = np.gradient(np.nan_to_num(dem, nan=0.0))
    slope = np.hypot(gx, gy)
    if np.isfinite(slope).any():
        s_p2, s_p98 = np.nanpercentile(slope, (2, 98))
        s_denom = (s_p98 - s_p2) if (s_p98 - s_p2) != 0 else 1.0
        slope_norm = np.clip((slope - s_p2) / s_denom, 0, 1)
    else:
        slope_norm = np.zeros_like(slope, dtype=float)
    X = np.stack([dem_norm, slope_norm], axis=-1).reshape(H * W, 2)
    
    # Final cleanup to ensure no NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    X = np.clip(X, 0, 1)
    
    return X


def _train_flood_classifier(dem: np.ndarray, water: np.ndarray, threshold: float, n_estimators: int = 100, max_depth: int = 12) -> RandomForestClassifier:
    """Train a Random Forest binary classifier for flood inundation prediction.

    The classifier is trained on topographic indices derived from the DEM
    (see :func:`_compute_topographic_features`) with binary labels derived from the
    simulated water depth field. Class imbalance is addressed via
    *class_weight='balanced'* (Breiman, 2001).

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m].
    water : np.ndarray, shape (H, W)
        Simulated water depth array [m] from :class:`DiffusionWaveFloodModel`.
    threshold : float
        Water depth threshold [m] used to assign positive class labels.
    n_estimators : int, optional
        Number of decision trees in the ensemble (default 100).
    max_depth : int, optional
        Maximum tree depth; controls model complexity (default 12).

    Returns
    -------
    clf : RandomForestClassifier
        Fitted scikit-learn classifier.

    References
    ----------
    Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
    """
    X = _compute_topographic_features(dem)
    y = (water.reshape(-1) > float(threshold)).astype(np.uint8)
    
    # Ensure no NaN/Inf in X before training
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    X = np.clip(X, 0, 1)
    
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def _predict_probability(model: RandomForestClassifier, dem: np.ndarray) -> np.ndarray:
    X = _compute_topographic_features(dem)
    
    # Ensure no NaN/Inf before prediction
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    X = np.clip(X, 0, 1)
    
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        p1 = np.zeros((dem.size,), dtype=float)
    else:
        p1 = proba[:, 1]
    return p1.reshape(dem.shape)


def _plot_probability_overlay(prob: np.ndarray, transform, crs, rf_threshold: float, rf_alpha: float, dem_back: Optional[np.ndarray] = None):
    """Plota um mapa com o DEM (se fornecido) e sobrepõe a probabilidade (IA) com transparência abaixo do limiar."""
    bounds = array_bounds(prob.shape[0], prob.shape[1], transform)
    fig, ax = plt.subplots(figsize=(10, 8))
    if dem_back is not None and np.isfinite(dem_back).any():
        vmin, vmax = np.nanpercentile(dem_back, (5, 95)) if np.isfinite(
            dem_back).any() else (0, 1)
        ax.imshow(dem_back, extent=bounds, cmap="terrain",
                  alpha=0.75, vmin=vmin, vmax=vmax)
    # Prob com máscara/under transparente
    reds = cm.get_cmap("Reds").copy()
    reds.set_under(cast(Any, (0, 0, 0, 0.0)))
    masked = np.ma.masked_less_equal(prob, rf_threshold)
    im = ax.imshow(masked, extent=bounds, cmap=reds, vmin=max(
        1e-6, rf_threshold + 1e-6), vmax=1.0, alpha=rf_alpha)
    with contextlib.suppress(Exception):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')
    ax.set_title("RF Flood Probability")
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="Probability")  # type: ignore
    st.pyplot(fig, clear_figure=True)


def _probability_geotiff_bytes(prob: np.ndarray, transform, crs) -> bytes:
    """Gera um GeoTIFF em memória com a probabilidade (0..1)."""
    prob = prob.astype(np.float32)
    profile = {
        'driver': 'GTiff',
        'height': prob.shape[0],
        'width': prob.shape[1],
        'count': 1,
        'dtype': 'float32',
        'compress': 'deflate',
        'nodata': np.nan,
        'transform': transform,
        'crs': crs,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(prob, 1)
        return memfile.read()


def _probability_rgba_geotiff_bytes(prob: np.ndarray, transform, crs, vmin: float = 0.0, vmax: float = 1.0, cmap_name: str = "Reds", under_transparent: bool = True) -> bytes:
    """Gera um GeoTIFF RGBA (uint8) com estilo aplicado (colormap) e transparência abaixo de vmin.

    - vmin: valores <= vmin ficam transparentes quando under_transparent=True.
    - vmax: topo do mapeamento de cores.
    - cmap_name: nome do colormap Matplotlib.
    """
    data = np.asarray(prob, dtype=float)
    H, W = data.shape
    # Normalização para 0..1 dentro do intervalo informado
    vmin_eff = max(1e-6, float(vmin))
    vmax_eff = max(vmin_eff + 1e-6, float(vmax))
    norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff, clip=True)

    cmap = cm.get_cmap(cmap_name).copy()
    # Transparência para valores 'under' e mascarados
    if under_transparent:
        cmap.set_under(cast(Any, (0, 0, 0, 0.0)))
        cmap.set_bad(cast(Any, (0, 0, 0, 0.0)))
        masked = np.ma.masked_less_equal(data, vmin_eff)
        mapped = cmap(norm(masked))  # HxWx4 em float 0..1
    else:
        mapped = cmap(norm(data))

    rgba = (np.clip(mapped, 0, 1) * 255).astype(np.uint8)
    # Separar bandas
    r = rgba[:, :, 0]
    g = rgba[:, :, 1]
    b = rgba[:, :, 2]
    a = rgba[:, :, 3]

    profile = {
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': 4,
        'dtype': 'uint8',
        'compress': 'deflate',
        'transform': transform,
        'crs': crs,
        # Mantemos photometric padrão; a banda 4 serve como alpha
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(r, 1)
            dst.write(g, 2)
            dst.write(b, 3)
            dst.write(a, 4)
        return memfile.read()


def _check_docker_available(timeout: float = 6.0) -> Tuple[bool, str]:
    """Verifica se o Docker está disponível no PATH e retorna (ok, detalhe)."""
    try:
        res = subprocess.run(["docker", "--version"], capture_output=True,
                             text=True, timeout=timeout, check=False)
        if res.returncode == 0:
            out = (res.stdout or res.stderr or "").strip()
            return True, out
        return False, (res.stderr or res.stdout or "").strip()
    except FileNotFoundError:
        return False, "binário 'docker' não encontrado no PATH"
    except subprocess.TimeoutExpired:
        return False, "tempo esgotado ao consultar 'docker --version'"
    except (OSError, RuntimeError) as e:
        return False, f"erro ao consultar docker: {e}"


def _check_trimesh_installed() -> Tuple[bool, str]:
    try:
        # Primeiro tenta obter versão via metadata
        try:
            from importlib import metadata as importlib_metadata  # py3.8+
            ver = importlib_metadata.version("trimesh")  # type: ignore
            return True, f"trimesh {ver}"
        # type: ignore[attr-defined]
        except importlib_metadata.PackageNotFoundError:
            pass
        except (ValueError, RuntimeError):
            pass
        # Fallback: tenta import direto
        try:
            trimesh = importlib_mod.import_module("trimesh")  # type: ignore
            ver = getattr(trimesh, "__version__", "")
            label = f"trimesh {ver}".strip() if ver else "trimesh"
            return True, label
        except ImportError:
            return False, "não instalado"
        except (RuntimeError, AttributeError) as e:
            return False, f"erro ao checar: {e}"
    except (RuntimeError, AttributeError) as e:
        return False, f"erro ao checar: {e}"


def _install_trimesh() -> Tuple[int, str, str]:
    """Instala trimesh no Python atual. Retorna (code, stdout, stderr)."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "trimesh"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout, res.stderr
    except (OSError, RuntimeError) as e:
        return 1, "", str(e)


def create_lisflood_minimal_xml(output_path, replacements):
    """Cria um XML mínimo compatível com LISFLOOD - VERSÃO ROBUSTA"""
    import xml.etree.ElementTree as ET

    # Valores padrão COMPLETOS
    default_replacements = {
        "MaskMap": "/input/MASK.map",
        "Ldd": "/input/Ldd.map",
        "PathOut": "/input/output/",
        "StepStart": "01/01/2024 00:00",
        "StepEnd": "01/01/2024 01:00",
        "DtSec": "3600",
        "DtInit": "3600",
        "RepStep": "1",
        "CalendarConvention": "proleptic_gregorian",
    }

    # Atualiza com os valores fornecidos
    default_replacements.update(replacements or {})

    # Construção XML
    root = ET.Element("lisfloodSettings")
    lfuser = ET.SubElement(root, "lfuser")

    params = [
        ("MaskMap", default_replacements["MaskMap"]),
        ("Ldd", default_replacements["Ldd"]),
        ("PathOut", default_replacements["PathOut"]),
        ("StepStart", default_replacements["StepStart"]),
        ("StepEnd", default_replacements["StepEnd"]),
        ("DtSec", default_replacements["DtSec"]),
        ("DtInit", default_replacements["DtInit"]),
        # Sinônimos para compatibilidade
        ("timestep", default_replacements["DtSec"]),
        ("timestep_init", default_replacements["DtInit"]),
        ("CalendarType", default_replacements["CalendarConvention"]),
        ("RepStep", default_replacements["RepStep"]),
        ("CalendarConvention", default_replacements["CalendarConvention"]),
        ("simulateWaterBodies", "0"),
        ("simulateLakes", "0"),
        ("simulateReservoirs", "0"),
        ("simulateSnow", "0"),
        ("simulateGlaciers", "0"),
        ("simulateFrost", "0"),
        ("simulateInfiltration", "0"),
        ("simulatePercolation", "0"),
        ("simulateGroundwater", "0"),
        ("simulateCapillaryRise", "0"),
        ("simulateInterception", "0"),
        ("simulateEvapotranspiration", "0"),
        ("simulateWaterQuality", "0"),
        ("simulateSediment", "0"),
        ("simulateNutrients", "0"),
        ("RepMapSteps", "1"),
        ("RepStateFiles", "0"),
        ("RepDischarge", "0"),
        ("RepStateVars", "0"),
        ("InitialConditions", "0"),
        ("InitLisflood", "1"),
    ]
    for name, value in params:
        ET.SubElement(lfuser, "textvar", name=name, value=value)

    lfbinding = ET.SubElement(root, "lfbinding")
    # Essenciais no binding (incluir variações)
    ET.SubElement(lfbinding, "textvar", name="CalendarConvention",
                  value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "textvar", name="CalendarType",
                  value=default_replacements["CalendarConvention"])  # alias
    # Algumas versões usam tag <text> em vez de <textvar>
    ET.SubElement(lfbinding, "text", name="CalendarConvention",
                  value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "text", name="CalendarType",
                  value=default_replacements["CalendarConvention"])  # alias
    ET.SubElement(lfbinding, "map", name="MASK", file="MASK.map")
    ET.SubElement(lfbinding, "map", name="LDD", file="Ldd.map")

    # Formatar e gravar
    try:
        ET.indent(root, space="\t", level=0)  # py>=3.9
    except Exception:
        pass
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # Verificação básica
    try:
        _ = ET.parse(output_path)
        print(f"✅ XML válido criado em: {output_path}")
    except ET.ParseError as e:
        print(f"❌ Erro no XML gerado: {e}")
        raise

# ========= FUNÇÕES AUXILIARES PARA PÓS-PROCESSAMENTO =========


def _plot_temporal_diagnostics(model):
    """Cria gráficos de evolução temporal"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico 1: Área inundada e volume
    times = [h['time_minutes']/60 for h in model.history]
    areas = [h['flooded_percent'] for h in model.history]
    volumes = [h['total_water_volume_m3'] for h in model.history]

    ax1.plot(times, areas, 'b-', linewidth=2, label='Área Inundada (%)')
    ax1.set_xlabel('Elapsed time (h)')
    ax1.set_ylabel('Inundated fraction (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, volumes, 'r-', linewidth=2, label='Volume (m³)')
    ax1_twin.set_ylabel('Water volume (m³)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('Inundated area and water volume', fontsize=10)

    # Gráfico 2: Profundidade e células ativas
    depths = [h['max_depth'] for h in model.history]
    cells = [h['active_cells'] for h in model.history]

    ax2.plot(times, depths, 'g-', linewidth=2, label='Profundidade Máx (m)')
    ax2.set_xlabel('Elapsed time (h)')
    ax2.set_ylabel('Maximum water depth (m)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, cells, 'orange', linewidth=2, label='Células Ativas')
    ax2_twin.set_ylabel('Active cells', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('Maximum depth and active cell count', fontsize=10)

    plt.tight_layout()
    return fig


def _render_simulation_outputs(model, _tmp_dir, anim_path, anim_format, total_rain, cell_size, sources_mask):
    """Processa resultados e gera downloads"""

    st.markdown("---")
    st.subheader("Simulation Results")

    # 1. Estatísticas finais
    # Se não houver fontes definidas (ou a simulação usou chuva uniforme), considere chuva sobre toda a área
    if np.any(sources_mask):
        rain_area_cells = np.sum(sources_mask > 0)
    else:
        rain_area_cells = sources_mask.size
    total_rain_m3 = float((total_rain/1000.0) *
                          rain_area_cells * (cell_size*cell_size))
    final_time = model.simulation_time_minutes
    h, m = divmod(final_time, 60)
    transbordo = (f"{divmod(model.overflow_time_minutes,60)[0]}h {divmod(model.overflow_time_minutes,60)[1]}m"
                  if model.overflow_time_minutes else "N/A")

    # Summary metrics table (expanded)
    last = model.history[-1]
    peak_depth_m = float(np.nanmax(np.clip(model.water_height, 0, None)))
    total_cells = int(model.water_height.size)
    inundated_cells = int(np.sum(model.water_height > model.flood_threshold))
    inundated_area_km2 = inundated_cells * (cell_size ** 2) / 1e6

    resumo = pd.DataFrame([{
        "Total precipitation [mm]": round(float(total_rain), 2),
        "Rainfall input volume [m³]": round(float(total_rain_m3), 1),
        "Simulated duration": f"{h}h {m}m",
        "Time to overflow": transbordo,
        "Peak inundated area [km²]": round(inundated_area_km2, 4),
        "Peak inundated fraction [%]": round(float(last["flooded_percent"]), 2),
        "Peak water depth [m]": round(peak_depth_m, 3),
        "Peak water volume [m³]": round(float(last["total_water_volume_m3"]), 1),
        "Active cells (final step)": int(last["active_cells"]),
        "Total grid cells": total_cells,
    }])
    st.dataframe(resumo, use_container_width=True)
    st.caption(
        f"Inundation threshold h\u002a = {model.flood_threshold:.3f} m \u00b7 "
        f"Cell size \u2248 {cell_size:.1f} m \u00b7 "
        f"Diffusion coefficient \u03b1 = {model.diffusion_rate:.2f}"
    )

    # 2. Downloads (gerar artefatos e persistir no session_state para sobreviver ao rerun)
    st.subheader("Exported Outputs")
    col1, col2, col3 = st.columns(3)

    # CSV com dados
    df = pd.DataFrame(model.history)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.session_state["dl_history_csv"] = csv_data
    with col1:
        st.download_button(
            "Diagnostics time series (CSV)",
            csv_data,
            "dados_simulacao.csv",
            "text/csv",
            key="dl_csv_now",
        )

    # Animação
    anim_bytes = None
    anim_mime = None
    anim_ext = None
    if anim_path and os.path.exists(anim_path):
        try:
            with open(anim_path, "rb") as f:
                anim_bytes = f.read()
            anim_ext = anim_format.lower()
            anim_mime = f"video/{anim_ext}" if anim_ext == 'mp4' else "image/gif"
            st.session_state["dl_anim_bytes"] = anim_bytes
            st.session_state["dl_anim_ext"] = anim_ext
            st.session_state["dl_anim_mime"] = anim_mime
        except Exception as e:
            st.error(f"Erro ao carregar animação: {e}")
    with col2:
        if anim_bytes is not None:
            _ext_label = (anim_ext or "").upper(
            ) if isinstance(anim_ext, str) else ""
            st.download_button(
                f"🎬 Animação ({_ext_label})",
                anim_bytes,
                f"simulacao.{anim_ext}",
                anim_mime,
                key="dl_anim_now",
            )
        else:
            st.info("Ative 'Pré‑visualização rápida' DESMARCADA para salvar animação")

    # Gráficos — gerar UMA vez e reutilizar para download e exibição
    graph_png_bytes = None
    fig_evolution = None
    if len(model.history) > 0:
        fig_evolution = _plot_temporal_diagnostics(model)
        buf = io.BytesIO()
        fig_evolution.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        graph_png_bytes = buf.getvalue()
        st.session_state["dl_graph_png"] = graph_png_bytes
        plt.close(fig_evolution)  # Liberar memória após salvar
    with col3:
        if graph_png_bytes is not None:
            st.download_button(
                "Diagnostic plots (PNG)",
                graph_png_bytes,
                "evolucao_simulacao.png",
                "image/png",
                key="dl_graph_now",
            )

    # Overlay PNG da água simulada (não IA) sobre DOM/DEM
    try:
        dem_last = st.session_state.get("last_dem_data")
        transform_last = st.session_state.get("last_transform")
        bg = st.session_state.get("last_background_rgb")
        if dem_last is not None and transform_last is not None:
            bounds_sim = array_bounds(
                dem_last.shape[0], dem_last.shape[1], transform_last)
            water = np.asarray(model.water_height, dtype=float)
            masked_sim = np.ma.masked_less_equal(water, float(
                max(1e-6, getattr(model, 'flood_threshold', 0.0))))
            fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
            if bg is not None:
                img = (bg * 255).astype(np.uint8) if np.issubdtype(bg.dtype,
                                                                   np.floating) and bg.max() <= 1.0 else bg.astype(np.uint8)
                ax_sim.imshow(img, extent=bounds_sim, alpha=1.0)
            else:
                dem_b = dem_last.astype(float)
                vmin_b, vmax_b = np.nanpercentile(
                    dem_b, (5, 95)) if np.isfinite(dem_b).any() else (0, 1)
                ax_sim.imshow(dem_b, extent=bounds_sim, cmap="terrain",
                              vmin=vmin_b, vmax=vmax_b, alpha=0.85)
            # Usar o mesmo colormap da água e PowerNorm se disponível
            water_cmap = mcolors.LinearSegmentedColormap.from_list(
                "water_export",
                [
                    (0.70, 0.82, 1.00),
                    (0.18, 0.34, 0.85),
                    (0.05, 0.12, 0.45),
                    (0.00, 0.02, 0.18),
                ], N=256,
            )
            water_cmap.set_under(cast(Any, (0, 0, 0, 0.0)))
            vmin_w = float(
                max(1e-6, getattr(model, 'flood_threshold', 0.0) + 1e-6))
            _sim_data = np.ma.filled(masked_sim, fill_value=np.nan) if hasattr(masked_sim, 'filled') else masked_sim
            vmax_w = float(np.nanmax(_sim_data)) if np.isfinite(_sim_data).any() else vmin_w + 1e-3
            try:
                norm = mcolors.PowerNorm(gamma=float(st.session_state.get(
                    "water_gamma", 0.7)), vmin=vmin_w, vmax=max(vmin_w + 1e-6, vmax_w))
                ax_sim.imshow(masked_sim, extent=bounds_sim, cmap=water_cmap, norm=norm, alpha=float(
                    st.session_state.get("water_alpha", 0.5)))
            except Exception:
                ax_sim.imshow(masked_sim, extent=bounds_sim, cmap=water_cmap, vmin=vmin_w, vmax=max(
                    vmin_w + 1e-6, vmax_w), alpha=float(st.session_state.get("water_alpha", 0.5)))
            ax_sim.set_axis_off()
            buf_sim = io.BytesIO()
            fig_sim.savefig(buf_sim, format='png', dpi=200,
                            bbox_inches='tight', pad_inches=0)
            plt.close(fig_sim)
            buf_sim.seek(0)
            overlay_sim_png = buf_sim.getvalue()
            st.session_state["dl_overlay_sim_png"] = overlay_sim_png
            with st.container():
                st.download_button(
                    "Simulated inundation overlay (PNG)",
                    overlay_sim_png,
                    "overlay_dom_agua_simulada.png",
                    "image/png",
                    key="dl_overlay_sim_now",
                )
    except Exception as _e_sim_png:
        st.caption(f"(PNG de água simulada indisponível: {_e_sim_png})")

    # ========== Pacote ZIP do cenário (todos os artefatos disponíveis) ==========
    try:
        buf_zip = io.BytesIO()
        with zipfile.ZipFile(buf_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Parâmetros/JSON
            params: Dict[str, Any] = {}
            try:
                last = model.history[-1] if len(model.history) else {}
                params.update({
                    "total_rain_mm": float(total_rain),
                    "cell_size_m": float(cell_size),
                    "final_time_min": int(model.simulation_time_minutes),
                    "final_flooded_percent": float(last.get('flooded_percent', 0.0)),
                    "final_max_depth_m": float(last.get('max_depth', 0.0)),
                    "final_total_water_m3": float(last.get('total_water_volume_m3', 0.0)),
                })
            except Exception:
                pass
            params["created_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            zf.writestr('parametros.json', json.dumps(
                params, ensure_ascii=False, indent=2))

            # Dados & gráficos
            if st.session_state.get('dl_history_csv'):
                zf.writestr('dados_simulacao.csv',
                            st.session_state['dl_history_csv'])
            if st.session_state.get('dl_graph_png'):
                zf.writestr('evolucao_simulacao.png',
                            st.session_state['dl_graph_png'])

            # Overlays
            if st.session_state.get('dl_overlay_sim_png'):
                zf.writestr('overlay_dom_agua_simulada.png',
                            st.session_state['dl_overlay_sim_png'])
            if st.session_state.get('dl_overlay_png'):
                zf.writestr('overlay_dom_probabilidade.png',
                            st.session_state['dl_overlay_png'])

            # Animação
            if st.session_state.get('dl_anim_bytes') and st.session_state.get('dl_anim_ext'):
                ext = st.session_state['dl_anim_ext']
                zf.writestr(f'simulacao.{ext}',
                            st.session_state['dl_anim_bytes'])

            # IA (raster de probabilidade)
            if st.session_state.get('dl_prob_tif'):
                zf.writestr('prob_inundacao_ia.tif',
                            st.session_state['dl_prob_tif'])
            if st.session_state.get('dl_prob_rgba_tif'):
                zf.writestr('prob_inundacao_ia_rgba.tif',
                            st.session_state['dl_prob_rgba_tif'])

            # Pasta de estado (quando existe no workspace): inclui arquivos gerados em disco
            try:
                _base_dir = os.path.dirname(os.path.abspath(__file__))
                estado_dir = os.path.join(_base_dir, "Simulacao estado")
                if os.path.isdir(estado_dir):
                    for root, _dirs, files in os.walk(estado_dir):
                        for fn in files:
                            full_path = os.path.join(root, fn)
                            arcname = os.path.relpath(full_path, _base_dir)
                            try:
                                zf.write(full_path, arcname)
                            except Exception:
                                continue
            except Exception:
                pass

        buf_zip.seek(0)
        zip_bytes = buf_zip.getvalue()
        st.session_state["dl_zip_bytes"] = zip_bytes
        st.download_button(
            "Download complete output package (ZIP)",
            zip_bytes,
            "simulacao_pacote_completo.zip",
            "application/zip",
            use_container_width=True,
            key="dl_zip_now",
        )
    except Exception as _e_zip:
        st.caption(f"(Falha ao montar ZIP do pacote: {_e_zip})")

    # 3. Gráficos de evolução (reutiliza fig_evolution já gerada acima)
    if fig_evolution is not None:
        st.subheader("Temporal evolution of flood diagnostics")
        st.pyplot(fig_evolution, clear_figure=True)
    elif len(model.history) > 0:
        st.subheader("📈 Evolução Temporal")
        st.pyplot(_plot_temporal_diagnostics(model), clear_figure=True)

    # 4. Painel persistente de downloads recentes (sobrevive ao rerun)
    with st.expander("Exported files (persistent across reruns)"):
        cols = st.columns(3)
        with cols[0]:
            csv_buf = st.session_state.get("dl_history_csv")
            if csv_buf:
                st.download_button(
                    "📊 Dados (CSV)",
                    csv_buf,
                    "dados_simulacao.csv",
                    "text/csv",
                    key="dl_csv_persist",
                )
        with cols[1]:
            anim_buf = st.session_state.get("dl_anim_bytes")
            anim_ext = st.session_state.get("dl_anim_ext") or ""
            anim_mime = st.session_state.get(
                "dl_anim_mime") or "application/octet-stream"
            if anim_buf:
                st.download_button(
                    f"🎬 Animação ({str(anim_ext).upper()})",
                    anim_buf,
                    f"simulacao.{anim_ext}",
                    anim_mime,
                    key="dl_anim_persist",
                )
        with cols[2]:
            graph_buf = st.session_state.get("dl_graph_png")
            if graph_buf:
                st.download_button(
                    "📈 Gráficos (PNG)",
                    graph_buf,
                    "evolucao_simulacao.png",
                    "image/png",
                    key="dl_graph_persist",
                )
        # Linha adicional para overlay PNG
        cols2 = st.columns(3)
        with cols2[0]:
            overlay_buf = st.session_state.get("dl_overlay_png")
            if overlay_buf:
                st.download_button(
                    "Flood probability overlay (PNG)",
                    overlay_buf,
                    "overlay_dom_probabilidade.png",
                    "image/png",
                    key="dl_overlay_persist",
                )
        with cols2[1]:
            overlay_sim_buf = st.session_state.get("dl_overlay_sim_png")
            if overlay_sim_buf:
                st.download_button(
                    "🖼️ PNG (DOM + água simulada)",
                    overlay_sim_buf,
                    "overlay_dom_agua_simulada.png",
                    "image/png",
                    key="dl_overlay_sim_persist",
                )

        zip_buf = st.session_state.get("dl_zip_bytes")
        if zip_buf:
            st.download_button(
                "Complete output package (ZIP)",
                zip_buf,
                "simulacao_pacote_completo.zip",
                "application/zip",
                key="dl_zip_persist",
            )

# ── LISFLOOD integration (feature-flagged via ENABLE_LISFLOOD) ──────────


def main():
    st.set_page_config(
        page_title="HydroSim-RF — Hybrid Flood Inundation Simulator",
        page_icon=os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logos", "logo.png"),
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    apply_custom_styles()

    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_MAIN = os.path.join(_BASE_DIR, "logos", "logo.png")

    with st.container():
        col_l, col_c = st.columns([1, 5])
        with col_l:
            if os.path.exists(LOGO_MAIN):
                st.image(LOGO_MAIN, width=160)
        with col_c:
            st.markdown(
                "<h1 style='text-align:center; margin:0; font-size:1.7rem;'>"
                "HydroSim-RF: Hybrid Raster-Based Flood Inundation Simulator"
                "</h1>"
                "<p style='text-align:center; color:#555; margin:4px 0 0;'>"
                "Diffusion-wave solver · Random Forest flood probability · "
                "Spatial mitigation analysis"
                "</p>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    tab_numpy, tab_validation = st.tabs([
        "Simulation", "Model Validation (RF)"
    ])

    with tab_numpy:
        st.header("Flood Inundation Simulation")
        # Telemetria removida (não utilizada)

        # Container para uploads
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Input Data")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Required files**")
                dem_file = st.file_uploader(
                    "Digital Elevation Model (DEM)",
                    type=["tif", "tiff"],
                    help="GeoTIFF raster with ground-surface elevation [m a.s.l.]",
                    key="np_dem"
                )

            with col2:
                st.markdown("**Rainfall / source polygons (optional)**")
                vector_files = st.file_uploader(
                    "Rainfall source areas",
                    type=["gpkg", "shp", "shx", "dbf", "prj"],
                    accept_multiple_files=True,
                    help="Vector polygons defining spatial extent of rainfall application (GeoPackage or Shapefile)",
                    key="np_vec"
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # Container para parâmetros
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Model Parameters")

            col_params1, col_params2, col_params3 = st.columns(3)

            with col_params1:
                st.markdown("**Precipitation forcing**")
                rain_mm_per_cycle = st.number_input(
                    "Rainfall depth per time step (mm)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=5.0,
                    step=0.1,
                    help="Gross rainfall depth [mm] applied uniformly at each time step"
                )

                total_cycles = st.number_input(
                    "Number of time steps",
                    min_value=1,
                    max_value=2000,
                    value=100,
                    help="Total number of simulation time steps"
                )

                time_step_minutes = st.number_input(
                    "Time step duration (min)",
                    min_value=1,
                    max_value=1440,
                    value=10,
                    help="Duration of each computational time step [min]"
                )

            with col_params2:
                st.markdown("**Hydraulic parameters**")
                diffusion_rate = st.number_input(
                    "Diffusion coefficient α (–)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="Fraction of available water depth transferred to lower neighbours per time step (0 < α ≤ 1); lower values increase numerical diffusion"
                )

                flood_threshold = st.number_input(
                    "Inundation depth threshold h* (m)",
                    min_value=0.001,
                    max_value=2.0,
                    value=0.1,
                    step=0.01,
                    help="Minimum water depth [m] for a cell to be classified as inundated in reporting metrics"
                )

                rain_mode = st.selectbox(
                    "Rainfall application mode",
                    ["Spatially uniform", "Source polygons only"],
                    index=0,
                    help="Spatially uniform: rainfall applied to all cells | Source polygons: rainfall restricted to uploaded vector layer"
                )

            with col_params3:
                st.markdown("**Output / visualisation**")
                animation_format = st.selectbox(
                    "Formato da animação", ["GIF", "MP4"], index=0)

                animation_duration = st.slider(
                    "Animation duration (s)",
                    min_value=2,
                    max_value=60,
                    value=10,
                    help="Target duration of the exported animation [s]"
                )

                water_alpha = st.slider(
                    "Water layer opacity",
                    0.05, 0.9, 0.40, 0.05,
                    help="Alpha transparency of the inundation depth overlay (0 = transparent, 1 = opaque)"
                )
                water_min_threshold = st.slider(
                    "Visualisation depth threshold (m)",
                    0.0, 0.1, 0.005, 0.001,
                    help="Cells with water depth below this value are rendered transparent to suppress near-zero artefacts"
                )
                water_gamma = st.slider(
                    "Colour stretch exponent γ",
                    0.3, 2.0, 0.9, 0.05,
                    help="Power-norm exponent applied to the depth colormap (γ < 1 enhances shallow-water contrast; γ > 1 compresses the colour ramp)"
                )

                st.markdown("**Basemap**")
                basemap_choice = st.selectbox(
                    "Basemap provider",
                    [
                        "Esri.WorldImagery",
                        "CartoDB.Positron",
                        "Nenhum",
                        "Custom XYZ tile URL",
                    ],
                    index=0,
                    help="Web tile provider used as cartographic background (requires internet connection)"
                )
                basemap_xyz = ""
                if basemap_choice == "XYZ personalizado":
                    basemap_xyz = st.text_input(
                        "URL XYZ",
                        value="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                        help="Ex.: https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                    )
                show_dem_on_basemap = st.checkbox(
                    "Overlay DEM hillshade on basemap",
                    value=False,
                    help="Renders a semi-transparent DEM hillshade over the web basemap for terrain context"
                )

                show_rain_particles = st.checkbox(
                    "Show rainfall particle animation",
                    value=False,
                    help="Renders stochastic rainfall particles for presentation purposes; disable for publication-quality output"
                )

                show_water_contour = st.checkbox(
                    "Show inundation boundary contour",
                    value=False,
                    help="Draws the h = h* isoline to delineate the inundation extent boundary"
                )

                st.markdown("**Puddles / accumulations**")
                highlight_puddles = st.checkbox(
                    "Realçar poças/acúmulos (hotspots)",
                    value=True,
                    help="Adiciona uma camada quente sobre as maiores lâminas d'água para destacar áreas de acúmulo/empoçamento."
                )
                puddle_quantile = st.slider(
                    "Percentil para hotspot (%)",
                    min_value=70,
                    max_value=99,
                    value=90,
                    step=1,
                    help="O limiar de hotspot é definido como este percentil (entre as células molhadas) da lâmina d'água."
                )
                puddle_strength = st.slider(
                    "Intensidade do hotspot",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.65,
                    step=0.05,
                    help="Multiplicador de opacidade da camada de hotspots."
                )

                basemap_source = None
                if basemap_choice == "Nenhum":
                    basemap_source = "Nenhum"
                elif basemap_choice == "XYZ personalizado":
                    basemap_source = basemap_xyz
                else:
                    basemap_source = basemap_choice

            # Persistir parâmetros visuais para uso em exportações
            st.session_state["water_alpha"] = float(water_alpha)
            st.session_state["water_gamma"] = float(water_gamma)
            st.session_state["basemap_source"] = basemap_source
            st.session_state["show_dem_on_basemap"] = bool(show_dem_on_basemap)
            st.session_state["show_rain_particles"] = bool(show_rain_particles)
            st.session_state["show_water_contour"] = bool(show_water_contour)
            st.session_state["highlight_puddles"] = bool(highlight_puddles)
            st.session_state["puddle_quantile"] = float(puddle_quantile)
            st.session_state["puddle_strength"] = float(puddle_strength)

            # Parâmetros adicionais
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                grid_reduction_factor = st.select_slider(
                    "Grid resampling factor",
                    options=[1, 2, 4, 8, 16],
                    value=4,
                    help="Spatial resampling factor applied to the DEM prior to simulation (1 = native resolution; higher values increase speed at reduced accuracy)"
                )

                quick_preview = st.checkbox(
                    "Quick preview (skip animation export)",
                    value=False,
                    help="Runs the full simulation but skips animation rendering; use for rapid parameter sensitivity testing"
                )

            with col_adv2:
                dom_bg_file = st.file_uploader(
                    "Background orthoimage / DOM (optional)",
                    type=["tif", "tiff"],
                    help="GeoTIFF orthophoto or DOM used as cartographic background in place of a web basemap (no internet required)",
                    key="np_dom_bg"
                )

                river_vector_files = st.file_uploader(
                    "River / channel vector (optional)",
                    type=["gpkg", "shp", "shx", "dbf", "prj"],
                    accept_multiple_files=True,
                    help="Vector layer delineating the river or channel network; used to compute time-to-overflow and to guide levee placement in the mitigation analysis",
                    key="np_river"
                )

                apply_hs = st.checkbox(
                    "Apply hillshade to DOM background",
                    value=True,
                    help="Blends an analytical hillshade derived from the DEM over the orthoimage to emphasise terrain morphology"
                )

                hs_intensity = st.slider(
                    "Hillshade blending intensity",
                    0.0, 1.0, 0.6, 0.05,
                    help="Relative weight of the hillshade layer blended with the orthoimage [0 = no hillshade; 1 = full hillshade]"
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # Normalizações/casts para tipos estáveis (evita alertas do Pylance)
        total_cycles_int = int(total_cycles)
        time_step_minutes_int = int(time_step_minutes)
        # Persist grid factor for display in the summary table
        _gf = int(grid_reduction_factor[0]) if isinstance(grid_reduction_factor, tuple) else int(grid_reduction_factor)
        st.session_state["grid_factor"] = _gf

        # Métricas em tempo real
        st.subheader("Real-time Diagnostics")
        stats_cols = st.columns(4)
        overflow_ph, time_ph, flooded_ph, vol_ph = stats_cols
        overflow_ph.metric("Time to overflow", "N/A")
        time_ph.metric("Elapsed simulation time", "0h 0m")
        flooded_ph.metric("Inundated area", "0.00%")
        vol_ph.metric("Total water volume", "0.00 m³")

        # Botão de ação
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            simular = st.button(
                "Run Simulation",
                type="primary",
                use_container_width=True,
                key="simular_numpy"
            )
        with col_info:
            if not dem_file:
                st.warning("Upload a DEM raster to enable the simulation.")
            else:
                st.success("Ready — click Run Simulation to proceed.")

        # Área para a animação
        anim_area = st.empty()

        # ========== IA (na mesma aba) ==========
        with st.expander("Random Forest Inundation Probability (Beta)"):
            st.caption(
                "Train a Random Forest classifier using topographic indices (elevation, slope) derived from the DEM and binary flood labels from the current simulation.")
            col_ia1, col_ia2 = st.columns(2)
            with col_ia1:
                rf_show_map = st.checkbox(
                    "Display flood probability map (RF)", value=False)
                rf_threshold = st.slider(
                    "Probability display threshold", 0.0, 1.0, 0.5, 0.05)
                rf_alpha = st.slider(
                    "Probability overlay opacity", 0.05, 1.0, 0.6, 0.05)
            with col_ia2:
                rf_n_estimators = st.slider(
                    "Number of trees (RF ensemble)", 10, 300, 80, 10)
                rf_max_depth = st.slider("Maximum tree depth", 2, 30, 12, 1)
                rf_train = st.button(
                    "Train RF classifier", type="secondary")

            st.caption(
                "Features: percentile-normalised elevation and normalised slope magnitude. Labels: binary flood classification at depth threshold h*.")

        # Botão de treino IA (usa última simulação concluída)
        if 'rf_model' not in st.session_state:
            st.session_state['rf_model'] = None
        if rf_train:
            dem_last = st.session_state.get('last_dem_data')
            water_last = st.session_state.get('last_water_height')
            if dem_last is None or water_last is None:
                st.warning("Rode uma simulação primeiro para treinar a IA.")
            else:
                with st.spinner("Treinando modelo IA (RandomForest)..."):
                    try:
                        clf = _train_flood_classifier(
                            dem_last, water_last, threshold=water_min_threshold, n_estimators=rf_n_estimators, max_depth=rf_max_depth)
                        st.session_state['rf_model'] = clf
                        st.success("Modelo IA treinado com sucesso!")
                    except Exception as e:
                        st.error(f"Falha ao treinar IA: {e}")

        # Predição/overlay de probabilidade (em tempo real após treino)
        if rf_show_map and st.session_state.get('rf_model') is not None:
            try:
                dem_last = st.session_state.get('last_dem_data')
                transform_last = st.session_state.get('last_transform')
                crs_last = st.session_state.get('last_crs')
                if dem_last is None or transform_last is None or crs_last is None:
                    st.warning(
                        "Rode uma simulação primeiro para gerar o mapa de probabilidade.")
                else:
                    prob = _predict_probability(
                        st.session_state['rf_model'], dem_last)
                    # guardar para validação e download
                    st.session_state['rf_flood_prob'] = prob
                    _plot_probability_overlay(
                        prob, transform_last, crs_last, rf_threshold, rf_alpha, dem_back=dem_last)
                    # botão de download do raster de probabilidade
                    try:
                        gtiff_bytes = _probability_geotiff_bytes(
                            prob, transform_last, crs_last)
                        st.session_state["dl_prob_tif"] = gtiff_bytes
                        st.download_button(
                            label="Download flood probability (GeoTIFF)",
                            data=gtiff_bytes,
                            file_name="prob_inundacao_ia.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )
                        # Versão estilizada (RGBA) com transparência abaixo do limiar para melhor visualização em QGIS/ArcGIS
                        rgba_bytes = _probability_rgba_geotiff_bytes(
                            prob, transform_last, crs_last,
                            vmin=max(1e-6, rf_threshold), vmax=1.0,
                            cmap_name="Reds", under_transparent=True,
                        )
                        st.session_state["dl_prob_rgba_tif"] = rgba_bytes
                        st.download_button(
                            label="Download styled probability (GeoTIFF RGBA)",
                            data=rgba_bytes,
                            file_name="prob_inundacao_ia_rgba.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )
                        # PNG do overlay (DOM/DEM + probabilidade) para visualização direta
                        try:
                            bg = st.session_state.get("last_background_rgb")
                            bounds_png = array_bounds(
                                prob.shape[0], prob.shape[1], transform_last)
                            fig_png, ax_png = plt.subplots(figsize=(10, 8))
                            if bg is not None:
                                img = (bg * 255).astype(np.uint8) if np.issubdtype(bg.dtype,
                                                                                   np.floating) and bg.max() <= 1.0 else bg.astype(np.uint8)
                                ax_png.imshow(
                                    img, extent=bounds_png, alpha=1.0)
                            else:
                                dem_back = dem_last.astype(float)
                                vmin_b, vmax_b = np.nanpercentile(
                                    dem_back, (5, 95)) if np.isfinite(dem_back).any() else (0, 1)
                                ax_png.imshow(
                                    dem_back, extent=bounds_png, cmap="terrain", vmin=vmin_b, vmax=vmax_b, alpha=0.85)
                            reds = cm.get_cmap("Reds").copy()
                            reds.set_under(cast(Any, (0, 0, 0, 0.0)))
                            masked_png = np.ma.masked_less_equal(
                                prob, rf_threshold)
                            ax_png.imshow(masked_png, extent=bounds_png, cmap=reds, vmin=max(
                                1e-6, rf_threshold+1e-6), vmax=1.0, alpha=rf_alpha)
                            ax_png.set_axis_off()
                            buf_png = io.BytesIO()
                            fig_png.savefig(
                                buf_png, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
                            plt.close(fig_png)
                            buf_png.seek(0)
                            overlay_png = buf_png.getvalue()
                            st.session_state["dl_overlay_png"] = overlay_png
                            st.download_button(
                                label="Download probability overlay (PNG)",
                                data=overlay_png,
                                file_name="overlay_dom_probabilidade.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                        except Exception as _e_png:
                            st.caption(
                                f"(PNG de overlay indisponível: {_e_png})")
                    except Exception as e:
                        st.warning(
                            f"Falha ao gerar GeoTIFF de probabilidade: {e}")
            except Exception as e:
                st.error(f"Falha ao gerar probabilidade: {e}")

    # ── Model Validation tab (Random Forest) ────────────────────────────────
    with tab_validation:
        st.header("Random Forest Model Validation")
        st.caption(
            "Compares the Random Forest flood probability against the simulated inundation depth field. Run a simulation and train the RF classifier first.")
        dem_last = st.session_state.get('last_dem_data')
        water_last = st.session_state.get('last_water_height')
        prob_last = st.session_state.get('rf_flood_prob')
        if any(x is None for x in [dem_last, water_last, prob_last]):
            st.info(
                "⚠️ Rode uma simulação, treine a IA e gere o mapa de probabilidade para habilitar a validação.")
        else:
            colv1, colv2 = st.columns(2)
            with colv1:
                label_threshold = st.slider(
                    "Positive-class depth threshold h_label (m)", 0.0, 0.3, 0.01, 0.005,
                    help="Water depth threshold [m] used to assign positive inundation labels from the simulation output"
                )
            with colv2:
                st.caption("ROC and precision-recall curves computed over all valid grid cells.")

            water = np.asarray(water_last, dtype=float)
            y_true = (water.reshape(-1) >
                      float(label_threshold)).astype(np.uint8)
            y_score = np.asarray(prob_last, dtype=float).reshape(-1)
            valid = np.isfinite(y_true) & np.isfinite(y_score)
            if valid.sum() < 2:
                st.warning("Dados insuficientes para validar.")
            else:
                yt = y_true[valid]
                ys = y_score[valid]
                # ROC
                try:
                    fpr, tpr, _ = roc_curve(yt, ys)
                    auc_roc = roc_auc_score(yt, ys)
                except Exception as e:
                    fpr, tpr, auc_roc = np.array(
                        [0, 1]), np.array([0, 1]), float('nan')
                    st.warning(f"Falha ROC: {e}")
                # PR
                try:
                    prec, rec, _ = precision_recall_curve(yt, ys)
                    ap = average_precision_score(yt, ys)
                except Exception as e:
                    prec, rec, ap = np.array(
                        [1, 0]), np.array([0, 1]), float('nan')
                    st.warning(f"Falha PR: {e}")

                fig_val, axs = plt.subplots(1, 2, figsize=(12, 5))
                axs[0].plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}")
                axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
                axs[0].set_title("Curva ROC")
                axs[0].set_xlabel("Falso Positivo (FPR)")
                axs[0].set_ylabel("Verdadeiro Positivo (TPR)")
                axs[0].grid(True, alpha=0.3)
                axs[0].legend()
                axs[1].plot(rec, prec, label=f"AP = {ap:.3f}")
                axs[1].set_title("Curva Precisão-Revocação (PR)")
                axs[1].set_xlabel("Revocação")
                axs[1].set_ylabel("Precisão")
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()
                st.pyplot(fig_val, clear_figure=True)

        # ── Core simulation logic ───────────────────────────────────────────────
        if simular and dem_file:
            tmp = None
            tmp_anim = None  # Caminho do arquivo de animação (se gerado)
            try:
                sim_start_ts = time.time()
                vector_path = None
                river_path = None

                # Processar DEM e vetores de fonte
                if vector_files:
                    dem_path, vector_path, tmp = _save_input_files(
                        dem_file, vector_files)
                    if not vector_path:
                        st.error("Falha ao processar o arquivo vetorial.")
                        raise RuntimeError("Arquivo vetorial inválido")
                else:
                    # Salvar apenas DEM
                    tmp = tempfile.mkdtemp(prefix="sim_numpy_")
                    dem_path = os.path.join(tmp, dem_file.name)
                    with open(dem_path, "wb") as f:
                        f.write(dem_file.getbuffer())

                # Processar vetor de rio se enviado
                if river_vector_files:
                    for f in river_vector_files:
                        if f.name.lower().endswith((".gpkg", ".shp")):
                            rp = os.path.join(tmp or tempfile.mkdtemp(
                                prefix="sim_numpy_"), f"river_{f.name}")
                            with open(rp, "wb") as out:
                                out.write(f.getbuffer())
                            if f.name.lower().endswith(".gpkg"):
                                river_path = rp
                            elif f.name.lower().endswith(".shp") and river_path is None:
                                river_path = rp

                # Configurar dados geoespaciais
                assert dem_path is not None, "Caminho do DEM não pode ser None"
                gf = int(grid_reduction_factor[0]) if isinstance(
                    grid_reduction_factor, tuple) else int(grid_reduction_factor)
                dem_data, sources_mask, transform, crs, cell_size, river_mask = _prepare_spatial_domain(
                    dem_path, vector_path, gf, river_path
                )

                # Inicializar modelo
                model = DiffusionWaveFloodModel(
                    dem_data, sources_mask, diffusion_rate,
                    flood_threshold, cell_size, river_mask
                )
                model.uniform_rain = (rain_mode == "Uniforme na área")

                # Fallback: se usuário escolheu "Somente nas fontes", mas não há fontes nem rio, aplicar chuva uniforme para evitar resultado em branco
                if (rain_mode != "Uniforme na área") and (not np.any(sources_mask)) and (not np.any(river_mask)):
                    st.info(
                        "Nenhuma fonte vetorial ou rio definidos. Aplicando chuva uniforme na área para evitar resultado vazio.")
                    model.uniform_rain = True

                # Preparar fundo visual
                background_rgb = None
                if dom_bg_file is not None and tmp:
                    dom_tmp = os.path.join(tmp, f"bg_{dom_bg_file.name}")
                    with open(dom_tmp, "wb") as out:
                        out.write(dom_bg_file.getbuffer())
                    background_rgb = _load_orthoimage(
                        dom_tmp, dem_data.shape, crs)
                # Guardar o fundo atual (DOM reamostrado) para futuras exportações
                st.session_state["last_background_rgb"] = background_rgb

                # Configurar visualização
                fig, _, water_layer, puddle_layer, rain_particles, title, bounds = _init_animation_figure(
                    dem_data,
                    transform,
                    crs,
                    background_rgb,
                    apply_hs,
                    hs_intensity,
                    basemap_source=st.session_state.get(
                        "basemap_source", 'CartoDB.Positron'),
                    show_dem_on_basemap=bool(
                        st.session_state.get("show_dem_on_basemap", False)),
                )
                x_min, y_min, x_max, y_max = bounds

                # Grelha nas coordenadas geográficas para alinhar contorno com o imshow (extent)
                xs = np.linspace(x_min, x_max, dem_data.shape[1])
                # inverter eixo Y (origin='upper')
                ys = np.linspace(y_max, y_min, dem_data.shape[0])
                Xw, Yw = np.meshgrid(xs, ys)

                # Coleção para contornos de água (atualizados a cada frame)
                water_contour_artists = []
                puddle_contour_artists = []

                progress = st.progress(0, text="Inicializando...")

                # Função de atualização da animação
                def update(frame):
                    # Adicionar chuva
                    model.apply_rainfall(rain_mm_per_cycle)

                    # Executar passo de fluxo
                    model.advance_flow()

                    # Atualizar estatísticas
                    model.record_diagnostics(time_step_minutes_int)

                    # Atualizar visualização da água: somente onde acumula
                    water = model.water_height
                    masked = np.ma.masked_less_equal(
                        water, water_min_threshold)
                    # definir vmin ligeiramente acima do threshold para que valores abaixo fiquem 'under' (transparentes)
                    current_vmin = max(1e-9, float(water_min_threshold) + 1e-9)
                    masked = np.ma.masked_less_equal(
                        water, water_min_threshold)
                    # vmax dinâmico (percentil 99) para evitar “lavar” a escala por picos pontuais
                    try:
                        vals = np.asarray(water, dtype=float)
                        vals = vals[np.isfinite(vals) & (
                            vals > float(water_min_threshold))]
                        if vals.size > 0:
                            wmax = float(np.nanpercentile(vals, 99))
                        else:
                            wmax = current_vmin + 1e-3
                    except Exception:
                        wmax = current_vmin + 1e-3
                    vmax_eff = max(current_vmin + 1e-6, wmax)
                    water_layer.set_clim(vmin=current_vmin, vmax=vmax_eff)
                    # Aplicar PowerNorm para aumentar contraste visual
                    try:
                        norm = mcolors.PowerNorm(gamma=float(
                            water_gamma), vmin=current_vmin, vmax=vmax_eff)
                        water_layer.set_norm(norm)
                    except Exception:
                        pass
                    water_layer.set_data(masked)

                    # Alpha por-pixel com “piso” para água rasa ficar mais visível
                    try:
                        denom = max(
                            1e-6, float(vmax_eff - water_min_threshold))
                        disp = np.clip(
                            (water - float(water_min_threshold)) / denom, 0.0, 1.0)
                        # Realce moderado (evita escurecer demais em mapa base com muito azul)
                        disp = disp**0.55
                        alpha_floor = 0.18
                        alpha_map = float(
                            water_alpha) * (alpha_floor + (1.0 - alpha_floor) * disp)
                        alpha_map = np.where(water > float(
                            water_min_threshold), alpha_map, 0.0)
                        water_layer.set_alpha(alpha_map)
                    except Exception:
                        water_layer.set_alpha(water_alpha)

                    # Realce de poças/acúmulos (hotspots): camada quente acima de um percentil
                    highlight_puddles = bool(st.session_state.get("highlight_puddles", True))
                    puddle_q = float(st.session_state.get("puddle_quantile", 90.0))
                    puddle_strength = float(st.session_state.get("puddle_strength", 0.65))

                    # Limpar contornos anteriores de hotspots
                    try:
                        for artist in puddle_contour_artists:
                            artist.remove()
                    except Exception:
                        pass
                    puddle_contour_artists.clear()

                    try:
                        vals_hi = np.asarray(water, dtype=float)
                        vals_hi = vals_hi[np.isfinite(vals_hi) & (vals_hi > float(water_min_threshold))]
                    except Exception:
                        vals_hi = np.asarray([], dtype=float)

                    if highlight_puddles and vals_hi.size > 20 and (float(np.nanmax(water)) > float(water_min_threshold)):
                        thr = float(np.nanpercentile(vals_hi, np.clip(puddle_q, 70.0, 99.0)))
                        thr = max(thr, float(current_vmin))

                        # Máscara e normalização apenas para valores acima do limiar de hotspot
                        puddle_masked = np.ma.masked_less_equal(water, thr)
                        puddle_layer.set_data(puddle_masked)
                        vmin_h = float(thr + 1e-9)
                        puddle_layer.set_clim(vmin=vmin_h, vmax=float(vmax_eff))
                        try:
                            puddle_norm = mcolors.PowerNorm(gamma=0.55, vmin=vmin_h, vmax=float(vmax_eff))
                            puddle_layer.set_norm(puddle_norm)
                        except Exception:
                            pass

                        try:
                            denom_h = max(1e-6, float(vmax_eff - thr))
                            nd = np.clip((water - thr) / denom_h, 0.0, 1.0)
                            alpha_h = float(puddle_strength) * (0.20 + 0.80 * (nd ** 0.35))
                            alpha_h = np.where(water > thr, alpha_h, 0.0)
                            puddle_layer.set_alpha(alpha_h)
                        except Exception:
                            puddle_layer.set_alpha(min(0.85, max(0.0, float(puddle_strength))))

                        # Contornos dos hotspots (melhora leitura de poças/acúmulos)
                        try:
                            ax_plot = puddle_layer.axes
                            _before = set(id(a) for a in ax_plot.collections)
                            thr2 = float(np.nanpercentile(vals_hi, 97.0)) if vals_hi.size > 50 else (thr + (vmax_eff - thr) * 0.60)
                            thr2 = max(thr2, thr + 1e-6)
                            cs2 = ax_plot.contour(
                                Xw, Yw, water,
                                levels=[thr, thr2],
                                colors=[(1.00, 0.96, 0.55, 0.98), (1.00, 0.62, 0.15, 0.98)],
                                linewidths=[1.6, 2.4],
                                zorder=12.5,
                            )
                            new_artists = getattr(cs2, 'collections', [])
                            if not new_artists:
                                new_artists = [a for a in ax_plot.collections if id(a) not in _before]
                            puddle_contour_artists.extend(new_artists)
                        except Exception:
                            pass
                    else:
                        # Desativar camada de hotspots
                        puddle_layer.set_alpha(0.0)
                        puddle_layer.set_data(np.ma.masked_all_like(water))

                    # Atualizar contorno das áreas com água (destacar limites)
                    # Remover contornos anteriores
                    try:
                        for artist in water_contour_artists:
                            artist.remove()
                    except Exception:
                        pass
                    water_contour_artists.clear()

                    # Desenhar contorno (opcional)
                    if bool(st.session_state.get("show_water_contour", False)) and (np.nanmax(water) > water_min_threshold):
                        ax_plot = water_layer.axes
                        # Capturar artistas ANTES do contour para identificar os novos
                        _before = set(id(a) for a in ax_plot.collections)
                        cs = ax_plot.contour(
                            Xw, Yw, water,
                            levels=[water_min_threshold],
                            colors=[(1.0, 1.0, 1.0, 0.90)],  # branco discreto
                            linewidths=2.2,
                            zorder=11,
                        )
                        # Compatível com Matplotlib >= 3.8 (cs.collections pode ser vazio)
                        new_artists = getattr(cs, 'collections', [])
                        if not new_artists:
                            # Fallback: identificar artistas adicionados pelo contour
                            new_artists = [a for a in ax_plot.collections if id(a) not in _before]
                        water_contour_artists.extend(new_artists)

                    # Partículas de chuva (efeito visual) — opcional
                    if bool(st.session_state.get("show_rain_particles", False)):
                        n = int(rain_mm_per_cycle * 150)
                        rx = np.random.uniform(x_min, x_max, n)
                        ry = np.random.uniform(y_min, y_max, n)
                        rain_particles.set_visible(True)
                        rain_particles.set_alpha(0.7)
                        rain_particles.set_data(rx, ry)
                    else:
                        rain_particles.set_visible(False)
                        rain_particles.set_alpha(0.0)
                        rain_particles.set_data([], [])

                    # Atualizar título e métricas
                    h, m = divmod(model.simulation_time_minutes, 60)
                    title.set_text(
                        f"Simulação de Inundação | Tempo: {h}h {m}m")

                    # Atualizar métricas em tempo real
                    latest = model.history[-1]
                    if model.overflow_time_minutes is not None:
                        ho, mo = divmod(model.overflow_time_minutes, 60)
                        overflow_ph.metric(
                            "Tempo para Transbordar", f"{ho}h {mo}m")
                    time_ph.metric("Tempo de Simulação", f"{h}h {m}m")
                    flooded_ph.metric(
                        "Área Inundada", f"{latest['flooded_percent']:.2f}%")
                    vol_ph.metric("Volume de Água",
                                  f"{latest['total_water_volume_m3']:.2f} m³")

                    # Atualizar barra de progresso
                    progress.progress(
                        int(100 * (frame + 1) / max(1, total_cycles_int)),
                        text=f"Simulando ciclo {frame + 1}/{total_cycles_int}"
                    )

                    return [water_layer, puddle_layer, rain_particles, title]

                # Executar simulação
                if quick_preview:
                    # Modo pré-visualização: executar sem salvar animação
                    for frame in range(total_cycles_int):
                        update(frame)
                    # Mostrar resultado final
                    anim_area.pyplot(fig, clear_figure=False)
                else:
                    # Modo completo: gerar animação
                    # FPS calculado como float para evitar truncamento (ex.: 10 ciclos / 10s = 1 fps)
                    animation_duration_safe = max(1, int(animation_duration))
                    fps_float = max(1.0, float(total_cycles_int) / float(animation_duration_safe))
                    fps = max(1, round(fps_float))
                    interval = max(1, round(1000.0 / fps_float))  # ms entre frames (float evita jitter)

                    # Garantir fundo branco para evitar fundo preto/transparente no GIF
                    try:
                        fig.set_facecolor('white')
                    except Exception:
                        # Se por algum motivo o backend não suportar, ignoramos silenciosamente
                        pass
                    ext = str(animation_format).lower()  # valor padrão (pode ser sobrescrito por fallback)

                    # blit=False evita “fantasmas” de artists (ex.: partículas) no MP4/GIF
                    anim = FuncAnimation(
                        fig, update, frames=total_cycles_int,
                        blit=False, interval=interval
                    )

                    # Salvar animação
                    ext = str(animation_format).lower()
                    tmp_anim = os.path.join(
                        tmp or tempfile.gettempdir(), f"simulation.{ext}")
                    saved_ok = False
                    try:
                        if ext == 'gif':
                            # GIF via Pillow: dpi menor (90) para tamanho razoável
                            anim.save(tmp_anim, writer='pillow', dpi=90, fps=fps)
                            saved_ok = True
                        else:
                            # MP4: garantir ffmpeg disponível via imageio-ffmpeg
                            ffmpeg_bin = None
                            try:
                                import imageio_ffmpeg  # type: ignore
                                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                            except (ImportError, OSError):
                                subprocess.run([sys.executable, '-m', 'pip', 'install',
                                               'imageio-ffmpeg', '--quiet'], capture_output=True, check=False)
                                try:
                                    import importlib as _im
                                    imageio_ffmpeg = _im.import_module(
                                        'imageio_ffmpeg')  # type: ignore
                                except ModuleNotFoundError:
                                    ffmpeg_bin = None
                                else:
                                    try:
                                        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                                    except OSError:
                                        ffmpeg_bin = None

                            if ffmpeg_bin:
                                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_bin
                                from matplotlib.animation import FFMpegWriter
                                writer = FFMpegWriter(
                                    fps=fps,
                                    codec='libx264',
                                    bitrate=1800,
                                    extra_args=['-pix_fmt', 'yuv420p']
                                )
                                anim.save(tmp_anim, writer=writer, dpi=150)
                                saved_ok = True
                            else:
                                anim.save(
                                    tmp_anim,
                                    writer='ffmpeg',
                                    dpi=150,
                                    fps=fps,
                                    extra_args=['-vcodec', 'libx264',
                                                '-pix_fmt', 'yuv420p']
                                )
                                saved_ok = True
                    except (RuntimeError, ValueError, OSError) as e:
                        st.warning(
                            f"Falha ao salvar em {ext.upper()} ({e}). Tentando GIF...")

                    # Fallback para GIF executado UMA vez somente se o save original falhou
                    if not saved_ok:
                        ext = 'gif'
                        tmp_anim = os.path.join(
                            tmp or tempfile.gettempdir(), "simulation.gif")
                        try:
                            anim.save(tmp_anim, writer='pillow', dpi=90, fps=fps)
                        except Exception as e_gif:
                            st.error(f"Não foi possível salvar a animação: {e_gif}")

                    # Exibir animação
                    with open(tmp_anim, "rb") as f:
                        if ext == 'gif':
                            anim_area.image(f.read())
                        else:
                            anim_area.video(f.read())

                # Salvar dados para IA (estado da sessão)
                st.session_state["last_dem_data"] = dem_data
                st.session_state["last_transform"] = transform
                st.session_state["last_crs"] = crs
                st.session_state["last_water_height"] = model.water_height.copy()
                st.session_state["last_river_mask"] = river_mask
                st.session_state["last_cell_size"] = float(cell_size)
                st.session_state["last_flood_threshold"] = float(
                    flood_threshold)

                # Pós-processamento e downloads
                total_rain_mm = float(rain_mm_per_cycle) * \
                    float(total_cycles_int)
                # Passar ext (não animation_format) pois ext reflete o formato real após fallback
                _render_simulation_outputs(
                    model, tmp,
                    tmp_anim,
                    ext if not quick_preview else str(animation_format).lower(),
                    total_rain_mm,
                    cell_size,
                    sources_mask
                )
                sim_dur = time.time() - sim_start_ts

            except (RuntimeError, ValueError, OSError) as e:
                st.error(f"Erro na simulação: {e}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                # Limpeza
                if tmp and os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                plt.close('all')

        # ── Mitigation analysis panel ───────────────────────────────────────────
        with st.expander("Spatial Flood Mitigation Analysis (Beta)"):
            st.caption(
                "Rule-based spatial analysis identifying candidate zones for nature-based solutions and engineering interventions.")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                mit_threshold = st.slider(
                    "Flood probability threshold for intervention eligibility",
                    0.1, 1.0, 0.45, 0.05,
                    help="Minimum RF probability above which a cell is classified as high flood risk and eligible for interventions"
                )
                mit_min_slope = st.slider(
                    "Normalised slope threshold for reforestation eligibility",
                    0.001, 0.2, 0.05, 0.005,
                    help="Cells with normalised slope below this value are eligible for reforestation / green infrastructure"
                )
            with col_m2:
                _has_mit_data = bool(st.session_state.get('mitigation_data'))
                show_mitigation = st.checkbox(
                    "Display intervention map", value=_has_mit_data,
                    help="Ative antes de clicar em 'Analisar Terreno' para ver o mapa imediatamente")
                generate_report = st.button(
                    "Generate Mitigation Report", type="secondary",
                    disabled=not bool(st.session_state.get('mitigation_data')),
                    help="Execute 'Analisar Terreno para Mitigação' primeiro")
                use_icons = st.checkbox("Overlay intervention icons", value=False,
                                        help="Overlays schematic icons (reforestation, levee, drainage, terrain-raising) at cluster centroids")
                icon_size = st.slider("Icon size (px)", 12, 64, 24, 2)
                icon_dir = st.text_input("Icon directory (optional)", value="",
                                         help="Directory containing PNG icons (tree.png, dike.png, drainage.png, fill.png). If empty, defaults to ./logos/icons, ./icons, ./logos")

            if st.button("Run Mitigation Analysis", type="primary"):
                dem_last = st.session_state.get('last_dem_data')
                prob_last = st.session_state.get('rf_flood_prob')
                river_last = st.session_state.get('last_river_mask')
                transform_last = st.session_state.get('last_transform')
                crs_last = st.session_state.get('last_crs')
                bg_last = st.session_state.get('last_background_rgb')
                cell_size = float(st.session_state.get(
                    'last_cell_size') or 10.0)

                if dem_last is None:
                    st.warning("Run a flood simulation first to generate the required DEM and probability fields.")
                else:
                    # Se a probabilidade IA ainda não foi gerada, mas há modelo treinado, calcule agora
                    if prob_last is None and st.session_state.get('rf_model') is not None:
                        try:
                            prob_last = _predict_probability(
                                st.session_state['rf_model'], dem_last)
                            st.session_state['rf_flood_prob'] = prob_last
                        except Exception as _e_pred:
                            st.warning(
                                f"Falha ao gerar probabilidade automaticamente: {_e_pred}")

                    if prob_last is None:
                        # Fallback: estimar probabilidade a partir da lâmina d'água simulada
                        water_last = st.session_state.get('last_water_height')
                        ft = float(st.session_state.get(
                            'last_flood_threshold') or 0.1)
                        if water_last is not None:
                            w = np.asarray(water_last, dtype=float)
                            if np.isfinite(w).any():
                                wmax = float(np.nanmax(w))
                                if wmax > 0:
                                    eps = 1e-6
                                    prob_last = np.clip(
                                        (w - ft) / max(eps, (wmax - ft)), 0.0, 1.0)
                                    st.info(
                                        "Using approximate flood probability derived from the simulated water depth field (no RF classifier trained).")
                                    st.session_state['rf_flood_prob'] = prob_last
                        if prob_last is None:
                            st.warning(
                                "Generate the flood probability map in the RF section first, or increase rainfall / number of time steps to produce a non-trivial water depth field.")
                    if prob_last is not None:
                        with st.spinner("Analysing terrain and generating intervention suggestions..."):
                            try:
                                intervention_mask, suggestions = _identify_intervention_zones(
                                    np.asarray(dem_last), np.asarray(
                                        prob_last),
                                    river_last, mit_threshold, mit_min_slope, cell_size
                                )
                                st.session_state['mitigation_data'] = {
                                    'intervention_mask': intervention_mask,
                                    'suggestions': suggestions,
                                    'cell_size': cell_size,
                                    'transform': transform_last,
                                    'crs': crs_last,
                                    'background': bg_last,
                                }
                                st.success("Mitigation analysis complete. Enable 'Display intervention map' to visualise the results.")
                            except Exception as e:
                                st.error(f"Erro na análise: {e}")

            mitigation_data = st.session_state.get('mitigation_data')
            # Exibir mapa automaticamente se já existe dados e o checkbox está ativo
            if show_mitigation and mitigation_data:
                dem_last = st.session_state.get('last_dem_data')
                transform_last = mitigation_data.get('transform')
                crs_last = mitigation_data.get('crs')
                bg_last = mitigation_data.get('background')
                if dem_last is not None and transform_last is not None:
                    fig_mit = _plot_intervention_map(
                        np.asarray(dem_last),
                        mitigation_data['intervention_mask'],
                        mitigation_data['suggestions'],
                        transform_last, crs_last, bg_last,
                        use_icons=bool(use_icons),
                        icon_dir=(icon_dir or None),
                        icon_size=int(icon_size)
                    )
                    st.pyplot(fig_mit, clear_figure=True)
                    if not np.any(mitigation_data['intervention_mask']):
                        st.info(
                            "No intervention zones were identified under the current parameter configuration. Consider lowering the flood probability threshold or relaxing the slope criterion.")
                    buf = io.BytesIO()
                    fig_mit.savefig(buf, format='png', dpi=300,
                                    bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        "Download intervention map (PNG)",
                        buf.getvalue(),
                        "mapa_intervencoes_mitigacao.png",
                        "image/png",
                        use_container_width=True,
                    )

            if generate_report and mitigation_data:
                report_text = _build_mitigation_report(
                    mitigation_data['suggestions'], float(
                        mitigation_data.get('cell_size') or 10.0)
                )
                st.markdown("---")
                st.markdown("### Flood Mitigation Assessment Report")
                st.markdown(report_text)
                st.download_button(
                    "Download report (TXT)",
                    report_text.encode('utf-8'),
                    "relatorio_mitigacao.txt",
                    "text/plain",
                    use_container_width=True,
                )
                import json as _json
                suggestions_json = _json.dumps(
                    mitigation_data['suggestions'], indent=2, ensure_ascii=False)
                st.download_button(
                    "Download intervention data (JSON)",
                    suggestions_json.encode('utf-8'),
                    "dados_intervencoes.json",
                    "application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()


# NOTA: Sugestões prontas (todas totalizam 174 mm em 24 h):

# Ciclo de 5 min: 288 ciclos; 0,604 mm por ciclo
# Ciclo de 10 min: 144 ciclos; 1,208 mm por ciclo
# Ciclo de 15 min: 96 ciclos; 1,8125 mm por ciclo
# Ciclo de 30 min: 48 ciclos; 3,625 mm por ciclo
# Ciclo de 60 min: 24 ciclos; 7,25 mm por ciclo
