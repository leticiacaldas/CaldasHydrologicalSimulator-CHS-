"""Export functionality for HydroSim-RF simulation outputs."""

import numpy as np
import rasterio as rio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import io as io_lib
from typing import Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def export_geotiff_probability(prob: np.ndarray, transform: Any, crs: Any) -> bytes:
    """
    Export flood probability raster as GeoTIFF bytes.

    Parameters
    ----------
    prob : np.ndarray, shape (H, W)
        Flood probability grid [0, 1].
    transform : rasterio.transform.Affine
        Georeferencing transform.
    crs : CRS
        Coordinate reference system.

    Returns
    -------
    tif_bytes : bytes
        GeoTIFF data as bytes.
    """
    H, W = prob.shape
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=H, width=W,
            count=1,
            dtype=np.float32,
            transform=transform,
            crs=crs,
        ) as mem:
            mem.write(prob.astype(np.float32), 1)
        tif_bytes = memfile.read()
    
    logger.info(f"Exported probability GeoTIFF: {len(tif_bytes)} bytes")
    return tif_bytes


def export_png_overlay(prob: np.ndarray, dem: Optional[np.ndarray], 
                      transform: Any, crs: Any, threshold: float = 0.5,
                      alpha: float = 0.6) -> bytes:
    """
    Export flood probability overlay as PNG.

    Parameters
    ----------
    prob : np.ndarray, shape (H, W)
        Flood probability grid.
    dem : np.ndarray, optional
        Background DEM for hillshade.
    transform : rasterio.transform.Affine
        Georeferencing transform.
    crs : CRS
        Coordinate reference system.
    threshold : float
        Probability threshold for display.
    alpha : float
        Overlay transparency.

    Returns
    -------
    png_bytes : bytes
        PNG data as bytes.
    """
    from rasterio.transform import array_bounds
    
    bounds = array_bounds(prob.shape[0], prob.shape[1], transform)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dem is not None:
        dem_norm = dem.astype(float)
        vmin, vmax = np.percentile(dem_norm[np.isfinite(dem_norm)], (5, 95))
        ax.imshow(dem_norm, extent=bounds, cmap='terrain', vmin=vmin, vmax=vmax, alpha=0.85)
    
    reds = cm.get_cmap('Reds').copy()
    reds.set_under((0, 0, 0, 0.0))
    masked_prob = np.ma.masked_less_equal(prob, threshold)
    ax.imshow(masked_prob, extent=bounds, cmap=reds, 
             vmin=max(1e-6, threshold + 1e-6), vmax=1.0, alpha=alpha)
    ax.set_axis_off()
    
    buf = io_lib.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.getvalue()
    
    logger.info(f"Exported PNG overlay: {len(png_bytes)} bytes")
    return png_bytes
