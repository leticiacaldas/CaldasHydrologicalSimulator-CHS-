"""Raster and vector input handling for HydroSim-RF."""

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.features import rasterize
from typing import Tuple, Optional, Any
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


def load_raster(path: str, target_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Any, Any]:
    """
    Load a GeoTIFF raster file.

    Parameters
    ----------
    path : str
        Path to GeoTIFF file.
    target_shape : tuple, optional
        Target output shape (H, W). If provided, raster will be resampled.

    Returns
    -------
    data : np.ndarray
        Raster data.
    transform : rasterio.transform.Affine
        Georeferencing transform.
    crs : CRS
        Coordinate reference system.
    """
    with rio.open(path) as src:
        if target_shape is None:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
        else:
            H, W = target_shape
            data = src.read(1, out_shape=(H, W), resampling=Resampling.bilinear)
            # Recompute transform for resampled grid
            bounds = src.bounds
            transform = from_origin(bounds.left, bounds.top, 
                                   (bounds.right - bounds.left) / W,
                                   (bounds.top - bounds.bottom) / H)
            crs = src.crs
    
    logger.info(f"Loaded raster {path}: shape={data.shape}, dtype={data.dtype}")
    return data, transform, crs


def save_raster(path: str, data: np.ndarray, transform: Any, crs: Any) -> None:
    """
    Save array as GeoTIFF raster.

    Parameters
    ----------
    path : str
        Output file path.
    data : np.ndarray
        Raster data.
    transform : rasterio.transform.Affine
        Georeferencing transform.
    crs : CRS
        Coordinate reference system.
    """
    H, W = data.shape
    with rio.open(
        path, 'w',
        driver='GTiff',
        height=H, width=W,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(data, 1)
    logger.info(f"Saved raster {path}")


def load_vector_mask(path: str, raster_shape: Tuple[int, int], 
                    raster_transform: Any, raster_crs: Any) -> np.ndarray:
    """
    Rasterize a vector layer to create a binary mask.

    Parameters
    ----------
    path : str
        Path to vector file (GeoPackage or Shapefile).
    raster_shape : tuple
        Target raster shape (H, W).
    raster_transform : rasterio.transform.Affine
        Georeferencing transform of target raster.
    raster_crs : CRS
        Coordinate reference system of target raster.

    Returns
    -------
    mask : np.ndarray, dtype bool
        Binary rasterized mask.
    """
    try:
        gdf = gpd.read_file(path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        
        H, W = raster_shape
        shapes = [(geom, 1) for geom in gdf.geometry]
        mask = rasterize(shapes, out_shape=(H, W), transform=raster_transform,
                        default_value=0, dtype=np.uint8).astype(bool)  # type: ignore
        logger.info(f"Rasterized {path}: {np.sum(mask)} cells in mask")
        return mask
    except Exception as e:
        logger.error(f"Failed to rasterize vector {path}: {e}")
        raise
