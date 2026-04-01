"""Utilities for input validation, caching, and enhanced error handling."""

import os
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import numpy as np
import rasterio as rio

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path("outputs/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DURATION_MINUTES = 30


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class CacheManager:
    """Manage file caching with TTL."""
    
    @staticmethod
    def get_cache_key(file_path: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from file path and parameters."""
        content = f"{file_path}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def is_cache_valid(cache_file: Path, max_age_minutes: int = CACHE_DURATION_MINUTES) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return age < timedelta(minutes=max_age_minutes)
    
    @staticmethod
    def save_to_cache(data: np.ndarray, cache_key: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Save array to cache."""
        cache_file = CACHE_DIR / f"{cache_key}.npz"
        try:
            # Save only data as npz, store metadata separately if needed
            np.savez_compressed(cache_file, data=data)
            logger.info(f"✅ Cached: {cache_file}")
            return cache_file
        except Exception as e:
            logger.warning(f"⚠️ Could not cache: {e}")
            return None
    
    @staticmethod
    def load_from_cache(cache_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load array from cache if valid."""
        cache_file = CACHE_DIR / f"{cache_key}.npz"
        if CacheManager.is_cache_valid(cache_file):
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                data = loaded['data']
                metadata = dict(loaded['metadata'].item()) if 'metadata' in loaded else {}
                logger.info(f"✅ Cache hit: {cache_key}")
                return data, metadata
            except Exception as e:
                logger.warning(f"⚠️ Could not load cache: {e}")
        return None


def validate_geotiff(file_path: str, max_size_mb: float = 500.0) -> Tuple[bool, str]:
    """
    Validate GeoTIFF file format and constraints.
    
    Returns: (is_valid, message)
    """
    try:
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.1f} MB (max: {max_size_mb} MB)"
        
        # Check extension
        if path.suffix.lower() not in ['.tif', '.tiff', '.geotiff']:
            return False, f"Invalid extension: {path.suffix}. Expected .tif/.tiff/.geotiff"
        
        # Try to open with rasterio
        with rio.open(file_path) as src:
            # Check CRS
            if src.crs is None:
                logger.warning("⚠️ Warning: GeoTIFF has no CRS defined, assuming WGS84")
            
            # Check data type
            if src.dtypes[0] not in ['float32', 'float64', 'int16', 'int32', 'uint8', 'uint16']:
                return False, f"Unsupported data type: {src.dtypes[0]}"
            
            # Check shape
            if src.height < 10 or src.width < 10:
                return False, f"DEM too small: {src.height}x{src.width} (min: 10x10)"
            
            if src.height > 10000 or src.width > 10000:
                return False, f"DEM too large: {src.height}x{src.width} (max: 10000x10000)"
        
        return True, "✅ Valid GeoTIFF"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_shapefile(file_path: str) -> Tuple[bool, str]:
    """Validate shapefile/GeoPackage format."""
    try:
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        if path.suffix.lower() not in ['.gpkg', '.shp', '.geojson']:
            return False, f"Invalid format: {path.suffix}. Expected .gpkg/.shp/.geojson"
        
        # Try to open with geopandas
        try:
            import geopandas as gpd
            gdf = gpd.read_file(file_path)
            if len(gdf) == 0:
                return False, "Shapefile is empty"
            return True, f"✅ Valid shapefile ({len(gdf)} features)"
        except ImportError:
            return True, "✅ File format OK (geopandas not available for full validation)"
        except Exception as e:
            return False, f"Shapefile error: {str(e)}"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_dem_values(dem: np.ndarray, cell_size_m: float = 25.0) -> Tuple[bool, Dict[str, Any]]:
    """Validate DEM data integrity and statistics."""
    stats = {
        "min": float(np.nanmin(dem)),
        "max": float(np.nanmax(dem)),
        "mean": float(np.nanmean(dem)),
        "std": float(np.nanstd(dem)),
        "nodata_count": int(np.isnan(dem).sum()),
        "negative_count": int((dem < 0).sum()),
    }
    
    issues = []
    
    if stats["nodata_count"] > dem.size * 0.5:
        issues.append(f"More than 50% NoData: {stats['nodata_count']} cells")
    
    if stats["negative_count"] > dem.size * 0.1:
        issues.append(f"More than 10% negative elevations: {stats['negative_count']} cells")
    
    if stats["std"] < 0.1:
        issues.append("DEM is too flat (std < 0.1 m)")
    
    if stats["max"] - stats["min"] < 1.0:
        issues.append("Elevation range too small (< 1 m)")
    
    is_valid = len(issues) == 0
    if not is_valid:
        logger.warning(f"⚠️ DEM validation issues: {'; '.join(issues)}")
    
    return is_valid, stats


class EnhancedLogging:
    """Enhanced logging with progress tracking."""
    
    @staticmethod
    def log_step(step_name: str, current: int, total: int, status: str = ""):
        """Log a simulation step with progress."""
        pct = int((current / total) * 100) if total > 0 else 0
        msg = f"[{pct:3d}%] {step_name}"
        if status:
            msg += f" - {status}"
        logger.info(msg)
    
    @staticmethod
    def log_performance(operation: str, elapsed_seconds: float, result_description: str = ""):
        """Log performance metrics."""
        msg = f"✅ {operation}: {elapsed_seconds:.2f}s"
        if result_description:
            msg += f" ({result_description})"
        logger.info(msg)


def ensure_valid_crs(crs) -> str:
    """Ensure CRS is valid, default to WGS84 if None."""
    if crs is None:
        logger.warning("⚠️ No CRS provided, defaulting to EPSG:4326 (WGS84)")
        return "EPSG:4326"
    return str(crs)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safe division avoiding division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator, where=denominator!=0, out=np.full_like(numerator, default, dtype=float))
    return np.nan_to_num(result, nan=default)
