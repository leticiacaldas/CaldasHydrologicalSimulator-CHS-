"""Export functionality for multiple formats and simulation history management."""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Optional imports for scientific formats
try:
    import xarray as xr  # type: ignore
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import h5py  # type: ignore
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Ensure database directory exists
DB_DIR = Path("outputs/.database")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "simulations.db"


class SimulationHistory:
    """Manage simulation history in SQLite database."""
    
    @staticmethod
    def init_db():
        """Initialize database schema."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT,  -- JSON string
                results_path TEXT,
                max_depth_m REAL,
                flooded_area_percent REAL,
                total_volume_m3 REAL,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                format TEXT,  -- tif, netcdf, hdf5, etc
                file_path TEXT,
                export_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(simulation_id) REFERENCES simulations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Simulation database initialized")
    
    @staticmethod
    def add_simulation(name: str, parameters: Dict[str, Any], results_path: str,
                      max_depth: float, flooded_area: float, volume: float,
                      notes: str = "") -> Optional[int]:
        """Add a simulation record to history."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO simulations 
                (name, parameters, results_path, max_depth_m, flooded_area_percent, total_volume_m3, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                name,
                json.dumps(parameters),
                results_path,
                max_depth,
                flooded_area,
                volume,
                notes
            ))
            
            sim_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"✅ Simulation saved: {name} (ID: {sim_id})")
            return sim_id
        except Exception as e:
            logger.error(f"❌ Error saving simulation: {e}")
            return None
    
    @staticmethod
    def get_simulations() -> List[Dict[str, Any]]:
        """Get all simulations from history."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM simulations ORDER BY timestamp DESC')
            simulations = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return simulations
        except Exception as e:
            logger.error(f"❌ Error retrieving simulations: {e}")
            return []
    
    @staticmethod
    def add_export(simulation_id: int, format_type: str, file_path: str):
        """Record an export."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO exports (simulation_id, format, file_path)
                VALUES (?, ?, ?)
            ''', (simulation_id, format_type, file_path))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Export recorded: {format_type} -> {file_path}")
        except Exception as e:
            logger.error(f"❌ Error recording export: {e}")


def export_to_netcdf(water: np.ndarray, dem: np.ndarray, transform=None, crs=None,
                     output_path: str = "outputs/test_run/simulation.nc") -> Optional[str]:
    """Export water and DEM data to NetCDF format (scientific standard)."""
    try:
        if not XARRAY_AVAILABLE:
            logger.warning("⚠️ xarray not installed, skipping NetCDF export")
            return None
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create data arrays
        height, width = water.shape
        x = np.arange(width)
        y = np.arange(height)
        
        # Create Dataset
        ds = xr.Dataset(
            {
                'water_depth_m': (['y', 'x'], water),
                'elevation_m': (['y', 'x'], dem),
            },
            coords={
                'x': x,
                'y': y,
            }
        )
        
        # Add attributes
        ds.attrs['title'] = 'HydroSim Flood Simulation'
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['crs'] = str(crs) if crs else 'EPSG:4326'
        
        ds['water_depth_m'].attrs['units'] = 'meters'
        ds['water_depth_m'].attrs['description'] = 'Water depth above ground surface'
        ds['elevation_m'].attrs['units'] = 'meters'
        ds['elevation_m'].attrs['description'] = 'Digital Elevation Model'
        
        # Save to NetCDF
        ds.to_netcdf(output_path, encoding={'water_depth_m': {'zlib': True, 'complevel': 9}})
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"✅ NetCDF exported: {output_path} ({file_size_mb:.1f} MB)")
        return output_path
    except ImportError:
        logger.warning("⚠️ xarray not installed, skipping NetCDF export")
        return None
    except Exception as e:
        logger.error(f"❌ NetCDF export failed: {e}")
        return None


def export_to_hdf5(water: np.ndarray, dem: np.ndarray, transform=None, crs=None,
                   output_path: str = "outputs/test_run/simulation.h5") -> Optional[str]:
    """Export complete simulation data to HDF5 format (hierarchical scientific data)."""
    try:
        if not H5PY_AVAILABLE:
            logger.warning("⚠️ h5py not installed, skipping HDF5 export")
            return None
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Store raster data
            f.create_dataset('water_depth_m', data=water, compression='gzip', compression_opts=9)
            f.create_dataset('elevation_m', data=dem, compression='gzip', compression_opts=9)
            
            # Store metadata
            f.attrs['title'] = 'HydroSim Flood Simulation'
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['crs'] = str(crs) if crs else 'EPSG:4326'
            
            # Add metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['grid_shape'] = water.shape
            meta_group.attrs['max_depth_m'] = float(np.nanmax(water)) if not np.isnan(water).all() else 0.0
            meta_group.attrs['total_volume_m3'] = float(np.nansum(water) * 625)  # 25m cell
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"✅ HDF5 exported: {output_path} ({file_size_mb:.1f} MB)")
        return output_path
    except ImportError:
        logger.warning("⚠️ h5py not installed, skipping HDF5 export")
        return None
    except Exception as e:
        logger.error(f"❌ HDF5 export failed: {e}")
        return None


def export_comparison_report(simulations: List[Dict[str, Any]], 
                            output_path: str = "outputs/comparison_report.json") -> Optional[str]:
    """Export comparison report of multiple simulations."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "export_date": datetime.now().isoformat(),
            "num_simulations": len(simulations),
            "simulations": [
                {
                    "id": s.get('id'),
                    "name": s.get('name'),
                    "timestamp": s.get('timestamp'),
                    "max_depth_m": s.get('max_depth_m'),
                    "flooded_area_percent": s.get('flooded_area_percent'),
                    "total_volume_m3": s.get('total_volume_m3'),
                    "notes": s.get('notes'),
                }
                for s in simulations
            ]
        }
        
        # Add comparative statistics
        if simulations:
            depths = [float(s.get('max_depth_m', 0)) for s in simulations if s.get('max_depth_m') is not None]
            if depths:
                report['statistics'] = {
                    'min_max_depth': float(min(depths)),
                    'max_max_depth': float(max(depths)),
                    'avg_max_depth': float(np.mean(depths)),
                }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Comparison report: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Comparison export failed: {e}")
        return None
