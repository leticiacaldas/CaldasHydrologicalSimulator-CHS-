# CaldasHydrologicalSimulator (CHS): Hybrid Raster-Based Urban Flood Simulation Framework

## Overview

**CaldasHydrologicalSimulator (CHS)** is an interactive web application for rapid 2D urban flood simulation using Digital Elevation Models (DEMs). It implements:

- **Hydrodynamic Core**: Diffusion wave approximation via vectorized NumPy solver (`DiffusionWaveFloodModel`)
- **Machine Learning Classifier**: Random Forest for flood probability estimation without requiring calibration data
- **Spatial Analysis**: Automatic identification of eligible zones for flood mitigation (reforestation, levees, drainage, land elevation)
- **Interactive Visualization**: Streamlit with online basemaps, animations (GIF/MP4), and data export

## System Requirements

### Local (with venv)

- **Python**: 3.9+
- **GDAL**: Operating system (Linux/macOS/Windows)
- **Memory**: 4GB recommended
- **Processor**: Any modern processor

### Docker

- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Disk Space**: 2-3GB for the image

## Installation

### Option 1: Local Virtual Environment (venv)

```bash
cd /path/to/CaldasHydrologicalSimulator

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run hydrosim_rf.py
```

The application will be available at: **[http://localhost:8501](http://localhost:8501)**

### Option 2: Docker (Recommended)

```bash
cd /path/to/CaldasHydrologicalSimulator

# Build image
docker build -t chs .

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -p 8501:8501 -v $(pwd)/data:/app/data chs
```

**The application will be available at**: [http://localhost:8501](http://localhost:8501)

## Project Structure

```text
CaldasHydrologicalSimulator/
├── hydrosim_rf.py              # Main application (Streamlit UI)
├── web_server_v3.py            # REST API server
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Container orchestration
├── .dockerignore               # Docker exclusions
├── .gitignore                  # Git exclusions
├── README.md                   # This file
├── configs/
│   └── default.json            # Configuration file
├── data/
│   ├── input/                  # Input DEMs and vectors
│   └── output/                 # Simulation results
├── src/
│   ├── core/                   # Core hydrodynamic models
│   ├── data/                   # Data loading modules
│   ├── io/                     # I/O and export formats
│   ├── ml/                     # Machine learning classifiers
│   └── ui/                     # UI components
├── logs/                       # Log files
└── venv/                       # Python virtual environment
```

## Key Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `streamlit` | ≥1.28.0 | Interactive web framework |
| `numpy` | ≥1.24.0 | Numerical computation (solver) |
| `rasterio` | ≥1.3.0 | Geospatial raster I/O |
| `geopandas` | ≥0.13.0 | Vector data processing |
| `scikit-learn` | ≥1.3.0 | Random Forest classifier |
| `matplotlib` | ≥3.7.0 | Visualization and animations |
| `imageio-ffmpeg` | ≥0.4.8 | MP4 video export |

## Usage Guide

### 1. Prepare Input Data

You will need:

- **DEM (GeoTIFF)**: Elevation raster in geographic coordinates
- **Source Polygons (optional)**: Polygons (GeoPackage/.shp) defining rainfall areas
- **River Network (optional)**: Polygons/lines defining the drainage network
- **Orthophoto (optional)**: Aerial imagery for visualization

### 2. Run Simulation

1. Open the application: [http://localhost:8501](http://localhost:8501)
2. **Simulation Tab**:
   - Upload DEM (required)
   - Upload source polygons (optional)
   - Configure rainfall parameters, time step, diffusion coefficient
   - Click **"Run Simulation"**
3. Animation is generated in real-time

### 3. ML-Based Validation

1. After simulation, in the **"Random Forest Inundation Probability"** section:
   - Train model with customizable `n_estimators` and `max_depth`
   - Model learns topography-inundation relationships

### 4. Mitigation Analysis

1. In the **"Spatial Flood Mitigation Analysis"** section:
   - Run automatic spatial analysis
   - Identify eligible zones for:
     - **Reforestation / green infrastructure**
     - **Levees / flood barriers**
     - **Drainage systems**
     - **Land elevation / terracing**
2. Generate detailed report with recommendations

## Advanced Features

### Data Export

- **CSV**: Time series diagnostics (flooded area, volume, depth)
- **GeoTIFF**: Probability rasters (ML) and simulated flood extent
- **PNG**: Map overlays
- **GIF/MP4**: Flood animations
- **ZIP**: Complete package with all artifacts

### Visual Customization

- **Basemap**: Choose from Esri, CartoDB, OpenStreetMap, or none
- **Hillshade**: Analytical relief shading of DEM
- **Transparency & Colors**: Control water visualization
- **Contours**: Highlight flood extent boundaries

### Hydrodynamic Parameters

- **Diffusion coefficient (α)**: 0.01–1.0 (determines lateral propagation)
- **Flood threshold (h\*)**: Minimum depth for classification
- **Time step size**: 1–1440 minutes
- **Resampling factor**: 1×–16× (speed vs. accuracy)

## Troubleshooting

### "ModuleNotFoundError: No module named 'rasterio'"

```bash
source venv/bin/activate
pip install --upgrade rasterio geopandas
```

### "GDAL not found"

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev

# macOS
brew install gdal proj geos

# Use Docker if problems persist
docker-compose up
```

### "AnimationError: no writer for format 'mp4'"

```bash
pip install imageio-ffmpeg
```

### Slow application (large-scale simulation)

- Increase **"Grid resampling factor"** to 4, 8, or 16
- Reduce animation duration ("Quick preview" for testing)
- Run in Docker with memory limits: `docker-compose.yml`

## Code Structure

### Core Classes

#### `DiffusionWaveFloodModel`

Simulation engine. Implements 2D water propagation with:

- Water redistribution to lower-elevation neighboring cells
- Water volume conservation
- Temporal diagnostics logging
- Support for source and river masks

#### `RandomForestClassifier` (scikit-learn)

Trained on:

- **Inputs**: Normalized elevation, normalized slope (DEM derivatives)
- **Output**: Binary flood probability (0–1)
- **Usage**: Transfer to new DEMs without full simulation

### Analysis Functions

- `_prepare_spatial_domain()`: Read DEM, resample, rasterize vectors
- `_identify_intervention_zones()`: Mitigation analysis (DBSCAN, morphological filters)
- `_train_flood_classifier()`: RF training with class balancing
- `_predict_probability()`: Prediction on new DEM
- `_build_mitigation_report()`: Structured report with scientific citations

## Scientific References

- **Hunter et al. (2005)**: Diffusion-wave formulation for flood modeling
- **Breiman (2001)**: Random Forests (classifier theory)
- **Neal et al. (2012)**: Zero-inertia approximation for hydrodynamics
- **Rogger et al. (2017)**: Impact of land-use change on floods
- **EU Directive 2007/60/EC**: Flood risk management

## License

MIT License — See LICENSE file for details

## Support

For issues, open an issue on the GitHub repository or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Author**: Letícia Caldas  
**Institution**: HydroLab Research Group
