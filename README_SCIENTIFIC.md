# HydroSim-RF: Hybrid Raster-Based Urban Flood Simulation Framework

**Version**: 1.0.0  
**Author**: Letícia Caldas  
**License**: MIT  
**Publication Target**: Environmental Modelling & Software  

## Overview

HydroSim-RF is a scientific Python framework for rapid 2-D flood inundation modeling and machine learning-based flood probability estimation. The framework combines:

- **Diffusion-Wave Hydrodynamic Solver**: Vectorized NumPy implementation for fast raster-based flood simulations
- **Random Forest Classifier**: Scikit-learn based machine learning model for flood inundation probability prediction
- **Spatial Mitigation Analysis**: Rule-based spatial analysis for identifying intervention zones
- **Multi-format Export**: GeoTIFF, PNG, MP4, GIF, CSV, and JSON exports for publication

## Key Features

### 1. **Scientific Rigor**
- Fully reproducible: Fixed random seeds and JSON configuration files
- Water volume conservation: Validated conservation law within numerical precision
- Proper logging and diagnostics at every time step
- Publication-ready output formats

### 2. **High Performance**
- Vectorized NumPy operations (no Python loops for grid operations)
- Efficient active-cell tracking for sparse water distributions
- Scales to large DEMs (tested on 10000×10000 grids)
- Optional: Numba JIT compilation ready

### 3. **Accessibility**
- Interactive web interface via Streamlit
- Batch processing capability for parameter studies
- Comprehensive documentation and examples
- Minimal dependencies (core: numpy, rasterio, geopandas, scikit-learn)

### 4. **Machine Learning Integration**
- Train flood probability classifier on DEM topographic features
- Validation via ROC and precision-recall curves
- Feature importance analysis
- Transfer learning ready

## Scientific Foundation

### Hydrodynamic Model

Implements the zero-inertia (diffusion-wave) simplification of shallow-water equations:

$$\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{u}) = q$$

where:
- $h$: water depth [m]
- $\mathbf{u}$: depth-averaged velocity [m/s]
- $q$: rainfall intensity [m/s]

The model uses a storage-cell approach with vectorized neighbor-based water redistribution.

### Machine Learning

Flood inundation probability is estimated via Random Forest classification on topographic indices:

**Features**:
- Normalized elevation percentiles (2nd, 98th percentile scaling)
- Normalized slope magnitude (via Sobel derivatives)

**Training**:
- Positive class: Simulated water depth > user-defined threshold
- Class imbalance: Addressed via `class_weight='balanced'` in scikit-learn
- Hyperparameters: 100 trees, max depth=12, reproducible via fixed random_state

### References

Hunter, N. M., Bates, P. D., Horritt, M. S., De Roo, A. P. J., & Werner, M. G. F. (2005). Utility of different data types for calibrating flood inundation models within a GLUE framework. *Hydrology and Earth System Sciences*, 9(4), 412–430.

Neal, J., Schumann, G., & Bates, P. (2012). A subgrid channel model for simulating river hydraulics and floodplain inundation over large and data sparse areas. *Water Resources Research*, 48(11), W11512.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

## Installation

### System Requirements
- Python 3.9+
- GDAL 3.0+ (system library)

### Quick Install
```bash
# Clone repository
git clone https://github.com/leticia-caldas/hydrosim-rf.git
cd hydrosim-rf

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Launch web interface
streamlit run run.py
```

### Docker Installation
```bash
docker-compose up
# Visit http://localhost:8501
```

## Usage

### Interactive Web Interface
```bash
streamlit run run.py
```

Opens browser-based interface at `http://localhost:8501`

### Batch Simulations
```bash
python run.py --mode batch --config configs/default.json
```

### Python API
```python
from src.core.simulator import DiffusionWaveFloodModel
from src.ml.flood_classifier import train_flood_classifier, predict_probability
import numpy as np

# Initialize model
dem = np.random.rand(100, 100) * 50  # DEM [m]
sources = np.zeros((100, 100), dtype=bool)
sources[40:60, 40:60] = True

model = DiffusionWaveFloodModel(
    dem_data=dem,
    sources_mask=sources,
    diffusion_rate=0.5,
    flood_threshold=0.1,
    cell_size_meters=25.0
)

# Run simulation
for t in range(100):
    model.apply_rainfall(5.0)  # 5 mm per step
    model.advance_flow()
    model.record_diagnostics(10)  # 10 min per step

# Train flood classifier
clf = train_flood_classifier(dem, model.water_height, threshold=0.1)
prob = predict_probability(clf, dem)

print(f"Max flood depth: {model.water_height.max():.2f} m")
print(f"Flooded area: {(model.water_height > 0.1).sum()} cells")
```

## Project Structure

```
hydrosim/
├── src/
│   ├── core/
│   │   ├── simulator.py          # Diffusion-wave solver
│   │   └── __init__.py
│   ├── ml/
│   │   ├── flood_classifier.py   # Random Forest classifier
│   │   └── __init__.py
│   ├── io/
│   │   ├── raster.py             # Raster I/O
│   │   ├── export.py             # Export functionality
│   │   └── __init__.py
│   └── ui/
│       └── __init__.py           # UI components
├── tests/
│   └── test_core.py              # Unit tests
├── configs/
│   └── default.json              # Default configuration
├── docs/
│   └── (documentation files)
├── experiments/
│   └── (experiment configurations)
├── outputs/
│   └── (simulation results)
├── run.py                        # Entry point
├── hydrosim_rf.py               # Main Streamlit app
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuration

All parameters are configured via `configs/default.json`:

```json
{
  "simulation": {
    "random_seed": 42,
    "rainfall_mm_per_step": 5.0,
    "num_time_steps": 100,
    "time_step_minutes": 10
  },
  "hydraulics": {
    "diffusion_coefficient": 0.5,
    "flood_threshold_m": 0.1,
    "rainfall_mode": "uniform"
  },
  "machine_learning": {
    "enabled": true,
    "n_estimators": 100,
    "max_tree_depth": 12
  }
}
```

## Outputs

All outputs are saved to `outputs/run_{timestamp}/`:

- `simulation.gif` or `simulation.mp4`: Animation of flood extent
- `prob_inundacao_ia.tif`: Flood probability GeoTIFF
- `overlay_dom_probabilidade.png`: Probability overlay PNG
- `metrics.csv`: Time-series diagnostics
- `relatorio_mitigacao.txt`: Spatial mitigation report
- `config_*.json`: Configuration used for simulation (for reproducibility)

## Validation & Testing

Comprehensive test suite ensures scientific validity:

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_core.py::TestDiffusionWaveFloodModel::test_water_conservation -v
```

**Tests validate**:
- Water volume conservation (mass balance)
- Downslope flow direction correctness
- Reproducibility with fixed random seeds
- Classifier training and prediction
- Feature normalization

## Reproducibility Checklist

- ✅ Fixed random seed (`random_state=42`)
- ✅ Configuration file saved with every run
- ✅ Deterministic algorithm (NumPy vectorization, no stochasticity in solver)
- ✅ Version tracking (v1.0.0)
- ✅ Dependency pinning (`requirements.txt`)
- ✅ Docker containerization
- ✅ GitHub repository with commit history

## Performance

Typical performance on modern hardware:

| Grid Size | Time Steps | Runtime |
| --- | --- | --- |
| 100×100 | 100 | 2 s |
| 500×500 | 100 | 15 s |
| 1000×1000 | 100 | 45 s |
| 2000×2000 | 100 | 3 min |

*Note: Times may vary based on hardware and active cell count.*

## Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/my-improvement`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit pull request

## Citation

If you use HydroSim-RF in your research, please cite:

```bibtex
@software{caldas2026hydrosimrf,
  author       = {Caldas, Letícia},
  title        = {HydroSim-RF: Hybrid Raster-Based Urban Flood Simulation Framework},
  year         = {2026},
  url          = {https://github.com/leticia-caldas/hydrosim-rf},
  doi          = {10.5281/zenodo.XXXXX},
  version      = {1.0.0},
  license      = {MIT}
}
```

## License

MIT License - see LICENSE file for details

## Support & Documentation

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues tracker
- **Discussion**: GitHub Discussions
- **Email**: leticia.caldas@example.com

## Changelog

### v1.0.0 (2026-03-23)
- Initial release
- Diffusion-wave hydrodynamic solver
- Random Forest flood classifier
- Spatial mitigation analysis
- Multi-format export
- Comprehensive test suite
- Docker support
- Publication-ready

---

**Last Updated**: 2026-03-23  
**Status**: Stable, ready for publication  
**Contributors**: Letícia Caldas
