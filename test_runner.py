#!/usr/bin/env python3
"""
Quick test runner for HydroSim-RF - generates synthetic test data and runs simulation.

This script demonstrates the scientific capabilities of the framework:
- Generates a synthetic DEM (Digital Elevation Model)
- Defines rainfall source areas
- Runs diffusion-wave hydrodynamic simulation
- Trains Random Forest flood classifier
- Exports results

Author: Letícia Caldas
License: MIT
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import DiffusionWaveFloodModel
from src.ml.flood_classifier import (
    compute_topographic_features,
    train_flood_classifier,
    predict_probability,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_synthetic_dem(shape=(100, 100), seed=42):
    """Generate synthetic DEM with valley for flood testing."""
    np.random.seed(seed)
    
    # Create base elevation with gentle slope
    H, W = shape
    x = np.arange(W) / W
    y = np.arange(H) / H
    X, Y = np.meshgrid(x, y)
    
    # Base: sloping plane
    dem = 50.0 + 30.0 * (X + Y)
    
    # Add valley (sink for water to accumulate)
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 6
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                dem[i, j] -= 15.0 * (1.0 - dist / radius) ** 2
    
    # Add small noise
    dem += np.random.normal(0, 0.5, shape)
    
    logger.info(f"Generated synthetic DEM: shape={dem.shape}, range=[{dem.min():.2f}, {dem.max():.2f}] m")
    return dem

def generate_rainfall_sources(shape=(100, 100), seed=42):
    """Define rainfall source areas (e.g., urban catchment)."""
    np.random.seed(seed)
    
    H, W = shape
    sources = np.zeros((H, W), dtype=bool)
    
    # Define multiple source areas
    # Source 1: top-left
    sources[10:25, 10:25] = True
    
    # Source 2: top-right
    sources[10:25, 75:90] = True
    
    # Source 3: bottom center
    sources[75:90, 40:60] = True
    
    logger.info(f"Generated rainfall sources: {np.sum(sources)} cells ({100*np.sum(sources)/(H*W):.1f}% of domain)")
    return sources

def run_simulation_test(dem, sources, rainfall_mm=100.0, num_steps=50):
    """Run flood simulation with synthetic data."""
    logger.info("="*70)
    logger.info("STARTING FLOOD SIMULATION")
    logger.info("="*70)
    
    # Initialize model
    model = DiffusionWaveFloodModel(
        dem_data=dem,
        sources_mask=sources,
        diffusion_rate=0.5,
        flood_threshold=0.1,
        cell_size_meters=25.0,
    )
    
    # Run simulation
    rainfall_per_step = rainfall_mm / num_steps
    logger.info(f"Simulation parameters:")
    logger.info(f"  - Total rainfall: {rainfall_mm} mm")
    logger.info(f"  - Rainfall per step: {rainfall_per_step:.2f} mm")
    logger.info(f"  - Number of steps: {num_steps}")
    logger.info(f"  - Time step duration: 10 minutes")
    logger.info(f"  - Total simulation time: {num_steps * 10} minutes")
    
    for step in range(num_steps):
        model.apply_rainfall(rainfall_per_step)
        model.advance_flow()
        model.record_diagnostics(10)  # 10 minute timestep
        
        if (step + 1) % 10 == 0:
            summary = model.get_summary()
            logger.info(f"Step {step+1}/{num_steps}: "
                       f"Flooded={summary['flooded_area_percent']:.2f}%, "
                       f"Max depth={summary['max_water_depth']:.3f} m, "
                       f"Vol={summary['total_water_volume_m3']:.1f} m³")
    
    logger.info("="*70)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*70)
    
    return model

def run_ml_test(dem, model_water):
    """Train and evaluate Random Forest flood classifier."""
    logger.info("="*70)
    logger.info("TRAINING RANDOM FOREST CLASSIFIER")
    logger.info("="*70)
    
    # Train classifier
    logger.info("Computing topographic features...")
    X = compute_topographic_features(dem)
    
    logger.info("Training Random Forest (100 trees)...")
    clf = train_flood_classifier(dem, model_water, threshold=0.2, n_estimators=100)
    
    logger.info(f"Trained model: {clf.n_estimators} trees, {clf.max_depth} max depth")  # type: ignore
    logger.info(f"Feature importances: elevation={clf.feature_importances_[0]:.3f}, slope={clf.feature_importances_[1]:.3f}")
    
    # Generate predictions
    logger.info("Generating flood probability predictions...")
    prob = predict_probability(clf, dem)
    
    logger.info(f"Probability range: [{prob.min():.3f}, {prob.max():.3f}]")
    logger.info(f"Mean probability: {prob.mean():.3f}")
    logger.info(f"High-risk cells (p>0.7): {np.sum(prob > 0.7)} ({100*np.sum(prob > 0.7)/prob.size:.1f}%)")
    
    logger.info("="*70)
    logger.info("ML CLASSIFIER COMPLETE")
    logger.info("="*70)
    
    return clf, prob

def save_results(dem, sources, model, prob, output_dir="outputs/test_run"):
    """Save results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}...")
    
    # Save as NPZ (NumPy compressed)
    np.savez(
        output_path / "results.npz",
        dem=dem,
        sources=sources,
        water_final=model.water_height,
        probability=prob,
    )
    logger.info(f"  ✓ Saved: results.npz")
    
    # Save simulation history as JSON
    with open(output_path / "history.json", 'w') as f:
        history_serializable = []
        for entry in model.history:
            entry_copy = entry.copy()
            # Convert any numpy types to Python types
            for k, v in entry_copy.items():
                if isinstance(v, np.integer):
                    entry_copy[k] = int(v)
                elif isinstance(v, np.floating):
                    entry_copy[k] = float(v)
            history_serializable.append(entry_copy)
        json.dump(history_serializable, f, indent=2)
    logger.info(f"  ✓ Saved: history.json ({len(model.history)} timesteps)")
    
    # Save summary statistics
    summary = {
        "simulation": {
            "timesteps": len(model.history),
            "final_time_minutes": model.simulation_time_minutes,
            "total_water_volume_m3": float(np.sum(model.water_height) * 625.0),  # 625 = 25m x 25m
            "flooded_cells": int(np.sum(model.water_height > model.flood_threshold)),
            "max_depth_m": float(np.max(model.water_height)),
        },
        "probability": {
            "mean_p": float(prob.mean()),
            "std_p": float(prob.std()),
            "high_risk_cells": int(np.sum(prob > 0.7)),
            "high_risk_pct": float(100 * np.sum(prob > 0.7) / prob.size),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved: summary.json")
    
    logger.info(f"\n📁 All results saved to: {output_path}/")
    return output_path

def main():
    """Main test runner."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              HydroSim-RF Test Runner                          ║
║         Comprehensive Simulation + ML Evaluation              ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    logger.info("Starting HydroSim-RF test run...")
    
    # Generate test data
    dem = generate_synthetic_dem(shape=(100, 100))
    sources = generate_rainfall_sources(shape=(100, 100))
    
    # Run simulation
    model = run_simulation_test(dem, sources, rainfall_mm=100.0, num_steps=50)
    
    # Train ML classifier
    clf, prob = run_ml_test(dem, model.water_height)
    
    # Save results
    output_path = save_results(dem, sources, model, prob)
    
    logger.info("""
✅ TEST RUN COMPLETE!

📊 Results generated:
   • results.npz      - NumPy arrays (DEM, sources, water, probability)
   • history.json     - Timestep diagnostics
   • summary.json     - Summary statistics
   • test_run.log     - Full execution log

🎯 Next steps:
   1. Open Streamlit app: streamlit run run.py
   2. Load results from outputs/test_run/
   3. Generate publication-ready visualizations
   4. Write paper for Environmental Modelling & Software
""")

if __name__ == "__main__":
    main()
