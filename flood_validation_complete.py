#!/usr/bin/env python3
"""
Complete example: Flood simulation validation using ML and real INMET data.

This script demonstrates:
1. Load INMET rainfall data (real weather)
2. Run flood simulation
3. Load real flood extent map
4. Create integrated dataset
5. Train Random Forest model
6. Validate against real floods
7. Generate comparison visualizations

Reference:
    Environmental Modelling & Software journal publication
    DOI: 10.1016/j.envsoft.xxx
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import DiffusionWaveFloodModel
from src.data.inmet_loader import INMETDataLoader, create_inmet_dataset
from src.ml.flood_validation import FloodValidationModel, compare_predictions
from src.ml.flood_validation_viz import (
    plot_comparison_maps, plot_metrics_comparison,
    plot_confusion_matrices, plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete flood validation workflow."""
    
    print("\n" + "="*70)
    print("FLOOD SIMULATION VALIDATION WITH ML")
    print("Integrating INMET Real Data + Physics Simulation + Machine Learning")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: Load Real Rainfall Data from INMET
    # ========================================================================
    logger.info("STEP 1️⃣  Loading Real Rainfall Data")
    logger.info("-" * 70)
    
    inmet = INMETDataLoader()
    
    # Load rainfall from INMET for Rio Grande do Sul flood event (May 2024)
    rain_df = inmet.load_rainfall_data(
        station_id='RS_PORTO_ALEGRE',
        date_start='2024-04-01',
        date_end='2024-05-31'
    )
    
    logger.info(f"✅ Loaded {len(rain_df)} days of rainfall data")
    logger.info(f"   Total rainfall: {rain_df['rainfall_mm'].sum():.1f} mm")
    logger.info(f"   Peak daily rainfall: {rain_df['rainfall_mm'].max():.1f} mm")
    
    # ========================================================================
    # STEP 2: Generate Synthetic DEM (or load real DEM)
    # ========================================================================
    logger.info("\n\nSTEP 2️⃣  Generating Digital Elevation Model (DEM)")
    logger.info("-" * 70)
    
    H, W = 100, 100
    dem = _generate_synthetic_dem(H, W)
    
    logger.info(f"✅ Generated {H}x{W} DEM")
    logger.info(f"   Elevation range: {dem.min():.1f} - {dem.max():.1f} m")
    
    # ========================================================================
    # STEP 3: Run Physics-based Flood Simulation
    # ========================================================================
    logger.info("\n\nSTEP 3️⃣  Running Physics-based Flood Simulation")
    logger.info("-" * 70)
    
    sources = _generate_rainfall_sources(H, W)
    
    model = DiffusionWaveFloodModel(
        dem_data=dem,
        sources_mask=sources,
        diffusion_rate=0.5,
        flood_threshold=0.1,
        cell_size_meters=25.0,
    )
    
    # Simulate with rainfall from INMET
    total_rainfall = rain_df['rainfall_mm'].sum()
    num_steps = 100  # Simulation timesteps
    rainfall_per_step = total_rainfall / num_steps
    
    logger.info(f"Running {num_steps} timesteps")
    logger.info(f"Rainfall per step: {rainfall_per_step:.2f} mm")
    
    for step in range(num_steps):
        model.apply_rainfall(rainfall_per_step)
        model.advance_flow()
        model.record_diagnostics(10)
        
        if (step + 1) % 25 == 0:
            logger.info(f"   Step {step+1}/{num_steps}")
    
    simulated_flood = (model.water_height > 0.1).astype(int)
    
    logger.info(f"✅ Simulation complete")
    logger.info(f"   Flooded cells: {simulated_flood.sum()} ({100*simulated_flood.mean():.1f}%)")
    
    # ========================================================================
    # STEP 4: Load Real Flood Extent Map
    # ========================================================================
    logger.info("\n\nSTEP 4️⃣  Loading Real Flood Extent (from INMET observations)")
    logger.info("-" * 70)
    
    real_flood = inmet.load_real_flood_map('RS_2024_05')
    
    logger.info(f"✅ Real flood map loaded")
    logger.info(f"   Flooded cells: {real_flood.sum()} ({100*real_flood.mean():.1f}%)")
    
    # ========================================================================
    # STEP 5: Create Integrated Dataset
    # ========================================================================
    logger.info("\n\nSTEP 5️⃣  Creating Integrated Dataset")
    logger.info("-" * 70)
    
    dataset = create_inmet_dataset(
        dem=dem,
        simulated_flood=simulated_flood,
        real_flood=real_flood,
        rainfall_df=rain_df,
        cell_size_meters=25.0
    )
    
    logger.info(f"✅ Dataset created")
    logger.info(f"   Shape: {dataset.shape}")
    logger.info(f"\n   Features:")
    for col in dataset.columns[:-1]:
        logger.info(f"      • {col}")
    logger.info(f"   Target: real_flood")
    
    # ========================================================================
    # STEP 6: Train Machine Learning Model
    # ========================================================================
    logger.info("\n\nSTEP 6️⃣  Training Random Forest Classifier")
    logger.info("-" * 70)
    
    ml_model = FloodValidationModel(
        n_estimators=100,
        random_state=42,
        test_size=0.2
    )
    
    # Prepare data
    feature_cols = ['elevation', 'slope', 'rainfall_total', 'rainfall_max_daily', 'simulated_flood']
    ml_model.prepare_data(dataset, feature_cols=feature_cols, target_col='real_flood')
    
    # Train
    ml_model.train()
    
    # Cross-validation
    cv_scores = ml_model.cross_validate(cv=5)
    
    # ========================================================================
    # STEP 7: Evaluate Model
    # ========================================================================
    logger.info("\n\nSTEP 7️⃣  Model Evaluation")
    logger.info("-" * 70)
    
    metrics = ml_model.evaluate(verbose=True)
    
    # Get feature importance
    importance_df = ml_model.get_feature_importance(top_n=5)
    logger.info("\n📊 Top Feature Importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"   {row['feature']:.<30} {row['importance']:.4f}")
    
    # ========================================================================
    # STEP 8: Compare All Three Maps
    # ========================================================================
    logger.info("\n\nSTEP 8️⃣  Comparing Predictions")
    logger.info("-" * 70)
    
    predicted_proba = ml_model.predict_flood_map(dataset).reshape(H, W)
    
    comparison = compare_predictions(
        real_flood=real_flood,
        simulated_flood=simulated_flood,
        predicted_flood=predicted_proba
    )
    
    # ========================================================================
    # STEP 9: Generate Visualizations
    # ========================================================================
    logger.info("\n\nSTEP 9️⃣  Generating Visualizations")
    logger.info("-" * 70)
    
    output_dir = Path("outputs/flood_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comparison maps
    plot_comparison_maps(
        real_flood=real_flood,
        simulated_flood=simulated_flood,
        predicted_flood=predicted_proba,
        dem=dem,
        output_path=output_dir / "01_comparison_maps.png"
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        metrics=comparison,
        output_path=output_dir / "02_metrics_comparison.png"
    )
    
    # Confusion matrices
    plot_confusion_matrices(
        real_flood=real_flood,
        simulated_flood=simulated_flood,
        predicted_flood=predicted_proba,
        output_path=output_dir / "03_confusion_matrices.png"
    )
    
    # Feature importance
    plot_feature_importance(
        importance_df,
        output_path=output_dir / "04_feature_importance.png",
        top_n=len(importance_df)
    )
    
    # ========================================================================
    # STEP 10: Summary Report
    # ========================================================================
    logger.info("\n\nSTEP 🔟  Summary Report")
    logger.info("-" * 70)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total cells: {H*W:,}")
    print(f"   Real flooded: {real_flood.sum()} ({100*real_flood.mean():.1f}%)")
    print(f"   Simulated flooded: {simulated_flood.sum()} ({100*simulated_flood.mean():.1f}%)")
    print(f"   ML predicted flooded: {(predicted_proba > 0.5).sum()} ({100*(predicted_proba > 0.5).mean():.1f}%)")
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    
    print(f"\n📈 IMPROVEMENT OVER SIMULATION:")
    print(f"   Simulation Accuracy: {comparison['simulation_accuracy']:.4f}")
    print(f"   ML Accuracy: {comparison['prediction_accuracy']:.4f}")
    print(f"   ✅ Improvement: {comparison['improvement']:+.4f} ({100*comparison['improvement']/comparison['simulation_accuracy']:+.1f}%)")
    
    print(f"\n📁 OUTPUTS:")
    print(f"   Saved to: {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"      • {f.name}")
    
    print("\n" + "="*70 + "\n")
    
    logger.info("✅ WORKFLOW COMPLETE!")
    logger.info("   Next steps:")
    logger.info("   1. Review generated PNG files")
    logger.info("   2. Adjust model parameters if needed")
    logger.info("   3. Integrate into web interface")
    logger.info("   4. Prepare for publication")


def _generate_synthetic_dem(H: int, W: int) -> np.ndarray:
    """Generate synthetic DEM."""
    x = np.arange(W) / W
    y = np.arange(H) / H
    X, Y = np.meshgrid(x, y)
    
    dem = 50.0 + 30.0 * (X + Y)
    
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 6
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                dem[i, j] -= 15.0 * (1.0 - dist / radius) ** 2
    
    dem += np.random.normal(0, 0.5, (H, W))
    return dem


def _generate_rainfall_sources(H: int, W: int) -> np.ndarray:
    """Generate rainfall source areas."""
    sources = np.zeros((H, W), dtype=bool)
    sources[10:25, 10:25] = True
    sources[10:25, 75:90] = True
    sources[75:90, 40:60] = True
    return sources


if __name__ == '__main__':
    main()
