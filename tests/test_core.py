"""
Unit tests for HydroSim-RF core modules.

Comprehensive test suite ensuring scientific validity and reproducibility
of flood simulation engine and machine learning classifier.

Author: Letícia Caldas
License: MIT
"""

import pytest  # type: ignore
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulator import DiffusionWaveFloodModel
from src.ml.flood_classifier import (
    compute_topographic_features,
    train_flood_classifier,
    predict_probability,
)


class TestDiffusionWaveFloodModel:
    """Test suite for diffusion-wave hydrodynamic solver."""

    @pytest.fixture
    def setup_model(self):
        """Create a test flood model."""
        dem = np.random.rand(50, 50) * 10  # 50x50 grid, elevation 0-10m
        sources = np.zeros((50, 50), dtype=bool)
        sources[20:30, 20:30] = True  # 10x10 source area
        
        model = DiffusionWaveFloodModel(
            dem_data=dem,
            sources_mask=sources,
            diffusion_rate=0.5,
            flood_threshold=0.1,
            cell_size_meters=25.0,
        )
        return model

    def test_model_initialization(self, setup_model):
        """Test model initializes with correct dimensions."""
        model = setup_model
        assert model.water_height.shape == (50, 50)
        assert model.altitude.shape == (50, 50)
        assert model.simulation_time_minutes == 0
        assert model.overflow_time_minutes is None

    def test_water_conservation(self, setup_model):
        """Test that water volume is conserved during simulation."""
        model = setup_model
        model.apply_rainfall(10.0)  # 10 mm rainfall
        initial_volume = np.sum(model.water_height)
        
        for _ in range(10):
            model.advance_flow()
        
        final_volume = np.sum(model.water_height)
        # Allow small tolerance for floating-point errors
        assert abs(initial_volume - final_volume) / (initial_volume + 1e-9) < 0.01

    def test_rainfall_application(self, setup_model):
        """Test rainfall is correctly applied."""
        model = setup_model
        initial_water = model.water_height.copy()
        
        model.apply_rainfall(5.0)  # 5 mm = 0.005 m
        
        # Check that water increased
        added_water = model.water_height - initial_water
        assert np.any(added_water > 0)
        
        # Check volume increase matches rainfall
        expected_increase = 5.0 / 1000.0 * np.sum(model._valid_mask)
        actual_increase = np.sum(added_water)
        assert abs(expected_increase - actual_increase) < 1.0  # Allow some tolerance

    def test_flow_downslope(self, setup_model):
        """Test that water flows downslope."""
        model = setup_model
        # Create simplified DEM with clear slope
        model.altitude = np.arange(2500, dtype=float).reshape(50, 50) / 100.0
        model.water_height[:] = 0
        model.water_height[0, 0] = 1.0  # Add water at high point
        
        initial_pos_water = model.water_height[0, 0]
        
        model.advance_flow()
        
        # Water should have moved away from initial cell
        assert model.water_height[0, 0] < initial_pos_water

    def test_diagnostics_recording(self, setup_model):
        """Test that diagnostics are properly recorded."""
        model = setup_model
        model.apply_rainfall(10.0)
        model.record_diagnostics(10)
        
        assert len(model.history) == 1
        assert model.history[0]["time_minutes"] == 10
        assert model.history[0]["flooded_percent"] >= 0
        assert model.history[0]["total_water_volume_m3"] >= 0

    def test_reproducibility_with_seed(self):
        """Test simulation reproducibility with fixed seed."""
        dem = np.random.RandomState(42).rand(30, 30) * 10
        sources = np.zeros((30, 30), dtype=bool)
        sources[10:20, 10:20] = True
        
        # Run first simulation
        model1 = DiffusionWaveFloodModel(dem, sources, 0.5, 0.1, 25.0)
        for _ in range(20):
            model1.apply_rainfall(5.0)
            model1.advance_flow()
        result1 = model1.water_height.copy()
        
        # Run second simulation with identical parameters
        model2 = DiffusionWaveFloodModel(dem, sources, 0.5, 0.1, 25.0)
        for _ in range(20):
            model2.apply_rainfall(5.0)
            model2.advance_flow()
        result2 = model2.water_height.copy()
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2, decimal=6)


class TestFloodClassifier:
    """Test suite for Random Forest flood classifier."""

    @pytest.fixture
    def setup_data(self):
        """Create synthetic training data."""
        dem = np.random.rand(50, 50) * 100  # Synthetic DEM
        water = np.random.rand(50, 50) * 0.5
        water[10:30, 10:30] *= 2  # Add a flooded region
        return dem, water

    def test_feature_computation(self, setup_data):
        """Test topographic feature extraction."""
        dem, _ = setup_data
        X = compute_topographic_features(dem)
        
        assert X.shape == (2500, 2)  # 50x50 grid = 2500 cells, 2 features
        assert np.all((X >= 0) & (X <= 1))  # Features should be normalized

    def test_classifier_training(self, setup_data):
        """Test Random Forest classifier training."""
        dem, water = setup_data
        clf = train_flood_classifier(dem, water, threshold=0.2, n_estimators=10)
        
        assert clf.n_estimators == 10  # type: ignore
        assert clf.n_features_in_ == 2
        assert hasattr(clf, 'feature_importances_')

    def test_probability_prediction(self, setup_data):
        """Test flood probability prediction."""
        dem, water = setup_data
        clf = train_flood_classifier(dem, water, threshold=0.2, n_estimators=10)
        prob = predict_probability(clf, dem)
        
        assert prob.shape == dem.shape
        assert np.all((prob >= 0) & (prob <= 1))

    def test_classifier_reproducibility(self, setup_data):
        """Test classifier reproducibility with fixed seed."""
        dem, water = setup_data
        
        # Train first classifier
        clf1 = train_flood_classifier(
            dem, water, threshold=0.2, n_estimators=10, random_state=42
        )
        prob1 = predict_probability(clf1, dem)
        
        # Train second classifier
        clf2 = train_flood_classifier(
            dem, water, threshold=0.2, n_estimators=10, random_state=42
        )
        prob2 = predict_probability(clf2, dem)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(prob1, prob2, decimal=6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
