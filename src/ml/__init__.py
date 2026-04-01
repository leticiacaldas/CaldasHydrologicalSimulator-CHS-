"""
Machine learning module for HydroSim-RF.

Implements flood inundation probability estimation using scikit-learn's
Random Forest classifier trained on topographic indices.
"""

from .flood_classifier import (
    compute_topographic_features,
    train_flood_classifier,
    predict_probability,
    evaluate_classifier,
)

__all__ = [
    "compute_topographic_features",
    "train_flood_classifier",
    "predict_probability",
    "evaluate_classifier",
]
