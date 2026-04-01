"""
Random Forest flood inundation classifier module.

Implements scikit-learn based machine learning approach for flood probability
estimation based on topographic indices derived from DEM.

References:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

Author: Letícia Caldas
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging

logger = logging.getLogger(__name__)


def compute_topographic_features(dem: np.ndarray) -> np.ndarray:
    """
    Compute topographic indices from a Digital Elevation Model.

    Derives normalized elevation and slope magnitude as features for machine
    learning classification.

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m].

    Returns
    -------
    X : np.ndarray, shape (H*W, 2)
        Feature matrix containing [normalized_elevation, normalized_slope] 
        for each grid cell.

    Notes
    -----
    Features are normalized to [0, 1] range using percentile scaling to ensure
    robustness to outliers.
    """
    dem = np.asarray(dem, dtype=float)
    dem_valid = dem[np.isfinite(dem)]

    # Normalize elevation
    if dem_valid.size > 0:
        p2, p98 = np.percentile(dem_valid, (2, 98))
        p2 = max(float(p2), 0.0)
        denom = max(float(p98) - p2, 1e-6)
        dem_norm = np.clip((dem - p2) / denom, 0.0, 1.0)
    else:
        dem_norm = np.zeros_like(dem)

    # Compute slope via Sobel derivatives
    gy, gx = np.gradient(dem, edge_order=2)
    slope = np.sqrt(gx**2 + gy**2 + 1e-9)
    slope_valid = slope[np.isfinite(slope)]

    if slope_valid.size > 0:
        p2, p98 = np.percentile(slope_valid, (2, 98))
        p2 = max(float(p2), 0.0)
        denom = max(float(p98) - p2, 1e-6)
        slope_norm = np.clip((slope - p2) / denom, 0.0, 1.0)
    else:
        slope_norm = np.zeros_like(slope)

    X = np.column_stack([dem_norm.ravel(), slope_norm.ravel()])
    return X


def train_flood_classifier(
    dem: np.ndarray,
    water: np.ndarray,
    threshold: float,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest binary classifier for flood inundation prediction.

    The classifier is trained on topographic indices derived from the DEM with
    binary labels from simulated water depth. Class imbalance is handled via
    class_weight='balanced'.

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m].
    water : np.ndarray, shape (H, W)
        Simulated water depth array [m].
    threshold : float
        Water depth threshold [m] for positive class assignment.
    n_estimators : int, optional
        Number of decision trees (default 100).
    max_depth : int, optional
        Maximum tree depth (default 12).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    clf : RandomForestClassifier
        Fitted classifier.

    Notes
    -----
    This function is reproducible when run with the same random_state.
    Seeding should be handled externally via np.random.seed() if required.
    """
    X = compute_topographic_features(dem)
    y = (water.reshape(-1) > float(threshold)).astype(np.uint8)

    logger.info(
        f"Training RandomForest: {n_estimators} trees, "
        f"max_depth={max_depth}, {np.sum(y)} positive samples"
    )

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        class_weight="balanced",
        n_jobs=-1,
        random_state=int(random_state),
    )
    clf.fit(X, y)

    logger.info(f"Training complete. Feature importances: {clf.feature_importances_}")
    return clf


def predict_probability(model: RandomForestClassifier, dem: np.ndarray) -> np.ndarray:
    """
    Predict flood inundation probability from a trained Random Forest model.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained flood classifier.
    dem : np.ndarray, shape (H, W)
        Ground-surface elevation raster [m].

    Returns
    -------
    proba : np.ndarray, shape (H, W)
        Predicted inundation probability [0, 1] at each grid cell.
    """
    X = compute_topographic_features(dem)
    proba = model.predict_proba(X)

    if proba.shape[1] == 1:
        p_flood = np.zeros((dem.size,), dtype=float)
    else:
        p_flood = proba[:, 1]

    return p_flood.reshape(dem.shape)


def evaluate_classifier(
    model: RandomForestClassifier,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate classifier performance using standard metrics.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained classifier.
    y_true : np.ndarray, shape (N,)
        True binary labels.
    y_score : np.ndarray, shape (N,)
        Predicted probability scores.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - roc_auc: Area under ROC curve
        - ap: Average precision
        - feature_importances: Feature importance array
    """
    metrics: Dict[str, Any] = {}

    # Handle edge cases
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    if valid.sum() < 2:
        logger.warning("Insufficient valid samples for evaluation")
        return metrics

    y_true_valid = y_true[valid]
    y_score_valid = y_score[valid]

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_valid, y_score_valid))
        logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
    except Exception as e:
        logger.warning(f"ROC AUC computation failed: {e}")

    try:
        metrics["ap"] = float(average_precision_score(y_true_valid, y_score_valid))
        logger.info(f"Average Precision: {metrics['ap']:.3f}")
    except Exception as e:
        logger.warning(f"AP computation failed: {e}")

    try:
        metrics["feature_importances"] = model.feature_importances_.tolist()
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")

    return metrics
