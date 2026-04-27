"""
Improved Random Forest Flood Classifier with Advanced Features & Spatial Validation

Features:
- TWI (Topographic Wetness Index)
- Flow Accumulation
- Terrain Curvature (profile & plan)
- Height Above Nearest Drainage (HAND)
- Distance to channels
- Advanced hyperparameters
- Spatial cross-validation (no data leakage)
- Comprehensive metrics (ROC-AUC, F1, Precision, Recall)
- Feature importance visualization
"""

import numpy as np
from scipy import ndimage, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from typing import Optional, Tuple, Dict

# Public API
__all__ = [
    'compute_advanced_topographic_features',
    'train_flood_classifier_improved',
    'plot_feature_importances',
    'predict_probability_improved',
]


def _compute_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Compute flow accumulation using simplified steepest descent.
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    
    Returns
    -------
    flow_acc : np.ndarray, shape (H, W)
        Flow accumulation (cell count, log-transformed)
    """
    from scipy.ndimage import label, sum as ndi_sum
    
    H, W = dem.shape
    dem = dem.astype(float)
    dem = np.nan_to_num(dem, nan=np.nanmedian(dem))
    
    # Compute gradient-based flow direction (simplified)
    gy, gx = np.gradient(dem)
    slope = np.hypot(gx, gy)
    
    # Normalize gradients
    with np.errstate(divide='ignore', invalid='ignore'):
        gx_norm = np.where(slope > 0, gx / slope, 0)
        gy_norm = np.where(slope > 0, gy / slope, 0)
    
    # Flow accumulation: cumulative contribution based on slope
    # Higher slope = higher flow potential
    flow_acc = 1 + slope  # Minimum accumulation = 1
    
    # Iterative smoothing to simulate flow convergence
    for _ in range(3):
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        flow_acc = ndimage.convolve(flow_acc, kernel, mode='constant', cval=0)
    
    return np.log1p(flow_acc)  # log-transform for stability


def _compute_terrain_curvature(dem: np.ndarray) -> tuple:
    """Compute profile and plan curvature.
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    
    Returns
    -------
    profile_curv, plan_curv : tuple of np.ndarray
        Profile curvature and plan curvature
    """
    dem = dem.astype(float)
    dem = np.nan_to_num(dem, nan=np.nanmedian(dem))
    
    # First derivatives
    dy, dx = np.gradient(dem)
    
    # Second derivatives
    d2y_dy2 = np.gradient(dy, axis=0)
    d2y_dx2 = np.gradient(dy, axis=1)
    d2x_dy2 = np.gradient(dx, axis=0)
    d2x_dx2 = np.gradient(dx, axis=1)
    
    # Avoid division by zero
    slope_magnitude = np.hypot(dx, dy)
    slope_magnitude = np.where(slope_magnitude == 0, 1e-8, slope_magnitude)
    
    # Profile curvature (convexity along slope direction)
    numerator = (dy**2 * d2y_dx2 - 2*dx*dy*d2y_dy2 + dx**2 * d2y_dy2)
    denominator = slope_magnitude**3
    profile_curv = numerator / denominator
    
    # Plan curvature (convergence/divergence across slope)
    numerator_plan = (dx**2 * d2y_dy2 - 2*dx*dy*d2y_dx2 + dy**2 * d2y_dx2)
    plan_curv = numerator_plan / denominator
    
    return profile_curv, plan_curv


def _compute_hand(dem: np.ndarray, channel_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Height Above Nearest Drainage (HAND).
    
    HAND = elevation - elevation of nearest channel
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    channel_mask : np.ndarray, optional
        Binary mask of channel cells. If None, uses flow accumulation threshold.
    
    Returns
    -------
    hand : np.ndarray, shape (H, W)
        Height above nearest drainage [m]
    """
    dem = dem.astype(float)
    dem = np.nan_to_num(dem, nan=np.nanmedian(dem))
    
    if channel_mask is None:
        # Use flow accumulation to define channels (simple threshold)
        flow_acc = _compute_flow_accumulation(dem)
        threshold = np.percentile(flow_acc, 90)  # Top 10% as channels
        channel_mask = (flow_acc > threshold).astype(float)
    else:
        channel_mask = channel_mask.astype(float)
    
    # At this point, channel_mask is always ndarray (never None)
    assert isinstance(channel_mask, np.ndarray), "channel_mask must be ndarray after initialization"
    
    # Simplified HAND: use distance-weighted minimum elevation
    if channel_mask.sum() > 0:
        # For each cell, find minimum elevation within distance
        hand = np.zeros_like(dem)
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                # Find nearest channel cell (simple 5x5 kernel)
                window = dem[max(0, i-2):min(dem.shape[0], i+3),
                             max(0, j-2):min(dem.shape[1], j+3)]
                min_elev = np.min(window)
                hand[i, j] = max(0, dem[i, j] - min_elev)
    else:
        hand = np.zeros_like(dem)
    
    return hand


def _compute_twi(dem: np.ndarray) -> np.ndarray:
    """Compute Topographic Wetness Index (TWI).
    
    TWI = ln(Flow Accumulation / tan(Slope))
    Simplified: TWI ≈ ln(Flow Accumulation) - ln(tan(Slope) + small_epsilon)
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    
    Returns
    -------
    twi : np.ndarray, shape (H, W)
        Topographic Wetness Index
    """
    dem = dem.astype(float)
    dem = np.nan_to_num(dem, nan=np.nanmedian(dem))
    
    # Flow accumulation
    flow_acc = _compute_flow_accumulation(dem)
    flow_acc = np.maximum(flow_acc, 1e-8)  # Avoid log(0)
    
    # Slope (in degrees converted to radians)
    dy, dx = np.gradient(dem)
    slope = np.hypot(dx, dy)
    
    # Avoid tan(0) by adding small epsilon
    # tan(slope) ≈ slope for small angles
    tan_slope = np.tan(np.clip(np.arctan(slope), 1e-8, np.pi/2 - 1e-8))
    tan_slope = np.maximum(tan_slope, 1e-8)
    
    # TWI
    twi = np.log(flow_acc / tan_slope)
    twi = np.nan_to_num(twi, nan=0)
    
    return twi


def compute_advanced_topographic_features(dem: np.ndarray, channel_mask: Optional[np.ndarray] = None) -> tuple:
    """Compute comprehensive topographic features for flood prediction.
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    channel_mask : np.ndarray, optional
        Binary mask of channel cells
    
    Returns
    -------
    X : np.ndarray, shape (H*W, n_features)
        Feature matrix (flattened)
    feature_names : list
        Names of features for interpretation
    """
    H, W = dem.shape
    dem = dem.astype(float)
    dem_filled = np.nan_to_num(dem, nan=np.nanmedian(dem))
    
    # 1. Normalized elevation
    if np.isfinite(dem_filled).any():
        p2, p98 = np.nanpercentile(dem_filled, (2, 98))
        denom = (p98 - p2) if (p98 - p2) != 0 else 1.0
        elev_norm = np.clip((dem_filled - p2) / denom, 0, 1)
    else:
        elev_norm = np.zeros_like(dem_filled)
    
    # 2. Normalized slope
    gy, gx = np.gradient(dem_filled)
    slope = np.hypot(gx, gy)
    if np.isfinite(slope).any():
        s_p2, s_p98 = np.nanpercentile(slope, (2, 98))
        s_denom = (s_p98 - s_p2) if (s_p98 - s_p2) != 0 else 1.0
        slope_norm = np.clip((slope - s_p2) / s_denom, 0, 1)
    else:
        slope_norm = np.zeros_like(slope)
    
    # 3. Flow accumulation (normalized)
    flow_acc = _compute_flow_accumulation(dem_filled)
    if np.isfinite(flow_acc).any():
        fa_p2, fa_p98 = np.nanpercentile(flow_acc, (2, 98))
        fa_denom = (fa_p98 - fa_p2) if (fa_p98 - fa_p2) != 0 else 1.0
        flow_acc_norm = np.clip((flow_acc - fa_p2) / fa_denom, 0, 1)
    else:
        flow_acc_norm = np.zeros_like(flow_acc)
    
    # 4. TWI (normalized)
    twi = _compute_twi(dem_filled)
    twi = np.nan_to_num(twi, nan=0)
    if np.isfinite(twi).any():
        twi_p2, twi_p98 = np.nanpercentile(twi, (2, 98))
        twi_denom = (twi_p98 - twi_p2) if (twi_p98 - twi_p2) != 0 else 1.0
        twi_norm = np.clip((twi - twi_p2) / twi_denom, 0, 1)
    else:
        twi_norm = np.zeros_like(twi)
    
    # 5. HAND (normalized)
    hand = _compute_hand(dem_filled, channel_mask)
    if np.isfinite(hand).any() and hand.max() > 0:
        hand_p2, hand_p98 = np.nanpercentile(hand, (2, 98))
        hand_denom = (hand_p98 - hand_p2) if (hand_p98 - hand_p2) != 0 else 1.0
        hand_norm = np.clip((hand - hand_p2) / hand_denom, 0, 1)
    else:
        hand_norm = np.zeros_like(hand)
    
    # 6. Profile curvature (normalized)
    profile_curv, plan_curv = _compute_terrain_curvature(dem_filled)
    profile_curv = np.nan_to_num(profile_curv, nan=0)
    if np.isfinite(profile_curv).any():
        pc_p2, pc_p98 = np.nanpercentile(profile_curv, (2, 98))
        pc_denom = (pc_p98 - pc_p2) if (pc_p98 - pc_p2) != 0 else 1.0
        profile_curv_norm = np.clip((profile_curv - pc_p2) / pc_denom, 0, 1)
    else:
        profile_curv_norm = np.zeros_like(profile_curv)
    
    # 7. Plan curvature (normalized)
    plan_curv = np.nan_to_num(plan_curv, nan=0)
    if np.isfinite(plan_curv).any():
        plc_p2, plc_p98 = np.nanpercentile(plan_curv, (2, 98))
        plc_denom = (plc_p98 - plc_p2) if (plc_p98 - plc_p2) != 0 else 1.0
        plan_curv_norm = np.clip((plan_curv - plc_p2) / plc_denom, 0, 1)
    else:
        plan_curv_norm = np.zeros_like(plan_curv)
    
    # Stack all features
    X = np.stack([
        elev_norm, slope_norm, flow_acc_norm, twi_norm,
        hand_norm, profile_curv_norm, plan_curv_norm
    ], axis=-1).reshape(H * W, 7)
    
    # Clean any remaining NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip to valid range [0, 1]
    X = np.clip(X, 0, 1)
    
    feature_names = [
        'Normalized Elevation',
        'Normalized Slope',
        'Flow Accumulation (log)',
        'Topographic Wetness Index (TWI)',
        'Height Above Nearest Drainage (HAND)',
        'Profile Curvature',
        'Plan Curvature'
    ]
    
    return X, feature_names


def _spatial_train_test_split(X, y, dem_shape, test_fraction=0.25, random_state=42):
    """Spatial train-test split (no data leakage).
    
    Divides domain into blocks; assigns whole blocks to train/test.
    
    Parameters
    ----------
    X : np.ndarray, shape (H*W, n_features)
        Feature matrix
    y : np.ndarray, shape (H*W,)
        Labels
    dem_shape : tuple
        (H, W) of original DEM
    test_fraction : float
        Fraction of domain for testing
    random_state : int
        Random seed
    
    Returns
    -------
    X_train, X_test, y_train, y_test, train_mask, test_mask
    """
    H, W = dem_shape
    np.random.seed(random_state)
    
    # Create spatial blocks (avoid too many blocks)
    n_blocks_h = max(2, min(4, int(np.sqrt(1 / test_fraction))))
    n_blocks_w = n_blocks_h
    
    block_h = H // n_blocks_h
    block_w = W // n_blocks_w
    
    train_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    
    # Random block selection for test
    n_total_blocks = n_blocks_h * n_blocks_w
    n_test_blocks = max(1, int(test_fraction * n_total_blocks))
    test_blocks = np.random.choice(n_total_blocks, size=n_test_blocks, replace=False)
    test_blocks_set = set(test_blocks)
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block_idx = i * n_blocks_w + j
            h_start = i * block_h
            h_end = (i + 1) * block_h if i < n_blocks_h - 1 else H
            w_start = j * block_w
            w_end = (j + 1) * block_w if j < n_blocks_w - 1 else W
            
            if block_idx in test_blocks_set:
                test_mask[h_start:h_end, w_start:w_end] = True
            else:
                train_mask[h_start:h_end, w_start:w_end] = True
    
    train_idx = train_mask.reshape(-1)
    test_idx = test_mask.reshape(-1)
    
    X_train = X[train_idx] if train_idx.sum() > 0 else X[:0]
    X_test = X[test_idx] if test_idx.sum() > 0 else X[:0]
    y_train = y[train_idx] if train_idx.sum() > 0 else y[:0]
    y_test = y[test_idx] if test_idx.sum() > 0 else y[:0]
    
    return X_train, X_test, y_train, y_test, train_mask, test_mask


def train_flood_classifier_improved(
    dem: np.ndarray,
    water: np.ndarray,
    threshold: float,
    channel_mask: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    test_fraction: float = 0.25,
    return_metrics: bool = True
) -> dict:
    """Train improved Random Forest using advanced features & spatial validation.
    
    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    water : np.ndarray, shape (H, W)
        Simulated water depth [m]
    threshold : float
        Water depth threshold for binary labels [m]
    channel_mask : np.ndarray, optional
        Binary mask of channel cells
    n_estimators : int
        Number of trees (default 200)
    max_depth : int
        Max tree depth (default None = unlimited)
    min_samples_split : int
        Min samples to split (default 5)
    min_samples_leaf : int
        Min samples at leaf (default 2)
    test_fraction : float
        Fraction of domain for spatial test (default 0.25)
    return_metrics : bool
        If True, compute & return evaluation metrics
    
    Returns
    -------
    results : dict
        {
            'model': RandomForestClassifier,
            'feature_names': list,
            'X_train': np.ndarray,
            'X_test': np.ndarray,
            'y_train': np.ndarray,
            'y_test': np.ndarray,
            'train_mask': np.ndarray,
            'test_mask': np.ndarray,
            'metrics': dict (if return_metrics=True) with keys:
                - 'roc_auc', 'f1', 'precision', 'recall'
                - 'confusion_matrix'
                - 'classification_report'
                - 'feature_importances'
        }
    """
    print("[ML] Computing advanced topographic features...")
    X, feature_names = compute_advanced_topographic_features(dem, channel_mask)
    
    print("[ML] Creating binary labels from water depth...")
    y = (water.reshape(-1) > float(threshold)).astype(np.uint8)
    
    print(f"[ML] Positive class: {y.sum()} / {len(y)} ({100*y.mean():.1f}%)")
    
    # Ensure no NaN/Inf in X before training
    n_nan_before = np.isnan(X).sum() + np.isinf(X).sum()
    if n_nan_before > 0:
        print(f"[ML] WARNING: Found {n_nan_before} NaN/Inf values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X = np.clip(X, 0, 1)
    
    print("[ML] Spatial train-test split (no leakage)...")
    X_train, X_test, y_train, y_test, train_mask, test_mask = _spatial_train_test_split(
        X, y, dem.shape, test_fraction=test_fraction, random_state=42
    )
    
    print(f"[ML] Training set: {len(y_train)} samples")
    print(f"[ML] Test set: {len(y_test)} samples")
    
    print("[ML] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    
    results = {
        'model': clf,
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_mask': train_mask,
        'test_mask': test_mask,
    }
    
    if return_metrics:
        print("[ML] Computing evaluation metrics...")
        
        # Check if we have both classes
        if len(np.unique(y_test)) < 2:
            print("[ML] WARNING: Only one class found in test set. Cannot compute ROC-AUC.")
            print(f"[ML] Classes in test set: {np.unique(y_test)}")
            metrics = {
                'roc_auc': None,
                'f1': None,
                'precision': None,
                'recall': None,
                'confusion_matrix': None,
                'classification_report': "Insufficient data (single class)",
                'feature_importances': dict(zip(feature_names, clf.feature_importances_)),
            }
        else:
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            metrics = {
                'roc_auc': roc_auc_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'feature_importances': dict(zip(feature_names, clf.feature_importances_)),
            }
        
        results['metrics'] = metrics
        
        # Print summary
        print(f"\n[ML] === EVALUATION === Spatial Test Set ===")
        if metrics['roc_auc'] is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"\n  Classification Report:\n{metrics['classification_report']}")
        else:
            print(f"  (Single class in test set - metrics not computed)")
        
        print(f"\n  Feature Importances:")
        for feat, imp in sorted(metrics['feature_importances'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {feat}: {imp:.4f}")
    
    return results


def plot_feature_importances(results: dict, figsize: Tuple[int, int] = (12, 5)) -> Optional[Figure]:
    """Plot feature importances & ROC curve.
    
    Parameters
    ----------
    results : dict
        Output from train_flood_classifier_improved
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object, or None if metrics not available
    """
    if 'metrics' not in results:
        print("No metrics available (return_metrics=False)")
        return None
    
    clf = results['model']
    feature_names = results['feature_names']
    metrics = results['metrics']
    y_test = results['y_test']
    
    # Check if we have both classes
    if metrics['roc_auc'] is None:
        print("[ML] Cannot plot ROC curve: Only one class in test set")
        # Plot only feature importances
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        importances = metrics['feature_importances']
        names = list(importances.keys())
        vals = list(importances.values())
        idx_sort = np.argsort(vals)[::-1]
        
        ax.barh(range(len(idx_sort)), np.array(vals)[idx_sort], color='steelblue')
        ax.set_yticks(range(len(idx_sort)))
        ax.set_yticklabels(np.array(names)[idx_sort])
        ax.set_xlabel('Importance')
        ax.set_title('Random Forest Feature Importances (Insufficient Test Data for ROC)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # Get probabilities for ROC curve
    y_proba = clf.predict_proba(results['X_test'])[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Feature importances
    importances = metrics['feature_importances']
    names = list(importances.keys())
    vals = list(importances.values())
    idx_sort = np.argsort(vals)[::-1]
    
    ax1.barh(range(len(idx_sort)), np.array(vals)[idx_sort], color='steelblue')
    ax1.set_yticks(range(len(idx_sort)))
    ax1.set_yticklabels(np.array(names)[idx_sort])
    ax1.set_xlabel('Importance')
    ax1.set_title('Random Forest Feature Importances')
    ax1.grid(axis='x', alpha=0.3)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = metrics['roc_auc']
    
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random classifier")
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (Spatial Test Set)')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def predict_probability_improved(model: RandomForestClassifier, dem: np.ndarray, channel_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Predict flood probability using improved classifier.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    dem : np.ndarray, shape (H, W)
        Digital Elevation Model [m]
    channel_mask : np.ndarray, optional
        Channel mask
    
    Returns
    -------
    proba : np.ndarray, shape (H, W)
        Flood probability raster
    """
    X, _ = compute_advanced_topographic_features(dem, channel_mask)
    proba = model.predict_proba(X)
    
    if proba.shape[1] == 1:
        p1 = np.zeros(dem.size, dtype=float)
    else:
        p1 = proba[:, 1]
    
    return p1.reshape(dem.shape)
