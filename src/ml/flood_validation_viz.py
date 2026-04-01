"""
Visualization utilities for flood validation and model comparison.
Generates publication-ready figures comparing real, simulated, and predicted floods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_comparison_maps(real_flood: np.ndarray,
                        simulated_flood: np.ndarray,
                        predicted_flood: np.ndarray,
                        dem: Optional[np.ndarray] = None,
                        output_path: Optional[Path] = None,
                        title: str = "Flood Map Comparison: Real vs Simulated vs Predicted") -> Path:
    """
    Create 4-panel comparison of flood maps.
    
    Parameters
    ----------
    real_flood : np.ndarray
        Real flood map (HxW, 0/1)
    simulated_flood : np.ndarray
        Simulated flood map (HxW, 0/1)
    predicted_flood : np.ndarray
        Predicted flood map (HxW, 0-1 probabilities)
    dem : np.ndarray, optional
        DEM for background
    output_path : Path, optional
        Save location. If None, doesn't save.
    title : str
        Figure title
        
    Returns
    -------
    Path
        Output file path
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    
    # Normalize predicted to 0-1
    pred_norm = np.clip(predicted_flood, 0, 1)
    
    # Plot 1: Real flood
    im0 = axes[0, 0].imshow(real_flood, cmap='Blues', vmin=0, vmax=1)
    axes[0, 0].set_title('(A) Real Flood Extent\n(Observed/INMET)', fontweight='bold')
    axes[0, 0].set_ylabel('North-South (cells)')
    plt.colorbar(im0, ax=axes[0, 0], label='Flooded (1/0)')
    
    # Plot 2: Simulated flood
    im1 = axes[0, 1].imshow(simulated_flood, cmap='Purples', vmin=0, vmax=1)
    axes[0, 1].set_title('(B) Simulated Flood Extent\n(Physics-based Model)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], label='Flooded (1/0)')
    
    # Plot 3: Predicted flood
    im2 = axes[1, 0].imshow(pred_norm, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('(C) ML Predicted Flood Probability\n(Random Forest)', fontweight='bold')
    axes[1, 0].set_ylabel('North-South (cells)')
    axes[1, 0].set_xlabel('East-West (cells)')
    plt.colorbar(im2, ax=axes[1, 0], label='Probability (0-1)')
    
    # Plot 4: Difference/Error map
    sim_binary = (simulated_flood > 0.5).astype(int)
    pred_binary = (pred_norm > 0.5).astype(int)
    
    # Create difference map: 0=correct, 1=FP sim, 2=FN sim, 3=FP pred, 4=FN pred
    error_map = np.zeros_like(real_flood, dtype=int)
    
    # Correct predictions (both models)
    error_map[(sim_binary == real_flood) & (pred_binary == real_flood)] = 0
    
    # Simulation errors
    error_map[(sim_binary == 1) & (real_flood == 0)] = 1  # FP sim
    error_map[(sim_binary == 0) & (real_flood == 1)] = 2  # FN sim
    
    # Prediction errors
    error_map[(pred_binary == 1) & (real_flood == 0)] = 3  # FP pred
    error_map[(pred_binary == 0) & (real_flood == 1)] = 4  # FN pred
    
    im3 = axes[1, 1].imshow(error_map, cmap='tab10', vmin=0, vmax=4)
    axes[1, 1].set_title('(D) Error Classification\n(Compared to Real)', fontweight='bold')
    axes[1, 1].set_xlabel('East-West (cells)')
    
    # Custom colorbar for error map
    cbar = plt.colorbar(im3, ax=axes[1, 1], ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Correct', 'Sim FP', 'Sim FN', 'Pred FP', 'Pred FN'], fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Comparison map saved to {output_path}")
    
    return output_path or Path("comparison_maps.png")


def plot_metrics_comparison(metrics: dict,
                           output_path: Optional[Path] = None) -> Path:
    """
    Create bar chart comparing model metrics.
    
    Parameters
    ----------
    metrics : dict
        Metrics from compare_predictions()
    output_path : Path, optional
        Save location
        
    Returns
    -------
    Path
        Output file path
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Simulation', 'ML Prediction']
    accuracies = [metrics['simulation_accuracy'], metrics['prediction_accuracy']]
    
    bars1 = ax1.bar(models, accuracies, color=['#9C27B0', '#F44336'], alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Overall Accuracy vs Real Flood', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    f1_scores = [metrics['simulation_f1'], metrics['prediction_f1']]
    
    bars2 = ax2.bar(models, f1_scores, color=['#9C27B0', '#F44336'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('(B) F1-Score (Recall-Precision Balance)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Metrics chart saved to {output_path}")
    
    return output_path or Path("metrics_comparison.png")


def plot_confusion_matrices(real_flood: np.ndarray,
                           simulated_flood: np.ndarray,
                           predicted_flood: np.ndarray,
                           output_path: Optional[Path] = None) -> Path:
    """
    Create confusion matrix heatmaps for both models.
    
    Parameters
    ----------
    real_flood : np.ndarray
        Real flood map
    simulated_flood : np.ndarray
        Simulated flood map
    predicted_flood : np.ndarray
        Predicted flood map (probabilities)
    output_path : Path, optional
        Save location
        
    Returns
    -------
    Path
        Output file path
    """
    from sklearn.metrics import confusion_matrix  # type: ignore
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    real_flat = real_flood.flatten().astype(int)
    sim_flat = simulated_flood.flatten().astype(int)
    pred_flat = (predicted_flood.flatten() > 0.5).astype(int)
    
    # Simulation confusion matrix
    cm_sim = confusion_matrix(real_flat, sim_flat)
    
    im1 = ax1.imshow(cm_sim, cmap='Blues', aspect='auto')
    ax1.set_title('Simulation Model\nConfusion Matrix', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['No Flood', 'Flood'])
    ax1.set_yticklabels(['No Flood', 'Flood'])
    
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm_sim[i, j]), ha='center', va='center',
                    color='white' if cm_sim[i, j] > cm_sim.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1)
    
    # Prediction confusion matrix
    cm_pred = confusion_matrix(real_flat, pred_flat)
    
    im2 = ax2.imshow(cm_pred, cmap='Reds', aspect='auto')
    ax2.set_title('ML Prediction Model\nConfusion Matrix', fontweight='bold', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['No Flood', 'Flood'])
    ax2.set_yticklabels(['No Flood', 'Flood'])
    
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_pred[i, j]), ha='center', va='center',
                    color='white' if cm_pred[i, j] > cm_pred.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Confusion matrices saved to {output_path}")
    
    return output_path or Path("confusion_matrices.png")


def plot_feature_importance(feature_importance_df,
                           output_path: Optional[Path] = None,
                           top_n: int = 10) -> Path:
    """
    Create horizontal bar chart of feature importance.
    
    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    output_path : Path, optional
        Save location
    top_n : int
        Number of top features to plot
        
    Returns
    -------
    Path
        Output file path
    """
    
    top_features = feature_importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color='#1e88e5', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance\n(Random Forest)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.3f}",
               va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Feature importance plot saved to {output_path}")
    
    return output_path or Path("feature_importance.png")
