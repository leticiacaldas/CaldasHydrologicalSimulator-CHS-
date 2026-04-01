#!/usr/bin/env python3
"""
Results visualization script for HydroSim-RF test outputs.
Loads and displays simulation and ML results.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_results(results_dir="outputs/test_run"):
    """Load test results."""
    results_path = Path(results_dir)
    
    # Load NPZ
    data = np.load(results_path / "results.npz")
    dem = data['dem']
    sources = data['sources']
    water_final = data['water_final']
    probability = data['probability']
    
    # Load JSON
    with open(results_path / "summary.json") as f:
        summary = json.load(f)
    
    with open(results_path / "history.json") as f:
        history = json.load(f)
    
    return dem, sources, water_final, probability, summary, history

def plot_results(dem, sources, water_final, probability, summary):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('HydroSim-RF Test Results', fontsize=16, fontweight='bold')
    
    # Plot 1: DEM
    ax = axes[0, 0]
    im1 = ax.imshow(dem, cmap='terrain', alpha=0.8)
    ax.contour(dem, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_title('Digital Elevation Model (DEM)', fontweight='bold')
    ax.set_xlabel('X (cells)')
    ax.set_ylabel('Y (cells)')
    plt.colorbar(im1, ax=ax, label='Elevation (m)')
    
    # Plot 2: Rainfall Sources
    ax = axes[0, 1]
    ax.imshow(dem, cmap='gray', alpha=0.5)
    ax.contourf(sources.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.6)
    ax.set_title('Rainfall Source Areas', fontweight='bold')
    ax.set_xlabel('X (cells)')
    ax.set_ylabel('Y (cells)')
    
    # Plot 3: Final Water Depth
    ax = axes[1, 0]
    im3 = ax.imshow(water_final, cmap='Blues', interpolation='nearest')
    ax.set_title(f'Final Water Depth (max={water_final.max():.3f} m)', fontweight='bold')
    ax.set_xlabel('X (cells)')
    ax.set_ylabel('Y (cells)')
    cbar3 = plt.colorbar(im3, ax=ax, label='Depth (m)')
    
    # Plot 4: Flood Probability
    ax = axes[1, 1]
    im4 = ax.imshow(probability, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title(f'RF Flood Probability (mean={probability.mean():.3f})', fontweight='bold')
    ax.set_xlabel('X (cells)')
    ax.set_ylabel('Y (cells)')
    cbar4 = plt.colorbar(im4, ax=ax, label='Probability')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("outputs/test_run/results_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    return fig

def plot_timeseries(history):
    """Plot simulation time series."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('HydroSim-RF Simulation Time Series', fontsize=16, fontweight='bold')
    
    times = [h['time_minutes'] for h in history]
    flooded_pct = [h['flooded_percent'] for h in history]
    active_cells = [h['active_cells'] for h in history]
    max_depth = [h['max_depth'] for h in history]
    volume = [h['total_water_volume_m3'] for h in history]
    
    # Flooded percentage
    axes[0, 0].plot(times, flooded_pct, 'b-', linewidth=2)
    axes[0, 0].set_title('Flooded Area %', fontweight='bold')
    axes[0, 0].set_ylabel('Percentage (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Active cells
    axes[0, 1].plot(times, active_cells, 'g-', linewidth=2)
    axes[0, 1].set_title('Active Cells', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Cells')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max depth
    axes[1, 0].plot(times, max_depth, 'r-', linewidth=2)
    axes[1, 0].set_title('Maximum Water Depth', fontweight='bold')
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Depth (m)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Volume
    axes[1, 1].plot(times, volume, 'm-', linewidth=2)
    axes[1, 1].set_title('Total Water Volume', fontweight='bold')
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Volume (m³)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("outputs/test_run/timeseries.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved time series to {output_path}")
    
    return fig

def main():
    print("Loading test results...")
    dem, sources, water_final, probability, summary, history = load_results()
    
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"Final time: {summary['simulation']['final_time_minutes']} minutes")
    print(f"Total timesteps: {summary['simulation']['timesteps']}")
    print(f"Final water volume: {summary['simulation']['total_water_volume_m3']:.1f} m³")
    print(f"Maximum water depth: {summary['simulation']['max_depth_m']:.3f} m")
    print(f"Flooded cells: {summary['simulation']['flooded_cells']}")
    print()
    print("PROBABILITY SUMMARY")
    print("="*70)
    print(f"Mean probability: {summary['probability']['mean_p']:.3f}")
    print(f"Std probability: {summary['probability']['std_p']:.3f}")
    print(f"High-risk cells (p>0.7): {summary['probability']['high_risk_cells']}")
    print()
    
    print("Generating visualizations...")
    plot_results(dem, sources, water_final, probability, summary)
    plot_timeseries(history)
    
    print("\n✅ Visualization complete!")
    print("   • results_visualization.png")
    print("   • timeseries.png")

if __name__ == "__main__":
    main()
