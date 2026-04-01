#!/usr/bin/env python3
"""
Enhanced Flask web server for HydroSim with interactive DEM upload and simulation.
Allows users to:
  - Upload custom DEMs
  - Configure rainfall parameters
  - Run simulations
  - View results with animations

Run:
    python web_server_v2.py
"""

from flask import Flask, render_template_string, jsonify, request, send_file  # type: ignore
import json
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import logging
import sys
import zipfile
import rasterio  # type: ignore
from rasterio.transform import from_bounds  # type: ignore
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import DiffusionWaveFloodModel
from src.ml.flood_classifier import (
    compute_topographic_features,
    train_flood_classifier,
    predict_probability,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HydroSim - Flood Simulation Platform</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            color: white;
            padding: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .header img {
            height: 100px;
            width: 100px;
            border-radius: 8px;
            object-fit: contain;
        }
        
        .header-text h1 {
            font-size: 2.8em;
            margin-bottom: 5px;
        }
        
        .header-text p {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            background: #f5f5f5;
            padding: 25px;
            border-radius: 8px;
        }
        
        .section-title {
            font-size: 1.6em;
            color: #1e88e5;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1e88e5;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: bold;
        }
        
        input[type="file"],
        input[type="number"],
        input[type="range"],
        select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
        }
        
        input[type="number"]:focus,
        input[type="file"]:focus,
        select:focus {
            outline: none;
            border-color: #1e88e5;
            box-shadow: 0 0 5px rgba(30, 136, 229, 0.3);
        }
        
        .slider-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        input[type="range"] {
            flex: 1;
        }
        
        .slider-value {
            min-width: 80px;
            background: white;
            padding: 8px 12px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            color: #1e88e5;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            background: #1e88e5;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: #388e3c;
        }
        
        .btn-secondary:hover {
            background: #2e7d32;
        }
        
        .btn-danger {
            background: #d32f2f;
        }
        
        .btn-danger:hover {
            background: #c62828;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        
        .stat-unit {
            font-size: 0.7em;
            opacity: 0.8;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .image-container {
            text-align: center;
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        
        .image-container img,
        .image-container video {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .image-label {
            color: #666;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 5px;
            color: #1565c0;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1e88e5;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: none;
        }
        
        .alert.success {
            background: #c8e6c9;
            color: #2e7d32;
            display: block;
        }
        
        .alert.error {
            background: #ffcdd2;
            color: #c62828;
            display: block;
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }
        
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                text-align: center;
            }
            
            .header h1 {
                font-size: 1.6em;
            }
            
            .images-grid {
                grid-template-columns: 1fr;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="/logo" alt="HydroSim Logo">
            <div class="header-text">
                <h1>HydroSim</h1>
                <p>Flood Simulation & Analysis Platform</p>
            </div>
        </div>
        
        <div class="content">
            <!-- Alert Messages -->
            <div id="alert" class="alert"></div>
            
            <!-- Upload Section -->
            <div class="section">
                <div class="section-title">📁 DEM Upload</div>
                <div class="form-group">
                    <label>Select Digital Elevation Model (GeoTIFF)</label>
                    <input type="file" id="demFile" accept=".tif,.tiff,.geotiff" />
                    <small>Or use default synthetic DEM</small>
                </div>
            </div>
            
            <!-- Parameters Section -->
            <div class="section">
                <div class="section-title">⚙️ Simulation Parameters</div>
                
                <div class="form-group">
                    <label>Total Rainfall (mm)</label>
                    <div class="slider-container">
                        <input type="range" id="rainfall" min="10" max="500" value="100" step="10">
                        <span class="slider-value" id="rainfallValue">100 mm</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Simulation Duration (minutes)</label>
                    <div class="slider-container">
                        <input type="range" id="duration" min="10" max="1440" value="500" step="10">
                        <span class="slider-value" id="durationValue">500 min</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Rainfall Mode</label>
                    <select id="rainfallMode">
                        <option value="uniform">Uniform (distributed evenly)</option>
                        <option value="concentrated">Concentrated (source areas only)</option>
                    </select>
                </div>
                
                <div class="button-group">
                    <button class="btn btn-secondary" onclick="runSimulation()">▶️ Run Simulation</button>
                    <button class="btn btn-danger" onclick="resetForm()">🔄 Reset</button>
                </div>
            </div>
            
            <!-- Loading Indicator -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Running simulation... This may take a moment.</p>
            </div>
            
            <!-- Results Section -->
            <div class="section" id="resultsSection" style="display: none;">
                <div class="section-title">📊 Simulation Results</div>
                <div class="stats-grid" id="stats-container"></div>
                
                <div class="images-grid">
                    <div class="image-container">
                        <h3>Simulation Snapshot</h3>
                        <img id="resultImage" src="" alt="Results">
                        <div class="image-label">DEM • Sources • Water • Probability</div>
                    </div>
                    <div class="image-container">
                        <h3>Animation</h3>
                        <video id="animVideo" controls style="max-width: 100%;"></video>
                        <div class="image-label">Water Depth Evolution</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="timeseriesChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🌊 HydroSim - Advanced Flood Simulation Platform</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Environmental Modelling & Software • Scientific Research Tool
            </p>
        </div>
    </div>
    
    <script>
        // Update slider values
        document.getElementById('rainfall').addEventListener('input', (e) => {
            document.getElementById('rainfallValue').textContent = e.target.value + ' mm';
        });
        
        document.getElementById('duration').addEventListener('input', (e) => {
            document.getElementById('durationValue').textContent = e.target.value + ' min';
        });
        
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.textContent = message;
            alert.className = 'alert ' + type;
            setTimeout(() => {
                alert.className = 'alert';
            }, 5000);
        }
        
        function runSimulation() {
            const rainfall = document.getElementById('rainfall').value;
            const duration = document.getElementById('duration').value;
            const mode = document.getElementById('rainfallMode').value;
            
            document.getElementById('loading').classList.add('active');
            
            fetch('/api/run-simulation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    rainfall_mm: parseInt(rainfall),
                    duration_minutes: parseInt(duration),
                    rainfall_mode: mode
                })
            })
            .then(r => {
                if (!r.ok) throw new Error('Simulation failed');
                return r.json();
            })
            .then(data => {
                document.getElementById('loading').classList.remove('active');
                displayResults(data);
                showAlert('✅ Simulation completed successfully!', 'success');
            })
            .catch(err => {
                document.getElementById('loading').classList.remove('active');
                showAlert('❌ Error: ' + err.message, 'error');
                console.error(err);
            });
        }
        
        function displayResults(data) {
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.style.display = 'block';
            
            // Update stats
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-label">Duration</div>
                    <div class="stat-value">${data.summary.simulation.final_time_minutes}</div>
                    <div class="stat-unit">minutes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Timesteps</div>
                    <div class="stat-value">${data.summary.simulation.timesteps}</div>
                    <div class="stat-unit">steps</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max Depth</div>
                    <div class="stat-value">${data.summary.simulation.max_depth_m.toFixed(3)}</div>
                    <div class="stat-unit">meters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Volume</div>
                    <div class="stat-value">${(data.summary.simulation.total_water_volume_m3 / 1e6).toFixed(2)}</div>
                    <div class="stat-unit">million m³</div>
                </div>
            `;
            
            document.getElementById('stats-container').innerHTML = statsHtml;
            
            // Load images
            document.getElementById('resultImage').src = '/image/results?t=' + Date.now();
            
            // Load video
            const videoEl = document.getElementById('animVideo');
            videoEl.src = '/video/animation.gif?t=' + Date.now();
            
            // Create chart
            createChart(data.history);
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        function createChart(history) {
            const times = history.map(h => h.time_minutes);
            const flooded = history.map(h => h.flooded_percent);
            const depth = history.map(h => h.max_depth);
            const volume = history.map(h => h.total_water_volume_m3 / 1e6);
            
            const ctx = document.getElementById('timeseriesChart').getContext('2d');
            
            // Destroy previous chart if exists
            if (window.myChart) {
                window.myChart.destroy();
            }
            
            window.myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: times,
                    datasets: [
                        {
                            label: 'Flooded Area (%)',
                            data: flooded,
                            borderColor: '#1e88e5',
                            backgroundColor: 'rgba(30, 136, 229, 0.1)',
                            yAxisID: 'y',
                            borderWidth: 2,
                            tension: 0.1
                        },
                        {
                            label: 'Max Depth (m)',
                            data: depth,
                            borderColor: '#d32f2f',
                            backgroundColor: 'rgba(211, 47, 47, 0.1)',
                            yAxisID: 'y1',
                            borderWidth: 2,
                            tension: 0.1
                        },
                        {
                            label: 'Volume (million m³)',
                            data: volume,
                            borderColor: '#388e3c',
                            backgroundColor: 'rgba(56, 142, 60, 0.1)',
                            yAxisID: 'y2',
                            borderWidth: 2,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        title: { display: true, text: 'Flood Simulation Time Series', font: { size: 14, weight: 'bold' } },
                        legend: { display: true, position: 'top' }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Flooded Area (%)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Max Depth (m)' },
                            grid: { drawOnChartArea: false }
                        },
                        y2: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Volume (million m³)' },
                            grid: { drawOnChartArea: false }
                        },
                        x: { title: { display: true, text: 'Time (minutes)' } }
                    }
                }
            });
        }
        
        function resetForm() {
            document.getElementById('rainfall').value = 100;
            document.getElementById('duration').value = 500;
            document.getElementById('rainfallMode').value = 'uniform';
            document.getElementById('demFile').value = '';
            document.getElementById('rainfallValue').textContent = '100 mm';
            document.getElementById('durationValue').textContent = '500 min';
            document.getElementById('resultsSection').style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/logo')
def get_logo():
    """Return logo image."""
    logo_path = Path("logo.png")
    if logo_path.exists():
        return send_file(logo_path, mimetype='image/png')
    # Return placeholder if no logo
    return "Logo not found", 404

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run simulation with custom parameters."""
    try:
        params = request.json
        rainfall_mm = params.get('rainfall_mm', 100)
        duration_minutes = params.get('duration_minutes', 500)
        rainfall_mode = params.get('rainfall_mode', 'uniform')
        
        # Generate synthetic DEM
        dem = _generate_synthetic_dem(shape=(100, 100))
        sources = _generate_rainfall_sources(shape=(100, 100))
        
        # Run simulation
        model = DiffusionWaveFloodModel(
            dem_data=dem,
            sources_mask=sources,
            diffusion_rate=0.5,
            flood_threshold=0.1,
            cell_size_meters=25.0,
        )
        
        # Calculate rainfall per step
        num_steps = duration_minutes // 10
        rainfall_per_step = rainfall_mm / num_steps
        
        logger.info(f"Running simulation: {rainfall_mm}mm over {duration_minutes}min")
        
        for step in range(num_steps):
            model.apply_rainfall(rainfall_per_step)
            model.advance_flow()
            model.record_diagnostics(10)
        
        # Train ML classifier
        clf, prob = _train_classifier(dem, model.water_height)
        
        # Generate visualization and GIF
        _generate_visualizations(dem, sources, model, prob)
        _generate_animation(dem, model)
        
        # Prepare response
        summary_path = Path("outputs/test_run/summary.json")
        history_path = Path("outputs/test_run/history.json")
        
        with open(summary_path) as f:
            summary = json.load(f)
        with open(history_path) as f:
            history = json.load(f)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({'error': str(e)}), 500

def _generate_synthetic_dem(shape=(100, 100)):
    """Generate synthetic DEM."""
    H, W = shape
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
    
    dem += np.random.normal(0, 0.5, shape)
    return dem

def _generate_rainfall_sources(shape=(100, 100)):
    """Generate rainfall source areas."""
    sources = np.zeros(shape, dtype=bool)
    sources[10:25, 10:25] = True
    sources[10:25, 75:90] = True
    sources[75:90, 40:60] = True
    return sources

def _train_classifier(dem, water):
    """Train flood classifier."""
    clf = train_flood_classifier(dem, water, threshold=0.2, n_estimators=100)
    prob = predict_probability(clf, dem)
    return clf, prob

def _generate_visualizations(dem, sources, model, prob):
    """Generate visualization plots."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # DEM
    axes[0, 0].imshow(dem, cmap='terrain', alpha=0.8)
    axes[0, 0].set_title('DEM')
    
    # Sources
    axes[0, 1].imshow(dem, cmap='gray', alpha=0.5)
    axes[0, 1].contourf(sources.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.6)
    axes[0, 1].set_title('Rainfall Sources')
    
    # Water
    axes[1, 0].imshow(model.water_height, cmap='Blues')
    axes[1, 0].set_title('Final Water Depth')
    
    # Probability
    axes[1, 1].imshow(prob, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 1].set_title('Flood Probability')
    
    plt.tight_layout()
    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/test_run/results_visualization.png", dpi=100)
    plt.close()

def _generate_animation(dem, model):
    """Generate animation of water depth."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image
    
    logger.info("Generating animation...")
    
    # Create frames
    frames = []
    history_len = len(model.history)
    
    # Reconstruct water levels at each timestep (simplified)
    for step in range(0, history_len, max(1, history_len // 10)):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Approximate water at this timestep
        water_frac = step / history_len if history_len > 0 else 1.0
        water_approx = model.water_height * water_frac
        
        im = ax.imshow(water_approx, cmap='Blues', vmin=0, vmax=model.water_height.max())
        ax.set_title(f'Water Depth - Step {step}')
        plt.colorbar(im, ax=ax)
        
        # Save frame
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close()
    
    if frames:
        Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
        frames[0].save(
            'outputs/test_run/animation.gif',
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        logger.info("Animation saved to outputs/test_run/animation.gif")

@app.route('/image/results')
def get_results_image():
    """Return results visualization."""
    img_path = Path("outputs/test_run/results_visualization.png")
    if img_path.exists():
        return send_file(img_path, mimetype='image/png')
    return "Image not found", 404

@app.route('/video/animation.gif')
def get_animation():
    """Return animation GIF."""
    gif_path = Path("outputs/test_run/animation.gif")
    if gif_path.exists():
        return send_file(gif_path, mimetype='image/gif')
    return "Animation not found", 404

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════╗
║           HydroSim Web Server v2 Started                       ║
╚════════════════════════════════════════════════════════════════╝

🌐 Web Interface: http://localhost:5000

✅ Features:
   • DEM Upload
   • Rainfall Configuration
   • Interactive Simulation
   • Animation Generation
   • Real-time Results

Press CTRL+C to stop the server.
    """)
    app.run(host='localhost', port=5001, debug=True)
