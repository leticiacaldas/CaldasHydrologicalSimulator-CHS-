#!/usr/bin/env python3
"""
Enhanced Flask web server for HydroSim with GeoTIFF export, animated GIF, and downloads.
"""

# Configure matplotlib to use non-GUI backend BEFORE importing anything else
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template_string, jsonify, request, send_file  # type: ignore
from werkzeug.exceptions import RequestEntityTooLarge  # type: ignore
import json
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import logging
import os
import sys
import tempfile
import zipfile
import rasterio  # type: ignore
from rasterio.transform import from_bounds  # type: ignore
from typing import Optional, Tuple, Any
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import DiffusionWaveFloodModel, NumpyDiffusionWaveEngine, GamaFloodModelNumpy
from src.ml.flood_classifier import (
    compute_topographic_features,
    train_flood_classifier,
    predict_probability,
)
from src.io.utilities import (
    ValidationError,
    CacheManager,
    validate_geotiff,
    validate_shapefile,
    validate_dem_values,
    EnhancedLogging,
    ensure_valid_crs,
    safe_divide,
)
from src.io.export_formats import (
    SimulationHistory,
    export_to_netcdf,
    export_to_hdf5,
    export_comparison_report,
)

app = Flask(__name__)
try:
    _MAX_UPLOAD_MB = int(os.environ.get('HYDROSIM_MAX_UPLOAD_MB', '4096'))
except Exception:
    _MAX_UPLOAD_MB = 4096
app.config['MAX_CONTENT_LENGTH'] = int(_MAX_UPLOAD_MB) * 1024 * 1024
# Werkzeug/Flask form parsing limits (helps avoid 413/400 for large multipart forms)
app.config['MAX_FORM_MEMORY_SIZE'] = app.config['MAX_CONTENT_LENGTH']
app.config['MAX_FORM_PARTS'] = int(os.environ.get('HYDROSIM_MAX_FORM_PARTS', '2000'))
_db_initialized = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
        """Return a helpful 413 error message with instructions to increase limits."""
        limit_bytes = app.config.get('MAX_CONTENT_LENGTH')
        limit_mb = (float(limit_bytes) / (1024.0 * 1024.0)) if limit_bytes else None
        message = (
                "Upload muito grande (413). "
                + (f"Limite atual: {limit_mb:.0f} MB. " if limit_mb is not None else "")
                + "Ajuste a variável de ambiente HYDROSIM_MAX_UPLOAD_MB e reinicie o servidor."
        )

        wants_json = (
                request.path.startswith('/api')
                or request.accept_mimetypes.best == 'application/json'
                or request.is_json
        )
        if wants_json:
                return jsonify({'success': False, 'error': message}), 413

        return (
                f"""<!doctype html>
<html lang='pt-BR'>
    <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>HydroSim - Upload grande</title>
        <style>
            body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }}
            .card {{ max-width: 880px; margin: 0 auto; border: 1px solid #ddd; border-radius: 12px; padding: 18px; }}
            code, pre {{ background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }}
            pre {{ padding: 12px; overflow: auto; }}
        </style>
    </head>
    <body>
        <div class='card'>
            <h2>Erro 413: upload excedeu o limite</h2>
            <p>{message}</p>
            <p>Exemplo para subir com 8192 MB (8 GB):</p>
            <pre>HYDROSIM_MAX_UPLOAD_MB=8192 HYDROSIM_PORT=8888 /home/leticia/Desktop/hydrosim/venv/bin/python web_server_v3.py</pre>
            <p><a href='/'>Voltar</a></p>
        </div>
    </body>
</html>""",
                413,
        )

@app.before_request
def init_database():
    """Initialize database on first request."""
    global _db_initialized
    if not _db_initialized:
        try:
            SimulationHistory.init_db()
            _db_initialized = True
            logger.info("✅ Simulation database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en-US">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HydroSim - Flood Simulation Platform</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
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
            background: linear-gradient(135deg, #061a40 0%, #082a63 55%, #0a3478 100%);
            color: white;
            padding: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            border-bottom: 3px solid rgba(255, 255, 255, 0.18);
            box-shadow: inset 0 -10px 24px rgba(0, 0, 0, 0.25);
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
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
        }
        
        .header-text p {
            font-size: 1em;
            opacity: 0.95;
            text-shadow: 0 1px 6px rgba(0, 0, 0, 0.35);
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
        
        input[type="range"], input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
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
        
        .btn-download {
            background: #f57c00;
        }
        
        .btn-download:hover {
            background: #e65100;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
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
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1e88e5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stat-card h3 {
            color: #1e88e5;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        .stat-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .image-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .image-container h3 {
            background: #1e88e5;
            color: white;
            padding: 10px;
            margin: 0;
        }
        
        .image-container img, .image-container video {
            width: 100%;
            height: 300px;
            object-fit: contain;
            object-position: center;
            background: #f4f7fb;
            display: block;
        }
        
        .image-label {
            padding: 10px;
            font-size: 0.9em;
            color: #666;
            border-top: 1px solid #eee;
        }
        
        .map-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        #map {
            height: 500px;
            width: 100%;
        }
        
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .download-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            border: 2px dashed #1e88e5;
        }
        
        .download-section h3 {
            color: #1e88e5;
            margin-bottom: 15px;
        }
        
        .download-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .download-item {
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
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
            <div id="alert" class="alert"></div>
            
            <!-- Input Section -->
            <div class="section">
                <div class="section-title">📥 Input Configuration</div>
                
                <div class="form-group">
                    <label>Digital Elevation Model (DEM)</label>
                    <input type="file" id="demFile" accept=".tif,.geotiff,.tiff">
                    <small>Or use default synthetic DEM (100x100m)</small>
                </div>

                <div class="form-group">
                    <label>Orthomosaic (Realistic Base)</label>
                    <input type="file" id="orthoFile" accept=".tif,.geotiff,.tiff,.png,.jpg,.jpeg">
                    <small>Optional: RGB image to use as background (GeoTIFF/PNG/JPG)</small>
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
                    <button class="btn" style="background: #757575;" onclick="resetForm()">↻ Reset</button>
                </div>
            </div>
            
            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Running simulation... Please wait.</p>
            </div>
            
            <!-- Results Section -->
            <div class="section" id="resultsSection" style="display: none;">
                <div class="section-title">📊 Simulation Results</div>
                
                <!-- Statistics -->
                <div class="stats-grid" id="stats-container"></div>
                
                <!-- Map -->
                <div class="map-container" id="mapContainer" style="display: none;">
                    <div id="map"></div>
                </div>
                
                <!-- Visualizations -->
                <div class="images-grid">
                    <div class="image-container" id="orthoContainer" style="display: none;">
                        <h3>Orthomosaic (Upload)</h3>
                        <img id="orthoImage" src="" alt="Orthomosaic" style="max-width: 100%;">
                        <div class="image-label">Realistic background used in GIF and maps</div>
                    </div>
                    <div class="image-container">
                        <h3>Digital Elevation Model (DEM)</h3>
                        <img id="demImage" src="" alt="DEM" style="max-width: 100%;">
                        <div class="image-label">Terrain Elevation Map</div>
                    </div>
                    <div class="image-container">
                        <h3>Preferential Water Flow</h3>
                        <img id="flowImage" src="" alt="Flow Direction" style="max-width: 100%;">
                        <div class="image-label">Flow Direction (D8 Algorithm)</div>
                    </div>
                    <div class="image-container">
                        <h3>Drainage Network (Accumulation)</h3>
                        <img id="flowAccImage" src="" alt="Flow Accumulation" style="max-width: 100%;">
                        <div class="image-label">Flow Accumulation (log scale)</div>
                    </div>
                    <div class="image-container">
                        <h3>Simulation Snapshot</h3>
                        <img id="resultImage" src="" alt="Results">
                        <div class="image-label">DEM • Sources • Water • Probability</div>
                    </div>
                    <div class="image-container">
                        <h3>Water Evolution (Animated)</h3>
                        <img id="animGif" src="" alt="Animation" style="max-width: 100%;">
                        <div class="image-label">GIF Animation - Water Depth Over Time</div>
                    </div>
                    <div class="image-container">
                        <h3>Peak Water Depth (Heatmap)</h3>
                        <img id="peakImage" src="" alt="Peak Water Depth" style="max-width: 100%;">
                        <div class="image-label">Maximum water depth intensity map - quick visualization of flow accumulation zones</div>
                    </div>
                </div>
                
                <!-- ML Validation Results -->
                <div class="section" id="mlResultsSection" style="display: none;">
                    <div class="section-title">🤖 Machine Learning Validation (Random Forest)</div>
                    
                    <div class="images-grid">
                        <div class="image-container">
                            <h3>ML Metrics & Feature Importance</h3>
                            <img id="mlMetricsImage" src="" alt="ML Metrics" style="max-width: 100%;">
                            <div class="image-label">ROC-AUC, F1-score, Precision, Recall, and Feature Importance Ranking</div>
                        </div>
                        <div class="image-container">
                            <h3>Flood Susceptibility Map</h3>
                            <img id="floodProbImage" src="" alt="Flood Probability" style="max-width: 100%;">
                            <div class="image-label">ML-predicted flood inundation probability (0-1 scale)</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <canvas id="mlMetricsChart"></canvas>
                    </div>
                    
                    <div class="download-section">
                        <h3>📥 ML Validation Downloads</h3>
                        <div class="download-grid">
                            <button class="btn btn-download" onclick="downloadMLMetrics()">📊 ML Metrics (JSON)</button>
                            <button class="btn btn-download" onclick="downloadFloodProbability()">🗺️ Flood Probability (GeoTIFF)</button>
                            <button class="btn btn-download" onclick="downloadMLValidationZip()">📦 ML Results (ZIP)</button>
                        </div>
                    </div>
                </div>
                
                <!-- Time Series Chart -->
                <div class="chart-container">
                    <canvas id="timeseriesChart"></canvas>
                </div>
                
                <!-- Downloads -->
                <div class="download-section">
                    <h3>📥 Download Results</h3>
                    <div class="download-grid">
                        <button class="btn btn-download" onclick="downloadGeoTIFF()">🗺️ DEM (GeoTIFF)</button>
                        <button class="btn btn-download" onclick="downloadOrthomosaic()">🛰️ Orthomosaic (GeoTIFF)</button>
                        <button class="btn btn-download" onclick="downloadWaterGeoTIFF()">💧 Water Depth (GeoTIFF Visual p/ QGIS)</button>
                        <button class="btn btn-download" onclick="downloadWaterGeoTIFFRaw()">💧 Water Depth RAW (float32)</button>
                        <button class="btn btn-download" onclick="downloadWaterRGBA()">💧 Water Depth (RGBA)</button>
                        <button class="btn btn-download" onclick="downloadWaterGifQGIS()">🎬🗺️ Lâmina do GIF p/ QGIS (GeoTIFF RGB)</button>
                        <button class="btn btn-download" onclick="downloadFloodGPKG()">🧱 Flood Extent (GPKG)</button>
                        <button class="btn btn-download" onclick="downloadFlowDirection()">🌊 Flow Direction (GeoTIFF)</button>
                        <button class="btn btn-download" onclick="downloadFlowAccumulation()">🕸️ Flow Accumulation (GeoTIFF)</button>
                        <button class="btn btn-download" onclick="downloadAnimGif()">🎬 Animated GIF</button>
                        <button class="btn btn-download" onclick="downloadAllData()">📦 All Data (ZIP)</button>
                    </div>
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
        let lastSimulationData = null;
        let timeseriesChartInstance = null;
        
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
            const demFile = document.getElementById('demFile').files[0];
            const orthoFile = document.getElementById('orthoFile').files[0];
            
            document.getElementById('loading').classList.add('active');

            // Use multipart to optionally send a DEM file
            const formData = new FormData();
            formData.append('rainfall_mm', parseInt(rainfall));
            formData.append('duration_minutes', parseInt(duration));
            formData.append('rainfall_mode', mode);
            if (demFile) formData.append('dem_file', demFile);
            if (orthoFile) formData.append('ortho_file', orthoFile);
            
            fetch('/api/run-simulation', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').classList.remove('active');
                
                if (!data.success) throw new Error(data.error);
                
                lastSimulationData = data;
                displayResults(data);
                showAlert('✅ Simulation completed successfully!', 'success');
            })
            .catch(e => {
                document.getElementById('loading').classList.remove('active');
                showAlert('❌ Error: ' + e.message, 'error');
                console.error(e);
            });
        }
        
        function displayResults(data) {
            const summary = data.summary;
            const history = data.history;

            // Garante que os elementos IMG estejam visíveis no DOM antes do carregamento
            document.getElementById('resultsSection').style.display = 'block';

            function setImageWithRetry(elementId, baseUrl, retries = 20, delayMs = 3000) {
                const img = document.getElementById(elementId);
                if (!img) return;

                img.loading = 'eager';
                img.decoding = 'sync';
                img.style.opacity = '0.45';
                img.style.transition = 'opacity 0.25s ease';

                let attempt = 0;
                const load = (resolve) => {
                    const url = baseUrl + (baseUrl.includes('?') ? '&' : '?') + 't=' + Date.now() + '_' + attempt;

                    img.onload = () => {
                        img.style.opacity = '1';
                        img.removeAttribute('title');
                        if (resolve) resolve(true);
                    };

                    img.onerror = () => {
                        attempt += 1;
                        if (attempt < retries) {
                            setTimeout(() => load(resolve), delayMs);
                        } else {
                            img.style.opacity = '1';
                            img.title = 'Falha ao carregar imagem no painel';
                            if (resolve) resolve(false);
                        }
                    };

                    img.src = url;
                };

                return new Promise((resolve) => load(resolve));
            }
            
            // Stats
            const statsHtml = `
                <div class="stat-card">
                    <h3>Simulation Duration</h3>
                    <div class="value">${summary.simulation_duration_minutes} min</div>
                </div>
                <div class="stat-card">
                    <h3>Time Steps</h3>
                    <div class="value">${history.length}</div>
                </div>
                <div class="stat-card">
                    <h3>Max Water Depth</h3>
                    <div class="value">${summary.max_water_depth_m.toFixed(2)} m</div>
                </div>
                <div class="stat-card">
                    <h3>Total Water Volume</h3>
                    <div class="value">${(summary.total_water_volume_m3 / 1e6).toFixed(2)}M m³</div>
                </div>
            `;
            document.getElementById('stats-container').innerHTML = statsHtml;
            
            // Images
            // Orthomosaic is optional; hide container if not available
            const orthoContainer = document.getElementById('orthoContainer');
            const orthoImg = document.getElementById('orthoImage');
            orthoImg.onerror = () => { orthoContainer.style.display = 'none'; };
            orthoImg.onload = () => { orthoContainer.style.display = 'block'; };
            orthoImg.src = '/image/orthomosaic?t=' + Date.now();

            // Carrega os painéis pesados em sequência (evita cancelamento/competição de requests)
            (async () => {
                await setImageWithRetry('demImage', '/image/dem', 20, 3000);
                await setImageWithRetry('flowImage', '/image/flow-direction', 20, 3000);
                await setImageWithRetry('flowAccImage', '/image/flow-accumulation', 20, 3000);
                await setImageWithRetry('peakImage', '/image/water-peak', 20, 3000);
                
                // Load ML validation results if available
                await loadMLValidationResults();
            })();

            setImageWithRetry('resultImage', '/image/results');
            setImageWithRetry('animGif', '/video/animation.gif');
            
            // Time Series
            const times = history.map(h => h.time_minutes);
            const flooded = history.map(h => h.flooded_area_percent);
            const maxdepth = history.map(h => h.max_water_depth_m);
            const volumes = history.map(h => h.total_water_volume_m3 / 1e6);
            
            // Destroy existing chart if it exists
            if (timeseriesChartInstance) {
                timeseriesChartInstance.destroy();
            }
            
            const ctx = document.getElementById('timeseriesChart').getContext('2d');
            timeseriesChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: times,
                    datasets: [
                        {
                            label: 'Flooded Area (%)',
                            data: flooded,
                            borderColor: '#1e88e5',
                            backgroundColor: 'rgba(30, 136, 229, 0.1)',
                            yAxisID: 'y'
                        },
                        {
                            label: 'Max Depth (m)',
                            data: maxdepth,
                            borderColor: '#d32f2f',
                            backgroundColor: 'rgba(211, 47, 47, 0.1)',
                            yAxisID: 'y1'
                        },
                        {
                            label: 'Volume (M m³)',
                            data: volumes,
                            borderColor: '#388e3c',
                            backgroundColor: 'rgba(56, 142, 60, 0.1)',
                            yAxisID: 'y2'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Flood Simulation Time Series', font: { size: 14, weight: 'bold' } },
                        legend: { display: true, position: 'top' }
                    },
                    scales: {
                        y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Flooded Area (%)' } },
                        y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Max Depth (m)' }, grid: { drawOnChartArea: false } },
                        y2: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Volume (M m³)' }, grid: { drawOnChartArea: false } },
                        x: { title: { display: true, text: 'Time (minutes)' } }
                    }
                }
            });
            
        }
        
        function downloadGeoTIFF() {
            if (!lastSimulationData) return;
            window.location.href = '/download/dem-geotiff';
        }

        function downloadOrthomosaic() {
            if (!lastSimulationData) return;
            window.location.href = '/download/orthomosaic';
        }
        
        function downloadWaterGeoTIFF() {
            if (!lastSimulationData) return;
            window.location.href = '/download/water-geotiff';
        }

        function downloadWaterGeoTIFFRaw() {
            if (!lastSimulationData) return;
            window.location.href = '/download/water-geotiff-raw';
        }

        function downloadWaterRGBA() {
            if (!lastSimulationData) return;
            window.location.href = '/download/lamina-agua-rgba';
        }

        function downloadWaterGifQGIS() {
            if (!lastSimulationData) return;
            window.location.href = '/download/lamina-gif-qgis';
        }

        function downloadFloodGPKG() {
            if (!lastSimulationData) return;
            window.location.href = '/download/fluxo-preferencial-gpkg';
        }
        
        function downloadFlowDirection() {
            if (!lastSimulationData) return;
            window.location.href = '/download/fluxo-preferencial-d8';
        }

        function downloadFlowAccumulation() {
            if (!lastSimulationData) return;
            window.location.href = '/download/flow-accumulation';
        }
        
        function downloadAnimGif() {
            if (!lastSimulationData) return;
            window.location.href = '/video/animation.gif';
        }
        
        function downloadAllData() {
            if (!lastSimulationData) return;
            window.location.href = '/download/all-data-zip';
        }
        
        async function loadMLValidationResults() {
            try {
                const response = await fetch('/api/ml-metrics');
                if (!response.ok) throw new Error('ML metrics not available');
                
                const metrics = await response.json();
                document.getElementById('mlResultsSection').style.display = 'block';
                
                // Load ML validation images
                setImageWithRetry('mlMetricsImage', '/image/ml-validation', 15, 2000);
                setImageWithRetry('floodProbImage', '/image/flood-probability', 15, 2000);
                
                // Display ML Metrics Chart
                displayMLMetricsChart(metrics);
            } catch (error) {
                console.log('ML validation results not available:', error.message);
                document.getElementById('mlResultsSection').style.display = 'none';
            }
        }
        
        function displayMLMetricsChart(metrics) {
            try {
                const mlCtx = document.getElementById('mlMetricsChart');
                if (!mlCtx) return;
                
                // Extract feature importances
                const featureImportances = metrics.feature_importances || {};
                const features = Object.keys(featureImportances);
                const importances = Object.values(featureImportances);
                
                // Destroy existing chart if it exists
                if (window.mlChartInstance) {
                    window.mlChartInstance.destroy();
                }
                
                const ctx = mlCtx.getContext('2d');
                window.mlChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: features,
                        datasets: [
                            {
                                label: 'Feature Importance',
                                data: importances,
                                backgroundColor: '#1e88e5',
                                borderColor: '#1565c0',
                                borderWidth: 2
                            }
                        ]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: `ML Validation Metrics | ROC-AUC: ${metrics.roc_auc.toFixed(4)} | F1: ${metrics.f1.toFixed(4)} | Precision: ${metrics.precision.toFixed(4)} | Recall: ${metrics.recall.toFixed(4)}`,
                                font: { size: 12, weight: 'bold' }
                            },
                            legend: { display: true, position: 'top' }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Importance Score' }, min: 0, max: 0.35 }
                        }
                    }
                });
            } catch (error) {
                console.error('Error displaying ML metrics chart:', error);
            }
        }
        
        function downloadMLMetrics() {
            window.location.href = '/api/ml-metrics';
        }
        
        function downloadFloodProbability() {
            window.location.href = '/download/flood-probability';
        }
        
        function downloadMLValidationZip() {
            window.location.href = '/download/ml-validation-zip';
        }
        
        function resetForm() {
            document.getElementById('rainfall').value = 100;
            document.getElementById('duration').value = 500;
            document.getElementById('rainfallMode').value = 'uniform';
            document.getElementById('demFile').value = '';
            document.getElementById('orthoFile').value = '';
            document.getElementById('rainfallValue').textContent = '100 mm';
            document.getElementById('durationValue').textContent = '500 min';
            document.getElementById('resultsSection').style.display = 'none';
            showAlert('Form reset', 'success');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/logo')
def get_logo():
    logo_path = Path("logo.png")
    if logo_path.exists():
        return send_file(logo_path, mimetype='image/png')
    return "Logo not found", 404

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    try:
        params = request.json if request.is_json else None
        rainfall_mm = (params or request.form).get('rainfall_mm', 100)
        duration_minutes = (params or request.form).get('duration_minutes', 500)
        diffusion_rate = (params or request.form).get('diffusion_rate', 0.22)
        flood_threshold = (params or request.form).get('flood_threshold', 0.05)
        
        rainfall_mm = float(rainfall_mm)
        duration_minutes = float(duration_minutes)
        diffusion_rate = float(np.clip(float(diffusion_rate), 0.05, 0.6))
        flood_threshold = float(np.clip(float(flood_threshold), 0.005, 0.5))
        
        # Check if custom DEM was uploaded
        custom_dem: Optional[np.ndarray] = None
        out_transform = None
        out_crs = None
        if 'dem_file' in request.files:
            dem_file = request.files['dem_file']
            if dem_file and (dem_file.filename or '').lower().endswith(('.tif', '.tiff', '.geotiff')):
                try:
                    custom_dem, out_transform, out_crs = _load_dem_upload(dem_file, target_shape=(100, 100))
                    logger.info(f"✅ Custom DEM loaded/resampled: shape {custom_dem.shape}")
                except Exception as e:
                    logger.warning(f"Could not load custom DEM: {e}, using synthetic")
        
        # Use custom or synthetic DEM
        dem = custom_dem if custom_dem is not None else _generate_synthetic_dem(shape=(100, 100))

        # Optional orthomosaic upload (RGB background)
        ortho_rgb = None
        
        # ⚠️ IMPORTANTE: Limpar ortomosaico anterior se nenhum foi selecionado
        ortho_file = request.files.get('ortho_file')
        if 'ortho_file' not in request.files or not ortho_file or not ortho_file.filename:
            # Remover arquivos do ortomosaico anterior
            for ortho_file in [Path("outputs/test_run/orthomosaic.png"), Path("outputs/test_run/orthomosaic.tif")]:
                if ortho_file.exists():
                    ortho_file.unlink()
                    logger.info(f"🗑️ Removido ortomosaico anterior: {ortho_file.name}")
        
        if 'ortho_file' in request.files:
            ortho_file = request.files['ortho_file']
            if ortho_file and ortho_file.filename:
                try:
                    ortho_rgb, ortho_transform, ortho_crs = _load_orthomosaic_upload(ortho_file, target_shape=dem.shape)
                    # If we didn't get a georef from DEM, try to adopt it from ortho GeoTIFF
                    if out_transform is None or out_crs is None:
                        if ortho_transform is not None and ortho_crs is not None:
                            out_transform, out_crs = ortho_transform, ortho_crs

                    if ortho_rgb is not None:
                        _save_orthomosaic_products(ortho_rgb, transform=out_transform, crs=out_crs)
                    logger.info("✅ Orthomosaic loaded and saved")
                except Exception as e:
                    logger.warning(f"Could not load orthomosaic: {e}")

        # If still missing georef (no DEM GeoTIFF and no orthomosaic GeoTIFF), fall back
        if out_transform is None or out_crs is None:
            out_transform, out_crs = _default_georef_for_array(dem)

        sources = _generate_rainfall_sources(shape=dem.shape)
        
        # Use GamaFloodModelNumpy (D8-based with directional flow)
        model = GamaFloodModelNumpy(
            dem_data=dem,
            sources_mask=sources,
            diffusion_rate=diffusion_rate,
            flood_threshold=flood_threshold,
            cell_size_meters=25.0,
            river_mask=None,  # Optional: provide river mask if available
            sources_intensity=None,  # Optional: provide intensity map if available
            intensity_mode="relative"
        )
        
        num_steps = int(duration_minutes // 10)
        rainfall_per_step = rainfall_mm / num_steps
        
        logger.info(f"Running GamaFloodModelNumpy simulation: {rainfall_mm}mm over {duration_minutes}min")
        
        for step in range(num_steps):
            model.add_water(rainfall_per_step)  # Add rainfall in mm
            model.run_flow_step()  # Route water using D8 directional flow
            model.update_stats(10)  # Record statistics with 10-minute timestep
        
        # Calculate probability based on ACTUAL water depths from simulation
        # This way 50mm vs 500mm creates different probability maps
        water_heights = model.water_height.astype(np.float32)
        
        # Normalize water depths to probability (0-1 scale)
        valid_dem = np.isfinite(dem) & (dem > -1e6)
        water_valid = water_heights[valid_dem]
        
        if water_valid.size > 0:
            max_water = float(np.nanmax(water_valid))
            min_water = float(np.nanmin(water_valid[water_valid > 0.0001])) if np.any(water_valid > 0.0001) else 0.0001
        else:
            max_water = 1.0
            min_water = 0.0001
        
        # Probability = normalized water depth (higher water = higher probability)
        # This automatically varies with rainfall amount!
        if max_water > min_water:
            prob = np.clip((water_heights - min_water) / (max_water - min_water + 1e-6), 0.0, 1.0)
        else:
            prob = np.zeros_like(water_heights, dtype=np.float32)
        
        # Where no water, probability is 0
        prob = np.where(water_heights > 0.001, prob, 0.0)
        
        clf = None  # Not needed anymore since we use water-based probability
        
        _generate_visualizations(dem, sources, model, prob, ortho_rgb=ortho_rgb)
        _generate_animation_improved(dem, model, ortho_rgb=ortho_rgb)
        _save_geotiff(dem, model.water_height, transform=out_transform, crs=out_crs)
        
        # Calculate and save preferential flow products
        flow_direction = _calculate_flow_direction(dem)
        _save_flow_direction(flow_direction, transform=out_transform, crs=out_crs)
        flow_acc = _calculate_flow_accumulation(flow_direction, dem)
        _save_flow_accumulation(flow_acc, transform=out_transform, crs=out_crs)
        
        # ALSO save water-weighted flow accumulation (varies with rainfall amount!)
        # This shows where water ACTUALLY accumulated in the simulation
        try:
            flow_acc_water_weighted = _calculate_flow_accumulation_water_weighted(
                flow_direction, dem, model.water_height
            )
            _save_flow_accumulation(
                flow_acc_water_weighted,
                transform=out_transform,
                crs=out_crs,
                filename="flow_accumulation_water_weighted.tif"
            )
        except Exception as e:
            logger.warning(f"Could not save water-weighted flow accumulation: {e}")

        # Save products similar to reference outputs (RGBA GeoTIFF + GPKG)
        _save_water_rgba_geotiff(model.water_height, transform=out_transform, crs=out_crs, dem=dem)
        _save_inundation_gpkg(model.water_height, transform=out_transform, crs=out_crs, dem=dem)
        
        # Create output directories
        output_dir = Path("outputs/test_run")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary data
        summary = {
            'simulation_duration_minutes': int(duration_minutes),
            'flood_threshold_m': float(flood_threshold),
            'max_water_depth_m': float(model.water_height.max()),
            'total_water_volume_m3': float(model.water_height.sum() * (dem.shape[0] * dem.shape[1])),
            'flooded_cells': int((model.water_height > 0.01).sum()),
            'total_cells': int(dem.size),
            'flooded_percent': float(100.0 * (model.water_height > 0.01).sum() / dem.size)
        }
        
        # Create history data from model diagnostics
        history = []
        if hasattr(model, 'history') and model.history:
            for i, h in enumerate(model.history):
                history.append({
                    'time_minutes': float(i * 10),
                    'flooded_area_percent': float(h.get('flooded_percent', 0)),
                    'max_water_depth_m': float(h.get('max_depth', 0)),
                    'total_water_volume_m3': float(h.get('total_water_volume_m3', 0))
                })
        else:
            # Generate synthetic history if not available
            for i in range(int(duration_minutes / 10) + 1):
                history.append({
                    'time_minutes': float(i * 10),
                    'flooded_area_percent': float(20 + i * 2),
                    'max_water_depth_m': float(0.1 + i * 0.05),
                    'total_water_volume_m3': float(1000 + i * 500)
                })
        
        # Save to JSON files
        summary_path = output_dir / "summary.json"
        history_path = output_dir / "history.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def _generate_synthetic_dem(shape=(100, 100)):
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
    sources = np.zeros(shape, dtype=bool)
    sources[10:25, 10:25] = True
    sources[10:25, 75:90] = True
    sources[75:90, 40:60] = True
    return sources

def _train_classifier(dem, water):
    clf = train_flood_classifier(dem, water, threshold=0.05, n_estimators=100)
    prob = predict_probability(clf, dem)
    return clf, prob

def _generate_visualizations(dem, sources, model, prob, ortho_rgb=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.close('all')  # Close any existing figures
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    dem_masked = np.ma.masked_invalid(dem)
    valid_dem = np.isfinite(dem)
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    # Fundo branco + DEM neutro em cinza (sem coloração de base)
    dem_float = dem.astype(np.float32)
    finite_vals = dem_float[np.isfinite(dem_float)]
    vmin_dem, vmax_dem = np.nanpercentile(finite_vals, (2, 98)) if finite_vals.size else (0, 1)
    dem_gray = cm.get_cmap('gray').copy()
    dem_gray.set_bad('white')
    dem_im = axes[0, 0].imshow(dem_masked, cmap=dem_gray, vmin=vmin_dem, vmax=vmax_dem, alpha=0.92)
    outside = np.where(valid_dem, 0.0, 1.0).astype(np.float32)
    axes[0, 0].imshow(outside, cmap='gray', vmin=0, vmax=1, alpha=np.where(valid_dem, 0.0, 1.0))
    axes[0, 0].set_title('DEM (neutral base / base neutra)')
    # Escala de cores do DEM (mesma base neutra)
    try:
        fig.colorbar(dem_im, ax=axes[0, 0], label='Altitude (m)')
    except Exception:
        pass
    axes[0, 0].legend(
        handles=[
            Patch(facecolor='lightgray', edgecolor='none', label='DEM relief'),
            Patch(facecolor='white', edgecolor='#666', label='Outside DEM domain'),
        ],
        loc='lower right',
        fontsize=8,
        framealpha=0.92,
    )
    
    # Rainfall panel: fundo branco + DEM em contornos (sem preenchimento de fundo)
    axes[0, 1].imshow(np.ones_like(dem_float, dtype=np.float32), cmap='gray', vmin=0, vmax=1, zorder=1)
    try:
        dem_levels = np.linspace(vmin_dem, vmax_dem, 16) if np.isfinite(vmin_dem) and np.isfinite(vmax_dem) else []
        if len(dem_levels) > 0:
            axes[0, 1].contour(
                np.ma.masked_where(~valid_dem, dem_float),
                levels=dem_levels,
                colors='k',
                linewidths=0.40,
                alpha=0.20,
                zorder=2,
            )
    except Exception:
        pass
    axes[0, 1].contourf(sources.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.65, zorder=3)
    axes[0, 1].set_title('Rainfall Sources')
    axes[0, 1].legend(
        handles=[
            Patch(facecolor='red', edgecolor='none', alpha=0.6, label='Rainfall source'),
            Patch(facecolor='lightgray', edgecolor='none', alpha=0.8, label='Background relief'),
        ],
        loc='lower right',
        fontsize=8,
        framealpha=0.92,
    )
    
    # Water depth over background - com sensibilidade adaptativa
    import matplotlib.colors as mcolors
    water_raw = np.where(np.isfinite(model.water_height), np.clip(model.water_height.astype(np.float32), 0.0, None), np.nan)

    positive = water_raw[np.isfinite(water_raw) & (water_raw > 0.0)]
    if positive.size > 0:
        p10 = float(np.nanpercentile(positive, 10.0))
        water_threshold = float(np.clip(min(0.005, p10 * 0.35), 1e-6, 0.005))
    else:
        water_threshold = 1e-6

    wet_mask = np.isfinite(water_raw) & (water_raw > water_threshold)
    if np.count_nonzero(wet_mask) < 20 and positive.size > 0:
        # fallback para eventos com lâmina muito rasa
        wet_mask = np.isfinite(water_raw) & (water_raw > 0.0)

    # Mascarar água: mostrar apenas onde há água acima do threshold
    water_display = np.ma.masked_where(~wet_mask, water_raw)

    # Colormap com gradiente legível (ciano -> azul), sem preto
    water_cmap = mcolors.LinearSegmentedColormap.from_list(
        "water_layer",
        [
            (0.86, 0.98, 1.00),
            (0.58, 0.90, 1.00),
            (0.20, 0.76, 0.99),
            (0.02, 0.54, 0.95),
            (0.00, 0.34, 0.78),
        ],
        N=256,
    )
    water_cmap.set_under("white")
    water_cmap.set_bad("white")
    
    # Final water panel: fundo branco + DEM em contornos (sem preenchimento)
    if dem is not None:
        dem_float = dem.astype(np.float32)
        finite_vals = dem_float[np.isfinite(dem_float)]
        vmin_dem, vmax_dem = np.nanpercentile(finite_vals, (2, 98)) if finite_vals.size else (0, 1)
        axes[1, 0].imshow(np.ones_like(dem_float, dtype=np.float32), cmap='gray', vmin=0, vmax=1, zorder=1)
        try:
            dem_levels_w = np.linspace(vmin_dem, vmax_dem, 16) if np.isfinite(vmin_dem) and np.isfinite(vmax_dem) else []
            if len(dem_levels_w) > 0:
                axes[1, 0].contour(
                    np.ma.masked_where(~valid_dem, dem_float),
                    levels=dem_levels_w,
                    colors='k',
                    linewidths=0.35,
                    alpha=0.18,
                    zorder=1.15,
                )
        except Exception:
            pass
        outside = np.where(valid_dem, 0.0, 1.0).astype(np.float32)
        axes[1, 0].imshow(outside, cmap='gray', vmin=0, vmax=1, alpha=np.where(valid_dem, 0.0, 1.0), zorder=1.2)
    
    # Sobrepor água com escala robusta e alpha por profundidade
    if np.any(wet_mask):
        wet_vals = water_raw[wet_mask]
        # Use GLOBAL fixed scales to preserve difference between rainfall amounts
        global_vmin = 0.001  # 1mm threshold
        global_vmax = 0.50   # 500mm reference scale
        vis_vmin = float(max(global_vmin, np.clip(np.nanpercentile(wet_vals, 5), global_vmin, global_vmax)))
        vis_vmax = float(max(global_vmin + 1e-6, global_vmax))

        # Estatísticas para diagnosticar "tudo raso" (unidades em metros)
        try:
            p90 = float(np.nanpercentile(wet_vals, 90.0))
            wmax = float(np.nanmax(wet_vals))
        except Exception:
            p90, wmax = (np.nan, np.nan)

        depth_norm = np.zeros_like(water_raw, dtype=np.float32)
        depth_norm[wet_mask] = np.clip((water_raw[wet_mask] - vis_vmin) / (vis_vmax - vis_vmin + 1e-6), 0.0, 1.0)
        alpha_water = np.zeros_like(water_raw, dtype=np.float32)
        alpha_water[wet_mask] = np.clip(0.68 + 0.32 * (depth_norm[wet_mask] ** 0.60), 0.0, 1.0)

        im = axes[1, 0].imshow(
            water_display,
            cmap=water_cmap,
            norm=mcolors.PowerNorm(gamma=0.72, vmin=vis_vmin, vmax=vis_vmax),
            alpha=alpha_water,
            zorder=2,
            interpolation='bilinear',
        )
    else:
        im = axes[1, 0].imshow(
            np.zeros_like(water_raw),
            cmap=water_cmap,
            vmin=0.0,
            vmax=1.0,
            alpha=0.0,
            zorder=2,
        )
    plt.colorbar(im, ax=axes[1, 0], label='Water Depth (m)')
    if np.isfinite(wmax) and np.isfinite(p90):
        # Mostra em cm quando é bem raso para facilitar leitura
        if wmax < 0.30:
            axes[1, 0].set_title(f'Final Water Depth (p90={p90*100:.1f} cm; max={wmax*100:.1f} cm)')
        else:
            axes[1, 0].set_title(f'Final Water Depth (p90={p90:.3f} m; max={wmax:.3f} m)')
    else:
        axes[1, 0].set_title('Final Water Depth')
    axes[1, 0].legend(
        handles=[
            Patch(facecolor=(0.70, 0.94, 1.00), edgecolor='none', label='Shallow water'),
            Patch(facecolor=(0.20, 0.76, 0.99), edgecolor='none', label='Moderate depth'),
            Patch(facecolor=(0.00, 0.34, 0.78), edgecolor='none', label='Deep accumulation'),
        ],
        loc='lower right',
        fontsize=8,
        framealpha=0.92,
        title='Water depth guide',
        title_fontsize=8,
    )
    
    # Flood Probability - display ALL valid probability values
    prob_f = np.where(np.isfinite(prob), np.clip(prob.astype(np.float32), 0.0, 1.0), np.nan)
    prob_valid = prob_f[np.isfinite(prob_f) & valid_dem]
    if prob_valid.size > 0:
        p5 = float(np.nanpercentile(prob_valid, 5.0))
        p995 = float(np.nanpercentile(prob_valid, 99.5))
        prob_vmin = max(0.0, min(p5, 0.05))
        prob_vmax = max(prob_vmin + 1e-6, p995)
        # Guarantee vmin < vmax
        if prob_vmax <= prob_vmin:
            prob_vmax = prob_vmin + 0.1
    else:
        prob_vmin, prob_vmax = 0.0, 1.0

    # Display probability everywhere in valid DEM (no excessive masking)
    prob_display = np.ma.masked_where(~valid_dem, prob_f)
    
    # Criar cópia do colormap
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    prob_cmap = cm.get_cmap('RdYlGn_r').copy()
    prob_cmap.set_under("white")  # Fundo branco para valores mascarados
    prob_cmap.set_bad("white")
    
    # Flood probability panel: fundo branco + DEM em contornos (sem preenchimento)
    dem_float = dem.astype(np.float32)
    finite_vals = dem_float[np.isfinite(dem_float)]
    vmin_dem, vmax_dem = np.nanpercentile(finite_vals, (5, 95)) if finite_vals.size else (0, 1)
    axes[1, 1].imshow(np.ones_like(dem_float, dtype=np.float32), cmap='gray', vmin=0, vmax=1, zorder=1)
    try:
        dem_levels_p = np.linspace(vmin_dem, vmax_dem, 16) if np.isfinite(vmin_dem) and np.isfinite(vmax_dem) else []
        if len(dem_levels_p) > 0:
            axes[1, 1].contour(
                np.ma.masked_where(~valid_dem, dem_float),
                levels=dem_levels_p,
                colors='k',
                linewidths=0.33,
                alpha=0.16,
                zorder=1.15,
            )
    except Exception:
        pass
    
    # Render probability with adaptive alpha: stronger alpha for higher values
    prob_alpha = np.ones_like(prob_f, dtype=np.float32) * 0.20
    if prob_valid.size > 0:
        prob_norm = np.clip((prob_f - prob_vmin) / (prob_vmax - prob_vmin + 1e-6), 0.0, 1.0)
        prob_alpha = np.where(valid_dem, 0.20 + 0.70 * (prob_norm ** 0.65), 0.0)

    im = axes[1, 1].imshow(
        prob_display,
        cmap=prob_cmap,
        norm=mcolors.Normalize(vmin=prob_vmin, vmax=prob_vmax),
        alpha=prob_alpha,
        zorder=2,
    )
    axes[1, 1].set_title('Flood Probability')
    plt.colorbar(im, ax=axes[1, 1], label='Probability (adaptive scale)')
    axes[1, 1].legend(
        handles=[
            Patch(facecolor=prob_cmap(0.15), edgecolor='none', label='Low probability'),
            Patch(facecolor=prob_cmap(0.50), edgecolor='none', label='Medium probability'),
            Patch(facecolor=prob_cmap(0.85), edgecolor='none', label='High probability'),
            Line2D([0], [0], color='white', lw=0, label='(white area: very low/none)'),
        ],
        loc='lower right',
        fontsize=8,
        framealpha=0.92,
    )

    fig.suptitle(
        'Simulation summary: terrain, rainfall sources, final water depth, and flood probability',
        fontsize=12,
        fontweight='bold',
        y=0.995,
    )
    
    plt.tight_layout()
    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/test_run/results_visualization.png", dpi=100)
    plt.close()

def _calculate_d8_flow_directions(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate D8 flow directions (steepest descent) as dy/dx displacements.
    
    Returns (dy, dx) as int8 arrays with same shape as dem.
    Cells with no valid outlet (pits/flats) have (0,0).
    """
    dem_f = np.asarray(dem, dtype=np.float32)
    H, W = dem_f.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.int8), np.zeros((H, W), dtype=np.int8)
    
    # Pad DEM with edge values to handle boundaries
    pad = np.pad(dem_f, pad_width=1, mode='edge')
    center = pad[1:-1, 1:-1]
    
    sqrt2 = np.float32(np.sqrt(2.0))
    dirs = [
        (-1, -1, sqrt2), (-1, 0, np.float32(1.0)), (-1, 1, sqrt2),
        (0, -1, np.float32(1.0)),                   (0, 1, np.float32(1.0)),
        (1, -1, sqrt2),  (1, 0, np.float32(1.0)),   (1, 1, sqrt2),
    ]
    
    # Calculate slopes for each D8 direction
    slopes: list = []
    for dy, dx, dist in dirs:
        neigh = pad[(1 + dy):(1 + dy + H), (1 + dx):(1 + dx + W)]
        diff = center - neigh
        s = diff / dist
        # Only allow downslope flow
        s = np.where(np.isfinite(s) & (s > 0), s, -np.inf)
        slopes.append(s)
    
    # Find steepest direction
    stack = np.stack(slopes, axis=0)  # (8, H, W)
    idx = np.argmax(stack, axis=0).astype(np.int8)
    max_s = np.take_along_axis(stack, idx[None, :, :].astype(np.int64), axis=0)[0]
    valid = np.isfinite(max_s)
    
    dy_list = np.array([d[0] for d in dirs], dtype=np.int8)
    dx_list = np.array([d[1] for d in dirs], dtype=np.int8)
    out_dy = dy_list[idx]
    out_dx = dx_list[idx]
    out_dy[~valid] = 0
    out_dx[~valid] = 0
    return out_dy, out_dx


def _generate_animation_improved(dem, model, ortho_rgb=None):
    """Generate beautiful animated GIF showing water flowing and accumulating in topographic lows.
    
    Similar style to simulacao_inundacao.gif:
    - Terrain/DEM as background with shaded relief
    - Water overlay in blue tones (white → ciano → turquoise → blue)
    - Shows real simulation progress and statistics
    """
    from matplotlib.colors import LinearSegmentedColormap, LightSource, PowerNorm, Normalize
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.close('all')
    
    logger.info("Generating animated GIF with DEM/DOM background + rainfall + accumulation...")
    
    history_len = len(model.history)
    if history_len == 0:
        logger.warning("No history to animate")
        return
    
    frames = []
    
    # DEM mask (valid domain). Outside domain remains white.
    dem_float = dem.astype(np.float32, copy=False)
    valid_dem = np.isfinite(dem_float)

    # Restringe ao domínio original do DEM uploadado (quando disponível)
    try:
        mask_path = Path("outputs/test_run/dem_valid_mask.npy")
        if mask_path.exists():
            original_mask = np.load(mask_path)
            if original_mask.shape == valid_dem.shape:
                valid_dem = valid_dem & (original_mask.astype(np.uint8) > 0)
    except Exception:
        pass

    finite_vals = dem_float[valid_dem]

    # Normalização e relevo para destacar diferenças altimétricas
    if finite_vals.size:
        dem_p2, dem_p98 = np.nanpercentile(finite_vals, (2, 98))
        dem_norm = np.where(valid_dem, np.clip((dem_float - dem_p2) / (dem_p98 - dem_p2 + 1e-6), 0.0, 1.0), 0.0)
        dem_fill = np.where(valid_dem, dem_float, np.nanmedian(finite_vals))
    else:
        dem_norm = np.zeros_like(dem_float, dtype=np.float32)
        dem_fill = np.zeros_like(dem_float, dtype=np.float32)

    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_fill, vert_exag=1.6, dx=1.0, dy=1.0)

    # Guias topográficos (declividade + rota preferencial) para visual não uniforme
    gy, gx = np.gradient(dem_fill)
    slope_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    slope_valid = slope_mag[valid_dem]
    if slope_valid.size > 0:
        s_p10, s_p90 = np.nanpercentile(slope_valid, (10, 90))
        slope_norm = np.clip((slope_mag - s_p10) / (s_p90 - s_p10 + 1e-6), 0.0, 1.0)
    else:
        slope_norm = np.zeros_like(dem_fill, dtype=np.float32)
    low_slope_factor = np.clip(1.0 - slope_norm, 0.0, 1.0)
    low_slope_factor[~valid_dem] = 0.0

    try:
        flow_dir_vis = _calculate_flow_direction(dem_fill)
        flow_acc_vis = _calculate_flow_accumulation(flow_dir_vis, dem_fill).astype(np.float32)
        acc_log = np.log1p(np.clip(flow_acc_vis, 0.0, None))
        acc_vals = acc_log[valid_dem]
        if acc_vals.size > 0:
            a_p30, a_p99 = np.nanpercentile(acc_vals, (30, 99))
            acc_norm = np.clip((acc_log - a_p30) / (a_p99 - a_p30 + 1e-6), 0.0, 1.0)
        else:
            acc_norm = np.zeros_like(acc_log, dtype=np.float32)
    except Exception:
        acc_norm = np.zeros_like(dem_fill, dtype=np.float32)

    topo_pref = np.clip(0.55 * low_slope_factor + 0.45 * acc_norm, 0.0, 1.0)

    # Chuva visual distribuída por todo o DEM válido
    source_ys, source_xs = np.where(valid_dem)
    has_sources = source_ys.size > 0
    
    # Find max water depth for consistent coloring
    max_water_depth = 0.0
    snapshots_for_persistence = []
    for h in model.history:
        if "water_height_snapshot" in h:
            snap = h["water_height_snapshot"].astype(np.float32)
            snapshots_for_persistence.append(snap)
            max_water_depth = max(max_water_depth, float(np.nanmax(snap)))
    # NÃO forçar mínimo - deixar refletir a chuva real
    # Historicamente era: max_water_depth = max(max_water_depth, 0.5)
    max_water_depth = max(max_water_depth, 0.01)  # Apenas um mínimo tiny para evitar divisão por zero
    
    # Garantir que max_water_depth is valid (not NaN or inf)
    if not np.isfinite(max_water_depth):
        max_water_depth = 0.01
    max_water_depth = float(max_water_depth)

    # CORREÇÃO: Escala GLOBAL para todo o GIF - não varia por frame
    # Isso garante que diferentes chuvas tenham diferentes escalas visuais
    global_vmin_scale = 0.003  # 3mm - limiar físico mínimo
    global_vmax_scale = max(global_vmin_scale + 0.008, global_vmin_scale + 0.85 * max_water_depth)
    
    # Validate scale values - prevent vmin >= vmax
    if not np.isfinite(global_vmax_scale) or global_vmax_scale <= global_vmin_scale:
        global_vmax_scale = global_vmin_scale + 0.05  # 50mm default fallback
    
    logger.info(f"GIF Scale: vmin={global_vmin_scale:.4f}m, vmax={global_vmax_scale:.4f}m (max_water_depth={max_water_depth:.4f}m)")

    # Mapa de persistência de água (onde acumula ao longo do tempo)
    if snapshots_for_persistence:
        stack = np.stack(snapshots_for_persistence, axis=0)
        pool_persistence = np.mean(stack > 0.015, axis=0).astype(np.float32)
    else:
        pool_persistence = np.zeros_like(dem_float, dtype=np.float32)
    
    # Water colormap: azul bem destacado para leitura operacional
    water_cmap = LinearSegmentedColormap.from_list(
        "water_flow",
        [
            (0.90, 0.98, 1.00),  # água rasa (muito clara)
            (0.62, 0.90, 1.00),  # azul/ciano claro
            (0.03, 0.48, 0.92),  # azul forte
            (0.00, 0.36, 0.82),  # acúmulo
            (0.00, 0.26, 0.68),  # acúmulo profundo
            (0.00, 0.16, 0.52),  # núcleo de poça
        ],
        N=256,
    )

    # Hotspots para poças/acúmulos: amarelo/laranja por cima da água
    puddle_hot_cmap = LinearSegmentedColormap.from_list(
        "puddle_hotspots",
        [
            (1.00, 0.96, 0.55),
            (1.00, 0.78, 0.20),
            (0.98, 0.40, 0.12),
        ],
        N=256,
    )

    # Background DOM/ortomosaico (se fornecido)
    ortho_bg = None
    try:
        if ortho_rgb is not None and hasattr(ortho_rgb, "shape") and ortho_rgb.ndim == 3 and ortho_rgb.shape[2] >= 3:
            ob = ortho_rgb[..., :3]
            if ob.dtype == np.uint8:
                ob = ob.astype(np.float32) / 255.0
            else:
                ob = ob.astype(np.float32)
                if np.nanmax(ob) > 1.5:
                    ob = ob / 255.0
            ob = np.clip(ob, 0.0, 1.0)
            # Leve clareamento para o overlay de água continuar legível
            ob = np.clip(0.92 * ob + 0.08, 0.0, 1.0)
            if ob.shape[:2] == dem_float.shape:
                ortho_bg = ob
    except Exception:
        ortho_bg = None
    
    # Generate frames
    step_interval = max(1, history_len // 40)
    total_steps = len(range(0, history_len, step_interval))
    
    prev_water_snapshot = None
    for frame_num, step_idx in enumerate(range(0, history_len, step_interval)):
        if "water_height_snapshot" not in model.history[step_idx]:
            continue
        
        water_at_step = model.history[step_idx]["water_height_snapshot"].astype(np.float32)
        if prev_water_snapshot is None:
            flow_front = np.zeros_like(water_at_step, dtype=np.float32)
        else:
            flow_front = np.clip(water_at_step - prev_water_snapshot, 0.0, None)
        
        # Create figure
        fig = Figure(figsize=(12, 10), dpi=80, facecolor='white')
        ax = fig.add_subplot(111)
        
        # ===== Background: DOM (se existir) + DEM neutro (contornos) =====
        white_bg = np.ones((*dem_float.shape, 3), dtype=np.float32)
        ax.imshow(white_bg, zorder=1, interpolation='nearest')

        if ortho_bg is not None:
            ax.imshow(ortho_bg, zorder=1.22, interpolation='bilinear')

        # DEM more visible with higher alpha + hillshade for better relief visualization
        # Show terrain topography clearly so water accumulation patterns are obvious
        hillshade_overlay = np.ma.masked_where(~valid_dem, hillshade)
        ax.imshow(
            hillshade_overlay,
            cmap='gray',
            vmin=0.0,
            vmax=1.0,
            alpha=(0.35 if ortho_bg is not None else 0.45),  # INCREASED from 0.08/0.22
            zorder=1.30,
            interpolation='bilinear',
        )
        
        dem_gray = np.ma.masked_where(~valid_dem, dem_norm)
        ax.imshow(
            dem_gray,
            cmap='gray',
            vmin=0.0,
            vmax=1.0,
            alpha=(0.15 if ortho_bg is not None else 0.25),  # INCREASED from 0.08/0.22
            zorder=1.35,
            interpolation='bilinear',
        )

        # Contornos do DEM para leitura clara do relevo - MUCH DARKER AND THICKER
        try:
            levels = np.linspace(float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals)), 24) if finite_vals.size else []  # MORE CONTOURS
            if len(levels) > 0:
                dem_contour = np.ma.masked_where(~valid_dem, dem_float)
                ax.contour(
                    dem_contour,
                    levels=levels,
                    colors='darkgray',  # DARKER GRAY instead of black
                    linewidths=0.45,  # THICKER lines (was 0.25)
                    alpha=(0.32 if ortho_bg is not None else 0.42),  # MUCH DARKER (was 0.14/0.18)
                    zorder=1.40,  # ABOVE DEM
                )
        except Exception:
            pass

        # Branco fora do domínio válido do DEM
        outside = np.where(valid_dem, 0.0, 1.0).astype(np.float32)
        ax.imshow(outside, cmap='gray', vmin=0, vmax=1, alpha=np.where(valid_dem, 0.0, 1.0), zorder=2)
        
        # ===== WATER LAYER: mostrar escoamento + acúmulo =====
        threshold = 0.003  # limiar físico mínimo (3 mm)
        spread_threshold = 0.004  # trilha de escoamento/espalhamento no terreno

        wet_phys_mask = (water_at_step > threshold) & np.isfinite(water_at_step) & valid_dem
        wet_vals = water_at_step[wet_phys_mask]
        if wet_vals.size > 20:
            # Mais seletivo: evita película azul contínua em todo o domínio
            vis_threshold = float(max(threshold, np.nanpercentile(wet_vals, 58)))
        else:
            vis_threshold = float(threshold)

        # Evita "filme azul" em todo o quadro quando há água muito rasa espalhada
        valid_count = int(np.count_nonzero(valid_dem))
        wet_count = int(np.count_nonzero(wet_phys_mask))
        wet_cov = (wet_count / valid_count) if valid_count > 0 else 0.0
        if wet_vals.size > 40 and wet_cov > 0.22:
            vis_threshold = float(max(vis_threshold, np.nanpercentile(wet_vals, 72)))
            spread_threshold = float(max(spread_threshold, np.nanpercentile(wet_vals, 56)))

        # Fallback anti-quadro "branco": se a máscara de acúmulo ficar muito pequena,
        # afrouxa o limiar para voltar a mostrar a água sobre o DEM.
        vis_mask_test = (water_at_step >= vis_threshold) & np.isfinite(water_at_step) & valid_dem
        if np.count_nonzero(vis_mask_test) < 30 and wet_vals.size > 0:
            vis_threshold = float(max(threshold, np.nanpercentile(wet_vals, 35)))

        water_display = np.ma.masked_where((water_at_step < vis_threshold) | (~valid_dem), water_at_step)
        # Camada base fraca (água rasa), para manter contexto sem poluir
        water_base = np.ma.masked_where((water_at_step <= threshold) | (~valid_dem), water_at_step)

        # Base opaca para garantir leitura "por cima" do DEM
        accum_mask = (water_at_step >= max(vis_threshold, spread_threshold)) & (acc_norm > 0.55) & valid_dem
        if np.any(accum_mask):
            ax.imshow(
                np.ma.masked_where(~accum_mask, np.ones_like(water_at_step, dtype=np.float32)),
                cmap=LinearSegmentedColormap.from_list("water_opaque_base", [(0.08, 0.52, 0.96), (0.08, 0.52, 0.96)], N=2),
                alpha=0.18,
                zorder=3.55,
                interpolation='nearest',
            )

        # Trilha de escoamento: mostra por onde a água está se espalhando
        water_spread = np.ma.masked_where((water_at_step <= spread_threshold) | (~valid_dem), water_at_step)
        spread_alpha = np.zeros_like(water_at_step, dtype=np.float32)
        spread_mask = (water_at_step >= vis_threshold) & (acc_norm > 0.70) & valid_dem
        if np.any(spread_mask):
            spread_vmax = float(np.nanpercentile(water_at_step[spread_mask], 98))
            spread_vmax = max(spread_vmax, spread_threshold + 1e-6)
            # Ensure vmin < vmax before creating norm
            spread_vmin = spread_threshold + 1e-9
            if spread_vmax <= spread_vmin:
                spread_vmax = spread_vmin + 1e-6
            spread_norm = np.clip((water_at_step[spread_mask] - spread_threshold) / (spread_vmax - spread_threshold + 1e-6), 0, 1)
            spread_alpha[spread_mask] = np.clip((0.20 + 0.22 * (spread_norm ** 0.65)) * (0.78 + 0.38 * topo_pref[spread_mask]), 0.0, 0.60)
            ax.imshow(
                water_spread,
                cmap=LinearSegmentedColormap.from_list(
                    "water_spread",
                    [(0.92, 1.00, 1.00), (0.46, 0.92, 1.00), (0.10, 0.70, 1.00)],
                    N=128,
                ),
                norm=PowerNorm(gamma=1.0, vmin=spread_vmin, vmax=spread_vmax),
                alpha=spread_alpha,
                zorder=5.05,
                interpolation='bilinear',
            )

        # Frente de avanço (movimento entre frames)
        front_mask = (flow_front > 0.0030) & (water_at_step >= vis_threshold) & valid_dem
        if np.any(front_mask):
            front_alpha = np.zeros_like(flow_front, dtype=np.float32)
            front_norm = np.clip(flow_front[front_mask] / (np.nanpercentile(flow_front[front_mask], 95) + 1e-6), 0, 1)
            front_alpha[front_mask] = 0.20 + 0.16 * front_norm
            ax.imshow(
                np.ma.masked_where(~front_mask, flow_front),
                cmap=LinearSegmentedColormap.from_list("flow_front", [(0.85, 0.98, 1.00), (0.42, 0.88, 1.00), (0.08, 0.66, 0.98)], N=64),
                alpha=front_alpha,
                zorder=3.9,
                interpolation='bilinear',
            )

        # Escala dinâmica estável para destacar acúmulo (poças)
        current_vmin = float(global_vmin_scale)
        vmax_eff = float(global_vmax_scale)
        
        # Usar escala GLOBAL para garantir consistência entre frames
        # vis_threshold varia por frame para mascarar, mas normalização é fixa
        
        # Subtle highlighting of LOW AREAS (natural accumulation zones based on terrain)
        try:
            # Identify low-lying areas: bottom 20% of terrain elevation
            terrain_vals = dem_float[valid_dem]
            if terrain_vals.size > 100:
                low_terrain_threshold = float(np.nanpercentile(terrain_vals, 20))
                low_areas = (dem_float <= low_terrain_threshold) & valid_dem
                
                if np.any(low_areas):
                    # Subtle purple/blue tint on low areas to show natural accumulation zones
                    low_area_display = np.ma.masked_where(~low_areas, np.ones_like(dem_float, dtype=np.float32))
                    ax.imshow(
                        low_area_display,
                        cmap=LinearSegmentedColormap.from_list(
                            "low_areas",
                            [(0.0, 0.0, 0.0), (0.15, 0.25, 0.45)],  # Subtle dark blue
                            N=2
                        ),
                        alpha=0.08,  # Very subtle
                        zorder=1.45,
                        interpolation='nearest',
                    )
        except Exception:
            pass

        # 1) Render base (água rasa): deixa a lâmina mais visível sem virar "filme" pesado
        base_alpha = np.zeros_like(water_at_step, dtype=np.float32)
        # Inclui água acima do limiar físico (threshold) mesmo se vis_threshold estiver alto
        base_mask = (water_at_step > threshold) & np.isfinite(water_at_step) & valid_dem
        if np.any(base_mask):
            base_norm = np.clip((water_at_step[base_mask] - threshold) / (vmax_eff - threshold + 1e-6), 0, 1)
            # Mais presente em água rasa (clareia/realça) - AUMENTADO para melhor visualização
            base_alpha[base_mask] = 0.16 + 0.28 * (base_norm ** 0.70)
            base_alpha[base_mask] = np.clip(base_alpha[base_mask], 0.0, 0.55)
            # Ensure vmin < vmax before creating norm
            base_vmin = threshold + 1e-9
            base_vmax = vmax_eff
            if base_vmax <= base_vmin:
                base_vmax = base_vmin + 1e-6
            ax.imshow(
                water_base,
                cmap=water_cmap,
                norm=PowerNorm(gamma=0.52, vmin=base_vmin, vmax=base_vmax),
                alpha=base_alpha,
                zorder=3.7,
                interpolation='nearest',
            )

        # Per-pixel alpha: água fraca no terreno todo e destaque nas poças
        alpha_water = np.zeros_like(water_at_step, dtype=np.float32)
        wet_mask = np.isfinite(water_at_step) & (water_at_step >= vis_threshold) & valid_dem

        if np.any(wet_mask):
            normalized_depth = (water_at_step[wet_mask] - current_vmin) / (vmax_eff - current_vmin + 1e-6)
            # água rasa com pouca opacidade; acúmulo aumenta gradualmente - AUMENTADO
            alpha_main = 0.14 + 0.42 * (np.clip(normalized_depth, 0, 1) ** 0.75)
            alpha_water[wet_mask] = np.clip(alpha_main, 0.0, 0.68)

        # Gamma MENOR para melhor separação entre água rasa vs acúmulo
        water_vmin = current_vmin
        water_vmax = vmax_eff
        if water_vmax <= water_vmin:
            water_vmax = water_vmin + 1e-6
        norm = PowerNorm(gamma=0.40, vmin=water_vmin, vmax=water_vmax)
        im = ax.imshow(
            water_display,
            cmap=water_cmap,
            norm=norm,
            alpha=alpha_water,
            zorder=4,
            interpolation='nearest',
        )

        # Mapa de profundidade normalizada (para linhas graduadas)
        depth_norm_full = np.zeros_like(water_at_step, dtype=np.float32)
        try:
            denom_d = float(max(1e-6, vmax_eff - vis_threshold))
            finite_valid = np.isfinite(water_at_step) & valid_dem
            depth_norm_full[finite_valid] = np.clip((water_at_step[finite_valid] - vis_threshold) / denom_d, 0.0, 1.0)
        except Exception:
            pass

        def _mask_boundary(mask: np.ndarray) -> np.ndarray:
            """Return 1px boundary for a boolean mask (no scipy)."""
            m = mask.astype(bool)
            if m.size == 0:
                return np.zeros_like(m, dtype=bool)
            # pixel is interior if all 4-neighbors are true
            up = np.roll(m, 1, axis=0)
            dn = np.roll(m, -1, axis=0)
            lf = np.roll(m, 1, axis=1)
            rg = np.roll(m, -1, axis=1)
            interior = m & up & dn & lf & rg
            return m & (~interior)

        def _dilate1(mask: np.ndarray) -> np.ndarray:
            m = mask.astype(bool)
            if m.size == 0:
                return np.zeros_like(m, dtype=bool)
            return (
                m
                | np.roll(m, 1, axis=0)
                | np.roll(m, -1, axis=0)
                | np.roll(m, 1, axis=1)
                | np.roll(m, -1, axis=1)
            )

        # Overlay para poças/acúmulo: escurece apenas regiões realmente profundas
        if wet_vals.size > 12:
            # Máscara de acúmulo real: evita deixar tudo lilás/roxo
            accum_core_mask = valid_dem & ((acc_norm >= 0.72) | (pool_persistence >= 0.28))

            # Use percentis mais altos para não "pintar" quase tudo como acúmulo
            deep_thr = float(np.nanpercentile(wet_vals, 88))
            pool_thr = float(np.nanpercentile(wet_vals, 96))
            ultra_thr = float(np.nanpercentile(wet_vals, 99))

            # Exige separação mínima do limiar visível para não colapsar as classes
            deep_thr = float(max(deep_thr, current_vmin + 0.004))
            pool_thr = float(max(pool_thr, deep_thr + 0.004))
            ultra_thr = float(max(ultra_thr, pool_thr + 0.004))

            # Garante ordenação mínima entre níveis (evita níveis colapsados)
            deep_thr = float(min(deep_thr, vmax_eff - 1e-6))
            pool_thr = float(min(max(pool_thr, deep_thr + 1e-6), vmax_eff - 1e-6))
            ultra_thr = float(min(max(ultra_thr, pool_thr + 1e-6), vmax_eff - 1e-6))

            deep_cmap = LinearSegmentedColormap.from_list(
                "deep_navy",
                [(0.02, 0.30, 0.72), (0.00, 0.18, 0.52), (0.00, 0.08, 0.26)],
                N=32,
            )
            pool_cmap = LinearSegmentedColormap.from_list(
                "pool_core_navy",
                [(0.00, 0.18, 0.52), (0.00, 0.08, 0.26), (0.00, 0.03, 0.12)],
                N=16,
            )
            ultra_cmap = LinearSegmentedColormap.from_list(
                "pool_ultra_navy",
                [(0.00, 0.08, 0.26), (0.00, 0.01, 0.05)],
                N=8,
            )

            deep_mask = np.where((water_at_step >= deep_thr) & accum_core_mask, 1.0, np.nan)
            deep_alpha = np.clip((water_at_step - deep_thr) / (vmax_eff - deep_thr + 1e-6), 0.0, 1.0)
            ax.imshow(
                np.ma.masked_invalid(deep_mask),
                cmap=deep_cmap,
                alpha=0.12 + 0.28 * deep_alpha,
                zorder=4.4,
                interpolation='bilinear',
            )

            # Núcleo das poças (mais escuro) para leitura imediata de acúmulo
            pool_mask = np.where((water_at_step >= pool_thr) & accum_core_mask, 1.0, np.nan)
            pool_alpha = np.clip((water_at_step - pool_thr) / (vmax_eff - pool_thr + 1e-6), 0.0, 1.0)
            ax.imshow(
                np.ma.masked_invalid(pool_mask),
                cmap=pool_cmap,
                alpha=0.18 + 0.42 * pool_alpha,
                zorder=4.6,
                interpolation='bilinear',
            )

            # Ultra núcleo: topo do acúmulo com azul marinho quase sólido
            ultra_mask = np.where((water_at_step >= ultra_thr) & accum_core_mask, 1.0, np.nan)
            ultra_alpha = np.clip((water_at_step - ultra_thr) / (vmax_eff - ultra_thr + 1e-6), 0.0, 1.0)
            ax.imshow(
                np.ma.masked_invalid(ultra_mask),
                cmap=ultra_cmap,
                alpha=0.22 + 0.45 * ultra_alpha,
                zorder=4.75,
                interpolation='bilinear',
            )

            # Contornos de classes de acúmulo para leitura imediata (forte contraste)
            try:
                ax.contour(
                    np.ma.masked_where(~valid_dem, water_at_step),
                    levels=[deep_thr, pool_thr, ultra_thr],
                    colors=[(0.70, 0.95, 1.00, 0.92), (1.00, 0.94, 0.35, 0.98), (1.00, 0.78, 0.22, 1.00)],
                    linewidths=[0.6, 0.8, 1.0],
                    zorder=5.25,
                )
            except Exception:
                pass

            # Hotspots (amarelo/laranja) para destacar as maiores lâminas/poças
            try:
                # Hotspots bem seletivos (foco no topo das poças)
                hot_thr = float(np.nanpercentile(wet_vals, 97))
                if hot_thr < pool_thr:
                    hot_thr = pool_thr
                # Ensure hot_thr doesn't exceed vmax_eff to prevent normalize error
                hot_thr = min(float(hot_thr), float(vmax_eff - 1e-6))
                hot_mask = (water_at_step >= hot_thr) & valid_dem
                if np.any(hot_mask):
                    denom_h = max(1e-6, float(vmax_eff - hot_thr))
                    nd = np.clip((water_at_step - hot_thr) / denom_h, 0.0, 1.0)
                    hot_alpha = np.zeros_like(water_at_step, dtype=np.float32)
                    hot_alpha[hot_mask] = 0.22 + 0.55 * (nd[hot_mask] ** 0.35)
                    hot_alpha = np.clip(hot_alpha, 0.0, 0.85)
                    # Double-check vmin < vmax before creating norm
                    vmin_check = hot_thr + 1e-9
                    vmax_check = vmax_eff
                    if vmax_check <= vmin_check:
                        vmax_check = vmin_check + 1e-6
                    ax.imshow(
                        np.ma.masked_where(~hot_mask, water_at_step),
                        cmap=puddle_hot_cmap,
                        norm=PowerNorm(gamma=0.65, vmin=vmin_check, vmax=vmax_check),
                        alpha=hot_alpha,
                        zorder=5.35,
                        interpolation='bilinear',
                    )
            except Exception:
                pass

            # Persistência temporal: poças recorrentes ficam bem destacadas
            persistent_mask = (pool_persistence >= 0.45) & (water_at_step > vis_threshold) & valid_dem
            if np.any(persistent_mask):
                p_alpha = np.zeros_like(water_at_step, dtype=np.float32)
                p_alpha[persistent_mask] = 0.30
                ax.imshow(
                    np.ma.masked_where(~persistent_mask, pool_persistence),
                    cmap=LinearSegmentedColormap.from_list("pool_persist", [(0.02, 0.14, 0.42), (0.00, 0.02, 0.10)], N=32),
                    alpha=p_alpha,
                    zorder=4.9,
                    interpolation='bilinear',
                )

            # Realce de rota preferencial: mostra para onde a água tende a escoar
            route_mask = (water_at_step > threshold) & (acc_norm > 0.62) & valid_dem
            if np.any(route_mask):
                route_alpha = np.zeros_like(water_at_step, dtype=np.float32)
                route_alpha[route_mask] = 0.30 + 0.34 * np.clip(acc_norm[route_mask], 0.0, 1.0)
                route_alpha = np.clip(route_alpha, 0.0, 0.78)

                # halo claro por trás para aparecer melhor
                ax.imshow(
                    np.ma.masked_where(~route_mask, acc_norm),
                    cmap=LinearSegmentedColormap.from_list(
                        "route_halo",
                        [(1.00, 1.00, 1.00), (0.88, 0.98, 1.00)],
                        N=64,
                    ),
                    alpha=0.10 + 0.12 * np.clip(acc_norm, 0.0, 1.0),
                    zorder=5.08,
                    interpolation='bilinear',
                )
                ax.imshow(
                    np.ma.masked_where(~route_mask, acc_norm),
                    cmap=LinearSegmentedColormap.from_list("route_glow", [(0.60, 0.92, 1.00), (0.08, 0.58, 0.98)], N=64),
                    alpha=route_alpha,
                    zorder=5.12,
                    interpolation='bilinear',
                )

                # Linhas graduadas pela profundidade: quanto mais fundo, mais forte
                try:
                    rb = _mask_boundary(route_mask)
                    if np.any(rb):
                        rb_halo = _dilate1(rb)
                        halo_alpha = np.zeros_like(water_at_step, dtype=np.float32)
                        line_alpha = np.zeros_like(water_at_step, dtype=np.float32)
                        # Halo: aparece mesmo em lâmina moderada
                        halo_alpha[rb_halo] = np.clip(0.25 + 0.55 * (depth_norm_full[rb_halo] ** 0.65), 0.0, 0.90)
                        line_alpha[rb] = np.clip(0.35 + 0.65 * (depth_norm_full[rb] ** 0.55), 0.0, 0.98)

                        ax.imshow(
                            np.ma.masked_where(~rb_halo, np.ones_like(water_at_step, dtype=np.float32)),
                            cmap=LinearSegmentedColormap.from_list("route_line_halo", [(1, 1, 1), (1, 1, 1)], N=2),
                            alpha=halo_alpha,
                            zorder=5.55,
                            interpolation='nearest',
                        )
                        # Cyan-to-blue depth gradient for preferential flow route lines
                        route_line_cmap = LinearSegmentedColormap.from_list(
                            "route_depth_gradient",
                            [
                                (0.50, 0.95, 1.00),  # cyan claro (raso)
                                (0.25, 0.85, 1.00),  # cyan
                                (0.10, 0.70, 0.95),  # azul-cyan
                                (0.05, 0.55, 0.88),  # azul médio
                                (0.00, 0.35, 0.70),  # azul profundo
                            ],
                            N=128,
                        )
                        # Ensure vis_threshold < vmax_eff for normalization
                        route_vmin = min(float(vis_threshold), float(vmax_eff - 1e-6))
                        route_vmax = vmax_eff
                        if route_vmax <= route_vmin:
                            route_vmax = route_vmin + 1e-6
                        ax.imshow(
                            np.ma.masked_where(~rb, water_at_step),
                            cmap=route_line_cmap,
                            norm=Normalize(vmin=route_vmin, vmax=route_vmax),
                            alpha=line_alpha,
                            zorder=5.60,
                            interpolation='nearest',
                        )
                except Exception:
                    pass

            # Rede de caminhos pela acumulação (acc_norm), graduada pela profundidade
            try:
                acc_valid = acc_norm[np.isfinite(acc_norm) & valid_dem]
                if acc_valid.size > 50:
                    a80 = float(np.nanpercentile(acc_valid, 80))
                    a90 = float(np.nanpercentile(acc_valid, 90))
                    a96 = float(np.nanpercentile(acc_valid, 96))

                    # Só desenha rede onde há água visível (evita linhas em seco)
                    stream_base = valid_dem & np.isfinite(acc_norm) & (water_at_step >= vis_threshold)
                    m1 = stream_base & (acc_norm >= max(0.55, a80))
                    m2 = stream_base & (acc_norm >= max(0.65, a90))
                    m3 = stream_base & (acc_norm >= max(0.75, a96))

                    # Cmap para linhas de corrente: azul claro → azul escuro conforme profundidade
                    stream_line_cmap = LinearSegmentedColormap.from_list(
                        "stream_depth_gradient",
                        [
                            (0.50, 0.90, 1.00),  # azul muito claro (raso)
                            (0.20, 0.70, 1.00),  # azul claro
                            (0.08, 0.50, 0.90),  # azul médio
                            (0.00, 0.30, 0.70),  # azul profundo
                            (0.00, 0.15, 0.45),  # azul muito escuro
                        ],
                        N=128,
                    )

                    for m, w_halo, w_line, z0 in [
                        (m1, 0.18, 0.22, 5.44),
                        (m2, 0.26, 0.32, 5.46),
                        (m3, 0.34, 0.42, 5.48),
                    ]:
                        b = _mask_boundary(m)
                        if not np.any(b):
                            continue
                        b_halo = _dilate1(b)
                        ha = np.zeros_like(water_at_step, dtype=np.float32)
                        la = np.zeros_like(water_at_step, dtype=np.float32)
                        ha[b_halo] = np.clip(w_halo + 0.55 * (depth_norm_full[b_halo] ** 0.70), 0.0, 0.88)
                        la[b] = np.clip(w_line + 0.65 * (depth_norm_full[b] ** 0.55), 0.0, 0.98)

                        ax.imshow(
                            np.ma.masked_where(~b_halo, np.ones_like(water_at_step, dtype=np.float32)),
                            cmap=LinearSegmentedColormap.from_list("stream_halo", [(1, 1, 1), (1, 1, 1)], N=2),
                            alpha=ha,
                            zorder=z0,
                            interpolation='nearest',
                        )
                        # Render stream line with water depth colors from blue gradient
                        # Ensure vis_threshold < vmax_eff for normalization
                        stream_vmin = min(float(vis_threshold), float(vmax_eff - 1e-6))
                        stream_vmax = vmax_eff
                        if stream_vmax <= stream_vmin:
                            stream_vmax = stream_vmin + 1e-6
                        ax.imshow(
                            np.ma.masked_where(~b, water_at_step),
                            cmap=stream_line_cmap,
                            norm=Normalize(vmin=stream_vmin, vmax=stream_vmax),
                            alpha=la,
                            zorder=z0 + 0.02,
                            interpolation='nearest',
                        )
            except Exception:
                pass

            # Faixas preenchidas para leitura instantânea de acúmulo
            try:
                ax.contourf(
                    np.ma.masked_where(~valid_dem, water_at_step),
                    levels=[deep_thr, pool_thr, vmax_eff],
                    colors=[(0.00, 0.18, 0.52, 0.12), (0.00, 0.03, 0.12, 0.18)],
                    zorder=4.8,
                )
            except Exception:
                pass

        # Contorno de lâmina para destacar poças/acúmulo
        try:
            if np.nanmax(water_at_step) > vis_threshold:
                ax.contour(
                    water_at_step,
                    levels=[vis_threshold, max(vis_threshold * 1.45, vis_threshold + 0.008)],
                    colors=[(0.08, 0.78, 1.00, 0.80), (0.00, 0.36, 0.90, 0.95)],
                    linewidths=[0.5, 0.7],
                    zorder=5,
                )
                # Contorno adicional para núcleos de poças
                if wet_vals.size > 10:
                    core_lvl = float(np.nanpercentile(wet_vals, 93))
                    ax.contour(
                        water_at_step,
                        levels=[core_lvl],
                        colors=[(0.00, 0.03, 0.12, 1.00)],
                        linewidths=[0.8],
                        zorder=5.2,
                    )
        except Exception:
            pass

        # Fixar domínio visual para evitar "zoom" entre frames
        h_img, w_img = water_at_step.shape
        ax.set_xlim(-0.5, w_img - 0.5)
        ax.set_ylim(h_img - 0.5, -0.5)
        ax.set_autoscale_on(False)

        # ===== Chuva visual (bem sutil, para não escurecer a cena) =====
        if has_sources:
            rng = np.random.default_rng(frame_num + 2026)
            rain_pulse = 0.75 + 0.25 * np.sin(frame_num * 0.42)
            n_streaks = int(min(280, max(55, (source_ys.size // 24) * rain_pulse)))
            sel = rng.integers(0, source_ys.size, size=n_streaks)
            rx = source_xs[sel].astype(np.float32) + rng.uniform(-0.35, 0.35, size=n_streaks)
            ry = source_ys[sel].astype(np.float32) + rng.uniform(-0.35, 0.35, size=n_streaks)
            fall = (frame_num % 8) * 0.22
            y0 = ry - (1.00 + fall)
            y1 = ry + (0.22 + fall)
            for k in range(n_streaks):
                ax.plot(
                    [rx[k], rx[k]],
                    [y0[k], y1[k]],
                    color=(0.82, 0.95, 1.0, 0.16),
                    linewidth=0.55,
                    zorder=6,
                    solid_capstyle='round',
                    clip_on=True,
                )
        
        # Colorbar showing depth scale
        # Ensure vmax > vmin before creating Normalize
        if vmax_eff <= current_vmin:
            vmax_eff = current_vmin + 0.01
        sm = cm.ScalarMappable(cmap=water_cmap, norm=Normalize(vmin=current_vmin, vmax=vmax_eff))
        sm.set_array(water_display)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (m) / Profundidade (m)', fontsize=10, fontweight='bold')
        
        # Title
        time_min = model.history[step_idx].get("time_minutes", frame_num * 10)
        progress_pct = int((frame_num / max(1, total_steps)) * 100)
        ax.set_title(
            f'Rainfall + Flow + Terrain Accumulation (Chuva + Escoamento + Acúmulo) | Time (Tempo): {time_min:.1f} min | Progress (Progresso): {progress_pct}%',
            fontsize=13, fontweight='bold', pad=15
        )
        
        # Statistics
        flooded_cells = int(((water_at_step > max(0.0015, 0.5 * threshold)) & valid_dem).sum())
        max_depth = float(np.nanmax(water_at_step)) if np.isfinite(water_at_step).any() else 0.0
        volume_m3 = float(water_at_step.sum() * 625)  # 25x25 cell
        
        stats_text = (
            f'Max Depth (Prof. Máx): {max_depth:.3f} m\n'
            f'Flooded Cells (Células Inundadas): {flooded_cells}\n'
            f'Volume: {volume_m3/1e6:.3f} Mm³'
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9,
                     edgecolor='darkblue', linewidth=2)
        )

        legend_handles = [
            Patch(facecolor=(0.70, 0.95, 1.00), edgecolor='none', label='Shallow flow'),
            Patch(facecolor=(0.18, 0.64, 0.96), edgecolor='none', label='Moderate depth'),
            Patch(facecolor=(0.02, 0.24, 0.60), edgecolor='none', label='Pools/accumulation'),
            Line2D([0], [0], color=(0.98, 0.88, 0.24), lw=2.0, label='Accumulation contour'),
            Line2D([0], [0], color=(0.82, 0.95, 1.0), lw=1.2, alpha=0.6, label='Rain visual'),
        ]
        ax.legend(
            handles=legend_handles,
            loc='lower right',
            fontsize=8,
            framealpha=0.93,
            title='Frame legend (Legenda do quadro)',
            title_fontsize=8,
        )

        ax.text(
            0.02,
            0.02,
            'Quick read (Leitura rápida): light blue = flow | dark blue = accumulation | yellow lines = depth classes',
            transform=ax.transAxes,
            fontsize=8.5,
            color='#1b1b1b',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.86, edgecolor='#7a7a7a', linewidth=1.1),
            va='bottom',
        )
        
        ax.set_xlabel('X (cells / células)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cells / células)', fontsize=10, fontweight='bold')
        ax.grid(False)

        # Layout fixo para evitar variação de escala/posição entre frames
        fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.08)
        
        # Render to PIL image (using modern matplotlib API)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get buffer RGBA
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:,:,:3]  # Drop alpha, keep RGB
        frames.append(Image.fromarray(img_array, 'RGB'))
        
        fig.clear()
        plt.close(fig)
        
        if (frame_num + 1) % 5 == 0:
            logger.info(f"Frame {frame_num + 1}/{total_steps} generated")

        prev_water_snapshot = water_at_step.copy()
    
    if not frames:
        logger.error("No frames generated")
        return
    
    # Duração total alvo: mais lenta para melhorar leitura da dinâmica da água
    total_duration_ms = 40_000
    per_frame_ms = max(90, int(round(total_duration_ms / max(1, len(frames)))))

    output_path = Path("outputs/test_run/animation.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=per_frame_ms,
        loop=0,
        optimize=False,
    )
    logger.info(
        f"✅ Animation saved: {output_path} ({len(frames)} frames, {total_steps} total steps, "
        f"~{(len(frames) * per_frame_ms) / 1000:.1f}s)"
    )
    return output_path

def _default_georef_for_array(arr: np.ndarray, cell_size_meters: float = 25.0):
    height, width = arr.shape
    transform = from_bounds(0, 0, width * cell_size_meters, height * cell_size_meters, width, height)
    crs = 'EPSG:4326'
    return transform, crs


def _save_upload_to_temp(file_storage) -> Path:
    suffix = Path(file_storage.filename or '').suffix
    tmp = tempfile.NamedTemporaryFile(prefix='hydrosim_upload_', suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    file_storage.save(tmp_path)
    return tmp_path


def _load_dem_upload(file_storage, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, Any, Any]:
    """Load DEM from uploaded GeoTIFF with validation and caching."""
    from rasterio.enums import Resampling  # type: ignore
    import time

    tmp_path = _save_upload_to_temp(file_storage)
    start_time = time.time()
    
    try:
        # Validate GeoTIFF before processing
        is_valid, validation_msg = validate_geotiff(str(tmp_path), max_size_mb=500.0)
        if not is_valid:
            raise ValidationError(f"Invalid GeoTIFF: {validation_msg}")
        
        logger.info(f"✅ Validation passed: {validation_msg}")
        
        # Generate cache key
        cache_key = CacheManager.get_cache_key(tmp_path.name, {"shape": target_shape})
        
        # Try to load from cache
        cached = CacheManager.load_from_cache(cache_key)
        if cached:
            dem_arr, metadata = cached
            logger.info(f"Loaded from cache: {metadata}")
            return dem_arr, metadata.get("transform"), metadata.get("crs")
        
        with rasterio.open(tmp_path) as src:
            target_h, target_w = int(target_shape[0]), int(target_shape[1])
            dem_arr = src.read(
                1,
                out_shape=(target_h, target_w),
                resampling=Resampling.bilinear,
            )

            nodata = src.nodata
            dem_arr = dem_arr.astype(np.float32, copy=False)
            if nodata is not None:
                dem_arr = np.where(dem_arr == nodata, np.nan, dem_arr)
            # Preservar NaN/NoData para que áreas fora do DEM possam ser
            # exibidas em branco nas visualizações.

            # Salva máscara original do domínio válido (antes de preencher NoData)
            # para evitar água fora do terreno no GIF.
            try:
                original_valid_mask = np.isfinite(dem_arr)
                Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
                np.save("outputs/test_run/dem_valid_mask.npy", original_valid_mask.astype(np.uint8))
            except Exception:
                pass

            # Se o DEM veio com bordas NoData (conteúdo parcial), preenche para
            # estabilizar a simulação e evitar DEM "desconfigurado" no painel.
            valid_mask = np.isfinite(dem_arr)
            valid_ratio = float(np.count_nonzero(valid_mask) / dem_arr.size) if dem_arr.size else 0.0
            if valid_ratio > 0.0 and valid_ratio < 0.85:
                fill_val = float(np.nanmedian(dem_arr[valid_mask]))
                dem_arr = np.where(valid_mask, dem_arr, fill_val).astype(np.float32)
                logger.info(f"DEM had low valid coverage ({valid_ratio:.2%}); filled NoData to stabilize dashboard rendering")

            # Scale transform to the requested out_shape
            scale_x = src.width / float(target_w)
            scale_y = src.height / float(target_h)
            out_transform = src.transform * src.transform.scale(scale_x, scale_y)
            out_crs = ensure_valid_crs(src.crs)
            
            # Validate DEM values
            is_valid, stats = validate_dem_values(dem_arr)
            logger.info(f"DEM Statistics: min={stats['min']:.2f}m, max={stats['max']:.2f}m, "
                       f"mean={stats['mean']:.2f}m, std={stats['std']:.2f}m")
            
            # Cache the result
            metadata = {"transform": out_transform, "crs": out_crs}
            CacheManager.save_to_cache(dem_arr, cache_key, metadata)
            
            elapsed = time.time() - start_time
            EnhancedLogging.log_performance("DEM Load & Resample", elapsed, f"{target_h}x{target_w}")
            
            return dem_arr, out_transform, out_crs
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading DEM: {e}")
        raise
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _save_geotiff(dem, water, transform=None, crs=None):
    """Save DEM and water depth as GeoTIFF with compression."""

    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)

    height, width = dem.shape
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(dem)

    # Save DEM with compression
    dem_path = Path("outputs/test_run/dem.tif")
    with rasterio.open(
        dem_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=np.float32,
        transform=transform,
        crs=crs,
        compress='deflate',  # Added compression
        compress_level=9,
        nodata=np.nan,
    ) as dst:
        dst.write(dem.astype(np.float32, copy=False), 1)

    # Save Water depth REAL (float32) para análise hidrológica em SIG
    threshold_m = 0.01
    water_f = water.astype(np.float32, copy=False)
    wet = water_f > float(threshold_m)
    
    if np.any(wet):
        vmax = float(np.nanpercentile(water_f[wet], 99))
    else:
        vmax = float(np.nanmax(water_f)) if np.isfinite(water_f).any() else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    water_out = np.where(np.isfinite(water_f), np.clip(water_f, 0.0, None), np.nan).astype(np.float32)

    water_path = Path("outputs/test_run/water_depth.tif")
    with rasterio.open(
        water_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1,
        dtype=np.float32,
        transform=transform,
        crs=crs,
        compress='deflate',  # Added compression
        compress_level=9,
        nodata=0.0,
    ) as dst:
        dst.write(water_out, 1)

    logger.info(f"✅ GeoTIFFs saved with compression: {dem_path}, {water_path}")


def _save_water_rgba_geotiff(water: np.ndarray, transform=None, crs=None, threshold_m: float = 0.001, dem: Optional[np.ndarray] = None):
    """Save water depth as RGBA GeoTIFF (uint8) with alpha and custom water colormap.

    Output name: outputs/test_run/lamina_agua_rgba.tif
    Uses custom water colormap (light cyan → dark blue) for better flood visualization.
    Alpha channel increases with depth for better visual impact.
    """
    from rasterio.enums import ColorInterp  # type: ignore
    import matplotlib.colors as mcolors

    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/test_run/lamina_agua_rgba.tif")

    water_f = np.asarray(water, dtype=np.float32)
    water_f = np.where(np.isfinite(water_f), np.clip(water_f, 0.0, None), np.nan).astype(np.float32)

    # Modulação topográfica opcional para evitar lâmina visual uniforme
    water_vis = water_f.copy()
    if dem is not None and dem.shape == water_f.shape:
        try:
            dem_f = np.asarray(dem, dtype=np.float32)
            valid_dem = np.isfinite(dem_f)
            if np.any(valid_dem):
                dem_fill = np.where(valid_dem, dem_f, np.nanmedian(dem_f[valid_dem]))
                gy, gx = np.gradient(dem_fill)
                slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
                svals = slope[valid_dem]
                if svals.size > 0:
                    s10, s90 = np.nanpercentile(svals, (10, 90))
                    low_slope = np.clip(1.0 - (slope - s10) / (s90 - s10 + 1e-6), 0.0, 1.0)
                else:
                    low_slope = np.zeros_like(slope, dtype=np.float32)

                flow_dir = _calculate_flow_direction(dem_fill)
                flow_acc = _calculate_flow_accumulation(flow_dir, dem_fill).astype(np.float32)
                acc_log = np.log1p(np.clip(flow_acc, 0.0, None))
                avals = acc_log[valid_dem]
                if avals.size > 0:
                    a30, a99 = np.nanpercentile(avals, (30, 99))
                    acc_norm = np.clip((acc_log - a30) / (a99 - a30 + 1e-6), 0.0, 1.0)
                else:
                    acc_norm = np.zeros_like(acc_log, dtype=np.float32)

                topo_pref = np.clip(0.55 * low_slope + 0.45 * acc_norm, 0.0, 1.0)
                water_vis = water_f * (0.55 + 0.95 * topo_pref)
        except Exception:
            pass

    # CORREÇÃO: Usar valores REAIS de água (water_f) para normalização, não modulados (water_vis)
    # Isso garante que diferentes chuvas produzam diferentes visualizações
    positive = np.isfinite(water_f) & (water_f > 0.0)
    # Use GLOBAL fixed scales to preserve difference between rainfall amounts
    global_vmin = 0.001  # 1mm threshold
    global_vmax = 0.50   # 500mm reference scale
    vmin_eff = float(max(global_vmin, threshold_m))
    vmax_eff = float(global_vmax)
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(water_f)

    # Gradiente calibrado para o estilo do lami.png (azul claro -> azul escuro)
    water_cmap = mcolors.LinearSegmentedColormap.from_list(
        "water_grad_export",
        [
            (0.76, 0.90, 1.00),
            (0.58, 0.80, 0.99),
            (0.34, 0.68, 0.98),
            (0.14, 0.56, 0.95),
            (0.03, 0.42, 0.88),
            (0.01, 0.30, 0.74),
            (0.00, 0.18, 0.54),
        ],
        N=256,
    )
    water_cmap = water_cmap.copy()
    water_cmap.set_under((0, 0, 0, 0.0))
    water_cmap.set_bad((0, 0, 0, 0.0))

    # Limiar visual adaptativo - baseado em valores REAIS (water_f), não modulados
    if np.any(positive):
        p30 = float(np.nanpercentile(water_f[positive], 30.0))
        p85 = float(np.nanpercentile(water_f[positive], 85.0))
        vis_threshold = max(vmin_eff, p30)
        # Evita imagem em branco quando a distribuição é muito estreita
        if (p85 - p30) < 1e-4:
            vis_threshold = vmin_eff
        else:
            above_vis = int(np.count_nonzero(np.isfinite(water_f) & (water_f > vis_threshold)))
            if above_vis < max(20, int(0.0015 * water_f.size)):
                vis_threshold = vmin_eff
    else:
        vis_threshold = vmin_eff

    # Curva para realçar contraste - normalização baseada em valores reais
    norm = mcolors.PowerNorm(gamma=0.95, vmin=vis_threshold, vmax=vmax_eff, clip=True)
    
    # Aplicar colormap a TODOS os valores (vai interpolar gradiente)
    normalized_vals = norm(water_f)
    rgba_f = water_cmap(normalized_vals)  # Aplica colormap com interpolação
    rgba_f = np.clip(rgba_f, 0.0, 1.0)
    
    # Alpha seletivo: raso quase transparente; profundo opaco
    depth_norm = np.clip((water_f - vis_threshold) / (vmax_eff - vis_threshold + 1e-6), 0.0, 1.0)
    alpha = np.zeros_like(water_f, dtype=np.float32)
    wet = np.isfinite(water_f) & (water_f > vis_threshold)
    alpha_curve = np.clip((depth_norm - 0.10) / 0.90, 0.0, 1.0)
    alpha[wet] = 0.10 + 0.64 * (alpha_curve[wet] ** 1.20)
    
    # Zeros onde não tem água
    alpha[~wet] = 0.0
    rgba_f[..., 3] = alpha

    rgba = (rgba_f * 255).astype(np.uint8)
    data = np.transpose(rgba, (2, 0, 1))  # (4, H, W)

    height, width = water_f.shape
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=4,
        dtype=np.uint8,
        transform=transform,
        crs=crs,
        compress='deflate',
        compress_level=9,
    ) as dst:
        dst.write(data)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )

    wet_count = int(np.count_nonzero(np.isfinite(water_f) & (water_f > vmin_eff)))
    logger.info(f"✅ RGBA water GeoTIFF saved: {out_path} | wet_pixels={wet_count} | vmin={vmin_eff:.6f} m | vvis={vis_threshold:.6f} m | vmax={vmax_eff:.6f} m")
    return out_path


def _water_over_terrain_geotiff_bytes(dem: np.ndarray, water: np.ndarray, transform=None, crs=None, threshold_m: float = 0.005):
    """Generate RGB GeoTIFF bytes with water depth over terrain (GIF-like style) for QGIS."""
    from rasterio.io import MemoryFile
    import matplotlib.colors as mcolors
    from matplotlib.colors import LightSource, PowerNorm

    dem_f = np.asarray(dem, dtype=np.float32)
    water_f = np.asarray(water, dtype=np.float32)
    water_f = np.where(np.isfinite(water_f), np.clip(water_f, 0.0, None), np.nan).astype(np.float32)

    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(dem_f)

    valid_dem = np.isfinite(dem_f)
    dem_vals = dem_f[valid_dem]
    if dem_vals.size == 0:
        dem_fill = np.zeros_like(dem_f, dtype=np.float32)
        dem_norm = np.zeros_like(dem_f, dtype=np.float32)
    else:
        d2, d98 = np.nanpercentile(dem_vals, (2, 98))
        dem_fill = np.where(valid_dem, dem_f, np.nanmedian(dem_vals)).astype(np.float32)
        dem_norm = np.clip((dem_fill - d2) / (d98 - d2 + 1e-6), 0.0, 1.0)

    dem_cmap = mcolors.LinearSegmentedColormap.from_list(
        "dem_no_blue_qgis",
        [
            (0.95, 0.94, 0.86),
            (0.88, 0.82, 0.64),
            (0.74, 0.67, 0.49),
            (0.58, 0.51, 0.36),
            (0.52, 0.45, 0.34),
            (0.70, 0.70, 0.70),
        ],
        N=256,
    )
    terrain = np.array(dem_cmap(dem_norm), dtype=np.float32)
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(dem_fill, vert_exag=1.45, dx=1.0, dy=1.0).astype(np.float32)
    terrain[..., :3] = np.clip(terrain[..., :3] * (0.74 + 0.26 * shade[..., None]), 0.0, 1.0)
    terrain[~valid_dem, :3] = 1.0

    positive = np.isfinite(water_f) & (water_f > 0.0) & valid_dem
    if np.any(positive):
        max_pos = float(np.nanmax(water_f[positive]))
    else:
        max_pos = 0.0
    
    # CORREÇÃO: Usar escala GLOBAL fixa, não normalizada por simulação
    # Permite comparar visualmente simulações diferentes
    global_vmin = 0.001  # 1mm de threshold
    global_vmax = 0.50   # Escala máxima de referência (500mm de chuva)
    
    vvis = global_vmin
    vmax_eff = max(max_pos, global_vmax)  # Respeita valores maiores que max se ocorrerem

    wet = np.isfinite(water_f) & (water_f > vvis) & valid_dem
    water_cmap = mcolors.LinearSegmentedColormap.from_list(
        "water_on_terrain",
        [
            (0.76, 0.92, 1.00),
            (0.52, 0.83, 1.00),
            (0.18, 0.69, 0.99),
            (0.02, 0.50, 0.92),
            (0.00, 0.30, 0.74),
            (0.00, 0.12, 0.46),
        ],
        N=256,
    )
    
    # Renderizar água com cores azuis do colormap
    norm = PowerNorm(gamma=0.92, vmin=vvis, vmax=vmax_eff, clip=True)
    normalized_water = norm(water_f)  # valores 0-1
    rgba_water = water_cmap(normalized_water)  # aplica colormap = RGBA
    water_rgb = np.array(rgba_water)[..., :3]  # pega só RGB (azul claro -> azul escuro)

    depth_norm = np.zeros_like(water_f, dtype=np.float32)
    if np.any(wet):
        depth_norm[wet] = np.clip((water_f[wet] - vvis) / (vmax_eff - vvis + 1e-6), 0.0, 1.0)
    
    # Mostrar APENAS a água (colormap azul), NÃO o terreno
    # Isso vai sair: sem água = branco/claro, com água = azul variado
    out_rgb = water_rgb.copy()

    data = (out_rgb * 255.0).astype(np.uint8)
    data = np.transpose(data, (2, 0, 1))
    height, width = dem_f.shape

    with MemoryFile() as mem:
        with mem.open(
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            transform=transform,
            crs=crs,
            compress='deflate',
            compress_level=9,
        ) as dst:
            dst.write(data)
        return mem.read()


def _water_rgba_geotiff_bytes(water: np.ndarray, transform=None, crs=None, vmin: float = 0.001, vmax: Optional[float] = None, threshold_m: float = 0.001, cmap=None, dem: Optional[np.ndarray] = None):
    """Generate RGBA GeoTIFF bytes for water depth with custom colormap.
    
    Returns bytes suitable for download (MemoryFile).
    Used by Streamlit interface for download_button.
    """
    from io import BytesIO
    from rasterio.enums import ColorInterp  # type: ignore
    from rasterio.io import MemoryFile
    import matplotlib.colors as mcolors

    water_f = np.asarray(water, dtype=np.float32)
    water_f = np.where(np.isfinite(water_f), np.clip(water_f, 0.0, None), np.nan).astype(np.float32)

    water_vis = water_f.copy()
    if dem is not None and dem.shape == water_f.shape:
        try:
            dem_f = np.asarray(dem, dtype=np.float32)
            valid_dem = np.isfinite(dem_f)
            if np.any(valid_dem):
                dem_fill = np.where(valid_dem, dem_f, np.nanmedian(dem_f[valid_dem]))
                gy, gx = np.gradient(dem_fill)
                slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
                svals = slope[valid_dem]
                if svals.size > 0:
                    s10, s90 = np.nanpercentile(svals, (10, 90))
                    low_slope = np.clip(1.0 - (slope - s10) / (s90 - s10 + 1e-6), 0.0, 1.0)
                else:
                    low_slope = np.zeros_like(slope, dtype=np.float32)

                flow_dir = _calculate_flow_direction(dem_fill)
                flow_acc = _calculate_flow_accumulation(flow_dir, dem_fill).astype(np.float32)
                acc_log = np.log1p(np.clip(flow_acc, 0.0, None))
                avals = acc_log[valid_dem]
                if avals.size > 0:
                    a30, a99 = np.nanpercentile(avals, (30, 99))
                    acc_norm = np.clip((acc_log - a30) / (a99 - a30 + 1e-6), 0.0, 1.0)
                else:
                    acc_norm = np.zeros_like(acc_log, dtype=np.float32)

                topo_pref = np.clip(0.55 * low_slope + 0.45 * acc_norm, 0.0, 1.0)
                water_vis = water_f * (0.55 + 0.95 * topo_pref)
        except Exception:
            pass

    # CORREÇÃO: Usar valores REAIS de água (water_f) para normalização, não modulados (water_vis)
    positive = np.isfinite(water_f) & (water_f > 0.0)
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(water_f)

    # vmin efetivo com fallback para não zerar tudo quando o campo é muito raso
    vmin_eff = max(1e-6, float(vmin), float(threshold_m))

    # Calcular vmax se não fornecido - usar GLOBAL fixed scales
    if vmax is None:
        # Use GLOBAL fixed scales to preserve difference between rainfall amounts
        vmax_eff = 0.50  # 500mm reference scale
    else:
        vmax_eff = max(vmin_eff + 1e-6, float(vmax))

    # Colormap padrão no estilo do lami.png (se não informado)
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "water_grad_export",
            [
                (0.76, 0.90, 1.00),
                (0.58, 0.80, 0.99),
                (0.34, 0.68, 0.98),
                (0.14, 0.56, 0.95),
                (0.03, 0.42, 0.88),
                (0.01, 0.30, 0.74),
                (0.00, 0.18, 0.54),
            ],
            N=256,
        )

    cmap = cmap.copy()
    cmap.set_under((0, 0, 0, 0.0))
    cmap.set_bad((0, 0, 0, 0.0))

    # Limiar visual adaptativo - baseado em valores REAIS (water_f), não modulados
    if np.any(positive):
        p30 = float(np.nanpercentile(water_f[positive], 30.0))
        p85 = float(np.nanpercentile(water_f[positive], 85.0))
        vis_threshold = max(vmin_eff, p30)
        if (p85 - p30) < 1e-4:
            vis_threshold = vmin_eff
        else:
            above_vis = int(np.count_nonzero(np.isfinite(water_f) & (water_f > vis_threshold)))
            if above_vis < max(20, int(0.0015 * water_f.size)):
                vis_threshold = vmin_eff
    else:
        vis_threshold = vmin_eff

    norm = mcolors.PowerNorm(gamma=0.95, vmin=vis_threshold, vmax=vmax_eff, clip=True)
    
    # Aplicar colormap a TODOS os valores (vai interpolar gradiente)
    normalized_vals = norm(water_f)
    rgba_f = cmap(normalized_vals)  # Aplica colormap com interpolação
    rgba_f = np.clip(rgba_f, 0.0, 1.0)
    
    # Alpha seletivo: raso quase transparente; profundo opaco
    depth_norm = np.clip((water_f - vis_threshold) / (vmax_eff - vis_threshold + 1e-6), 0.0, 1.0)
    alpha = np.zeros_like(water_f, dtype=np.float32)
    wet = np.isfinite(water_f) & (water_f > vis_threshold)
    alpha_curve = np.clip((depth_norm - 0.10) / 0.90, 0.0, 1.0)
    alpha[wet] = 0.10 + 0.64 * (alpha_curve[wet] ** 1.20)
    
    # Zeros onde não tem água
    alpha[~wet] = 0.0
    rgba_f[..., 3] = alpha

    rgba = (rgba_f * 255).astype(np.uint8)
    data = np.transpose(rgba, (2, 0, 1))  # (4, H, W)

    height, width = water_f.shape
    
    # Gerar em memória
    with MemoryFile() as mem:
        with mem.open(
            driver='GTiff',
            height=height,
            width=width,
            count=4,
            dtype=np.uint8,
            transform=transform,
            crs=crs,
            compress='deflate',
            compress_level=9,
        ) as dst:
            dst.write(data)
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
        
        return mem.read()


def _save_inundation_gpkg(water: np.ndarray, transform=None, crs=None, threshold_m: float = 0.01, dem: Optional[np.ndarray] = None):
    """Vectorize flooded cells and preferential flow paths into GPKG.

    Output name: outputs/test_run/fluxo_preferencial.gpkg
    Layers: 
      - inundacao: flooded polygons
      - fluxo_preferencial: flow direction lines
    """
    from rasterio.features import shapes as rio_shapes  # type: ignore
    from rasterio.transform import xy as xy_from_transform  # type: ignore
    from shapely.geometry import LineString  # type: ignore

    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/test_run/fluxo_preferencial.gpkg")

    water_f = water.astype(np.float32, copy=False)
    flood_threshold_val = float(threshold_m)

    # 1) Máscara binária de inundação (água > limiar)
    try:
        mask = (water_f > flood_threshold_val).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Could not create flood mask: {e}")
        return None

    flooded = mask == 1
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(water_f)

    if not np.any(flooded):
        logger.warning("⚠️ No flooded cells above threshold; skipping GPKG export")
        return None

    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import shape  # type: ignore
        from shapely.ops import unary_union  # type: ignore

        # 2) Polygonize raster mask -> geometrias vetoriais
        shapes_gen = rio_shapes(mask, transform=transform)
        geoms = []
        for geom, val in shapes_gen:
            try:
                if int(val) == 1:
                    geoms.append(shape(geom))
            except Exception:
                continue
        
        # Layer 2: Preferential flow paths (if DEM provided)
        flow_lines = []
        if dem is not None and len(geoms) > 0:
            flow_direction = _calculate_flow_direction(dem)
            flow_acc = _calculate_flow_accumulation(flow_direction, dem).astype(np.float32)
            height, width = dem.shape
            
            # Define 8 directions and their offsets
            directions = {
                1: ((-1, 1), "NE"),   # NE
                2: ((-1, 0), "N"),    # N
                3: ((-1, -1), "NW"),  # NW
                8: ((0, 1), "E"),     # E
                4: ((0, -1), "W"),    # W
                7: ((1, 1), "SE"),    # SE
                6: ((1, 0), "S"),     # S
                5: ((1, -1), "SW"),   # SW
            }
            
            # Sementes apenas em áreas com maior acumulação para evitar excesso de linhas
            seed_mask = (flow_direction > 0) & flooded & np.isfinite(flow_acc)
            acc_seed_vals = flow_acc[seed_mask]
            if acc_seed_vals.size > 0:
                seed_thr = float(np.nanpercentile(acc_seed_vals, 88))
            else:
                seed_thr = 0.0
            seed_cells = np.argwhere(seed_mask & (flow_acc >= seed_thr))

            # fallback para não zerar se houver poucos pontos acima do limiar
            if seed_cells.size == 0:
                seed_cells = np.argwhere((flow_direction > 0) & flooded)

            if seed_cells.size > 0:
                order = np.argsort(flow_acc[seed_cells[:, 0], seed_cells[:, 1]])[::-1]
                seed_cells = seed_cells[order]

            # controle de sobreposição: evita desenhar muitas linhas quase iguais
            occupied = np.zeros((height, width), dtype=np.uint8)
            max_lines = 420

            for i, j in seed_cells:
                if len(flow_lines) >= max_lines:
                    break

                if occupied[i, j] >= 2:
                    continue

                path = [(int(i), int(j))]
                ci, cj = int(i), int(j)

                # Follow flow downstream
                for _ in range(height * width):  # Prevent infinite loops
                    dir_code = flow_direction[ci, cj]
                    if dir_code == 0 or dir_code not in directions:
                        break

                    di, dj = directions[dir_code][0]
                    ni, nj = ci + di, cj + dj

                    if not (0 <= ni < height and 0 <= nj < width):
                        break
                    if not flooded[ni, nj]:
                        break
                    if (ni, nj) == (ci, cj):
                        break

                    # corta ramificações muito redundantes
                    if occupied[ni, nj] >= 2 and len(path) >= 6:
                        break

                    path.append((int(ni), int(nj)))
                    ci, cj = int(ni), int(nj)

                # Mantém somente linhas com comprimento útil
                if len(path) >= 6:
                    # simplificação: reduz zigue-zague e densidade de vértices
                    simplified = path[::2]
                    if simplified[-1] != path[-1]:
                        simplified.append(path[-1])

                    coords = [xy_from_transform(transform, p[0], p[1]) for p in simplified]
                    flow_lines.append(LineString(coords))

                    # marca vizinhança para espaçar linhas próximas
                    for pi, pj in simplified:
                        r0, r1 = max(0, pi - 1), min(height, pi + 2)
                        c0, c1 = max(0, pj - 1), min(width, pj + 2)
                        occupied[r0:r1, c0:c1] = np.clip(occupied[r0:r1, c0:c1] + 1, 0, 3)
            
            if flow_lines:
                logger.info(f"✅ Generated {len(flow_lines)} preferential flow paths")

        # Save to GPKG
        if out_path.exists():
            out_path.unlink(missing_ok=True)

        # Write inundation layer
        if geoms:
            dissolved = unary_union(geoms)
            if dissolved.geom_type == 'MultiPolygon':
                final_geoms = list(dissolved.geoms)
            else:
                final_geoms = [dissolved]
            
            gdf_inund = gpd.GeoDataFrame({'geometry': final_geoms}, crs=crs)  # type: ignore[arg-type]
            gdf_inund.to_file(out_path, layer='inundacao', driver='GPKG')

        # Write flow paths layer
        if flow_lines:
            gdf_flow = gpd.GeoDataFrame({'geometry': flow_lines}, crs=crs)  # type: ignore[arg-type]
            gdf_flow.to_file(out_path, layer='fluxo_preferencial', driver='GPKG')

        file_size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
        logger.info(f"✅ GPKG saved: {out_path} ({file_size_kb:.1f} KB)")
        return out_path
    except Exception as e:
        logger.warning(f"Could not export GPKG (missing deps or error): {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None

def _calculate_flow_direction(dem):
    """Calculate preferential flow direction using steepest descent."""
    logger.info("Calculating preferential water flow direction...")
    
    # Flow direction: 1-8 direction encoding (D8 algorithm)
    # 1-8 encoded as follows (like QGIS):
    # 3 2 1
    # 4 x 8
    # 5 6 7
    
    height, width = dem.shape
    flow_direction = np.zeros_like(dem, dtype=np.uint8)
    
    # Limiar mínimo de declividade para evitar mapa muito poluído em áreas planas
    slope_eps = 0.02

    # For each cell, find steepest descent direction
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Get elevation differences to 8 neighbors
            center = dem[i, j]
            
            # Define 8 directions (and corresponding flow codes)
            directions = [
                (i-1, j+1, 1),  # NE
                (i-1, j,   2),  # N
                (i-1, j-1, 3),  # NW
                (i,   j+1, 8),  # E
                (i,   j-1, 4),  # W
                (i+1, j+1, 7),  # SE
                (i+1, j,   6),  # S
                (i+1, j-1, 5),  # SW
            ]
            
            max_slope = 0
            steepest_dir = 0
            
            for ni, nj, dir_code in directions:
                if 0 <= ni < height and 0 <= nj < width:
                    slope = (center - dem[ni, nj]) / (1.0 if abs(ni-i) + abs(nj-j) == 1 else 1.414)
                    if slope > max_slope:
                        max_slope = slope
                        steepest_dir = dir_code
            
            flow_direction[i, j] = steepest_dir if max_slope > slope_eps else 0
    
    return flow_direction


def _load_orthomosaic_upload(file_storage, target_shape):
    """Load an orthomosaic from upload (GeoTIFF or PNG/JPG) and resample to target_shape.

    Returns: (rgb_float32, transform_or_None, crs_or_None)
      - rgb_float32: shape (H, W, 3) in [0,1]
      - transform/crs: only available for GeoTIFF inputs
    """
    from PIL import Image
    from rasterio.enums import Resampling  # type: ignore

    filename = (file_storage.filename or '').lower()
    target_h, target_w = int(target_shape[0]), int(target_shape[1])

    if filename.endswith(('.tif', '.tiff', '.geotiff')):
        tmp_path = _save_upload_to_temp(file_storage)
        try:
            with rasterio.open(tmp_path) as src:
                count = int(src.count)
                if count >= 3:
                    rgb = src.read(
                        indexes=[1, 2, 3],
                        out_shape=(3, target_h, target_w),
                        resampling=Resampling.bilinear,
                    )
                    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32, copy=False)
                else:
                    band = src.read(
                        1,
                        out_shape=(target_h, target_w),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32, copy=False)
                    rgb = np.repeat(band[..., None], 3, axis=2)

                vmin = float(np.nanpercentile(rgb, 2))
                vmax = float(np.nanpercentile(rgb, 98))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                rgb = np.clip((rgb - vmin) / (vmax - vmin), 0.0, 1.0)

                scale_x = src.width / float(target_w)
                scale_y = src.height / float(target_h)
                out_transform = src.transform * src.transform.scale(scale_x, scale_y)
                out_crs = src.crs or 'EPSG:4326'
                return rgb, out_transform, out_crs
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Fallback: raster image formats
    img = Image.open(file_storage.stream).convert('RGB')
    img = img.resize((target_w, target_h))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0), None, None


def _save_orthomosaic_products(ortho_rgb: np.ndarray, transform=None, crs=None):
    """Save orthomosaic preview as PNG and as GeoTIFF (RGB)."""
    from PIL import Image

    Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
    png_path = Path("outputs/test_run/orthomosaic.png")
    Image.fromarray((np.clip(ortho_rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(png_path)

    # Save GeoTIFF (3 bands)
    tif_path = Path("outputs/test_run/orthomosaic.tif")
    height, width, _ = ortho_rgb.shape
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(ortho_rgb[..., 0])
    data = (np.clip(ortho_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    data = np.transpose(data, (2, 0, 1))  # (3, H, W)
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(data)

    return png_path, tif_path


def _calculate_flow_accumulation(flow_direction: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Compute D8 flow accumulation (number of upstream cells)."""
    logger.info("Calculating flow accumulation...")

    height, width = flow_direction.shape
    acc = np.ones((height, width), dtype=np.float32)

    # Downstream offsets for our D8 encoding
    offsets = {
        1: (-1, 1),   # NE
        2: (-1, 0),   # N
        3: (-1, -1),  # NW
        4: (0, -1),   # W
        5: (1, -1),   # SW
        6: (1, 0),    # S
        7: (1, 1),    # SE
        8: (0, 1),    # E
    }

    # Process cells from high to low elevation so upstream adds to downstream
    flat_idx = np.argsort(dem.ravel())[::-1]
    for idx in flat_idx:
        i = int(idx // width)
        j = int(idx % width)
        d = int(flow_direction[i, j])
        if d == 0:
            continue
        di, dj = offsets.get(d, (0, 0))
        ni, nj = i + di, j + dj
        if 0 <= ni < height and 0 <= nj < width:
            acc[ni, nj] += acc[i, j]

    return acc

def _calculate_flow_accumulation_water_weighted(
    flow_direction: np.ndarray,
    dem: np.ndarray,
    water_height: np.ndarray
) -> np.ndarray:
    """
    Compute D8 flow accumulation weighted by water depth.
    
    Simple approach: Calculate standard D8 accumulation, then weight by water depth.
    This avoids numerical explosion while maintaining variation between scenarios.
    
    Args:
        flow_direction: D8 flow direction (1-8) array
        dem: Digital elevation model
        water_height: Water depth at each location (from simulation)
    
    Returns:
        Weighted accumulation where higher water = higher values in downstream cells
    """
    logger.info("Calculating water-weighted flow accumulation...")
    
    height, width = flow_direction.shape
    
    # First, calculate standard D8 accumulation (this is stable and proven)
    acc_standard = np.ones((height, width), dtype=np.float32)
    
    # D8 offsets
    offsets = {
        1: (-1, 1),   # NE
        2: (-1, 0),   # N
        3: (-1, -1),  # NW
        4: (0, -1),   # W
        5: (1, -1),   # SW
        6: (1, 0),    # S
        7: (1, 1),    # SE
        8: (0, 1),    # E
    }
    
    # Process cells from high to low elevation to accumulate downstream
    flat_idx = np.argsort(dem.ravel())[::-1]
    for idx in flat_idx:
        i = int(idx // width)
        j = int(idx % width)
        d = int(flow_direction[i, j])
        if d == 0:
            continue
        di, dj = offsets.get(d, (0, 0))
        ni, nj = i + di, j + dj
        if 0 <= ni < height and 0 <= nj < width:
            acc_standard[ni, nj] += acc_standard[i, j]
    
    # Now weight the accumulation by water depth
    # Water weight: add to base contribution based on local water depth
    # Normalized by 1 meter reference depth
    # Use stronger multiplier to show clear differences between rain scenarios
    water_factor = 1.0 + 3.0 * np.clip(water_height / 1.0, 0.0, 1.0)
    
    # Apply multiplicative weight (ranges from 1.0x to 4.0x)
    acc_weighted = acc_standard * water_factor
    
    # Normalize to 1-10 scale (standard hydrological intensity scale)
    # This is more intuitive and widely used in flood risk mapping
    acc_min = np.nanmin(acc_weighted)
    acc_max = np.nanmax(acc_weighted)
    if acc_max > acc_min:
        acc_normalized = (acc_weighted - acc_min) / (acc_max - acc_min + 1e-6)
    else:
        acc_normalized = np.zeros_like(acc_weighted)
    acc_weighted = 1.0 + acc_normalized * 9.0  # Scale to 1-10
    
    return acc_weighted

def _save_flow_direction(flow_direction, transform=None, crs=None):
    """Save flow direction as GeoTIFF."""
    flow_path = Path("outputs/test_run/flow_direction.tif")
    height, width = flow_direction.shape
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(flow_direction.astype(np.float32, copy=False))
    
    with rasterio.open(
        flow_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=flow_direction.dtype,
        transform=transform,
        crs=crs,
        nodata=0,
        compress='deflate',
        compress_level=9,
    ) as dst:
        dst.write(flow_direction, 1)
    
    logger.info(f"✅ Flow direction saved: {flow_path}")
    return flow_path


def _save_flow_accumulation(flow_acc: np.ndarray, transform=None, crs=None, filename="flow_accumulation.tif"):
    """Save flow accumulation as GeoTIFF."""
    acc_path = Path(f"outputs/test_run/{filename}")
    height, width = flow_acc.shape
    if transform is None or crs is None:
        transform, crs = _default_georef_for_array(flow_acc.astype(np.float32, copy=False))

    with rasterio.open(
        acc_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=flow_acc.dtype,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(flow_acc, 1)

    logger.info(f"✅ Flow accumulation saved: {acc_path}")
    return acc_path

@app.route('/image/results')
def get_results_image():
    img_path = Path("outputs/test_run/results_visualization.png")
    if img_path.exists():
        resp = send_file(img_path, mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    return "Image not found", 404

@app.route('/video/animation.gif')
def get_animation():
    gif_path = Path("outputs/test_run/animation.gif")
    if gif_path.exists():
        return send_file(gif_path, mimetype='image/gif', max_age=0)
    return "Animation not found", 404

@app.route('/download/dem-geotiff')
def download_dem():
    dem_path = Path("outputs/test_run/dem.tif")
    if dem_path.exists():
        return send_file(dem_path, as_attachment=True, download_name='dem.tif')
    return "File not found", 404

@app.route('/download/water-geotiff')
def download_water():
    """Download da lâmina em formato visual (RGB) pronta para leitura no QGIS."""
    try:
        dem_path = Path("outputs/test_run/dem.tif")
        water_path = Path("outputs/test_run/water_depth.tif")
        if not dem_path.exists() or not water_path.exists():
            return "File not found", 404

        with rasterio.open(dem_path) as dsrc:
            dem_arr = dsrc.read(1).astype(np.float32, copy=False)
            transform_last = dsrc.transform
            crs_last = dsrc.crs

        with rasterio.open(water_path) as wsrc:
            water_arr = wsrc.read(1).astype(np.float32, copy=False)

        threshold_default = 0.005
        summary_path = Path("outputs/test_run/summary.json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                threshold_default = float(summary.get('flood_threshold_m', threshold_default))
            except Exception:
                pass

        threshold_m = float(max(1e-6, float(request.args.get('threshold', threshold_default))))
        rgb_bytes = _water_over_terrain_geotiff_bytes(
            dem_arr,
            water_arr,
            transform_last,
            crs_last,
            threshold_m=threshold_m,
        )

        return send_file(
            BytesIO(rgb_bytes),
            as_attachment=True,
            download_name='water_depth_visual_qgis.tif',
            mimetype='image/tiff',
            max_age=0,
        )
    except Exception as e:
        logger.error(f"Error generating visual water GeoTIFF: {e}")
        return f"Visual water GeoTIFF indisponível: {e}", 500


@app.route('/download/water-geotiff-raw')
def download_water_raw():
    """Download da lâmina bruta (float32), para análise quantitativa."""
    water_path = Path("outputs/test_run/water_depth.tif")
    if water_path.exists():
        return send_file(water_path, as_attachment=True, download_name='water_depth_raw.tif')
    return "File not found", 404


@app.route('/download/lamina-agua-rgba')
def download_water_rgba():
    """Download de lâmina RGBA estilizada, transparente abaixo do limiar (vmin)."""
    try:
        water_path = Path("outputs/test_run/water_depth.tif")
        if not water_path.exists():
            return "File not found", 404

        with rasterio.open(water_path) as src:
            water_arr = src.read(1).astype(np.float32, copy=False)
            transform_last = src.transform
            crs_last = src.crs

        # vmin default: automático pela distribuição da lâmina (evita export em branco)
        vmin_default = 0.004
        positive = water_arr[np.isfinite(water_arr) & (water_arr > 0.0)]
        if positive.size > 0:
            p30 = float(np.nanpercentile(positive, 30.0))
            # limiar visual mais seletivo para evitar "lâmina parelha" em todo quadro
            vmin_default = float(np.clip(max(1e-6, 0.9 * p30), 1e-6, 0.03))

        vmin_rgba = float(max(1e-6, float(request.args.get('vmin', vmin_default))))
        vmax_arg = request.args.get('vmax')
        vmax_rgba = float(max(vmin_rgba + 1e-6, float(vmax_arg))) if vmax_arg is not None else None

        dem_arr = None
        dem_path = Path("outputs/test_run/dem.tif")
        if dem_path.exists():
            try:
                with rasterio.open(dem_path) as dsrc:
                    dem_arr = dsrc.read(1).astype(np.float32, copy=False)
            except Exception:
                dem_arr = None

        rgba_bytes = _water_rgba_geotiff_bytes(
            water_arr,
            transform_last,
            crs_last,
            vmin=vmin_rgba,
            vmax=vmax_rgba,
            dem=dem_arr,
        )

        return send_file(
            BytesIO(rgba_bytes),
            as_attachment=True,
            download_name='lamina_agua_rgba.tif',
            mimetype='image/tiff',
            max_age=0,
        )
    except Exception as e:
        logger.error(f"Error generating styled RGBA GeoTIFF: {e}")
        return f"GeoTIFF RGBA estilizado indisponível: {e}", 500


@app.route('/download/lamina-agua-terreno')
def download_water_on_terrain():
    """Download da lâmina de água renderizada sobre o terreno (RGB), pronta para visualização no QGIS."""
    try:
        dem_path = Path("outputs/test_run/dem.tif")
        water_path = Path("outputs/test_run/water_depth.tif")
        if not dem_path.exists() or not water_path.exists():
            return "Files not found", 404

        with rasterio.open(dem_path) as dsrc:
            dem_arr = dsrc.read(1).astype(np.float32, copy=False)
            transform_last = dsrc.transform
            crs_last = dsrc.crs

        with rasterio.open(water_path) as wsrc:
            water_arr = wsrc.read(1).astype(np.float32, copy=False)

        threshold_default = 0.005
        summary_path = Path("outputs/test_run/summary.json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                threshold_default = float(summary.get('flood_threshold_m', threshold_default))
            except Exception:
                pass

        threshold_m = float(max(1e-6, float(request.args.get('threshold', threshold_default))))
        rgb_bytes = _water_over_terrain_geotiff_bytes(
            dem_arr,
            water_arr,
            transform_last,
            crs_last,
            threshold_m=threshold_m,
        )

        return send_file(
            BytesIO(rgb_bytes),
            as_attachment=True,
            download_name='lamina_agua_sobre_terreno_rgb.tif',
            mimetype='image/tiff',
            max_age=0,
        )
    except Exception as e:
        logger.error(f"Error generating water-over-terrain GeoTIFF: {e}")
        return f"GeoTIFF lâmina sobre terreno indisponível: {e}", 500


@app.route('/download/lamina-gif-qgis')
def download_lamina_gif_qgis():
    """Alias explícito: exporta a lâmina no estilo visual do GIF para uso no QGIS."""
    return download_water_on_terrain()


@app.route('/download/fluxo-preferencial-gpkg')
def download_fluxo_preferencial_gpkg():
    gpkg_path = Path("outputs/test_run/fluxo_preferencial.gpkg")
    if gpkg_path.exists():
        return send_file(gpkg_path, as_attachment=True, download_name='fluxo_preferencial.gpkg')
    return "File not found", 404


@app.route('/image/inundacao-gpkg')
def get_inundacao_gpkg_image():
    """Preview da camada `inundacao` do GPKG em PNG."""
    import matplotlib.pyplot as plt
    gpkg_path = Path("outputs/test_run/fluxo_preferencial.gpkg")
    if not gpkg_path.exists():
        return "Inundation GPKG not found", 404

    try:
        import geopandas as gpd  # type: ignore
        gdf = gpd.read_file(gpkg_path, layer='inundacao')
        if gdf.empty:
            return "Layer 'inundacao' is empty", 404

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_facecolor('white')
        gdf.plot(
            ax=ax,
            color=(0.38, 0.67, 0.97, 0.55),
            edgecolor=(0.07, 0.24, 0.55, 0.95),
            linewidth=0.7,
        )
        ax.set_title('Flood Extent (GPKG vector)', fontsize=13, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(alpha=0.15, linestyle='--')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error rendering inundacao layer: {e}")
        return f"Could not render inundacao layer: {e}", 500

@app.route('/download/flow-direction')
def download_flow():
    """Download preferential water flow direction."""
    flow_path = Path("outputs/test_run/flow_direction.tif")
    if flow_path.exists():
        return send_file(flow_path, as_attachment=True, download_name='flow_direction.tif')
    return "File not found", 404


@app.route('/download/fluxo-preferencial-d8')
def download_fluxo_preferencial_d8():
    """Download preferential flow direction raster (D8)."""
    flow_path = Path("outputs/test_run/flow_direction.tif")
    if flow_path.exists():
        return send_file(flow_path, as_attachment=True, download_name='fluxo_preferencial_d8.tif')
    return "File not found", 404

@app.route('/download/flow-accumulation')
def download_flow_accumulation():
    """Download flow accumulation (upstream contributing area proxy)."""
    acc_path = Path("outputs/test_run/flow_accumulation.tif")
    if acc_path.exists():
        return send_file(acc_path, as_attachment=True, download_name='flow_accumulation.tif')
    return "File not found", 404

@app.route('/image/dem')
def get_dem_image():
    """Get DEM visualization as PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colors import LightSource
    dem_path = Path("outputs/test_run/dem.tif")
    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        dem_masked = np.ma.masked_invalid(dem)
        finite_vals = dem[np.isfinite(dem)]
        if finite_vals.size:
            vmin_dem, vmax_dem = np.nanpercentile(finite_vals, (2, 98))

            # Base neutra: elevação em escala de cinza + hillshade sutil (sem "tinta" de cor)
            dem_gray = cm.get_cmap('gray').copy()
            dem_gray.set_bad('white')
            im = ax.imshow(dem_masked, cmap=dem_gray, vmin=float(vmin_dem), vmax=float(vmax_dem), alpha=0.98)

            try:
                dem_fill = np.where(np.isfinite(dem), dem, np.nanmedian(finite_vals)).astype(np.float32, copy=False)
                ls = LightSource(azdeg=315, altdeg=45)
                shade = ls.hillshade(dem_fill, vert_exag=1.6, dx=1.0, dy=1.0)
                ax.imshow(shade, cmap='gray', vmin=0.0, vmax=1.0, alpha=0.22)
            except Exception:
                pass

            fig.colorbar(im, ax=ax, label='Altitude (m)')
        else:
            im = ax.imshow(np.zeros_like(dem, dtype=np.float32), cmap='gray', vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax, label='Altitude (m)')

        ax.set_title('Digital Elevation Model (DEM) — neutral base / base neutra', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        resp = send_file(buf, mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    return "DEM not found", 404

@app.route('/image/flow-direction')
def get_flow_direction_image():
    """Get flow direction visualization as PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LightSource
    flow_path = Path("outputs/test_run/flow_direction.tif")
    if flow_path.exists():
        with rasterio.open(flow_path) as src:
            flow = src.read(1)

        # Visual mais limpo: relevo suave + vetores D8 apenas
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        dem = None
        acc_full = None
        dem_path = Path("outputs/test_run/dem.tif")
        if dem_path.exists():
            with rasterio.open(dem_path) as dsrc:
                dem = dsrc.read(1)
            dem_masked = np.ma.masked_invalid(dem)
            dem_vals = dem[np.isfinite(dem)]
            if dem_vals.size > 0:
                dem_p2, dem_p98 = np.nanpercentile(dem_vals, (2, 98))
                dem_norm = np.where(np.isfinite(dem), np.clip((dem - dem_p2) / (dem_p98 - dem_p2 + 1e-6), 0.0, 1.0), np.nan)
                ls = LightSource(azdeg=315, altdeg=45)
                shade = ls.hillshade(np.where(np.isfinite(dem), dem, np.nanmedian(dem_vals)), vert_exag=1.4, dx=1.0, dy=1.0)
                terrain = np.array(cm.get_cmap('Greys_r')(dem_norm))
                terrain[..., :3] = np.clip(terrain[..., :3] * (0.70 + 0.30 * shade[..., None]), 0.0, 1.0)
                terrain[~np.isfinite(dem), :3] = 1.0
                ax.imshow(terrain, zorder=1, interpolation='bilinear')

                # Força consistência: visual D8 guiado pela declividade real do DEM
                try:
                    dem_fill = np.where(np.isfinite(dem), dem, np.nanmedian(dem_vals)).astype(np.float32, copy=False)
                    flow_dem = _calculate_flow_direction(dem_fill).astype(np.uint8, copy=False)
                    if flow_dem.shape == flow.shape:
                        flow = flow_dem
                        acc_full = _calculate_flow_accumulation(flow, dem_fill).astype(np.float32, copy=False)
                except Exception:
                    pass

        # Vetores D8 com amostragem mais espaçada para reduzir poluição
        h, w = flow.shape
        step = max(8, int(min(h, w) / 70))
        yy, xx = np.mgrid[0:h:step, 0:w:step]
        fsub = flow[0:h:step, 0:w:step]
        valid = fsub > 0
        acc_sub = None

        # Se houver acumulação, filtra para células mais relevantes (menos clutter)
        acc_path = Path("outputs/test_run/flow_accumulation.tif")
        if acc_full is None and acc_path.exists():
            try:
                with rasterio.open(acc_path) as asrc:
                    acc = asrc.read(1).astype(np.float32, copy=False)
                acc_full = acc
            except Exception:
                pass

        if acc_full is not None:
            acc_sub = acc_full[0:h:step, 0:w:step]
            acc_valid = acc_sub[valid]
            if acc_valid.size > 0:
                thr = float(np.nanpercentile(acc_valid, 65))
                valid = valid & (acc_sub >= thr)

        # rarefaz adicional (xadrez) para evitar sobrecarga visual de setas
        sparse_mask = (((yy // step) + (xx // step)) % 2) == 0
        valid = valid & sparse_mask

        # fallback: se ficar vazio, volta para vetores D8 básicos
        if np.count_nonzero(valid) < 30:
            valid = (fsub > 0)

        # Códigos D8 (1..8) consistentes com _calculate_flow_direction:
        # 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE, 8=E
        d8_dx = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.float32)
        d8_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.float32)
        d8_dx_i = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int16)
        d8_dy_i = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int16)
        u = np.zeros_like(fsub, dtype=np.float32)
        v = np.zeros_like(fsub, dtype=np.float32)
        codes = np.clip(fsub.astype(np.int16), 0, 8)
        u[valid] = d8_dx[codes[valid]]
        v[valid] = d8_dy[codes[valid]]

        # Cor por intensidade de acumulação para melhorar leitura operacional
        if acc_sub is not None and np.any(valid):
            avals = np.log1p(np.clip(acc_sub[valid], 0.0, None))
            if avals.size > 0:
                a95 = float(np.nanpercentile(avals, 95))
                a05 = float(np.nanpercentile(avals, 5))
                cvals = np.clip((avals - a05) / (a95 - a05 + 1e-6), 0.0, 1.0)
            else:
                cvals = np.zeros_like(avals)

            # halo branco primeiro
            ax.quiver(
                xx[valid], yy[valid], u[valid], v[valid],
                color=(1.0, 1.0, 1.0, 0.50),
                angles='xy',
                scale_units='xy',
                scale=1.7,
                width=0.0034,
                headwidth=3.9,
                headlength=4.5,
                headaxislength=4.0,
                minlength=0.0,
                zorder=4,
            )

            q = ax.quiver(
                xx[valid], yy[valid], u[valid], v[valid],
                cvals,
                cmap='turbo',
                angles='xy',
                scale_units='xy',
                scale=1.7,
                width=0.0024,
                headwidth=3.6,
                headlength=4.2,
                headaxislength=3.8,
                minlength=0.0,
                zorder=5,
            )
            cbar = plt.colorbar(q, ax=ax, fraction=0.042, pad=0.02)  # type: ignore
            cbar.set_label('Preferential flow intensity (normalized)')
        else:
            ax.quiver(
                xx[valid], yy[valid], u[valid], v[valid],
                color=(0.03, 0.44, 0.96, 0.90),
                angles='xy',
                scale_units='xy',
                scale=1.7,
                width=0.0024,
                headwidth=3.6,
                headlength=4.2,
                headaxislength=3.8,
                minlength=0.0,
                zorder=5,
            )

        # Trilhas principais com seta de direção (deixa explícito para onde a água vai)
        if acc_full is not None:
            try:
                seed_mask = (flow > 0) & np.isfinite(acc_full)
                seed_vals = acc_full[seed_mask]
                if seed_vals.size > 0:
                    seed_thr = float(np.nanpercentile(seed_vals, 86))
                    seeds = np.argwhere(seed_mask & (acc_full >= seed_thr))
                    if seeds.size == 0:
                        seeds = np.argwhere(seed_mask)
                    if seeds.size > 0:
                        order = np.argsort(acc_full[seeds[:, 0], seeds[:, 1]])[::-1]
                        seeds = seeds[order]

                        occupied = np.zeros_like(flow, dtype=np.uint8)
                        max_paths = 54
                        paths_drawn = 0

                        for si, sj in seeds:
                            if paths_drawn >= max_paths:
                                break
                            if occupied[si, sj] >= 2:
                                continue

                            ci, cj = int(si), int(sj)
                            path = [(ci, cj)]

                            for _ in range(260):
                                code = int(flow[ci, cj])
                                if code <= 0 or code > 8:
                                    break
                                ni = ci + int(d8_dy_i[code])
                                nj = cj + int(d8_dx_i[code])
                                if not (0 <= ni < h and 0 <= nj < w):
                                    break
                                if (ni, nj) == (ci, cj):
                                    break
                                if occupied[ni, nj] >= 2 and len(path) >= 8:
                                    break

                                path.append((ni, nj))
                                ci, cj = ni, nj

                            if len(path) < 7:
                                continue

                            sampled = path[::2]
                            if sampled[-1] != path[-1]:
                                sampled.append(path[-1])
                            py = np.array([p[0] for p in sampled], dtype=np.float32)
                            px = np.array([p[1] for p in sampled], dtype=np.float32)

                            # halo + linha principal
                            ax.plot(px, py, color=(0.92, 1.00, 1.00, 0.62), linewidth=4.4, zorder=6)
                            ax.plot(px, py, color=(0.00, 0.30, 0.82, 0.96), linewidth=2.8, zorder=7)

                            # seta no final da trilha
                            if px.size >= 2:
                                # halo da seta (melhora leitura sobre a linha)
                                ax.annotate(
                                    '',
                                    xy=(float(px[-1]), float(py[-1])),
                                    xytext=(float(px[-2]), float(py[-2])),
                                    arrowprops=dict(arrowstyle='-|>', color=(0.92, 1.00, 1.00, 0.90), lw=4.2),
                                    zorder=7.8,
                                )
                                ax.annotate(
                                    '',
                                    xy=(float(px[-1]), float(py[-1])),
                                    xytext=(float(px[-2]), float(py[-2])),
                                    arrowprops=dict(arrowstyle='-|>', color=(0.30, 0.88, 1.00, 0.99), lw=2.6),
                                    zorder=8.0,
                                )

                            for pi, pj in sampled:
                                r0, r1 = max(0, pi - 1), min(h, pi + 2)
                                c0, c1 = max(0, pj - 1), min(w, pj + 2)
                                occupied[r0:r1, c0:c1] = np.clip(occupied[r0:r1, c0:c1] + 1, 0, 3)

                            paths_drawn += 1
            except Exception:
                pass

        ax.text(
            0.015,
            0.02,
            'D8 arrows/lines guided by terrain slope (DEM)',
            transform=ax.transAxes,
            fontsize=9,
            color=(0.08, 0.18, 0.35, 0.95),
            bbox=dict(boxstyle='round,pad=0.25', facecolor=(1, 1, 1, 0.70), edgecolor=(0.70, 0.78, 0.90, 0.9)),
            zorder=9,
        )

        # Força enquadramento no domínio do raster para evitar imagem "encolhida" no canto
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title('Preferential Flow (D8) - Clean Visualization', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(False)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        resp = send_file(buf, mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    return "Flow direction not found", 404

@app.route('/image/flow-accumulation')
def get_flow_accumulation_image():
    """Get flow accumulation visualization as PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    acc_path = Path("outputs/test_run/flow_accumulation.tif")
    if acc_path.exists():
        with rasterio.open(acc_path) as src:
            acc = src.read(1).astype(np.float32, copy=False)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        # Log scale + stretch robusto para evitar imagem "apagada"
        img = np.log1p(np.clip(acc, 0.0, None))
        valid = np.isfinite(img)
        if np.any(valid):
            p5, p995 = np.nanpercentile(img[valid], (5, 99.5))
            imgn = np.clip((img - p5) / (p995 - p5 + 1e-6), 0.0, 1.0)
        else:
            imgn = np.zeros_like(img, dtype=np.float32)

        cmap_acc = cm.get_cmap('magma')
        im = ax.imshow(imgn, cmap=cmap_acc, vmin=0.0, vmax=1.0)

        # Sobreposição das linhas principais de drenagem para leitura rápida
        if np.any(valid):
            main_thr = float(np.nanpercentile(img[valid], 93))
            main = np.where(img >= main_thr, 1.0, np.nan)
            ax.imshow(
                np.ma.masked_invalid(main),
                cmap=cm.get_cmap('winter'),
                alpha=0.50,
                interpolation='nearest',
                zorder=3,
            )

        ax.set_title('Flow Accumulation (log scale)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Normalized accumulation (log scale)')  # type: ignore
        ax.legend(
            handles=[
                Patch(facecolor=cmap_acc(0.25), edgecolor='none', label='Low accumulation'),
                Patch(facecolor=cmap_acc(0.55), edgecolor='none', label='Medium accumulation'),
                Patch(facecolor=cmap_acc(0.90), edgecolor='none', label='High accumulation'),
                Patch(facecolor=cm.get_cmap('winter')(0.75), edgecolor='none', label='Main drainage network'),
            ],
            loc='lower right',
            fontsize=8,
            framealpha=0.92,
            title='Drainage Legend',
            title_fontsize=8,
        )

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        resp = send_file(buf, mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    return "Flow accumulation not found", 404


@app.route('/image/water-peak')
def get_water_peak_image():
    """Get peak/final water depth heatmap with enhanced visualization and proper scaling."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.patches import Rectangle

    water_path = Path("outputs/test_run/water_depth.tif")
    if not water_path.exists():
        return "Water depth not found", 404

    try:
        with rasterio.open(water_path) as src:
            water = src.read(1).astype(np.float64, copy=False)
        
        # Filter out NaN and invalid values
        valid = np.isfinite(water)
        h, w = water.shape
        
        # Create figure with high resolution
        fig = plt.figure(figsize=(12, 10), dpi=120, facecolor='white')
        ax = fig.add_subplot(111)
        
        if np.any(valid):
            water_valid = water[valid]
            min_depth = np.nanmin(water_valid)
            max_depth = np.nanmax(water_valid)
            mean_depth = np.nanmean(water_valid)
            
            # Create masked array (only show valid cells with water > 0)
            water_masked = np.ma.masked_array(water, mask=~valid | (water <= 0))
            
            # CORREÇÃO: Usar escala GLOBAL fixa, não normalizada por simulação
            # Permite comparar visualmente simulações diferentes
            # Assume que max_depth esperado é ~0.5m (500mm de chuva)
            global_vmax = 0.50  # Escala máxima de referência
            global_vmin = 0.001  # 1mm de threshold
            
            # Aderir à escala global mesmo que haja valores maiores (raro)
            norm = Normalize(vmin=global_vmin, vmax=max(max_depth, global_vmax))
            
            # Use high-contrast colormap (better than RdYlBu for water)
            # Para 50mm: cor mais clara; para 500mm: cor mais escura
            cmap = cm.get_cmap('Blues')
            
            # Display with interpolation for smoothness
            im = ax.imshow(water_masked, cmap=cmap, norm=norm, interpolation='bilinear', origin='upper')
            
            # Enhanced colorbar
            cbar = plt.colorbar(im, ax=ax, label='Water Depth (m)', pad=0.02, shrink=0.9)  # type: ignore
            cbar.set_label('Water Depth (m)', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # Add statistics text box
            stats_text = f'Min: {min_depth:.3f}m | Max: {max_depth:.3f}m | Mean: {mean_depth:.3f}m\nValid cells: {valid.sum()}/{valid.size}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # No valid data - show empty grid
            ax.imshow(np.zeros((h, w)), cmap='Blues', vmin=0, vmax=1)
            ax.text(0.5, 0.5, 'No water depth data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color='red')
        
        # Title and labels
        ax.set_title('Peak Water Depth Map (Maximum Inundation Intensity)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('X (pixels)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (pixels)', fontsize=11, fontweight='bold')
        
        # Grid for reference
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='black')
        ax.set_facecolor('#eeeeee')
        
        # Set proper axis limits
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.set_aspect('equal', adjustable='box')
        
        # Save figure
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        buf.seek(0)
        plt.close(fig)
        
        # Return with no-cache headers
        resp = send_file(buf, mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
        
    except Exception as e:
        logger.error(f"Error generating peak water depth image: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@app.route('/image/orthomosaic')
def get_orthomosaic_image():
    """Serve orthomosaic preview (if uploaded) as PNG."""
    png_path = Path("outputs/test_run/orthomosaic.png")
    if png_path.exists():
        return send_file(png_path, mimetype='image/png', max_age=0)
    return "Orthomosaic not found", 404


@app.route('/image/ml-validation')
def get_ml_validation_image():
    """Serve ML validation results (metrics + feature importance) as PNG."""
    png_path = Path("outputs/ml_validation/feature_importance_roc.png")
    if png_path.exists():
        return send_file(png_path, mimetype='image/png', max_age=0)
    return "ML validation not found", 404


@app.route('/image/flood-probability')
def get_flood_probability_image():
    """Serve flood probability map as PNG heatmap."""
    tif_path = Path("outputs/ml_validation/flood_probability.tif")
    if not tif_path.exists():
        return "Flood probability map not found", 404
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    with rasterio.open(tif_path) as src:
        prob = src.read(1).astype(np.float32, copy=False)
    
    valid = np.isfinite(prob)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100, facecolor='white')
    
    if np.any(valid):
        pmask = np.ma.masked_where(~valid, prob)
        im = ax.imshow(pmask, cmap=cm.get_cmap('RdYlGn_r'), vmin=0.0, vmax=1.0, interpolation='bilinear')
    else:
        im = ax.imshow(np.zeros_like(prob), cmap=cm.get_cmap('RdYlGn_r'), vmin=0.0, vmax=1.0)
    
    ax.set_title('ML-Predicted Flood Susceptibility', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    cbar = plt.colorbar(im, ax=ax, label='Probability (0-1)', shrink=0.8)  # type: ignore
    cbar.set_label('Inundation Probability', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#f0f0f0')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    resp = send_file(buf, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/api/ml-metrics')
def get_ml_metrics():
    """Serve ML validation metrics as JSON."""
    metrics_path = Path("outputs/ml_validation/ml_metrics_improved.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({"error": "ML metrics not found"}), 404


@app.route('/download/orthomosaic')
def download_orthomosaic():
    """Download orthomosaic GeoTIFF (if uploaded)."""
    tif_path = Path("outputs/test_run/orthomosaic.tif")
    if tif_path.exists():
        return send_file(tif_path, as_attachment=True, download_name='orthomosaic.tif')
    return "File not found", 404


@app.route('/download/all-data-zip')
def download_all():
    """Download all results as ZIP."""
    try:
        output_dir = Path("outputs/test_run")
        Path("outputs").mkdir(parents=True, exist_ok=True)
        zip_path = Path("outputs/results.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in output_dir.glob('*'):
                if file.is_file() and file.name not in ['results.zip']:
                    zf.write(file, arcname=file.name)
        
        return send_file(zip_path, as_attachment=True, download_name='hydrosim_results.zip')
    except Exception as e:
        logger.error(f"ZIP error: {e}")
        return "Error creating ZIP", 500


@app.route('/download/flood-probability')
def download_flood_probability():
    """Download ML-predicted flood probability as GeoTIFF."""
    tif_path = Path("outputs/ml_validation/flood_probability.tif")
    if tif_path.exists():
        return send_file(tif_path, as_attachment=True, download_name='flood_susceptibility_probability.tif')
    return "File not found", 404


@app.route('/download/ml-validation-zip')
def download_ml_validation():
    """Download all ML validation results as ZIP."""
    try:
        ml_dir = Path("outputs/ml_validation")
        Path("outputs").mkdir(parents=True, exist_ok=True)
        ml_zip_path = Path("outputs/ml_validation_results.zip")
        
        if not ml_dir.exists():
            return "ML validation results not available", 404
        
        with zipfile.ZipFile(ml_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in ml_dir.glob('*'):
                if file.is_file() and file.name not in ['ml_validation_results.zip']:
                    zf.write(file, arcname=f"ml_validation/{file.name}")
        
        return send_file(ml_zip_path, as_attachment=True, download_name='ml_validation_results.zip')
    except Exception as e:
        logger.error(f"ML ZIP error: {e}")
        return "Error creating ML validation ZIP", 500


# ===========================
# 🆕 NEW REST API ROUTES
# ===========================

@app.route('/api/simulations', methods=['GET'])
def list_simulations():
    """
    List all simulations from database.
    Returns JSON with simulation history.
    """
    try:
        sims = SimulationHistory.get_simulations()
        return jsonify({
            'success': True,
            'count': len(sims),
            'simulations': sims
        })
    except Exception as e:
        logger.error(f"Error listing simulations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/simulations/<int:sim_id>', methods=['GET'])
def get_simulation(sim_id):
    """
    Get details of a specific simulation.
    """
    try:
        sims = SimulationHistory.get_simulations()
        for sim in sims:
            if sim.get('id') == sim_id:
                return jsonify({
                    'success': True,
                    'simulation': sim
                })
        return jsonify({
            'success': False,
            'error': 'Simulation not found'
        }), 404
    except Exception as e:
        logger.error(f"Error fetching simulation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/netcdf', methods=['POST'])
def export_netcdf_api():
    """
    Export simulation results to NetCDF format.
    Expects: POST with JSON containing water_depth and dem arrays
    """
    try:
        data = request.get_json()
        if not data or 'water_depth' not in data or 'dem' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing water_depth or dem data'
            }), 400
        
        water = np.array(data['water_depth'])
        dem = np.array(data['dem'])
        transform = data.get('transform', None)
        crs = data.get('crs', 'EPSG:4326')
        
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'simulation_results.nc'
        
        export_to_netcdf(water, dem, transform, crs, str(output_path))
        
        return jsonify({
            'success': True,
            'message': 'NetCDF export successful',
            'file': str(output_path)
        })
    except Exception as e:
        logger.error(f"NetCDF export error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/simulation-netcdf', methods=['GET'])
def download_netcdf():
    """
    Download latest NetCDF export file.
    """
    try:
        nc_path = Path('outputs/simulation_results.nc')
        if nc_path.exists():
            return send_file(
                nc_path,
                as_attachment=True,
                download_name='hydrosim_results.nc',
                mimetype='application/netcdf'
            )
        return jsonify({
            'success': False,
            'error': 'NetCDF file not found. Run export first.'
        }), 404
    except Exception as e:
        logger.error(f"NetCDF download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/hdf5', methods=['POST'])
def export_hdf5_api():
    """
    Export simulation results to HDF5 format.
    Expects: POST with JSON containing water_depth and dem arrays
    """
    try:
        data = request.get_json()
        if not data or 'water_depth' not in data or 'dem' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing water_depth or dem data'
            }), 400
        
        water = np.array(data['water_depth'])
        dem = np.array(data['dem'])
        transform = data.get('transform', None)
        crs = data.get('crs', 'EPSG:4326')
        
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'simulation_results.h5'
        
        export_to_hdf5(water, dem, transform, crs, str(output_path))
        
        return jsonify({
            'success': True,
            'message': 'HDF5 export successful',
            'file': str(output_path)
        })
    except Exception as e:
        logger.error(f"HDF5 export error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/simulation-hdf5', methods=['GET'])
def download_hdf5():
    """
    Download latest HDF5 export file.
    """
    try:
        h5_path = Path('outputs/simulation_results.h5')
        if h5_path.exists():
            return send_file(
                h5_path,
                as_attachment=True,
                download_name='hydrosim_results.h5',
                mimetype='application/x-hdf5'
            )
        return jsonify({
            'success': False,
            'error': 'HDF5 file not found. Run export first.'
        }), 404
    except Exception as e:
        logger.error(f"HDF5 download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_simulations():
    """
    Compare multiple simulations and generate report.
    Expects: POST with JSON containing list of simulation IDs or names
    """
    try:
        data = request.get_json()
        if not data or 'simulation_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing simulation_ids in request'
            }), 400
        
        sim_ids = data['simulation_ids']
        if not isinstance(sim_ids, list) or len(sim_ids) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 simulations to compare'
            }), 400
        
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'comparison_report.json'
        
        # Get simulations from database
        all_sims = SimulationHistory.get_simulations()
        selected_sims = [
            s for s in all_sims 
            if s.get('id') in sim_ids or s.get('name') in sim_ids
        ]
        
        if len(selected_sims) < 2:
            return jsonify({
                'success': False,
                'error': f'Found only {len(selected_sims)} matching simulations. Need at least 2.'
            }), 400
        
        # Generate comparison report
        report = export_comparison_report(selected_sims, str(report_path))
        
        return jsonify({
            'success': True,
            'message': 'Comparison report generated',
            'file': str(report_path),
            'simulations_compared': len(selected_sims),
            'report': report
        })
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/comparison-report', methods=['GET'])
def download_comparison():
    """
    Download latest comparison report.
    """
    try:
        report_path = Path('outputs/comparison_report.json')
        if report_path.exists():
            return send_file(
                report_path,
                as_attachment=True,
                download_name='comparison_report.json',
                mimetype='application/json'
            )
        return jsonify({
            'success': False,
            'error': 'Comparison report not found. Run comparison first.'
        }), 404
    except Exception as e:
        logger.error(f"Report download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns API status and database info.
    """
    try:
        db_exists = (Path('outputs/.database') / 'simulations.db').exists()
        cache_dir = Path('outputs/.cache')
        cache_count = len(list(cache_dir.glob('*.npz'))) if cache_dir.exists() else 0
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'database_initialized': db_exists,
            'cache_items': cache_count,
            'timestamp': str(np.datetime64('now'))
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    host = os.environ.get('HYDROSIM_HOST', os.environ.get('HOST', 'localhost'))
    try:
        port = int(os.environ.get('HYDROSIM_PORT', os.environ.get('PORT', '8888')))
    except Exception:
        port = 8888

    print(f"""
╔════════════════════════════════════════════════════════════════╗
║          HydroSim Web Server v3 Started                        ║
║  • Improved GIF Animation                                      ║
║  • GeoTIFF Export for QGIS                                     ║
║  • Download Package Support                                    ║
╚════════════════════════════════════════════════════════════════╝

🌐 Web Interface: http://{host}:{port}
    """)
    app.run(host=host, port=port, debug=True, use_reloader=False)
