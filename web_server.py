#!/usr/bin/env python3
"""
Flask web server for HydroSim-RF results visualization.
Serves HTML interface on http://localhost:5000

Run:
    python web_server.py
"""

from flask import Flask, render_template_string, jsonify  # type: ignore
import json
import numpy as np
from pathlib import Path
import base64
from io import BytesIO

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HydroSim-RF - Flood Simulation Results</title>
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
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section-title {
            font-size: 1.8em;
            color: #1e88e5;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1e88e5;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .stat-unit {
            font-size: 0.8em;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .image-container {
            text-align: center;
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
        }
        
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .image-label {
            margin-top: 10px;
            color: #666;
            font-weight: bold;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin-bottom: 30px;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
        }
        
        .json-container {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            border-left: 4px solid #1e88e5;
        }
        
        .json-container pre {
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #333;
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 10px 20px;
            background: #1e88e5;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s ease;
        }
        
        .btn:hover {
            background: #1565c0;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
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
            <h1>🌊 HydroSim-RF</h1>
            <p>Flood Simulation & Machine Learning Results Viewer</p>
        </div>
        
        <div class="content">
            <!-- Summary Section -->
            <div class="section">
                <div class="section-title">📊 Simulation Summary</div>
                <div class="stats-grid" id="stats-container"></div>
            </div>
            
            <!-- Visualizations -->
            <div class="section">
                <div class="section-title">📈 Visualizations</div>
                <div class="images-grid">
                    <div class="image-container">
                        <h3>Simulation Results</h3>
                        <img src="/image/results" alt="Results visualization">
                        <div class="image-label">DEM • Sources • Water • Probability</div>
                    </div>
                    <div class="image-container">
                        <h3>Time Series</h3>
                        <img src="/image/timeseries" alt="Time series">
                        <div class="image-label">Flooded Area • Active Cells • Depth • Volume</div>
                    </div>
                </div>
            </div>
            
            <!-- Time Series Charts -->
            <div class="section">
                <div class="section-title">📉 Time Series Analysis</div>
                <div class="chart-container">
                    <canvas id="timeseriesChart"></canvas>
                </div>
            </div>
            
            <!-- JSON Data -->
            <div class="section">
                <div class="section-title">📋 Raw Data (JSON)</div>
                <div class="button-group">
                    <button class="btn" onclick="showSummary()">Summary</button>
                    <button class="btn" onclick="showHistory()">History (First 10)</button>
                </div>
                <div class="json-container">
                    <pre id="json-data"></pre>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>✅ All results generated successfully | 📁 outputs/test_run/</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                HydroSim-RF • Scientific Flood Simulation Framework • Environmental Modelling & Software
            </p>
        </div>
    </div>
    
    <script>
        let fullHistory = [];
        
        // Load summary on page load
        async function loadSummary() {
            const summary = await fetch('/api/summary').then(r => r.json());
            const history = await fetch('/api/history').then(r => r.json());
            fullHistory = history;
            
            // Populate stats
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-label">Simulation Time</div>
                    <div class="stat-value">${summary.simulation.final_time_minutes}</div>
                    <div class="stat-unit">minutes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Timesteps</div>
                    <div class="stat-value">${summary.simulation.timesteps}</div>
                    <div class="stat-unit">steps</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max Water Depth</div>
                    <div class="stat-value">${summary.simulation.max_depth_m.toFixed(3)}</div>
                    <div class="stat-unit">meters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Volume</div>
                    <div class="stat-value">${(summary.simulation.total_water_volume_m3 / 1e6).toFixed(2)}</div>
                    <div class="stat-unit">million m³</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Flooded Cells</div>
                    <div class="stat-value">${summary.simulation.flooded_cells}</div>
                    <div class="stat-unit">cells</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Mean Probability</div>
                    <div class="stat-value">${summary.probability.mean_p.toFixed(3)}</div>
                    <div class="stat-unit">dimensionless</div>
                </div>
            `;
            
            document.getElementById('stats-container').innerHTML = statsHtml;
            
            // Create chart
            createChart(history);
            
            // Show summary JSON
            showSummary();
        }
        
        function createChart(history) {
            const times = history.map(h => h.time_minutes);
            const flooded = history.map(h => h.flooded_percent);
            const depth = history.map(h => h.max_depth);
            const volume = history.map(h => h.total_water_volume_m3 / 1e6);
            
            const ctx = document.getElementById('timeseriesChart').getContext('2d');
            new Chart(ctx, {
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
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Simulation Time Series',
                            font: { size: 14, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Flooded Area (%)' },
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Max Depth (m)' },
                            grid: { drawOnChartArea: false },
                        },
                        y2: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Volume (million m³)' },
                            grid: { drawOnChartArea: false },
                        },
                        x: {
                            title: { display: true, text: 'Time (minutes)' }
                        }
                    }
                }
            });
        }
        
        function showSummary() {
            fetch('/api/summary')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('json-data').textContent = JSON.stringify(data, null, 2);
                });
        }
        
        function showHistory() {
            const first10 = fullHistory.slice(0, 10);
            document.getElementById('json-data').textContent = JSON.stringify(first10, null, 2);
        }
        
        // Load on page load
        window.onload = loadSummary;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/summary')
def api_summary():
    """Return summary JSON."""
    summary_path = Path("outputs/test_run/summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Summary not found"}), 404

@app.route('/api/history')
def api_history():
    """Return history JSON."""
    history_path = Path("outputs/test_run/history.json")
    if history_path.exists():
        with open(history_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "History not found"}), 404

@app.route('/image/<image_type>')
def get_image(image_type):
    """Return image as base64."""
    if image_type == "results":
        img_path = Path("outputs/test_run/results_visualization.png")
    elif image_type == "timeseries":
        img_path = Path("outputs/test_run/timeseries.png")
    else:
        return "Image not found", 404
    
    if img_path.exists():
        with open(img_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" />'
    
    return "Image not found", 404

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════╗
║           HydroSim-RF Web Server Started                       ║
╚════════════════════════════════════════════════════════════════╝

🌐 Web Interface: http://localhost:5000

✅ Features:
   • Real-time visualization
   • Interactive charts
   • JSON data export
   • Responsive design

Press CTRL+C to stop the server.
    """)
    app.run(host='localhost', port=5000, debug=True)
