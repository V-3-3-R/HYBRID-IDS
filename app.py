"""
Hybrid IDS - Web Dashboard
Flask Application for Real-Time Monitoring
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Load trained models
try:
    rf_model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Models loaded successfully!")
except:
    print("⚠️ Models not found. Using dummy data.")
    rf_model = None
    scaler = None

# In-memory storage for alerts
alerts_db = []
stats_db = {
    'total_connections': 0,
    'total_alerts': 0,
    'signature_detections': 0,
    'anomaly_detections': 0,
    'blocked_threats': 0
}

# Attack categories
ATTACK_TYPES = ['DoS', 'Probe', 'R2L', 'U2R', 'SQL Injection', 'Port Scan', 'Brute Force']
SEVERITY_LEVELS = ['Low', 'Medium', 'High', 'Critical']

# ====================
# ROUTES
# ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current system statistics"""
    return jsonify(stats_db)

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(alerts_db[-limit:])

@app.route('/api/alerts/realtime')
def get_realtime_alerts():
    """Get alerts from last minute"""
    one_minute_ago = datetime.now() - timedelta(minutes=1)
    recent_alerts = [
        alert for alert in alerts_db 
        if datetime.fromisoformat(alert['timestamp']) > one_minute_ago
    ]
    return jsonify(recent_alerts)

@app.route('/api/attack-distribution')
def attack_distribution():
    """Get attack type distribution"""
    distribution = {}
    for alert in alerts_db:
        attack_type = alert['attack_type']
        distribution[attack_type] = distribution.get(attack_type, 0) + 1
    return jsonify(distribution)

@app.route('/api/severity-distribution')
def severity_distribution():
    """Get severity level distribution"""
    distribution = {}
    for alert in alerts_db:
        severity = alert['severity']
        distribution[severity] = distribution.get(severity, 0) + 1
    return jsonify(distribution)

@app.route('/api/timeline')
def timeline():
    """Get alerts timeline (last 24 hours)"""
    hours = []
    counts = []
    
    for i in range(24):
        hour_start = datetime.now() - timedelta(hours=23-i)
        hour_end = hour_start + timedelta(hours=1)
        
        hour_alerts = [
            alert for alert in alerts_db
            if hour_start <= datetime.fromisoformat(alert['timestamp']) < hour_end
        ]
        
        hours.append(hour_start.strftime('%H:00'))
        counts.append(len(hour_alerts))
    
    return jsonify({'hours': hours, 'counts': counts})

@app.route('/api/detection-methods')
def detection_methods():
    """Get detection method statistics"""
    signature_count = sum(1 for a in alerts_db if a['detection_method'] == 'Signature')
    anomaly_count = sum(1 for a in alerts_db if a['detection_method'] == 'Anomaly-ML')
    
    return jsonify({
        'signature': signature_count,
        'anomaly': anomaly_count
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if traffic is malicious"""
    if not rf_model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = rf_model.predict(features_scaled)[0]
        probability = rf_model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Attack' if prediction == 1 else 'Normal',
            'confidence': float(max(probability)),
            'attack_probability': float(probability[1] if len(probability) > 1 else 0)
        }
        
        # Log alert if attack detected
        if prediction == 1:
            add_alert(
                attack_type='Unknown_Anomaly',
                severity='Medium',
                source_ip=data.get('source_ip', '0.0.0.0'),
                detection_method='Anomaly-ML',
                confidence=float(probability[1])
            )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/simulate', methods=['POST'])
def simulate_traffic():
    """Simulate network traffic and generate alerts"""
    num_connections = request.json.get('count', 10)
    
    for _ in range(num_connections):
        # Simulate random attack or normal traffic
        is_attack = random.random() < 0.3  # 30% attack rate
        
        if is_attack:
            attack_type = random.choice(ATTACK_TYPES)
            severity = random.choice(SEVERITY_LEVELS)
            detection_method = random.choice(['Signature', 'Anomaly-ML'])
            
            add_alert(
                attack_type=attack_type,
                severity=severity,
                source_ip=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                detection_method=detection_method,
                confidence=random.uniform(0.7, 1.0)
            )
        
        stats_db['total_connections'] += 1
    
    return jsonify({'message': f'Simulated {num_connections} connections'})

@app.route('/api/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    global alerts_db, stats_db
    alerts_db = []
    stats_db = {
        'total_connections': 0,
        'total_alerts': 0,
        'signature_detections': 0,
        'anomaly_detections': 0,
        'blocked_threats': 0
    }
    return jsonify({'message': 'All alerts cleared'})

# ====================
# HELPER FUNCTIONS
# ====================

def add_alert(attack_type, severity, source_ip, detection_method, confidence):
    """Add new alert to database"""
    alert = {
        'id': len(alerts_db) + 1,
        'timestamp': datetime.now().isoformat(),
        'attack_type': attack_type,
        'severity': severity,
        'source_ip': source_ip,
        'destination_ip': f"192.168.1.{random.randint(1,255)}",
        'port': random.randint(1, 65535),
        'detection_method': detection_method,
        'confidence': confidence,
        'status': 'Blocked' if severity in ['High', 'Critical'] else 'Logged'
    }
    
    alerts_db.append(alert)
    
    # Update statistics
    stats_db['total_alerts'] += 1
    if detection_method == 'Signature':
        stats_db['signature_detections'] += 1
    else:
        stats_db['anomaly_detections'] += 1
    
    if alert['status'] == 'Blocked':
        stats_db['blocked_threats'] += 1
    
    # Keep only last 1000 alerts
    if len(alerts_db) > 1000:
        alerts_db.pop(0)

def generate_sample_alerts():
    """Generate some sample alerts for demo"""
    sample_attacks = [
        ('SQL Injection', 'High', '203.0.113.45', 'Signature', 1.0),
        ('DoS', 'Critical', '198.51.100.23', 'Signature', 1.0),
        ('Port Scan', 'Medium', '192.0.2.100', 'Anomaly-ML', 0.87),
        ('Brute Force', 'High', '203.0.113.12', 'Signature', 1.0),
        ('Unknown_Anomaly', 'Low', '198.51.100.89', 'Anomaly-ML', 0.72),
    ]
    
    for attack in sample_attacks:
        add_alert(*attack)

# ====================
# HTML TEMPLATE
# ====================

def create_html_template():
    """Create dashboard HTML template"""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid IDS Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stat-card h3 { font-size: 0.9em; color: #ccc; margin-bottom: 10px; }
        .stat-card .value { font-size: 2.5em; font-weight: bold; }
        .alerts-section {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th { background: rgba(255,255,255,0.1); font-weight: 600; }
        .severity-high { color: #ff6b6b; font-weight: bold; }
        .severity-critical { color: #ff0000; font-weight: bold; }
        .severity-medium { color: #ffa500; font-weight: bold; }
        .severity-low { color: #4ecdc4; font-weight: bold; }
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 10px;
        }
        button {
            background: #4ecdc4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        button:hover { background: #45b7d1; }
        .status-active { color: #00ff00; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Hybrid Intrusion Detection System</h1>
            <p>Real-time Network Security Monitoring</p>
            <p class="status-active">● System Active</p>
        </div>

        <div class="stats-grid" id="stats-grid">
            <!-- Stats will be loaded here -->
        </div>

        <div class="controls">
            <button onclick="simulateTraffic()">🔄 Simulate Traffic</button>
            <button onclick="clearAlerts()">🗑️ Clear Alerts</button>
            <button onclick="refreshData()">↻ Refresh</button>
        </div>

        <div class="alerts-section">
            <h2>Recent Security Alerts</h2>
            <table id="alerts-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Type</th>
                        <th>Severity</th>
                        <th>Source IP</th>
                        <th>Method</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="alerts-body">
                    <!-- Alerts will be loaded here -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function loadStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('stats-grid').innerHTML = `
                        <div class="stat-card">
                            <h3>Total Connections</h3>
                            <div class="value">${data.total_connections}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Total Alerts</h3>
                            <div class="value">${data.total_alerts}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Signature Detections</h3>
                            <div class="value">${data.signature_detections}</div>
                        </div>
                        <div class="stat-card">
                            <h3>ML Detections</h3>
                            <div class="value">${data.anomaly_detections}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Blocked Threats</h3>
                            <div class="value">${data.blocked_threats}</div>
                        </div>
                    `;
                });
        }

        function loadAlerts() {
            fetch('/api/alerts?limit=20')
                .then(r => r.json())
                .then(alerts => {
                    const tbody = document.getElementById('alerts-body');
                    tbody.innerHTML = alerts.reverse().map(alert => `
                        <tr>
                            <td>${new Date(alert.timestamp).toLocaleTimeString()}</td>
                            <td>${alert.attack_type}</td>
                            <td class="severity-${alert.severity.toLowerCase()}">${alert.severity}</td>
                            <td>${alert.source_ip}</td>
                            <td>${alert.detection_method}</td>
                            <td>${(alert.confidence * 100).toFixed(1)}%</td>
                            <td>${alert.status}</td>
                        </tr>
                    `).join('');
                });
        }

        function simulateTraffic() {
            fetch('/api/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({count: 20})
            }).then(() => refreshData());
        }

        function clearAlerts() {
            if (confirm('Clear all alerts?')) {
                fetch('/api/clear', {method: 'POST'})
                    .then(() => refreshData());
            }
        }

        function refreshData() {
            loadStats();
            loadAlerts();
        }

        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
    """
    
    # Save template
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)

# ====================
# MAIN
# ====================

if __name__ == '__main__':
    # Create templates directory
    import os
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Create HTML template
    create_html_template()
    
    # Generate sample alerts
    generate_sample_alerts()
    
    print("="*60)
    print("🚀 Hybrid IDS Dashboard Starting...")
    print("="*60)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"🔍 API Docs: http://localhost:5000/api/stats")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)