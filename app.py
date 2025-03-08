from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import shutil
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Serve index.html
@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

# Serve realtime.html
@app.route('/realtime.html')
def serve_realtime():
    return send_from_directory('templates', 'realtime.html')

# Serve train.html
@app.route('/train.html')
def serve_train():
    return send_from_directory('templates', 'train.html')

# Serve speech.html
@app.route('/speech.html')
def serve_speech():
    return send_from_directory('templates', 'speech.html')

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Existing API endpoints
@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    gestures = data.get('gestures', []) if data else []
    if not gestures:
        return jsonify({'error': 'No gestures provided'}), 400

    try:
        proc = subprocess.run(
            ['python', 'train.py', '--gestures'] + gestures,
            check=True,
            capture_output=True,
            text=True
        )
        return jsonify({
            'message': 'Training completed',
            'output': proc.stdout,
            'error': proc.stderr
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': 'Training failed',
            'output': e.stdout,
            'error': e.stderr
        }), 500

@app.route('/realtime', methods=['POST'])
def realtime_recognition():
    try:
        proc = subprocess.Popen(
            ['python', 'realtime_recognition.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return jsonify({
            'message': 'Realtime recognition started',
            'pid': proc.pid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speech', methods=['POST'])
def speech_recognition():
    try:
        proc = subprocess.Popen(
            ['python', 'speech_recognition.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return jsonify({
            'message': 'Speech recognition started',
            'pid': proc.pid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        result = subprocess.run(
            ['python', 'clear_cache.py'],
            check=True,
            capture_output=True,
            text=True
        )
        return jsonify({
            'message': 'Cache cleared',
            'output': result.stdout,
            'error': result.stderr
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': 'Cache clearance failed',
            'output': e.stdout,
            'error': e.stderr
        }), 500

# Block hybridaction requests
@app.route('/hybridaction/<path:subpath>', methods=['GET', 'POST'])
def block_hybridaction(subpath):
    logger.debug(f"Blocked request to /hybridaction/{subpath}")  # Suppress logs
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
