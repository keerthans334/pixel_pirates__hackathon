from flask import Flask, request, jsonify, send_file
import subprocess
import os
import shutil

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    gestures = request.json.get('gestures')
    if not gestures:
        return jsonify({'error': 'No gestures provided'}), 400

    # Call the train.py script with the specified gestures
    subprocess.run(['python', 'train.py', '--gestures', *gestures])
    return jsonify({'message': 'Training completed'})

@app.route('/realtime', methods=['GET'])
def realtime_recognition():
    subprocess.Popen(['python', 'realtime_recognition.py'])
    return jsonify({'message': 'Realtime recognition started'})

@app.route('/speech', methods=['GET'])
def speech_recognition():
    subprocess.Popen(['python', 'speech_recognition.py'])
    return jsonify({'message': 'Speech recognition started'})

@app.route('/clear_cache', methods=['GET'])
def clear_cache():
    result = subprocess.run(['python', 'clear_cache.py'], capture_output=True, text=True)
    return jsonify({'message': result.stdout})

if __name__ == '__main__':
    app.run(debug=True, port=5000)