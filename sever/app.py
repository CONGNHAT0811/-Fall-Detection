from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO
from datetime import datetime
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.json_encoder = JSONEncoder
socketio = SocketIO(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
API_KEY = "08112003"
MAX_STREAM_CLIENTS = 2
active_streams = 0
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['fall_detection']
    history_collection = db['history']
    devices_collection = db['devices']
    logging.info("MongoDB connected successfully.")
except ConnectionFailure as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# Load TFLite model
try:
    logging.info("Loading fall_detection TFLite model...")
    interpreter = tflite.Interpreter(model_path="fall_model_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load TFLite model: {e}")
    raise

# Send request with retry
def send_request_to_esp32(url, data, retries=15, backoff_factor=3, timeout=60):
    session = requests.Session()
    retries_config = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries_config))
    try:
        start_time = time.time()
        response = session.post(url, data=data, timeout=timeout)
        response.raise_for_status()
        logging.info(f"Sent request to {url}, response time: {time.time() - start_time:.2f}s")
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send request to {url}: {str(e)}")
        return None

# Check ESP32 status
def is_esp32_online(ip):
    device = devices_collection.find_one({"ip": ip})
    if not device:
        return False
    last_seen = device.get('last_seen')
    if not last_seen:
        return False
    time_diff = (datetime.now() - last_seen).total_seconds()
    return time_diff < 60

@app.before_request
def log_request():
    logging.debug(f"Request: {request.method} {request.path}, payload: {request.get_data(as_text=True)}")

@app.route('/')
@app.route('/camera')
def camera():
    error_message = None
    try:
        device = devices_collection.find_one({"mac": "F8:B3:B7:7B:32:A8"})
        esp32_ip = device.get('ip', '192.168.0.101') if device else '192.168.0.101'
        if not is_esp32_online(esp32_ip):
            error_message = "Camera is offline. Please check the device."
        else:
            response = send_request_to_esp32(
                f"http://{esp32_ip}:81/set_mode",
                data={'mode': 'stream'},
                retries=15,
                backoff_factor=3,
                timeout=60
            )
            if response:
                logging.info("Set ESP32 to stream mode")
            else:
                error_message = "Failed to connect to camera."
    except Exception as e:
        logging.error(f"Error setting stream mode: {str(e)}")
        error_message = "Failed to connect to camera."
    return render_template('camera.html', error=error_message)

@app.route('/messenger')
def messenger():
    error_message = None
    try:
        device = devices_collection.find_one({"mac": "F8:B3:B7:7B:32:A8"})
        esp32_ip = device.get('ip', '192.168.0.101') if device else '192.168.0.101'
        if not is_esp32_online(esp32_ip):
            error_message = "Camera is offline. Please check the device."
        else:
            # Kiểm tra trạng thái trước khi gửi yêu cầu
            ping_response = send_request_to_esp32(
                f"http://{esp32_ip}:81/set_mode",
                data={'mode': 'ping'},
                timeout=5
            )
            if not ping_response:
                error_message = "ESP32 is not responding. Please wait and try again."
            else:
                response = send_request_to_esp32(
                    f"http://{esp32_ip}:81/set_mode",
                    data={'mode': 'detection'},
                    retries=15,
                    backoff_factor=3,
                    timeout=60
                )
                if response:
                    logging.info("Set ESP32 to detection mode")
                else:
                    error_message = "Failed to set detection mode."
    except Exception as e:
        logging.error(f"Error setting detection mode: {str(e)}")
        error_message = "Failed to set detection mode."
    return render_template('messenger.html', error=error_message)

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        results = list(history_collection.find({}, {"_id": 0}))
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error fetching history: {str(e)}")
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    try:
        mode = request.form.get('mode')
        if mode not in ['stream', 'detection']:
            return jsonify({'error': 'Invalid mode'}), 400
        device = devices_collection.find_one({"mac": "F8:B3:B7:7B:32:A8"})
        esp32_ip = device.get('ip', '192.168.0.101') if device else '192.168.0.101'
        if not is_esp32_online(esp32_ip):
            return jsonify({'status': 'error', 'message': 'Camera is offline.'}), 500
        response = send_request_to_esp32(
            f"http://{esp32_ip}:81/set_mode",
            data={'mode': mode},
            retries=15,
            backoff_factor=3,
            timeout=60
        )
        if response:
            logging.info(f"Set ESP32 to {mode} mode")
            return jsonify({'status': 'success', 'mode': mode})
        return jsonify({'status': 'error', 'message': 'Failed to connect to ESP32.'}), 500
    except Exception as e:
        logging.error(f"Error setting mode: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to ESP32: {str(e)}'}), 500

@app.route('/api/ping_esp32', methods=['GET'])
def ping_esp32():
    try:
        device = devices_collection.find_one({"mac": "F8:B3:B7:7B:32:A8"})
        esp32_ip = device.get('ip', '192.168.0.101') if device else '192.168.0.101'
        if not is_esp32_online(esp32_ip):
            return jsonify({"status": "offline"}), 500
        response = send_request_to_esp32(
            f"http://{esp32_ip}:81/set_mode",
            data={'mode': 'ping'},
            timeout=5
        )
        return jsonify({"status": "online" if response else "offline"})
    except Exception as e:
        logging.error(f"Ping ESP32 failed: {str(e)}")
        return jsonify({"status": "offline"}), 500

@app.route('/api/report_ip', methods=['POST'])
def report_ip():
    try:
        data = request.get_json()
        ip = data.get('ip')
        mac = data.get('mac')
        logging.info(f"Updated ESP32 IP to {ip} (MAC: {mac})")
        devices_collection.update_one(
            {"mac": mac},
            {"$set": {"ip": ip, "last_seen": datetime.now()}},
            upsert=True
        )
        return jsonify({"status": "success", "ip": ip})
    except Exception as e:
        logging.error(f"Failed to update IP: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/fall_detect', methods=['POST'])
def fall_detect():
    try:
        logging.debug(f"Received /fall_detect, data size: {len(request.data)} bytes")
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'fall': False, 'error': "Invalid API key"}), 401

        if not request.data:
            return jsonify({'fall': False, 'error': "No image data received"}), 400

        raw = np.frombuffer(request.data, dtype=np.uint8)
        if raw.size < 96 * 96:
            return jsonify({'fall': False, 'error': f"Data size too small: {raw.size} bytes"}), 400
        raw = raw[:96 * 96]

        image = raw ^ 0x80
        image = image.astype(np.int8).reshape((96, 96, 1))

        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        fall_score = output_data.item() if output_data.size == 1 else output_data[1]

        if fall_score > 1.0 or fall_score < 0.0:
            fall_score = max(0.0, min(1.0, fall_score))

        status = "Ngã" if fall_score > 0.7 else "Bình thường"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        img_data = (image.astype(np.uint8) ^ 0x80).reshape((96, 96))
        img = Image.fromarray(img_data, mode='L')
        filename = f"fall_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        img.save(filepath)
        image_url = f"/{filepath}"

        entry = {
            "timestamp": timestamp,
            "location": request.headers.get('X-Location', 'Phòng khách'),
            "status": status,
            "probability": f"{fall_score * 100:.0f}%",
            "image_path": image_url
        }
        history_collection.insert_one(entry)

        socketio.emit('new_detection', entry)
        logging.info(f"Detection: fall={fall_score > 0.7}, score={fall_score}")
        return jsonify({"fall": fall_score > 0.7, "score": float(fall_score)})

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'fall': False, 'error': str(e)}), 500

@app.route('/stream')
def stream():
    global active_streams
    try:
        if active_streams >= MAX_STREAM_CLIENTS:
            return "<p>Error: Maximum clients reached</p>", 429

        active_streams += 1
        def proxy_stream():
            global active_streams
            try:
                device = devices_collection.find_one({"mac": "F8:B3:B7:7B:32:A8"})
                esp32_ip = device.get('ip', '192.168.0.101') if device else '192.168.0.101'
                url = f"http://{esp32_ip}:81/stream"
                while True:
                    if not is_esp32_online(esp32_ip):
                        break
                    try:
                        req = requests.get(url, stream=True, timeout=10)
                        for chunk in req.iter_content(chunk_size=1024):
                            if chunk:
                                yield chunk
                    except requests.exceptions.RequestException as e:
                        time.sleep(3)
                        continue
            except Exception as e:
                logging.error(f"Stream proxy failed: {str(e)}")
                raise
            finally:
                active_streams -= 1
        return Response(proxy_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Error proxying stream: {str(e)}")
        active_streams -= 1
        return "<p>Error: Failed to connect to camera</p>", 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    try:
        return send_from_directory('static/audio', filename)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'Audio file not found'}), 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)