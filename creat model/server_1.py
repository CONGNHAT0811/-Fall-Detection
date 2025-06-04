from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO
from datetime import datetime
import numpy as np
try:
    from ai_edge_litert import lite_rt as tflite
except ImportError:
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

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.json_encoder = JSONEncoder
socketio = SocketIO(app)

# Cấu hình
UPLOAD_FOLDER = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
API_KEY = "08112003"
ESP32_CAM_IP = "192.168.0.100"
MAX_STREAM_CLIENTS = 2
active_streams = 0
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Kết nối MongoDB
try:
    client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['fall_detection']
    history_collection = db['history']
    devices_collection = db['devices']
    print("[INFO] Kết nối MongoDB thành công.")
except ConnectionFailure as e:
    print(f"[ERROR] Không thể kết nối MongoDB: {e}")
    raise

# Load mô hình TFLite
try:
    print("[INFO] Loading fall_detection TFLite model...")
    interpreter = tflite.Interpreter(model_path="fall_model_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load TFLite model: {e}")
    raise

# Biến trạng thái thiết bị
device_status = {"mode": "unknown", "streaming": False, "last_updated": None}

# Hàm kiểm tra trạng thái thiết bị
def check_device_status():
    global device_status
    try:
        response = requests.get(f"http://{ESP32_CAM_IP}:81/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            device_status["mode"] = status["mode"]
            device_status["streaming"] = status["streaming"]
            device_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Failed to check device status: {str(e)}")
        device_status["mode"] = "offline"
        device_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return False

# Hàm gửi yêu cầu với cơ chế thử lại
def send_request_to_esp32(url, data, retries=5, backoff_factor=2, timeout=15):
    session = requests.Session()
    retries_config = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries_config))
    try:
        response = session.post(url, data=data, timeout=timeout)
        response.raise_for_status()
        print(f"[INFO] Successfully sent request to {url}")
        check_device_status()  # Cập nhật trạng thái sau mỗi yêu cầu
        return response
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to send request to {url}: {str(e)}")
        check_device_status()
        return None

# Log all incoming requests
@app.before_request
def log_request():
    print(f"[DEBUG] Incoming request: {request.method} {request.path}")

@app.route('/')
@app.route('/camera')
def camera():
    error_message = None
    try:
        # Kiểm tra trạng thái thiết bị trước
        if not check_device_status() or device_status["mode"] != "stream":
            response = send_request_to_esp32(
                f"http://{ESP32_CAM_IP}:81/set_mode",
                data={'mode': 'stream'},
                retries=5,
                backoff_factor=2,
                timeout=15
            )
            if not response:
                error_message = "Không thể kết nối với camera. Vui lòng kiểm tra mạng hoặc thiết bị."
    except Exception as e:
        print(f"[ERROR] Failed to send stream mode request: {str(e)}")
        error_message = "Không thể kết nối với camera. Vui lòng kiểm tra mạng hoặc thiết bị."
    return render_template('camera.html', error=error_message, device_status=device_status)

@app.route('/messenger')
def messenger():
    error_message = None
    try:
        # Kiểm tra trạng thái thiết bị trước
        if not check_device_status() or device_status["mode"] != "detection":
            response = send_request_to_esp32(
                f"http://{ESP32_CAM_IP}:81/set_mode",
                data={'mode': 'detection'},
                retries=5,
                backoff_factor=2,
                timeout=15
            )
            if not response:
                error_message = "Không thể chuyển ESP32 sang chế độ phát hiện. Vui lòng kiểm tra mạng hoặc thiết bị."
    except Exception as e:
        print(f"[ERROR] Failed to send detection mode request: {str(e)}")
        error_message = "Không thể chuyển ESP32 sang chế độ phát hiện. Vui lòng kiểm tra mạng hoặc thiết bị."
    return render_template('messenger.html', error=error_message, device_status=device_status)

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        results = list(history_collection.find({}, {"_id": 0}))
        return jsonify(results)
    except Exception as e:
        print(f"[ERROR] Lỗi khi lấy lịch sử: {str(e)}")
        return jsonify({"error": f"Không thể lấy lịch sử: {str(e)}"}), 500

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    try:
        mode = request.form.get('mode')
        if mode not in ['stream', 'detection']:
            return jsonify({'error': 'Invalid mode'}), 400
        
        # Kiểm tra trạng thái thiết bị trước
        if check_device_status() and device_status["mode"] == mode:
            return jsonify({'status': 'success', 'mode': mode})

        response = send_request_to_esp32(
            f"http://{ESP32_CAM_IP}:81/set_mode",
            data={'mode': mode},
            retries=5,
            backoff_factor=2,
            timeout=15
        )
        if response:
            print(f"[INFO] Successfully set ESP32 to {mode} mode")
            return jsonify({'status': 'success', 'mode': mode})
        return jsonify({
            'status': 'error',
            'message': 'Không thể kết nối với ESP32. Vui lòng kiểm tra mạng hoặc thiết bị.'
        }), 500
    except Exception as e:
        print(f"[ERROR] Error setting mode: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Không thể kết nối với ESP32: {str(e)}'
        }), 500

@app.route('/api/ping_esp32', methods=['GET'])
def ping_esp32():
    if check_device_status():
        return jsonify({"status": "online", "mode": device_status["mode"]})
    return jsonify({"status": "offline"}), 500

@app.route('/api/device_status', methods=['GET'])
def get_device_status():
    check_device_status()
    return jsonify(device_status)

@app.route('/fall_detect', methods=['POST'])
def fall_detect():
    try:
        print(f"[DEBUG] Received POST request to /fall_detect, data size: {len(request.data)} bytes")
        print(f"[DEBUG] Headers: {dict(request.headers)}")
        if request.headers.get('X-API-Key') != API_KEY:
            print("[ERROR] Invalid API key")
            return jsonify({'fall': False, 'error': "Khóa API không hợp lệ"}), 401

        if not request.data:
            print("[ERROR] No image data received")
            return jsonify({'fall': False, 'error': "Không nhận được dữ liệu ảnh"}), 400

        raw = np.frombuffer(request.data, dtype=np.uint8)
        if raw.size < 96 * 96:
            print(f"[ERROR] Data size too small: {raw.size} bytes, expected: {96 * 96} bytes")
            return jsonify({'fall': False, 'error': f"Kích thước dữ liệu quá nhỏ: {raw.size} bytes"}), 400
        raw = raw[:96 * 96]

        image = raw ^ 0x80
        image = image.astype(np.int8).reshape((96, 96, 1))

        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Debug thông tin đầu ra
        print(f"[DEBUG] Output data shape: {output_data.shape}, size: {output_data.size}, dtype: {output_data.dtype}")
        print(f"[DEBUG] Output data: {output_data}")

        # Điều chỉnh logic dựa trên kích thước đầu ra
        if output_data.size == 1:
            fall_score = output_data.item()  # Lấy giá trị duy nhất nếu chỉ có 1 phần tử
        else:
            fall_score = output_data[0] if len(output_data.shape) == 1 else output_data[0, 0]  # Lấy giá trị đầu tiên nếu có nhiều hơn

        if fall_score > 1.0 or fall_score < 0.0:
            print(f"[WARNING] Invalid fall_score: {fall_score}, clamping to 0-1 range")
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
        result = history_collection.insert_one(entry)

        socketio.emit('new_detection', entry)
        print(f"[DEBUG] Detection result: fall={fall_score > 0.7}, score={fall_score}, saved to {filepath}")
        return jsonify({"fall": fall_score > 0.7, "score": float(fall_score)})
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý ảnh: {str(e)}")
        return jsonify({'fall': False, 'error': str(e)}), 500

@app.route('/stream')
def stream():
    global active_streams
    try:
        if active_streams >= MAX_STREAM_CLIENTS:
            print("[ERROR] Maximum stream clients reached")
            return "<p>Lỗi: Đã đạt số lượng client tối đa</p>", 429

        # Kiểm tra trạng thái thiết bị
        if not check_device_status() or device_status["mode"] != "stream":
            response = send_request_to_esp32(
                f"http://{ESP32_CAM_IP}:81/set_mode",
                data={'mode': 'stream'},
                timeout=15
            )
            if not response:
                return "<p>Lỗi: ESP32 không ở chế độ stream</p>", 500

        active_streams += 1
        def proxy_stream():
            global active_streams
            try:
                url = f"http://{ESP32_CAM_IP}:81/stream"
                print(f"[DEBUG] Starting stream proxy to {url}")
                while True:
                    try:
                        req = requests.get(url, stream=True, timeout=30)
                        print(f"[DEBUG] Stream connected, status: {req.status_code}")
                        for chunk in req.iter_content(chunk_size=1024):
                            if chunk:
                                yield chunk
                    except requests.exceptions.RequestException as e:
                        print(f"[WARNING] Stream disconnected: {str(e)}, attempting to reconnect...")
                        time.sleep(2)
                        continue
            except Exception as e:
                print(f"[ERROR] Stream proxy failed: {str(e)}")
                active_streams -= 1
                raise
            finally:
                print("[DEBUG] Stream proxy closed")
                active_streams -= 1
        return Response(proxy_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"[ERROR] Lỗi khi proxy stream: {str(e)}")
        active_streams -= 1
        return "<p>Lỗi: Không thể kết nối camera</p>", 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    try:
        return send_from_directory('static/audio', filename)
    except FileNotFoundError:
        print(f"[WARNING] Audio file {filename} not found")
        return jsonify({'status': 'error', 'message': 'Audio file not found'}), 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)