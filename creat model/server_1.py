# === Server-side Python Flask: Nhận ảnh từ ESP32-CAM và phát hiện ngã ===
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load mô hình phát hiện ngã
print("[INFO] Loading fall_detection TFLite model...")
interpreter = tf.lite.Interpreter(model_path="fall_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[INFO] Model loaded successfully.")

@app.route('/fall_detect', methods=['POST'])
def fall_detect():
    try:
        print("🟢 [SERVER] Nhận ảnh từ ESP32-CAM...")

        # Nhận dữ liệu ảnh từ ESP32 gửi sang
        raw = np.frombuffer(request.data, dtype=np.uint8)

        # Chuyển đổi uint8 → int8 đúng như ESP32 xử lý
        image = raw ^ 0x80
        image = image.astype(np.int8).reshape((96, 96, 1))

        # Chạy mô hình
        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"[DEBUG] Output model: {output_data}")

        # Xử lý fall score
        if isinstance(output_data, np.ndarray) and output_data.size == 1:
            fall_score = output_data.item()
        else:
            fall_score = output_data[1] if len(output_data) > 1 else 0.0

        print(f"🔍 [SERVER] Fall score: {fall_score:.2f}")

        if fall_score > 0.7:
            print("🔴 [SERVER] PHÁT HIỆN NGƯỜI NGÃ!")
        else:
            print("✅ [SERVER] Không phát hiện ngã.")

        print("✅ [SERVER] Đã xử lý và gửi kết quả về ESP32.\n")

        return jsonify({'fall': bool(fall_score > 0.7)})

    except Exception as e:
        print("❌ [ERROR] Lỗi khi xử lý ảnh nhận từ ESP32:")
        print(e)
        return jsonify({'fall': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)