#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include <WiFiClient.h>
#include <WebServer.h>
#include "esp_heap_caps.h"
#include "esp_system.h"

// WiFi and server
const char* ssid = "Nguyễn Hữu Dũng";
const char* password = "nguyenhuudung2003";
const char* server_url = "http://192.168.0.100:5000/fall_detect";
const char* report_ip_url = "http://192.168.0.100:5000/api/report_ip";
const char* api_key = "08112003";
WebServer stream_server(81);

// Static IP configuration
IPAddress local_IP(192, 168, 0, 101);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);
IPAddress secondaryDNS(8, 8, 4, 4);

// TensorFlow Lite
#include "TensorFlowLite_ESP32.h"
#include "person_detect_model_data.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TensorFlow variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t* tensor_arena = nullptr;
bool tflite_initialized = false;

// Model settings
float latest_person_score = 0.0;
bool fall_detected = false;
bool streaming_active = true;
String device_mode = "stream";
bool camera_initialized = false;
unsigned long last_detection_time = 0;
const unsigned long DETECTION_INTERVAL = 2000;
unsigned long last_ip_report_time = 0;
const unsigned long IP_REPORT_INTERVAL = 10000;

// === CAMERA FUNCTIONS ===
bool initCamera(bool for_streaming = true) {
  if (camera_initialized) {
    esp_camera_deinit();
    camera_initialized = false;
    delay(200);
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 10000000;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 1;

  if (for_streaming) {
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 30;
  } else {
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
  }

  Serial.printf("🔍 Free PSRAM before camera init: %u bytes\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Serial.printf("🔍 Free heap before camera init: %u bytes\n", esp_get_free_heap_size());
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x, attempting recovery...\n", err);
    config.fb_count = 1;
    config.frame_size = for_streaming ? FRAMESIZE_QQVGA : FRAMESIZE_96X96;
    err = esp_camera_init(&config);
    if (err != ESP_OK) {
      Serial.printf("❌ Camera reinitialization failed with error 0x%x\n", err);
      return false;
    }
  }
  Serial.println("✅ Camera initialized for " + String(for_streaming ? "streaming" : "detection"));
  camera_initialized = true;
  Serial.printf("🔍 Free PSRAM after camera init: %u bytes\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  return true;
}

camera_fb_t* captureGrayscaleFrame() {
  if (!camera_initialized || !setCameraFormat(false)) {
    Serial.println("❌ Camera not ready for grayscale capture");
    return nullptr;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Failed to capture grayscale frame");
    camera_initialized = false;
    return nullptr;
  }
  Serial.println("✅ Grayscale frame captured");
  return fb;
}

bool setCameraFormat(bool for_streaming) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) {
    Serial.println("❌ Failed to get camera sensor");
    return false;
  }

  if (for_streaming) {
    s->set_pixformat(s, PIXFORMAT_JPEG);
    s->set_framesize(s, FRAMESIZE_QQVGA);
    s->set_quality(s, 30);
  } else {
    s->set_pixformat(s, PIXFORMAT_GRAYSCALE);
    s->set_framesize(s, FRAMESIZE_96X96);
  }
  delay(100);
  return true;
}

// === MJPEG STREAMING ===
void handleStream() {
  WiFiClient client = stream_server.client();
  if (!client) {
    Serial.println("❌ No client connected for streaming");
    return;
  }

  if (!streaming_active) {
    client.stop();
    Serial.println("✅ Streaming stopped due to inactive state");
    return;
  }

  if (!camera_initialized || !setCameraFormat(true)) {
    Serial.println("❌ Failed to prepare camera for streaming");
    client.stop();
    return;
  }
  Serial.println("✅ Streaming started");

  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  unsigned long lastFrameTime = millis();
  while (client.connected() && streaming_active && device_mode == "stream") {
    if (millis() - lastFrameTime < 50) {
      continue; // Giới hạn frame rate
    }
    lastFrameTime = millis();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("❌ Camera capture failed for stream");
      camera_initialized = false;
      break;
    }
    String boundary = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
    client.write(boundary.c_str(), boundary.length());
    client.write(fb->buf, fb->len);
    client.write("\r\n", 2);
    esp_camera_fb_return(fb);

    // Kiểm tra client còn kết nối không
    if (!client.connected()) {
      Serial.println("❌ Client disconnected during streaming");
      break;
    }
  }

  streaming_active = false;
  client.stop();
  Serial.println("✅ Streaming stopped");
}

// === MODE HANDLING ===
void handleSetMode() {
  Serial.println("📡 Received /set_mode request");
  if (!stream_server.hasArg("mode")) {
    Serial.println("❌ No mode argument provided");
    stream_server.send(400, "application/json", "{\"error\":\"No mode provided\"}");
    return;
  }

  String mode = stream_server.arg("mode");
  Serial.println("🔍 Received mode: " + mode);

  if (mode == "stream" || mode == "detection") {
    if (device_mode != mode) {
      Serial.println("🔄 Changing mode from " + device_mode + " to " + mode);
      streaming_active = false;

      // Đợi streaming dừng hoàn toàn
      unsigned long stopTimeout = millis();
      while (millis() - stopTimeout < 2000) {
        if (!streaming_active) break;
        delay(10);
      }
      Serial.println("✅ Streaming fully stopped");

      // Giải phóng tài nguyên camera
      if (camera_initialized) {
        esp_camera_deinit();
        camera_initialized = false;
        Serial.println("✅ Camera deinitialized to free memory");
        delay(200);
      }

      device_mode = mode;
      Serial.println("✅ Set device mode to: " + device_mode);
      streaming_active = (device_mode == "stream");

      if (device_mode == "stream") {
        if (!initCamera(true)) {
          Serial.println("❌ Failed to prepare camera for streaming");
          streaming_active = false;
          device_mode = "detection";
        }
      } else if (device_mode == "detection") {
        streaming_active = false;
        if (!initCamera(false)) {
          Serial.println("❌ Failed to prepare camera for detection");
          device_mode = "stream";
        }
        if (!tflite_initialized && !initTensorFlow()) {
          Serial.println("❌ TensorFlow reinitialization failed");
        }
      }
    } else {
      Serial.println("🔄 Mode unchanged: " + device_mode);
    }
    stream_server.send(200, "application/json", "{\"status\":\"success\",\"mode\":\"" + mode + "\"}");
  } else {
    Serial.println("❌ Invalid mode received");
    stream_server.send(400, "application/json", "{\"error\":\"Invalid mode\"}");
  }
}

// === PERSON DETECTION ===
void runPersonDetection(camera_fb_t* fb) {
  if (!tflite_initialized) {
    Serial.println("❌ TensorFlow Lite not initialized, skipping detection");
    return;
  }

  if (!fb || fb->width != kNumCols || fb->height != kNumRows) {
    Serial.printf("❌ Invalid frame: %dx%d, expected: %dx%d\n", fb->width, fb->height, kNumCols, kNumRows);
    return;
  }

  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = static_cast<int8_t>(fb->buf[i] - 128);
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("❌ Inference failed: %d", invoke_status);
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];

  latest_person_score = (person_score + 128) / 255.0;
  Serial.printf("🔍 Person score: %.2f\n", latest_person_score);
}

// === INIT TENSORFLOW ===
bool initTensorFlow() {
  if (tflite_initialized) return true;

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("❌ Model version mismatch");
    return false;
  }

  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.println("❌ Tensor arena allocation failed");
    return false;
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("❌ Tensor allocation failed, required: %d, available: %d",
                           interpreter->arena_used_bytes(), kTensorArenaSize);
    heap_caps_free(tensor_arena);
    tensor_arena = nullptr;
    return false;
  }

  input = interpreter->input(0);
  tflite_initialized = true;
  Serial.println("✅ TensorFlow Lite initialized");
  return true;
}

// === REPORT IP ===
void reportIPToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi not connected, skipping IP report");
    return;
  }

  HTTPClient http;
  http.setTimeout(15000);
  http.setConnectTimeout(5000);
  if (!http.begin(report_ip_url)) {
    Serial.println("❌ Failed to initialize HTTP client");
    return;
  }

  http.addHeader("Content-Type", "application/json");
  String payload = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"mac\":\"" + WiFi.macAddress() + "\"}";
  Serial.println("📤 Reporting IP: " + payload);

  int httpCode = http.POST(payload);
  if (httpCode == HTTP_CODE_OK) {
    Serial.println("✅ Reported IP to server");
  } else {
    Serial.printf("❌ HTTP POST failed, code: %d\n", httpCode);
  }

  http.end();
  delay(100);
}

// === SEND IMAGE ===
bool sendImageToServer(camera_fb_t* fb, String location = "Phòng Ngủ") {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected, reconnecting...");
    WiFi.reconnect();
    delay(5000);
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("❌ Reconnect failed");
      return false;
    }
    Serial.println("✅ WiFi reconnected");
    reportIPToServer();
  }

  HTTPClient http;
  http.setTimeout(20000);
  http.setConnectTimeout(5000);
  if (!http.begin(server_url)) {
    Serial.println("❌ Failed to initialize HTTP client");
    return false;
  }

  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Location", location);
  http.addHeader("X-API-Key", api_key);

  Serial.println("📤 Sending image...");
  int httpCode = http.POST(fb->buf, fb->len);
  bool fall = false;
  if (httpCode == HTTP_CODE_OK) {
    String res = http.getString();
    Serial.println("📤 Server response: " + res);
    fall = res.indexOf("\"fall\":true") >= 0;
    Serial.println(fall ? "🔴 Fall detected!" : "✅ No fall detected");
  } else {
    Serial.printf("❌ HTTP POST failed, code: %d\n", httpCode);
  }
  http.end();
  return fall;
}

// === CHECK WIFI ===
void checkWiFi() {
  static unsigned long last_check = 0;
  const unsigned long CHECK_INTERVAL = 10000;

  if (millis() - last_check < CHECK_INTERVAL) return;
  last_check = millis();

  Serial.printf("🔍 WiFi RSSI: %ld dBm\n", WiFi.RSSI());
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected, reconnecting...");
    WiFi.reconnect();
    delay(5000);
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("✅ WiFi reconnected: " + WiFi.localIP().toString());
      reportIPToServer();
    } else {
      Serial.println("❌ Reconnect failed");
    }
  }
}

// === SETUP ===
void setup() {
  Serial.begin(115200);
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) {
    Serial.println("❌ Failed to configure static IP");
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());
  reportIPToServer();

  if (!initCamera(true)) {
    Serial.println("❌ Camera init failed, halting...");
    while (1);
  }

  if (!initTensorFlow()) {
    Serial.println("❌ TensorFlow initialization failed, halting...");
    while (1);
  }

  stream_server.on("/stream", handleStream);
  stream_server.on("/set_mode", handleSetMode);
  stream_server.begin();
  Serial.println("✅ MJPEG stream server started on port 81");
}

// === LOOP ===
void loop() {
  checkWiFi();
  stream_server.handleClient();

  if (device_mode == "detection" && !streaming_active && tflite_initialized) {
    if (!camera_initialized || !initCamera(false)) {
      Serial.println("❌ Camera not ready, attempting reinitialization...");
      if (!initCamera(false)) {
        Serial.println("❌ Reinitialization failed");
        delay(2000);
        return;
      }
    }

    unsigned long current_time = millis();
    if (current_time - last_detection_time >= DETECTION_INTERVAL) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("❌ Camera capture failed");
        camera_initialized = false;
        delay(2000);
        return;
      }

      runPersonDetection(fb);
      esp_camera_fb_return(fb);

      if (latest_person_score > 0.7) {
        Serial.println("👤 Person detected, sending to server...");
        camera_fb_t* fb_grayscale = captureGrayscaleFrame();
        if (fb_grayscale) {
          fall_detected = sendImageToServer(fb_grayscale, "Phòng Ngủ");
          esp_camera_fb_return(fb_grayscale);
          digitalWrite(4, fall_detected ? HIGH : LOW);
          Serial.println(fall_detected ? "🔴 Fall detected!" : "✅ No fall detected");
        }
      } else {
        fall_detected = false;
        digitalWrite(4, LOW);
      }

      last_detection_time = current_time;
    }
  }
}