#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include <WiFiClient.h>
#include <WebServer.h>

// WiFi và server
const char *ssid = "Nguyễn Hữu Dũng";
const char *password = "nguyenhuudung2003";
const char *server_url = "http://192.168.0.103:5000/fall_detect"; // Địa chỉ server Flask
const char *api_key = "08112003";                                 // Khóa API
WebServer stream_server(81);                                      // Server MJPEG trên cổng 81

// TensorFlow Lite
#include "TensorFlowLite_ESP32.h"
#include "person_detect_model_data.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TensorFlow variables
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;

constexpr int kTensorArenaSize = 100 * 1024; // 100KB cho tensor arena
static uint8_t *tensor_arena = nullptr;

// Model settings
float latest_person_score = 0.0;
bool fall_detected = false;
bool streaming_active = false;
unsigned long stream_start_time = 0;
const unsigned long STREAM_TIMEOUT = 30000; // 30 giây timeout

// Trạng thái camera
bool camera_initialized = false;

// === CAMERA FUNCTIONS ===
bool initCamera(bool for_streaming = false)
{
  if (camera_initialized)
  {
    esp_camera_deinit();
    camera_initialized = false;
    delay(100); // Đảm bảo tài nguyên được giải phóng
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
  config.xclk_freq_hz = 20000000;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if (for_streaming)
  {
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QQVGA; // 160x120
    config.jpeg_quality = 15;            // Giảm chất lượng để tiết kiệm băng thông
    config.fb_count = 1;
  }
  else
  {
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
    config.fb_count = 1;
  }

  Serial.println("🔧 Initializing camera...");
  int retry_count = 0;
  const int max_retries = 3;
  esp_err_t err;
  while (retry_count < max_retries)
  {
    err = esp_camera_init(&config);
    if (err == ESP_OK)
    {
      Serial.println("✅ Camera initialized for " + String(for_streaming ? "streaming" : "detection"));
      camera_initialized = true;
      return true;
    }
    Serial.printf("❌ Camera init failed with error 0x%x\n", err);
    esp_camera_deinit(); // Buộc deinit để dọn dẹp tài nguyên
    delay(100);          // Chờ trước khi thử lại
    retry_count++;
  }
  Serial.println("❌ Camera init failed after retries");
  return false;
}

camera_fb_t *captureGrayscaleFrame()
{
  if (!initCamera(false))
  {
    Serial.println("❌ Failed to initialize camera for grayscale capture");
    return nullptr;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
  {
    Serial.println("❌ Failed to capture grayscale frame");
    esp_camera_deinit();
    camera_initialized = false;
    return nullptr;
  }
  Serial.println("✅ Grayscale frame captured");
  return fb;
}

// === MJPEG STREAMING ===
void handleStream()
{
  WiFiClient client = stream_server.client();
  if (!streaming_active)
  {
    streaming_active = true;
    stream_start_time = millis();
    if (!initCamera(true))
    {
      Serial.println("❌ Failed to initialize camera for streaming");
      streaming_active = false;
      client.stop();
      return;
    }
  }
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.write(response.c_str());

  while (client.connected() && streaming_active)
  {
    if (millis() - stream_start_time > STREAM_TIMEOUT)
    {
      Serial.println("❌ Streaming timeout, stopping...");
      break;
    }
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("❌ Camera capture failed for stream");
      camera_initialized = false;
      break;
    }
    String boundary = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
    client.write(boundary.c_str(), boundary.length());
    client.write(fb->buf, fb->len);
    client.write("\r\n", 2);
    esp_camera_fb_return(fb);
    delay(100); // Giảm tốc độ khung hình để tránh quá tải
  }
  streaming_active = false;
  client.stop();
  initCamera(false); // Khôi phục chế độ detection
}

// === MÔ HÌNH PHÁT HIỆN NGƯỜI ===
void runPersonDetection(camera_fb_t *fb)
{
  if (!fb || fb->width != kNumCols || fb->height != kNumRows)
  {
    Serial.printf("❌ Invalid frame buffer. Got: %dx%d, expected: %dx%d\n", fb->width, fb->height, kNumCols, kNumRows);
    return;
  }

  for (int i = 0; i < kNumCols * kNumRows; i++)
  {
    input->data.int8[i] = static_cast<int8_t>(fb->buf[i] - 128);
  }

  Serial.println("🔍 Input data sample:");
  for (int i = 0; i < 10; i++)
  {
    Serial.printf("%d ", input->data.int8[i]);
  }
  Serial.println();

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    error_reporter->Report("❌ Model inference failed with status: %d", invoke_status);
    Serial.println("❌ Model inference failed");
    return;
  }

  TfLiteTensor *output = interpreter->output(0);
  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];

  latest_person_score = (person_score + 128) / 255.0;
  float no_person_score_float = (no_person_score + 128) / 255.0;

  Serial.printf("🔍 Person score: %.2f, No person score: %.2f\n", latest_person_score, no_person_score_float);
}

// === GỬI ẢNH VỀ SERVER ===
bool sendImageToServer(camera_fb_t *fb, String location = "Phòng Ngủ")
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("❌ WiFi disconnected, reconnecting...");
    WiFi.reconnect();
    unsigned long start_time = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start_time < 5000)
    {
      delay(500);
      Serial.print(".");
    }
    if (WiFi.status() != WL_CONNECTED)
    {
      Serial.println("❌ Reconnect failed");
      return false;
    }
    Serial.println("✅ WiFi reconnected");
  }

  // Kiểm tra kết nối đến server trước khi gửi
  WiFiClient client;
  if (!client.connect("192.168.0.103", 5000))
  {
    Serial.println("❌ Server ping failed");
    return false;
  }
  client.stop();

  HTTPClient http;
  http.setTimeout(5000); // Đặt timeout 5 giây
  http.begin(server_url);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Location", location);
  http.addHeader("X-API-Key", api_key);

  int httpCode = http.POST(fb->buf, fb->len);
  bool fall = false;
  if (httpCode > 0)
  {
    if (httpCode == HTTP_CODE_OK)
    {
      String res = http.getString();
      Serial.println("📤 Server response: " + res);
      fall = res.indexOf("\"fall\":true") >= 0;
    }
    else
    {
      Serial.printf("❌ HTTP POST failed, code: %d\n", httpCode);
    }
  }
  else
  {
    Serial.printf("❌ HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
  return fall;
}

// === SETUP ===
void setup()
{
  Serial.begin(115200);
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());

  if (!initCamera(false))
  {
    Serial.println("❌ Camera init failed, halting...");
    while (1)
      ;
  }

  // Khởi tạo error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("❌ Model version mismatch: model version %d, expected %d",
                           model->version(), TFLITE_SCHEMA_VERSION);
    Serial.println("❌ Model version mismatch");
    while (1)
      ;
  }

  tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena)
  {
    Serial.println("❌ Tensor arena allocation failed");
    while (1)
      ;
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
  if (allocate_status != kTfLiteOk)
  {
    error_reporter->Report("❌ Tensor allocation failed with status: %d", allocate_status);
    Serial.println("❌ Tensor allocation failed");
    while (1)
      ;
  }
  input = interpreter->input(0);

  Serial.printf("📦 Input tensor type: %d\n", input->type);
  TfLiteTensor *output = interpreter->output(0);
  Serial.printf("📦 Output tensor type: %d\n", output->type);
  Serial.printf("🔍 Input tensor dims: %d x %d x %d\n",
                input->dims->data[1], input->dims->data[2], input->dims->data[3]);

  stream_server.on("/stream", handleStream);
  stream_server.begin();
  Serial.println("✅ MJPEG stream server started on port 81");

  Serial.println("✅ ESP32-CAM initialized");
}

// === LOOP ===
void loop()
{
  stream_server.handleClient();

  if (!streaming_active)
  {
    if (!camera_initialized)
    {
      Serial.println("❌ Camera not initialized, attempting to reinitialize...");
      if (!initCamera(false))
      {
        Serial.println("❌ Reinitialization failed");
        delay(2000); // Tăng delay để giảm tải
        return;
      }
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("❌ Camera capture failed");
      camera_initialized = false;
      delay(2000); // Tăng delay để giảm tải
      return;
    }

    runPersonDetection(fb);

    if (latest_person_score > 0.7)
    {
      Serial.println("👤 Person detected, sending to server...");
      camera_fb_t *fb_grayscale = captureGrayscaleFrame();
      if (fb_grayscale)
      {
        fall_detected = sendImageToServer(fb_grayscale, "Phòng Ngủ");
        digitalWrite(4, fall_detected ? HIGH : LOW);
        Serial.println(fall_detected ? "🔴 Fall detected!" : "✅ No fall detected");
        esp_camera_fb_return(fb_grayscale);
      }
      else
      {
        Serial.println("❌ Failed to capture grayscale image for sending");
      }
    }
    else
    {
      fall_detected = false;
      digitalWrite(4, LOW);
    }

    esp_camera_fb_return(fb);
  }
  delay(2000); // Tăng delay để giảm tải
}