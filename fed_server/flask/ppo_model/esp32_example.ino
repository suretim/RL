// ESP32 Plant HVAC Control Example
#include <TensorFlowLite.h>
#include "actor.h"
#include "critic.h"

// 錯誤報告器
tflite::MicroErrorReporter error_reporter;

// 加載模型
const tflite::Model* actor_model = tflite::GetModel(actor_model);
const tflite::Model* critic_model = tflite::GetModel(critic_model);

// 創建解釋器
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter actor_interpreter(actor_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);
static tflite::MicroInterpreter critic_interpreter(critic_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);

// 張量緩衝區（根據模型大小調整）
const int tensor_arena_size = 20 * 1024;
uint8_t tensor_arena[tensor_arena_size];

void setup() {
  Serial.begin(115200);

  // 分配內存
  if (actor_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Actor模型內存分配失敗");
    return;
  }

  if (critic_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Critic模型內存分配失敗");
    return;
  }

  Serial.println("ESP32 Plant HVAC Controller Ready");
}

void get_action(float state[5], float action[4]) {
  // 獲取輸入張量
  TfLiteTensor* input = actor_interpreter.input(0);

  // 複製狀態數據
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // 運行推理
  if (actor_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Actor推理失敗");
    return;
  }

  // 獲取輸出並二值化
  TfLiteTensor* output = actor_interpreter.output(0);
  for (int i = 0; i < 4; i++) {
    action[i] = (output->data.f[i] > 0.5) ? 1.0 : 0.0;
  }
}

float get_value(float state[5]) {
  // 獲取輸入張量
  TfLiteTensor* input = critic_interpreter.input(0);

  // 複製狀態數據
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // 運行推理
  if (critic_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Critic推理失敗");
    return 0.0;
  }

  // 獲取輸出
  TfLiteTensor* output = critic_interpreter.output(0);
  return output->data.f[0];
}

void loop() {
  // 示例狀態數據
  float state[5] = {0.0, 25.0, 0.5, 500.0, 600.0};
  float action[4];

  // 獲取動作
  get_action(state, action);

  // 獲取值預測
  float value = get_value(state);

  // 輸出結果
  Serial.print("Action: ");
  for (int i = 0; i < 4; i++) {
    Serial.print(action[i]);
    Serial.print(" ");
  }
  Serial.print("Value: ");
  Serial.println(value);

  delay(1000);
}
