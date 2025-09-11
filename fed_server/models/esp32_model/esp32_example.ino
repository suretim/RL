// ESP32 Plant HVAC Control Example
#include <TensorFlowLite.h>
#include "actor.h"
#include "critic.h"

// �e�`�����
tflite::MicroErrorReporter error_reporter;

// ���dģ��
const tflite::Model* actor_model = tflite::GetModel(actor_model);
const tflite::Model* critic_model = tflite::GetModel(critic_model);

// ���������
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter actor_interpreter(actor_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);
static tflite::MicroInterpreter critic_interpreter(critic_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);

// �������n�^������ģ�ʹ�С�{����
const int tensor_arena_size = 20 * 1024;
uint8_t tensor_arena[tensor_arena_size];

void setup() {
  Serial.begin(115200);

  // ����ȴ�
  if (actor_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Actorģ�̓ȴ����ʧ��");
    return;
  }

  if (critic_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Criticģ�̓ȴ����ʧ��");
    return;
  }

  Serial.println("ESP32 Plant HVAC Controller Ready");
}

void get_action(float state[5], float action[4]) {
  // �@ȡݔ�돈��
  TfLiteTensor* input = actor_interpreter.input(0);

  // �}�u��B����
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // �\������
  if (actor_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Actor����ʧ��");
    return;
  }

  // �@ȡݔ���K��ֵ��
  TfLiteTensor* output = actor_interpreter.output(0);
  for (int i = 0; i < 4; i++) {
    action[i] = (output->data.f[i] > 0.5) ? 1.0 : 0.0;
  }
}

float get_value(float state[5]) {
  // �@ȡݔ�돈��
  TfLiteTensor* input = critic_interpreter.input(0);

  // �}�u��B����
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // �\������
  if (critic_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Critic����ʧ��");
    return 0.0;
  }

  // �@ȡݔ��
  TfLiteTensor* output = critic_interpreter.output(0);
  return output->data.f[0];
}

void loop() {
  // ʾ����B����
  float state[5] = {0.0, 25.0, 0.5, 500.0, 600.0};
  float action[4];

  // �@ȡ����
  get_action(state, action);

  // �@ȡֵ�A�y
  float value = get_value(state);

  // ݔ���Y��
  Serial.print("Action: ");
  for (int i = 0; i < 4; i++) {
    Serial.print(action[i]);
    Serial.print(" ");
  }
  Serial.print("Value: ");
  Serial.println(value);

  delay(1000);
}
