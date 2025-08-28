#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// -------------------------
// 模型数据 (TFLite flatbuffer)
// -------------------------
extern const unsigned char lstm_encoder_contrastive_tflite[];
extern const unsigned int lstm_encoder_contrastive_tflite_len;

extern const unsigned char meta_lstm_classifier_tflite[];
extern const unsigned int meta_lstm_classifier_tflite_len;

// -------------------------
// TensorArena
// -------------------------
constexpr int kTensorArenaSize = 450 * 1024; // 调整大小
uint8_t tensor_arena[kTensorArenaSize];

// -------------------------
// 运行推理
// -------------------------
float run_inference(float* input_seq, int seq_len, int num_feats, float* out_logits) {
    // 1) 加载模型
    const tflite::Model* model = tflite::GetModel(meta_lstm_classifier_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("模型版本不匹配\n");
        return -1;
    }

    // 2) Ops Resolver
    tflite::AllOpsResolver resolver;

    // 3) Interpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // 4) Allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("分配张量失败\n");
        return -1;
    }

    // 5) 填充输入
    TfLiteTensor* input = interpreter.input(0);
    memcpy(input->data.f, input_seq, seq_len * num_feats * sizeof(float));

    // 6) 推理
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("推理失败\n");
        return -1;
    }

    // 7) 读取输出
    TfLiteTensor* output = interpreter.output(0);
    int num_classes = output->dims->data[1];
    memcpy(out_logits, output->data.f, num_classes * sizeof(float));

    return 0; // 成功
}

// -------------------------
// 示例
// -------------------------
int main() {
    // 假设输入 seq_len x num_feats
    float input_seq[SEQ_LEN * NUM_FEATS] = {0};  // 从传感器读取
    float logits[NUM_CLASSES];

    int ret = run_inference(input_seq, SEQ_LEN, NUM_FEATS, logits);
    if (ret == 0) {
        printf("预测 logits: ");
        for (int i=0;i<NUM_CLASSES;i++) printf("%.3f ", logits[i]);
        printf("\n");
    }

    return 0;
}
