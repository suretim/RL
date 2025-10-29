#pragma once


#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


struct CLASSIFIER_Prams {
    int infer_case; // 0: PPO, 1: META
    int seq_len;
    int feature_dim;
    int num_classes;
 };

struct ModelContext {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    // 公用的 Op resolver（根据模型需求配置）
    tflite::MicroMutableOpResolver<24> micro_op_resolver;
    // tensor arena
    uint8_t* tensor_arena = nullptr;
    size_t tensor_arena_size = 0;
    CLASSIFIER_Prams classifier_params;

}; 