#pragma once
#include "model.pb.h"

#define MAX_PARAM_LENGTH 1024  // 可按需要扩大

typedef struct {
    ParamType param_type;
    float values[MAX_PARAM_LENGTH];
    size_t value_count;
    int32_t client_id;
} ParsedModelParams;

//bool decode_model_params(const uint8_t *payload, size_t length, ParsedModelParams *out_params);
bool decode_model_params(  uint8_t *data, size_t len, ParsedModelParams *out_params) ;

