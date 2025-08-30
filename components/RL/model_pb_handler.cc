#include "model_pb_handler.h"
 
#include <string.h>
#include <stdio.h>


#include "pb.h"
#include "pb_decode.h"
#include "model.pb.h"  // 包含你生成的 message 类型定义

// model_pb_handler.c
#if 1

bool decode_float_array(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    ParsedModelParams *out = (ParsedModelParams *)(*arg);

    while (stream->bytes_left) {
        if (out->value_count >= MAX_PARAM_LENGTH) {
            printf("参数溢出，最多 %d 个\n", MAX_PARAM_LENGTH);
            return false;
        }

        // 解码一个 float（用 pb_decode_fixed32 或 pb_decode）
        if (!pb_decode_fixed32(stream, &(out->values[out->value_count]))) {
            return false;
        }

        out->value_count++;
    }

    return true;
}



/* 私有回调函数 - 只在当前文件使用 */
#if 1
bool decode_values_cb(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    ParsedModelParams *out = (ParsedModelParams *)(*arg);

    while (stream->bytes_left > 0) {
        if (out->value_count >= MAX_PARAM_LENGTH) {
            return false;
        }
        float value;
        if (!pb_decode_fixed32(stream, &value)) {
            return false;
        }
        out->values[out->value_count++] = value;
    }
    return true;
}
#else
static bool decode_values_cb(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    float **values = (float **)*arg;
    size_t count = stream->bytes_left / sizeof(float);
    *values =(float *) malloc(count * sizeof(float));
    return pb_read(stream, (pb_byte_t*)*values, count * sizeof(float));
}
#endif
/* 公开的解码接口 */
bool decode_model_params(  uint8_t *data, size_t len, ParsedModelParams *out_params) {
 #if 1
    ModelParams msg=ModelParams_init_zero;
    //memset(&msg, 0, sizeof(msg));
    memset(out_params, 0, sizeof(ParsedModelParams));

    msg.values.funcs.decode = decode_values_cb;
    msg.values.arg = out_params;
    pb_istream_t stream = pb_istream_from_buffer(data, len);
    bool status = pb_decode(&stream, ModelParams_fields,  &msg);

    if (status) {
        out_params->param_type = msg.param_type;
        out_params->client_id = msg.client_id;
    }
#else
    float *value_buf = NULL;
    msg->values.funcs.decode = decode_values_cb;  // 绑定回调
    msg->values.arg = &value_buf;

    pb_istream_t stream = pb_istream_from_buffer(data, len);
    //bool status = pb_decode(&stream, ModelParams_fields, msg);
    bool status = decode_float_array(&stream, ModelParams_fields, msg);
    
    // for (int i = 0; i < value_buf.value_count; ++i) {
    // printf("Value %d: %f\n", i, value_buf.values[i]);
    // }
    out_params->param_type = msg->param_type;
    out_params->value_count = value_buf.value_count;
    memcpy(out_params->values,(float *) value_buf.values  ,MAX_PARAM_LENGTH);//sizeof(float) * msg->values.size);
    out_params->client_id = msg->client_id;
    if (status && value_buf) {
        // 使用解码后的数据...
        free(value_buf);  // 释放内存
    }
    #endif
    return status;
}
#else
bool decode_model_params(const uint8_t *payload, size_t length, ParsedModelParams *out_params) {
    ModelParams msg = ModelParams_init_zero;

    pb_istream_t stream = pb_istream_from_buffer(payload, length);

    // 临时数组接收 repeated float
    float value_buf[MAX_PARAM_LENGTH] = {0};
    msg.values.values = value_buf;
    msg.values.size = MAX_PARAM_LENGTH;

    if (!pb_decode(&stream, ModelParams_fields, &msg)) {
        printf("Decoding failed: %s\n", PB_GET_ERROR(&stream));
        return false;
    }

    out_params->param_type = msg.param_type;
    out_params->value_count = msg.values.size;
    memcpy(out_params->values, msg.values.values, sizeof(float) * msg.values.size);
    out_params->client_id = msg.client_id;

    return true;
}
#endif