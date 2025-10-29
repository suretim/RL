#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//#include "model.h"          //aa_bb_model_f32_lite  84716

#include <esp_log.h>
#include "esp_timer.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"

#include "esp_nn.h"

#include "tflm.h"

#define INPUT_IMAGE_WIDTH       320
#define INPUT_IMAGE_HEIGHT      240
#define MODEL_INPUT_CHANNELS    3
#define NUM_CLASSES             2
#define THRESHOLD               0.5
const unsigned int MODEL_INPUT_WIDTH = 96; //224 160 96;
const unsigned int MODEL_INPUT_HEIGHT = 96;  
static const char *TAG = "tflm";
const float detect_threshold = 0.08;
const float conf_threshold = 0.1f;
// extern const unsigned char  g_model[];
// extern const unsigned int   MODEL_INPUT_WIDTH;
// extern const unsigned int   MODEL_INPUT_HEIGHT;

extern const unsigned char _binary_best_float32_tflite_start[] asm("_binary_best_float32_tflite_start");
extern const unsigned char _binary_best_float32_tflite_end[]   asm("_binary_best_float32_tflite_end");
const size_t   esp32_best_float32_tflite_len=_binary_best_float32_tflite_end-_binary_best_float32_tflite_start;

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

static uint8_t *tensor_arena = nullptr;
}

void tflite_init(void) 
{
    unsigned int kTensorArenaSize;
    model = tflite::GetModel(_binary_best_float32_tflite_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) 
    {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    if(     MODEL_INPUT_WIDTH == 224) kTensorArenaSize = 3 * 1024 * 1024;
    else if(MODEL_INPUT_WIDTH == 160) kTensorArenaSize = 3 * 1024 * 1024 / 2;
    else                              kTensorArenaSize = 3 * 1024 * 1024 / 4;
    if (tensor_arena == NULL) 
    {        
        tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }

    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddConv2D();
    resolver.AddSoftmax();
    resolver.AddConcatenation();
    resolver.AddStridedSlice();
    resolver.AddReshape();
    resolver.AddTranspose();
    resolver.AddPad();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddSub();
    resolver.AddMaxPool2D();
    resolver.AddResizeNearestNeighbor();
    resolver.AddLogistic(); 
    resolver.AddQuantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    printf("Input shape: [%d, %d, %d, %d]\n", 
        input->dims->data[0], input->dims->data[1], 
        input->dims->data[2], input->dims->data[3]);
    printf("Input type: %s\n", 
        (input->type == kTfLiteInt8) ? "INT8" : "FLOAT32");
    if (input->quantization.type == kTfLiteAffineQuantization) {
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        printf("Quantization: scale=%.6f, zero_point=%d\n", scale, zero_point);
    }

    output = interpreter->output(0);

    printf("output shape: [%d, %d, %d, %d]\n", 
        output->dims->data[0], output->dims->data[1], 
        output->dims->data[2], output->dims->data[3]);
    printf("output type: %s\n", 
        (output->type == kTfLiteInt8) ? "INT8" : "FLOAT32");
    if (output->quantization.type == kTfLiteAffineQuantization) {
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        printf("Quantization: scale=%.6f, zero_point=%d\n", scale, zero_point);
    }

//Input shape: [1, 160, 160, 3]
//Input type: FLOAT32
//output shape: [1, 6, 525, -34594]
//output type: FLOAT32    
}

//resize_image(image_data, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 
//        resized_image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS);
    
void resize_image(const uint8_t *src, int src_width, int src_height, 
                  uint8_t *dst, int dst_width, int dst_height, int channels) {
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            // BGR → RGB 转换
            dst[(y * dst_width + x) * 3 + 0] = src[(src_y * src_width + src_x) * 3 + 2]; // R
            dst[(y * dst_width + x) * 3 + 1] = src[(src_y * src_width + src_x) * 3 + 1]; // G
            dst[(y * dst_width + x) * 3 + 2] = src[(src_y * src_width + src_x) * 3 + 0]; // B 
        }
    }
}

void post_process(uint8_t *image_data);

unsigned int run_inference(uint8_t *image_data) 
{
    unsigned int i;
    if(input == NULL) return 1;

    int64_t start_time = esp_timer_get_time();

    uint8_t *resized_image = (uint8_t *)heap_caps_malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS,  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if(resized_image == NULL)
    {
        ESP_LOGE(TAG, "heap_caps_malloc failed");
        return 1;
    }

    resize_image(image_data, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 
                resized_image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS);
    
    if(input->quantization.type == kTfLiteAffineQuantization)
    {
        for (i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS; i++) {
            input->data.int8[i] = resized_image[i] ^ 0x80;
        }    

        //float scale = input->params.scale;
        //int zero_point = input->params.zero_point;
        //for (i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS; i++) {
            //float normalized = resized_image[i] / 255.0f; 
            //input->data.int8[i] = normalized / scale + zero_point;
        //    input->data.int8[i] = (char)(((float)resized_image[i] - 128.0f) / (128.0f * scale)) + zero_point;
        //} 
    }   
    else
    {             
        for (i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS; i++) {
            input->data.f[i] = resized_image[i] / 255.0f;
        }    
    }

    free(resized_image);

    if (interpreter->Invoke() != kTfLiteOk) {        
        ESP_LOGE(TAG, "Invoke failed");
        return 1;
    }

    #if 0
    unsigned int len = output->dims->data[2];
    if (output->quantization.type == kTfLiteAffineQuantization)
    {
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        float   f32[6];
        for(i = 0; i < len; i++)
        {
            for(int x = 0; x < 6; x++)
            {
                f32[x] = (output->data.int8[i + x * len] - zero_point) * scale;
            }
            printf("%04d,%.3f %.3f %.3f %.3f %.3f %.3f\r\n", i, f32[0], f32[1], f32[2], f32[3], f32[4], f32[5]);
        }
    }
    else
    {
        for(i = 0; i < len; i++)
        {
            printf("%04d,%.3f %.3f %.3f %.3f %.3f %.3f\r\n", i, output->data.f[i], output->data.f[i+ len * 1],
                output->data.f[i + len * 2],output->data.f[i + len * 3],output->data.f[i + len * 4],output->data.f[i + len * 5]
            );
        }
    }
    #endif

    post_process(image_data);

    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "Inference time: %.2f ms", (end_time - start_time) / 1000.0); //150ms

    return 0;
}

struct st_box 
{
    float           x1, y1;            
    float           ww, hh;
    float           score;              // 置信度
    unsigned char   class_id;           // 类别ID
    unsigned char   keep;
};

struct st_box_arr
{
    struct st_box*  box;               // 指向边界框数组的指针
    size_t          cnt;               // 当前存储的框数量
    size_t          len;               // 数组容量
};

typedef struct 
{
    int         width;
    int         height;
    uint8_t     *data;
} fb_data_t;

void fb_gfx_fillRect(fb_data_t *fb, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t color)
{
    int32_t line_step = (fb->width - w) * 3;
    uint8_t *data = fb->data + ((x + (y * fb->width)) * 3);
    uint8_t c0 = color >> 16;
    uint8_t c1 = color >> 8;
    uint8_t c2 = color;
    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            data[0] = c0;
            data[1] = c1;
            data[2] = c2;
            data+=3;
        }
        data += line_step;
    }
}

void fb_gfx_drawFastHLine(fb_data_t *fb, int32_t x, int32_t y, int32_t w, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, w, 1, color);
}

void fb_gfx_drawFastVLine(fb_data_t *fb, int32_t x, int32_t y, int32_t h, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, 1, h, color);
}


// 计算两个框的交并比(IOU)
float calculate_iou(struct st_box box1, struct st_box box2) 
{
    float box1_x1 = box1.x1 - box1.ww / 2;
    float box1_y1 = box1.y1 - box1.hh / 2;
    float box1_x2 = box1.x1 + box1.ww / 2;
    float box1_y2 = box1.y1 + box1.hh / 2;
    
    float box2_x1 = box2.x1 - box2.ww / 2;
    float box2_y1 = box2.y1 - box2.hh / 2;
    float box2_x2 = box2.x1 + box2.ww / 2;
    float box2_y2 = box2.y1 + box2.hh / 2;
    
    float inter_x1 = fmaxf(box1_x1, box2_x1);
    float inter_y1 = fmaxf(box1_y1, box2_y1);
    float inter_x2 = fminf(box1_x2, box2_x2);
    float inter_y2 = fminf(box1_y2, box2_y2);
    
    float inter_width = fmaxf(0, inter_x2 - inter_x1);
    float inter_height = fmaxf(0, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;
    
    float box1_area = box1.ww * box1.hh;
    float box2_area = box2.ww * box2.hh;
    float union_area = box1_area + box2_area - inter_area;
    
    if (union_area <= 0) return 0;
    
    return inter_area / union_area;
}

// 非极大抑制(NMS)函数
void nms(struct st_box* boxes, int num_boxes, float iou_threshold) 
{
    for (int i = 0; i < num_boxes; i++) {
        boxes[i].keep = 1;
    }
    
    for (int i = 0; i < num_boxes - 1; i++) {
        for (int j = i + 1; j < num_boxes; j++) {
            if (boxes[i].score < boxes[j].score) {
                struct st_box temp = boxes[i];
                boxes[i] = boxes[j];
                boxes[j] = temp;
            }
        }
    }
    
    for (int i = 0; i < num_boxes; i++) {
        printf("post_process boxes[i].class_id =%d!\r\n",boxes[i].class_id );
        if (!boxes[i].keep) continue;
        
        for (int j = i + 1; j < num_boxes; j++) {
            if (!boxes[j].keep) continue;
            
            if (boxes[i].class_id == boxes[j].class_id) {
                float iou = calculate_iou(boxes[i], boxes[j]);
                
                if (iou > iou_threshold) {
                    boxes[j].keep = 0; // 抑制重叠框
                }
            }
        }
    }
}

void post_process(uint8_t *image_data)
{
    struct st_box_arr   arr = {0};
    float conf_1, conf_0,  f32[6], scale = 1;
    unsigned int i, len;
    int zero_point = 0;

    arr.box = (struct st_box*)heap_caps_malloc(output->dims->data[2] * sizeof(struct st_box),  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if(arr.box == NULL)
    {
        printf("box_arr_malloc error!\r\n");
        return;
    }
    arr.cnt = 0;

    if (output->quantization.type == kTfLiteAffineQuantization)
    {
        scale = output->params.scale;
        zero_point = output->params.zero_point;
    }

    len = output->dims->data[2];
    for (i = 0; i < len; i++) 
    {
        if (output->quantization.type == kTfLiteAffineQuantization)
        {
            f32[0] = (output->data.int8[i + 0 * len] - zero_point) * scale;
            f32[1] = (output->data.int8[i + 1 * len] - zero_point) * scale;
            f32[2] = (output->data.int8[i + 2 * len] - zero_point) * scale;
            f32[3] = (output->data.int8[i + 3 * len] - zero_point) * scale;
            f32[4] = (output->data.int8[i + 4 * len] - zero_point) * scale;
            f32[5] = (output->data.int8[i + 5 * len] - zero_point) * scale;
        }
        else
        {
            f32[0] = output->data.f[0 * len + i];
            f32[1] = output->data.f[1 * len + i];
            f32[2] = output->data.f[2 * len + i];
            f32[3] = output->data.f[3 * len + i];
            f32[4] = output->data.f[4 * len + i];
            f32[5] = output->data.f[5 * len + i];
        }
        conf_0 = f32[5];
        conf_1 = f32[4];        
        if((conf_0 >= conf_threshold)||(conf_1 >= conf_threshold))
        {
            arr.box[arr.cnt].x1 = f32[0]; 
            arr.box[arr.cnt].y1 = f32[1]; 
            arr.box[arr.cnt].ww = f32[2]; 
            arr.box[arr.cnt].hh = f32[3]; 
            arr.box[arr.cnt].score = (conf_0 >= conf_1 ? conf_0 : conf_1);
            arr.box[arr.cnt].class_id = (conf_0 >= conf_1 ? 0 : 1); 
            arr.cnt++;
        }
    }
    printf("post_process arr.cnt=%d!\r\n",arr.cnt);
    nms(arr.box, arr.cnt,detect_threshold);

    fb_data_t fb_data;
    fb_data.width = INPUT_IMAGE_WIDTH;
    fb_data.height = INPUT_IMAGE_HEIGHT;
    fb_data.data = image_data;

    for (i = 0; i < arr.cnt; i++) {
        if (arr.box[i].keep) {
            float x1, y1, ww, hh;
            unsigned int color;

            x1 = (arr.box[i].x1 - arr.box[i].ww / 2) * INPUT_IMAGE_WIDTH;
            y1 = (arr.box[i].y1 - arr.box[i].hh / 2) * INPUT_IMAGE_HEIGHT;
            ww = arr.box[i].ww * INPUT_IMAGE_WIDTH;
            hh = arr.box[i].hh * INPUT_IMAGE_HEIGHT;

            color = arr.box[i].class_id == 0 ? 0x0000ff00 : 0x00ff0000;
            fb_gfx_drawFastHLine(&fb_data, x1, y1, ww, color);
            fb_gfx_drawFastHLine(&fb_data, x1, y1 + hh - 1, ww, color);
            fb_gfx_drawFastVLine(&fb_data, x1, y1, hh, color);
            fb_gfx_drawFastVLine(&fb_data, x1 + ww - 1, y1, hh, color);
        }
    }

    free(arr.box); 
    arr.box = NULL;
}
