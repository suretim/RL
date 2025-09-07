#include "esp_log.h"
#include "esp_system.h"
#include "esp_spiffs.h"
#include "esp_http_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"


//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"
#include <stdio.h>
#include <stdlib.h>

static const char *TAG = "NN_OTA_AUTO";

#define TENSOR_ARENA_SIZE 10*1024
#define OTA_INTERVAL_MS 3600000   // 1 小时

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::MicroInterpreter *interpreter = NULL;
TfLiteTensor *input_tensor = NULL;
TfLiteTensor *output_tensor = NULL;
const char *model_path = "/spiffs/model.tflite";

// 初始化 SPIFFS
void init_spiffs() {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if(ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount SPIFFS (%s)", esp_err_to_name(ret));
    }
}

// 下载 TFLite 文件
esp_err_t download_tflite(const char *url, const char *save_path) {
    esp_http_client_config_t config = {.url = url, .timeout_ms = 10000};
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if(esp_http_client_perform(client) != ESP_OK) {
        ESP_LOGE(TAG, "HTTP GET failed");
        esp_http_client_cleanup(client);
        return ESP_FAIL;
    }

    int content_length = esp_http_client_get_content_length(client);
    if(content_length <= 0) { esp_http_client_cleanup(client); return ESP_FAIL; }

    FILE *f = fopen(save_path, "wb");
    if(!f) { esp_http_client_cleanup(client); return ESP_FAIL; }

    char buffer[1024];
    int total_read = 0;
    while(total_read < content_length) {
        int read_len = esp_http_client_read(client, buffer, sizeof(buffer));
        if(read_len <= 0) break;
        fwrite(buffer, 1, read_len, f);
        total_read += read_len;
    }
    fclose(f);
    esp_http_client_cleanup(client);
    ESP_LOGI(TAG, "Downloaded %s (%d bytes)", save_path, total_read);
    return ESP_OK;
}

// 初始化 TFLite Micro interpreter
bool init_interpreter(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if(!f) { ESP_LOGE(TAG,"Failed to open model"); return false; }
    fseek(f, 0, SEEK_END);
    size_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *model_data = (uint8_t*)malloc(model_size);
    fread(model_data, 1, model_size, f);
    fclose(f);

    const tflite::Model *model = tflite::GetModel(model_data);
    if(model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG,"Model schema mismatch"); free(model_data); return false;
    }

    //static tflite::AllOpsResolver resolver;

// 假设模型只用 10 种算子
tflite::MicroMutableOpResolver<10> resolver;
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddReshape();


    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG,"AllocateTensors failed"); free(model_data); return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    ESP_LOGI(TAG,"Interpreter ready. Input=%d Output=%d",
             input_tensor->bytes/sizeof(float), output_tensor->bytes/sizeof(float));
    free(model_data);
    return true;
}

// 推理
bool run_inference(float *input_data, float *output_data) {
    for(int i=0;i<input_tensor->bytes/sizeof(float);i++) input_tensor->data.f[i]=input_data[i];
    if(interpreter->Invoke() != kTfLiteOk) return false;
    for(int i=0;i<output_tensor->bytes/sizeof(float);i++) output_data[i]=output_tensor->data.f[i];
    return true;
}

// OTA 任务
void ota_task(void *param) {
    const char *model_url = (const char*)param;
    while(1) {
        if(download_tflite(model_url, model_path) == ESP_OK) {
            ESP_LOGI(TAG, "Model downloaded, re-initialize interpreter...");
            if(init_interpreter(model_path)) {
                ESP_LOGI(TAG, "Interpreter updated with new model");
            }
        }
        vTaskDelay(OTA_INTERVAL_MS / portTICK_PERIOD_MS);
    }
}

extern "C" void app_main() {
    init_spiffs();

    const char *model_url = "http://192.168.68.237:5000/download_tflite";

    // 第一次初始化
    if(download_tflite(model_url, model_path) == ESP_OK) {
        if(init_interpreter(model_path)) {
            ESP_LOGI(TAG, "Initial model loaded");
        }
    }

    // 创建 OTA 更新任务
    xTaskCreate(ota_task, "ota_task", 8192, (void*)model_url, 5, NULL);

    // 模拟传感器推理循环
    float input_data[5] = {0};
    float output_data[4] = {0};

    while(1) {
        // 填充输入数据
        for(int i=0;i<5;i++) input_data[i] = (float)esp_random() / UINT32_MAX;

        if(run_inference(input_data, output_data)) {
            ESP_LOGI(TAG,"Inference output:");
            for(int i=0;i<4;i++) ESP_LOGI(TAG,"class %d: %f", i, output_data[i]);
        }

        vTaskDelay(5000 / portTICK_PERIOD_MS); // 每 5 秒推理一次
    }
}
