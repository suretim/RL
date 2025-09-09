
#define SPIFFS_FL      0
#define NVS_FL        1
#define FLTYPE        SPIFFS_FL
 
#include "esp_spiffs.h"
 #include "esp_log.h"
#include "esp_system.h"
#include "esp_spiffs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
 
#include "esp_http_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "SPIFFS";
const char* model_path = "/spiffs/ppo_model.bin";

extern const unsigned char meta_model_tflite[];
extern const unsigned int meta_model_tflite_len;

extern float *fisher_matrix;
extern float *theta ; 
extern bool ewc_ready;
extern void parse_model_weights(uint8_t *buffer, size_t size); 
// ---------------- Local fallback ----------------
bool load_local_model() {
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Local model not found!");
        return false;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buffer = (uint8_t*)malloc(size);
    if (!buffer) { fclose(f); return false; }
    fread(buffer, 1, size, f);
    fclose(f);
    parse_model_weights(buffer, size);
    free(buffer);
    ESP_LOGI(TAG, "Local model loaded successfully!");
    return true;
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

/**
 * 初始化 SPIFFS
 */
extern "C" void spiffs_init(void) {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",       // 挂载路径
        .partition_label = NULL,      // 默认分区
        .max_files = 5,               // 最大同时打开文件数
        .format_if_mount_failed = true
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to find SPIFFS partition");
        } else {
            ESP_LOGE(TAG, "Failed to init SPIFFS (%s)", esp_err_to_name(ret));
        }
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get SPIFFS partition information (%s)", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
    }
}

/**
 * 从 SPIFFS 加载二进制 float 文件
 */
float* load_float_bin(const char* path, size_t &length) {
    char full_path[64];
    snprintf(full_path, sizeof(full_path), "/spiffs/%s", path);

    FILE* f = fopen(full_path, "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file: %s", full_path);
        length = 0;
        return NULL;
    }

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    rewind(f);

    if (file_size % sizeof(float) != 0) {
        ESP_LOGW(TAG, "File size %d not aligned with float size", file_size);
    }

    // 分配内存并读取数据
    float* buffer = (float*)malloc(file_size);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Malloc failed for file: %s", full_path);
        fclose(f);
        length = 0;
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, file_size, f);
    fclose(f);

    if (read_bytes != file_size) {
        ESP_LOGW(TAG, "Read size mismatch (%d != %d)", read_bytes, file_size);
    }

    length = file_size / sizeof(float);
    ESP_LOGI(TAG, "Loaded %d floats from %s", length, full_path);
    return buffer;
}

 