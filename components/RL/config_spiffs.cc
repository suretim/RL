
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
 #include "esp_partition.h"   

#include "esp_http_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
#include "esp_err.h" 
#include "mbedtls/esp_config.h"
#include "mbedtls/base64.h"
#include "mbedtls/md5.h"  
#include "hvac_q_agent.h"  
#include "config_wifi.h"
#include "nn.h"
#include <vector>



static const char *TAG = "OTA SPIFFS";
//const char* spiffs_ppo_model_bin_path = "/spiffs/ppo_model.bin";

//extern const unsigned char meta_model_tflite[];
//extern const unsigned int meta_model_tflite_len;
//bool spiffs_token=false;
extern uint8_t  flask_state_get_flag[FLASK_GET_COUNT] ;
extern uint8_t  flask_state_put_flag[FLASK_PUT_COUNT] ;
extern float *fisher_matrix;
extern float *theta ; 
extern bool ewc_ready; 

 
bool save_model_to_spiffs(uint8_t type, const char *b64_str, const char *spi_file_name) {
    size_t bin_len;
    uint8_t *model_bin = nullptr;
    size_t decoded_len = 0;

     if (type == HTTP_DATA_TYPE_B64) {
        // Base64解码
        // 更准确的长度计算
        bin_len = (strlen(b64_str) * 3 + 3) / 4;
        model_bin = (uint8_t *)malloc(bin_len + 1);  // +1 for safety
        if (!model_bin) {
            ESP_LOGE(TAG, "malloc failed");
            return false;
        }

        int ret = mbedtls_base64_decode(model_bin, bin_len, &decoded_len,
                                        (const unsigned char *)b64_str,
                                        strlen(b64_str));
        if (ret != 0) {
            ESP_LOGE(TAG, "Base64 decode failed, ret=%d", ret);
            free(model_bin);
            return false;
        }
        ESP_LOGI(TAG, "Base64 Model decoded, length=%zu", decoded_len);
    }
     if (type == HTTP_DATA_TYPE_BIN)    
     {
        // 直接二进制数据
        decoded_len = strlen(b64_str);  // 注意：这里假设b64_str包含二进制数据
        bin_len = decoded_len;
        model_bin = (uint8_t *)malloc(bin_len);
        if (!model_bin) {
            ESP_LOGE(TAG, "malloc failed");
            return false;
        }
        memcpy(model_bin, b64_str, bin_len);
        ESP_LOGI(TAG, "Binary Model copied, length=%zu", decoded_len);
    }
    if(model_bin==nullptr)
    {
        ESP_LOGE(TAG, "model_bin is null");
        return false;
    }
    // 验证模型头（可选但推荐）
    if (decoded_len >= 16) {  // 确保文件至少有 16 字节
        // 检查 FlatBuffer 头（字节0-3）
        if (model_bin[4] == 'T' && model_bin[5] == 'F' && model_bin[6] == 'L' && model_bin[7] == '3') {
            // 验证 TFLite 魔术数字（字节 4-7）
            ESP_LOGI(TAG, "TFLite model save_model_to_spiffs verified");
        } else {
            ESP_LOGW(TAG, "Unknown file format, may not be TFLite header %x",model_bin[0]);
            return false;
        }
    } else {
        ESP_LOGW(TAG, "Model file is too short, cannot verify header");
        return false;
    }

    // 保存到SPIFFS
    FILE *f = fopen(spi_file_name, "wb");
    if (f) {
        size_t written = fwrite(model_bin, 1, decoded_len, f);
        fclose(f);
        if (written == decoded_len) {
            ESP_LOGI(TAG, "Model saved to SPIFFS: %s, size=%zu", spi_file_name, written);
        } else {
            ESP_LOGE(TAG, "Write incomplete: %zu/%zu", written, decoded_len);
            free(model_bin);
            return false;
        }
    } else {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", spi_file_name);
        free(model_bin);
        return false;
    }

    free(model_bin);
    flask_state_get_flag[SPIFFS_DOWN_LOAD_MODEL] = SPIFFS_MODEL_SAVED;
     
    return true;
}
 
 


/**
 * 从 SPIFFS 加载二进制 float 文件
 */
float* load_float_bin(const char* path, size_t &length) {
    char full_path[64];
    snprintf(full_path, sizeof(full_path), "/spiffs1/%s", path);

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

  
// 初始化兩個 SPIFFS 分區
extern "C" void spiffs_init(void) {
    // 第一個 SPIFFS，用於普通文件
    esp_vfs_spiffs_conf_t conf1 = {
        .base_path = "/spiffs1",
        .partition_label = "spiffs1",  // 分區名
        .max_files = 5,
        .format_if_mount_failed = true
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&conf1));

    size_t total1 = 0, used1 = 0;
    esp_spiffs_info("spiffs1", &total1, &used1);
    ESP_LOGI(TAG, "SPIFFS1: total=%d, used=%d", total1, used1);

    // 第二個 SPIFFS，用於模型文件
    esp_vfs_spiffs_conf_t conf2 = {
        .base_path = "/spiffs2",
        .partition_label = "spiffs2",  // 分區名
        .max_files = 5,
        .format_if_mount_failed = true
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&conf2));

    size_t total2 = 0, used2 = 0;
    esp_spiffs_info("spiffs2", &total2, &used2);
    ESP_LOGI(TAG, "SPIFFS2: total=%d, used=%d", total2, used2);
}


  

 