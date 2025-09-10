
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
 
#include "esp_err.h" 
#include "mbedtls/md5.h"  
  
 bool spiffs_flag[2]={false};

static const char *TAG = "OTA SPIFFS";
const char* model_path = "/spiffs/ppo_model.bin";

extern const unsigned char meta_model_tflite[];
extern const unsigned int meta_model_tflite_len;

extern float *fisher_matrix;
extern float *theta ; 
extern bool ewc_ready;
extern void parse_model_weights(uint8_t *buffer, size_t size); 


void verify_downloaded_file(void) {
    const char *save_path = "/spiffs/ppo_model.bin";
    
    // 检查文件是否存在
    FILE *f = fopen(save_path, "rb");
    if (!f) {
        ESP_LOGI(TAG, "Downloaded file does not exist in SPIFFS!");
         
        return;
    }
    
    // 获取文件大小
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fclose(f);
    
    ESP_LOGI(TAG, "Downloaded file size: %d bytes", file_size);
    
    if (file_size == 0) {
        ESP_LOGI(TAG, "File exists but is empty!");
    }
    
     
}


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

 
 bool calc_file_md5(const char *file_path, char *md5_str) {
    FILE *f = fopen(file_path, "rb");
    if (!f) return false;

    mbedtls_md5_context ctx;
    unsigned char digest[16];
    unsigned char buf[1024];
    size_t len;

    mbedtls_md5_init(&ctx);
    mbedtls_md5_starts(&ctx);  

    while ((len = fread(buf, 1, sizeof(buf), f)) > 0) {
        mbedtls_md5_update(&ctx, buf, len);   
    }

    mbedtls_md5_finish(&ctx, digest);  
    mbedtls_md5_free(&ctx);
    fclose(f);

    for (int i = 0; i < 16; i++) {
        sprintf(md5_str + i*2, "%02x", digest[i]);
    }
    md5_str[32] = 0;
    return true;
}


 char local_md5[64]={0};
 char server_md5[64]={0};
  
// JSON 解析函数（简单版本）
char* extract_md5_from_json(const char* json_str) {
    char* md5_start = strstr(json_str, "\"md5\":\"");
    if (md5_start == NULL) {
        ESP_LOGE(TAG, "MD5 field not found in JSON");
        return NULL;
    }
    
    md5_start += 7; // 跳过 "\"md5\":\""
    char* md5_end = strchr(md5_start, '"');
    if (md5_end == NULL) {
        ESP_LOGE(TAG, "Invalid JSON format");
        return NULL;
    }
    
    size_t md5_len = md5_end - md5_start;
    char* md5 = (char*) malloc(md5_len + 1);
    strncpy(md5, md5_start, md5_len);
    md5[md5_len] = '\0';
    memcpy(server_md5, md5, sizeof(server_md5));
     
    return md5;
}
 
// HTTP 事件处理函数
esp_err_t http_md5_event_handler(esp_http_client_event_t *evt) {
    static char* response_buffer = NULL;
    static size_t response_len = 0;

    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "Received data, len=%d", evt->data_len);
            // 累积接收到的数据
            response_buffer = (char*)realloc(response_buffer, response_len + evt->data_len + 1);
            memcpy(response_buffer + response_len, evt->data, evt->data_len);
            response_len += evt->data_len;
            response_buffer[response_len] = '\0';
            break;
            
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP request finished");
            if (response_buffer != NULL) {
                ESP_LOGI(TAG, "Full response: %s", response_buffer);
                
                // 解析 MD5
                char *md5 = extract_md5_from_json(response_buffer);
                
                if (md5 != NULL) {
                    ESP_LOGI(TAG, "Extracted MD5: %s", md5);
                    // if (strcmp(md5, local_md5) == 0) {
                    //     ESP_LOGI(TAG, "MD5 match! OTA success.");
                    // } 
                    // else {
                    //     ESP_LOGE(TAG, "MD5 mismatch! OTA corrupted.");
                    //     free(md5);
                    //     return ESP_FAIL;
                    // }
                    free(md5);
                }
                
                free(response_buffer);
                response_buffer = NULL;
                response_len = 0;
            }
            break;
            
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGI(TAG, "Connected to server");
            break;
            
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "Disconnected from server");
            if (response_buffer != NULL) {
                free(response_buffer);
                response_buffer = NULL;
                response_len = 0;
            }
            break;
            
        default:
            break;
    }
    return ESP_OK;
}
  


typedef struct {
    FILE *file;
    int total_bytes;
} download_ctx_t;

esp_err_t download_event_handler(esp_http_client_event_t *evt) {
    download_ctx_t *ctx = (download_ctx_t *)evt->user_data;

    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            if (ctx && ctx->file && evt->data_len > 0) {
                size_t written = fwrite(evt->data, 1, evt->data_len, ctx->file);
                ctx->total_bytes += written;
                ESP_LOGI(TAG, "Written %d bytes, total: %d", written, ctx->total_bytes);
            }
            break;

        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_FINISH");
            break;

        default:
            break;
    }
    return ESP_OK;
}

esp_err_t ota_download_event_based(const char *url, const char *save_path) {
    ESP_LOGI(TAG, "Event-based download from: %s", url);

    download_ctx_t ctx = {0};
    
    // 删除旧文件
    remove(save_path);

    ctx.file = fopen(save_path, "wb");
    if (!ctx.file) {
        ESP_LOGE(TAG, "Failed to open file for writing");
        return ESP_FAIL;
    }

    esp_http_client_config_t config = {
        .url = url,
        .timeout_ms = 30000,
        .event_handler = download_event_handler,
        .user_data = &ctx
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        fclose(ctx.file);
        ESP_LOGE(TAG, "HTTP client init failed");
        return ESP_FAIL;
    }

    esp_err_t err = esp_http_client_perform(client);
    int status = esp_http_client_get_status_code(client);

    fclose(ctx.file);
    esp_http_client_cleanup(client);

    if (err != ESP_OK || status != 200) {
        ESP_LOGE(TAG, "Download failed: %s, status=%d", esp_err_to_name(err), status);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Download completed: %d bytes", ctx.total_bytes);
    spiffs_flag[0]=true;
    return ESP_OK;
}

void download_model(void *pvParameters) {

    const char *ota_url = "http://192.168.30.132:5001/ota_model";
    const char *save_path = "/spiffs/ppo_model.bin";
     
 
    if (ota_download_event_based(ota_url, save_path) != ESP_OK) {
        ESP_LOGE(TAG, "OTA download failed");
        vTaskDelete(NULL); 
        return;
    }
    ESP_LOGI(TAG, "OTA download Suceess!");
    // 验证下载的文件
    if (!calc_file_md5("/spiffs/ppo_model.bin", local_md5)) {
        ESP_LOGI(TAG, "Local MD5 calculation ppo_model.bin empty");
        //vTaskDelete(NULL);
        //return;
    
        if (  strcmp(server_md5, local_md5) == 0) {
            ESP_LOGI(TAG, "Md5 verified equally!");
                
        } else {
            ESP_LOGI(TAG, "Md5 verified not equally!");
        }   
    }
    vTaskDelete(NULL);
 
}



void get_server_md5(void *pvParameters ) { 


    esp_http_client_config_t config = {
        .url =  "http://192.168.30.132:5001/ota_model_md5",
        .timeout_ms = 10000,
        .event_handler = http_md5_event_handler,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    
    ESP_LOGI(TAG, "Performing HTTP GET request...");
    esp_err_t err = esp_http_client_perform(client);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP GET request successful");
        int status_code = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "Status code: %d", status_code);
    } else {
        ESP_LOGE(TAG, "HTTP GET request failed: %s", esp_err_to_name(err));
        server_md5[0]=0;
    }

    esp_http_client_cleanup(client);
 

    vTaskDelete(NULL);
} 

void ota_update_process(void *pvParameters) {
    ESP_LOGI(TAG, "Starting OTA update process...");

    // 1. 获取服务器 MD5
    //get_server_md5();
    xTaskCreate(get_server_md5, "get_server_md5", 8192, NULL, 5, NULL);
    vTaskDelay(5000 / portTICK_PERIOD_MS);
    if (server_md5[0] == 0) {
        ESP_LOGE(TAG, "Failed to get server MD5");
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Server MD5: %s", server_md5);

    // 2. 计算本地 MD5
     
    if (!calc_file_md5("/spiffs/ppo_model.bin", local_md5)) {
        ESP_LOGI(TAG, "Local MD5 calculation Model Empty");
        //vTaskDelete(NULL);
        //return;
    }
    ESP_LOGI(TAG, "Local  MD5: %s", local_md5);
    // 3. 比较 MD5，决定是否需要下载
    if (  strcmp(server_md5, local_md5) != 0) {
        ESP_LOGI(TAG, "MD5 different, downloading new model...");
        //download_model();
        
        
    } else {
        ESP_LOGI(TAG, "Model is up to date, no download needed");
    }

    // 清理内存
    //free(server_md5);
    //if (local_md5 != NULL) free(local_md5);
    
    // 在 download_model 函数末尾添加验证
    verify_downloaded_file();
    spiffs_flag[1]=true;
    vTaskDelete(NULL);
}
// ---------------- 启动 OTA ----------------
extern "C" void start_ota() {

     
     while(spiffs_flag[0]==false){
        xTaskCreate(download_model, "download_model", 16384, NULL, 5, NULL);
        vTaskDelay(10000 / portTICK_PERIOD_MS);
     }
     while(spiffs_flag[1]==false){
        xTaskCreate(ota_update_process, "ota_update_process", 16384, NULL, 5, NULL);
        vTaskDelay(10000 / portTICK_PERIOD_MS);
     }

}


 