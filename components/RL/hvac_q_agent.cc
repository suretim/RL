#include <stdio.h>
#include <vector>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_http_client.h"
#include "ota_model_updater.h"
#include "ml_pid.h"
#include "nn.h" 
#include "hvac_q_agent.h"
#include "config_wifi.h"
#include "config_mqtt.h"
 
#include "cJSON.h"
 #include "mbedtls/base64.h"

#include "esp_crc.h"   
#include <string.h> 
#include "esp_system.h" 
#include "cJSON.h"
#include "esp_system.h"
#include "esp_log.h" 
#include "esp_err.h" 
#include "esp_partition.h"    
#include "infer_esp32_lstm_lll.h"
#define CURRENT_VERSION  "1.0.0" 
static const char *TAG = "OTA HVAC";

  uint8_t flask_state_get_flag[FLASK_GET_COUNT]={0};
  uint8_t flask_state_put_flag[FLASK_PUT_COUNT]={0};
extern std::vector<float> health_result;
extern const char* spiffs1_model_path[SPIFFS1_MODEL_COUNT];
extern const char* spiffs2_model_path[SPIFFS2_MODEL_COUNT];
//extern const char optimized_model_path[];
//extern const char spiffs_ppo_model_bin_path[];

//#define  post_data   "{\"model_path\": \"./saved_models/ppo_policy_actor\"}" 

//const int num_flask_task=3;
char flask_get_name[FLASK_STATES_GET_COUNT][64]={
     "optimized_model_path",
     "spiffs_ppo_model_bin_path"
};
//const int num_flask_task=3;
char flask_put_name[FLASK_PUT_COUNT][64]={
    "post_data"
}; 
// ---------------- OTA Callback ----------------
esp_err_t _http_down_load_event_handler(esp_http_client_event_t *evt) {
    static FILE *f_model = NULL;
    switch(evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            //if (f_model) fwrite(evt->data, 1, evt->data_len, f_model);
            if (evt->user_data) {
                f_model = (FILE *)evt->user_data;
                fwrite(evt->data, 1, evt->data_len, f_model);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            if (f_model) {
                fclose(f_model);
                f_model = NULL;
                ESP_LOGI(TAG, "OTA file saved to %s", spiffs1_model_path[FLASK_OPTI_MODEL]);
            }
            break;
        case HTTP_EVENT_DISCONNECTED:
        case HTTP_EVENT_ERROR:
            if (f_model) { fclose(f_model); f_model = NULL; }
            ESP_LOGE(TAG, "HTTP error or disconnected");
            break;
        default:
            break;
    }
    return ESP_OK;
}
 
uint32_t swap_endian(uint32_t value) {
    return ((value >> 24) & 0xFF) | 
           ((value << 8) & 0xFF0000) | 
           ((value >> 8) & 0xFF00) | 
           ((value << 24) & 0xFF000000);
} 

bool save_model_to_spiffs(unsigned int decoded_len,   unsigned char  *model_bin, const char *spi_file_name) {
          
    ESP_LOGI(TAG, "Binary Model copied, length=%zu", decoded_len);
      
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
             
            return false;
        }
    } else {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", spi_file_name);
         
        return false;
    }

     
    flask_state_get_flag[FLASK_OPTI_MODEL] = SPIFFS_MODEL_SAVED;
     
    return true;
}
 
 

bool parse_policy_model_json(const char *json_str) {
     
    //printf("Raw JSON string: %s\n", json_str); // 直接使用，不需要.c_str()
    //printf("JSON string length: %d\n", strlen(json_str)); // 使用strlen函数
    cJSON *root = cJSON_Parse(json_str); // 直接使用，不需要.c_str()
    if (!root) {
        ESP_LOGE(TAG, "Failed to parse_model_json JSON");
        return false;
    } 
    // 解析 metadata
    cJSON *metadata = cJSON_GetObjectItem(root, "metadata");
    if (metadata) {
        const char *fw = cJSON_GetObjectItem(metadata, "firmware_version")->valuestring;
        const char *model_type = cJSON_GetObjectItem(metadata, "model_type")->valuestring;
        ESP_LOGI(TAG, "Firmware: %s, ModelType: %s", fw, model_type);
    }
    // Parse CRC32
    //const char * crc32_espect_str =nullptr;
    cJSON *cjson_crc32_espect = cJSON_GetObjectItem(root, "crc32");
    if (cjson_crc32_espect==NULL || !cJSON_IsString(cjson_crc32_espect)) {
        ESP_LOGE(TAG, "cjson_crc32_espect failed");
        return false;
    } 
    ESP_LOGI(TAG, "crc32_espect (hex): 0x%s", cjson_crc32_espect->valuestring);

    //uint32_t crc32_espect = (uint32_t) strtol(cjson_crc32_espect->valuestring, NULL, 10);
    uint32_t crc32_espect = strtoul(cjson_crc32_espect->valuestring, NULL, 10); 
    //ESP_LOGI(TAG, "crc32_espect: %s ",  crc32_espect_str);

    // Parse model_data_b64
    cJSON *model_b64 = cJSON_GetObjectItem(root, "model_data_b64");
    if (model_b64 && cJSON_IsString(model_b64)) {
        const char *b64_str = model_b64->valuestring;
        size_t bin_len;
        unsigned char *model_bin = nullptr;
        unsigned int decoded_len = 0;

        bin_len = (strlen(b64_str) * 3 + 3) / 4;
        model_bin = (unsigned char *)malloc(bin_len + 1);  // +1 for safety
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
        ESP_LOGI(TAG, "Base64 Model decoded, length=%d",(int) decoded_len);
        // Calculate CRC32 of the Base64 string (using strlen instead of sizeof)
        uint32_t crc32_result = esp_crc32_le(0, (const uint8_t *)model_bin, decoded_len);
        //crc32_result = swap_endian(crc32_result);
        if (crc32_espect != crc32_result) {
            ESP_LOGE(TAG, "CRC32 mismatch, expected %lx, got %lx", crc32_espect, crc32_result);
            cJSON_Delete(root);
            return false;
        }
         
        // Save model to SPIFFS or Flash
        save_model_to_spiffs(decoded_len, model_bin, spiffs1_model_path[OPTIMIZED_MODEL]);
        
        // save_model_to_flash(b64_str); 
    }

    // 解析 optimal_params
    cJSON *opt_params = cJSON_GetObjectItem(root, "optimal_params");
    if (opt_params) {
        cJSON *dense24_kernel = cJSON_GetObjectItem(opt_params, "dense_24/kernel:0");
        if (dense24_kernel) {
            int rows = cJSON_GetArraySize(dense24_kernel);
            ESP_LOGI(TAG, "dense_24/kernel:0 has %d rows", rows);
            for (int i = 0; i < rows; i++) {
                cJSON *row = cJSON_GetArrayItem(dense24_kernel, i);
                int cols = cJSON_GetArraySize(row);
                for (int j = 0; j < cols; j++) {
                    float val = (float)cJSON_GetArrayItem(row, j)->valuedouble;
                    // TODO: 存到你的 dense layer 权重数组
                    ESP_LOGD(TAG, "W[%d][%d]=%f", i, j, val);
                }
            }
        }

        cJSON *dense25_bias = cJSON_GetObjectItem(opt_params, "dense_25/bias:0");
        if (dense25_bias) {
            int size = cJSON_GetArraySize(dense25_bias);
            for (int i = 0; i < size; i++) {
                float val = (float)cJSON_GetArrayItem(dense25_bias, i)->valuedouble;
                ESP_LOGI(TAG, "dense_25/bias[%d]=%f", i, val);
            }
        }
    }

    cJSON_Delete(root);
    return true;
}


// 添加全局变量或使用静态变量来存储数据
static std::string http_response_data;
static std::string http_put_response_data;

// HTTP事件处理器
esp_err_t _http_event_model_json_handler(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            // 接收到数据时追加到字符串
            if (evt->data_len > 0) {
                http_response_data.append((char*)evt->data, evt->data_len);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            // 请求完成
            ESP_LOGI(TAG, "HTTP request finished, received %d bytes", http_response_data.length());
            break;
        default:
            break;
    }
    return ESP_OK;
}



 

void http_get_model_json(void *pvParameters) {
     
    http_response_data.clear(); // 清空之前的数据
    flask_state_get_flag[FLASK_OPTI_MODEL]=SPIFFS_MODEL_ERR;
    esp_http_client_config_t config = {
        .url = HTTP_GET_MODEL_JSON_URL,        
        .timeout_ms = 10000,
        .event_handler = _http_event_model_json_handler, // 设置事件处理器
    };
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status_code = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "HTTP Status Code: %d", status_code);
        
        if (status_code == 200) {
            ESP_LOGI(TAG, "Total received data length: %d", http_response_data.length());
            
            if (!http_response_data.empty()) {
                // 调试输出：打印前100个字符
                //ESP_LOGI(TAG, "First 100 chars: %.100s", http_response_data.c_str());
                
                // 打印前16字节的十六进制
                //ESP_LOGI(TAG, "First 16 bytes in hex:");
                //for (int i = 0; i < std::min(16, (int)http_response_data.length()); i++) {
                //    printf("%02X ", (unsigned char)http_response_data[i]);
                //}
                //printf("\n");
                
                // 解析JSON
                parse_policy_model_json(http_response_data.c_str());
            } else {
                ESP_LOGE(TAG, "No data received");
                
            }
        } else {
            ESP_LOGE(TAG, "HTTP request failed with status: %d", status_code);
        }
    } else {
        ESP_LOGE(TAG, "HTTP GET failed: %s", esp_err_to_name(err));
    }
    
    esp_http_client_cleanup(client);
    vTaskDelete(NULL);
}

   
 char local_md5[64]={0};
 char server_md5[64]={0};

 
typedef struct {
    FILE *file;
    int total_bytes;
} download_ctx_t;


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


esp_err_t download_event_md5_handler(esp_http_client_event_t *evt) {
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

esp_err_t ota_download_md5_event_based(const char *url, const char *save_path) {
    ESP_LOGI(TAG, "Event-based download from: %s", url);

    download_ctx_t ctx = {0};
    
    // 删除旧文件
    remove(save_path);

    ctx.file = fopen(save_path, "wb");
    if (!ctx.file) {
        ESP_LOGE(TAG, "Failed to open file for writing %s",save_path);
        
        return ESP_FAIL;
    }

    esp_http_client_config_t config = {
        .url = url,
        .timeout_ms = 30000,
        .event_handler = download_event_md5_handler,
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

    ESP_LOGI(TAG, "Download MODEL_BIN_PPO_MD5 completed: %d bytes", ctx.total_bytes);
    //spiffs_token=true;
    flask_state_get_flag[MODEL_BIN_PPO_MD5]=SPIFFS_MODEL_SAVED;
    return ESP_OK;
}

void download_ota_md5_model(void *pvParameters) {

    //const char *ota_url = "http://192.168.30.132:5001/ota_model";
    char task_str[32]="ota_model";
        char task_url[128];
        sprintf(task_url, "http://%s:%s/%s", BASE_URL,POLICY_PORT,task_str);
          
 
    if (ota_download_md5_event_based(task_url, spiffs2_model_path[BIN_MODEL]) != ESP_OK) {
        ESP_LOGE(TAG, "OTA download failed");
        flask_state_get_flag[MODEL_BIN_PPO_MD5]=SPIFFS_MODEL_ERR;
        vTaskDelete(NULL); 
        return;
    }
    ESP_LOGI(TAG, "OTA download Suceess!");
    // 验证下载的文件
    if (!calc_file_md5(spiffs2_model_path[BIN_MODEL], local_md5)) {
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
  

void get_ota_model_md5(void *pvParameters ) { 
        char task_str[32]="ota_model_md5";
        char task_url[128];
        sprintf(task_url, "http://%s:%s/%s", BASE_URL,POLICY_PORT,task_str);
     

    esp_http_client_config_t config = {
        .url =  task_url   ,//"http://192.168.30.132:5001/ota_model_md5",
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

 
void ota_update_ppo_model_md5_process(void *pvParameters) {
     
    while(flask_state_get_flag[MODEL_BIN_PPO_MD5]==SPIFFS_MODEL_EMPTY){
        xTaskCreate(download_ota_md5_model, "download_model", 16384, NULL, 5, NULL);
        vTaskDelay(10000 / portTICK_PERIOD_MS);
    }
    flask_state_get_flag[MODEL_BIN_PPO_MD5]=SPIFFS_MODEL_ERR;
    ESP_LOGI(TAG, "Starting OTA update process...");

    // 1. 获取服务器 MD5
    //get_server_md5();
    xTaskCreate(get_ota_model_md5, "get_ota_model_md5", 8192, NULL, 5, NULL);
    vTaskDelay(5000 / portTICK_PERIOD_MS);
    if (server_md5[0] == 0) {
        ESP_LOGE(TAG, "Failed to get server MD5");
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Server MD5: %s", server_md5);

    // 2. 计算本地 MD5
     
    if (!calc_file_md5(spiffs1_model_path[MODEL_BIN_PPO_MD5], local_md5)) {
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
    //verify_downloaded_file(spiffs_ppo_model_bin_path);
    //spiffs_flag[1]=true;
    flask_state_get_flag[MODEL_BIN_PPO_MD5]=SPIFFS_MODEL_SAVED;
    vTaskDelete(NULL);
}

 
void (*functionGetArray[2])(void *pvParameters) = {
    http_get_model_json,  
    ota_update_ppo_model_md5_process,
};



// HTTP事件处理器
esp_err_t _http_event_exporter_handler(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            // 接收到数据时追加到字符串
            if (evt->data_len > 0) {
                http_put_response_data.append((char*)evt->data, evt->data_len);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            // 请求完成
            ESP_LOGI(TAG, "http_put_response_data finished, received %d bytes", http_put_response_data.length());
            break;
        default:
            break;
    }
    return ESP_OK;
}


void flask_init_exporter_task(void *pvParameters) {
        flask_state_put_flag[INIT_EXPORTER]=SPIFFS_MODEL_ERR;
        //const char* base_url ="192.168.0.57:5000";
        char task_str[32]="init_exporter";
        char task_url_str[128];
        sprintf(task_url_str, "http://%s:%s/%s", BASE_URL,BASE_PORT,task_str); 
        ESP_LOGI(TAG, "URL: %s", task_url_str);
        //sprintf(url, "http://%s/init_exporter", BASE_URL);
        esp_http_client_config_t config = {
            //.url = "http://<SERVER_IP>:5000/init_exporter",   // Flask 服务器地址
            .url = task_url_str,   
            .event_handler = _http_event_exporter_handler, // 设置事件处理器 
        };

        esp_http_client_handle_t client = esp_http_client_init(&config);

        // 发送 JSON 数据
        //const char *post_data = "{\"model_path\": \"./saved_models/my_model.keras\"}";
         
 
        //const char *post_data = "{\"model_path\": \"./saved_models/ppo_policy_actor\"}";
        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, flask_put_name[INIT_EXPORTER], strlen(flask_put_name[INIT_EXPORTER]));

        
     
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status_code = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "HTTP Status Code: %d", status_code);
        
        if (status_code == 200) {
            ESP_LOGI(TAG, "Total received data length: %d", http_put_response_data.length());
            
            if (!http_put_response_data.empty()) {
                // 调试输出：打印前100个字符
                if(http_put_response_data.length()>=16)
                 ESP_LOGI(TAG, "First 16 chars: %.16s", http_put_response_data.c_str());
                
                // 打印前16字节的十六进制
                //ESP_LOGI(TAG, "First 16 bytes in hex:");
                //for (int i = 0; i < std::min(16, (int)http_put_response_data.length()); i++) {
                //    printf("%02X ", (unsigned char)http_put_response_data[i]);
                //}
                //printf("\n");
                 flask_state_put_flag[INIT_EXPORTER]=SPIFFS_MODEL_SAVED;
            } else {
                
                ESP_LOGE(TAG, "No data received");
            }
        } else {
            ESP_LOGE(TAG, "HTTP request failed with status: %d", status_code);
        }
    } else {
        ESP_LOGE(TAG, "HTTP GET failed: %s", esp_err_to_name(err));
    }
    
    esp_http_client_cleanup(client);
    vTaskDelete(NULL);
 
}

  

void (*functionPutArray[FLASK_PUT_COUNT])(void *pvParameters) = {
    flask_init_exporter_task,  
};
 

void wifi_get_package(int type ) {
    if(type>=0 && type<FLASK_STATES_GET_COUNT)
    //if(type==0)
    {
        xTaskCreatePinnedToCore(
        functionGetArray[type],  //download_model_ota_task,download_model_task
        flask_get_name[type],
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   

    } 
    return   ;
}

void wifi_put_package(int type ) {
    if(type>=0 && type<FLASK_PUT_COUNT)
    //if(type==0)
    {
        xTaskCreatePinnedToCore(
        functionPutArray[type],  //download_model_ota_task,download_model_task
        flask_put_name[type],
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   

    } 
    return   ;
}
//load model.bin
extern "C" pid_run_output_st nn_ppo_infer(void) {
    pid_run_output_st output_speed;

    // 尝试下载 OTA model_version == 0 &&
    //if( !ppoModel.downloadModel(bin_url, model_path)) {
    //   ESP_LOGW("downloadModel", "Model will use existing model if available");
    //}

    //const char* version = "1.0.0";
    //ModelOTAUpdater otaUpdater( check_url, download_url, "/spiffs/model.bin",version  );
    
    // OTA 拉取最新 PPO 参数
    //if (otaUpdater.checkUpdate()) {
    //    printf("✅ Model updated from server\n");
    //}

    // 初始化 PPO 推理网络
    NN  ppoNN(5, 32, 4);  // state_dim=5, hidden=32, action_dim=4
    if (ppoNN.loadWeights("/spiffs1/ppo_model.bin")==false) {
        ESP_LOGI(TAG, "Failed to load weights\n");
        output_speed.speed[0] =11;
        return output_speed;
    }
    ESP_LOGI(TAG," Weights loaded into PPO NN\n");
    
 
    std::vector<float> state = { bp_pid_th.t_feed, bp_pid_th.h_feed, bp_pid_th.l_feed, bp_pid_th.c_feed, 1.0};
    auto action_probs = ppoNN.forwardActor(state);
    float value = ppoNN.forwardCritic(state);
    if(value>0.5)
    {
        output_speed.speed[0] = action_probs[0]>0.5?1:0;
        output_speed.speed[1] = action_probs[1]>0.5?1:0;
        output_speed.speed[2] = action_probs[2]>0.5?1:0;
        output_speed.speed[3] = action_probs[3]>0.5?1:0;
    }
    printf("Action probs: ");
    for (auto p : action_probs) printf("%.3f ", p);
    printf("\nCritic value: %.3f\n", value);
    return output_speed ;
}

 