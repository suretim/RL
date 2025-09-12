#include <stdio.h>
#include <vector>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_http_client.h"
#include "config_mqtt.h"
#include "ota_model_updater.h"
#include "esp32_model_loader.h"
#include "ml_pid.h"
#include "nn.h" 
#include "hvac_q_agent.h"

 
#include "cJSON.h"
 #include "mbedtls/base64.h"

#include "esp_crc.h"   
#include <string.h> 
#include "esp_system.h" 
#include "cJSON.h"
//#include "esp_base64.h"   // ESP-IDF 自带 base64
#include "esp_err.h"
 
#include <string.h> 
#include "esp_system.h"
#include "esp_partition.h"   


#define CURRENT_VERSION  "1.0.0"
extern const char * spiffs_model_path ;
static const char *TAG = "OTA HVAC";

bool flask_state_flag[NUM_FLASK_TASK]={false};

// ---------------- NN Placeholder ----------------
// 权重向量示例
std::vector<float> W1, W2, b1, b2, Vw;
float Vb = 0.0f;
extern std::vector<float> health_result;

#define H1          32
#define H2          4
 

void parse_model_weights(uint8_t *buffer, size_t size) {
    ESP_LOGI(TAG, "Parsing model weights... (%d bytes)", size);

    // 将 buffer 强制转换为 float*
    float* ptr = reinterpret_cast<float*>(buffer);
    size_t offset = 0;

    // 清空之前的 vector 并填充新数据
    W1.assign(ptr + offset, ptr + offset + H1 * INPUT_DIM);
    offset += H1 * INPUT_DIM;

    b1.assign(ptr + offset, ptr + offset + H1);
    offset += H1;

    W2.assign(ptr + offset, ptr + offset + H2 * H1);
    offset += H2 * H1;

    b2.assign(ptr + offset, ptr + offset + H2);
    offset += H2;

    Vw.assign(ptr + offset, ptr + offset + H2);
    offset += H2;

    Vb = *(ptr + offset);
    offset += 1;

    ESP_LOGI(TAG, "Model weights parsed successfully. Total floats = %d", offset);
} 
 
 
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
                ESP_LOGI(TAG, "OTA file saved to %s", spiffs_model_path);
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
 
// ======= PPO 模型 =======
ESP32EWCModel ewcppoModel;



// 从 Flash 分区读取模型
void read_model_from_flash(void) {
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");

    if (!partition) {
        ESP_LOGE(TAG, "Model partition not found!");
        return;
    }

    size_t size = partition->size;
    uint8_t *buf = (uint8_t *) malloc(size);
    if (!buf) {
        ESP_LOGE(TAG, "malloc failed");
        return;
    }

    ESP_ERROR_CHECK(esp_partition_read(partition, 0, buf, size));

    ESP_LOGI(TAG, "Model read back, first 16 bytes:");
    for (int i = 0; i < 16 && i < size; i++) {
        printf("%02X ", buf[i]);
    }
    printf("\n");

    free(buf);
    flask_state_flag[FLASH_DOWN_LOAD_MODEL] = true;
}

// 保存模型到自定义 Flash 分区
bool save_model_to_flash(const char *b64_str) {
    size_t bin_len = strlen(b64_str) * 3 / 4;
    uint8_t *model_bin =(uint8_t *) malloc(bin_len);
    if (!model_bin) {
        ESP_LOGE(TAG, "malloc failed");
        return false;
    }

 
    size_t decoded_len = 0;
    int ret = mbedtls_base64_decode(model_bin, bin_len, &decoded_len,
                                    (const unsigned char *)b64_str,
                                    strlen(b64_str));
    if (ret != 0) {
        ESP_LOGE(TAG, "Base64 decode failed, ret=%d", ret);
        free(model_bin);
        return false;
    }
    ESP_LOGI(TAG, "Model decoded, length=%zu", decoded_len);

  
    // 查找 "spiffs2" 分区
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "spiffs2");

    if (!partition) {
        ESP_LOGE(TAG, "spiffs2 partition not found!");
        free(model_bin);
        return false;
    }

    // 擦除并写入
    ESP_ERROR_CHECK(esp_partition_erase_range(partition, 0, partition->size));
    ESP_ERROR_CHECK(esp_partition_write(partition, 0, model_bin, decoded_len));

    ESP_LOGI(TAG, "Model saved to Flash partition (size=%d)", decoded_len);

    free(model_bin);
    read_model_from_flash();
    return true;
}

void app_test(void) {
    // 模拟一个 JSON，里面包含 base64 模型
    const char *json_str = "{ \"model_data_b64\": \"QUJDREVGRw==\" }"; // "ABCDEFG"

    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        ESP_LOGE(TAG, "Failed to parse JSON");
        return;
    }

    cJSON *model_b64 = cJSON_GetObjectItem(root, "model_data_b64");
    if (model_b64 && cJSON_IsString(model_b64)) {
        save_model_to_flash(model_b64->valuestring);
    }

    cJSON_Delete(root);

    // 验证读取
    read_model_from_flash();
}

// 从 SPIFFS 读取模型验证
void read_model_from_spiffs(void) {
    FILE *f = fopen("/spiffs/esp32_optimized_model.tflite", "rb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open model file for reading");
        return;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    uint8_t *buf =(uint8_t *) malloc(file_size);
    if (!buf) {
        ESP_LOGE(TAG, "malloc failed");
        fclose(f);
        return;
    }

    fread(buf, 1, file_size, f);
    fclose(f);

    ESP_LOGI(TAG, "Model read back, size=%ld bytes", file_size);

    free(buf);
}

// 保存模型到 SPIFFS
bool save_model_to_spiffs(const char *b64_str) {
    size_t bin_len = strlen(b64_str) * 3 / 4;  // 预估长度
    uint8_t *model_bin =(uint8_t *) malloc(bin_len);
    if (!model_bin) {
        ESP_LOGE(TAG, "malloc failed");
        return false;
    }

     
    size_t decoded_len = 0;
    int ret = mbedtls_base64_decode(model_bin, bin_len, &decoded_len,
                                    (const unsigned char *)b64_str,
                                    strlen(b64_str));
    if (ret != 0) {
        ESP_LOGE(TAG, "Base64 decode failed, ret=%d", ret);
        free(model_bin);
        return false;
    }
    ESP_LOGI(TAG, "Model decoded, length=%zu", decoded_len);
 
    ESP_LOGI(TAG, "Model decoded, length=%d", decoded_len);

    FILE *f = fopen("/spiffs/esp32_optimized_model.tflite", "wb");
    if (f) {
        fwrite(model_bin, 1, decoded_len, f);
        fclose(f);
        ESP_LOGI(TAG, "Model saved to SPIFFS");
    } else {
        ESP_LOGE(TAG, "Failed to open file for writing");
    }

    free(model_bin);
    read_model_from_spiffs();
    return true;
}

void parse_model_json(const char *json_str) {
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        ESP_LOGE(TAG, "Failed to parse JSON");
        return;
    }

    // 解析 metadata
    cJSON *metadata = cJSON_GetObjectItem(root, "metadata");
    if (metadata) {
        const char *fw = cJSON_GetObjectItem(metadata, "firmware_version")->valuestring;
        const char *model_type = cJSON_GetObjectItem(metadata, "model_type")->valuestring;
        ESP_LOGI(TAG, "Firmware: %s, ModelType: %s", fw, model_type);
    }

    // 解析 model_data_b64
    cJSON *model_b64 = cJSON_GetObjectItem(root, "model_data_b64");
    if (model_b64 && cJSON_IsString(model_b64)) {
        const char *b64_str = model_b64->valuestring;
        size_t bin_len = strlen(b64_str) * 3 / 4;  // 预估长度
        uint8_t *model_bin =(uint8_t *) malloc(bin_len);
        if (model_bin) {
            #include "mbedtls/base64.h"

size_t decoded_len = 0;
int ret = mbedtls_base64_decode(model_bin, bin_len, &decoded_len,
                                (const unsigned char *)b64_str,
                                strlen(b64_str));
if (ret != 0) {
    ESP_LOGE(TAG, "Base64 decode failed, ret=%d", ret);
    free(model_bin);
    return  ;
}
ESP_LOGI(TAG, "Model decoded, length=%zu", decoded_len);


            //save_model_to_spiffs(b64_str);
            save_model_to_flash(b64_str);
            // TODO: 存到 SPIFFS / PSRAM / Flash 分区
            free(model_bin);
        }
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
}

void http_get_model_json(void *pvParameters) {
    esp_http_client_config_t config = {
        .url = HTTP_GET_MODEL_JSON_URL,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int content_len = esp_http_client_get_content_length(client);
        char *buffer =(char*) malloc(content_len + 1);
        if (buffer) {
            esp_http_client_read(client, buffer, content_len);
            buffer[content_len] = '\0';
            parse_model_json(buffer);
            free(buffer);
        }
    } else {
        ESP_LOGE(TAG, "HTTP GET failed: %s", esp_err_to_name(err));
    }
    esp_http_client_cleanup(client);
    vTaskDelete(NULL);
}

 


void flask_init_exporter_task(void *pvParameters) {
        flask_state_flag[INIT_EXPORTER]=false;

        //const char* base_url ="192.168.0.57:5000";
        char task_str[32]="init_exporter";
        char task_url_str[128];
        sprintf(task_url_str, "http://%s:%s/%s", BASE_URL,BASE_PORT,task_str); 
        ESP_LOGI(TAG, "URL: %s", task_url_str);
        //sprintf(url, "http://%s/init_exporter", BASE_URL);
        esp_http_client_config_t config = {
            //.url = "http://<SERVER_IP>:5000/init_exporter",   // Flask 服务器地址
            .url = task_url_str,    
        };

        esp_http_client_handle_t client = esp_http_client_init(&config);

        // 发送 JSON 数据
        //const char *post_data = "{\"model_path\": \"./saved_models/my_model.keras\"}";
         
 
       const char *post_data = "{\"model_path\": \"./saved_models/ppo_policy_actor\"}";
        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, post_data, strlen(post_data));

        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Status = %d, content_length = %lld",
                    esp_http_client_get_status_code(client),
                    esp_http_client_get_content_length(client));

                    
            // 获取响应头的大小
     
            
            int cont_len=esp_http_client_get_content_length(client);
            char buffer[512];
            if(cont_len<=512){
                
                //int data_len = esp_http_client_read_response(client, buffer, sizeof(buffer) - 1);
                int data_len = esp_http_client_read_response(client, buffer, cont_len);
                if (data_len >= 0) {
                    buffer[data_len] = 0;
                    ESP_LOGI(TAG, "Response = %s", buffer);
                    flask_state_flag[INIT_EXPORTER]=true;
                }
            }
        } else {
            ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
        } 

        esp_http_client_cleanup(client);
        vTaskDelete(NULL);
}

//print(f"GET http://{ip}:{port}/api/model?name=esp32_policy")
static void flask_download_model_task(void *param) {
    ESP_LOGI("MEM", "Before OTA: Heap free=%d, internal free=%d, largest=%d",
             (int)esp_get_free_heap_size(),
             (int)esp_get_free_internal_heap_size(),
             (int)heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));
        flask_state_flag[DOWN_LOAD_MODEL]=false;

    // 删除旧模型 
    remove(spiffs_model_path);
    static FILE *f_model =fopen(spiffs_model_path, "wb");
    if (!f_model) {
        ESP_LOGE("OTA", "Failed to open file for OTA");
        vTaskDelete(NULL);
        return;
    }
 
    esp_http_client_config_t config = {0}; // 初始化为全零
    char task_str[32]="api/bin-update";
        char task_url_str[128];
        sprintf(task_url_str, "http://%s:%s/%s", BASE_URL,BASE_PORT,task_str); 
        ESP_LOGI(TAG, "URL: %s", task_url_str);
    //sprintf(url, "http://%s:%d/api/bin-update",BASE_PORT, BASE_URL);
    //ESP_LOGE(TAG, "Server URL: %s",url);
    config.url = task_url_str;
    config.timeout_ms = 10000;
    config.user_data = f_model;
    config.event_handler = _http_down_load_event_handler;
    esp_http_client_handle_t client = esp_http_client_init(&config);
    //esp_http_client_set_buffer_size(client, 4096); 
    //FlaskState state = FlaskState::INIT_EXPORTER; 
    if (!client) {
        ESP_LOGE("OTA", "esp_http_client_init failed");
        fclose(f_model);
        f_model = NULL;
        vTaskDelete(NULL);
    }

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI("OTA", "Model downloaded successfully (%d bytes)",
                 (int)esp_http_client_get_content_length(client));
         flask_state_flag[DOWN_LOAD_MODEL]=true;

    } else {
        ESP_LOGE("OTA", "HTTP request failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    fclose(f_model);
    f_model = NULL;

    ESP_LOGI("MEM", "After OTA: Heap free=%d, internal free=%d",
             (int)esp_get_free_heap_size(),
             (int)esp_get_free_internal_heap_size());

    vTaskDelete(NULL);
}
  


void ota_check_update(void) {
    char url[128];
    snprintf(url, sizeof(url), "%s/api/ota/check-update?version=%s", OTA_SERVER_URL, CURRENT_VERSION);

    esp_http_client_config_t config = {
        .url = url,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        if (status == 200) {
            int len = esp_http_client_get_content_length(client);
            char *buffer =(char*) malloc(len + 1);
            esp_http_client_read(client, buffer, len);
            buffer[len] = '\0';
            
            ESP_LOGI(TAG, "Response: %s", buffer);

            // 解析 JSON
            cJSON *root = cJSON_Parse(buffer);
            if (root) {
                cJSON *update_available = cJSON_GetObjectItem(root, "update_available");
                if (cJSON_IsTrue(update_available)) {
                    ESP_LOGI(TAG, "New version available: %s", 
                             cJSON_GetObjectItem(root, "latest_version")->valuestring);
                } else {
                    ESP_LOGI(TAG, "No update available");
                }
                cJSON_Delete(root);
            }
            free(buffer);
        }
    } else {
        ESP_LOGE(TAG, "HTTP GET request failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
} 

// CRC32 計算
static uint32_t calc_crc32(const uint8_t *data, size_t len) {
    return esp_crc32_le(0, data, len);
}

void ota_download_package(void) {
    esp_http_client_config_t config = {
        .url = OTA_SERVER_URL "/api/ota/package",
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        if (status == 200) {
            int len = esp_http_client_get_content_length(client);
            char *buffer = (char *)malloc(len + 1);
            if (!buffer) {
                ESP_LOGE(TAG, "malloc failed");
                esp_http_client_cleanup(client);
                return;
            }

            int read_len = esp_http_client_read(client, buffer, len);
            buffer[read_len] = '\0';

            ESP_LOGI(TAG, "OTA Package JSON: %s", buffer);

            // === 1. 解析 JSON ===
            cJSON *root = cJSON_Parse(buffer);
            if (!root) {
                ESP_LOGE(TAG, "JSON parse error");
                free(buffer);
                esp_http_client_cleanup(client);
                return;
            }

            // 假設 OTA 包 JSON 格式:
            // { "data": "<base64 string>", "crc32": 12345678 }
            cJSON *data_b64 = cJSON_GetObjectItem(root, "data");
            cJSON *crc_item = cJSON_GetObjectItem(root, "crc32");

            if (!cJSON_IsString(data_b64) || !cJSON_IsNumber(crc_item)) {
                ESP_LOGE(TAG, "Invalid JSON format");
                cJSON_Delete(root);
                free(buffer);
                esp_http_client_cleanup(client);
                return;
            }

            const char *b64_str = data_b64->valuestring;
            uint32_t expected_crc = (uint32_t)crc_item->valuedouble;

            // === 2. base64 decode ===

            size_t out_len = 0;
            int ret = mbedtls_base64_decode(NULL, 0, &out_len,
                                            (const unsigned char *)b64_str, strlen(b64_str));
            if (ret != 0 && ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) {
                ESP_LOGE(TAG, "Base64 length calc failed (%d)", ret);
                return;
            }

            uint8_t *bin_data =(uint8_t *) malloc(out_len);
            if (!bin_data) {
                ESP_LOGE(TAG, "malloc failed");
                return;
            }

            ret = mbedtls_base64_decode(bin_data, out_len, &out_len,
                                        (const unsigned char *)b64_str, strlen(b64_str));
            if (ret != 0) {
                ESP_LOGE(TAG, "Base64 decode failed (%d)", ret);
                free(bin_data);
                return;
            }

            ESP_LOGI(TAG, "Decoded length = %d", (int)out_len);


 

            // === 3. CRC32 校驗 ===
            uint32_t actual_crc = calc_crc32(bin_data, out_len);
            if (actual_crc == expected_crc) {
                ESP_LOGI(TAG, "CRC32 OK (0x%d)",(int) actual_crc);

                // TODO: 把 bin_data 寫到 Flash / SPIFFS / 模型加載
            } else {
                ESP_LOGE(TAG, "CRC32 mismatch! expected=0x%d, got=0x%d",
                        (int)  expected_crc, (int) actual_crc);
            }

            // 釋放
            free(bin_data);
            cJSON_Delete(root);
            free(buffer);
        }
    } else {
        ESP_LOGE(TAG, "Failed to download OTA package: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
}

 

void flask_download_model_ota_task(void *param){
     // ======= OTA 更新模型 =======otaUpdater.downloadModel()
        char download_str[32]="api/download-update";
        char download_url[128];
        sprintf(download_url, "http://%s:%s/%s", BASE_URL,BASE_PORT,download_str); 
        ESP_LOGI(TAG, "download_url URL: %s", download_url);
    //char download_url[128];
    //sprintf(download_url, "http://%s/api/download-update", BASE_URL);
        char check_str[48]="api/check-update/device001/1.0.0";
        char check_url[128];
        sprintf(check_url, "http://%s:%s/%s", BASE_URL,BASE_PORT,check_str); 
        ESP_LOGI(TAG, "check_url URL: %s", check_url);
    //char check_url[128];
    //sprintf(check_url, "http://%s/api/check-update/device001/1.0.0", BASE_URL);
      
     ModelOTAUpdater otaUpdater(check_url, download_url, "/spiffs/model.tflite","0.0.1");
        flask_state_flag[DOWN_LOAD_MODEL_OTA]=false;

    if (otaUpdater.checkUpdate()) {
        ESP_LOGI(TAG, "Model OTA update completed!");
         
        flask_state_flag[DOWN_LOAD_MODEL_OTA]=true;
         
    } else {
        ESP_LOGW(TAG, "Model OTA update failed or no update available");
    }
return  ;
}

void (*functionArray[NUM_FLASK_TASK])(void *pvParameters) = {
    http_get_model_json,
    flask_init_exporter_task, 
    flask_download_model_task, 
    flask_download_model_ota_task
};

//const int num_flask_task=3;
char flask_name[NUM_FLASK_TASK][64]={
    "http_get_model_json",
    "flask_init_exporter_task",
    "flask_download_model_task",
    "flask_download_model_ota_task"
};

extern "C" void wifi_ota_ppo_package(int type ) {
    if(type>=0 && type<NUM_FLASK_TASK)
    //if(type==0)
    {
        xTaskCreatePinnedToCore(
        functionArray[type],  //download_model_ota_task,download_model_task
        flask_name[type],
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   

    } 
    return   ;
}

 
  
extern "C" void hvac_agent(void) {
     
    vTaskDelay(pdMS_TO_TICKS(5000)); // 等 WiFi 連線

   
    // ======= 從 SPIFFS 加載模型 =======
    if (ewcppoModel.loadModelFromSPIFFS("/spiffs/model.tflite")) {
        ESP_LOGI(TAG, "Model loaded successfully");
    } else {
        ESP_LOGE(TAG, "Failed to load model");
    } 
    std::vector<float> observation(5, 0.0f); // 5维状态
    // 填充实际数据
    observation[0] = bp_pid_th.t_feed;
    observation[1] = bp_pid_th.h_feed;
    observation[2] = bp_pid_th.l_feed;
    observation[3] = bp_pid_th.c_feed;
    observation[4] = 0;
    // ======= 推理 =======
    std::vector<float> action_probs = ewcppoModel.predict(observation);
    //std::vector<float> value = ewcppoModel.predictValue(observation);
    float value = ewcppoModel.predictValue(observation); 
    // 假设优势函数 = Q(s, a) - V(s)，我们计算优势。
    std::vector<float> advantages(action_probs.size(), 0.0f);
    for (size_t i = 0; i < action_probs.size(); ++i) {
        advantages[i] = action_probs[i] - value ;  // 这里是一个简单的示例，按需调整
    }

    printf("Action probs: ");
    for (auto v : action_probs) printf("%.3f ", v);
    printf("\n");

    // ======= 持续学习 (EWC) =======
    std::vector<float> newExperience = observation;  // 假设新经验就是当前观测值

    static PPOModel ppoModel;  // 只初始化一次，之后重复使用
    std::vector<float> old_probs = action_probs;  // 示例的旧概率

     
    // 更新旧动作概率
    static std::vector<float> old_action_probs = {0.0f, 0.0f, 0.0f};  

    std::vector<float> grads;
    ppoModel.calculateLossAndGradients(newExperience, old_probs, advantages, health_result, old_action_probs, grads);


    ewcppoModel.continualLearningEWC(grads);

    vTaskDelay(pdMS_TO_TICKS(5000)); // 每 5 秒推理一次
    // 更新旧动作概率以供下一轮使用
    old_action_probs = action_probs;

    // 可选：打印梯度
    ESP_LOGI(TAG, "Gradients: ");
    for (auto& grad : grads) {
        printf("%.3f ", grad);
    }
    printf("\n"); 
 
 
}
 
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
    if (ppoNN.loadWeights("/spiffs/model.bin")==false) {
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

 