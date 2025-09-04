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
 
#include "ota_model_updater.h"
#include "esp32_model_loader.h"
#include "ml_pid.h"
#include "nn.h" 
 


static const char *TAG = "OTA HVAC";
// ======= WiFi 與 OTA 配置 ======= 
#if 1
const char* check_url = "http://192.168.68.237:5000/api/check-update/device001/1.0.0";
const char* download_url = "http://192.168.68.237:5000/api/download-update";
#define OTA_URL "http://192.168.68.237:5000/api/bin-update"
#else
const char* check_url = "http://192.168.0.57:5000/api/check-update/device001/1.0.0";
const char* download_url = "http://192.168.0.57:5000/api/download-update";
#define OTA_URL "http://192.168.68.237:5000/api/bin-update"
#endif

#define LOCAL_MODEL_FILE "/spiffs/ppo_model.bin"
 
#define WIFI_SSID "YourWiFiSSID"
#define WIFI_PASS "YourWiFiPassword"
 
 #include "esp_http_client.h"

static FILE *f_model = NULL;
  
// ---------------- NN Placeholder ----------------
// 权重向量示例
std::vector<float> W1, W2, b1, b2, Vw;
float Vb = 0.0f;

void parse_model_weights(uint8_t *buffer, size_t size) {
    // TODO: 解析 buffer 填充 W1, W2, b1, b2, Vw, Vb
    ESP_LOGI(TAG, "Parsing model weights... (%d bytes)", size);
}
 
 
// ---------------- Local fallback ----------------
bool load_local_model() {
    FILE *f = fopen(LOCAL_MODEL_FILE, "rb");
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

// ---------------- OTA Callback ----------------
esp_err_t _http_event_handler0(esp_http_client_event_t *evt) {
    switch(evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            if (f_model) fwrite(evt->data, 1, evt->data_len, f_model);
            break;
        case HTTP_EVENT_ON_FINISH:
            if (f_model) {
                fclose(f_model);
                f_model = NULL;
                ESP_LOGI(TAG, "OTA file saved to %s", LOCAL_MODEL_FILE);
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
 
//ModelOTAUpdater otaUpdater(ssid, password, check_url, download_url, "/spiffs/model.tflite");
ModelOTAUpdater otaUpdater(check_url, download_url, "/spiffs/model.tflite");

// ======= PPO 模型 =======
ESP32EWCModel ppoModel;
  
// ======= 初始化 SPIFFS =======
extern "C" void init_spiffs(void) {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "spifs",
        .max_files = 5,
        .format_if_mount_failed = true
    };

     
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount or format filesystem (%s)", esp_err_to_name(ret));
        vTaskDelay(10000 / portTICK_PERIOD_MS); 
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "SPIFFS total: %d, used: %d", total, used);
    }
}
#if 0 
 // ======= 主任務 =======
//extern "C" void app_main(void) {
extern "C" void hvac_agent(void) {
    //ESP_ERROR_CHECK(nvs_flash_init());
    //init_spiffs();
    //wifi_init_sta();

    vTaskDelay(pdMS_TO_TICKS(5000)); // 等 WiFi 連線

    // ======= OTA 更新模型 =======otaUpdater.downloadModel()
    if (otaUpdater.checkUpdate()) {
        ESP_LOGI(TAG, "Model OTA update completed!");
    } else {
        ESP_LOGW(TAG, "Model OTA update failed or no update available");
    }

    // ======= 從 SPIFFS 加載模型 =======
    if (ppoModel.loadModelFromSPIFFS("/spiffs/model.tflite")) {
        ESP_LOGI(TAG, "Model loaded successfully");
    } else {
        ESP_LOGE(TAG, "Failed to load model");
    }

    while (1) {
        //read_all_sensor();
         
        std::vector<float> observation(5, 0.0f); // 5维状态
        // 填充实际数据
        observation[0] = bp_pid_th.t_feed;
        observation[1] = bp_pid_th.h_feed;
        observation[2] = bp_pid_th.l_feed;
        observation[3] = bp_pid_th.c_feed;
        observation[4] = 0;
        // ======= 推理 =======
        std::vector<float> action_probs = ppoModel.predict(observation);
        printf("Action probs: ");
        for (auto v : action_probs) printf("%.3f ", v);
        printf("\n");

        // ======= 持續學習 (EWC) =======
        std::vector<float> newExperience = observation; // 假設新經驗就是觀測值
        ppoModel.continualLearningEWC(newExperience);

        vTaskDelay(pdMS_TO_TICKS(5000)); // 每 5 秒推理一次
    }
}

#endif


esp_err_t _http_event_handler(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            if (evt->user_data) {
                FILE *file = (FILE *)evt->user_data;
                fwrite(evt->data, 1, evt->data_len, file);
            }
            break;
        default:
            break;
    }
    return ESP_OK;
}

static void download_model_task(void *param) {
    ESP_LOGI("MEM", "Before OTA: Heap free=%d, internal free=%d, largest=%d",
             (int)esp_get_free_heap_size(),
             (int)esp_get_free_internal_heap_size(),
             (int)heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));

    // 删除旧模型
    remove(LOCAL_MODEL_FILE);
    f_model = fopen(LOCAL_MODEL_FILE, "wb");
    if (!f_model) {
        ESP_LOGE("OTA", "Failed to open file for OTA");
        vTaskDelete(NULL);
        return;
    }
 
    esp_http_client_config_t config = {0}; // 初始化为全零

    config.url = OTA_URL;
    config.timeout_ms = 10000;
    config.user_data = f_model;
    config.event_handler = _http_event_handler;
    esp_http_client_handle_t client = esp_http_client_init(&config);
    //esp_http_client_set_buffer_size(client, 4096); 
     
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

void download_model(void) {
    // 独立任务下载模型，避免占用控制/推理循环的内部 RAM
    xTaskCreatePinnedToCore(
        download_model_task,
        "download_model_task",
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        1);             // 建议放到 Core1，避免和 Wi-Fi 核抢栈
}


bool is_network_connected() {
    // Check if WiFi is connected to AP
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) != ESP_OK) {
        return false; // Not connected to any AP
    }
    
    // Check if we have an IP address
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA");
    if (!netif) {
        return false;
    }
    
    esp_netif_ip_info_t ip_info;
    if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
        return false;
    }
    
    // Check if we have a valid IP (not 0.0.0.0)
    return (ip_info.ip.addr != 0);
}

static bool got_ip = false;

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                             int32_t event_id, void* event_data)
{
    if (event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        got_ip = false;
        ESP_LOGI(TAG, "WiFi disconnected, reconnecting...");
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_wifi_connect();
    } else if (event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        got_ip = true;
    }
}

void register_network_events() {
    got_ip = false;
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL);
}

bool wait_for_ip_event(int timeout_seconds) {
    ESP_LOGI(TAG, "Waiting for IP event...");
    
    for (int i = 0; i < timeout_seconds; i++) {
        if (got_ip) {
            ESP_LOGI(TAG, "IP event received!");
            return true;
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
    ESP_LOGE(TAG, "No IP event within %d seconds", timeout_seconds);
    return false;
}


void wait_for_network_connection() {
    ESP_LOGI(TAG, "Waiting for network connection...");
    
    for (int i = 0; i < 20; i++) { // Wait up to 20 seconds
        if (is_network_connected()) {
            ESP_LOGI(TAG, "Network fully connected with IP!");
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
        ESP_LOGD(TAG, "Waiting for network... (%d/20)", i + 1);
    }
    
    ESP_LOGE(TAG, "Failed to get network connection within 20 seconds");
}

void debug_current_network_status() {
    ESP_LOGI(TAG, "=== Current Network Status ===");
    
    // Check WiFi connection
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        ESP_LOGI(TAG, "WiFi: Connected to %s, RSSI: %d dBm", 
                ap_info.ssid, ap_info.rssi);
                
    } else {
        ESP_LOGI(TAG, "WiFi: Not connected to AP");
    }
    
    // Check IP address
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA");
    if (netif) {
        esp_netif_ip_info_t ip_info;
        if (esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
            ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&ip_info.ip));
            ESP_LOGI(TAG, "GW: " IPSTR, IP2STR(&ip_info.gw));
        }
    }
    
    ESP_LOGI(TAG, "got_ip flag: %d", got_ip);
    ESP_LOGI(TAG, "=================================");
}


void print_network_info() {
    // Check WiFi connection first
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) != ESP_OK) {
        ESP_LOGI(TAG, "Not connected to WiFi");
        return;
    }
    
    ESP_LOGI(TAG, "Connected to: %s", ap_info.ssid);
    ESP_LOGI(TAG, "RSSI: %d dBm", ap_info.rssi);
    
    // Get IP information
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA");
    if (!netif) {
        ESP_LOGE(TAG, "WiFi interface not found");
        return;
    }
    
    esp_netif_ip_info_t ip_info;
    if (esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
        if (ip_info.ip.addr == 0) {
            ESP_LOGI(TAG, "No IP address assigned (DHCP in progress)");
        } else {
            ESP_LOGI(TAG, "IP Address: " IPSTR, IP2STR(&ip_info.ip));
            ESP_LOGI(TAG, "Gateway: " IPSTR, IP2STR(&ip_info.gw));
            ESP_LOGI(TAG, "Netmask: " IPSTR, IP2STR(&ip_info.netmask));
        }
    } else {
        ESP_LOGE(TAG, "Failed to get IP information");
    }
}
bool check_ota_update() {
    ESP_LOGI(TAG, "Starting OTA check...");
    register_network_events();
    //debug_current_network_status();
    print_network_info();
    // Method 1: Wait for IP event (most reliable)
    if (!wait_for_ip_event(15)) {
        ESP_LOGE(TAG, "No IP address, skipping OTA");
        vTaskDelay(5000 / portTICK_PERIOD_MS);
        return false;
    }
    
    // Method 2: Or use the polling approach
    // wait_for_network_connection();
    
    if (!is_network_connected()) {
        ESP_LOGE(TAG, "Network not connected after waiting, skipping OTA");
        vTaskDelay(5000 / portTICK_PERIOD_MS);
        return false;
    }
    
    ESP_LOGI(TAG, "Network fully connected, attempting OTA update..."); 

        download_model();
        return true;
    // Your OTA update code here
    // esp_err_t result = your_ota_function();
}

extern "C" void wifi_ota_ppo_package(void ) {

    //const char* model_url = "http://127.0.0.1:5000/api/download-update";
    ///const char* bin_url = "http://192.168.0.57:5000/api/bin-update/device001/0.0.0";
    const char* bin_url = "http://192.168.0.57:5000/api/bin-update";
      
    const char* model_path = "/spiffs/ppo_model.bin";
    static int model_version = 0;
    vTaskDelay(1000 / portTICK_PERIOD_MS); // 等待 WiFi 连接
    
    while(1)
    {
       
         if(check_ota_update()==true) {
             break;
         }
        
    }
        model_version=1;  
    return;
}
extern "C" pid_run_output_st nn_ppo_infer() {

    // 尝试下载 OTA model_version == 0 &&
    //if( !ppoModel.downloadModel(bin_url, model_path)) {
    //   ESP_LOGW("downloadModel", "Model will use existing model if available");
    //}

    //const char* version = "1.0.0";
    //ModelOTAUpdater otaUpdater( check_url, download_url, "/spiffs/model.bin",version  );
    static pid_run_output_st 		output_speed;
    // OTA 拉取最新 PPO 参数
    //if (otaUpdater.checkUpdate()) {
    //    printf("✅ Model updated from server\n");
    //}

    // 初始化 PPO 推理网络
    NN  ppoNN(5, 32, 4);  // state_dim=5, hidden=32, action_dim=4
    if (ppoNN.loadWeights("/spiffs/model.bin")) {
        printf("✅ Weights loaded into PPO NN\n");
    }
 
    std::vector<float> state = { bp_pid_th.t_feed, bp_pid_th.h_feed, bp_pid_th.l_feed, bp_pid_th.c_feed, 1.0};
    auto action_probs = ppoNN.forwardActor(state);
    float value = ppoNN.forwardCritic(state);
    output_speed.speed[0] = action_probs[0]>0.5?1:0;
    output_speed.speed[1] = action_probs[1]>0.5?1:0;
    output_speed.speed[2] = action_probs[2]>0.5?1:0;
    output_speed.speed[3] = action_probs[3]>0.5?1:0;
    printf("Action probs: ");
    for (auto p : action_probs) printf("%.3f ", p);
    printf("\nCritic value: %.3f\n", value);
    return output_speed;
}

 