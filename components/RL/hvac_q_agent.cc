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

static const char *TAG = "MAIN";

// ======= WiFi 與 OTA 配置 =======
const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";
const char* check_url = "http://server_ip:5000/api/check-update/device001/1.0.0";
const char* download_url = "http://server_ip:5000/api/download-update";

//ModelOTAUpdater otaUpdater(ssid, password, check_url, download_url, "/spiffs/model.tflite");
ModelOTAUpdater otaUpdater(check_url, download_url, "/spiffs/model.tflite");

// ======= PPO 模型 =======
ESP32PPOModel ppoModel;

// ======= 範例觀測值 =======
std::vector<float> observation = {0.1, 0.2, 0.3, 0.4, 0.5};

// ======= 初始化 SPIFFS =======
void init_spiffs() {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount or format filesystem (%s)", esp_err_to_name(ret));
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "SPIFFS total: %d, used: %d", total, used);
    }
}

// ======= WiFi 連線 =======
void wifi_init_sta() {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, ssid);
    strcpy((char*)wifi_config.sta.password, password);

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    ESP_LOGI(TAG, "WiFi connecting to %s...", ssid);
    esp_wifi_connect();
}

// ======= 主任務 =======
//extern "C" void app_main(void) {
extern "C" void hvac_agent(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    init_spiffs();
    wifi_init_sta();

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
