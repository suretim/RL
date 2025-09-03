// esp32_ota_client.h
#pragma once

#include "esp_http_client.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "cJSON.h"
#include <string>
#include <cstring>

static const char *TAG = "ESP32_OTA";

class ESP32OTAClient {
private:
    std::string deviceId;
    std::string currentVersion;
    std::string otaServerUrl;

public:
    ESP32OTAClient(const std::string &serverUrl, const std::string &devId)
        : otaServerUrl(serverUrl), deviceId(devId), currentVersion("0.9.0") {}

    bool checkForUpdates() {
        char url[256];
        snprintf(url, sizeof(url), "%s/api/check-update/%s/%s",
                 otaServerUrl.c_str(), deviceId.c_str(), currentVersion.c_str());

        esp_http_client_config_t config = {
            .url = url,
            .timeout_ms = 5000,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);

        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            int status_code = esp_http_client_get_status_code(client);
            if (status_code == 200) {
                int len = esp_http_client_get_content_length(client);
                char *buffer = (char *)malloc(len + 1);
                if (buffer) {
                    esp_http_client_read(client, buffer, len);
                    buffer[len] = '\0';

                    cJSON *json = cJSON_Parse(buffer);
                    if (json) {
                        cJSON *updateAvailable = cJSON_GetObjectItem(json, "update_available");
                        if (cJSON_IsBool(updateAvailable) && cJSON_IsTrue(updateAvailable)) {
                            ESP_LOGI(TAG, "Update available! Downloading...");
                            cJSON_Delete(json);
                            free(buffer);
                            esp_http_client_cleanup(client);
                            return downloadAndInstallUpdate();
                        }
                        cJSON_Delete(json);
                    }
                    free(buffer);
                }
            }
        } else {
            ESP_LOGE(TAG, "HTTP GET failed: %s", esp_err_to_name(err));
        }

        esp_http_client_cleanup(client);
        return false;
    }

    bool downloadAndInstallUpdate() {
        char url[256];
        snprintf(url, sizeof(url), "%s/api/download-update", otaServerUrl.c_str());

        esp_http_client_config_t config = {
            .url = url,
            .timeout_ms = 10000,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        esp_err_t err = esp_http_client_open(client, 0);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to open HTTP connection: %s", esp_err_to_name(err));
            esp_http_client_cleanup(client);
            return false;
        }

        int binary_file_length = 0;
        esp_ota_handle_t ota_handle;
        const esp_partition_t *update_partition = esp_ota_get_next_update_partition(NULL);

        ESP_ERROR_CHECK(esp_ota_begin(update_partition, OTA_SIZE_UNKNOWN, &ota_handle));

        char buffer[1024];
        int data_read;
        while ((data_read = esp_http_client_read(client, buffer, sizeof(buffer))) > 0) {
            ESP_ERROR_CHECK(esp_ota_write(ota_handle, buffer, data_read));
            binary_file_length += data_read;
        }

        ESP_LOGI(TAG, "Written image length %d", binary_file_length);

        if (esp_ota_end(ota_handle) == ESP_OK) {
            ESP_ERROR_CHECK(esp_ota_set_boot_partition(update_partition));
            ESP_LOGI(TAG, "OTA update complete, restarting...");
            esp_restart();
        } else {
            ESP_LOGE(TAG, "OTA end failed");
            esp_http_client_cleanup(client);
            return false;
        }

        esp_http_client_cleanup(client);
        return true;
    }
};
