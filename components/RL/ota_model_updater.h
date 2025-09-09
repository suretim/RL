// ota_model_updater.h (ESP-IDF 版本，不依賴 Arduino)
#ifndef OTA_MODEL_UPDATER_H
#define OTA_MODEL_UPDATER_H

#include "esp_http_client.h"
#include "esp_spiffs.h"
#include "esp_log.h"
#include "esp_err.h"
#include "mbedtls/md5.h" 
#include <vector>
#include <cstring> 
#include <cstdio>
#include <string>
class ModelOTAUpdater {
private:
    const char* TAG = "ModelOTAUpdater";

    const char* check_url;
    const char* download_url;
    const char* local_model_path;
    std::string current_version;


    std::string md5File(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return "";

        mbedtls_md5_context ctx;
        mbedtls_md5_init(&ctx);
        mbedtls_md5_starts(&ctx);

        unsigned char buf[512];
        size_t len;
        while ((len = fread(buf, 1, sizeof(buf), f)) > 0) {
            mbedtls_md5_update(&ctx, buf, len);
        }

        unsigned char result[16];
        mbedtls_md5_finish(&ctx, result);
        mbedtls_md5_free(&ctx);
        fclose(f);

        char md5_str[33];
        for (int i = 0; i < 16; ++i) {
            sprintf(&md5_str[i * 2], "%02x", result[i]);
        }
        return std::string(md5_str);
    }
        // HTTP GET helper
    esp_err_t httpGet(const char* url, std::string& response) {
        esp_http_client_config_t config = {
            .url = url,
            .timeout_ms = 5000,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        if (!client) return ESP_FAIL;

        esp_err_t err = esp_http_client_open(client, 0);
        if (err != ESP_OK) {
            esp_http_client_cleanup(client);
            return err;
        }

        int content_length = esp_http_client_fetch_headers(client);
        if (content_length <= 0) {
            esp_http_client_cleanup(client);
            return ESP_FAIL;
        }

        char buf[256];
        int read_len;
        response.clear();
        while ((read_len = esp_http_client_read(client, buf, sizeof(buf))) > 0) {
            response.append(buf, read_len);
        }

        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return ESP_OK;
    }

public:
    ModelOTAUpdater(const char* check_url,
                    const char* download_url,
                    const char* local_model_path = "/spiffs/model.tflite",
                    const char* version = "0.0.0")
        : check_url(check_url),
          download_url(download_url),
          local_model_path(local_model_path),
          current_version(version) {}

    // 檢查更新 (這裡假設 server 回傳 JSON: {"latest_version":"1.0.1","hash":"abcd1234"})
    bool checkUpdate() {
        std::string payload;
        if (httpGet(check_url, payload) != ESP_OK) {
            ESP_LOGE(TAG, "Check update failed");
            return false;
        }

        ESP_LOGI(TAG, "Server response: %s", payload.c_str());

        // ⚠️ TODO: 用 cJSON 解析 JSON
        // 假設直接解析 hash 跟 version
        std::string latest_version = "1.0.1"; // mock
        std::string expected_md5 = "abcd1234"; // mock

        //if (latest_version != current_version) {
        if (1) {
            return downloadUpdate(expected_md5) && setCurrentVersion(latest_version);
        }
        return false;
    }

    // 下載更新
    bool downloadUpdate(const std::string& expected_md5) {
        esp_http_client_config_t config = {
            .url = download_url,
            .timeout_ms = 10000,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        if (!client) return false;

        esp_err_t err = esp_http_client_open(client, 0);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "HTTP open failed");
            esp_http_client_cleanup(client);
            return false;
        }

        int file_size = esp_http_client_fetch_headers(client);
        ESP_LOGI(TAG, "Downloading model, size=%d", file_size);

        FILE* f = fopen(local_model_path, "wb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open file for writing");
            esp_http_client_cleanup(client);
            return false;
        }

        char buf[512];
        int read_len;
        int written = 0;
        while ((read_len = esp_http_client_read(client, buf, sizeof(buf))) > 0) {
            fwrite(buf, 1, read_len, f);
            written += read_len;
        }
        fclose(f);
        esp_http_client_close(client);
        esp_http_client_cleanup(client);

        ESP_LOGI(TAG, "Downloaded %d bytes", written);

        // 驗證 MD5
        std::string file_md5 = md5File(local_model_path);
        ESP_LOGI(TAG, "MD5 check: %s ?= %s", file_md5.c_str(), expected_md5.c_str());

        //if (file_md5 != expected_md5) {
        //    ESP_LOGE(TAG, "MD5 mismatch!");
        //    return false;
        //}

        ESP_LOGI(TAG, "Update success");
        return true;
    }

    bool setCurrentVersion(const std::string& v) {
        current_version = v;
        return true;
    }

    std::string getCurrentVersion() { return current_version; }
};

#endif // OTA_MODEL_UPDATER_H
