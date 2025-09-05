#include "fl_client.h"
#include "esp_http_client.h"
#include "cJSON.h"
#include <stdio.h>

bool fetch_seq_from_server(std::vector<std::vector<float>>& seq_input, const std::string& url) {
    esp_http_client_config_t config = {};
    config.url = url.c_str();
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) return false;

    if (esp_http_client_perform(client) != ESP_OK) {
        printf("fetch_seq_from_server HTTP request failed\n");
        esp_http_client_cleanup(client);
        return false;
    }

    int content_length = esp_http_client_get_content_length(client);
    //std::string buffer(content_length, 0);
    std::vector<char> buffer(content_length);
    int read_len = esp_http_client_read(client, buffer.data(), content_length);


    //esp_http_client_read(client, buffer.data(), content_length);
    esp_http_client_cleanup(client); 

    // 转成 std::string
    std::string buf_str(buffer.begin(), buffer.end());


    cJSON* root = cJSON_Parse(buf_str.c_str());
    if (!root) return false;

    cJSON* arr = cJSON_GetObjectItem(root, "seq_input");
    if (!cJSON_IsArray(arr)) { cJSON_Delete(root); return false; }

    int seq_len = cJSON_GetArraySize(arr);
    int n_features = cJSON_GetArraySize(cJSON_GetArrayItem(arr,0));
    seq_input.resize(seq_len, std::vector<float>(n_features,0.0f));

    for (int i = 0; i < seq_len; ++i) {
        cJSON* row = cJSON_GetArrayItem(arr,i);
        for (int j = 0; j < n_features; ++j) {
            seq_input[i][j] = static_cast<float>(cJSON_GetArrayItem(row,j)->valuedouble);
        }
    }

    cJSON_Delete(root);
    return true;
}
