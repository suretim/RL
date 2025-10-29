#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "mqtt_client.h"
#include "cJSON.h"
 
#define MQTT_BROKER "mqtt://YOUR_MQTT_BROKER_IP"
#define MQTT_PORT 1883
#define MQTT_TOPIC "growlink/obs"

// Wi-Fi and MQTT client handle
static esp_mqtt_client_handle_t client;

static const char *TAG = "ESP32_MQTT";
 
// MQTT event handler
static esp_err_t mqtt_event_handler(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT Connected");
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT Disconnected");
            break;
        default:
            ESP_LOGI(TAG, "Other MQTT event: %d", event->event_id);
            break;
    }
    return ESP_OK;
}

// MQTT data publishing function
void mqtt_publish_data() {
    // Simulating sensor data
    float temperature = 25.0;
    float humidity = 50.0;
    float light = 0.5;
    int soil_moisture = 500;
    int co2_level = 600;

    // Create a JSON object to hold the observation data
    cJSON *obs_data = cJSON_CreateObject();
    cJSON *obs = cJSON_CreateArray();
    cJSON_AddItemToObject(obs_data, "obs", obs);

    cJSON_AddNumberToArray(obs, temperature);
    cJSON_AddNumberToArray(obs, humidity);
    cJSON_AddNumberToArray(obs, light);
    cJSON_AddNumberToArray(obs, soil_moisture);
    cJSON_AddNumberToArray(obs, co2_level);

    // Convert the JSON object to a string
    char *json_data = cJSON_Print(obs_data);

    // Publish the JSON data to the MQTT topic
    esp_mqtt_client_publish(client, MQTT_TOPIC, json_data, 0, 1, 0);

    ESP_LOGI(TAG, "Published: %s", json_data);

    // Clean up the JSON object after publishing
    cJSON_Delete(obs_data);
    free(json_data);
}

// MQTT client initialization
void mqtt_app_start() {
    esp_mqtt_client_config_t mqtt_cfg = {
        .uri = MQTT_BROKER,
        .event_handle = mqtt_event_handler,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_start(client);
}

void app_mqtt() {
    // Initialize NVS (required by Wi-Fi)
    //ESP_ERROR_CHECK(nvs_flash_init());

    // Initialize Wi-Fi
    //wifi_init_sta();

    // Initialize MQTT client
    mqtt_app_start();

    // Periodically publish sensor data every 5 seconds
    while (1) {
        mqtt_publish_data();
        vTaskDelay(5000 / portTICK_PERIOD_MS);  // Wait for 5 seconds before sending next data
    }
}
