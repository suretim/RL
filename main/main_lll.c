// main.c (ESP-IDF project single-file illustrative)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"
//#include "nvs_flash.h"
//#include "nvs.h" 
#include "config_mqtt.h" 
#include "config_wifi.h"  
//#include <sstream>
#include "esp_log.h"
#include <esp_task_wdt.h>
#include "classifier_storage.h"


static const char *TAG = "MAIN_LLL";
 
  
 
 
  // Example MQTT callback (pseudo): receives classifier_weights.bin as payload
// In real code wire up esp-mqtt and call this when message arrives

void periodic_task(void *pvParameter) {
    while (1) {
        publish_feature_vector(0,1);
        vTaskDelay(pdMS_TO_TICKS(120000)); // 延遲 60 秒
    }
}
//extern void start_mqtt_client(void);  
extern void lll_tensor_run(void );  
// Example main demonstrating flow
void app_main(void) {
 
 
 //initialize_nvs();
//    esp_err_t ret = nvs_flash_init();
//     if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
//         ESP_ERROR_CHECK(nvs_flash_erase());
//         ret = nvs_flash_init();
//     }
//     ESP_ERROR_CHECK(ret);
    init_classifier_from_header();

    initialize_nvs_robust();
    wifi_init_apsta();   
    start_mqtt_client(); 
    // 初始化 SPIFFS
    xTaskCreate(&periodic_task, "periodic_task", 8192, NULL, 5, NULL);
    // init tflite
    lll_tensor_run();
 
   
}
