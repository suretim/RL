#include "esp_log.h"
#include "esp_system.h"
#include "esp_spiffs.h"
#include "esp_http_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

//#include "tensorflow/lite/micro/all_ops_resolver.h"
//#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"
#include "esp_wifi.h"
#include "esp_event.h" 
#include "nvs_flash.h"
#include "ulog.h"

#include <stdio.h>
#include <stdlib.h>
#include "config_wifi.h"
#include "infer_esp32_lstm_lll.h"
#include "plant_make.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "config_spiffs.h"


const char *TAG = "NN_OTA_AUTO";

//#define TENSOR_ARENA_SIZE 10*1024
//#define OTA_INTERVAL_MS 3600000   // 1 小时

//uint8_t tensor_arena[TENSOR_ARENA_SIZE];
//tflite::MicroInterpreter *interpreter = NULL;
//TfLiteTensor *input_tensor = NULL;
//TfLiteTensor *output_tensor = NULL;
//extern const char *model_path ;
#if 0
// OTA 任务
void ota_task(void *param) {
    const char *model_url = (const char*)param;
    while(1) {
        if(download_tflite(model_url, model_path) == ESP_OK) {
            ESP_LOGI(TAG, "Model downloaded, re-initialize interpreter...");
            if(init_spiffs_interpreter(model_path)) {
                ESP_LOGI(TAG, "Interpreter updated with new model");
            }
        }
        vTaskDelay(OTA_INTERVAL_MS / portTICK_PERIOD_MS);
    }
}
#endif
static SemaphoreHandle_t s_infer_mutex = NULL;

SemaphoreHandle_t mutex_mainTask;
EventGroupHandle_t app_event_group;
// 事件组 bit 定义
#define WIFI_CONNECTED_BIT BIT0
#define MQTT_CONNECTED_BIT BIT1
#define MODEL_READY_BIT    BIT2
const char *model_url = "http://192.168.0.57:5000/download_tflite";
void model_init() {
    s_infer_mutex = xSemaphoreCreateMutex();
     
    // 第一次初始化
    //if(download_tflite(model_url, model_path) == ESP_OK) {
    //    if(init_spiffs_interpreter(model_path)) {
    //        ESP_LOGI(TAG, "Initial model loaded");
    //    }
    //}
    xEventGroupSetBits(app_event_group, MODEL_READY_BIT);

    // 在使用者
    if (!(xEventGroupWaitBits(app_event_group, MODEL_READY_BIT, pdFALSE, pdTRUE, 0) & MODEL_READY_BIT)) {
        ESP_LOGW(TAG, "model not ready yet, skip inference");
        return;
    }
} 
//extern void wifi_ota_ppo_package(void);

extern bool init_exporter_flag;

void plant_env_make_task(void *pvParameters)
{ 
	vTaskDelay(3000 / portTICK_PERIOD_MS);
	
	//wifi_ota_ppo_package(0); 
	while(1)
	{  
        if(init_exporter_flag==false)
            wifi_ota_ppo_package(0); 
		if(plant_env_step() ==0){
			vTaskDelay(pdMS_TO_TICKS(5000));
			break;
		}
		vTaskDelay(pdMS_TO_TICKS(20000));
	} 
}

void app_main(void) {

    esp_err_t ret = 0;
	ret = nvs_flash_init();
	if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
	{
		ESP_ERROR_CHECK(nvs_flash_erase());
		ret = nvs_flash_init();
	}
	ESP_ERROR_CHECK(ret); 
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    spiffs_init();

	wifi_init_sta();
    

    // 创建 OTA 更新任务
   // xTaskCreate(ota_task, "ota_task", 8192, (void*)model_url, 5, NULL);

    

     xTaskCreate(plant_env_make_task,TASK_NAME_COMMDECODE, TASK_STACK_COMMDECODE, NULL, TASK_PRIO_COMMDECODE, NULL);	// 通信数据解析
 
	vTaskDelete(NULL);
}
