// main.c (ESP-IDF project single-file illustrative)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"

#include "includes.h"
  

#include "esp_log.h"
#include <esp_task_wdt.h>

//#include "classifier_storage.h"
#include "config_mqtt.h" 
#include "config_wifi.h" 

static const char *TAG = "main      ";


SemaphoreHandle_t mutex_mainTask;
EventGroupHandle_t main_eventGroup; 
//extern void publish_feature_vector(int label,int type );   

void main_creat_objs(void)
{
    mutex_mainTask  = xSemaphoreCreateMutex();
    main_eventGroup = xEventGroupCreate();
}

void main_task_suspend(void) { xSemaphoreTake(mutex_mainTask, portMAX_DELAY); }

void main_task_resume(void) { xSemaphoreGive(mutex_mainTask); }

void main_task(void *pvParameters)
{
	EventBits_t bits = 0;

	//uint8_t cnt = 0;
    static uint16_t flag_100ms=0;
    extern void gpio_init(void);
    extern void gpio_sw_cfg_all(void);
	 gpio_init();
    gpio_sw_cfg_all();
	while (1)
	{
 
        flag_100ms++;
        if (flag_100ms>=100)
        {
            flag_100ms = 0;	
            lll_tensor_run(); 
        } 

		xSemaphoreGive(mutex_mainTask);

		vTaskDelay(pdMS_TO_TICKS(1)); 
	}

	vTaskDelete(NULL);
}




void periodic_task(void *pvParameter) {
    while (1) {
        publish_feature_vector(0,1);
        vTaskDelay(pdMS_TO_TICKS(120000)); // 延遲 60 秒
    }
}
#define TASK_NAME_MQTT          "MqttTask"
#define TASK_PRIO_MQTT          5//2
#define TASK_STACK_MQTT         (5 * 1024)  //约3K
void app_main(void) {
 
 main_creat_objs(); 
    init_classifier_from_header();

    initialize_nvs_robust();
#if 1    
    wifi_init_apsta();  
#else    
   	xTaskCreate(wifi_task, 			TASK_NAME_WIFI, 	TASK_STACK_WIFI, 	NULL, TASK_PRIO_WIFI, 	NULL);
#endif   
    start_mqtt_client(NULL);
    //xTaskCreate(start_mqtt_client,  TASK_NAME_MQTT, TASK_STACK_MQTT, 	NULL, TASK_PRIO_MQTT, 	NULL);
    // 初始化 SPIFFS
    xTaskCreate(main_task, 			TASK_NAME_MAIN, 	TASK_STACK_MAIN, 	NULL, TASK_PRIO_MAIN, 	NULL);
    xTaskCreate(&periodic_task, "periodic_task", 8192, NULL, 5, NULL);
     
 
   
}
