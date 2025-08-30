// main.c (ESP-IDF project single-file illustrative)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"

#include "includes.h"
  

#include "esp_log.h"
#include <esp_task_wdt.h>
 
#include "config_mqtt.h"  
#include "infer_esp32_lstm_lll.h"
#include "ml_pid.h"
static const char *TAG = "main      ";
#define PID_RULE_EN

SemaphoreHandle_t mutex_mainTask;
EventGroupHandle_t main_eventGroup;    

void main_creat_objs(void)
{
    mutex_mainTask  = xSemaphoreCreateMutex();
    main_eventGroup = xEventGroupCreate();
}

void main_task_suspend(void) { xSemaphoreTake(mutex_mainTask, portMAX_DELAY); }

void main_task_resume(void) { xSemaphoreGive(mutex_mainTask); }
#define CHECK_ERROR(expr, message) \
    do { \
        esp_err_t __err = (expr); \
        if (__err != ESP_OK) { \
            ESP_LOGE("MAIN", "%s: 0x%x", message, __err); \
            /* 可以添加恢復邏輯 */ \
        } \
    } while (0)
void main_task(void *pvParameters)
{
	EventBits_t bits = 0;
    static uint16_t flag_100ms=0;
    extern void gpio_init(void);
    extern void gpio_sw_cfg_all(void); 
	gpio_init();
    gpio_sw_cfg_all();
    
	while (1)
	{   
        bits = xEventGroupWaitBits(main_eventGroup, BIT0 | BIT1, pdTRUE, pdFALSE, portMAX_DELAY); 
		xSemaphoreTake(mutex_mainTask, portMAX_DELAY);
		flag_100ms++;
          
        if (flag_100ms>=1000)
        {
            flag_100ms = 0;	
            CHECK_ERROR(lll_tensor_run(), "Tensor run failed");
            // lll_tensor_run(); 
        }  
         
        //#if ((_TYPE_of(VER_HARDWARE) == _TYPE_(_OUTLET)))
		//	switch_sync_sta();
		//#endif
		//}
		vTaskDelay(pdMS_TO_TICKS(1)); 

		xSemaphoreGive(mutex_mainTask);  
	} 
	//vTaskDelete(NULL);
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
if(1)    
{
    init_classifier_from_header();

    initialize_nvs_robust();
    
    //wifi_init_apsta();  
     
   	//xTaskCreate(wifi_task, 			TASK_NAME_WIFI, 	TASK_STACK_WIFI, 	NULL, TASK_PRIO_WIFI, 	NULL);
    
    //start_mqtt_client(NULL);

    //xTaskCreate(&periodic_task, "periodic_task", 8192, NULL, 5, NULL);
}     
    xTaskCreate(main_task, 			TASK_NAME_MAIN, 	TASK_STACK_MAIN, 	NULL, TASK_PRIO_MAIN, 	NULL);
     
 
   
}
