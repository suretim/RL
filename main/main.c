#include "esp_system.h"


#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "ulog.h"

#include "esp_bt_device.h"

#include "esp_ota_ops.h"

#include "includes.h"

#include "mcu.h"
#include "can_server_api.h"
#include "ble.h"
#include "wifi.h"
#include "tcpClient.h"
#include "infer_esp32_lstm_lll.h"
#include "plant_make.h"
#include "hvac_q_agent.h"
#include "config_spiffs.h"
#include "classifier_storage.h"
#define TAG "main      " // 10个字符


SemaphoreHandle_t mutex_mainTask;
EventGroupHandle_t main_eventGroup;
extern void ai_check_plug_in(uint8_t track_cc);
extern uint8_t flask_state_get_flag[FLASK_GET_COUNT];
extern uint8_t flask_state_put_flag[FLASK_PUT_COUNT];

/*
****************************************************************************************************
*	函数名称：
*	功能说明：
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
void main_creat_objs(void)
{
    mutex_mainTask  = xSemaphoreCreateMutex();
    main_eventGroup = xEventGroupCreate();
}

void main_task_suspend(void) { xSemaphoreTake(mutex_mainTask, portMAX_DELAY); }

void main_task_resume(void) { xSemaphoreGive(mutex_mainTask); }

/*
****************************************************************************************************
*    函数名称：
*    功能说明：
*    参   数：
*    返 回 值：
*
****************************************************************************************************
*/
static void maintask_timer10ms_callback(void *arg)
{
	static uint8_t cnt = 0;

	xEventGroupSetBits(main_eventGroup, BIT0);
	if (++cnt >= 10)
	{
		cnt = 0;
		xEventGroupSetBits(main_eventGroup, BIT1);
	}
}

static void maintask_timer10ms_Init(void)
{
	const esp_timer_create_args_t periodic_timer_args = {
		.callback = &maintask_timer10ms_callback,
		/* name is optional, but may help identify the timer when debugging */
		.name = "periodic"};

	esp_timer_handle_t periodic_timer;
	ESP_ERROR_CHECK(esp_timer_create(&periodic_timer_args, &periodic_timer));
	/* The timer has been created but is not running yet */

	/* Start the timers */
	ESP_ERROR_CHECK(esp_timer_start_periodic(periodic_timer, 10000));
	ESP_LOGI(TAG, "%s() Started timers, time since boot: %lld us", __func__, esp_timer_get_time());
} 



static void _Init(void)
{
	extern void fg_init(void);

	ulog_init();
	ulog_output_lock_enabled(true);
	ulog_i(TAG, "maintask init()");

	comm_wait_next_timer_init();

	// vTaskDelay(pdMS_TO_TICKS(5000));
	gpio_init();
	//fixbug-v40: 初始化移到前面, 开机自动进入自测模式

#if (_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER))
	key_init(); // touch key ic
#elif(_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))
	extern void uart_Init(void);
	uart_Init();  
#endif

	// hardy: 将 CC 和 Clin 的驱动移动到上面，为自测模式做准备
#if	(_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER)|| _TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))
	fg_init();
	cc_adc_config();

#if (_PORT_of(VER_HARDWARE) == _PORT_(8))
	cc_adc_sel_port(1);
#endif
#endif

	Variable_Init();
#if (_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER)|| _TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))
	CAN_Init();
	CAN_Server_Init();
#endif	

	icon_init();
	InitSpi();
	pwm_init();

	//IIC1_init();  // for sc05b key, pcf8563 rtc, ex-humid sensor sht40 and etc.
    //IIC2_init();  // for zone sht3c, ex-humid sensor sht40 and etc.
#if 0  //tim modify
	ReadAdvance();	// outlet 时， ReadAllParam 中有处理 ADV ， 所以先读ADV
	ReadAllParam(); 
	InitRunData(); // 读取 设备已经运行的时间 sysClk1s 以及 定时模式的剩余时间
	
	if (!rtc_init_check())
	{
		Clear_Log_Data();
		Clear_RunData();
	}
#endif
//#if (_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER))
//		key_init(); // touch key ic
//#endif

#if (_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER))
	lcd_init();
	Power_On_Test_LCD_BZ(); // 按住 mode 键上电开机自检, 必须放在按键初始化后

#elif ((_TYPE_of(VER_HARDWARE) == _TYPE_(_OUTLET)))
	gpio_sw_cfg_all();
	led_init();	
	Power_On_Test_outlet();
	ocDetect_Init();
	//tim modify
	//Enter_start();	
#endif	

	InitHistoryLog();
	InitHisDataInfo();

	get_sun_dynamic_history_data();
	ai_com_mutex_init();

	ESP_LOGI(TAG, "--------------------------------------------------------------------------------");
	ESP_LOGI(TAG, "\t\t%s() init finish....", __func__);
	ESP_LOGI(TAG, "--------------------------------------------------------------------------------\r\n\r\n");

	maintask_timer10ms_Init();
	ulog_i(TAG, "maintask init done!");
}
/*
****************************************************************************************************
*    函数名称：
*    功能说明： 上电获取 ESP32 工作在 蓝牙 还是 WiFi 
*    参    数：
*    返 回 值：
*    
****************************************************************************************************
*/
static void powerON_Pro(void)
{
	esp_err_t ret = 0;
	sw_version_t swV_nvs = {0};

	// Temp_st sock_data = {0};
	// wifi_param_t wifi_Param = {0};
	/*
		代码上电运行时的版本状态， 
		0:  无升级降级动作
		1： 升级， 无群组到有群组
		2： 升级， 有群组到有群组
		3： 降级
	*/
	u8 version_sta = 0;
	
	getVersion((u8 *)(SOFTWARE_VERSION), sw_version, SW_VER_CNT);
	ESP_LOGI(TAG, "%s : sw = %d.%d.%d", __func__, sw_version[0], sw_version[1], sw_version[2]);

	ret = nvs_read_SWV(&swV_nvs);
	/* nvs 区没有版本信息 */
	if (ret != ESP_OK)
	{
		version_sta = 1;	// 升级， 无群组到有群组
	}
	else 
	{
		ESP_LOGI(TAG, "%s : sw_nvs = %d.%d.%d", __func__, swV_nvs.num_0, swV_nvs.num_1, swV_nvs.num_2);

		u8 n = 0;
		u8 sw_temp[3] = {swV_nvs.num_0, swV_nvs.num_1, swV_nvs.num_2};

		do
		{
			if (sw_version[n] > sw_temp[n])
			{
				version_sta = 2;	// 升级， 有群组到有群组
				ESP_LOGI(TAG, "%s : (sw_version[%d] > swV_nvs[%d])", __func__, n, n);
			}
			else if (sw_version[n] < sw_temp[n])
			{
				version_sta = 3;	// 绛级
				ESP_LOGI(TAG, "%s : (sw_version[%d] < swV_nvs[%d])", __func__, n, n);
			}
			else
			{
				ESP_LOGI(TAG, "%s : (sw_version[%d] = swV_nvs[%d])", __func__, n, n);

				// wifi 先降级到 frank 代码， 再升级， 会进入该分支
				// 从 12.0.1 降级到 frank 代码， nvs 区 已经存在 WiFi 参数， 再升级时也不会有问题， 
				// 只不过代码运行的流程 与 首次从 frank 代码升级时 有点区别

			}

		} while ((sw_version[n] == sw_temp[n]) && (++ n < 3));
	}
	ESP_LOGW(TAG, "%s : version_sta = %d\n", __func__, version_sta);

	/* 存储版本信息 */
	if (version_sta)
	{
		swV_nvs.num_0 = sw_version[0];
		swV_nvs.num_1 = sw_version[1];
		swV_nvs.num_2 = sw_version[2];
		nvs_write_SWV(&swV_nvs);
	}

	nvs_read_espMode(&ESP32WorkMode);
	nvs_read_Mac(&Mac);
	ESP_LOGW(TAG, "MAC : " ESP_BD_ADDR_STR "\n", Mac.addr[0], Mac.addr[1], Mac.addr[2], Mac.addr[3], Mac.addr[4], Mac.addr[5]);

	/* 启动 蓝牙 或者 WiFi */
	if (ESP32WorkMode == ESP32_MODE_BLE)
	{
		ESP_LOGW(TAG, "%s --- run ble --- \n", __func__);
		ulog_i(TAG, "%s --- run ble --- \n", __func__);
		ble_set_start();	// ble_setStart();
	}
	else if (ESP32WorkMode == ESP32_MODE_WIFI)
	{
		ESP_LOGW(TAG, "%s --- run wifi --- \n", __func__);
		ulog_i(TAG, "%s --- run wifi --- \n", __func__);
		wifi_set_start();	// wifi_setStart();
	}
}

#ifdef _SYS_TASK_RUN_INFO_OUT
void print_task_minstack(char* task_name)
{
	ESP_LOGW(TAG, "%s:%d",task_name, uxTaskGetStackHighWaterMark( xTaskGetHandle(task_name) ));
}

char task_list_buff[1024];
void print_task_stack_info()
{
#if 1
	vTaskList(task_list_buff);
	ESP_LOGW("TAG","Task List:\nTaskName\tSta\tPro\tMinFree\tCreatName\t\n%s",task_list_buff);
#endif
#if 1 
	char *pc_line, *pc_lattice, *pc_line_end;
	size_t word_len = 0, task_occupied = 0;
	size_t i=0;
	bool is_digit = false, is_filter;
	const char * pc_task_name[3] = {
		"IDLE",
		"ipc0",
		"ipc1",
	};
	const char * TASK_TAG = "TASK_Monitor";

	vTaskGetRunTimeStats(task_list_buff);
	//ESP_LOGI(TASK_TAG, "state: %s", task_list_buff);
	pc_line = strtok(task_list_buff, "\n");
	while (pc_line != NULL)
	{
		/* 过滤指定名称 的任务 */
		is_filter = false;
		for(i = 0; i < ARRAY_SIZE(pc_task_name); i++)
		{
			if (strstr(pc_line, pc_task_name[i]) != NULL){
				is_filter = true;
				break;
			}
		}
		if (is_filter == true)
		{
			pc_line = strtok(NULL, "\n");
			continue;
		}
		/* 找出占用 >% x 的任务 */
		pc_line_end = pc_line + strlen(pc_line);
		pc_lattice = pc_line_end;
		while((*pc_lattice != ' ' && *pc_lattice != '\t') && (pc_lattice - pc_line > 0)) {
			pc_lattice--;
		}
		word_len = pc_line_end - pc_lattice;
		is_digit = false;
		
		for(i = 0; i < word_len; i++)
		{
			if ( isdigit( ((uint8_t*)pc_lattice)[i] ) )
			{
				task_occupied = strtol(pc_lattice + i, NULL, 10);
				is_digit = true;
				break;
			}
		}
		if (is_digit && task_occupied > 5)
			ESP_LOGI(TASK_TAG, "%s", pc_line);
		
		pc_line = strtok(NULL, "\n");
	}
#endif
#if 0
	print_task_minstack("MainTask");
	print_task_minstack("BleTask");
	print_task_minstack("WifiTask");
	print_task_minstack("UpdataTask");
	print_task_minstack("tcpClientTask");
	print_task_minstack("tcpClientRcvTas");
	print_task_minstack("FgRecvTask");
#endif
}

void sys_monitor_task(void *pvParameters)
{
	uint8_t delay_cnt = 0;
	while(1){
		ESP_LOGW(TAG, "heap free=%ld, min=%ld",esp_get_free_heap_size(),esp_get_minimum_free_heap_size() );
		ESP_LOGW(TAG, "inter heap free=%ld, min=%d",esp_get_free_internal_heap_size(), heap_caps_get_minimum_free_size( MALLOC_CAP_8BIT | MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL ) );

		if( ++delay_cnt >= 1 ){
			delay_cnt = 0;
			// heap_caps_print_heap_info( MALLOC_CAP_SPIRAM );
			// heap_caps_print_heap_info( MALLOC_CAP_INTERNAL );
			print_task_stack_info();
		}
		
		vTaskDelay(pdMS_TO_TICKS(1000));	//1s节拍
	}
}
#endif

static uint16_t main_heart_cnt = 0;
void main_flash_heart()
{
	if( xTaskGetCurrentTaskHandle() == xTaskGetHandle("MainTask") ){
		main_heart_cnt++;
		// ESP_LOGI(TAG,"main task updata !!!");
	}
}

void tcp_recive_flash_heart()
{
	main_heart_cnt++;
}

uint16_t get_main_heart_cnt()
{
	return main_heart_cnt;
}

/**
 * 	主程序监控程序 
 */
extern uint16_t port_get_wait_poweron();
void main_monitor_task(void *pvParameters)
{
	static uint16_t local_main_heart_cnt = 0;
	static uint16_t	heart_delay_cnt = 0;
	while(1){
		uint16_t temp = get_main_heart_cnt();
		if( local_main_heart_cnt != temp || port_get_wait_poweron() ){
			heart_delay_cnt = 0;
			local_main_heart_cnt = temp;
		}else{
			if( ! is_updating() ){
				heart_delay_cnt++;
			}
		}
		//	5s未运行
		if( heart_delay_cnt >= 5 ){
			//	reset system
			ESP_LOGE(TAG,"sys main stop run \n reset now!!! \n");
			vTaskDelay(pdMS_TO_TICKS(1000));
			esp_restart();
		}	
		// print_task_stack_info();
		vTaskDelay(pdMS_TO_TICKS(1000));	//1s节拍
	}
}

/*
****************************************************************************************************
*    函数名称：
*    功能说明：
*    参   数：
*    返 回 值：
*
****************************************************************************************************
*/
void main_task(void *pvParameters)
{
	EventBits_t bits = 0;

	//uint8_t cnt = 0;
    static uint16_t flag_100ms=0;
	
	_Init();

	//powerON_Pro();
    
	while (1)
	{

#if 1
        bits = xEventGroupWaitBits(main_eventGroup, BIT0 | BIT1, pdTRUE, pdFALSE, portMAX_DELAY);

		xSemaphoreTake(mutex_mainTask, portMAX_DELAY);
		
		if (bits & BIT0)
		{
			run_per10ms();			
		}
		if (bits & BIT1)
		{
			run_per100ms_2();
			run_per100ms();	
			main_flash_heart();
			// ai_check_plug_in(1);   // called per sec
			if(flag_100ms==0) 
			{
				
				for(int i=0;i<FLASK_STATES_GET_COUNT;i++)
				{
					vTaskDelay(pdMS_TO_TICKS(10000));
					if(flask_state_get_flag[i]==SPIFFS_MODEL_EMPTY)
					{
						wifi_get_package(i);    
					}
					//if(flask_state_get_flag[FLASK_OPTI_MODEL]==SPIFFS_MODEL_SAVED){
						 
					//	break;
					//}
					 
				}
			}
			
			flag_100ms++;
		    //if (  flag_100ms>=100 && flask_state_get_flag[FLASK_OPTI_MODEL]==SPIFFS_MODEL_SAVED )
			if (  flag_100ms>=100   )
			{
				flag_100ms = 1;	
				if(	lll_tensor_run(PPO_CASE)==ESP_FAIL){
					 
					break;

				}
				 	
				//ai_check_plug_in(1);   // called per sec
				// ESP_LOGW(TAG, "heap free=%ld, min=%ld",esp_get_free_heap_size(),esp_get_minimum_free_heap_size() );
				//ESP_LOGW(TAG, "inter heap free=%ld, min=%d",esp_get_free_internal_heap_size(), heap_caps_get_minimum_free_size( MALLOC_CAP_8BIT | MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL ) );
				// heap_caps_print_heap_info( MALLOC_CAP_SPIRAM );
				// heap_caps_print_heap_info( MALLOC_CAP_INTERNAL );
		    }
			vTaskDelay(pdMS_TO_TICKS(100));
		#if ((_TYPE_of(VER_HARDWARE) == _TYPE_(_OUTLET)))
			switch_sync_sta();
		#endif
		}

		xSemaphoreGive(mutex_mainTask);

		// vTaskDelay(pdMS_TO_TICKS(1));

#else

		xSemaphoreTake(mutex_mainTask, portMAX_DELAY);

		/* 以下方式 倒计时显示 会出现不连续，原因不明 */
		run_per10ms();
		if (++cnt >= 10)
		{
			cnt = 0;
			run_per100ms_2();
			run_per100ms();
		}
		vTaskDelay(pdMS_TO_TICKS(10));

		xSemaphoreGive(mutex_mainTask);


#endif

	}

	vTaskDelete(NULL);
}
 


void plant_env_make_task(void *pvParameters)
{

	vTaskDelay(3000 / portTICK_PERIOD_MS);
	 
	while(1)
	{  
		//for(int i=0;i<NUM_FLASK_TASK;i++)
        for(int i=0;i<FLASK_PUT_COUNT;i++)
        {
            vTaskDelay(pdMS_TO_TICKS(10000));
            if(flask_state_put_flag[i]==SPIFFS_MODEL_EMPTY)
                wifi_put_package(i);    

        }
		if(plant_env_step() ==0){
			vTaskDelay(pdMS_TO_TICKS(1000));
			break;
		}
		hvac_agent();
		vTaskDelay(pdMS_TO_TICKS(10000));
	} 
}

/*
****************************************************************************************************
*    函数名称：
*    功能说明：
*    参   数：
*    返 回 值：
*
****************************************************************************************************
*/
void app_main(void)
{
	ESP_LOGI(TAG, "--------------------------------------------------------------------------------");
	ESP_LOGI(TAG, "\t\t%s()", __func__);
	ESP_LOGI(TAG, "--------------------------------------------------------------------------------\r\n\r\n");
 
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
	 
	//--------------------------------------------------------------
	// const esp_partition_t *configured = esp_ota_get_boot_partition();
	// const esp_partition_t *running = esp_ota_get_running_partition();

	// if (configured != running)
	// {
	// 	ESP_LOGW(TAG, "Configured OTA boot partition at offset 0x%08x, but running from offset 0x%08x", configured->address, running->address);
	// 	ESP_LOGW(TAG, "(This can happen if either the OTA boot data or preferred boot image become corrupted somehow.)");
	// }
	// ESP_LOGI(TAG, "Running partition type %d subtype %d (offset 0x%08x)\n", running->type, running->subtype, running->address);

      
    spiffs_init();

	//--------------------------------------------------------------
    //ble_creat_event();
    //wifi_creat_event();
	//wifi_init_apsta();  
	wifi_init_sta();
//updata_creat_event();
 	main_creat_objs();
//tcp_client_creat_objs();
	//     start_ota();

//comm_creat_objs();
    //////////////////
	// xTaskCreate(ble_task, 			TASK_NAME_BLE, 		TASK_STACK_BLE, 	NULL, TASK_PRIO_BLE, 	NULL);
	//xTaskCreate(wifi_task, 			TASK_NAME_WIFI, 	TASK_STACK_WIFI, 	NULL, TASK_PRIO_WIFI, 	NULL);
//xTaskCreate(updata_task,		TASK_NAME_UPDATA,	TASK_STACK_UPDATA, 	NULL, TASK_PRIO_UPDATA, NULL);
xTaskCreate(main_task, 			TASK_NAME_MAIN, 	TASK_STACK_MAIN, 	NULL, TASK_PRIO_MAIN, 	NULL);
	// xTaskCreate(main_monitor_task, 	TASK_NAME_MOMITOR, 	TASK_STACK_MOMITOR, NULL, TASK_PRIO_MOMITOR, NULL);

//tcp_client_creat_task();
//tcp_client_creat_rcv_task();
//xTaskCreate(chgup_load_task, 	TASK_NAME_CHGUPLOAD, TASK_STACK_CHGUPLOAD, NULL, TASK_PRIO_CHGUPLOAD, NULL);

	 xTaskCreate(plant_env_make_task,TASK_NAME_COMMDECODE, TASK_STACK_COMMDECODE, NULL, TASK_PRIO_COMMDECODE, NULL);	// 通信数据解析

	// xTaskCreate(comm_decode_task, 	TASK_NAME_COMMDECODE, TASK_STACK_COMMDECODE, NULL, TASK_PRIO_COMMDECODE, NULL);	// 通信数据解析
	// xTaskCreate(cmd_decode_task, 	TASK_NAME_CMDDECODE, TASK_STACK_CMDDECODE, NULL, TASK_PRIO_CMDDECODE, NULL);	// 通信数据包解析

#ifdef _SYS_TASK_RUN_INFO_OUT
	xTaskCreate(sys_monitor_task, 	TASK_NAME_SYS_MONITOR, 	TASK_STACK_SYS_MONITOR, NULL, TASK_PRIO_SYS_MONITOR, NULL);
#endif

	vTaskDelete(NULL);

}
