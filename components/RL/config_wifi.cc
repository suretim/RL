
#include "esp_wifi.h"   
#include "esp_event.h" 
//#include <esp_netif.h>
#include <esp_log.h>
#include <string.h>  
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h> 
 
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "config_wifi.h"


 
 // Wi-Fi事件处理器
static TimerHandle_t reconnect_timer = NULL;
static int reconnect_attempts = 0;
#define MAX_RECONNECT_ATTEMPTS 10

// 全局变量 
static volatile bool wifi_connected = false;
EventGroupHandle_t wifi_event_group;
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
bool got_ip = false;
static const char *TAG = "wifi_config";

// ---------------- 等待 WiFi 连接 ----------------
void wait_for_wifi_connection() {
    ESP_LOGI(TAG, "Waiting for WiFi...");
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdTRUE,
                                           pdMS_TO_TICKS(15000));
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected.");
    } else {
        ESP_LOGE(TAG, "WiFi connection timed out.");
    }
}


static void wifi_ap_event_handler(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (id == WIFI_EVENT_AP_START) ESP_LOGI(TAG, "AP启动成功");
    else if (id == WIFI_EVENT_AP_STACONNECTED) ESP_LOGI(TAG, "设备接入");
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) ESP_LOGI(TAG, "设备断开");
}
void wifi_scan_networks(void) {
    uint16_t number = 10;
    wifi_ap_record_t ap_info[10];
    uint16_t ap_count = 0;
    memset(ap_info, 0, sizeof(ap_info));

    ESP_ERROR_CHECK(esp_wifi_scan_start(NULL, true));
    ESP_ERROR_CHECK(esp_wifi_scan_get_ap_records(&number, ap_info));
    ESP_ERROR_CHECK(esp_wifi_scan_get_ap_num(&ap_count));

    ESP_LOGI(TAG, "Found %d WiFi networks:", ap_count);
    for (int i = 0; i < ap_count; i++) {
        ESP_LOGI(TAG, "SSID: %s, RSSI: %d", ap_info[i].ssid, ap_info[i].rssi);
    }
}

// 重连回调函数
void reconnect_callback(void* arg) {
    ESP_LOGI(TAG, "Attempting to reconnect...");
    esp_wifi_connect();
}
 void wifi_sta_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    static int reconnect_delay = 3000; // 重连延迟，使用指数退避
    
    if (event_base == WIFI_EVENT) {
        if (event_id == WIFI_EVENT_STA_START) {
            ESP_LOGI(TAG, "WiFi station started");
            esp_err_t ret = esp_wifi_connect();
                if (ret != ESP_OK) {
                    if (ret == ESP_ERR_WIFI_CONN) {
                        ESP_LOGW(TAG, "WiFi is already connecting/connected (0x%x)", ret);
                        // 这不是一个致命错误，可以忽略或者记录日志
                    } else {
                        ESP_LOGE(TAG, "WiFi connect failed: %s", esp_err_to_name(ret));
                        // 其他错误可能需要处理
                    }
                } 
            
        } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
            wifi_event_sta_disconnected_t *disconn = (wifi_event_sta_disconnected_t *)event_data;
    
            // 显示详细的断开信息
           ESP_LOGW(TAG, "STA Disconnected. Reason: %d, RSSI: %d" , disconn->reason,     disconn->rssi);
            
            got_ip = false;
            
            // 使用定时器延迟重连，避免阻塞
            esp_timer_handle_t timer;
            esp_timer_create_args_t timer_args = {
                .callback = reconnect_callback,
                .arg = NULL,
                .dispatch_method = ESP_TIMER_TASK,
                .name = "wifi_reconnect"
            };
            esp_timer_create(&timer_args, &timer);
            esp_timer_start_once(timer, reconnect_delay * 1000);
        }
        
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "STA Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        
        // 重置重连延迟
        reconnect_delay = 1000;
        
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
        got_ip = true;
    }
}


static void reconnect_timer_callback(TimerHandle_t xTimer) {
    ESP_LOGI(TAG, "Attempting to connect...");
    esp_err_t ret = esp_wifi_connect();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Connect attempt returned: %s", esp_err_to_name(ret));
    }
}

static void wifi_apsta_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ESP_LOGI(TAG, "APSTA WIFI_EVENT_STA_START");
                if (reconnect_timer == NULL) {
                    reconnect_timer = xTimerCreate(
                        "wifi_reconnect", 
                        pdMS_TO_TICKS(2000), 
                        pdFALSE, 
                        NULL, 
                        reconnect_timer_callback
                    );
                }
                xTimerStart(reconnect_timer, 0);
                break;
                
            case WIFI_EVENT_STA_DISCONNECTED:
            {
                wifi_event_sta_disconnected_t *disconnected = 
                    (wifi_event_sta_disconnected_t *)event_data;
                ESP_LOGW(TAG, "Disconnected from AP (reason: %d), attempt %d/%d", 
                        disconnected->reason, ++reconnect_attempts, MAX_RECONNECT_ATTEMPTS);
                
                if (reconnect_attempts >= MAX_RECONNECT_ATTEMPTS) {
                    ESP_LOGE(TAG, "Max reconnect attempts reached");
                    // 触发恢复机制
                    return;
                }
                
                xTimerStart(reconnect_timer, 0);
                break;
            }
            
            case WIFI_EVENT_AP_STACONNECTED:
                ESP_LOGI(TAG, "Device connected to AP");
                break;
                
            case WIFI_EVENT_AP_STADISCONNECTED:
                ESP_LOGI(TAG, "Device disconnected from AP");
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        reconnect_attempts = 0;  // 重置计数器
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
} 




void wifi_init_apsta(void) {
    ESP_LOGI(TAG, "Initializing Wi-Fi AP+STA mode...");
    
    // 创建事件组
    s_wifi_event_group = xEventGroupCreate();
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "Failed to create event group");
        return;
    }

    // 创建默认网络接口
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    // WiFi初始化配置
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_apsta_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_apsta_event_handler, NULL));

    // STA配置
    wifi_config_t sta_config = {0};
    strlcpy((char *)sta_config.sta.ssid, WIFI_SSID_STA, sizeof(sta_config.sta.ssid));
    strlcpy((char *)sta_config.sta.password, WIFI_PASS_STA, sizeof(sta_config.sta.password));
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    // AP配置
    wifi_config_t ap_config = {0};
    strlcpy((char *)ap_config.ap.ssid, WIFI_SSID_AP, sizeof(ap_config.ap.ssid));
    strlcpy((char *)ap_config.ap.password, WIFI_PASS_AP, sizeof(ap_config.ap.password));
    ap_config.ap.ssid_len = strlen(WIFI_SSID_AP);
    ap_config.ap.max_connection = 4;
    ap_config.ap.authmode = (strlen(WIFI_PASS_AP) == 0) ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA_WPA2_PSK;

    // 设置WiFi模式
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    
    // 启动WiFi
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi AP+STA Initialized Successfully");
    ESP_LOGI(TAG, "AP SSID: %s", WIFI_SSID_AP);
    ESP_LOGI(TAG, "STA connecting to: %s", WIFI_SSID_STA);
}

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- WiFi ----------------
void wifi_init_sta(void) {
    ESP_LOGI(TAG, "Initializing WiFi...");
    got_ip = false;
    wifi_event_group = xEventGroupCreate();

    // Create default WiFi station
    esp_netif_create_default_wifi_sta(); 
    // WiFi init
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_sta_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_sta_event_handler, NULL));

    wifi_config_t sta_config = { 0 };
    strcpy((char*)sta_config.sta.ssid, WIFI_SSID_STA);
    strcpy((char*)sta_config.sta.password, WIFI_PASS_STA);
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
 
    ESP_ERROR_CHECK(esp_wifi_start()); 
    wifi_scan_networks(); // 先扫描网络
    
    // 正确的高效阻塞等待
    EventBits_t bits = xEventGroupWaitBits(
        wifi_event_group,            // 事件组句柄
        WIFI_CONNECTED_BIT,          // 等待的位
        pdFALSE,                     // 成功等待后不清除位
        pdTRUE,                      // 等待所有位都置位
        pdMS_TO_TICKS(10000)         // 等待10秒超时
    );

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to AP  " );
    } else {
        ESP_LOGE(TAG, "Failed to connect within timeout");
    }
}


void wifi_init_ap(void) 
{
    //esp_netif_init();
    //esp_event_loop_create_default();

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t ap_config = {};

    strcpy((char*)ap_config.ap.ssid,WIFI_SSID_AP);
    strcpy((char*)ap_config.ap.password,WIFI_PASS_AP);
    ap_config.ap.max_connection = 4;
    ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_ap_event_handler, NULL);

    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_config);
    
    //esp_wifi_set_ps(WIFI_PS_NONE);  // 禁用省电模式，避免频繁协商
    esp_wifi_set_max_tx_power(80);  // 设置最大发射功率（单位0.25dBm）
    //esp_wifi_set_max_tx_power(70);
    //esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G);   //Wi-Fi 4 (802.11n)
    //esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);  // 指定信道6减少干扰

    esp_wifi_start();

    esp_netif_ip_info_t ip_info;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_AP_DEF"), &ip_info);
    ESP_LOGI(TAG, "got ip:" IPSTR "\n", IP2STR(&ip_info.ip));
}


 

// 可选：等待连接函数（非阻塞方式）
bool wifi_wait_for_connection(TickType_t timeout_ticks) {
    if (wifi_connected) {
        return true;
    }
    
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                         WIFI_CONNECTED_BIT,
                                         pdFALSE,
                                         pdFALSE,
                                         timeout_ticks);
    return (bits & WIFI_CONNECTED_BIT) != 0;
}

// 如果需要重新配置WiFi（而不是完全清理）
esp_err_t wifi_reconnect_sta(const char* ssid, const char* password) {
    wifi_config_t sta_config = {0};
    strlcpy((char *)sta_config.sta.ssid, ssid, sizeof(sta_config.sta.ssid));
    strlcpy((char *)sta_config.sta.password, password, sizeof(sta_config.sta.password));
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    // 断开当前连接
    esp_wifi_disconnect();
    
    // 设置新的配置
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    
    // 重新连接
    return esp_wifi_connect();
}  
#ifdef __cplusplus
}
#endif
 