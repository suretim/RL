#pragma once

//#include "esp_event.h"

#if 1
  //#define BASE_URL  "192.168.0.57" 
  #define BASE_URL  "192.168.30.132" 
  #define MQTT_BROKER_URI "mqtt://192.168.30.132:1883"   
  #define WIFI_SSID_STA      "1573-2.4G"
  #define WIFI_PASS_STA      "987654321"
  //#define WIFI_SSID_STA       "ESP32-AP"
  //#define WIFI_PASS_STA       "12345678"
  

#else
  #define BASE_URL  "192.168.68.237"
  #define MQTT_BROKER_URI "mqtt://192.168.68.237:1883"     
  #define WIFI_SSID_STA      "JD803"
  #define WIFI_PASS_STA      "18825213948"
#endif
#define BASE_PORT "5000"
#define POLICY_PORT "5001"
#define WIFI_SSID_AP       "ESP32-AP"
#define WIFI_PASS_AP       "12345678"


void wifi_init(void) ;
void wifi_init_apsta(void);

#ifdef __cplusplus
extern "C" {
#endif
void wifi_init_sta(void);
#ifdef __cplusplus
}
#endif
//void wait_for_wifi_connection() ;
// void wifi_sta_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);
