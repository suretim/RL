#pragma once

//#include "esp_event.h"

#if 1
  #define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"   
  #define WIFI_SSID_STA      "1573"
  #define WIFI_PASS_STA      "987654321"
#else
  #define MQTT_BROKER_URI "mqtt://192.168.68.237:1883"     
  #define WIFI_SSID_STA      "JD803"
  #define WIFI_PASS_STA      "18825213948"
#endif
void wifi_init(void) ;

#ifdef __cplusplus
extern "C" {
#endif
void wifi_init_apsta(void);

#ifdef __cplusplus
}
#endif
void wait_for_wifi_connection() ;
// void wifi_sta_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);
