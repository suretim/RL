#pragma once

void init_classifier_from_header(void ); 
void initialize_nvs_robust(void);
 

void  get_mqtt_feature(  float *f_in);
void wifi_init_apsta(void);
void publish_feature_vector(int label,int type );
void start_mqtt_client(void *pvParameters); 

//void wifi_init_apsta(void);
 