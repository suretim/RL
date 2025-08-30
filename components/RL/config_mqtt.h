#pragma once

void init_classifier_from_header(void ); 
void initialize_nvs_robust(void);
 

void  get_mqtt_feature(  float *f_in);
//void mqtt_app_start();
void publish_feature_vector(int label,int type );
void start_mqtt_client(void *pvParameters); 
void lll_tensor_run(void ); 

//void wifi_init_apsta(void);
 