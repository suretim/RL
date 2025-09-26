#ifndef INFER_ESP32_LSTM_LLL
#define INFER_ESP32_LSTM_LLL
 
esp_err_t lll_tensor_run(void) ;

 void wifi_init_apsta(void);  
 void wifi_init_sta(void);  
void wifi_get_package(int type); 
void wifi_put_package(int type); 

void hvac_agent(void); 
#endif // INFER_ESP32_LSTM_LLL
