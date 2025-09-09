#include <stdio.h>
#include <vector>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_http_client.h"
#include "config_mqtt.h"
#include "ota_model_updater.h"
#include "esp32_model_loader.h"
#include "ml_pid.h"
#include "nn.h" 
 


static const char *TAG = "OTA HVAC";
// ======= WiFi 與 OTA 配置 ======= 
#if 0
    const char* check_url = "http://192.168.68.237:5000/api/check-update/device001/1.0.0";
    const char* download_url = "http://192.168.68.237:5000/api/download-update";
    //#define OTA_URL "http://192.168.68.237:5000/api/bin-update"
#else
    const char* base_url ="192.168.0.57:5000";
    const char* check_url = "http://192.168.0.57:5000/api/check-update/device001/1.0.0";
    const char* download_url = "http://192.168.0.57:5000/api/download-update";
    const char* download_bin_url = "http://192.168.0.57:5000/api/bin-update";
    //#define OTA_BIN_UPDATE_URL "http://192.168.0.57:5000/api/bin-update"
#endif
//#define LOCAL_MODEL_FILE "/spiffs/ppo_model.bin"
extern const char * model_path ;

  
// ---------------- NN Placeholder ----------------
// 权重向量示例
std::vector<float> W1, W2, b1, b2, Vw;
float Vb = 0.0f;
extern std::vector<float> health_result;

 #define H1          32
#define H2          4
 

void parse_model_weights(uint8_t *buffer, size_t size) {
    ESP_LOGI(TAG, "Parsing model weights... (%d bytes)", size);

    // 将 buffer 强制转换为 float*
    float* ptr = reinterpret_cast<float*>(buffer);
    size_t offset = 0;

    // 清空之前的 vector 并填充新数据
    W1.assign(ptr + offset, ptr + offset + H1 * INPUT_DIM);
    offset += H1 * INPUT_DIM;

    b1.assign(ptr + offset, ptr + offset + H1);
    offset += H1;

    W2.assign(ptr + offset, ptr + offset + H2 * H1);
    offset += H2 * H1;

    b2.assign(ptr + offset, ptr + offset + H2);
    offset += H2;

    Vw.assign(ptr + offset, ptr + offset + H2);
    offset += H2;

    Vb = *(ptr + offset);
    offset += 1;

    ESP_LOGI(TAG, "Model weights parsed successfully. Total floats = %d", offset);
} 
 


esp_err_t _http_down_load_event_handler0(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            if (evt->user_data) {
                FILE *file = (FILE *)evt->user_data;
                fwrite(evt->data, 1, evt->data_len, file);
            }
            break;
        default:
            break;
    }
    return ESP_OK;
}
// ---------------- OTA Callback ----------------
esp_err_t _http_down_load_event_handler(esp_http_client_event_t *evt) {
    static FILE *f_model = NULL;
    switch(evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            //if (f_model) fwrite(evt->data, 1, evt->data_len, f_model);
            if (evt->user_data) {
                f_model = (FILE *)evt->user_data;
                fwrite(evt->data, 1, evt->data_len, f_model);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            if (f_model) {
                fclose(f_model);
                f_model = NULL;
                ESP_LOGI(TAG, "OTA file saved to %s", model_path);
            }
            break;
        case HTTP_EVENT_DISCONNECTED:
        case HTTP_EVENT_ERROR:
            if (f_model) { fclose(f_model); f_model = NULL; }
            ESP_LOGE(TAG, "HTTP error or disconnected");
            break;
        default:
            break;
    }
    return ESP_OK;
}
 


// ======= PPO 模型 =======
ESP32EWCModel ewcppoModel;
bool init_exporter_flag=false;

void http_post_task(void *pvParameters) {
       
        const char* base_url ="192.168.0.57:5000";
        char url[128];

       sprintf(url, "http://%s/init_exporter", base_url);

        esp_http_client_config_t config = {
            //.url = "http://<SERVER_IP>:5000/init_exporter",   // Flask 服务器地址
            .url = url,    
        };

        esp_http_client_handle_t client = esp_http_client_init(&config);

        // 发送 JSON 数据
        //const char *post_data = "{\"model_path\": \"./saved_models/my_model.keras\"}";
         
 
       const char *post_data = "{\"model_path\": \"./saved_models/ppo_policy_actor\"}";
        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, post_data, strlen(post_data));

        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Status = %d, content_length = %lld",
                    esp_http_client_get_status_code(client),
                    esp_http_client_get_content_length(client));

                    
            // 获取响应头的大小
     
            
            
            char buffer[512];
            int data_len = esp_http_client_read_response(client, buffer, sizeof(buffer) - 1);
            if (data_len >= 0) {
                buffer[data_len] = 0;
                ESP_LOGI(TAG, "Response = %s", buffer);
            }
        } else {
            ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
        }






        esp_http_client_cleanup(client);
        vTaskDelete(NULL);
}


  
extern "C" void hvac_agent(void) {
     
    vTaskDelay(pdMS_TO_TICKS(5000)); // 等 WiFi 連線

   
    // ======= 從 SPIFFS 加載模型 =======
    if (ewcppoModel.loadModelFromSPIFFS("/spiffs/model.tflite")) {
        ESP_LOGI(TAG, "Model loaded successfully");
    } else {
        ESP_LOGE(TAG, "Failed to load model");
    } 
    std::vector<float> observation(5, 0.0f); // 5维状态
    // 填充实际数据
    observation[0] = bp_pid_th.t_feed;
    observation[1] = bp_pid_th.h_feed;
    observation[2] = bp_pid_th.l_feed;
    observation[3] = bp_pid_th.c_feed;
    observation[4] = 0;
    // ======= 推理 =======
    std::vector<float> action_probs = ewcppoModel.predict(observation);
    //std::vector<float> value = ewcppoModel.predictValue(observation);
    float value = ewcppoModel.predictValue(observation); 
    // 假设优势函数 = Q(s, a) - V(s)，我们计算优势。
    std::vector<float> advantages(action_probs.size(), 0.0f);
    for (size_t i = 0; i < action_probs.size(); ++i) {
        advantages[i] = action_probs[i] - value ;  // 这里是一个简单的示例，按需调整
    }

    printf("Action probs: ");
    for (auto v : action_probs) printf("%.3f ", v);
    printf("\n");

    // ======= 持续学习 (EWC) =======
    std::vector<float> newExperience = observation;  // 假设新经验就是当前观测值

    static PPOModel ppoModel;  // 只初始化一次，之后重复使用
    std::vector<float> old_probs = action_probs;  // 示例的旧概率

     
    // 更新旧动作概率
    static std::vector<float> old_action_probs = {0.0f, 0.0f, 0.0f};  

    std::vector<float> grads;
    ppoModel.calculateLossAndGradients(newExperience, old_probs, advantages, health_result, old_action_probs, grads);


    ewcppoModel.continualLearningEWC(grads);

    vTaskDelay(pdMS_TO_TICKS(5000)); // 每 5 秒推理一次
    // 更新旧动作概率以供下一轮使用
    old_action_probs = action_probs;

    // 可选：打印梯度
    ESP_LOGI(TAG, "Gradients: ");
    for (auto& grad : grads) {
        printf("%.3f ", grad);
    }
    printf("\n"); 
 
 
}
 

void download_model_ota_task(void *param){
     // ======= OTA 更新模型 =======otaUpdater.downloadModel()

      
     ModelOTAUpdater otaUpdater(check_url, download_url, "/spiffs/model.tflite","0.0.1");

    if (otaUpdater.checkUpdate()) {
        ESP_LOGI(TAG, "Model OTA update completed!");
         
    } else {
        ESP_LOGW(TAG, "Model OTA update failed or no update available");
        
    }
return  ;
}

static void download_model_task(void *param) {
    ESP_LOGI("MEM", "Before OTA: Heap free=%d, internal free=%d, largest=%d",
             (int)esp_get_free_heap_size(),
             (int)esp_get_free_internal_heap_size(),
             (int)heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));

    // 删除旧模型 
    remove(model_path);
    static FILE *f_model =fopen(model_path, "wb");
    if (!f_model) {
        ESP_LOGE("OTA", "Failed to open file for OTA");
        vTaskDelete(NULL);
        return;
    }
 
    esp_http_client_config_t config = {0}; // 初始化为全零

    config.url = download_bin_url;
    config.timeout_ms = 10000;
    config.user_data = f_model;
    config.event_handler = _http_down_load_event_handler;
    esp_http_client_handle_t client = esp_http_client_init(&config);
    //esp_http_client_set_buffer_size(client, 4096); 
     
    if (!client) {
        ESP_LOGE("OTA", "esp_http_client_init failed");
        fclose(f_model);
        f_model = NULL;
        vTaskDelete(NULL);
    }

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI("OTA", "Model downloaded successfully (%d bytes)",
                 (int)esp_http_client_get_content_length(client));
    } else {
        ESP_LOGE("OTA", "HTTP request failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    fclose(f_model);
    f_model = NULL;

    ESP_LOGI("MEM", "After OTA: Heap free=%d, internal free=%d",
             (int)esp_get_free_heap_size(),
             (int)esp_get_free_internal_heap_size());

    vTaskDelete(NULL);
}
   
 
extern "C" void wifi_ota_ppo_package(int type ) {
    if(type==0)
    {
        xTaskCreatePinnedToCore(
        http_post_task,  //download_model_ota_task,download_model_task
        "http_post_task",
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   
    }
    if(type==1)
    {
        xTaskCreatePinnedToCore(
        download_model_task,  //download_model_ota_task,download_model_task
        "download_model_task",
        8 * 1024,       // 栈大小，可根据实际增加
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   
    }
    if(type==2)
    {
        xTaskCreatePinnedToCore(
        download_model_ota_task, 
        "download_model_ota_task",
        8 * 1024,     
        NULL,
        tskIDLE_PRIORITY + 1,
        NULL,
        0);   
    }
    return   ;
}
extern "C" pid_run_output_st nn_ppo_infer() {

    // 尝试下载 OTA model_version == 0 &&
    //if( !ppoModel.downloadModel(bin_url, model_path)) {
    //   ESP_LOGW("downloadModel", "Model will use existing model if available");
    //}

    //const char* version = "1.0.0";
    //ModelOTAUpdater otaUpdater( check_url, download_url, "/spiffs/model.bin",version  );
    static pid_run_output_st 		output_speed;
    // OTA 拉取最新 PPO 参数
    //if (otaUpdater.checkUpdate()) {
    //    printf("✅ Model updated from server\n");
    //}

    // 初始化 PPO 推理网络
    NN  ppoNN(5, 32, 4);  // state_dim=5, hidden=32, action_dim=4
    if (ppoNN.loadWeights("/spiffs/model.bin")) {
        printf("✅ Weights loaded into PPO NN\n");
    }
 
    std::vector<float> state = { bp_pid_th.t_feed, bp_pid_th.h_feed, bp_pid_th.l_feed, bp_pid_th.c_feed, 1.0};
    auto action_probs = ppoNN.forwardActor(state);
    float value = ppoNN.forwardCritic(state);
    if(value>0.5)
    {
        output_speed.speed[0] = action_probs[0]>0.5?1:0;
        output_speed.speed[1] = action_probs[1]>0.5?1:0;
        output_speed.speed[2] = action_probs[2]>0.5?1:0;
        output_speed.speed[3] = action_probs[3]>0.5?1:0;
    }
    printf("Action probs: ");
    for (auto p : action_probs) printf("%.3f ", p);
    printf("\nCritic value: %.3f\n", value);
    return output_speed;
}

 