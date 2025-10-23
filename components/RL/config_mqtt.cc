#include "mqtt_client.h"
#include "esp_log.h" 
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <stdlib.h>  
#include "cJSON.h"
#include "esp_wifi.h"   
#include <mbedtls/sha1.h>
#include "classifier_storage.h"
#include "config_wifi.h"
//#include "esp_system.h" 
//#include "esp_macros.h
#include "model_pb_handler.h"
 
#define DENSE_IN_FEATURES 64
#define DENSE_OUT_CLASSES 3 
#define TAG "MQTT"


#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_CLIENT_ID_PREFIX "mqttx_" 
#define MQTT_TOPIC_PUB "grpc_sub/weights"

#define MQTT_TOPIC_SUB "federated_model/parameters"
#define WEIGHT_FISH_SUB "ewc/weight_fisher"
#define FISH_SHAP_SUB  "ewc/layer_shapes"

#define MQTT_KEEPALIVE_SECONDS 120 

std::vector<uint8_t> ewc_buffer;  // 接收 buffer
bool ewc_ready=false;

// ===================== worker 消息结构 =====================
typedef struct {
    size_t len;
    uint8_t *data;
} mqtt_worker_msg_t;

static QueueHandle_t mqtt_msg_queue = NULL;

static esp_mqtt_client_handle_t mqtt_client = NULL; 
float  f_out[DENSE_IN_FEATURES ]= {
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1
};
//char client_id[64]=MQTT_CLIENT_ID_PREFIX;
void  get_mqtt_feature(  float *f_in)
{
    for (int i=0;i<DENSE_IN_FEATURES;i++)
    {
        f_out[i]=f_in[i];
    }
   return  ;
}
uint32_t get_client_id()
{
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
      unsigned char hash[20]; // SHA-1 produces a 20-byte hash
    mbedtls_sha1(mac, 6, hash);
    uint32_t hashNumber = 0;
  for(int i=0; i<6; i++) {
    hashNumber = (hashNumber << 4) | hash[i];
  }
  ESP_LOGI(TAG, "hashNumber =%d",(int) hashNumber);
    return hashNumber;
}

#ifdef __cplusplus
extern "C" {
#endif
// 将 float 数组格式化为 JSON 并上传
void publish_feature_vector(int label,int type ) {
    std::stringstream ss;
    uint32_t  client_id=get_client_id();

    //int label=1;
    //int client_id=1;
    if(type==1)
    {
       ss << "{\"client_request\":"; 
       ss << type; 
       ss << ","; 
       ss << "\"client_id\":";     
            ss << client_id ;

        ss << "}";
    }
    else{
        ss << "{\"fea_weights\":[";
        for (int i = 0; i < DENSE_IN_FEATURES; ++i) {
            ss << f_out[i];
            if (i != DENSE_IN_FEATURES - 1) ss << ",";
        }
        ss << "],";

        ss << "\"fea_labels\":[";     
            ss << label; 

        ss << "],";

        ss << "\"client_request\":";     
            ss << type; 

        ss << ",";
        
        ss << "\"client_id\":";     
            ss << client_id ;

        ss << "}";
    }
    std::string payload = ss.str();

    int msg_id = esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC_PUB, payload.c_str(), payload.length(), 1, 0);
      
    if (msg_id != -1) {
        ESP_LOGI("MQTT", "%s %s, msg_id=%d",MQTT_TOPIC_PUB,payload.c_str(), msg_id);
    } else {
        ESP_LOGE("MQTT", "Failed to publish feature vector");
    }
    return;
}

#ifdef __cplusplus
}
#endif
 
#if 0
int test_protobuf() {
    // 模拟收到 protobuf 编码的数据（你实际中来自 MQTT）
    // 建议使用 Python 端生成真实数据更方便

    // 这里是手动写的一个 buffer 示例
    uint8_t encoded_data[] = {
        0x08, 0x00,                   // param_type = 0 (WEIGHTS)
        0x15, 0x00, 0x00, 0x20, 0x41, // float 10.0
        0x15, 0x00, 0x00, 0x40, 0x41, // float 12.0
        0x15, 0x00, 0x00, 0x48, 0x41, // float 12.5
        0x18, 0x05                    // client_id = 5
    };

    ParsedModelParams params;
    if(decode_model_params( (uint8_t *)encoded_data, sizeof(encoded_data) , &params))
    //if (decode_model_params(encoded_data, sizeof(encoded_data), &params))
    {
        printf("Decoded ParamType: %d\n", params.param_type);
        printf("Client ID: %d\n",(int) params.client_id);
        printf("Values (%zu):\n", params.value_count);
        for (size_t i = 0; i < params.value_count; ++i) {
            printf("  [%zu] = %f\n", i, params.values[i]);
        }
    } else {
        printf("Failed to decode model parameters.\n");
    }

    return 0;
}


// #define MAX_PARAM_LENGTH 1024  // 可按需要扩大

// /* Enum definitions */
// typedef enum _ParamType {
//     ParamType_UNKNOWN = 0,
//     ParamType_ENCODER_WEIGHT = 1,
//     ParamType_ENCODER_BIAS = 2,
//     ParamType_CLASSIFIER_WEIGHT = 3,
//     ParamType_CLASSIFIER_BIAS = 4,
//     ParamType_FULL_MODEL = 5
// } ParamType;
 
// /* Struct definitions */
// typedef struct _ModelParams {
//     ParamType param_type;
//     float * values;
//     int32_t client_id;
// } ModelParams;
// typedef struct {
//     ParamType param_type;
//     float values[MAX_PARAM_LENGTH];
//     size_t value_count;
//     int32_t client_id;
// } ParsedModelParams;



#define MAX_BIAS_SIZE 3  // 根据你的模型类别数调整

// 全局偏置数组
float classifier_bias[MAX_BIAS_SIZE];
float dense_weights[DENSE_IN_FEATURES][DENSE_OUT_CLASSES];
// 处理 classifier bias 函数
void handle_classifier_bias(const float *values, size_t len) {
    if (len > MAX_BIAS_SIZE) {
        ESP_LOGE(TAG, "Bias length %d exceeds max size %d", len, MAX_BIAS_SIZE);
        return;
    }

    ESP_LOGI(TAG, "Updating classifier bias with length %d", len);

    for (size_t i = 0; i < len; i++) {
        classifier_bias[i] = values[i];
        ESP_LOGD(TAG, "bias[%d] = %f", (int)i, values[i]);
    }
} 

void handle_classifier_weight(const float *values, size_t len, const int32_t *shape, size_t shape_len) {
    if (shape_len != 2) {
        ESP_LOGE(TAG, "Invalid shape");
        return;
    }

    int rows = shape[0]; // 例如 64
    int cols = shape[1]; // 例如 3

    if (rows * cols != len) {
        ESP_LOGE(TAG, "Shape mismatch with values_count");
        return;
    }

    ESP_LOGI(TAG, "Updating classifier weights: %d x %d", rows, cols);
    
    // 在此将 values 保存到你的内存结构中
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float w = values[i * cols + j];
            // 假设你有全局数组 float dense_weights[64][3];
            dense_weights[i][j] = w;
        }
    }
}

 
#endif

 
 
std::vector<std::vector<float>> trainable_layers;
std::vector<std::vector<float>> fisher_layers;
std::vector<std::vector<int>> layer_shapes;
 
std::vector<std::vector<int>> parse_layer_shapes(const std::string& json_str)
{
    std::vector<std::vector<int>> shapes;

    printf("Raw JSON string: %s\n", json_str.c_str());
    printf("JSON string length: %d\n", json_str.length());

    cJSON *root = cJSON_Parse(json_str.c_str());
    if (!root) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            printf("Error before: %s\n", error_ptr);
        }
        printf("Failed to parse_layer_shapes JSON. String: %s\n", json_str.c_str());
        return shapes;
    }

     

    int array_size = cJSON_GetArraySize(root);
    for (int i = 0; i < array_size; i++) {
        cJSON *layer = cJSON_GetArrayItem(root, i);
        std::vector<int> shape;
        int dim_size = cJSON_GetArraySize(layer);
        for (int j = 0; j < dim_size; j++) {
            cJSON *dim = cJSON_GetArrayItem(layer, j);
            shape.push_back(dim->valueint);
        }
        shapes.push_back(shape);
    }

    cJSON_Delete(root);

    return shapes;   
}

static std::vector<std::string> recv_chunks;
static int expected_total = 0;


#define MAX_BUFFER_SIZE (1024 * 60)  // 根據最大模型大小調整

static uint8_t *g_big_buffer = NULL;
static size_t g_received_len = 0;
static int g_total_chunks = -1;


// 將 hex 字串轉成 bytes
static size_t hex_to_bytes(const char *hex_str, uint8_t *out_buf, size_t out_buf_size) {
    size_t len = strlen(hex_str);
    if (len % 2 != 0) return 0;
    size_t out_len = len / 2;
    if (out_len > out_buf_size) return 0;

    for (size_t i = 0; i < out_len; i++) {
        sscanf(hex_str + 2*i, "%2hhx", &out_buf[i]);
    }
    return out_len;
}

void mqtt_data_handler(const char *payload) {
    cJSON *root = cJSON_Parse(payload);
    if (!root) {
        ESP_LOGE(TAG, "JSON parse mqtt_data_handler failed");
        return;
    }

    cJSON *seq_id = cJSON_GetObjectItem(root, "seq_id");
    cJSON *total  = cJSON_GetObjectItem(root, "total");
    cJSON *data   = cJSON_GetObjectItem(root, "data");

    if (!cJSON_IsNumber(seq_id) || !cJSON_IsNumber(total) || !cJSON_IsString(data)) {
        ESP_LOGE(TAG, "Invalid JSON fields");
        cJSON_Delete(root);
        return;
    }

    int seq = seq_id->valueint;
    int total_chunks = total->valueint;
    const char *hex_str = data->valuestring;

    if (g_big_buffer == NULL) {
        g_big_buffer =(uint8_t*) malloc(MAX_BUFFER_SIZE);
        g_received_len = 0;
        g_total_chunks = total_chunks;
        ESP_LOGI(TAG, "Allocated big buffer for %d chunks", total_chunks);
    }

    // decode hex string
    uint8_t tmp_buf[960];  // 一片大小
    size_t chunk_len = hex_to_bytes(hex_str, tmp_buf, sizeof(tmp_buf));
    if (chunk_len == 0) {
        ESP_LOGE(TAG, "Hex decode failed for chunk %d", seq);
        cJSON_Delete(root);
        return;
    }

    // append to big buffer
    memcpy(g_big_buffer + g_received_len, tmp_buf, chunk_len);
    g_received_len += chunk_len;

    ESP_LOGI(TAG, "Received chunk %d/%d, chunk_len=%d, total_received=%d",
             seq, total_chunks, (int)chunk_len, (int)g_received_len);

    if (seq == total_chunks - 1) {
        ESP_LOGI(TAG, "All chunks received! total size=%d bytes", (int)g_received_len);
        // TODO: 在這裡處理完整的 g_big_buffer (例如寫檔或載入模型)
        
 ewc_buffer.assign(g_big_buffer, g_big_buffer + g_received_len);

        ESP_LOGI(TAG, "完整 Fisher buffer 长度=%d", ewc_buffer.size());
ewc_ready=true;
                    // TODO: 这里可以把 buffer 转成 float* 然后传给模型
                  
        free(g_big_buffer);
        g_big_buffer = NULL;
        g_received_len = 0;
        g_total_chunks = -1;
    }

    cJSON_Delete(root);
}

static esp_err_t mqtt_event_handler_cb (esp_mqtt_event_handle_t event) {
    
    switch (event->event_id) {        
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            //test_protobuf();
            esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
            esp_mqtt_client_subscribe(event->client, FISH_SHAP_SUB, 1);
            esp_mqtt_client_subscribe(event->client, WEIGHT_FISH_SUB, 1);
            //publish_feature_vector(0,1); 
            break;
        case MQTT_EVENT_DATA:{
             

            if (event->topic_len == strlen(WEIGHT_FISH_SUB) &&
                strncmp(event->topic, WEIGHT_FISH_SUB, event->topic_len) == 0)
            {
                //ESP_LOGI(TAG, "MQTT data received: %.*s", event->data_len, event->data);
                ESP_LOGI(TAG, "len=%d, preview=%.*s", event->data_len, event->data_len>100?100:event->data_len, event->data);

                mqtt_data_handler(event->data);
                
                break;
            }

             
            //ESP_LOGI("MQTT", "Received topic: %.*s", event->topic_len, event->topic);
            if (event->topic_len == strlen(FISH_SHAP_SUB) &&
                strncmp(event->topic, FISH_SHAP_SUB, event->topic_len) == 0) {
                std::string json_str(event->data, event->data + event->data_len);
                layer_shapes = parse_layer_shapes(json_str);
                break;
            }

            if (event->topic_len == strlen(MQTT_TOPIC_SUB) &&
                strncmp(event->topic, MQTT_TOPIC_SUB, event->topic_len) == 0) {
             
                ParsedModelParams params;
                //ModelParams *msg = (ModelParams*)strndup(event->data, event->data_len);
                if(decode_model_params( (uint8_t *)event->data, event->data_len, &params)) { 
                    // 处理参数类型
                    switch (params.param_type) {
                        case ParamType_CLASSIFIER_WEIGHT:
                            ESP_LOGI(TAG, "Classifier weight received, values: %d", params.value_count);
                            //handle_classifier_weight(params.values, params.values_count, params.shape, params.shape_count);
                            update_classifier_weights_bias(params.values, params.value_count,0);
                            break;
                        case ParamType_CLASSIFIER_BIAS:
                            ESP_LOGI(TAG, "Classifier bias received");
                            //handle_classifier_bias(params.values, params.value_count);
                            update_classifier_weights_bias(params.values, params.value_count,1);
                            
                            break;
                        case ParamType_ENCODER_WEIGHT:  
                                ESP_LOGI(TAG, "Fisher weight received");
                                update_fishermatrix_theta(params.values, params.value_count,0);
                            
                            break;
                        case ParamType_ENCODER_BIAS:
                            ESP_LOGI(TAG, "Fisher bias received");
                            update_fishermatrix_theta(params.values, params.value_count,1);                        
                            
                            break;
                        default:
                            ESP_LOGW(TAG, "Unknown or unsupported param_type: %d", params.param_type);
                            break;
                    } 
                }
                else{
                    //printf("MQTT decode error data_len=%d : data=%s\n", event->data_len, event->data);
                    printf("MQTT decode error data_len=%d \r\n", event->data_len );
                }
            }
            break;
        }
        default:
            break;
    }
    vTaskDelay(1); 
    return ESP_OK;
} 

void handle_mqtt_message(const char *json_str) {
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("Failed to parse handle_mqtt_message JSON\n");
        return;
    }

    // weights
    cJSON *weights_array = cJSON_GetObjectItem(root, "mqtrx_weights");
    cJSON *bias_array = cJSON_GetObjectItem(root, "mqtrx_bias");

    if (!weights_array || !bias_array) {
        printf("Missing fields in JSON\n");
        cJSON_Delete(root);
        return;
    }

    int weight_len = cJSON_GetArraySize(weights_array);
    int bias_len = cJSON_GetArraySize(bias_array);

    float *weights = (float *)malloc(sizeof(float) * weight_len);
    float *bias =  (float *)malloc(sizeof(float) * bias_len);

    for (int i = 0; i < weight_len; ++i)
        weights[i] = (float)cJSON_GetArrayItem(weights_array, i)->valuedouble;

    for (int i = 0; i < bias_len; ++i)
        bias[i] = (float)cJSON_GetArrayItem(bias_array, i)->valuedouble;

     
    //classifier_set_params(weights, bias, weight_len, bias_len);

    cJSON_Delete(root);
    free(weights);
    free(bias);
    return;
}

  
static void mqtt_worker_task(void *arg)
{
    mqtt_worker_msg_t msg;
    ParsedModelParams params;

    for (;;) {
        if (xQueueReceive(mqtt_msg_queue, &msg, portMAX_DELAY)) {
            ESP_LOGI(TAG, "Worker 收到模型数据 len=%d", (int)msg.len);

            if (decode_model_params(msg.data, msg.len, &params)) {
                //update_classifier_weights_bias(params.values,
                //                               params.value_count,
                //                               params.param_type);
                // 处理参数类型
                switch (params.param_type) {
                    case ParamType_CLASSIFIER_WEIGHT:
                        ESP_LOGI(TAG, "Classifier weight received, values: %d", params.value_count);
                        //handle_classifier_weight(params.values, params.values_count, params.shape, params.shape_count);
                        update_classifier_weights_bias(params.values, params.value_count,0);
                        break;
                    case ParamType_CLASSIFIER_BIAS:
                        ESP_LOGI(TAG, "Classifier bias received");
                        //handle_classifier_bias(params.values, params.value_count);
                        update_classifier_weights_bias(params.values, params.value_count,1);
                        
                        break;
                    case ParamType_ENCODER_WEIGHT:  
                            ESP_LOGI(TAG, "Fisher weight received");
                            update_fishermatrix_theta(params.values, params.value_count,0);
                        
                        break;
                    case ParamType_ENCODER_BIAS:
                        ESP_LOGI(TAG, "Fisher bias received");
                        update_fishermatrix_theta(params.values, params.value_count,1);                        
                        
                        break;
                    default:
                        ESP_LOGW(TAG, "Unknown or unsupported param_type: %d", params.param_type);
                        break;
                } 
            }

            free(msg.data);
        }
    }
} 

extern "C" void mqtt_event_handler(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data)
{
    // ✅ 显式强制转换 event_data 为 esp_mqtt_event_handle_t 类型
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t) event_data;
    esp_mqtt_client_handle_t client = event->client;

    switch ((esp_mqtt_event_id_t)event_id)
    {
        case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
            //test_protobuf();
            esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
            esp_mqtt_client_subscribe(event->client, FISH_SHAP_SUB, 1);
            esp_mqtt_client_subscribe(event->client, WEIGHT_FISH_SUB, 1);
            publish_feature_vector(0,1); 
         
        break;

        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT 收到消息 topic=%.*s, len=%d",
                    event->topic_len, event->topic, event->data_len);
            if (event->topic_len == strlen(MQTT_TOPIC_SUB) &&
                strncmp(event->topic, MQTT_TOPIC_SUB, event->topic_len) == 0) {
             
                mqtt_worker_msg_t msg;
                msg.len = event->data_len;
                msg.data =(uint8_t *) malloc(event->data_len);
                memcpy(msg.data, event->data, event->data_len);

                if (xQueueSend(mqtt_msg_queue, &msg, 0) != pdTRUE) {
                    ESP_LOGW(TAG, "队列已满，丢弃消息");
                    free(msg.data);
                }
            }
            break;

        default:
            break;
    }
}


// static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
//                                int32_t event_id, void *event_data) {
//     mqtt_event_handler_cb((esp_mqtt_event_handle_t)event_data);
    
//     return;
// }





#ifdef __cplusplus
extern "C" {
#endif  


//extern EventGroupHandle_t wifi_event_group;
//#define WIFI_CONNECTED_BIT BIT0
//#define MQTT_CONNECTED_BIT BIT1
//#define MODEL_READY_BIT    BIT2
// ---------------- 启动 MQTT 客户端 ----------------
void mqtt_task(void * pvParameters) { 
  
    // 构造唯一 client_id，例如：leaf_detector_1234s
    static char client_id[64];
    snprintf(client_id, sizeof(client_id), MQTT_CLIENT_ID_PREFIX "%ld", random() % 10000);
    //snprintf(client_id, sizeof(client_id), "mqttx_fefb0396");
    //xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

    static esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = MQTT_BROKER_URI
            }
        },
        .credentials = {
            .username = MQTT_USERNAME,
            .client_id = client_id,
            .authentication = {
                .password = MQTT_PASSWORD
            }
        },
        .session = {
            .last_will = { 0 },  // 可选：配置 LWT
            .disable_clean_session = false,
            .keepalive = MQTT_KEEPALIVE_SECONDS      
        },
        .network = {
            .reconnect_timeout_ms = 10000
        }  
    };
 
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);

    //#define ESP_EVENT_ANY_ID     ((esp_mqtt_event_id_t) -1)
    //esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);

    esp_mqtt_client_register_event(mqtt_client, (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client)); 
    // for (;;) {
    //     //可定期检查 MQTT 状态或执行心跳任务
    //      UBaseType_t free_words = uxTaskGetStackHighWaterMark(NULL);
    // ESP_LOGI("MQTT", "Stack high water mark: %u words (≈%u bytes)",
    //          free_words, free_words * 4);
     
    //    vTaskDelay(pdMS_TO_TICKS(5000));
    // }
    vTaskDelete(NULL);
    //return;
}


void periodic_task(void *pvParameter) {
    while (1) {
        publish_feature_vector(0,1);
        vTaskDelay(pdMS_TO_TICKS(120000)); // 延遲 60 秒
    }
}
// ===================== 初始化入口 =====================
void mqtt_system_start(void)
{
    mqtt_msg_queue = xQueueCreate(4, sizeof(mqtt_worker_msg_t));
    if (!mqtt_msg_queue) {
        ESP_LOGE(TAG, "创建队列失败");
        return;
    }

    xTaskCreate(mqtt_worker_task, "mqtt_worker", 8192, NULL, 5, NULL);
    xTaskCreate(mqtt_task, "mqtt_task", 4096, NULL, 5, NULL);
    xTaskCreate(periodic_task, "periodic_task", 8192, NULL, 5, NULL);
}



#ifdef __cplusplus
}
#endif

 