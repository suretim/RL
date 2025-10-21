#include "classifier_storage.h"
//#include "classifier.h"
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"

#include "esp_system.h" 
#include <sstream> 
#include <esp_task_wdt.h>  
#include "config_mqtt.h"   
#include <math.h> 
#include <stdio.h>   
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  

  
 
float *fisher_matrix ;   // 每个变量的 Fisher 数组
float *theta ;    // 上一次权重
size_t theta_len=0;
size_t fisher_len=0; 
char bin_str[2][10]={"clf_bin","fish_bin"};



// 全局存放权重和偏置
float classifier_weights[MAX_FEATURE_DIM * MAX_NUM_CLASSES];
float classifier_bias[MAX_NUM_CLASSES];

static const char* TAG = "ClassifierStorage";

constexpr int FisherArenaSize = FISHER_LAYER * 256 *sizeof(float); // 调整大小


void free_fisher_matrix()
{
    free(fisher_matrix);
    free(theta);
}
static float softmax(float x[], int len, int *out_index) {
    float max = x[0];
    for (int i = 1; i < len; ++i)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    float probs[len];
    for (int i = 0; i < len; ++i) {
        probs[i] = expf(x[i] - max);
        sum += probs[i];
    }

    int max_idx = 0;
    float max_prob = 0;
    for (int i = 0; i < len; ++i) {
        probs[i] /= sum;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }

    if (out_index) *out_index = max_idx;
    return max_prob;
}


int classifier_predict(const float *features) {
    float logits[NUM_CLASSES] = {0};
    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < FEATURE_DIM; ++j) {
            logits[i] += classifier_weights[i*FEATURE_DIM + j] * features[j];
        }
        logits[i] += classifier_bias[i];
    }

    int predicted_class = 0;
    softmax(logits, NUM_CLASSES, &predicted_class);
    publish_feature_vector(predicted_class,0);
    return predicted_class;
} 
 

// 更新分类器的权重和偏置
void update_classifier_weights_bias(const float* values, int value_count,int type) {
    
    
    int expected = NUM_CLASSES * FEATURE_DIM ;
    if(type==1)
    {
        expected=NUM_CLASSES;
    }
    
    if (value_count != expected) {
        printf("权重参数数量不匹配，期望 %d，实际 %d\n", expected, value_count);
        return;
    }
    if(type==0)
    {
        // 前 NUM_CLASSES * NUM_INPUTS 是权重
        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < FEATURE_DIM; ++j) {
                classifier_weights[i*FEATURE_DIM + j] = values[i * FEATURE_DIM + j];
            }
        }
    }
    else{    
        // 后 NUM_CLASSES 是 bias
        for (int i = 0; i < NUM_CLASSES; ++i) {
            classifier_bias[i] = values[ i];
        }
    }

}
void update_fishermatrix_theta(const float* values, int value_count,int type) {
      
    if(type==0)
    {
        memcpy(fisher_matrix, values, value_count);
        fisher_len=value_count;
        printf("分类器 fisher_matrix 权重更新完成\n");
        return;
    }
    else{
        memcpy(theta, values,value_count);
        theta_len=value_count;
        printf("分类器 theta 权重更新完成\n");
        return;    
    } 
} 

 
void set_classifier_from_buffer(const uint8_t* buf, size_t len,size_t type) {
    size_t expected = (size_t)(FEATURE_DIM * NUM_CLASSES + NUM_CLASSES) * sizeof(float);
    if(type==1)
        expected = FisherArenaSize*2;
    if (len <= expected) {
        ESP_LOGE(TAG, "Buffer too small: len=%d expected=%d", (int)len, (int)expected);
        return; //   绝对不能 memcpy
    } 
     if(type==0){
        
        memcpy(classifier_weights, buf, FEATURE_DIM * NUM_CLASSES * sizeof(float));
        memcpy(classifier_bias, buf + FEATURE_DIM * NUM_CLASSES * sizeof(float),NUM_CLASSES * sizeof(float));
        ESP_LOGI(TAG, "Classifier updated in memory");
    }
    if(type==1){
        size_t fish_len=len/2; 
        //assert(fish_len==fisher_len);
        memcpy(fisher_matrix, buf, fish_len);
        memcpy(theta, buf + fish_len,fish_len);
        ESP_LOGI(TAG, "Classifier updated in memory");
    }
}

// ==============================
// 从 buffer 更新 + 写入 NVS
// ==============================
int update_classifier_from_bin(const uint8_t* data, size_t len,size_t type) {
    size_t expected = (size_t)(FEATURE_DIM * NUM_CLASSES + NUM_CLASSES) * sizeof(float);
    if (len < expected) {
        ESP_LOGE(TAG, "update_classifier_from_bin: insufficient data len=%d expected=%d",
                 (int)len, (int)expected);
        return -1;
    }

    // 更新内存
    set_classifier_from_buffer(data, len,type);

    // 存入 NVS
    nvs_handle_t h;
    esp_err_t err = nvs_open("model", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "NVS open error %s", esp_err_to_name(err));
        return -1;
    } 
    err = nvs_set_blob(h, bin_str[type], data, expected);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Classifier persisted to NVS");
        return 0;
    } else {
        ESP_LOGE(TAG, "NVS write error %s", esp_err_to_name(err));
        return -1;
    }
}

// ==============================
// 从 NVS 恢复
// ==============================
 static void init_default_classifier(void) {
    memset(classifier_weights, 0, sizeof(classifier_weights));
    memset(classifier_bias, 0, sizeof(classifier_bias));
    ESP_LOGI(TAG, "Classifier initialized with zeros (default)");
}

int restore_classifier_from_nvs(size_t type) {
    ESP_LOGI(TAG, "Attempting to restore classifier from NVS...");
    size_t free_heap = esp_get_free_heap_size();
    ESP_LOGI("SAFE", "Free heap: %d bytes", free_heap);
    
    if (free_heap < 10240) { // 少于 10KB
        ESP_LOGE("SAFE", "Critical: Low memory, skipping NVS operations");
        return 0;
    }
    nvs_handle_t h;
    esp_err_t err = nvs_open("model", NVS_READONLY, &h);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "nvs_open failed: %s (normal if first boot)", esp_err_to_name(err));
        init_default_classifier();
        return 0;
    }

    size_t required = 0;
    
     
    err = nvs_get_blob(h,bin_str[type], NULL, &required);
    if (err != ESP_OK || required == 0) {
        ESP_LOGW(TAG, "Classifier blob not found or size=0");
        nvs_close(h);
        init_default_classifier();
        return 0;
    }

    uint8_t* buf =(uint8_t*) malloc(required);
    if (!buf) {
        ESP_LOGE(TAG, "malloc failed");
        nvs_close(h);
        init_default_classifier();
        return 0;
    }

    err = nvs_get_blob(h, bin_str[type], buf, &required);
    nvs_close(h);

    if (err == ESP_OK) {
        set_classifier_from_buffer(buf, required,type);
        free(buf);
        ESP_LOGI(TAG, "Classifier restored from NVS successfully");
        return required;
    }

    free(buf);
    ESP_LOGE(TAG, "Failed to read classifier from NVS: %s", esp_err_to_name(err));
    init_default_classifier();
    return 0;
}

void initialize_nvs() {
    ESP_LOGI(TAG, "Initializing NVS...");
    
    // 尝试初始化 NVS
    esp_err_t err = nvs_flash_init();
    
    // 如果 NVS 分区损坏
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition corrupted, erasing and recreating...");
        
        // 完全擦除 NVS 分区
        ESP_ERROR_CHECK(nvs_flash_erase());
        
        // 重新初始化
        err = nvs_flash_init();
    }
    
    // 检查最终结果
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "NVS initialization failed: %s", esp_err_to_name(err));
        ESP_LOGE(TAG, "Restarting in 5 seconds...");
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }
    
    ESP_LOGI(TAG, "NVS initialized successfully");
}
//esptool.py --port COM5 erase_flash
 // Restore classifier weights from NVS if present - UPDATED API
 
// 安全的命名空间检查函数
bool check_namespace_exists(const char* namespace_name) {
    nvs_iterator_t it = NULL;
    esp_err_t err = nvs_entry_find("nvs", namespace_name, NVS_TYPE_ANY, &it);
    
    bool exists = (err == ESP_OK && it != NULL);
    
    if (it != NULL) {
        nvs_release_iterator(it);
    }
    
    return exists;
}

// 调试函数：列出所有命名空间和键
void debug_nvs_contents() {
    nvs_iterator_t it = NULL;
    esp_err_t err = nvs_entry_find("nvs", NULL, NVS_TYPE_ANY, &it);
    
    if (err != ESP_OK) {
        ESP_LOGI("NVS_DEBUG", "No entries found in NVS");
        return;
    }
    
    ESP_LOGI("NVS_DEBUG", "NVS contents:");
    
    while (err == ESP_OK) {
        nvs_entry_info_t info;
        nvs_entry_info(it, &info);
        
        ESP_LOGI("NVS_DEBUG", "  Namespace: %s, Key: %s, Type: %d", 
                info.namespace_name, info.key, info.type);
        
        err = nvs_entry_next(&it);
    }
    
    nvs_release_iterator(it);
} 



#ifdef __cplusplus
extern "C" {
#endif
void initialize_nvs_robust(void) {
    ESP_LOGI(TAG, "Initializing NVS with robust handling...");
    
    // 首先尝试正常初始化
    esp_err_t err = nvs_flash_init();
    
    // 处理常见的 NVS 错误
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition corrupted, erasing...");
        
        // 完全擦除 NVS 分区
        ESP_ERROR_CHECK(nvs_flash_erase());
        
        // 重新初始化
        err = nvs_flash_init();
    }
    
    // 如果还有其他错误
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "NVS initialization failed: %s", esp_err_to_name(err));
        ESP_LOGE(TAG, "This is a critical error, restarting...");
        
        // 等待一段时间让日志输出
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_restart();
    }
    
    ESP_LOGI(TAG, "NVS initialized successfully");
} 


void init_classifier_from_header(void)
{
    if (fisher_matrix == NULL) {
        fisher_matrix = (float*) heap_caps_malloc(FisherArenaSize, MALLOC_CAP_SPIRAM);
    }
    if (theta == NULL) {
        theta = (float*) heap_caps_malloc(FisherArenaSize, MALLOC_CAP_SPIRAM);
    }
    int fcls_len= restore_classifier_from_nvs(0) ;  //get from nvs
    int fish_len= restore_classifier_from_nvs(1) ;  //get from nvs
    //assert(fcls_len == FEATURE_DIM * NUM_CLASSES);
    //assert(fish_len == fisher_len + theta_len);
 
    // classifier_weights_data and classifier_bias_data are in classifier_weights.h
#ifdef classifier_weights_data
    // The header contains shapes; but we just copy flatten array
    // NOTE: header written classifier_weights_data in shape [feature_dim][num_classes]
    // We need to copy into our flat buffer in row-major
    int header_len = classifier_weights_len; // produced by header generator
    if (header_len >= FEATURE_DIM * NUM_CLASSES) {
        for (int i = 0; i < FEATURE_DIM * NUM_CLASSES; i++) {
            classifier_weights[i] = classifier_weights_data[i];
        }
    }
    // bias
    for (int j = 0; j < NUM_CLASSES; j++) {
        classifier_bias[j] = classifier_bias_data[j];
    }
#else
    //ESP_LOGI(TAG, "weights @%p size=%d", classifier_weights, (int)sizeof(classifier_weights));
    //ESP_LOGI(TAG, "bias    @%p size=%d", classifier_bias, (int)sizeof(classifier_bias));

    // no header: zero-init
    //memset(classifier_weights, 0, sizeof(classifier_weights));
    //memset(classifier_bias, 0, sizeof(classifier_bias));
#endif
} 

#ifdef __cplusplus
}
#endif
