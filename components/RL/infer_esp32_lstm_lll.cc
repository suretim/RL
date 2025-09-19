// Add this at the VERY top of the file
#ifdef __cplusplus
extern "C" {
#endif

// Undefine any potentially conflicting macros
#ifdef CHOOSE_MACRO_VA_ARG
#undef CHOOSE_MACRO_VA_ARG
#endif
#ifdef foo
#undef foo
#endif
// 彻底清理所有可能冲突的宏
#pragma push_macro("CHOOSE_MACRO_VA_ARG")
#pragma push_macro("foo")
#pragma push_macro("ESP_STATIC_ASSERT")
#pragma push_macro("__SELECT_MACRO_VA_ARG_SIZE__")

#undef CHOOSE_MACRO_VA_ARG
#undef foo
#undef ESP_STATIC_ASSERT
#undef __SELECT_MACRO_VA_ARG_SIZE__
 
// 首先包含最基本的ESP头文件
#include "esp_system.h"
#include "soc/soc.h"

// 然后包含其他ESP头文件
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// 恢复宏（在包含所有ESP头文件之后）
#pragma pop_macro("__SELECT_MACRO_VA_ARG_SIZE__")
#pragma pop_macro("ESP_STATIC_ASSERT")
#pragma pop_macro("foo")
#pragma pop_macro("CHOOSE_MACRO_VA_ARG")


  
 
//#include "esp_macros.h"

// Then include your other headers
#include "infer_esp32_lstm_lll.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h> 
#include "sensor_module.h" 
#include "hvac_q_agent.h"
#include "classifier_storage.h"


  
#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"  

#include "classifier_storage.h" 
#include "config_mqtt.h"
#include "esp_partition.h"  
#include "spi_flash_mmap.h"   // 替代 esp_spi_flash.h
#include "nn.h"

// -------------------------
// 模型数据 (TFLite flatbuffer) lstm_encoder_contrastive 和 meta_lstm_classifier
// -------------------------
//extern const unsigned char lstm_encoder_contrastive_tflite[];
//extern const unsigned int lstm_encoder_contrastive_tflite_len;

extern const unsigned char esp32_optimized_model_tflite[];
//extern const size_t   esp32_optimized_model_tflite_len;
extern const unsigned char meta_model_tflite[];
//extern const size_t   meta_model_tflite_len;

extern const unsigned char student_model_tflite[];
//extern const size_t    student_model_tflite_len;

const unsigned char* bin_model_tflite[3] = {
    esp32_optimized_model_tflite, 
    meta_model_tflite, 
    student_model_tflite
};
// const unsigned int bin_model_tflite_len[3] = {
//     esp32_optimized_model_tflite_len, 
//     meta_model_tflite_len, 
//     student_model_tflite_len
// };


//extern const unsigned char actor_tflite[];
//extern const unsigned int meta_model_tflite_len;
const char optimized_model_path[] = "/spiffs1/esp32_optimized_model.tflite" ;
const char meta_model_path[] = "/spiffs1/meta_model.tflite" ;
const char student_model_path[] = "/spiffs1/student_model.tflite";
const char spiffs_ppo_model_bin_path[]=  "/spiffs1/ppo_model.bin" ;
const char ppo_policy_actor[]="spiffs1/ppo_policy_actor.tflite";
const char* spiffs1_model_path[FLASK_GET_COUNT] = {
    optimized_model_path,
    meta_model_path,
    student_model_path,
    spiffs_ppo_model_bin_path,
    ppo_policy_actor
};
extern float *fisher_matrix;
extern float *theta ; 
extern bool ewc_ready;
// -------------------------
// TensorArena
// -------------------------

static const char *TAG = "Inference_lstm";

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor= nullptr; 
  tflite::MicroMutableOpResolver<24> micro_op_resolver;

  constexpr int kTensorArenaSize = 256 * 1024;  
  static uint8_t *tensor_arena= nullptr;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

//float *gradients ;       // 模拟梯度

float LAMBDA_EWC = 0.001f;
float LR = 0.01f;


extern std::vector<uint8_t> ewc_buffer;
bool received_flag = false;

// 層資訊（依 Python 端 trainable_variables）
extern std::vector<std::vector<float>> trainable_layers;
extern std::vector<std::vector<float>> fisher_layers;
extern std::vector<std::vector<int>> layer_shapes;
std::vector<int> trainable_tensor_indices;     // 存 dense 層的 tensor index

float input_seq[MAX_SEQ_LEN * MAX_FEATURE_DIM] = {0.0};  // 从传感器读取
float logits[MAX_NUM_CLASSES]= {0.0};
     
// 全局變量
//trainable_tensor_indices = [0, 1, 2, 3, 6, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50, 223, 224, 225]; 
// 將 TFLite FULLY_CONNECTED 層的 shape 提取到 layer_shapes

// ---------------- NN Placeholder ----------------
// 权重向量示例
std::vector<float> W1, W2, b1, b2, Vw;
float Vb = 0.0f;
extern std::vector<float> health_result;

#define H1          32
#define H2          4


static const void *model_data_ptr = NULL;
static spi_flash_mmap_handle_t model_mmap_handle;
#include "tensorflow/lite/schema/schema_generated.h"

bool load_model_from_flash(void) {
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "spiffs2");

    if (!partition) {
        ESP_LOGE(TAG, "Partition 'spiffs2' not found");
        return false;
    }

    // 首先读取模型文件头信息来确定实际模型大小
    uint32_t model_size = 0;
    esp_err_t err = esp_partition_read(partition, 0, &model_size, sizeof(model_size));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read model size, err=%d", (int)err);
        return false;
    }
    ESP_LOGI(TAG, "Read from spiffs2 model , model_size=%d", (int)model_size);
      
    // 映射实际模型大小，而不是整个分区
    err = esp_partition_mmap(
        partition,
        0,                    // offset
        model_size,           // 使用实际模型大小，而不是整个分区大小
        ESP_PARTITION_MMAP_DATA,
        &model_data_ptr,
        &model_mmap_handle
    );

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_partition_mmap failed, err=%d", (int)err);
        return false;
    }

    ESP_LOGI(TAG, "Model mapped at addr= %d, size=%d bytes",
           (int)model_data_ptr, (int)model_size);

    // 验证模型
    model = tflite::GetModel(model_data_ptr);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d",
               (int)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    ESP_LOGI(TAG, "Model loaded successfully, version: %d", (int)model->version());
    return true;
} 


std::string shape_to_string(const std::vector<int>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); i++) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1) s += ",";
    }
    return s;
}


// 修正的ESP32验证函数
bool verify_model_integrity(const void* model_data, size_t model_size) {
    if (model_size < 16) {
        ESP_LOGE(TAG, "model size too small: %d bytes", model_size);
        return false;
    }
    
    const uint8_t* data = static_cast<const uint8_t*>(model_data);
    
     
    // 验证模型头（可选但推荐）
    if (model_size >= 16) {  // 确保文件至少有 16 字节
        // 检查 FlatBuffer 头（字节0-3）
        if (data[4] == 'T' && data[5] == 'F' && data[6] == 'L' && data[7] == '3') {
            // 验证 TFLite 魔术数字（字节 4-7）
            ESP_LOGI(TAG, "verify model integrity TFLite model header verified");
            
        } else {
            ESP_LOGW(TAG, "Unknown file format, may not be TFLite");
            return false;
        }
    } else {
        ESP_LOGW(TAG, "Model file is too short, cannot verify header");
        return false;
    }

    
    return true;
} 


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
  
 
 

bool load_from_spiffs(int type, const char* filename) {
    // 打开文件
    FILE* file = fopen(filename, "rb");
    if (!file) {
        //char *bin_str=(char *)bin_model_tflite[type];
        //save_model_to_spiffs(bin_model_tflite_len[type], bin_str, filename);
         
        //file = fopen(filename, "rb");    
        //if (!file) {
            ESP_LOGE(TAG, "Failed to open load_from_spiffs file: %s", filename);
            return false;
        //}
    }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 分配内存
    unsigned char* buf = (unsigned char*)malloc(file_size);
    if (!buf) {
        ESP_LOGE(TAG, "Failed to allocate memory for load_from_spiffs");
        fclose(file);
        return false;
    }

    // 读取文件内容
    size_t read_size = fread(buf, 1, file_size, file);
    fclose(file);

    if (read_size != file_size) {
        ESP_LOGE(TAG, "Failed to read complete file. Expected %d, read %d", file_size, read_size);
        free(buf);
        return false;
    }

    
    if(type==SPIFFS_DATA_TYPE_WEIGHT)
    {
        parse_model_weights(buf, read_size);
    }
    else if(type==SPIFFS_DATA_TYPE_MODEL){
        // 验证模型完整性
        if (verify_model_integrity(buf, read_size)) {
            ESP_LOGI(TAG, "Model verification successful");
            // if (out_size) {
            //     *out_size = file_size;
            // }
            
        } else {
            ESP_LOGE(TAG, "Model verification failed");
            free(buf);
            return false;
        }
        model = tflite::GetModel(buf);
        ESP_LOGI(TAG, "TFLITE_SCHEMA_VERSION: %d", TFLITE_SCHEMA_VERSION);
        ESP_LOGI(TAG, "Model version: %d", (int)model->version());

        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "Model schema mismatch");
            free(buf);
            return false;
        }
    }
    // 使用模型（这里是模型推理的代码）
    
    free(buf);
    return true; 
}

 
//extern bool save_model_to_spiffs(uint8_t type, const char *b64_str, const char *spi_file_name);
bool init_tflite_model(const unsigned char model_tflite[]) {
    
#if 0
    model = tflite::GetModel(model_tflite);
     
    // 检查模型版本
    ESP_LOGI(TAG, "TFLITE_SCHEMA_VERSION: %d", TFLITE_SCHEMA_VERSION);
    ESP_LOGI(TAG, "Model version: %d", (int)model->version());

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "  Model version mismatch! Expected %d, got %d", 
                TFLITE_SCHEMA_VERSION,(int) model->version());
        
        return false;
    }
#endif
    return true;
}


  
bool init_model(int type)
{
    if(model!=nullptr) {
        ESP_LOGI(TAG,"Model ready"); 
        return true;
    }
    ESP_LOGI(TAG,"Model loadding..."); //
    //if(type == PPO_CASE && load_model_from_flash()==false) {
    if( load_from_spiffs(SPIFFS_DATA_TYPE_MODEL, spiffs1_model_path[type])==false) {
    //if(type == PPO_CASE && init_tflite_model(esp32_optimized_model_tflite)==false) {
        ESP_LOGE(TAG,"Failed to load model"); 
        return false;
    }
     
    ESP_LOGI(TAG,"Interpreter loadding...");
    if (tensor_arena != nullptr) {  
        ESP_LOGI(TAG,"tensor_arena ready"); 
        return true;
    } 
    
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
    
    if (tensor_arena == nullptr) {
      printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
      return false;
    }
    ESP_LOGI(TAG, "Total memory: %d bytes", kTensorArenaSize);
    ESP_LOGI(TAG, "Memory free size: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed!");
        free(tensor_arena);
        tensor_arena = nullptr;
        return false;
    }
 
    // if(interpreter->AllocateTensors() != kTfLiteOk) {
    //     ESP_LOGE(TAG,"AllocateTensors failed"); 
    //     free(tensor_arena); 
    //     tensor_arena = nullptr;
    //     return false;
    // }

    ESP_LOGI(TAG, "TFLite arena used bytes: %d", interpreter->arena_used_bytes());




    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    if (input_tensor == nullptr || output_tensor == nullptr) {
          ESP_LOGE(TAG, "failed to get input or output tensor");
          free(tensor_arena);
          tensor_arena = nullptr;
          return false;
      }
    ESP_LOGI(TAG,"Interpreter ready. Input=%d Output=%d",
             input_tensor->bytes/sizeof(float), output_tensor->bytes/sizeof(float)); 
    
    printf("input dims: ");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        printf("%d ", input_tensor->dims->data[i]);
    }
    printf("\n");

    printf("output dims: ");
    for (int i = 0; i < output_tensor->dims->size; i++) {
        printf("%d ", output_tensor->dims->data[i]);
    }
    printf("\n");
    //free(tensor_arena);
    //tensor_arena = nullptr; 

    // if(SEQ_LEN!=input_tensor->dims->data[1] || FEATURE_DIM!=input_tensor->dims->data[2] ) 
    // {
    //     printf("input tensor dims not match  %d %d but %d %d \n",SEQ_LEN,FEATURE_DIM,input_tensor->dims->data[1],input_tensor->dims->data[2]);  
    //     return false;
    // }  
    return true;
}

void extract_layer_shapes_from_model(const tflite::Model* model) {
    if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) return;

    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph) return;

    auto* operators = subgraph->operators();
    auto* tensors   = subgraph->tensors();
    if (!operators || !tensors) return;

    layer_shapes.clear();
    trainable_tensor_indices.clear();

    for (size_t op_idx = 0; op_idx < operators->size(); op_idx++) {
        const tflite::Operator* op = operators->Get(op_idx);
        if (!op) continue;

        if (!model->operator_codes() || op->opcode_index() >= model->operator_codes()->size()) continue;
        const tflite::OperatorCode* op_code = model->operator_codes()->Get(op->opcode_index());
        if (!op_code) continue;

        if (op_code->builtin_code() == tflite::BuiltinOperator_FULLY_CONNECTED) {
            // ---- Weights ----
            int weights_idx = op->inputs()->Get(1);
            if (weights_idx >= 0 && weights_idx < tensors->size()) {
                const tflite::Tensor* w = tensors->Get(weights_idx);
                if (w && w->shape()) {
                    std::string w_name = w->name() ? w->name()->str() : "unnamed";
                    if ((w_name.find("meta_dense") != std::string::npos) ||(w_name.find("hvac_dense") != std::string::npos))
                    {
                        std::vector<int> w_shape;
                        for (int d = 0; d < w->shape()->size(); d++) {
                            w_shape.push_back(w->shape()->Get(d));
                        }
                        layer_shapes.push_back(w_shape);
                        trainable_tensor_indices.push_back(weights_idx);

                        ESP_LOGI(TAG, "Dense Weights[%zu]: %s shape=[%s] -> Added idx %d",
                                 op_idx, w_name.c_str(),
                                 shape_to_string(w_shape).c_str(),
                                 weights_idx);
                    }
                }
            }

            // ---- Bias ----
            int bias_idx = (op->inputs()->size() > 2) ? op->inputs()->Get(2) : -1;
            if (bias_idx >= 0 && bias_idx < tensors->size()) {
                const tflite::Tensor* b = tensors->Get(bias_idx);
                if (b && b->shape()) {
                    std::string b_name = b->name() ? b->name()->str() : "unnamed";
                    if( (b_name.find("meta_dense") != std::string::npos)  ||(b_name.find("hvac_dense") != std::string::npos)  )
                    {
                        std::vector<int> b_shape;
                        for (int d = 0; d < b->shape()->size(); d++) {
                            b_shape.push_back(b->shape()->Get(d));
                        }
                        layer_shapes.push_back(b_shape);
                        trainable_tensor_indices.push_back(bias_idx);

                        ESP_LOGI(TAG, "Dense Bias[%zu]: %s shape=[%s] -> Added idx %d",
                                 op_idx, b_name.c_str(),
                                 shape_to_string(b_shape).c_str(),
                                 bias_idx);
                    }
                }
            }
        }
    }

    ESP_LOGI(TAG, "Extracted %zu dense layer tensors into layer_shapes, %zu trainable indices",
             layer_shapes.size(), trainable_tensor_indices.size());
}

 
void update_dense_layer_weights(void)
{
    extern std::vector<std::vector<float>> trainable_layers;
    extern std::vector<std::vector<float>> fisher_layers;
    extern std::vector<std::vector<int>> layer_shapes;
    extern std::vector<int> trainable_tensor_indices;  // 建議新增，存放哪些 tensor 是 trainable

    for (size_t k = 0; k < trainable_tensor_indices.size(); ++k) {
        int j = trainable_tensor_indices[k];   // 取得對應 tensor id
        TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(j);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        if (!tensor || tensor->type != kTfLiteFloat32 || tensor->dims->size < 1)
            continue;

        float* theta = tensor->data.f;   // 當前權重
        std::vector<float>& theta_star = trainable_layers[k];  // 舊任務權重
        std::vector<float>& fisher = fisher_layers[k];         // Fisher
        std::vector<int>& layer_shape = layer_shapes[k];

        // 計算該層權重總數
        size_t n = 1;
        for (int s : layer_shape) n *= s;
        printf("Layer %d train weights nums: %zu\n", j, n);

        // EWC 更新
        for (size_t i = 0; i < n; ++i) {
            float grad_ewc = 2.0f * LAMBDA_EWC * fisher[i] * (theta[i] - theta_star[i]);
            theta[i] -= LR * grad_ewc;   // 直接更新 interpreter tensor
        }
    }
}

 
float compute_ewc_loss( 
                       const std::vector<std::vector<float>> &prev_weights,
                       const std::vector<std::vector<float>> &fisher_matrix) {
    float loss = 0.0f;
    for(size_t i=0; i<prev_weights.size(); ++i) {
        //TfLiteTensor* tensor = interpreter.tensor(i);
         TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(i);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        for(size_t j=0; j<prev_weights[i].size(); ++j) {
            float diff = tensor->data.f[j] - prev_weights[i][j];
            loss += fisher_matrix[i][j] * diff * diff;
        }
    }
    return LAMBDA_EWC * loss;
}
  
  
// ---------------------------
// Flowering/HVAC 判定
// ---------------------------
int is_flowering_seq(float x_seq[SEQ_LEN][FEATURE_DIM], float th_light)
{
    float mean_light = 0.0f;
    for (int t=0; t<SEQ_LEN; t++) mean_light += x_seq[t][2];
    mean_light /= SEQ_LEN;
    return mean_light >= th_light;
}
float hvac_toggle_score(float x_seq[SEQ_LEN][FEATURE_DIM], float th_toggle, int *flag) {
    float diff_sum = 0.0f;
    int count = 0;
    for (int t=1; t<SEQ_LEN; t++)
        for (int f=3; f<7; f++) {
            diff_sum += fabsf(x_seq[t][f] - x_seq[t-1][f]);
            count++;
        }
    float rate = diff_sum / count;
    *flag = rate >= th_toggle;
    return rate;
}


void reset_tensor(void)
{
  //free(tensor_arena);
  heap_caps_free(tensor_arena);
}








#ifdef __cplusplus
extern "C" {
#endif
 

#include "ml_pid.h"



pid_run_input_st lll_tensor_run_input = {0};  
pid_run_output_st ml_pid_out_speed;

CLASSIFIER_Prams classifier_params;
extern float pid_map(float x, float in_min, float in_max, float out_min, float out_max);
int load_up_input_seq(int type,int seq_len)
{
    int cnt=0;
    float v_feed  = pid_map(bp_pid_th.v_feed,  c_pid_vpd_min, c_pid_vpd_max, 0, 1);
    float t_feed  = pid_map(bp_pid_th.t_feed,  c_pid_temp_min, c_pid_temp_max, 0, 1);
    float h_feed  = pid_map(bp_pid_th.h_feed,  c_pid_humi_min, c_pid_humi_max, 0, 1);
    float l_feed  = pid_map(bp_pid_th.l_feed,  c_pid_light_min, c_pid_light_max, 0, 1);
    float c_feed  = pid_map(bp_pid_th.c_feed,  c_pid_co2_min, c_pid_co2_max, 0, 1);
    if(type == PPO_CASE)
    {
        
        
        input_seq[cnt*FEATURE_DIM + 0] = (float) v_feed;
        input_seq[cnt*FEATURE_DIM + 1] = (float) t_feed ;
        input_seq[cnt*FEATURE_DIM + 2] = (float) h_feed ;
        input_seq[cnt*FEATURE_DIM + 3] = (float) l_feed;
        input_seq[cnt*FEATURE_DIM + 4] = (float) c_feed; 
       
    }     
    if(type == META_CASE)
    {
 
        int h_idx=-1;
        uint8_t geer[6];
        for(int port=1;port<9;port++)
        {     
            switch(  devs_type_list[port].real_type  ) 
            {
                case loadType_heater:	h_idx=DEV_TU;  break;
                case loadType_A_C:		h_idx=DEV_TD;  break;
                case loadType_humi:		h_idx=DEV_HU;  break;
                case loadType_dehumi:	h_idx=DEV_HD;  break;
                case loadType_inlinefan:h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD; break;
                case loadType_fan:      h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD;   break;
                default:               break;
            }
            if(h_idx>=0)
                geer[h_idx] = ml_pid_out_speed.speed[port];
        }
        input_seq[cnt*FEATURE_DIM + 0] = (float) t_feed;
        input_seq[cnt*FEATURE_DIM + 1] = (float) h_feed;
        input_seq[cnt*FEATURE_DIM + 2] = (float) l_feed;
        input_seq[cnt*FEATURE_DIM + 3] = (float)geer[0];
        input_seq[cnt*FEATURE_DIM + 4] = (float)geer[1];
        input_seq[cnt*FEATURE_DIM + 5] = (float)geer[2];
        input_seq[cnt*FEATURE_DIM + 6] = (float)geer[3];//+rand() % 10;
    }
    cnt++;
    cnt=cnt%seq_len;
    return (cnt);
}



 
// The name of this function is important for Arduino compatibility.
TfLiteStatus infer_loop() {
  
    // 推理範例
    //float* input = interpreter.input(0)->data.f;
    //for(int i=0; i<SEQ_LEN*NUM_FEATS; ++i) input[i] = input_data[i];

    int t=0,f=0;
    
    printf("Inference output: ");
    TfLiteType tensor_type = input_tensor->type;
    float* input_data= (float*) input_tensor->data.f;
    int seq_len= classifier_params.seq_len;//input_tensor->dims->data[1];
    int num_feats= classifier_params.feature_dim;//input_tensor->dims->data[2];
        
    
    // Print the type
    switch (tensor_type) {
        case kTfLiteFloat32:
            printf("Input tensor type: kTfLiteFloat32\n");
            for (  t=0; t<seq_len; t++){
                for (  f=0; f<num_feats; f++){
                    //input_tensor->data.f[t*num_feats + f] = input_data[t*num_feats+f];
                    input_tensor->data.f[t*num_feats + f] = (float)input_seq[t*num_feats+f] ;
                }
            }
            break;
        case kTfLiteInt8:
            printf("Input tensor type: kTfLiteInt8\n"); 
            //memcpy(input_tensor->data.int8, input_data, input->bytes); 
            for (  t=0; t<seq_len; t++){
                for (  f=0; f<num_feats; f++){
                    //input_tensor->data.f[t*num_feats + f] = input_data[t*num_feats+f];
                    input_tensor->data.int8[t*num_feats + f] = (int8)input_seq[t*num_feats+f] ;
                }
            }
            break;
        case kTfLiteUInt8:
            printf("Input tensor type: kTfLiteUInt8\n");
            break;
        case kTfLiteInt32:
            printf("Input tensor type: kTfLiteInt32\n");
            break;
        case kTfLiteBool:
            printf("Input tensor type: kTfLiteBool\n");
            break;
        default:
            printf("Unknown input tensor type: %d\n", tensor_type);
    }
    printf("\n");

    
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }
    float* output = output_tensor->data.f;
  
#if INFER_CASE == PPO_CASE
    for(int i=0; i<NUM_CLASSES; ++i){
         printf("%d ", (uint8_t)output[i]);
        ml_pid_out_speed.speed[i+1] = (uint8_t)output[i];
    }
#elif INFER_CASE == SARSA_CASE
    // 計算 EWC loss
    float ewc_loss = compute_ewc_loss( trainable_layers, fisher_layers);
    printf("EWC loss: %.6f\n", ewc_loss);
    // int flowering = is_flowering_seq(x_input, 550.0f);
    // int toggle_flag;
    // float toggle_rate = hvac_toggle_score(x_input, 0.15f, &toggle_flag);

    // printf("Flowering: %d, Toggle Rate: %.4f, Toggle Flag: %d\n", flowering, toggle_rate, toggle_flag);
    // printf("Predicted probabilities: ");
    // for (int i=0; i<NUM_CLASSES; i++) printf("%.4f ", out_prob[i]);
    // printf("\n");
    
    get_mqtt_feature(output_tensor->data.f); 
    int predicted = classifier_predict(output_tensor->data.f);
    printf("Predicted class: %d\n", predicted); 
 #endif     
   vTaskDelay(1); // to avoid watchdog trigger
  return kTfLiteOk;
} 
 

void parse_ewc_assets() {
    if (!ewc_ready || ewc_buffer.empty()) return;

    extract_layer_shapes_from_model(model);
    trainable_layers.clear();
    fisher_layers.clear();

    size_t offset = 0;
    
    // Trainable layers
    for (size_t i = 0; i < layer_shapes.size(); ++i) {
        size_t len = 1;
        for (auto s : layer_shapes[i]) len *= s;

        if (offset + len > ewc_buffer.size()) {
            ESP_LOGE("EWC", "Not enough data for trainable layer %zu", i);
            return; // 避免越界
        }

        std::vector<float> layer_data(len);
        memcpy(layer_data.data(), ewc_buffer.data() + offset, len * sizeof(float));
        trainable_layers.push_back(layer_data);
        offset += len;

        ESP_LOGI("EWC", "Parsed trainable layer %zu, len=%zu", i, len);
    }

    // Fisher layers
    for (size_t i = 0; i < layer_shapes.size(); ++i) {
        size_t len = 1;
        for (auto s : layer_shapes[i]) len *= s;

        if (offset + len > ewc_buffer.size()) {
            ESP_LOGE("EWC", "Not enough data for fisher layer %zu", i);
            return; // 避免越界
        }

        std::vector<float> arr(
            ewc_buffer.begin() + offset,
            ewc_buffer.begin() + offset + len
        );
        fisher_layers.push_back(std::move(arr));
        offset += len;
    }

    ESP_LOGI("EWC", "Parsed EWC assets: %zu trainable, %zu fisher layers",
             trainable_layers.size(), fisher_layers.size());

    // 用完清空 buffer
    ewc_buffer.clear();
    
    if (!trainable_layers.empty()) { 
        update_dense_layer_weights();
        

        trainable_layers.clear();  // 可選，保留 capacity
        fisher_layers.clear();
        ESP_LOGI("Main", "All layers updated");
    }
    ewc_ready = false;
}
 
bool ppo_inference(float *input_data) {
 
     ESP_LOGI(TAG, "ppo_inference Invoke ");
     // 假设模型只用 10 种算子
     //tflite::MicroMutableOpResolver<10> micro_op_resolver;
     if(model==nullptr){
        micro_op_resolver.AddUnidirectionalSequenceLSTM();
        micro_op_resolver.AddShape();            // SHAPE操作符 - 之前缺失的
        micro_op_resolver.AddStridedSlice();     // STRIDED_SLICE操作符 - 现在缺失的 ← 添加这一行
        micro_op_resolver.AddFullyConnected();   // 全连接层
        micro_op_resolver.AddReshape();          // 重塑层
        micro_op_resolver.AddSoftmax();          // Softmax
        micro_op_resolver.AddRelu();             // ReLU激活
        micro_op_resolver.AddMul();              // 乘法
        micro_op_resolver.AddAdd();              // 加法
        micro_op_resolver.AddSub();              // 减法

        micro_op_resolver.AddConcatenation();    // 连接操作     
        micro_op_resolver.AddSplit();            // 分割操作
        micro_op_resolver.AddTanh();             // Tanh激活（LSTM常用）
        micro_op_resolver.AddMean();              
        micro_op_resolver.AddAbs();              
        micro_op_resolver.AddFill();              
        micro_op_resolver.AddLogistic();              
        micro_op_resolver.AddLessEqual();
        micro_op_resolver.AddPack();             // Pack操作
        micro_op_resolver.AddUnpack();           // Unpack操作

        micro_op_resolver.AddTranspose();        // 转置操作
        if(    init_model(SPIFFS_DOWN_LOAD_MODEL)==false){
            ESP_LOGE(TAG,"Init ppo_inference Model Failed");
            return false;
        } 
    

        ESP_LOGI("INFERENCE", "Input dimensions: %dD", input_tensor->dims->size);
        for (int i = 0; i < input_tensor->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, input_tensor->dims->data[i]);
        }
        for (int i = 0; i < output_tensor->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, output_tensor->dims->data[i]);
        }
    
    
        // 准备输入数据 - 需要10个时间步，每个时间步7个特征
        //float* input_data = input_tensor->data.f;
        //float input_data[50] = {0};
        
        // 填充输入数据
        //    for(int i=0;i<50;i++) input_data[i] = (float) rand () / UINT32_MAX;
        //int8_t input_data[FEATURE_DIM ];  // 输入数据数组 
    
        //memcpy(input_tensor->data.int8, input_data, input->bytes);  
    }
    
    
    infer_loop();
#if 0
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }  
    float out_logits[40] = {0};  
    int num_classes = output_tensor->dims->data[1];
    memcpy(out_logits, output_tensor->data.f, num_classes * sizeof(float));
    printf("PPO Inference output: ");
    for(int i=0; i<num_classes; ++i) printf("%.3f ", out_logits[i]);
    printf("\n");           
#endif 
    return true;
}
  
// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
bool sarsa_inference( float* input_seq) {
    // int seq_len, int num_feats
   // tflite::MicroMutableOpResolver<24> micro_op_resolver;
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu(); 
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();  // 🔧 添加这个
    micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddSoftmax();

    micro_op_resolver.AddAdd(); 
    micro_op_resolver.AddSub();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddUnpack();  
    micro_op_resolver.AddFill();
    micro_op_resolver.AddSplit(); 
    micro_op_resolver.AddLogistic();  // This handles sigmoid activation CONCATENATION
    micro_op_resolver.AddTanh();

    micro_op_resolver.AddMean();
    micro_op_resolver.AddAbs();
    micro_op_resolver.AddConcatenation();  
     
    if(    init_model(META_MODEL)==false){
        ESP_LOGE(TAG,"Init sarsa_inference Model Failed");
        return false;
    }
    // 微调示意：更新权重，EWC参与 
    parse_ewc_assets();   
 
    for (int t=0; t<SEQ_LEN; t++)
        for (int f=0; f<FEATURE_DIM; f++)
            input_tensor->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
     
    
     
    // 7) 读取输出
     
    //int num_classes = output->dims->data[1];
    //memcpy(out_logits, output->data.f, num_classes * sizeof(float));
    infer_loop();
     
     
     
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return true;
}
  
bool img_inference( float* input_seq) {
    // int seq_len, int num_feats
    //tflite::MicroMutableOpResolver<24> micro_op_resolver;
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu(); 
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();  // 🔧 添加这个
    micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddSoftmax();

    micro_op_resolver.AddAdd(); 
    micro_op_resolver.AddSub();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddUnpack();  
    micro_op_resolver.AddFill();
    micro_op_resolver.AddSplit(); 
    micro_op_resolver.AddLogistic();  // This handles sigmoid activation CONCATENATION
    micro_op_resolver.AddTanh();

    micro_op_resolver.AddMean();
    micro_op_resolver.AddAbs();
    micro_op_resolver.AddConcatenation();  
     
    
    if(    init_model(IMG_MODEL)==false){
        ESP_LOGE(TAG,"Init image_inference Model Failed");
        return false;
    }
    for (int t=0; t<SEQ_LEN; t++)
        for (int f=0; f<FEATURE_DIM; f++)
            input_tensor->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
      
    // 7) 读取输出 
    //int num_classes = output->dims->data[1];
    //memcpy(out_logits, output->data.f, num_classes * sizeof(float));
    infer_loop(); 
     
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return true;
}



bool (*functionInferArray[3])(float* input_seq) = {
    ppo_inference,  
    sarsa_inference,
    img_inference,
};
 

void prepare_lstm_input() {
    // 获取输入张量
     
    
    // 检查输入维度
    ESP_LOGI("INFERENCE", "Input dimensions: %dD", input_tensor->dims->size);
    for (int i = 0; i < input_tensor->dims->size; i++) {
        ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, input_tensor->dims->data[i]);
    }
    
    // 输入应该是 [1, 10, 7] - batch=1, timesteps=10, features=7
    if (input_tensor->dims->size != 3 || 
        input_tensor->dims->data[0] != 1 || 
        input_tensor->dims->data[1] != 10 || 
        input_tensor->dims->data[2] != 7) {
        ESP_LOGE("INFERENCE", "Unexpected input shape");
        return;
    }
    
    // 准备输入数据 - 需要10个时间步，每个时间步7个特征
    float* input_data = input_tensor->data.f;
    
    // 示例：填充10个时间步的数据
    for (int timestep = 0; timestep < 10; timestep++) {
        for (int feature = 0; feature < 7; feature++) {
            // 这里根据您的实际数据填充
            // 例如：input_data[timestep * 7 + feature] = your_sensor_data[feature];
            input_data[timestep * 7 + feature] = 0.0f; // 临时用0填充
        }
    }
}

 


void catch_tensor_dim(enum CaseType type) {
    classifier_params.infer_case=INFER_CASE;
    classifier_params.feature_dim = FEATURE_DIM;
    classifier_params.num_classes = NUM_CLASSES;
    classifier_params.seq_len = SEQ_LEN;
    if (type == PPO_CASE) {
        classifier_params.infer_case=PPO_CASE;
        classifier_params.feature_dim = PPO_FEATURE_DIM;
        classifier_params.num_classes = PPO_CLASSES;
        classifier_params.seq_len = PPO_SEQ_LEN;
    }
    if (type == META_CASE) {
        classifier_params.infer_case=META_CASE;
        classifier_params.feature_dim = META_FEATURE_DIM;
        classifier_params.num_classes = META_CLASSES;
        classifier_params.seq_len = META_SEQ_LEN;
    }
    if (type == IMG_CASE) {
        classifier_params.infer_case=IMG_CASE;
        classifier_params.feature_dim = IMG_FEATURE_DIM;
        classifier_params.num_classes = IMG_CLASSES;
        classifier_params.seq_len = IMG_SEQ_LEN;
    }
    
}
extern std::array<int,PORT_CNT> plant_action;
//u_int8_t get_tensor_state(void);
esp_err_t  lll_tensor_run(void) 
{
    // int16_t sensor_val_list[ENV_CNT];
	// uint8_t cur_load_type[PORT_CNT];
	// uint8_t port_dev_origin[PORT_CNT];

	// pid_run_output_st pid_run_output;
	// pid_param_get(ai_setting, cur_load_type, port_dev_origin, sensor_val_list, &pid_run_input );
	// pid_run_output = pid_run_rule( &pid_run_input );
	// pid_rule_output_set_speed(pid_run_output, cur_load_type, output_port_list );

    // extern void pid_param_get(ai_setting_t *ai_setting, uint8_t* load_type_list, uint8_t* dev_origin_list, int16_t* env_value_list, pid_run_input_st* param);
     
    catch_tensor_dim(PPO_CASE); 
    read_all_sensor_trigger();
    // pid_param_get(&g_ai_setting, NULL, NULL, NULL, &pid_run_input );
     
    lll_tensor_run_input.env_en_bit  = 0xff;
    lll_tensor_run_input.ml_run_sta  = 1;
    lll_tensor_run_input.env_target[ENV_TEMP] =30.0;
    lll_tensor_run_input.env_target[ENV_HUMID]=60.0;
    lll_tensor_run_input.env_target[ENV_LIGHT]=32.0;
    lll_tensor_run_input.env_target[ENV_CO2]  =50.0;
    devs_type_list[1].real_type = lll_tensor_run_input.dev_type[1] =loadType_A_C;
    devs_type_list[2].real_type = lll_tensor_run_input.dev_type[2]= loadType_heater;
    devs_type_list[3].real_type = lll_tensor_run_input.dev_type[3]= loadType_dehumi;
    devs_type_list[4].real_type = lll_tensor_run_input.dev_type[4]= loadType_humi;
    devs_type_list[5].real_type = lll_tensor_run_input.dev_type[5]= loadType_A_C;
    devs_type_list[6].real_type = lll_tensor_run_input.dev_type[6]= loadType_heater;
    devs_type_list[7].real_type = lll_tensor_run_input.dev_type[7]= loadType_dehumi;
    devs_type_list[8].real_type = lll_tensor_run_input.dev_type[8]= loadType_humi;
    for(int port=1;port<9;port++)
    {    
        lll_tensor_run_input.is_switch[port] = 1;
    }
    pid_run_output_st out_speed = pid_run_rule( &lll_tensor_run_input );
    for(int port=1;port< PORT_CNT;port++)
    {    
        ml_pid_out_speed.speed[port] += out_speed.speed[port];
        plant_action[port-1]=ml_pid_out_speed.speed[port];
    }
    
    int ret=load_up_input_seq(classifier_params.infer_case,classifier_params.seq_len); 
     
    if(ret==0)
    {    
 
        if( false == functionInferArray[classifier_params.infer_case](input_seq))   
        {
            vTaskDelay(pdMS_TO_TICKS(10));
            return ESP_FAIL;  //kTfLiteOK
        }    
    } 
        //ESP_LOGI(TAG, "Set up 2 input tensor %d",cnt);
        
        vTaskDelay(pdMS_TO_TICKS(1000));  // 60000 每60秒输出一次
      //  vTaskDelay(30000 / portTICK_PERIOD_MS);
    //}  
    //reset_tensor();
    return ESP_OK;
}

#ifdef __cplusplus
}
#endif 
