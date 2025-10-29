// Add this at the VERY top of the file
#ifdef __cplusplus
extern "C" {
#endif
 
// 首先包含最基本的ESP头文件
#include "esp_system.h"
#include "soc/soc.h"

// 然后包含其他ESP头文件
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
  
// Then include your other headers
#include "infer_esp32_lstm_lll.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h> 
#include "sensor_module.h" 
#include "hvac_q_agent.h"

#ifdef __cplusplus
}
#endif

// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/schema/schema_generated.h"
#include "model_context.h"
#include "PlantHVACEnv.h" 
#include "classifier_storage.h" 
#include "config_mqtt.h"
#include "esp_partition.h"  
#include "spi_flash_mmap.h"   // 替代 esp_spi_flash.h
#include "nn.h"
#include "ni_debug.h"   
 
extern const unsigned char _binary_esp32_optimized_model_tflite_start[] asm("_binary_esp32_optimized_model_tflite_start");
extern const unsigned char _binary_esp32_optimized_model_tflite_end[]   asm("_binary_esp32_optimized_model_tflite_end");
const size_t   esp32_optimized_model_tflite_len=_binary_esp32_optimized_model_tflite_end-_binary_esp32_optimized_model_tflite_start;

extern const unsigned char _binary_meta_lstm_classifier_tflite_start[] asm("_binary_meta_lstm_classifier_tflite_start");
extern const unsigned char _binary_meta_lstm_classifier_tflite_end[]   asm("_binary_meta_lstm_classifier_tflite_end"); 
const size_t  meta_lstm_classifier_tflite_len=_binary_meta_lstm_classifier_tflite_end-_binary_meta_lstm_classifier_tflite_start;
 
extern const unsigned char _binary_meta_model_tflite_start[] asm("_binary_meta_model_tflite_start");
extern const unsigned char _binary_meta_model_tflite_end[]   asm("_binary_meta_model_tflite_end"); 
const size_t  meta_model_tflite_len=_binary_meta_model_tflite_end-_binary_meta_model_tflite_start;
 

extern const unsigned char _binary_actor_task0_tflite_start[] asm("_binary_actor_task0_tflite_start");
extern const unsigned char _binary_actor_task0_tflite_end[]   asm("_binary_actor_task0_tflite_end"); 
size_t actor_task0_tflite_len = _binary_actor_task0_tflite_end - _binary_actor_task0_tflite_start;
 
extern const unsigned char _binary_critic_task0_tflite_start[] asm("_binary_critic_task0_tflite_start");
extern const unsigned char _binary_critic_task0_tflite_end[]   asm("_binary_critic_task0_tflite_end"); 
const size_t    critic_task0_tflite_len=_binary_critic_task0_tflite_end - _binary_critic_task0_tflite_start;

const unsigned char* bin_model_tflite[NUM_INFER_CASE] = {
    _binary_esp32_optimized_model_tflite_start, 
    _binary_meta_lstm_classifier_tflite_start, 
    _binary_actor_task0_tflite_start,
    _binary_critic_task0_tflite_start
};
const unsigned int bin_model_tflite_len[NUM_INFER_CASE] = {
    esp32_optimized_model_tflite_len, 
    meta_lstm_classifier_tflite_len, 
    actor_task0_tflite_len,
    critic_task0_tflite_len
};
 
const char optimized_model_path[] = "/spiffs1/esp32_optimized_model.tflite" ;
const char meta_model_path[]      = "/spiffs1/meta_lstm_classifier.tflite" ;
const char actor_model_path[]     = "/spiffs1/actor_task0.tflite";
const char critic_model_path[]    = "/spiffs1/critic_task0.tflite";
const char spiffs_ppo_model_bin_path[]=  "/spiffs2/ppo_model.bin" ;
const char ppo_policy[]="spiffs2/policy.tflite";
 
const char* spiffs1_model_path[SPIFFS1_MODEL_COUNT] = {
    optimized_model_path,
    meta_model_path,
    actor_model_path,
    critic_model_path 
};
const char* spiffs2_model_path[SPIFFS2_MODEL_COUNT] = { 
    spiffs_ppo_model_bin_path,
    ppo_policy
};
extern float *fisher_matrix;
extern float *theta ; 
extern bool ewc_ready;
// -------------------------
// TensorArena
// -------------------------

static const char *TAG = "Inference_lstm";
#if 1
 
 
// 全局容器 (管理多个模型)
//namespace {
    //constexpr int kNumModels = 4;
    constexpr size_t kTensorArenaSize[NUM_INFER_CASE] = {
        64 * 1024, 64 * 1024, 64 * 1024, 64 * 1024, 256 * 1024
    };

    std::array<ModelContext, NUM_INFER_CASE> model_contexts; 
//}

#else
// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model[4] = {nullptr};
  tflite::MicroInterpreter* interpreter[4] = {nullptr};
  TfLiteTensor* input_tensor[4] = {nullptr};
  TfLiteTensor* output_tensor[4]= {nullptr}; 
  tflite::MicroMutableOpResolver<24> micro_op_resolver;

  constexpr int kTensorArenaSize[4] = { 16 * 1024, 16 * 1024, 128 * 1024, 256 * 1024 }; // 4 * 8k 256 * 1024;  
  static uint8_t *tensor_arena[4]= {nullptr, nullptr, nullptr, nullptr};//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace
 
#endif
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

bool load_model_from_flash(int type) {
    auto& ctx = model_contexts[type];
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
    ctx.model = tflite::GetModel(model_data_ptr);
    if (ctx.model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d",
               (int)ctx.model ->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    ESP_LOGI(TAG, "Model loaded successfully, version: %d", (int)ctx.model ->version());
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
    W1.assign(ptr + offset, ptr + offset + H1 * STATE_DIM);
    offset += H1 * STATE_DIM;

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
  
 
 

unsigned char*  load_from_spiffs(int type, const char* filename,size_t &file_size) {
    // 打开文件
    FILE* file = fopen(filename, "rb");
    if (!file) {
        //char *bin_str=(char *)bin_model_tflite[type];
        //save_model_to_spiffs(bin_model_tflite_len[type], bin_str, filename);
         
        //file = fopen(filename, "rb");    
        //if (!file) {
            ESP_LOGE(TAG, "Failed to open load_from_spiffs file: %s", filename);
            return nullptr;
        //}
    } 
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET); 
    // 分配内存
    unsigned char* buf = (unsigned char*)malloc(file_size);
    if (!buf) {
        ESP_LOGE(TAG, "Failed to allocate memory for load_from_spiffs from load_from_spiffs");
        fclose(file);
        return nullptr;
    } 
    // 读取文件内容
    size_t read_size = fread(buf, 1, file_size, file);
    fclose(file);

    if (read_size != file_size) {
        ESP_LOGE(TAG, "Failed to read complete file. Expected %d, read %d", file_size, read_size);
        free(buf);
        return nullptr;
    }  
    return buf; 
}

  bool init_model(int type)
{
    auto& ctx = model_contexts[type];  // 取出对应的 context

    if (ctx.model != nullptr) {
        ESP_LOGI(TAG, "Model[%d] already initialized", type);
        return true;
    }

    ESP_LOGI(TAG, "Loading model[%d]...", type);

    size_t file_size = 0;
    bool from_spiffs = false;
    unsigned char* buf = load_from_spiffs(
        SPIFFS_DATA_TYPE_MODEL,
        spiffs1_model_path[type],
        file_size
    );

    if (buf == nullptr || file_size == 0) {
        ESP_LOGW(TAG, "Falling back to embedded model[%d]", type);
        buf       = (unsigned char*)bin_model_tflite[type];
        file_size = bin_model_tflite_len[type];
        ESP_LOGI("Model", "Embedded model[%d] size = %d bytes", type, file_size);
    } else {
        from_spiffs = true;
    }

    if (!verify_model_integrity(buf, file_size)) {
        ESP_LOGE(TAG, "Model verification failed");
        if (from_spiffs) free(buf);
        return false;
    }

    ctx.model = tflite::GetModel(buf);
    if (ctx.model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: model=%d, expect=%d",
                 (int)ctx.model->version(), TFLITE_SCHEMA_VERSION);
        if (from_spiffs) free(buf);
        return false;
    }

    if (from_spiffs) {
        ESP_LOGI(TAG, "init_model [%d] loaded from SPIFFS (%d bytes)", type, file_size);
    } else {
        ESP_LOGI(TAG, "init_model [%d] loaded from embedded binary (%d bytes)", type, file_size);
    }

    // 确保旧 arena 被释放
    if (ctx.tensor_arena != nullptr) {
        free(ctx.tensor_arena);
        ctx.tensor_arena = nullptr;
    }

    ctx.tensor_arena_size = kTensorArenaSize[type];
    ctx.tensor_arena = (uint8_t*)heap_caps_malloc(
        ctx.tensor_arena_size,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (ctx.tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Couldn't allocate tensor arena (%d bytes)", ctx.tensor_arena_size);
        if (from_spiffs) free(buf);
        return false;
    }

    ctx.interpreter = new tflite::MicroInterpreter(
        ctx.model,
        ctx.micro_op_resolver,
        ctx.tensor_arena,
        ctx.tensor_arena_size
    );

    if (ctx.interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        free(ctx.tensor_arena);
        ctx.tensor_arena = nullptr;
        if (from_spiffs) free(buf);
        return false;
    }

    ctx.input_tensor  = ctx.interpreter->input(0);
    ctx.output_tensor = ctx.interpreter->output(0);

    if (ctx.input_tensor == nullptr || ctx.output_tensor == nullptr) {
        ESP_LOGE(TAG, "Failed to get input/output tensor");
        free(ctx.tensor_arena);
        ctx.tensor_arena = nullptr;
        if (from_spiffs) free(buf);
        return false;
    }

    ESP_LOGI(TAG, "Interpreter ready. Input=%d Output=%d",
             ctx.input_tensor->bytes / sizeof(float),
             ctx.output_tensor->bytes / sizeof(float));

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

 
void update_dense_layer_weights(int type)
{
    auto &ctx=model_contexts[type];
    extern std::vector<std::vector<float>> trainable_layers;
    extern std::vector<std::vector<float>> fisher_layers;
    extern std::vector<std::vector<int>> layer_shapes;
    extern std::vector<int> trainable_tensor_indices;  // 建議新增，存放哪些 tensor 是 trainable

    for (size_t k = 0; k < trainable_tensor_indices.size(); ++k) {
        int j = trainable_tensor_indices[k];   // 取得對應 tensor id
        TfLiteEvalTensor* eval_tensor = ctx.interpreter ->GetTensor(j);
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

 
float compute_ewc_loss( int type,
                       const std::vector<std::vector<float>> &prev_weights,
                       const std::vector<std::vector<float>> &fisher_matrix) {
    float loss = 0.0f;
    auto &ctx=model_contexts[type];
    for(size_t i=0; i<prev_weights.size(); ++i) {
        //TfLiteTensor* tensor = interpreter.tensor(i);
         TfLiteEvalTensor* eval_tensor = ctx.interpreter ->GetTensor(i);
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
// int is_flowering_seq(float x_seq[SEQ_LEN][FEATURE_DIM], float th_light)
// {
//     float mean_light = 0.0f;
//     for (int t=0; t<SEQ_LEN; t++) mean_light += x_seq[t][2];
//     mean_light /= SEQ_LEN;
//     return mean_light >= th_light;
// }
// float hvac_toggle_score(float x_seq[SEQ_LEN][FEATURE_DIM], float th_toggle, int *flag) {
//     float diff_sum = 0.0f;
//     int count = 0;
//     for (int t=1; t<SEQ_LEN; t++)
//         for (int f=3; f<7; f++) {
//             diff_sum += fabsf(x_seq[t][f] - x_seq[t-1][f]);
//             count++;
//         }
//     float rate = diff_sum / count;
//     *flag = rate >= th_toggle;
//     return rate;
// }


void reset_tensor(int type)
{
  //free(tensor_arena);
  heap_caps_free(model_contexts[type].tensor_arena);
}








#ifdef __cplusplus
extern "C" {
#endif
 


#include "ml_pid.h"


//pid_run_input_st pid_input = {0};  
//extern pid_run_output_st lstm_pid_out_speed;

extern float pid_map(float x, float in_min, float in_max, float out_min, float out_max);
int load_up_input_seq( int type,int seq_len,int feature_dim)
{
    static int cnt=0;
    // float t_feed  = pid_map(bp_pid_th.t_feed,   m_range_params.temp_range.first, m_range_params.temp_range.second, 0, 1);
    // float h_feed  = pid_map(bp_pid_th.h_feed,   m_range_params.humid_range.first, m_range_params.humid_range.second, 0, 1);
    // float w_feed  = pid_map(bp_pid_th.w_feed,   m_range_params.water_range.first, m_range_params.water_range.second, 0, 1);
    // float l_feed  = pid_map(bp_pid_th.l_feed,   m_range_params.light_range.first, m_range_params.light_range.second, 0, 1);
    // float c_feed  = pid_map(bp_pid_th.c_feed,   m_range_params.co2_range.first, m_range_params.co2_range.second, 0, 1);
    // float p_feed  = pid_map(bp_pid_th.p_feed,   m_range_params.ph_range.first, m_range_params.ph_range.second, 0, 1);
    // float v_feed  = pid_map(bp_pid_th.v_feed,   m_range_params.vpd_range.first, m_range_params.vpd_range.second, 0, 1);
    
    //if(type == PPO_CASE)
    //{ 
        for(int i=0;i< feature_dim;i++){
            input_seq[cnt* feature_dim + i] = (float) bp_pid_th.f[0][i]; 
        }
    //}     
    //if(type == META_CASE)
    //{
 
        //int h_idx=-1;
        //uint8_t geer[6];
        // for(int port=1;port<9;port++)
        // {     
        //     switch(  devs_type_list[port].real_type  ) 
        //     {
        //         case loadType_heater:	h_idx=DEV_TU;  break;
        //         case loadType_A_C:		h_idx=DEV_TD;  break;
        //         case loadType_humi:		h_idx=DEV_HU;  break;
        //         case loadType_dehumi:	h_idx=DEV_HD;  break;
        //         case loadType_inlinefan:h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD; break;
        //         case loadType_fan:      h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD;   break;
        //         default:               break;
        //     }
        //     if(h_idx>=0)
        //         geer[h_idx] = ml_pid_out_speed.speed[port];
        // }
        //for(int i=0;i<classifier_params.feature_dim;i++){
        //    input_seq[cnt*classifier_params.feature_dim + i] = (float) bp_pid_th.f[0][i]; 
        //}
    //}
    cnt++;
    cnt=cnt%seq_len;
    return (cnt);
}
struct TrainableArray {
    std::vector<float> weights;      // current trainable weights (shadow)
    std::vector<float> fisher;       // estimated fisher diagonal
    std::vector<float> old_weights;  // snapshot at time of consolidation

    TrainableArray() {}
    TrainableArray(size_t n) { resize(n); }
    void resize(size_t n) {
        weights.assign(n, 0.0f);
        fisher.assign(n, 0.0f);
        old_weights.assign(n, 0.0f);
    }
};


TfLiteStatus infer_loop( int type,int sub_type) {
    auto&ctx=model_contexts[type];
    int seq_len  = ctx.classifier_params.seq_len;
    int num_feats = ctx.classifier_params.feature_dim;

    // ===== 输入填充 =====
    switch (ctx.input_tensor->type) {
        case kTfLiteFloat32: {
            float* in_buf = ctx.input_tensor ->data.f;
            for (int t=0; t<seq_len; t++) {
                for (int f=0; f<num_feats; f++) {
                    in_buf[t*num_feats + f] = (float)input_seq[t*num_feats+f];
                }
            }
            break;
        }
        case kTfLiteInt8: {
            int8_t* in_buf = ctx.input_tensor ->data.int8;
            for (int t=0; t<seq_len; t++) {
                for (int f=0; f<num_feats; f++) {
                    in_buf[t*num_feats + f] = (int8_t)input_seq[t*num_feats+f];
                }
            }
            break;
        }
        case kTfLiteUInt8: {
            uint8_t* in_buf = ctx.input_tensor ->data.uint8;
            for (int t=0; t<seq_len; t++) {
                for (int f=0; f<num_feats; f++) {
                    in_buf[t*num_feats + f] = (uint8_t)input_seq[t*num_feats+f];
                }
            }
            break;
        }
        default:
            ESP_LOGE(TAG, "Unsupported input tensor type: %d", ctx.input_tensor ->type);
            return kTfLiteError;
    }

    // ===== 执行推理 =====
    if (ctx.interpreter ->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }

    float* output = ctx.output_tensor ->data.f;
    TfLiteTensor* out = ctx.interpreter ->output(0); 
    size_t m = 1;
    int i=0;
    for (  i = 0; i < out->dims->size; ++i) {
        m *= out->dims->data[i];
    } 
    std::vector<float>  output_vec ;
    // ===== 按模型类型输出 =====
    if (sub_type == CRITIC_MODEL) {
        
        output_vec.resize(m);
        memcpy(output_vec.data(), ctx.interpreter ->typed_output_tensor<float>(0), m * sizeof(float)); 
        printf("Critic output: %.4f\n", output_vec[0]); 
        lstm_pid_out_speed.speed[0] = output_vec[0]; 
    }
    else if (sub_type == OPTIMIZED_MODEL || sub_type == ACTOR_MODEL) {
        printf("Actor/Optimized output: ");
        float sum=0.0f;
        for (  i=0; i<m; i++) {
            printf("%.4f ", output[i]);
            sum+=output[i];
        }
        for (  i=0; i<m; i++) {
            if(sum< 1e-8)
                lstm_pid_out_speed.speed[i+1]=1.0f/m;
            else
                lstm_pid_out_speed.speed[i+1] = output[i]/sum;
        }
        printf("\n");
    }
    else if (sub_type == META_MODEL) {
        get_mqtt_feature(output);
        int predicted = classifier_predict(type,output);
        printf("META_MODEL Predicted class: %d\n", predicted);
    }

    vTaskDelay(1); // 防止 watchdog
    return kTfLiteOk;
}

  
void parse_ewc_assets(int type) {
    if (!ewc_ready || ewc_buffer.empty()) {
        return;
    }
    auto& ctx=model_contexts[type];
    extract_layer_shapes_from_model(ctx.model);
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
        update_dense_layer_weights(type);
        

        trainable_layers.clear();  // 可選，保留 capacity
        fisher_layers.clear();
        ESP_LOGI("Main", "All layers updated");
    }
    ewc_ready = false;
}
 
  

bool actor_critic_infer(int sub_type )
{ 
    auto&ctx=model_contexts[PPO_CASE];
    if(ctx.model==nullptr){
        ctx.micro_op_resolver.AddFullyConnected();
        ctx.micro_op_resolver.AddSoftmax();
        ctx.micro_op_resolver.AddReshape();
        ctx.micro_op_resolver.AddRelu();
        ctx.micro_op_resolver.AddQuantize();
        ctx.micro_op_resolver.AddDequantize();
        ctx.micro_op_resolver.AddTanh();
        
        if(    init_model(PPO_CASE)==false){
            ESP_LOGE(TAG,"Init actor_critic_inference Model Failed");
            return false;
        }   
        ESP_LOGI("INFERENCE", "Input dimensions: %dD", ctx.input_tensor->dims->size);
        for (int i = 0; i < ctx.input_tensor ->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, ctx.input_tensor ->dims->data[i]);
        }
        for (int i = 0; i < ctx.output_tensor ->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, ctx.output_tensor ->dims->data[i]);
        }
    
    }
    if(sub_type==ACTOR_MODEL)
    {
       ESP_LOGI("INFERENCE", "actor  sub_type= %d ", sub_type);
    }
    else
    {
       ESP_LOGI("INFERENCE", "critic  sub_type= %d ", sub_type);
    }
    infer_loop(PPO_CASE,sub_type);
    return true;
}


bool ppo_inference(int sub_type) {
 auto &ctx=model_contexts[NN_PPO_CASE];
     ESP_LOGI(TAG, "ppo_inference Invoke ");
     // 假设模型只用 10 种算子
     //tflite::MicroMutableOpResolver<10> micro_op_resolver;
     if(ctx.model==nullptr){
        ctx.micro_op_resolver.AddUnidirectionalSequenceLSTM();
        ctx.micro_op_resolver.AddShape();            // SHAPE操作符 - 之前缺失的
        ctx.micro_op_resolver.AddStridedSlice();     // STRIDED_SLICE操作符 - 现在缺失的 ← 添加这一行
        ctx.micro_op_resolver.AddFullyConnected();   // 全连接层
        ctx.micro_op_resolver.AddReshape();          // 重塑层
        ctx.micro_op_resolver.AddSoftmax();          // Softmax
        ctx.micro_op_resolver.AddRelu();             // ReLU激活
        ctx.micro_op_resolver.AddMul();              // 乘法
        ctx.micro_op_resolver.AddAdd();              // 加法
        ctx.micro_op_resolver.AddSub();              // 减法

        ctx.micro_op_resolver.AddConcatenation();    // 连接操作     
        ctx.micro_op_resolver.AddSplit();            // 分割操作
        ctx.micro_op_resolver.AddTanh();             // Tanh激活（LSTM常用）
        ctx.micro_op_resolver.AddMean();              
        ctx.micro_op_resolver.AddAbs();              
        ctx.micro_op_resolver.AddFill();              
        ctx.micro_op_resolver.AddLogistic();              
        ctx.micro_op_resolver.AddLessEqual();
        ctx.micro_op_resolver.AddPack();             // Pack操作
        ctx.micro_op_resolver.AddUnpack();           // Unpack操作

        ctx.micro_op_resolver.AddTranspose();        // 转置操作
        if(    init_model(NN_PPO_CASE)==false){
            ESP_LOGE(TAG,"Init ppo_inference Model Failed");
            return false;
        }  
        ESP_LOGI("INFERENCE", "Input dimensions: %dD", ctx.input_tensor->dims->size);
        for (int i = 0; i < ctx.input_tensor ->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, ctx.input_tensor ->dims->data[i]);
        }
        for (int i = 0; i < ctx.output_tensor ->dims->size; i++) {
            ESP_LOGI("INFERENCE", "  dim[%d]: %d", i, ctx.output_tensor ->dims->data[i]);
        }
     
    }
    
    
    infer_loop(NN_PPO_CASE,sub_type);
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
bool meta_inference(int sub_type) {
   auto &ctx=model_contexts[META_CASE];
   if(ctx.model==nullptr){
        ctx.micro_op_resolver.AddStridedSlice();
        ctx.micro_op_resolver.AddPack();
        ctx.micro_op_resolver.AddConv2D();
        ctx.micro_op_resolver.AddRelu(); 
        ctx.micro_op_resolver.AddAveragePool2D();
        ctx.micro_op_resolver.AddReshape();   
        ctx.micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
        ctx.micro_op_resolver.AddQuantize();
        ctx.micro_op_resolver.AddDequantize();
        ctx.micro_op_resolver.AddSoftmax();

        ctx.micro_op_resolver.AddAdd(); 
        ctx.micro_op_resolver.AddSub();
        ctx.micro_op_resolver.AddMul();
        ctx.micro_op_resolver.AddShape();
        ctx.micro_op_resolver.AddTranspose();
        ctx.micro_op_resolver.AddUnpack();  
        ctx.micro_op_resolver.AddFill();
        ctx.micro_op_resolver.AddSplit(); 
        ctx.micro_op_resolver.AddLogistic();  // This handles sigmoid activation CONCATENATION
        ctx.micro_op_resolver.AddTanh();

        ctx.micro_op_resolver.AddMean();
        ctx.micro_op_resolver.AddAbs();
        ctx.micro_op_resolver.AddConcatenation();  
        
        if( init_model(META_CASE)==false){
                ESP_LOGE(TAG,"Init Meta inference Model Failed");
                return false;
        }  
        ESP_LOGI("Meta INFERENCE", "Input dimensions: %dD", ctx.input_tensor->dims->size);
        for (int i = 0; i < ctx.input_tensor ->dims->size; i++) {
            ESP_LOGI("Meta INFERENCE", "  dim[%d]: %d", i, ctx.input_tensor ->dims->data[i]);
        }
        for (int i = 0; i < ctx.output_tensor ->dims->size; i++) {
            ESP_LOGI("Meta INFERENCE", "  dim[%d]: %d", i, ctx.output_tensor ->dims->data[i]);
        }
        
        
        
    } 
    // 微调 更新权重，EWC参与 
    parse_ewc_assets(META_CASE);   
    
    infer_loop(META_CASE,sub_type);   
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return true;
}
  
 
// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
bool meta_lstm_inference(int type) {
   auto &ctx=model_contexts[META_LSTM_CASE];
   if(ctx.model==nullptr){
        ctx.micro_op_resolver.AddStridedSlice();
        ctx.micro_op_resolver.AddPack();
        ctx.micro_op_resolver.AddConv2D();
        ctx.micro_op_resolver.AddRelu(); 
        ctx.micro_op_resolver.AddAveragePool2D();
        ctx.micro_op_resolver.AddReshape();   
        ctx.micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
        ctx.micro_op_resolver.AddQuantize();
        ctx.micro_op_resolver.AddDequantize();
        ctx.micro_op_resolver.AddSoftmax();

        ctx.micro_op_resolver.AddAdd(); 
        ctx.micro_op_resolver.AddSub();
        ctx.micro_op_resolver.AddMul();
        ctx.micro_op_resolver.AddShape();
        ctx.micro_op_resolver.AddTranspose();
        ctx.micro_op_resolver.AddUnpack();  
        ctx.micro_op_resolver.AddFill();
        ctx.micro_op_resolver.AddSplit(); 
        ctx.micro_op_resolver.AddLogistic();  // This handles sigmoid activation CONCATENATION
        ctx.micro_op_resolver.AddTanh();

        ctx.micro_op_resolver.AddMean();
        ctx.micro_op_resolver.AddAbs();
        ctx.micro_op_resolver.AddConcatenation();  
        
        if( init_model(META_LSTM_CASE)==false){
                ESP_LOGE(TAG,"Init Meta inference Model Failed");
                return false;
        }  
        ESP_LOGI("Meta INFERENCE", "Input dimensions: %dD", ctx.input_tensor->dims->size);
        for (int i = 0; i < ctx.input_tensor ->dims->size; i++) {
            ESP_LOGI("Meta INFERENCE", "  dim[%d]: %d", i, ctx.input_tensor ->dims->data[i]);
        }
        for (int i = 0; i < ctx.output_tensor ->dims->size; i++) {
            ESP_LOGI("Meta INFERENCE", "  dim[%d]: %d", i, ctx.output_tensor ->dims->data[i]);
        } 
        
    } 
    
    infer_loop(META_LSTM_CASE,type);   
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return true;
}
 

bool img_inference(int type) {
    auto &ctx= model_contexts[IMG_CASE];
    // int seq_len, int num_feats
    //tflite::MicroMutableOpResolver<24> micro_op_resolver;
    if(ctx.model==nullptr)
    {
        ctx.micro_op_resolver.AddStridedSlice();
        ctx.micro_op_resolver.AddPack();
        ctx.micro_op_resolver.AddConv2D();
        ctx.micro_op_resolver.AddRelu(); 
        ctx.micro_op_resolver.AddAveragePool2D();
        ctx.micro_op_resolver.AddReshape();  
        ctx.micro_op_resolver.AddFullyConnected();   
        ctx.micro_op_resolver.AddQuantize();
        ctx.micro_op_resolver.AddDequantize();
        ctx.micro_op_resolver.AddSoftmax();

        ctx.micro_op_resolver.AddAdd(); 
        ctx.micro_op_resolver.AddSub();
        ctx.micro_op_resolver.AddMul();
        ctx.micro_op_resolver.AddShape();
        ctx.micro_op_resolver.AddTranspose();
        ctx.micro_op_resolver.AddUnpack();  
        ctx.micro_op_resolver.AddFill();
        ctx.micro_op_resolver.AddSplit(); 
        ctx.micro_op_resolver.AddLogistic();   
        ctx.micro_op_resolver.AddTanh();

        ctx.micro_op_resolver.AddMean();
        ctx.micro_op_resolver.AddAbs();
        ctx.micro_op_resolver.AddConcatenation();  
        
        
        if( init_model(IMG_CASE)==false){
            ESP_LOGE(TAG,"Init image_inference Model Failed");
            return false;
        }
    }
     
    infer_loop(IMG_CASE,type); 
     
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return true;
}



bool (*functionInferArray[NUM_INFER_CASE])(int type) = {
    actor_critic_infer,
    ppo_inference, 
    meta_lstm_inference, 
    meta_inference,    
    img_inference
};
 
 

//void catch_tensor_dim(enum CaseType type) {
void catch_tensor_dim(int type) {
    auto&ctx=model_contexts[type];
    //ctx.classifier_params.infer_case=INFER_CASE;
    //ctx.classifier_params.feature_dim = META_FEATURE_DIM;
    //ctx.classifier_params.num_classes = META_CLASSES;
    //ctx.classifier_params.seq_len = META_SEQ_LEN;

    if (type == PPO_CASE|| type == NN_PPO_CASE) {
        ctx.classifier_params.infer_case=PPO_CASE;
        ctx.classifier_params.feature_dim = PPO_FEATURE_DIM;
        ctx.classifier_params.num_classes = PPO_CLASSES;
        ctx.classifier_params.seq_len = PPO_SEQ_LEN;
    }
    if (type == META_CASE || type == META_LSTM_CASE) {
        ctx.classifier_params.infer_case=META_CASE;
        ctx.classifier_params.feature_dim = META_FEATURE_DIM;
        ctx.classifier_params.num_classes = META_CLASSES;
        ctx.classifier_params.seq_len = META_SEQ_LEN;
    }
    if (type == IMG_CASE) {
        ctx.classifier_params.infer_case=IMG_CASE;
        ctx.classifier_params.feature_dim = IMG_FEATURE_DIM;
        ctx.classifier_params.num_classes = IMG_CLASSES;
        ctx.classifier_params.seq_len = IMG_SEQ_LEN;
    }
    
}


//float itm_heat      =  0.01f;        
// float itm_ac        = -0.005f;
// float itm_humid     =  0.0002f;
// float itm_dehumi    = -0.0005f;
// float itm_waterpump = 1.0;
// float itm_light     = 20.f;
// float itm_co2       = 50.0f;
// float itm_pump      = 1.1f;
//extern void set_plant_action(const std::array<int, ACTION_CNT>& action);
void set_plant_action(const std::array<int, ACTION_CNT>& action) {
    extern std::array<int, ACTION_CNT> plant_action ;
    extern std::vector<float> reward_history;
     
    plant_action = action;
}
//u_int8_t get_tensor_state(void);
 
extern curLoad_t curLoad[PORT_CNT] ;
extern pid_run_input_st pid_run_input;
//extern st_bp_pid_th  r_pid_th;

bool pid_env_init(void) 
{ 
    //pid_param_get(&g_ai_setting, NULL, NULL, NULL, &pid_run_input );
    
    pid_run_input.env_target[ENV_TEMP]  =(plant_range_params.temp_range.first  + plant_range_params.temp_range.second)/2.0;
    pid_run_input.env_target[ENV_HUMID] =(plant_range_params.humid_range.first + plant_range_params.humid_range.second)/2.0;
    pid_run_input.env_target[ENV_LIGHT] =(plant_range_params.light_range.first + plant_range_params.light_range.second)/2.0;
    pid_run_input.env_target[ENV_CO2]   =(plant_range_params.co2_range.first   + plant_range_params.co2_range.second)/2.0; 
    pid_run_input.env_target[ENV_SOIL]  =(plant_range_params.water_range.first + plant_range_params.water_range.second)/2.0; 
    if(pid_run_input.env_target[ENV_TEMP]==0||pid_run_input.env_target[ENV_HUMID]==0)
        return false;
    pid_run_input.dev_type[1] =loadType_heater ;
    pid_run_input.dev_type[2]= loadType_A_C;
    pid_run_input.dev_type[3]= loadType_humi ;
    pid_run_input.dev_type[4]= loadType_dehumi;
    pid_run_input.dev_type[5]= loadType_water_pump;
    pid_run_input.dev_type[6]= loadType_growLight;
    pid_run_input.dev_type[7]= loadType_co2_generator;
    pid_run_input.dev_type[8]= loadType_pump;
    curLoad[1].load_type=loadType_heater ;
    curLoad[2].load_type=loadType_A_C ;
    curLoad[3].load_type=loadType_humi ;
    curLoad[4].load_type=loadType_dehumi ;
    curLoad[5].load_type=loadType_water_pump ;
    curLoad[6].load_type=loadType_growLight ;
    curLoad[7].load_type=loadType_co2_generator ;
    curLoad[8].load_type=loadType_pump ;
     
    pid_run_input.env_en_bit  = (1 << ENV_TEMP) | (1 << ENV_HUMID)| (1 << ENV_VPD);
    pid_run_input.ml_run_sta  = 1;
      
    for(int port=1;port<PORT_CNT;port++)
    {    
        pid_run_input.is_switch[port] = 1;
    } 
    r_env_th.t_target=pid_run_input.env_target[ENV_TEMP];
    r_env_th.h_target=pid_run_input.env_target[ENV_HUMID];  
    bp_pid_dbg("pid_env_init Sucessfully \r\n");
    return true;
}

void pid_run(void) 
{   
    if(read_all_sensor_trigger()==false)
    {
        bp_pid_dbg("Sensor Working Error \r\n");
        return ;
    }  
    if( pid_env_init() ==false)
    {
        bp_pid_dbg("pid_env_init Error \r\n");
        return  ;
    } 
    
	
    if(pid_run_rule( &pid_run_input )==true)
    {
        std::array<int,ACTION_CNT>  action;
        for(int port=1;port<= ACTION_CNT;port++)
        {    
            
            action[port-1]=ml_pid_out_speed.speed[port];
            set_plant_action(action);

        }
        bp_pid_dbg("pid_run_output_st =(%d,%d,%d,%d)\r\n", action[0],action[1], action[2],action[3]);
    }
    return;
}


esp_err_t  lll_tensor_run(int type,int sub_type) 
{ 
    auto&ctx=model_contexts[type];
    catch_tensor_dim(type); 
    int ret=load_up_input_seq( ctx.classifier_params.infer_case,ctx.classifier_params.seq_len, ctx.classifier_params.feature_dim); 
     ESP_LOGI(TAG, "Input tensor type=%d,seq_cnt=%d,seq_len=%d",type,ret,ctx.classifier_params.seq_len); 
    if(ret==0)
    {    

        if(   false == functionInferArray[type](sub_type))   
        {
            vTaskDelay(pdMS_TO_TICKS(10));
            return ESP_FAIL;  //kTfLiteOK
        }
        
        // if( type== PPO_CASE && false == functionInferArray[type](ACTOR_MODEL) && false == functionInferArray[type](CRITIC_MODEL))   
        // {
        //     vTaskDelay(pdMS_TO_TICKS(10));
        //     return ESP_FAIL;  //kTfLiteOK
        // }  
            
     
        //ESP_LOGI(TAG, "Input tensor type=%d,sub_type=%d",type,sub_type);
    }    
        vTaskDelay(pdMS_TO_TICKS(1000));  // 60000 每60秒输出一次
      //  vTaskDelay(30000 / portTICK_PERIOD_MS);
    //}  
    //reset_tensor();
    return ESP_OK;
}

#ifdef __cplusplus
}
#endif 
