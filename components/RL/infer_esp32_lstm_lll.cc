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




#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"  

#include "classifier_storage.h" 
#include "config_mqtt.h"




// -------------------------
// 模型数据 (TFLite flatbuffer) lstm_encoder_contrastive 和 meta_lstm_classifier
// -------------------------
//extern const unsigned char lstm_encoder_contrastive_tflite[];
//extern const unsigned int lstm_encoder_contrastive_tflite_len;

extern const unsigned char meta_model_tflite[];
extern const unsigned int meta_model_tflite_len;

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

  constexpr int kTensorArenaSize = 1024 * 1024;  
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

     
// 全局變量
//trainable_tensor_indices = [0, 1, 2, 3, 6, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50, 223, 224, 225]; 
// 將 TFLite FULLY_CONNECTED 層的 shape 提取到 layer_shapes

 

std::string shape_to_string(const std::vector<int>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); i++) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1) s += ",";
    }
    return s;
}

// 初始化 TFLite Micro interpreter
bool init_spiffs_model(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if(!f) { ESP_LOGE(TAG,"Failed to open model"); return false; }
    fseek(f, 0, SEEK_END);
    size_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *model_data = (uint8_t*)malloc(model_size);
    fread(model_data, 1, model_size, f);
    fclose(f);

    const tflite::Model *model = tflite::GetModel(model_data);
    if(model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG,"Model schema mismatch"); free(model_data); return false;
    }
    return true;
}



bool init_tflite_model(void) {
    
    model = tflite::GetModel(meta_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      MicroPrintf("Model provided is schema version %d not equal to supported "
                  "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
      return false ;
    } 
    if (tensor_arena == NULL) {
       tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    }
    if (tensor_arena == NULL) {
      printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
      return false;
    }
    return true;
}


  
bool init_model(int type)
{
    if(type == 0) {
        init_spiffs_model("/model/lstm_encoder_contrastive.tflite");
        
    }
    if(type == 1) {
        init_tflite_model();
       
    }
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG,"AllocateTensors failed"); 
        free(tensor_arena); 
        tensor_arena = nullptr;
        return false;
    }

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
    
    
    printf("input dims: %d %d %d %d  output dims: %d %d  \n",
        input_tensor->dims->data[0],
        input_tensor->dims->data[1],
        input_tensor->dims->data[2],
        input_tensor->dims->data[3],
        output_tensor->dims->data[0],
        output_tensor->dims->data[1] 
    );

    free(tensor_arena);
    tensor_arena = nullptr;
    if(SEQ_LEN!=input_tensor->dims->data[1] || FEATURE_DIM!=input_tensor->dims->data[2] ) 
    {
        printf("input tensor dims not match  %d %d but %d %d \n",SEQ_LEN,FEATURE_DIM,input_tensor->dims->data[1],input_tensor->dims->data[2]);  
        return false;
    }

    
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


// The name of this function is important for Arduino compatibility.
TfLiteStatus sarsa_loop() {
  
    // 推理範例
    //float* input = interpreter.input(0)->data.f;
    //for(int i=0; i<SEQ_LEN*NUM_FEATS; ++i) input[i] = input_data[i];

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }
      
    
    float* output = output_tensor->data.f;
    printf("Inference output: ");
    for(int i=0; i<NUM_CLASSES; ++i) printf("%.3f ", output[i]);
    printf("\n");

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
 
esp_err_t ppo_inference(void) {

    
   
     
 // 假设模型只用 10 种算子
    tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();

  init_model(1);
 

     
 
    float input_data[50] = {0};
    float out_logits[40] = {0}; 
    // 填充输入数据
        for(int i=0;i<50;i++) input_data[i] = (float) rand () / UINT32_MAX;

     int seq_len= input_tensor->dims->data[1];
     int num_feats= input_tensor->dims->data[2];
     for (int t=0; t<seq_len; t++)
            for (int f=0; f<num_feats; f++)
                input_tensor->data.f[t*num_feats + f] = input_data[t*num_feats+f];

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }
      
    
     
    int num_classes = output_tensor->dims->data[1];
    memcpy(out_logits, output_tensor->data.f, num_classes * sizeof(float));
    printf("PPO Inference output: ");
    for(int i=0; i<num_classes; ++i) printf("%.3f ", out_logits[i]);
    printf("\n");           
     
    return kTfLiteOk;
}
  
// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
TfLiteStatus sarsa_inference(float* input_seq, int seq_len, int num_feats, float* out_logits) {
    
    tflite::MicroMutableOpResolver<24> micro_op_resolver;
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

     
          init_model(1);

    // 微调示意：更新权重，EWC参与 
    parse_ewc_assets();   
 
    for (int t=0; t<SEQ_LEN; t++)
            for (int f=0; f<FEATURE_DIM; f++)
                input_tensor->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
     
    
     
    // 7) 读取输出
     
    //int num_classes = output->dims->data[1];
    //memcpy(out_logits, output->data.f, num_classes * sizeof(float));
 
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }
      
    
    float* output = output_tensor->data.f;
    printf("Inference output: ");
    for(int i=0; i<NUM_CLASSES; ++i) printf("%.3f ", output[i]);
    printf("\n");

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
    vTaskDelay(1); // to avoid watchdog trigger 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
    return kTfLiteOk;
}

#ifdef __cplusplus
extern "C" {
#endif

 #include "ml_pid.h"

float input_seq[SEQ_LEN * FEATURE_DIM] = {25.0};  // 从传感器读取
float logits[NUM_CLASSES];
 


pid_run_output_st ml_pid_out_speed;

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
     
     
    read_all_sensor();
    pid_run_input_st pid_run_input = {0};
    // pid_param_get(&g_ai_setting, NULL, NULL, NULL, &pid_run_input );
    pid_run_input.dev_type[0] = loadType_heater;
    pid_run_input.is_switch[0] = 1;
    pid_run_input.env_en_bit  = 0xf;
    pid_run_input.ml_run_sta  = 1;
    pid_run_input.env_target[ENV_TEMP]=32.0;
    pid_run_input.env_target[ENV_HUMID]=32.0;
    pid_run_input.env_target[ENV_LIGHT]=320;
    ml_pid_out_speed= pid_run_rule( &pid_run_input );
    devs_type_list[1].real_type = loadType_heater;
    devs_type_list[2].real_type = loadType_A_C;
    devs_type_list[3].real_type = loadType_humi;
    devs_type_list[4].real_type = loadType_dehumi;
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
    //return ESP_OK;
     static int cnt=0; 
     //while (true)
     //{ 
        //
        if(cnt==SEQ_LEN)
        {
            //ESP_LOGI(TAG, "run_inference %d",cnt);
            if(kTfLiteError == ppo_inference())
            //if( kTfLiteError ==  sarsa_inference(input_seq, SEQ_LEN, FEATURE_DIM, logits) )
            {
                vTaskDelay(pdMS_TO_TICKS(10));
                return ESP_FAIL;  //kTfLiteOK
            }
            //ESP_LOGI(TAG, "run_inference logits %f,%f,%f",logits[0],logits[1],logits[2]);
            cnt=0;
        }
        input_seq[cnt*FEATURE_DIM + 0] = (float)bp_pid_th.t_feed;
        input_seq[cnt*FEATURE_DIM + 1] = (float)bp_pid_th.h_feed;
        input_seq[cnt*FEATURE_DIM + 2] = (float)bp_pid_th.l_feed;
        input_seq[cnt*FEATURE_DIM + 3] = (float)geer[0];
        input_seq[cnt*FEATURE_DIM + 4] = (float)geer[1];
        input_seq[cnt*FEATURE_DIM + 5] = (float)geer[2];
        input_seq[cnt*FEATURE_DIM + 6] = (float)geer[3];//+rand() % 10;
        //ESP_LOGI(TAG, "Set up 2 input tensor %d",cnt);
        cnt++;
        
      vTaskDelay(pdMS_TO_TICKS(1000));  // 60000 每60秒输出一次
      //  vTaskDelay(30000 / portTICK_PERIOD_MS);
    //}  
    //reset_tensor();
    return ESP_OK;
}

#ifdef __cplusplus
}
#endif 
