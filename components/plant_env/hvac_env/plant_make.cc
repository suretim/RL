#include <vector>
#include <array>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h> 
#include <sstream>

#include "esp_http_client.h"
#include "esp_log.h"
#include "cJSON.h"
#include "PlantHVACEnv.h"
#include "fl_client.h"   
#include "hvac_q_agent.h"
#include "config_wifi.h"
#include "version.h"
#include "nn.h"
#include "ml_pid.h"
std::vector<float> health_result;
//std::vector<float> data = {0.1f, 25.0f, 0.5f, 500.0f, 600.0f};
std::vector<std::vector<float>> state_history; // 存储每步的状态
std::vector<float> reward_history;

std::array<int, ACTION_CNT> plant_action{};   

PlantHVACEnv env(20, 3, 25.0f, 0.5f, 64);

static const char *TAG = "HTTP_CLIENT";
struct Transition {
    std::vector<float> state;   // 状态
    int action;                 // 动作
    float reward;               // 奖励
    float value;                // Critic 估值
    float old_log_prob;         // 动作的 log_prob
};
// 將向量轉 JSON
char* vec_to_json(const std::vector<float>& data) {
    cJSON *root = cJSON_CreateObject();
    cJSON *arr = cJSON_CreateArray();

    for (float v : data) {
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(v));
    }
    cJSON_AddItemToObject(root, "obs", arr);

    char *json_str = cJSON_PrintUnformatted(root); // 產生字串
    cJSON_Delete(root);
    return json_str; // 注意: 需要 free()
}

extern "C" bool send_seq_to_server(void) {
    
	const char* url = "http://192.168.0.57:5000/push_data";

    char* post_data = vec_to_json(health_result);

    esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_POST,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);

    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_perform(client);

    free(post_data); // 釋放記憶體
    esp_http_client_cleanup(client);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "POST success, status = %d",
                 esp_http_client_get_status_code(client));
        return true;
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
        return false;
    }
}



const std::array<int, ACTION_CNT>& get_plant_action() {
    return plant_action;
}


void fetch_seq_from_server_step() {
    
    env.set_seq_fetcher([](int t) -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> seq_input;
        char task_str[32]="seq_input";
        char task_url[128];
        sprintf(task_url, "http://%s:%s/%s", BASE_URL,BASE_PORT,task_str); 

        //sprintf(task_url, "http://%s/seq_input", BASE_URL);
        std::string url =task_url;// "http://192.168.0.57:5000/seq_input";
        bool ok = fetch_seq_from_server(seq_input, url);  
        if(!ok){
            // 默认填充
            seq_input = std::vector<std::vector<float>>(20, std::vector<float>(3, 0.0f));
        }
         
        return seq_input;
    }); 
}
extern "C" void plant_env_step(void) {  
        
        auto  result = env.step(plant_action,env.default_params);
        std::vector<float> new_state = result.state;    // 当前新的状态
        float reward = result.reward;                   // 当前的奖励值
        bool done = result.done;                        // 是否任务完成
 
        state_history.push_back(new_state);  // 存储当前状态
        reward_history.push_back(reward);    // 存储当前奖励

        // 如果任务完成，可能需要重新初始化环境
        if (done) { 
            std::cout << "Task finished. Resetting environment." << std::endl; 
            env.reset();
        }

        // 你还可以在这里进行其他操作，如更新策略、打印日志等
        std::cout << "New state: ";
        for (auto s : new_state) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Reward: " << reward << std::endl;
        
        std::vector<float> health_result={
            result.reward, 
            result.flower_prob           
        };
        return  ;
}
 

// ======= PPO 模型 =======
PPOEWCModel ewcppoModel; 
extern struct st_bp_pid_th bp_pid_th ;
extern float* load_float_bin(const char* path, size_t &length) ; 
extern "C" bool hvac_ewc(void) {
    vTaskDelay(pdMS_TO_TICKS(5000)); // 等 WiFi 连接

    // size_t length = 0;
    // float *buf = load_float_bin("/spiffs1/ppo_model.bin", length);
    // if (buf == NULL) {
    //     ESP_LOGI(TAG, "Model loaded successfully");
    // } else {
    //     ESP_LOGE(TAG, "Failed to load model");
    // }

    // 初始化环境
    bool done = false;
    
    env.reset(); 
    // PPO 模型 
    ewcppoModel.initModel();  // 如果有 initModel()
    
    static std::vector<float> old_action_probs = {0.0f, 0.0f, 0.0f};
 
    bp_pid_dbg("init hvac_ewc \r\n");  
    while(done==false){
        // ====== 环境状态 ======
        std::vector<float> observation = env.get_state();

        // ====== 模型推理 ======
        auto [action_probs, value] = ewcppoModel.predictFull(observation);
        
        // 选一个动作 (argmax)
        //int action = std::distance(action_probs.begin(),
        //                           std::max_element(action_probs.begin(), action_probs.end()));

        // ====== 环境交互 ======
        auto result = env.step(plant_action, env.default_params);
        std::vector<float> new_state = result.state;
        float reward = result.reward;
        done = result.done;
 
        // ====== 优势计算 ======
        std::vector<float> advantages(action_probs.size(), 0.0f);
        for (size_t i = 0; i < action_probs.size(); ++i) {
            // 这里简单用 reward 代替 TD 误差，可以换成 GAE
            advantages[i] = reward - value;
        }

        // // ====== health_result ======
        std::vector<float> health_result = {
            result.reward, 
            result.flower_prob
        };

        // ====== 计算梯度 + EWC 更新 ======
        std::vector<float> grads;
        ewcppoModel.calculateGradients(observation, action_probs,
                                       advantages, health_result,
                                       old_action_probs, grads);

        ewcppoModel.continualLearningEWC(grads);

        // ====== 更新 old_action_probs ======
        old_action_probs = action_probs;

        // ====== 打印调试 ======
        //printf("Action: %d, Reward: %.3f\n", plant_action, reward);
        printf("Action probs: ");
        for (auto v : action_probs) printf("%.3f ", v);
        printf("\n");

        ESP_LOGI(TAG, "Gradients: ");
        for (auto& grad : grads) printf("%.3f ", grad);
        printf("\n");

        // 如果环境结束，重置
        // if (done) {
        //     env.reset(); 
        // }

        vTaskDelay(pdMS_TO_TICKS(5000)); // 每 5 秒交互一次
    }
    return done;
}

