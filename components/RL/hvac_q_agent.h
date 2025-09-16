#pragma once


#ifndef HVAC_Q_AGENT_H
#define HVAC_Q_AGENT_H

#include <stdio.h>

// 假设状态维度 5：health + ac_target + dehum_target + light + humidity
#define STATE_DIM 5
//#define NUM_FLASK_GET_TASK 2
//#define NUM_FLASK_PUT_TASK 1

enum enum_flask_state{
    SPIFFS_MODEL_EMPTY=0, 
    SPIFFS_MODEL_SAVED=1, 
    SPIFFS_MODEL_READED=2,
    SPIFFS_MODEL_ERR=3,
};

enum enum_flask_get_state {
    SPIFFS_DOWN_LOAD_MODEL = 0, 
    MODEL_BIN_PPO_MD5 = 1,
    FLASK_STATE_GET_COUNT // This automatically becomes 2, useful for array sizing
};

 
enum enum_flask_put_state{
     
    INIT_EXPORTER=0, 
    FLASK_STATE_PUT_COUNT
};
// ======= WiFi 與 OTA 配置 ======= 
#if 1
    
    #define HTTP_GET_MODEL_JSON_URL "http://192.168.30.132:5001/api/model?name=esp32_policy"
    #define OTA_SERVER_URL          "http://192.168.30.132:5001"   // 换成你的 PC IP
#else
    
    #define HTTP_GET_MODEL_JSON_URL "http://192.168.30.132:5001/api/model?name=esp32_policy"
    #define OTA_SERVER_URL          "http://192.168.30.132:5001"   // 换成你的 PC IP
#endif
#define  spiffs_model_path  "/spiffs/esp32_optimized_model.tflite" 
#define  spiffs_ppo_model_bin_path  "/spiffs/ppo_model.bin" 
#define  post_data   "{\"model_path\": \"./saved_models/ppo_policy_actor\"}" 
#if 0
const int STATE_SIZES[STATE_DIM] = {2, 2, 2, 3, 3}; // 例：health 0/1, light 0/1/2 等

// 计算总状态数
#define TOTAL_STATES (2*2*2*3*3) // 72
#define N_ACTIONS 4  // [00,01,10,11]

// Q-table 固化 (示例随机值，可用 Python 导出)
const float Q_TABLE[TOTAL_STATES][N_ACTIONS] = {
    /* 填充从 Python 导出的 Q-values */
};

// 状态哈希函数：把状态映射到 Q_TABLE 行
int state_to_index(int state[STATE_DIM]) {
    int idx = 0;
    int mult = 1;
    for(int i=STATE_DIM-1; i>=0; i--){
        idx += state[i] * mult;
        mult *= STATE_SIZES[i];
    }
    return idx; // 返回 Q_TABLE 的行索引
}

// 选择动作
int select_action(int state[STATE_DIM]) {
    int row = state_to_index(state);
    if(row < 0 || row >= TOTAL_STATES) return 0; // 出界保护

    int best_action = 0;
    float max_q = Q_TABLE[row][0];
    for(int a=1;a<N_ACTIONS;a++){
        if(Q_TABLE[row][a] > max_q){
            max_q = Q_TABLE[row][a];
            best_action = a;
        }
    }
    return best_action;
}

// 转换为 AC/Dehum 控制位
void get_action_bits(int action_id, int action_bits[2]) {
    action_bits[0] = action_id / 2; // AC
    action_bits[1] = action_id % 2; // Dehum
}
#endif




#endif // HVAC_Q_AGENT_H
