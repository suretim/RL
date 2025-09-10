#ifndef HVAC_Q_AGENT_H
#define HVAC_Q_AGENT_H

#include <stdio.h>

// 假设状态维度 5：health + ac_target + dehum_target + light + humidity
#define STATE_DIM 5
#define NUM_FLASK_TASK 3

enum flask_state{
    INIT_EXPORTER=0,
    DOWN_LOAD_MODEL=1,
    DOWN_LOAD_MODEL_OTA=2,
};
// ======= WiFi 與 OTA 配置 ======= 
#if 0
    const char* base_url ="192.168.68.237:5000";
    const char* check_url = "http://192.168.68.237:5000/api/check-update/device001/1.0.0";
    const char* download_url = "http://192.168.68.237:5000/api/download-update";
    //#define OTA_URL "http://192.168.68.237:5000/api/bin-update"
#else
    #define BASE_URL  "192.168.30.132" 
    #define BASE_PORT "5000"
    //const char* check_url = "http://192.168.0.57:5000/api/check-update/device001/1.0.0";
    //const char* download_url = "http://192.168.0.57:5000/api/download-update";
    //const char* download_bin_url = "http://192.168.0.57:5000/api/bin-update";
    //#define OTA_BIN_UPDATE_URL "http://192.168.0.57:5000/api/bin-update"
#endif
//#define LOCAL_MODEL_FILE "/spiffs/ppo_model.bin"
// 每个维度可能取值数

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
