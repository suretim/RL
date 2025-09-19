#include "PlantHVACEnv.h"
#include "HVACEncoder.h"
#include "PrototypeClassifierSimple.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#include "ml_pid.h"
// 构建编码器
HVACEncoder* build_hvac_encoder(int seq_len, int n_features, int latent_dim) {
    return new HVACEncoder(seq_len, n_features, latent_dim);
}

// ==== 构造函数 ====
PlantHVACEnv::PlantHVACEnv(int seq_len_, int n_features_, float temp_init_,
                           float humid_init_, int latent_dim_)
    : seq_len(seq_len_), n_features(n_features_), latent_dim(latent_dim_),
      temp_init(temp_init_), humid_init(humid_init_),
      temp(temp_init_), humid(humid_init_), health(0), t(0), prev_action({0,0,0,0})
{
    encoder = build_hvac_encoder(seq_len, n_features, latent_dim);
    proto_cls = new PrototypeClassifierSimple(3, latent_dim, 0.1f);
}

// ==== 析构函数 ====
PlantHVACEnv::~PlantHVACEnv() {
    delete encoder;
    delete proto_cls;
}

// ==== 设置序列获取器 ====
void PlantHVACEnv::set_seq_fetcher(SeqFetcher fetcher) {
    seq_fetcher = fetcher;
}

// ==== 获取状态 ====
std::vector<float> PlantHVACEnv::_get_state() const {
    return {static_cast<float>(health), temp, humid};
}

std::vector<float> PlantHVACEnv::get_state() const {
    return _get_state();
}

// ==== 更新原型 ====
void PlantHVACEnv::update_prototypes(const std::vector<std::vector<float>>& features,
                                    const std::vector<int>& labels) {
    proto_cls->update_prototypes(features, labels);
}

// ==== VPD 计算 ====
float PlantHVACEnv::calc_vpd(float temp_, float humid_) const {
    float sat_vp = 0.6108f * std::exp(17.27f * temp_ / (temp_ + 237.3f));
    float vp = sat_vp * humid_;
    return sat_vp - vp;
}

// ==== 获取参数值 ====
float PlantHVACEnv::get_param(const std::map<std::string,float>& params, const std::string& key) const {
    auto it = params.find(key);
    if(it != params.end()) return it->second;
    auto def_it = default_params.find(key);
    if(def_it != default_params.end()) return def_it->second;
    return 0.0f;
}
//extern float pid_map(float x, float in_min, float in_max, float out_min, float out_max);
// ==== Step 函数 ====
PlantHVACEnv::StepResult PlantHVACEnv::step(const std::array<int,PORT_CNT>& action,
                                            const std::map<std::string,float>& params)
{
    StepResult result;

    int ac = action[0];
    int humi = action[1];
    int heat = action[2];
    int dehumi = action[3];

    // --- 环境动力学 ---
    
    float light_effect = get_param(params, "light_penalty") * (action[4] == 1 ? 0.1f : 0.0f);
    float co2_effect = get_param(params, "co2_penalty") * (action[5] == 1 ? 0.1f : 0.0f);

    temp += (ac==1 ? -0.5f : 0.2f) + (heat==1 ? 0.5f : 0.0f);
    humid += (humi==1 ? 0.05f : -0.02f) + (heat==1 ? -0.03f : 0.0f) + (dehumi==1 ? -0.05f : 0.0f);

    temp = std::max(15.0f, std::min(temp, 35.0f));
    humid = std::max(0.0f, std::min(humid, 1.0f));

    health = (22.0f <= temp && temp <= 28.0f && 0.4f <= humid && humid <= 0.7f) ? 0 : 1;

    // --- 获取序列 ---
    std::vector<std::vector<float>> seq_input;
    if(seq_fetcher) {
        seq_input = seq_fetcher(t);
    } else {
        seq_input = std::vector<std::vector<float>>(seq_len, std::vector<float>(n_features, 0.1f*(t+1)));
    }

    // --- 编码和原型分类 ---
    std::vector<float> z = encoder->encode(seq_input, false);
    std::vector<float> soft_label = (*proto_cls)(z);
    float flower_prob = soft_label.size()>2 ? soft_label[2] : 0.0f;

    // --- Reward 计算 ---
    float health_reward = (health==0 ? 1.0f : -1.0f);

    float action_sum = static_cast<float>(ac+humi+heat+dehumi);
    float energy_cost = get_param(params,"energy_penalty") * action_sum;

    float switch_diff = 0.0f;
    for(int i=0;i<4;++i) switch_diff += std::abs(action[i]-prev_action[i]);
    float switch_penalty = get_param(params,"switch_penalty_per_toggle")*switch_diff;

    float vpd_current = calc_vpd(temp, humid);
    float vpd_reward = -std::abs(vpd_current - get_param(params,"vpd_target"))*get_param(params,"vpd_penalty");

    float reward = health_reward - energy_cost - switch_penalty + flower_prob*vpd_reward;

    // --- 更新状态 ---
    prev_action = action;
    t++;
    bool done = t>=seq_len;


    float v_feed  = pid_map((bp_pid_th.v_feed+vpd_current)/2.0,  c_pid_vpd_min, c_pid_vpd_max, 0, 1);
    float t_feed  = pid_map((bp_pid_th.t_feed+temp)/2.0,  c_pid_temp_min, c_pid_temp_max, 0, 1);
    float h_feed  = pid_map((bp_pid_th.h_feed+humid)/2.0,  c_pid_humi_min, c_pid_humi_max, 0, 1);
    float l_feed  = pid_map((bp_pid_th.l_feed)/1.0,  c_pid_light_min, c_pid_light_max, 0, 1);
    float c_feed  = pid_map((bp_pid_th.c_feed)/1.0,  c_pid_co2_min, c_pid_co2_max, 0, 1);
    // --- 填充结果 ---
    result.state = _get_state();
    result.reward = reward;
    result.done = done;
    result.latent_soft_label = soft_label;
    result.flower_prob = flower_prob;
    result.temp = t_feed;
    result.humid = h_feed;
    result.light = l_feed;
    result.co2 =   c_feed;
    result.vpd = v_feed;

    return result;
}
