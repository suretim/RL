#include "PlantHVACEnv.h"
#include "HVACEncoder.h"
#include "PrototypeClassifierSimple.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#include "ml_pid.h"

ModeParam m_range_params;
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
    return {static_cast<float>(health), temp, humid,soil,light,co2,vpd};
} 

int PlantHVACEnv::_get_state_cnt() const {
    return static_cast<int>(_get_state().size());
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
PlantHVACEnv::StepResult PlantHVACEnv::step(const std::array<int,ACTION_CNT>& action,
                                            const std::map<std::string,float>& params)
{
    StepResult result;
    struct st_bp_pid_th    v_env_th = {0};
    float dev_heat      = action[0]>0?0.5f:0.0f;
    float dev_ac        = action[1]>0?-0.5f:0.2f;
    float dev_humid     = action[2]>0?0.05f:-0.02f;
    float dev_dehumi    = action[3]>0?-0.05f:0.0f;
    float dev_waterpump = action[4]>0?1:0;
    float dev_light     = action[5]>0?0.1f : 0.0f;
    float dev_co2       = action[6]>0?0.1f : 0.0f;
    float dev_pump      = action[7]>0?0.1f : 0.0f;
    m_range_params = mode_params[plant_mode]; 
    if(true_env){
        v_env_th   = bp_pid_th;
    }
    else{
        // --- 环境动力学 ---
        float light_effect = get_param(params, "light_penalty") * dev_light ;
        float co2_effect   = get_param(params, "co2_penalty")   * dev_co2 ;

        v_env_th.t_feed  += (dev_ac  +  dev_heat );
        v_env_th.h_feed  += (dev_humid  + dev_dehumi);//(dev_heat==1?-0.03f:0.0f) + (dev_dehumi==1?-0.05f:0.0f);
        v_env_th.h_feed  += (dev_heat>0.0f?-0.03f:0.0f);
        // 使用方法
        float light_noise = rand_normal(0.0f, 20.0f);
        float co2_noise   = rand_normal(0.0f, 10.0f);
        v_env_th.l_feed   += (light_effect+light_noise);
        v_env_th.c_feed   += (co2_effect+co2_noise);
        v_env_th.v_feed = calc_vpd(temp, humid);
    } 

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
    int optimal = 0;
    if(v_env_th.t_feed >=m_range_params.temp_range.first  && temp  <=m_range_params.temp_range.second) optimal++;
    if(v_env_th.h_feed >=m_range_params.humid_range.first && humid <=m_range_params.humid_range.second) optimal++;
    if(v_env_th.v_feed >=m_range_params.vpd_range.first   && vpd   <=m_range_params.vpd_range.second) optimal++;
    if(v_env_th.l_feed >=m_range_params.light_range.first && light <=m_range_params.light_range.second) optimal++;
    if(v_env_th.c_feed >=m_range_params.co2_range.first   && co2   <=m_range_params.co2_range.second) optimal++;
    float vpd_target   =(m_range_params.vpd_range.first  +m_range_params.vpd_range.second)/2.0;
    float light_target =(m_range_params.light_range.first+m_range_params.light_range.second)/2.0;
    float co2_target   =(m_range_params.co2_range.first  +m_range_params.co2_range.second)/2.0;
    health = (optimal>=4?0:(optimal>=2?1:2)); 

    // --- Reward ---
    float health_reward = (health==0?2.0f:(health==1?0.5f:-1.0f));
    float energy_cost = (params.count("energy_penalty")?params.at("energy_penalty"):0.1f)*(dev_ac+dev_humid+dev_heat+dev_dehumi);

    int switch_diff = 0;
    for(int i=0;i<4;i++) switch_diff += std::abs(action[i]-prev_action[i]);
    float switch_penalty = (params.count("switch_penalty_per_toggle")?params.at("switch_penalty_per_toggle"):0.05f)*switch_diff;

    float vpd_reward   = -std::abs(v_env_th.v_feed -vpd_target) *(params.count("vpd_penalty")?params.at("vpd_penalty"):0.5f);
    float light_reward = -std::abs(v_env_th.l_feed -light_target)*(params.count("light_penalty")?params.at("light_penalty"):0.5f);
    float co2_reward   = -std::abs(v_env_th.c_feed -co2_target)  *(params.count("co2_penalty")?params.at("co2_penalty"):0.5f);

    float flower_env_penalty = 0.0f;
    if(flower_prob>0.5f && plant_mode=="flowering") {
        if(!(humid>=m_range_params.humid_range.first && humid<=m_range_params.humid_range.second)) 
            flower_env_penalty -= (params.count("flower_humi_penalty")?params.at("flower_humi_penalty"):0.1f);
        if(!(temp>=m_range_params.temp_range.first && temp<=m_range_params.humid_range.second))    
            flower_env_penalty -= (params.count("flower_temp_penalty")?params.at("flower_temp_penalty"):0.1f);
    }

    float soft_label_bonus = 0.0f;
    if(plant_mode=="flowering") 
        soft_label_bonus = flower_prob*(params.count("soft_label_bonus")?params.at("soft_label_bonus"):0.5f);
    else if(plant_mode=="seeding") 
        soft_label_bonus = soft_label[1]*(params.count("soft_label_bonus")?params.at("soft_label_bonus"):0.5f);
    else //growing
        soft_label_bonus = soft_label[0]*(params.count("soft_label_bonus")?params.at("soft_label_bonus"):0.5f);

    float learning_reward = 0.0f;
    if (true_label != -1) {
        int pred_class = std::distance(
            soft_label.begin(),
            std::max_element(soft_label.begin(), soft_label.end())
        );
        learning_reward = (pred_class == true_label ? 0.5f : -0.3f);
    } 
    bool is_flowering = (flower_prob > 0.5f);  // 以 soft_label 作为阶段判定条件

    if(is_flowering) {
        // 将 humid 从 [0,1] 转换成 [%]
        float humid_pct = humid * 100.0f;

        // 湿度目标区间：早期/盛花/收获前
        float target_min_RH = 45.0f, target_max_RH = 55.0f;
        if (flower_prob > 0.6f && flower_prob <= 0.75f) {
            target_min_RH = 45.0f; target_max_RH = 55.0f;
        } else if (flower_prob > 0.75f && flower_prob <= 0.9f) {
            target_min_RH = 40.0f; target_max_RH = 50.0f;
        } else if (flower_prob > 0.9f) {
            target_min_RH = 35.0f; target_max_RH = 45.0f;
        }

        // 湿度 penalty
        if (humid_pct < target_min_RH || humid_pct > target_max_RH) {
            float diff = (humid_pct < target_min_RH) ? (target_min_RH - humid_pct) : (humid_pct - target_max_RH);
            flower_env_penalty -= get_param(params,"flower_humi_penalty") * diff;
        }

        // 温度 penalty (理想 20–26 ℃)
        if (temp < 20.0f || temp > 26.0f) {
            float diff = (temp < 20.0f) ? (20.0f - temp) : (temp - 26.0f);
            flower_env_penalty -= get_param(params,"flower_temp_penalty") * diff;
        }
    }
    
    
    vpd   = pid_map(v_env_th.v_feed,  m_range_params.vpd_range.first,   m_range_params.vpd_range.second,   0, 1);
    temp  = pid_map(v_env_th.t_feed,  m_range_params.temp_range.first,  m_range_params.temp_range.second,  0, 1);
    humid = pid_map(v_env_th.h_feed,  m_range_params.humid_range.first, m_range_params.humid_range.second, 0, 1);
    light = pid_map(v_env_th.l_feed,  m_range_params.light_range.first, m_range_params.light_range.second, 0, 1);
    co2   = pid_map(v_env_th.c_feed,  m_range_params.co2_range.first,   m_range_params.co2_range.second,   0, 1);
    
    
    float reward = health_reward 
                 - energy_cost 
                 - switch_penalty 
                 + vpd_reward 
                 + light_reward 
                 + co2_reward 
                 + flower_env_penalty 
                 + soft_label_bonus 
                 + learning_reward;
 
     

    // --- 更新状态 ---
    prev_action = action;
    t++;
    bool done = t>=seq_len;
    
    
    // --- 填充结果 ---
    result.state = _get_state();
    result.reward = reward;
    result.done = done;
    result.latent_soft_label = soft_label;
    result.flower_prob = flower_prob;
    result.temp  = temp;
    result.humid = humid;
    result.light = light;
    result.co2   = co2;
    result.vpd   = vpd;

    return result;
}
