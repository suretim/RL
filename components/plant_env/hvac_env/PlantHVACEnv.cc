#include "PlantHVACEnv.h"
#include "HVACEncoder.h"
#include "PrototypeClassifierSimple.h"
#include <algorithm>
#include <cmath>
#include <iostream> 

#include "ml_pid.h"
#include "ni_debug.h"

ModeParam plant_range_params;
ModeParam plant_limit_params;
StepResult plant_tres;
// 构建编码器
HVACEncoder* build_hvac_encoder(int seq_len, int n_features, int latent_dim) {
    return new HVACEncoder(seq_len, n_features, latent_dim);
}

// ==== 构造函数 ====
PlantHVACEnv::PlantHVACEnv(int seq_len_, int n_features_, float temp_init_,
                           float humid_init_, int latent_dim_)
    : seq_len(seq_len_), n_features(n_features_), latent_dim(latent_dim_),
      temp_init(temp_init_), humid_init(humid_init_), health(0), t(0), prev_action({0,0,0,0})
{
    encoder = build_hvac_encoder(seq_len, n_features, latent_dim);
    proto_cls = new PrototypeClassifierSimple(3, latent_dim, 0.1f);
    plant_limit_params = mode_params["limit"]; 
    plant_range_params = mode_params[plant_mode]; 

    
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
     
    return {static_cast<float>(health), v_env_th.t_feed, v_env_th.h_feed,v_env_th.w_feed,v_env_th.l_feed,v_env_th.c_feed,v_env_th.p_feed};
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
// float PlantHVACEnv::calc_vpd(float temp_, float humid_) const {
//     float sat_vp = 0.6108f * std::exp(17.27f * temp_ / (temp_ + 237.3f));
//     float vp = sat_vp * humid_;
//     return sat_vp - vp;
// }

// ==== 获取参数值 ====
float PlantHVACEnv::get_param(const std::map<std::string,float>& params, const std::string& key) const {
    auto it = params.find(key);
    if(it != params.end()) return it->second;
    auto def_it = default_params.find(key);
    if(def_it != default_params.end()) return def_it->second;
    return 0.0f;
}
void PlantHVACEnv::update_dynamics(float t_feed,StepResult result,const std::array<int,ACTION_CNT>& action,float measured_temp_next) {
    float predicted_temp_next = v_env_th.t_feed + t_feed;
    float error = measured_temp_next - predicted_temp_next;

    // Gradient-like update
    result.itm_heat += 0.0001f * error * action[0];
    result.itm_ac   += 0.0001f * error * action[1];
}
//extern float pid_map(float x, float in_min, float in_max, float out_min, float out_max);
StepResult PlantHVACEnv::step(const std::array<int,ACTION_CNT>& action,
                                            const std::map<std::string,float>& params)
{
    StepResult result; //    std::cout << "step " << t << std::endl;

    float dev_heat      = 0.1f;
    float dev_ac        = -0.05f;
    float dev_humid     = 0.5f;
    float dev_dehumi    = -0.5f;
    float dev_waterpump = 1.0;
    float dev_light     = 20.f;
    float dev_co2       = 50.0f;
    float dev_pump      = 1.1f;

    float temp_noise = rand_normal(0.0f , 0.10f);
    float humid_noise = rand_normal(0.0f, 0.10f);
    plant_range_params = mode_params[plant_mode]; 
    
        // --- 环境动力学 ---
        float light_effect = get_param(params, "light_penalty") * dev_light ;
        float co2_effect   = get_param(params, "co2_penalty")   * dev_co2 ;
        float itm_heat_fuse_humid  =  -0.001f;
        dev_heat      = action[0]*result.itm_heat;
        dev_ac        = action[1]*result.itm_ac ;
        dev_humid     = (action[2]*result.itm_humid + dev_heat *itm_heat_fuse_humid );
        dev_dehumi    = action[3]*result.itm_dehumi ;
        dev_waterpump = action[4]*result.itm_waterpump;
        dev_light     = action[5]*result.itm_light;
        dev_co2       = action[6]*result.itm_co2;
        dev_pump      = action[7]*result.itm_pump;
        if(true_env){
            v_env_th=r_env_th;
        }
        else{ 
            v_env_th.t_outside= 27.0f;//(v_env_th.t_feed- 32) * 5 / 9;
            v_env_th.h_outside= 0.55f;  
            //v_env_th.t_feed  += (dev_heat  + dev_ac     + (v_env_th.t_outside-v_env_th.t_feed)*0.05   ) ;
            //v_env_th.h_feed  += (dev_humid + dev_dehumi + (v_env_th.h_outside-v_env_th.h_feed)*0.05);//(dev_heat==1?-0.03f:0.0f) + (dev_dehumi==1?-0.05f:0.0f);
            v_env_th.t_feed  += (dev_heat  + dev_ac       );
            v_env_th.h_feed  += (dev_humid + dev_dehumi   );//(dev_heat==1?-0.03f:0.0f) + (dev_dehumi==1?-0.05f:0.0f);
        }
        if(r_env_th.t_feed>0.0 && r_env_th.h_feed>0.0)
        {
            float predicted_temp_next = v_env_th.t_feed + (dev_heat + dev_ac);
            float t_error = r_env_th.t_feed - predicted_temp_next;

            // Gradient-like update
            result.itm_heat   += 0.001f * t_error * action[0];
            result.itm_ac     += 0.001f * t_error * action[1];
            float predicted_humid_next = v_env_th.h_feed + (dev_humid + dev_dehumi);
            float h_error = r_env_th.h_feed - predicted_humid_next;
            result.itm_humid  += 0.001f * h_error * action[0];
            result.itm_dehumi += 0.001f * h_error * action[1];
        }
        // 使用方法
        float light_noise = rand_normal(0.0f, 20.0f);
        float co2_noise   = rand_normal(0.0f, 10.0f);
        v_env_th.l_feed   += (light_effect+light_noise);
        v_env_th.c_feed   += (co2_effect+co2_noise);
        //v_env_th.v_feed = calc_vpd(temp, humid);
     
    bp_pid_dbg(" PlantHVACEnv step =t,h(%.3f,%.3f  %.3f,%.3f  %.3f,%.3f,%.5f,%.5f)\r\n", 
        v_env_th.t_feed, r_env_th.t_feed,
        v_env_th.h_feed,r_env_th.h_feed,
        result.itm_heat ,result.itm_ac ,result.itm_humid ,result.itm_dehumi
    );
    //bp_pid_dbg("PlantHVACEnv StepResult");
    float flower_prob=0.0;
    std::vector<float> soft_label={0.0};
    if(t==seq_len-1) {
        // --- 获取序列 ---
        std::vector<std::vector<float>> seq_input;
        if(seq_fetcher) {
            seq_input = seq_fetcher(t);
        } else {
            seq_input = std::vector<std::vector<float>>(seq_len, std::vector<float>(n_features, 0.1f*(t+1)));
        }

        // --- 编码和原型分类 ---
        std::vector<float> z = encoder->encode(seq_input, false);
        soft_label = (*proto_cls)(z);
        flower_prob = soft_label.size()>2 ? soft_label[2] : 0.0f; 
        t=0;
    }
     
    int optimal = 0;
    if(v_env_th.t_feed >=plant_range_params.temp_range.first  && v_env_th.t_feed  <=plant_range_params.temp_range.second) optimal++;
    if(v_env_th.h_feed >=plant_range_params.humid_range.first && v_env_th.h_feed <=plant_range_params.humid_range.second) optimal++;
    if(v_env_th.v_feed >=plant_range_params.vpd_range.first   && v_env_th.v_feed   <=plant_range_params.vpd_range.second) optimal++;
    if(v_env_th.l_feed >=plant_range_params.light_range.first && v_env_th.l_feed <=plant_range_params.light_range.second) optimal++;
    if(v_env_th.c_feed >=plant_range_params.co2_range.first   && v_env_th.c_feed   <=plant_range_params.co2_range.second) optimal++;
    float vpd_target   =(plant_range_params.vpd_range.first  +plant_range_params.vpd_range.second)/2.0;
    float light_target =(plant_range_params.light_range.first+plant_range_params.light_range.second)/2.0;
    float co2_target   =(plant_range_params.co2_range.first  +plant_range_params.co2_range.second)/2.0;
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
        if(!(v_env_th.h_feed>=plant_range_params.humid_range.first && v_env_th.h_feed<=plant_range_params.humid_range.second)) 
            flower_env_penalty -= (params.count("flower_humi_penalty")?params.at("flower_humi_penalty"):0.1f);
        if(!(v_env_th.t_feed>=plant_range_params.temp_range.first && v_env_th.t_feed<=plant_range_params.humid_range.second))    
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
        float humid_pct = v_env_th.h_feed * 100.0f;

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
        if (v_env_th.t_feed < 20.0f || v_env_th.t_feed > 26.0f) {
            float diff = (v_env_th.t_feed < 20.0f) ? (20.0f - v_env_th.t_feed) : (v_env_th.t_feed - 26.0f);
            flower_env_penalty -= get_param(params,"flower_temp_penalty") * diff;
        }
    }
    
    
    // v_env_th.f[0][ENV_L] = pid_map(v_env_th.l_feed, plant_limit_params.light_range.first, plant_limit_params.light_range.second, 0, 1);
    // v_env_th.f[0][ENV_C] = pid_map(v_env_th.c_feed, plant_limit_params.co2_range.first  , plant_limit_params.co2_range.second,   0, 1);
    // v_env_th.f[0][ENV_W] = pid_map(v_env_th.w_feed, plant_limit_params.water_range.first, plant_limit_params.water_range.second,   0, 1);
    
	// double temp  =v_env_th.f[0][ENV_T] ;	
	// double humid =v_env_th.f[0][ENV_H] ;	
	// double vpd   =v_env_th.f[0][ENV_V] ;
	// double light =v_env_th.f[0][ENV_L];
    // double co2   =v_env_th.f[0][ENV_C];
    // double soil  =v_env_th.f[0][ENV_W];
    
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
    bool done =  reward<=-25.0f;
    

    // --- 填充结果 ---
    result.state = _get_state();
    result.reward = reward;
    result.done = done;
    result.latent_soft_label = soft_label;
    result.flower_prob = flower_prob;
    bp_pid_dbg("reward = %f done=%d\r\n",reward,done);
     

    return result;
}
