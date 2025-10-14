//PlantHVACEnv.h
#ifndef PLANTHVACENV_H
#define PLANTHVACENV_H

#include <vector>
#include <array>
#include <map>
#include <string>
#include <functional>
#include "version.h"
#include "math.h"
#include "ml_pid.h" 
// struct ModeParam {
//     std::pair<float,float> temp_range;
//     std::pair<float,float> humid_range;
//     std::pair<float,float> soil_range;
//     std::pair<float,float> light_range;
//     std::pair<float,float> co2_range;
//     std::pair<float,float> ph_range;
//     std::pair<float,float> vpd_range;
    
//     float soft_label_bonus;   
//     float penalty;   
// };

// 前向声明
class HVACEncoder;
class PrototypeClassifierSimple;
              // 存储每步的奖励
class PlantHVACEnv {
public:
    using SeqFetcher = std::function<std::vector<std::vector<float>>(int t)>;

    struct StepResult {
        std::vector<float> state;
        float reward;
        bool done;
        std::vector<float> latent_soft_label;
        float flower_prob;
        // float temp;
        // float humid;
        // float soil;
        // float light;
        // float co2;
        // float ph;
        // float vpd; 
        float itm_heat      =  0.01f;        
        float itm_ac        = -0.005f;
        float itm_humid     =  0.0002f;
        float itm_dehumi    = -0.0005f;
        float itm_waterpump = 1.0;
        float itm_light     = 20.f;
        float itm_co2       = 50.0f;
        float itm_pump      = 1.1f;
        
        StepResult() : reward(0.0f), done(false), flower_prob(0.0f),
                       itm_heat(0.01f), itm_ac(-0.005f), itm_humid(0.0002f), itm_dehumi(-0.0005f),
                       itm_waterpump(1.0f),itm_light(20.0f), itm_co2(50.0f), itm_pump(1.1f)   {}
    };
    std::map<std::string, ModeParam> mode_params= {
            {"seeding",  ModeParam{{24,30},{0.50f,0.70f},{0.20f, 0.30f},{200,400} ,{400,600} ,{5.5f,6.2f},{0.4f,0.8f}, 0.1f, -0.05f}},
            {"growing",  ModeParam{{22,28},{0.40f,0.70f},{0.25f, 0.35f},{300,600} ,{400,800} ,{5.8f,6.5f},{0.8f,1.2f}, 0.2f, -0.1f}},
            {"flowering",ModeParam{{20,26},{0.40f,0.60f},{0.30f, 0.40f},{500,800} ,{600,1000},{5.8f,6.3f},{1.0f,1.5f}, 0.3f, -0.15f}},
            {"testing"  ,ModeParam{{28,32},{0.50f,0.70f},{0.30f, 0.40f},{500,800} ,{600,1000},{5.8f,6.3f},{1.0f,1.5f}, 0.3f, -0.15f}},
            {"limit",    ModeParam{{18,37},{0.10f,0.80f},{0.10f, 0.70f},{100,1000},{200,1200},{5.0f,6.8f},{0.1f,2.8f}, 0.1f, -0.05f}}
        }; 
    uint32_t rng_seed = 12345;
    std::string plant_mode = "testing";
    int true_label= -1; 
    
    float rand_uniform() {
        // 简单 LCG 随机数生成器，返回 0~1
        rng_seed = 1664525 * rng_seed + 1013904223;
        return (rng_seed & 0xFFFFFF) / float(0x1000000);
    }
    void update_dynamics(float t_feed,StepResult result,const std::array<int,ACTION_CNT>& action,float measured_temp_next);
    float rand_normal(float mean, float stddev) {
        // 简单 Box-Muller
        float u1 = rand_uniform();
        float u2 = rand_uniform();
        float z0 = sqrt(-2.0f * log(u1)) * cos(2*M_PI*u2);
        return z0 * stddev + mean;
    } 
     
    PlantHVACEnv() {
       plant_limit_params = mode_params["limit"];  
    }
private:
    HVACEncoder* encoder;
    PrototypeClassifierSimple* proto_cls;

    int seq_len;
    int n_features;
    int latent_dim;
    float temp_init;
    float humid_init;

    // float temp;
    // float humid;
    // float soil;
    // float light;
    // float co2;
    // float vpd; // 蒸汽压差 (kPa)
    bool done;
    int health;
    int t;
    std::array<int,ACTION_CNT> prev_action; 
    SeqFetcher seq_fetcher; 

public:
    std::map<std::string,float> default_params = {
        // --- 基础 ---
        {"energy_penalty", 0.05f},                // 能耗惩罚
        {"switch_penalty_per_toggle", 0.1f},      // 开关切换惩罚
        {"light_penalty", 0.02f},                 // 光照过度惩罚
        {"co2_penalty", 0.02f},                   // CO2 过度惩罚

        // --- VPD ---
        
        {"vpd_penalty", 0.5f},                    // 偏差惩罚系数

        // --- 生长阶段 ---
        {"flowering_start_step", 500.0f},         // 从 step=500 开始进入开花期

        // --- 开花期专用 penalty ---
        {"flower_temp_penalty", 0.3f},            // 温度偏差惩罚系数
        {"flower_humi_penalty", 0.5f},            // 湿度偏差惩罚系数
    }; 
    PlantHVACEnv(int seq_len = 20, int n_features = 3, float temp_init = 25.0f,
                 float humid_init = 0.5f, int latent_dim = 64);
    ~PlantHVACEnv();

    void set_seq_fetcher(SeqFetcher fetcher);

    StepResult step(const std::array<int,ACTION_CNT>& action,
                    const std::map<std::string,float>& params = {});

    std::vector<float> get_state() const;
    void update_prototypes(const std::vector<std::vector<float>>& features,
                           const std::vector<int>& labels);
    void reset() {
            // 将环境状态重置为初始值
            v_env_th.t_feed   = (mode_params["seeding"].temp_range.first  +mode_params["seeding"].temp_range.second )/ 2.0f ;   //22.0f;  // 设定初始温度为22.0度
            v_env_th.h_feed   =(mode_params["seeding"].humid_range.first +mode_params["seeding"].humid_range.second )/ 2.0f ;   //;     // 设定初始湿度为50%
            v_env_th.w_feed   =(mode_params["seeding"].water_range.first +mode_params["seeding"].water_range.second )/ 2.0f ;   //;     
            v_env_th.l_feed   =(mode_params["seeding"].light_range.first +mode_params["seeding"].light_range.second )/ 2.0f ;   //; // 设定初始光照为300lux
            v_env_th.c_feed   =(mode_params["seeding"].co2_range.first   +mode_params["seeding"].co2_range.second )/ 2.0f ;   //; // 设定初始CO2浓度为400ppm
            v_env_th.p_feed   =(mode_params["seeding"].ph_range.first   +mode_params["seeding"].ph_range.second )/ 2.0f ;   //; // 设定初始CO2浓度为400ppm
            v_env_th.v_target =(mode_params["seeding"].vpd_range.first   +mode_params["seeding"].vpd_range.second )/ 2.0f ;   //;  
            done = false;         // 重置任务结束标志
            
          
        };
private:
    std::vector<float> _get_state() const;
    int _get_state_cnt() const;
    float get_param(const std::map<std::string,float>& params, const std::string& key) const;
    float calc_vpd(float temp, float humid) const;
};

#endif // PLANTHVACENV_H
