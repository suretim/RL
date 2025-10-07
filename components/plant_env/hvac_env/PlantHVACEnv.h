#ifndef PLANTHVACENV_H
#define PLANTHVACENV_H

#include <vector>
#include <array>
#include <map>
#include <string>
#include <functional>
#include "version.h"
#include "math.h"
struct ModeParam {
    std::pair<float,float> temp_range;
    std::pair<float,float> humid_range;
    std::pair<float,float> soil_range;
    std::pair<float,float> light_range;
    std::pair<float,float> co2_range;
    std::pair<float,float> vpd_range;
    
    float soft_label_bonus;   
    float penalty;   
};
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
        float temp;
        float humid;
        float soil;
        float light;
        float co2;
        float vpd; 
        
        StepResult() : reward(0.0f), done(false), flower_prob(0.0f),
                       temp(0.0f), humid(0.0f),soil(0.0f), light(0.0f), co2(0.0f), vpd(0.0f)  {}
    };
    std::map<std::string, ModeParam> mode_params;
    uint32_t rng_seed = 12345;
    std::string plant_mode = "growing";
    int true_label= -1; 
    bool true_env= false;
    float rand_uniform() {
        // 简单 LCG 随机数生成器，返回 0~1
        rng_seed = 1664525 * rng_seed + 1013904223;
        return (rng_seed & 0xFFFFFF) / float(0x1000000);
    }

    float rand_normal(float mean, float stddev) {
        // 简单 Box-Muller
        float u1 = rand_uniform();
        float u2 = rand_uniform();
        float z0 = sqrt(-2.0f * log(u1)) * cos(2*M_PI*u2);
        return z0 * stddev + mean;
    }

     
    PlantHVACEnv() {
        mode_params = {
            {"growing",  ModeParam{{22,28},{50.0f,70.0f},{25.0f, 35.0f},{300,600},{400,800} ,{0.8f,1.2f}, 0.2f, -0.1f}},
            {"flowering",ModeParam{{20,26},{40.0f,60.0f},{30.0f, 40.0f},{500,800},{600,1000},{1.0f,1.5f}, 0.3f, -0.15f}},
            {"seeding",  ModeParam{{24,30},{50.0f,70.0f},{20.0f, 30.0f},{200,400},{400,600} ,{0.4f,0.8f},0.1f, -0.05f}}
        };
         
    }
private:
    HVACEncoder* encoder;
    PrototypeClassifierSimple* proto_cls;

    int seq_len;
    int n_features;
    int latent_dim;
    float temp_init;
    float humid_init;

    float temp;
    float humid;
    float soil;
    float light;
    float co2;
    float vpd; // 蒸汽压差 (kPa)
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
            temp = 22.0f;  // 设定初始温度为22.0度
            humid = 50.0f;     // 设定初始湿度为50%
            soil  = 40.0f;     
            light=300.0f; // 设定初始光照为300lux
            co2=400.0f; // 设定初始CO2浓度为400ppm
            vpd=1.0f;  
            done = false;         // 重置任务结束标志
            
          
        };
private:
    std::vector<float> _get_state() const;
    int _get_state_cnt() const;
    float get_param(const std::map<std::string,float>& params, const std::string& key) const;
    float calc_vpd(float temp, float humid) const;
};

#endif // PLANTHVACENV_H
