#ifndef PLANTHVACENV_H
#define PLANTHVACENV_H

#include <vector>
#include <array>
#include <map>
#include <string>
#include <functional>
#include "version.h"
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
        float light;
        float co2;
        float vpd;

        StepResult() : reward(0.0f), done(false), flower_prob(0.0f),
                       temp(0.0f), humid(0.0f),light(0.0f), co2(0.0f), vpd(0.0f) {}
    };

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
    float light;
    float co2;
    bool done;
    int health;
    int t;
    std::array<int,PORT_CNT> prev_action;

    SeqFetcher seq_fetcher;

    std::map<std::string, float> default_params = {
        {"energy_penalty", 0.1f},
        {"switch_penalty_per_toggle", 0.2f},
        {"vpd_target", 1.2f},
        {"vpd_penalty", 2.0f},
        {"light_penalty", 0.3f},  // 光照惩罚系数
        {"co2_penalty", 0.4f}     // CO2惩罚系数
    };

public:
    PlantHVACEnv(int seq_len = 20, int n_features = 3, float temp_init = 25.0f,
                 float humid_init = 0.5f, int latent_dim = 64);
    ~PlantHVACEnv();

    void set_seq_fetcher(SeqFetcher fetcher);

    StepResult step(const std::array<int,PORT_CNT>& action,
                    const std::map<std::string,float>& params = {});

    std::vector<float> get_state() const;
    void update_prototypes(const std::vector<std::vector<float>>& features,
                           const std::vector<int>& labels);
    void reset() {
            // 将环境状态重置为初始值
            temp = 22.0f;  // 设定初始温度为22.0度
            humid = 50.0f;     // 设定初始湿度为50%
            light=300.0f; // 设定初始光照为300lux
            co2=400.0f; // 设定初始CO2浓度为400ppm
            done = false;         // 重置任务结束标志
            
          
        };
private:
    std::vector<float> _get_state() const;
    float get_param(const std::map<std::string,float>& params, const std::string& key) const;
    float calc_vpd(float temp, float humid) const;
};

#endif // PLANTHVACENV_H
