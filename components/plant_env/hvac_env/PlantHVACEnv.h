#ifndef PLANTHVACENV_H
#define PLANTHVACENV_H

#include <vector>
#include <array>
#include <map>
#include <string>
#include <functional>

// 前向声明
class HVACEncoder;
class PrototypeClassifierSimple;

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
        float vpd;

        StepResult() : reward(0.0f), done(false), flower_prob(0.0f),
                       temp(0.0f), humid(0.0f), vpd(0.0f) {}
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
    int health;
    int t;
    std::array<int, 4> prev_action;

    SeqFetcher seq_fetcher;

    std::map<std::string, float> default_params = {
        {"energy_penalty", 0.1f},
        {"switch_penalty_per_toggle", 0.2f},
        {"vpd_target", 1.2f},
        {"vpd_penalty", 2.0f}
    };

public:
    PlantHVACEnv(int seq_len = 20, int n_features = 3, float temp_init = 25.0f,
                 float humid_init = 0.5f, int latent_dim = 64);
    ~PlantHVACEnv();

    void set_seq_fetcher(SeqFetcher fetcher);

    StepResult step(const std::array<int,4>& action,
                    const std::map<std::string,float>& params = {});

    std::vector<float> get_state() const;
    void update_prototypes(const std::vector<std::vector<float>>& features,
                           const std::vector<int>& labels);

private:
    std::vector<float> _get_state() const;
    float get_param(const std::map<std::string,float>& params, const std::string& key) const;
    float calc_vpd(float temp, float humid) const;
};

#endif // PLANTHVACENV_H
