#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <map>
#include <string>
#include <functional>

// 前向声明
class PrototypeClassifierSimple;
class HVACEncoder;

// 辅助函数：计算VPD
float calc_vpd(float temp, float humid) {
    float saturation_vapor_pressure = 0.6108f * std::exp(17.27f * temp / (temp + 237.3f));
    float vapor_pressure = saturation_vapor_pressure * humid;
    return saturation_vapor_pressure - vapor_pressure;
}

// HVAC环境类
class PlantHVACEnv {
private:
    int seq_len;
    int n_features;
    float temp_init;
    float humid_init;
    int latent_dim;
    
    std::unique_ptr<HVACEncoder> encoder;
    std::unique_ptr<PrototypeClassifierSimple> proto_cls;
    
    float temp;
    float humid;
    int health;
    int t;
    std::array<int, 4> prev_action;
    
    // 参数映射
    std::map<std::string, float> default_params = {
        {"energy_penalty", 0.1f},
        {"switch_penalty_per_toggle", 0.2f},
        {"vpd_target", 1.2f},
        {"vpd_penalty", 2.0f}
    };
    
public:
    PlantHVACEnv(int seq_len = 20, int n_features = 3, float temp_init = 25.0f, 
                float humid_init = 0.5f, int latent_dim = 64)
        : seq_len(seq_len), n_features(n_features), temp_init(temp_init),
          humid_init(humid_init), latent_dim(latent_dim) {
        
        // 初始化编码器和分类器
        encoder = build_hvac_encoder(seq_len, n_features, latent_dim);
        proto_cls = std::make_unique<PrototypeClassifierSimple>(3, latent_dim, 0.1f);
        
        reset();
    }
    
    std::vector<float> reset() {
        temp = temp_init;
        humid = humid_init;
        health = 0;
        t = 0;
        prev_action = {0, 0, 0, 0};
        return _get_state();
    }
    
    std::vector<float> _get_state() {
        return {static_cast<float>(health), temp, humid};
    }
    
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
    
    StepResult step(const std::array<int, 4>& action, 
                   const std::vector<std::vector<float>>& seq_input,
                   const std::map<std::string, float>& params = {}) {
        
        StepResult result;
        
        int ac = action[0];
        int humi = action[1];
        int heat = action[2];
        int dehumi = action[3];
        
        // --- 环境动力学 ---
        temp += (ac == 1 ? -0.5f : 0.2f) + (heat == 1 ? 0.5f : 0.0f);
        humid += (humi == 1 ? 0.05f : -0.02f) + (heat == 1 ? -0.03f : 0.0f) + (dehumi == 1 ? -0.05f : 0.0f);
        
        // 数值裁剪
        temp = std::clamp(temp, 15.0f, 35.0f);
        humid = std::clamp(humid, 0.0f, 1.0f);
        
        // 健康判定
        health = (22.0f <= temp && temp <= 28.0f && 0.4f <= humid && humid <= 0.7f) ? 0 : 1;
        
        // --- latent soft label ---
        std::vector<float> z = encoder->encode(seq_input, false);
        std::vector<float> soft_label = (*proto_cls)(z);
        float flower_prob = soft_label.size() > 2 ? soft_label[2] : 0.0f;
        
        // --- reward 计算 ---
        // 基础 health + 能耗 + 开关惩罚
        float health_reward = (health == 0) ? 1.0f : -1.0f;
        
        float action_sum = static_cast<float>(ac + humi + heat + dehumi);
        float energy_penalty = get_param(params, "energy_penalty");
        float energy_cost = energy_penalty * action_sum;
        
        float action_diff = 0.0f;
        for (int i = 0; i < 4; ++i) {
            action_diff += std::abs(action[i] - prev_action[i]);
        }
        float switch_penalty = get_param(params, "switch_penalty_per_toggle") * action_diff;
        
        // 花期强化 VPD 控制
        float vpd_target = get_param(params, "vpd_target");
        float vpd_current = calc_vpd(temp, humid);
        float vpd_penalty = get_param(params, "vpd_penalty");
        float vpd_reward = -std::abs(vpd_current - vpd_target) * vpd_penalty;
        
        float reward = health_reward - energy_cost - switch_penalty + flower_prob * vpd_reward;
        
        // 更新状态
        prev_action = action;
        t++;
        bool done = t >= seq_len;
        
        // 填充结果
        result.state = _get_state();
        result.reward = reward;
        result.done = done;
        result.latent_soft_label = soft_label;
        result.flower_prob = flower_prob;
        result.temp = temp;
        result.humid = humid;
        result.vpd = vpd_current;
        
        return result;
    }
    
    // 获取当前环境状态
    std::vector<float> get_state() const {
        return _get_state();
    }
    
    // 获取当前时间步
    int get_timestep() const {
        return t;
    }
    
    // 更新原型分类器
    void update_prototypes(const std::vector<std::vector<float>>& features, 
                          const std::vector<int>& labels) {
        proto_cls->update_prototypes(features, labels);
    }
    
    // 获取编码器（用于外部访问）
    HVACEncoder* get_encoder() const {
        return encoder.get();
    }
    
    // 获取分类器（用于外部访问）
    PrototypeClassifierSimple* get_classifier() const {
        return proto_cls.get();
    }
    
private:
    // 获取参数值，如果用户提供了则使用用户值，否则使用默认值
    float get_param(const std::map<std::string, float>& params, const std::string& key) {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        
        auto default_it = default_params.find(key);
        if (default_it != default_params.end()) {
            return default_it->second;
        }
        
        // 如果找不到参数，返回0.0f
        return 0.0f;
    }
};

// PrototypeClassifierSimple 实现
class PrototypeClassifierSimple {
private:
    int n_classes;
    int latent_dim;
    float tau;
    std::vector<std::vector<float>> prototypes;
    
public:
    PrototypeClassifierSimple(int n_classes, int latent_dim, float tau = 0.1f)
        : n_classes(n_classes), latent_dim(latent_dim), tau(tau) {
        
        // 初始化原型向量
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        prototypes.resize(n_classes);
        for (int i = 0; i < n_classes; ++i) {
            prototypes[i].resize(latent_dim);
            for (int j = 0; j < latent_dim; ++j) {
                prototypes[i][j] = dist(gen);
            }
        }
    }
    
    // L2归一化
    std::vector<float> l2_normalize(const std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) {
            norm += x * x;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            std::vector<float> result(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                result[i] = v[i] / norm;
            }
            return result;
        }
        return v;
    }
    
    // 点积
    float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // 前向传播
    std::vector<float> operator()(const std::vector<float>& z) {
        std::vector<float> z_normalized = l2_normalize(z);
        
        // 计算相似度
        std::vector<float> sim(n_classes);
        for (int i = 0; i < n_classes; ++i) {
            std::vector<float> proto_normalized = l2_normalize(prototypes[i]);
            sim[i] = dot_product(z_normalized, proto_normalized) / tau;
        }
        
        // Softmax
        return softmax(sim);
    }
    
    // Softmax函数
    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> exp_x(x.size());
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            exp_x[i] = std::exp(x[i] - max_val);
            sum += exp_x[i];
        }
        
        for (size_t i = 0; i < exp_x.size(); ++i) {
            exp_x[i] /= sum;
        }
        
        return exp_x;
    }
    
    // 更新原型向量
    void update_prototypes(const std::vector<std::vector<float>>& features, 
                          const std::vector<int>& labels) {
        for (int k = 0; k < n_classes; ++k) {
            std::vector<std::vector<float>> class_features;
            
            // 收集属于类别k的特征
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == k && features[i].size() == latent_dim) {
                    class_features.push_back(features[i]);
                }
            }
            
            if (!class_features.empty()) {
                // 计算均值向量
                std::vector<float> mean_vec(latent_dim, 0.0f);
                for (const auto& feature : class_features) {
                    for (int j = 0; j < latent_dim; ++j) {
                        mean_vec[j] += feature[j];
                    }
                }
                
                for (int j = 0; j < latent_dim; ++j) {
                    mean_vec[j] /= class_features.size();
                }
                
                // 更新原型
                prototypes[k] = mean_vec;
            }
        }
    }
};

// HVACEncoder 实现（简化版本，使用标准库）
class HVACEncoder {
private:
    int seq_len;
    int n_features;
    int latent_dim;
    
    // 简化的编码逻辑 - 实际应用中应该使用真正的神经网络
    std::vector<float> encode_simple(const std::vector<std::vector<float>>& seq_input) {
        // 这里使用简单的平均池化作为示例
        // 实际应用中应该使用LSTM等神经网络
        
        std::vector<float> result(latent_dim, 0.0f);
        
        if (seq_input.empty()) return result;
        
        // 对每个特征维度求平均
        for (int i = 0; i < latent_dim; ++i) {
            for (const auto& timestep : seq_input) {
                if (i < timestep.size()) {
                    result[i] += timestep[i];
                }
            }
            result[i] /= seq_input.size();
        }
        
        // L2归一化
        float norm = 0.0f;
        for (float val : result) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            for (float& val : result) {
                val /= norm;
            }
        }
        
        return result;
    }
    
public:
    HVACEncoder(int seq_len, int n_features, int latent_dim)
        : seq_len(seq_len), n_features(n_features), latent_dim(latent_dim) {}
    
    std::vector<float> encode(const std::vector<std::vector<float>>& seq_input, bool training = false) {
        return encode_simple(seq_input);
    }
};

// 构建编码器的辅助函数
std::unique_ptr<HVACEncoder> build_hvac_encoder(int seq_len, int n_features, int latent_dim) {
    return std::make_unique<HVACEncoder>(seq_len, n_features, latent_dim);
}