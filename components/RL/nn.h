#ifndef NN_H
#define NN_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
 
#include <iostream> 
#include "esp_log.h" 
#include <numeric> 
#include "version.h"
#include "ml_pid.h"
static const char *NN_TAG = "PPOEWC";

// 假設 ACTION_DIM, STATE_DIM 已經定義
#define ACTION_DIM ACTION_CNT
#define STATE_DIM STATE_CNT 
#define HIDDEN_DIM 32 


// 自定义 clamp 函数
template <typename T>
T clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}
struct PPOModelStruct {
    float W1[STATE_DIM * HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    float W2[HIDDEN_DIM * ACTION_DIM];
    float b2[ACTION_DIM];
    float Vw[HIDDEN_DIM];
    float Vb[1];
};

  
class NN {
public:
   NN()
        : input_dim(STATE_DIM),
          hidden_dim(HIDDEN_DIM),
          action_dim(ACTION_DIM),
          Vb(0.0f),
          model_ready(false) {}

    // Param constructor
    NN(int input_dim_, int hidden_dim_, int action_dim_)
        : input_dim(input_dim_),
          hidden_dim(hidden_dim_),
          action_dim(action_dim_),
          Vb(0.0f),
          model_ready(false) {}
    std::vector<float> weights;  // 假設神經網路用一維權重存儲
    // 在 NN 类 public: 添加下面两个方法声明
    // 将网络参数打平成一个 vector（用于检查或备份）
    std::vector<float> get_weights_flat() const {
        std::vector<float> flat;
        flat.reserve(W1.size() + b1.size() + W2.size() + b2.size() + Vw.size() + 1);
        flat.insert(flat.end(), W1.begin(), W1.end());
        flat.insert(flat.end(), b1.begin(), b1.end());
        flat.insert(flat.end(), W2.begin(), W2.end());
        flat.insert(flat.end(), b2.begin(), b2.end());
        flat.insert(flat.end(), Vw.begin(), Vw.end());
        flat.push_back(Vb);
        return flat;
    }

    // 应用各参数的梯度（直接 SGD step）
    void apply_gradients(const std::vector<float>& dW1, const std::vector<float>& db1,
                        const std::vector<float>& dW2, const std::vector<float>& db2,
                        const std::vector<float>& dVw, float dVb,
                        float lr_shared, float lr_actor, float lr_critic) {
        // sizes must match
        if (dW1.size() == W1.size()) {
            for (size_t i = 0; i < W1.size(); ++i) W1[i] -= lr_shared * dW1[i];
        }
        if (db1.size() == b1.size()) {
            for (size_t i = 0; i < b1.size(); ++i) b1[i] -= lr_shared * db1[i];
        }
        if (dW2.size() == W2.size()) {
            // actor final layer use lr_actor
            for (size_t i = 0; i < W2.size(); ++i) W2[i] -= lr_actor * dW2[i];
        }
        if (db2.size() == b2.size()) {
            for (size_t i = 0; i < b2.size(); ++i) b2[i] -= lr_actor * db2[i];
        }
        if (dVw.size() == Vw.size()) {
            for (size_t i = 0; i < Vw.size(); ++i) Vw[i] -= lr_critic * dVw[i];
        }
        Vb -= lr_critic * dVb;
    }

    // 從 buffer 載入權重
    void loadWeights(const std::vector<float>& new_weights) {
       weights = new_weights;
    }    
    bool is_ready() const { return model_ready; }

    // 从二进制文件加载模型参数
    bool loadWeights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        size_t sizeW1 = input_dim * hidden_dim;
        size_t sizeb1 = hidden_dim;
        size_t sizeW2 = hidden_dim * action_dim;
        size_t sizeb2 = action_dim;
        size_t sizeVw = hidden_dim;

        W1.resize(sizeW1);
        b1.resize(sizeb1);
        W2.resize(sizeW2);
        b2.resize(sizeb2);
        Vw.resize(sizeVw);

        file.read(reinterpret_cast<char*>(W1.data()), sizeW1 * sizeof(float));
        file.read(reinterpret_cast<char*>(b1.data()), sizeb1 * sizeof(float));
        file.read(reinterpret_cast<char*>(W2.data()), sizeW2 * sizeof(float));
        file.read(reinterpret_cast<char*>(b2.data()), sizeb2 * sizeof(float));
        file.read(reinterpret_cast<char*>(Vw.data()), sizeVw * sizeof(float));
        file.read(reinterpret_cast<char*>(&Vb), sizeof(float));

        file.close();
        return true;
    }
    bool load_from_vector(const std::vector<float>& flat) {
        size_t sizeW1 = (size_t)input_dim * (size_t)hidden_dim;
        size_t sizeb1 = (size_t)hidden_dim;
        size_t sizeW2 = (size_t)hidden_dim * (size_t)action_dim;
        size_t sizeb2 = (size_t)action_dim;
        size_t sizeVw = (size_t)hidden_dim;
        size_t expected = sizeW1 + sizeb1 + sizeW2 + sizeb2 + sizeVw + 1;

        if (flat.size() < expected) {
            ESP_LOGE(NN_TAG, "load_from_vector: flat size %zu < expected %zu", flat.size(), expected);
            return false;
        }

        size_t offset = 0;
        W1.assign(flat.begin() + offset, flat.begin() + offset + sizeW1); offset += sizeW1;
        b1.assign(flat.begin() + offset, flat.begin() + offset + sizeb1); offset += sizeb1;
        W2.assign(flat.begin() + offset, flat.begin() + offset + sizeW2); offset += sizeW2;
        b2.assign(flat.begin() + offset, flat.begin() + offset + sizeb2); offset += sizeb2;
        Vw.assign(flat.begin() + offset, flat.begin() + offset + sizeVw); offset += sizeVw;
        Vb = flat[offset];

        model_ready = true;
        ESP_LOGI(NN_TAG, "Weights loaded: W1=%zu b1=%zu W2=%zu b2=%zu Vw=%zu Vb set",
                 W1.size(), b1.size(), W2.size(), b2.size(), Vw.size());
        return true;
    }
 // Load binary file and parse into internal vectors (calls load_from_vector)
    bool loadFromFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            ESP_LOGE(NN_TAG, "loadFromFile: failed to open %s", path.c_str());
            return false;
        }

        // compute expected count of floats
        size_t sizeW1 = (size_t)input_dim * (size_t)hidden_dim;
        size_t sizeb1 = (size_t)hidden_dim;
        size_t sizeW2 = (size_t)hidden_dim * (size_t)action_dim;
        size_t sizeb2 = (size_t)action_dim;
        size_t sizeVw = (size_t)hidden_dim;
        size_t expected = sizeW1 + sizeb1 + sizeW2 + sizeb2 + sizeVw + 1;

        std::vector<float> flat(expected);
        file.read(reinterpret_cast<char*>(flat.data()), expected * sizeof(float));
        if (file.gcount() != (std::streamsize)(expected * sizeof(float))) {
            ESP_LOGE(NN_TAG, "loadFromFile: read bytes %ld != expected %zu", (long)file.gcount(), expected * sizeof(float));
            file.close();
            return false;
        }
        file.close();
        return load_from_vector(flat);
    }
 
    // Forward actor returns softmax probabilities; always safe (if not ready returns uniform)
    std::vector<float> forwardActor(const std::vector<float>& state) {
        if (!model_ready) {
            //ESP_LOGW(NN_TAG, "forwardActor called but model not ready -> uniform output");
            std::vector<float> action_prob(ACTION_CNT);
            for (int i = 0; i < ACTION_CNT; i++)
                action_prob[i] = lstm_pid_out_speed.speed[i+1];
            return action_prob; 
        }
        if ((int)state.size() != input_dim) {
            ESP_LOGE(NN_TAG, "forwardActor: state size %d != input_dim %d", (int)state.size(), input_dim);
            return std::vector<float>((size_t)action_dim, 1.0f / (float)action_dim);
        }
        // compute hidden
        std::vector<float> hidden((size_t)hidden_dim, 0.0f);
        for (int j = 0; j < hidden_dim; ++j) {
            float sum = b1[(size_t)j];
            for (int i = 0; i < input_dim; ++i) {
                sum += state[(size_t)i] * W1[(size_t)i * (size_t)hidden_dim + (size_t)j];
            }
            hidden[(size_t)j] = std::tanh(sum);
        }
        std::vector<float> out((size_t)action_dim, 0.0f);
        for (int k = 0; k < action_dim; ++k) {
            float sum = b2[(size_t)k];
            for (int j = 0; j < hidden_dim; ++j) {
                sum += hidden[(size_t)j] * W2[(size_t)j * (size_t)action_dim + (size_t)k];
            }
            out[(size_t)k] = sum;
        }
        // softmax
        float maxv = *std::max_element(out.begin(), out.end());
        float s = 0.0f;
        for (auto &v : out) { v = std::exp(v - maxv); s += v; }
        if (s <= 1e-8f) {
            ESP_LOGW(NN_TAG, "forwardActor: sumExp small -> uniform");
            return std::vector<float>((size_t)action_dim, 1.0f / (float)action_dim);
        }
        for (auto &v : out) v /= s;
        return out;
    }

    // Critic forward (returns 0.0 if not ready)
    float forwardCritic(const std::vector<float>& state) {
        if (!model_ready) {
            //ESP_LOGW(NN_TAG, "forwardCritic called but model not ready -> 0.0");
            //return 0.0f; 
            return lstm_pid_out_speed.speed[0]; 
        }
        if ((int)state.size() != input_dim) {
            ESP_LOGE(NN_TAG, "forwardCritic: state size %d != input_dim %d", (int)state.size(), input_dim);
            return 0.0f;
        }
        std::vector<float> hidden((size_t)hidden_dim, 0.0f);
        for (int j = 0; j < hidden_dim; ++j) {
            float sum = b1[(size_t)j];
            for (int i = 0; i < input_dim; ++i) {
                sum += state[(size_t)i] * W1[(size_t)i * (size_t)hidden_dim + (size_t)j];
            }
            hidden[(size_t)j] = std::tanh(sum);
        }
        float value = Vb;
        for (int j = 0; j < hidden_dim; ++j) value += hidden[(size_t)j] * Vw[(size_t)j];
        return value;
    }
    
private:
    int input_dim, hidden_dim, action_dim;
    std::vector<float> W1, b1, W2, b2;
    std::vector<float> Vw;
    float Vb;
    bool model_ready;
};

inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
 
// ==================== PPO + EWC 模型 ====================
class PPOEWCModel {
private:
    int input_dim, hidden_dim, action_dim;
    NN actor_network;
    NN critic_network;

    std::vector<float> old_actor_weights;
    std::vector<float> old_critic_weights;

public:
    PPOEWCModel() {
        
        input_dim = STATE_DIM;   
        action_dim = ACTION_DIM ;  
        // 初始化网络参数
        initModel();
    }
     void initModel() {
        // 这里写初始化逻辑，比如分配参数、随机权重
        policy_weights.assign(input_dim * action_dim, 0.01f); // 小随机数
        value_weights.assign(input_dim, 0.01f);
        fisher_matrix.assign(policy_weights.size(), 0.0f);    // 初始化 EWC 鱼信息矩阵
    }

    // 你的成员变量
    std::vector<float> policy_weights;
    std::vector<float> value_weights;
    std::vector<float> fisher_matrix;
    PPOEWCModel(int input_dim, int hidden_dim, int action_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim), action_dim(action_dim),
          actor_network(input_dim, hidden_dim, action_dim),
          critic_network(input_dim, hidden_dim, 1) {}

    // ==================== 推理 ====================
    std::vector<float> predictAction(const std::vector<float>& obs) {
        return actor_network.forwardActor(obs);
    }
     
    float predictValue(const std::vector<float>& obs) {
        return critic_network.forwardCritic(obs);
    }

    struct PPOOutput {
        std::vector<float> action_probs;
        float value;
    };

    PPOOutput predictFull(const std::vector<float>& obs) {
        
        PPOOutput out;
        out.action_probs = predictAction(obs);
        out.value = predictValue(obs);
        return out;
    }

    float calculatePolicyLoss(const std::vector<float>& new_probs,
                          const std::vector<float>& old_probs,
                          const std::vector<float>& advantages) {
    float loss = 0.0f;
    for (size_t i = 0; i < new_probs.size(); ++i) {
        float ratio = new_probs[i] / (old_probs[i] + 1e-8f);
        float clipped = clamp(ratio, 1.0f - 0.2f, 1.0f + 0.2f);
        loss += std::min(ratio * advantages[i], clipped * advantages[i]);
    }
    return -loss / new_probs.size();
}

float calculateValueLoss(const std::vector<float>& returns, float predicted_value) {
    if (returns.empty()) return 0.0f;
    float value_loss = 0.0f;
    for (size_t i = 0; i < returns.size(); ++i) {
        value_loss += std::pow(returns[i] - predicted_value, 2);
    }
    return value_loss / returns.size();
}

float calculateEntropyLoss(const std::vector<float>& new_probs) {
    float entropy_loss = 0.0f;
    for (size_t i = 0; i < new_probs.size(); ++i) {
        entropy_loss -= new_probs[i] * std::log(new_probs[i] + 1e-8f);
    }
    return entropy_loss / new_probs.size();
}

std::vector<float> calculatePolicyGradients(const std::vector<float>& new_probs,
                                            const std::vector<float>& old_probs,
                                            const std::vector<float>& advantages) {
    std::vector<float> gradients(new_probs.size(), 0.0f);
    for (size_t i = 0; i < new_probs.size(); ++i) {
        float ratio = new_probs[i] / (old_probs[i] + 1e-8f);
        gradients[i] = ratio * advantages[i];
    }
    return gradients;
}
std::vector<float> calculateEntropyGradients(const std::vector<float>& action_probs) {
    std::vector<float> entropy_gradients(action_probs.size(), 0.0f);
    // 这里用 d( -Σ p log p ) / dp = -(log p + 1)
    for (size_t i = 0; i < action_probs.size(); ++i) {
        float p = std::max(action_probs[i], 1e-8f); // 避免 log(0)
        entropy_gradients[i] = -(std::log(p) + 1.0f);
    }
    return entropy_gradients;
}
std::vector<float> calculateValueGradients(const std::vector<float>& returns, float predicted_value) {
    std::vector<float> gradients(1, 0.0f);
    if (!returns.empty()) {
        gradients[0] = predicted_value - returns[0];
    }
    return gradients;
}
 


void calculateGradients(const std::vector<float>& state_batch,
                        const std::vector<float>& old_probs,
                        const std::vector<float>& advantages,
                        const std::vector<float>& returns,
                        const std::vector<float>& old_action_probs,
                        std::vector<float>& grads) {
    // 1) 基本检查
    if (state_batch.empty() || old_probs.empty() || advantages.empty()) {
        ESP_LOGE(NN_TAG, "calculateGradients: input batch empty");
        return;
    }

    // 2) 前向传播，得到新的 action_probs 和 value
    PPOOutput out = predictFull(state_batch);
    std::vector<float> action_probs = out.action_probs;
    float value = out.value;

    if (action_probs.size() != (size_t)action_dim) {
        ESP_LOGE(NN_TAG, "calculateGradients: action_probs size mismatch %zu != %d",
                 action_probs.size(), action_dim);
        return;
    }

    // 3) 计算 loss
    float policy_loss  = calculatePolicyLoss(action_probs, old_action_probs, advantages);
    float value_loss   = calculateValueLoss(returns, value);
    float entropy_loss = calculateEntropyLoss(action_probs);

    float total_loss = policy_loss + 0.5f * value_loss - 0.01f * entropy_loss;

    // 4) 计算梯度
    auto policy_gradients  = calculatePolicyGradients(action_probs, old_action_probs, advantages);
    auto value_gradients   = calculateValueGradients(returns, value);
    auto entropy_gradients = calculateEntropyGradients(action_probs);

    // 5) 合并梯度
    grads.clear();
    grads.insert(grads.end(), policy_gradients.begin(), policy_gradients.end());
    grads.insert(grads.end(), value_gradients.begin(), value_gradients.end());
    grads.insert(grads.end(), entropy_gradients.begin(), entropy_gradients.end());

    ESP_LOGI(NN_TAG, "Total Loss: %f, grads size: %zu", total_loss, grads.size());
}



       // ==================== EWC 更新 ====================
    void continualLearningEWC(const std::vector<float>& grads,
                              const std::vector<float>& fisher_actor,
                              const std::vector<float>& fisher_critic,
                              float lr=0.001f) {
        // 更新 Actor
        for (size_t i = 0; i < actor_network.weights.size(); i++) {
            float reg = fisher_actor[i] * (actor_network.weights[i] - old_actor_weights[i]);
            actor_network.weights[i] -= lr * (grads[i] + reg);
        }
        // 更新 Critic
        for (size_t i = 0; i < critic_network.weights.size(); i++) {
            float reg = fisher_critic[i] * (critic_network.weights[i] - old_critic_weights[i]);
            critic_network.weights[i] -= lr * (grads[actor_network.weights.size() + i] + reg);
        }
    }

    // 默認 fisher=0
    void continualLearningEWC(const std::vector<float>& grads) {
        std::vector<float> fisher_actor(actor_network.weights.size(), 0.0f);
        std::vector<float> fisher_critic(critic_network.weights.size(), 0.0f);
        continualLearningEWC(grads, fisher_actor, fisher_critic);
    }

    // ==================== OTA 模型加載 ====================
    bool loadModel(const uint8_t* data, size_t size) {
        size_t half = size / 2;
        const float* fdata = reinterpret_cast<const float*>(data);
        size_t num_floats = size / sizeof(float);

        std::vector<float> actor_w(fdata, fdata + num_floats/2);
        std::vector<float> critic_w(fdata + num_floats/2, fdata + num_floats);

        actor_network.loadWeights(actor_w);
        critic_network.loadWeights(critic_w);

        old_actor_weights = actor_network.weights;
        old_critic_weights = critic_network.weights;

        ESP_LOGI(NN_TAG, "Model loaded via OTA, actor=%zu, critic=%zu",
                 actor_w.size(), critic_w.size());
        return true;
    }
};


#endif // NN_H
