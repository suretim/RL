#ifndef NN_H
#define NN_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
 
#include <iostream>
   
#include "esp_log.h"
static const char *NN_TAG = "NN_TAG"; 

#define INPUT_DIM 5
#define HIDDEN_DIM 32
#define ACTION_DIM 4
// 自定义 clamp 函数
template <typename T>
T clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}
struct PPOModelStruct {
    float W1[INPUT_DIM * HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    float W2[HIDDEN_DIM * ACTION_DIM];
    float b2[ACTION_DIM];
    float Vw[HIDDEN_DIM];
    float Vb[1];
};

class PPOModel {
public:
    PPOModel() {
        // 初始化模型等
    }
    void calculateLossAndGradients(const std::vector<float>& state_batch,
                               const std::vector<float>& old_probs,
                               const std::vector<float>& advantages,
                               const std::vector<float>& returns,
                               const std::vector<float>& old_action_probs,
                               std::vector<float>& grads) {
        // 1) basic size checks
        if (state_batch.size() == 0 || old_probs.size() == 0 || advantages.size() == 0 || returns.size() == 0) {
            ESP_LOGE(NN_TAG, "calculateLossAndGradients: input batch empty");
            return;
        }
        if (old_probs.size() != (size_t)ACTION_DIM || old_action_probs.size() != (size_t)ACTION_DIM) {
            ESP_LOGW(NN_TAG, "calculateLossAndGradients: old_probs size mismatch: %zu", old_probs.size());
            // optionally return or continue with clamp to action_dim
        }

        // action_probs must be action_dim
        std::vector<float> action_probs((size_t)ACTION_DIM, 0.0f);
        float value = 0.0f;
        forwardPass(state_batch, action_probs, value); // 确保 forwardPass 将 action_probs 填满

        // ensure action_probs length is correct
        if (action_probs.size() != (size_t)ACTION_DIM) {
            ESP_LOGE(NN_TAG, "action_probs size %zu != ACTION_DIM %d", action_probs.size(), ACTION_DIM);
            return;
        }

        // sizes for advantages / returns likely represent batch size; make sure to use matching indexing
        // 以下示例仅为演示：请根据你的 batch 定义调整
        float policy_loss = calculatePolicyLoss(action_probs, old_action_probs, advantages);
        float value_loss = calculateValueLoss(returns, value);
        float entropy_loss = calculateEntropyLoss(action_probs);

        float total_loss = policy_loss + 0.5f * value_loss - 0.01f * entropy_loss;

        auto policy_gradients = calculatePolicyGradients(action_probs, old_action_probs, advantages);
        auto value_gradients = calculateValueGradients(returns, value);
        auto entropy_gradients = calculateEntropyGradients(action_probs);

        // 合并时先清空
        grads.clear();
        grads.insert(grads.end(), policy_gradients.begin(), policy_gradients.end());
        grads.insert(grads.end(), value_gradients.begin(), value_gradients.end());
        grads.insert(grads.end(), entropy_gradients.begin(), entropy_gradients.end());

        ESP_LOGI(NN_TAG, "Total Loss: %f, grads size: %zu", total_loss, grads.size());
    }

private:
 
    // 神经网络前向传播
    void forwardPass(const std::vector<float>& state, std::vector<float>& action_probs, float& value) {
        // 此处应通过 TensorFlow Lite 进行推理
        // 这里只是模拟，假设动作概率是 0.5，状态值是 0.2
        std::fill(action_probs.begin(), action_probs.end(), 0.5f);  // 假设所有动作概率为 0.5
        value = 0.2f;  // 假设价值为 0.2
    }

    // 计算 PPO 策略损失
    float calculatePolicyLoss(const std::vector<float>& new_probs, const std::vector<float>& old_probs, const std::vector<float>& advantages) {
        float loss = 0.0f;
        // 自定义 clamp 函数

        for (size_t i = 0; i < new_probs.size(); ++i) {
            float ratio = new_probs[i] / (old_probs[i] + 1e-8);
            float clipped_ratio =  clamp(ratio, 1.0f - 0.2f, 1.0f + 0.2f);  // 剪切 epsilon = 0.2
            loss += std::min(ratio * advantages[i], clipped_ratio * advantages[i]);
        }
        return -loss / new_probs.size();  // 负号是因为我们要最小化损失
    }

    // 计算价值损失
    float calculateValueLoss(const std::vector<float>& returns, float predicted_value) {
        float value_loss = 0.0f;
        for (size_t i = 0; i < returns.size(); ++i) {
            value_loss += std::pow(returns[i] - predicted_value, 2);
        }
        return value_loss / returns.size();
    }

    // 计算熵损失
    float calculateEntropyLoss(const std::vector<float>& new_probs) {
        float entropy_loss = 0.0f;
        for (size_t i = 0; i < new_probs.size(); ++i) {
            entropy_loss -= new_probs[i] * std::log(new_probs[i] + 1e-8) + (1 - new_probs[i]) * std::log(1 - new_probs[i] + 1e-8);
        }
        return entropy_loss / new_probs.size();
    }

    // 假设计算 PPO 策略的梯度
    std::vector<float> calculatePolicyGradients(const std::vector<float>& new_probs, const std::vector<float>& old_probs, const std::vector<float>& advantages) {
        std::vector<float> gradients(new_probs.size(), 0.0f);
        for (size_t i = 0; i < new_probs.size(); ++i) {
            float ratio = new_probs[i] / (old_probs[i] + 1e-8);
            gradients[i] = ratio * advantages[i];  // 这里只是示例，实际梯度计算更复杂
        }
        return gradients;
    }

    // 假设计算价值函数的梯度
    std::vector<float> calculateValueGradients(const std::vector<float>& returns, float predicted_value) {
        std::vector<float> gradients(1, 0.0f);  // 假设只有一个梯度
        gradients[0] = predicted_value - returns[0];  // 简单的梯度计算
        return gradients;
    }

    // 假设计算熵的梯度
    std::vector<float> calculateEntropyGradients(const std::vector<float>& action_probs) {
        std::vector<float> gradients(action_probs.size(), 0.0f);
        for (size_t i = 0; i < action_probs.size(); ++i) {
            gradients[i] = -std::log(action_probs[i] + 1e-8);  // 简单的梯度计算
        }
        return gradients;
    }
};
  
class NN {
public:
   NN()
        : input_dim(INPUT_DIM),
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
            ESP_LOGW(NN_TAG, "forwardActor called but model not ready -> uniform output");
            return std::vector<float>((size_t)action_dim, 1.0f / (float)action_dim);
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
            ESP_LOGW(NN_TAG, "forwardCritic called but model not ready -> 0.0");
            return 0.0f;
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

#endif // NN_H
