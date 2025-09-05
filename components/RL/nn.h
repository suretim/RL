#ifndef NN_H
#define NN_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
 
#include <iostream>
   
#include "esp_log.h"

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

    void calculateLossAndGradients(const std::vector<float>& newExperience, 
                                    const std::vector<float>& old_probs, 
                                    const std::vector<float>& advantages, 
                                    const std::vector<float>& returns, 
                                    const std::vector<float>& old_action_probs,
                                    std::vector<float>& grads) {
        // 执行前向传播，得到新的动作概率和状态价值
        std::vector<float> action_probs(newExperience.size(), 0.0f);
        float value = 0.0f;
        forwardPass(newExperience, action_probs, value);

        // 计算 PPO 损失
        float policy_loss = calculatePolicyLoss(action_probs, old_action_probs, advantages);
        float value_loss = calculateValueLoss(returns, value);
        float entropy_loss = calculateEntropyLoss(action_probs);
        
        // 总损失
        float total_loss = policy_loss + 0.5f * value_loss - 0.01f * entropy_loss;

        // 计算梯度
        // 注意，这里是一个示例，实际的梯度计算需要反向传播
        std::vector<float> policy_gradients = calculatePolicyGradients(action_probs, old_action_probs, advantages);
        std::vector<float> value_gradients = calculateValueGradients(returns, value);
        std::vector<float> entropy_gradients = calculateEntropyGradients(action_probs);

        // 将各个部分的梯度合并到 grads 向量中
        grads.insert(grads.end(), policy_gradients.begin(), policy_gradients.end());
        grads.insert(grads.end(), value_gradients.begin(), value_gradients.end());
        grads.insert(grads.end(), entropy_gradients.begin(), entropy_gradients.end());

        // 输出总损失（用于调试）
        std::cout << "Total Loss: " << total_loss << std::endl;
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
 
#if 0
class NN_EWC {
public:
    std::vector<float> weights;  // 假設神經網路用一維權重存儲

    NN_EWC() {}
    NN_EWC(int input_dim, int hidden_dim, int action_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim), action_dim(action_dim) {}
    PPOModel model;
    bool load_model(const char* path, PPOModel& model) {
        FILE* f = fopen(path, "rb");
        if (!f) {
            ESP_LOGE("PPO", "Failed to open file: %s", path);
            return false;
        }

        fread(model.W1, sizeof(float), INPUT_DIM * HIDDEN_DIM, f);
        fread(model.b1, sizeof(float), HIDDEN_DIM, f);
        fread(model.W2, sizeof(float), HIDDEN_DIM * ACTION_DIM, f);
        fread(model.b2, sizeof(float), ACTION_DIM, f);
        fread(model.Vw, sizeof(float), HIDDEN_DIM, f);
        fread(model.Vb, sizeof(float), 1, f);

        fclose(f);
        ESP_LOGI("PPO", "Model loaded successfully");
        return true;
    }


    std::vector<float> forward_actor(const std::vector<float>& input, PPOModel& model) {
        float hidden[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = model.b1[i];
            for (int j = 0; j < INPUT_DIM; j++) {
                hidden[i] += input[j] * model.W1[j * HIDDEN_DIM + i]; // column-major
            }
            hidden[i] = tanh(hidden[i]);
        }

        std::vector<float> logits(ACTION_DIM);
        for (int i = 0; i < ACTION_DIM; i++) {
            logits[i] = model.b2[i];
            for (int j = 0; j < HIDDEN_DIM; j++) {
                logits[i] += hidden[j] * model.W2[j * ACTION_DIM + i];
            }
        }

        return logits;
    }

    // 假設 forward 輸入是 observation，輸出是隨機結果（佔位）
    std::vector<float> forward(const std::vector<float>& input) {
        // TODO: 這裡替換成真實的 forward
        std::vector<float> output(1, 0.0f);
        return output;
    }
     
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

 // 從 buffer 載入權重
    void loadWeights(const std::vector<float>& new_weights) {
       weights = new_weights;
    }
private:
    int input_dim, hidden_dim, action_dim;
    std::vector<float> W1, b1, W2, b2;
    std::vector<float> Vw;
    float Vb;

   
  
};

#endif


class NN {
public:
    NN(){}
    NN(int input_dim, int hidden_dim, int action_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim), action_dim(action_dim) {}
    std::vector<float> weights;  // 假設神經網路用一維權重存儲
    // 從 buffer 載入權重
    void loadWeights(const std::vector<float>& new_weights) {
       weights = new_weights;
    }    
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
    
 
    std::vector<float> forwardActor(const std::vector<float>& state) {
        // === 输入检查 ===
        if ((int)state.size() != input_dim) {
            ESP_LOGE("NN", "forwardActor: state size %d != input_dim %d",
                    state.size(), input_dim);
            // 返回均匀分布，避免崩溃
            return std::vector<float>(action_dim, 1.0f / action_dim);
        }

        // === 权重检查 ===
        if ((int)b1.size() != hidden_dim ||
            (int)b2.size() != action_dim ||
            (int)W1.size() != input_dim * hidden_dim ||
            (int)W2.size() != hidden_dim * action_dim) {
            ESP_LOGE("NN", "forwardActor: weight dimensions mismatch "
                        "(b1=%d, b2=%d, W1=%d, W2=%d)",
                    (int)b1.size(), (int)b2.size(),
                    (int)W1.size(), (int)W2.size());
            // 返回均匀分布，避免崩溃
            return std::vector<float>(action_dim, 1.0f / action_dim);
        }

        // === 正常前向推理 ===
        std::vector<float> hidden(hidden_dim, 0.0f);

        for (int j = 0; j < hidden_dim; j++) {
            float sum = b1[j];
            for (int i = 0; i < input_dim; i++) {
                sum += state[i] * W1[i * hidden_dim + j];
            }
            hidden[j] = std::tanh(sum);
        }

        std::vector<float> output(action_dim, 0.0f);
        for (int k = 0; k < action_dim; k++) {
            float sum = b2[k];
            for (int j = 0; j < hidden_dim; j++) {
                sum += hidden[j] * W2[j * action_dim + k];
            }
            output[k] = sum;
        }

        // Softmax 归一化
        float maxLogit = *std::max_element(output.begin(), output.end());
        float sumExp = 0.0f;
        for (auto& v : output) {
            v = std::exp(v - maxLogit);  // 数值稳定性处理
            sumExp += v;
        }
        if (sumExp <= 1e-8f) {
            ESP_LOGW("NN", "forwardActor: sumExp very small, fallback to uniform");
            return std::vector<float>(action_dim, 1.0f / action_dim);
        }
        for (auto& v : output) v /= sumExp;

        return output;
    }

    
    // Critic: 计算状态价值
    float forwardCritic(const std::vector<float>& state) {
        // === 输入检查 ===
        if ((int)state.size() != input_dim) {
            ESP_LOGE("NN", "forwardCritic: state size %d != input_dim %d",
                    state.size(), input_dim);
            return 0.0f;  // 返回默认值，避免崩溃
        }

        // === 权重检查 ===
        if ((int)b1.size() != hidden_dim ||
            (int)W1.size() != input_dim * hidden_dim ||
            (int)Vw.size() != hidden_dim) {
            ESP_LOGE("NN", "forwardCritic: weight dimensions mismatch "
                        "(b1=%d, W1=%d, Vw=%d)",
                    (int)b1.size(), (int)W1.size(), (int)Vw.size());
            return 0.0f;  // 返回默认值
        }

        // === 正常前向计算 ===
        std::vector<float> hidden(hidden_dim, 0.0f);

        for (int j = 0; j < hidden_dim; j++) {
            float sum = b1[j];
            for (int i = 0; i < input_dim; i++) {
                sum += state[i] * W1[i * hidden_dim + j];
            }
            hidden[j] = std::tanh(sum);
        }

        float value = Vb;
        for (int j = 0; j < hidden_dim; j++) {
            value += hidden[j] * Vw[j];
        }

        return value;
    }

    // 假設 forward 輸入是 observation，輸出是隨機結果（佔位）
    // std::vector<float> forward(const std::vector<float>& input) {
    //     // TODO: 這裡替換成真實的 forward
    //     std::vector<float> output(1, 0.0f);
    //     return output;
    // }
    
private:
    int input_dim, hidden_dim, action_dim;
    std::vector<float> W1, b1, W2, b2;
    std::vector<float> Vw;
    float Vb;
};

#endif // NN_H
