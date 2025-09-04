#ifndef NN_H
#define NN_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

   
#include "esp_log.h"

#define INPUT_DIM 5
#define HIDDEN_DIM 32
#define ACTION_DIM 4

struct PPOModel {
    float W1[INPUT_DIM * HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    float W2[HIDDEN_DIM * ACTION_DIM];
    float b2[ACTION_DIM];
    float Vw[HIDDEN_DIM];
    float Vb[1];
};


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




class NN {
public:
    NN(int input_dim, int hidden_dim, int action_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim), action_dim(action_dim) {}

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
    

        // Actor: 输出动作概率分布
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

private:
    int input_dim, hidden_dim, action_dim;
    std::vector<float> W1, b1, W2, b2;
    std::vector<float> Vw;
    float Vb;
};

#endif // NN_H
