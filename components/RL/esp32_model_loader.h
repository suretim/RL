#ifndef ESP32_MODEL_LOADER_H
#define ESP32_MODEL_LOADER_H

#include <vector>
#include <stdio.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_spiffs.h"
#include "nn.h"  // 这里替换成你自己的神经网络实现

class ESP32EWCModel {
private:
    int input_dim, hidden_dim, action_dim;
    NN actor_network;   // 假設 NN 是你自定義的簡單神經網路類
    NN critic_network;

    std::vector<float> old_actor_weights;
    std::vector<float> old_critic_weights;

public:
    ESP32EWCModel() {}
    ESP32EWCModel(int input_dim, int hidden_dim, int action_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim), action_dim(action_dim) {}
    // ==================== dequantize（8-bit -> float） ====================
    std::vector<float> dequantize(const uint8_t* data, size_t size, float min, float max) {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; i++) {
            result[i] = min + (data[i] / 255.0f) * (max - min);
        }
        return result;
    }
    void loadWeightsToNetwork(NN& network, const uint8_t* otaData, size_t size) {
        size_t num_weights = size / sizeof(float);
        const float* floatData = reinterpret_cast<const float*>(otaData);
        std::vector<float> weights(floatData, floatData + num_weights);
        network.loadWeights(weights);  // NN 类需要有 loadWeights 方法
    } 

       // ==================== 從 OTA 數據加載模型 ====================
    bool loadModel(const uint8_t* otaData, size_t dataSize) {
        // TODO: 解析 OTA 包格式（可用 msgpack/自定義二進制）
        //loadWeightsToNetwork(actor_network, otaData);
        //loadWeightsToNetwork(critic_network, otaData);
        loadWeightsToNetwork(actor_network, otaData, dataSize/2);
        loadWeightsToNetwork(critic_network, otaData + dataSize/2, dataSize/2);

        // 保存快照，用於 EWC
        old_actor_weights = actor_network.weights;
        old_critic_weights = critic_network.weights;

        return true;
    }
 

    // ==================== 推理 ====================
    std::vector<float> predict(const std::vector<float>& observation) {
        //return actor_network.forward(observation);
        return actor_network.forwardActor(observation);
    }

    float predictValue(const std::vector<float>& observation) {
        //return critic_network.forward(observation)[0];
        return critic_network.forwardCritic( observation) ;
    }

    struct PPOOutput {
        std::vector<float> action_probs;
        float value;
    };

    PPOOutput predictFull(const std::vector<float>& observation) {
        PPOOutput out;
        out.action_probs = predict(observation);
        out.value = predictValue(observation);
        return out;
    }
    
    // ==================== EWC 持續學習 ====================
    void continualLearningEWC(const std::vector<float>& gradients,
                              const std::vector<float>& fisher_actor,
                              const std::vector<float>& fisher_critic,
                              float learning_rate = 0.001f) {
        // 假設 gradients 是 actor + critic 的合併梯度
        // actor
        for (size_t i = 0; i < actor_network.weights.size(); i++) {
            float reg = fisher_actor[i] * (actor_network.weights[i] - old_actor_weights[i]);
            actor_network.weights[i] -= learning_rate * (gradients[i] + reg);
        }
        // critic
        for (size_t i = 0; i < critic_network.weights.size(); i++) {
            float reg = fisher_critic[i] * (critic_network.weights[i] - old_critic_weights[i]);
            critic_network.weights[i] -= learning_rate * (gradients[actor_network.weights.size() + i] + reg);
        }
    }


    void continualLearningEWC(const std::vector<float>& gradients) {
        std::vector<float> fisher_actor(actor_network.weights.size(), 0.0f);
        std::vector<float> fisher_critic(critic_network.weights.size(), 0.0f);
        continualLearningEWC(gradients, fisher_actor, fisher_critic, 0.001f);
    }
};

#endif // ESP32_MODEL_LOADER_H
