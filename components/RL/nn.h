#ifndef NN_H
#define NN_H
#include <algorithm>

#include <vector>
#include <cmath>
#include <cstdlib>

// 一个非常简化的全连接网络 (仅演示)
class NN {
public:
    std::vector<float> weights;   // 扁平化存储所有权重
    size_t input_dim;
    size_t output_dim;

    NN(size_t in_dim = 4, size_t out_dim = 2) {
        input_dim = in_dim;
        output_dim = out_dim;
        // 初始化为随机权重
        weights.resize(input_dim * output_dim);
        for (auto &w : weights) {
            w = (float(rand()) / RAND_MAX - 0.5f) * 0.1f; // [-0.05, 0.05]
        }
    }

    // 简单线性层 + softmax (actor)
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(output_dim, 0.0f);
        for (size_t j = 0; j < output_dim; j++) {
            for (size_t i = 0; i < input_dim; i++) {
                output[j] += input[i] * weights[j * input_dim + i];
            }
        }
        // softmax
        float maxLogit = *max_element(output.begin(), output.end());
        float sumExp = 0.0f;
        for (auto &o : output) {
            o = expf(o - maxLogit);
            sumExp += o;
        }
        for (auto &o : output) {
            o /= sumExp;
        }
        return output;
    }
};

// ==================== 工具函數：將二進制 OTA 權重加載到 NN ====================
inline void loadWeightsToNetwork(NN& net, const uint8_t* data) {
    // 這裡假設 OTA 包裡直接是 float32 權重
    size_t total = net.weights.size();
    const float* fdata = reinterpret_cast<const float*>(data);
    for (size_t i = 0; i < total; i++) {
        net.weights[i] = fdata[i];
    }
}

#endif // NN_H
