#include "HVACEncoder.h"
#include <cmath>

HVACEncoder::HVACEncoder(int seq_len_, int n_features_, int latent_dim_)
    : seq_len(seq_len_), n_features(n_features_), latent_dim(latent_dim_) {}

std::vector<float> HVACEncoder::encode_simple(const std::vector<std::vector<float>>& seq_input) {
    std::vector<float> result(latent_dim, 0.0f);
    if (seq_input.empty()) return result;
    for (int i = 0; i < latent_dim; ++i) {
        for (size_t t = 0; t < seq_input.size(); ++t)
            if (i < seq_input[t].size()) result[i] += seq_input[t][i];
        result[i] /= seq_input.size();
    }
    float norm = 0.0f;
    for (size_t i = 0; i < result.size(); ++i) norm += result[i]*result[i];
    norm = std::sqrt(norm);
    if (norm > 0) for (size_t i = 0; i < result.size(); ++i) result[i] /= norm;
    return result;
}

std::vector<float> HVACEncoder::encode(const std::vector<std::vector<float>>& seq_input, bool training) {
    return encode_simple(seq_input);
}
