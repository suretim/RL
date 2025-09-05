#pragma once
#include <vector>

class HVACEncoder {
private:
    int seq_len;
    int n_features;
    int latent_dim;
    std::vector<float> encode_simple(const std::vector<std::vector<float>>& seq_input);

public:
    HVACEncoder(int seq_len, int n_features, int latent_dim);
    std::vector<float> encode(const std::vector<std::vector<float>>& seq_input, bool training=false);
};
