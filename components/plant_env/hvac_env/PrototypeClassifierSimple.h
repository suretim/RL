#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

class PrototypeClassifierSimple {
private:
    int n_classes;
    int latent_dim;
    float tau;
    std::vector<std::vector<float>> prototypes;

public:
    PrototypeClassifierSimple(int n_classes, int latent_dim, float tau = 0.1f);

    std::vector<float> operator()(const std::vector<float>& z);

    void update_prototypes(const std::vector<std::vector<float>>& features,
                           const std::vector<int>& labels);

private:
    std::vector<float> l2_normalize(const std::vector<float>& v);
    float dot_product(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> softmax(const std::vector<float>& x);
};
