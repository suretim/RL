#include "PrototypeClassifierSimple.h"

PrototypeClassifierSimple::PrototypeClassifierSimple(int n_classes_, int latent_dim_, float tau_)
    : n_classes(n_classes_), latent_dim(latent_dim_), tau(tau_) {

    prototypes.resize(n_classes);
    for (int i = 0; i < n_classes; ++i) {
        prototypes[i].resize(latent_dim);
        for (int j = 0; j < latent_dim; ++j) {
            prototypes[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

std::vector<float> PrototypeClassifierSimple::l2_normalize(const std::vector<float>& v) {
    float norm = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) norm += v[i] * v[i];
    norm = std::sqrt(norm);

    std::vector<float> result(v.size());
    if (norm > 0) {
        for (size_t i = 0; i < v.size(); ++i) result[i] = v[i] / norm;
    } else {
        result = v;
    }
    return result;
}

float PrototypeClassifierSimple::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float res = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
    return res;
}

std::vector<float> PrototypeClassifierSimple::softmax(const std::vector<float>& x) {
    std::vector<float> exp_x(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_val);
        sum += exp_x[i];
    }
    for (size_t i = 0; i < x.size(); ++i) exp_x[i] /= sum;
    return exp_x;
}

std::vector<float> PrototypeClassifierSimple::operator()(const std::vector<float>& z) {
    std::vector<float> z_norm = l2_normalize(z);
    std::vector<float> sim(n_classes);
    for (int i = 0; i < n_classes; ++i) {
        std::vector<float> proto_norm = l2_normalize(prototypes[i]);
        sim[i] = dot_product(z_norm, proto_norm) / tau;
    }
    return softmax(sim);
}

void PrototypeClassifierSimple::update_prototypes(const std::vector<std::vector<float>>& features,
                                                  const std::vector<int>& labels) {
    for (int k = 0; k < n_classes; ++k) {
        std::vector<std::vector<float>> class_features;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == k && features[i].size() == (size_t)latent_dim) {
                class_features.push_back(features[i]);
            }
        }
        if (!class_features.empty()) {
            std::vector<float> mean_vec(latent_dim, 0.0f);
            for (size_t m = 0; m < class_features.size(); ++m)
                for (int j = 0; j < latent_dim; ++j)
                    mean_vec[j] += class_features[m][j];
            for (int j = 0; j < latent_dim; ++j) mean_vec[j] /= class_features.size();
            prototypes[k] = mean_vec;
        }
    }
}
