#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <Eigen/Dense> // 需要Eigen库，用于矩阵运算

// 使用Eigen库进行矩阵运算
using Eigen::MatrixXf;
using Eigen::VectorXf;

class PrototypeClassifier {
private:
    int n_classes;
    int latent_dim;
    float tau;
    MatrixXf prototypes;
    std::default_random_engine generator;
    
public:
    PrototypeClassifier(int n_classes, int latent_dim, float tau = 0.1f)
        : n_classes(n_classes), latent_dim(latid_dim), tau(tau) {
        
        // 初始化原型向量
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        prototypes = MatrixXf::Zero(n_classes, latent_dim);
        
        for (int i = 0; i < n_classes; ++i) {
            for (int j = 0; j < latent_dim; ++j) {
                prototypes(i, j) = distribution(generator);
            }
        }
    }
    
    // L2归一化函数
    VectorXf l2_normalize(const VectorXf& v) {
        float norm = v.norm();
        if (norm > 0) {
            return v / norm;
        }
        return v;
    }
    
    // 前向传播
    std::vector<float> operator()(const std::vector<float>& z_vec) {
        // 转换为Eigen向量
        Eigen::Map<const VectorXf> z(z_vec.data(), latent_dim);
        
        // L2归一化输入向量
        VectorXf z_normalized = l2_normalize(z);
        
        // L2归一化原型向量
        MatrixXf protos_normalized(n_classes, latent_dim);
        for (int i = 0; i < n_classes; ++i) {
            protos_normalized.row(i) = l2_normalize(prototypes.row(i));
        }
        
        // 计算相似度
        VectorXf sim = z_normalized.transpose() * protos_normalized.transpose();
        sim = sim / tau;
        
        // Softmax
        VectorXf softmax_result = softmax(sim);
        
        // 转换为std::vector
        std::vector<float> result(n_classes);
        for (int i = 0; i < n_classes; ++i) {
            result[i] = softmax_result(i);
        }
        
        return result;
    }
    
    // Softmax函数
    VectorXf softmax(const VectorXf& x) {
        VectorXf exp_x = (x - x.maxCoeff()).array().exp();
        return exp_x / exp_x.sum();
    }
    
    // 更新原型向量
    void update_prototypes(const std::vector<std::vector<float>>& features, 
                          const std::vector<int>& labels) {
        
        for (int k = 0; k < n_classes; ++k) {
            std::vector<VectorXf> class_features;
            
            // 收集属于类别k的特征
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == k) {
                    Eigen::Map<const VectorXf> feature(features[i].data(), latent_dim);
                    class_features.push_back(feature);
                }
            }
            
            if (!class_features.empty()) {
                // 计算均值向量
                VectorXf mean_vec = VectorXf::Zero(latent_dim);
                for (const auto& feature : class_features) {
                    mean_vec += feature;
                }
                mean_vec /= class_features.size();
                
                // 更新原型
                prototypes.row(k) = mean_vec;
            }
        }
    }
    
    // 获取原型（用于调试）
    const MatrixXf& get_prototypes() const {
        return prototypes;
    }
};