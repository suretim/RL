#include <vector>
class PrototypeClassifierSimple;
class HVACEncoder;
class PlantHVACEnv;

extern "C" void plant_env_make_task(void *pvParameters)
{
    plant_env_make();
}

int plant_env_make() {
    // 创建环境
    PlantHVACEnv env(20, 3, 25.0f, 0.5f, 64);
    
    // 重置环境
    auto state = env.reset();
    
    // 准备输入数据
    std::vector<std::vector<float>> seq_input(20, std::vector<float>(3, 0.5f));
    
    // 设置参数
    std::map<std::string, float> params = {
        {"energy_penalty", 0.1f},
        {"switch_penalty_per_toggle", 0.2f},
        {"vpd_target", 1.2f},
        {"vpd_penalty", 2.0f}
    };
    
    // 执行一步
    std::array<int, 4> action = {1, 0, 0, 0}; // AC开启
    auto result = env.step(action, seq_input, params);
    
    // 输出结果
    std::cout << "Reward: " << result.reward << std::endl;
    std::cout << "Done: " << result.done << std::endl;
    std::cout << "Temperature: " << result.temp << std::endl;
    std::cout << "Humidity: " << result.humid << std::endl;
    std::cout << "Flower probability: " << result.flower_prob << std::endl;
    
    return 0;
}