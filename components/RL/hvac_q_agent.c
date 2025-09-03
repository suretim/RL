#include "hvac_q_agent.h"

#include "esp32_ota_client.h"
#include "esp32_model_loader.h"

void setup() {
    Serial.begin(115200);
    WiFi.begin(SSID, PASSWORD);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    
    // 检查并执行OTA更新
    if (otaClient.checkForUpdates()) {
        Serial.println("Update completed successfully");
    }
    
    // 加载新模型
    if (aiModel.loadModel(otaData, otaDataSize)) {
        Serial.println("Model loaded successfully");
    }
}
void loop() {
    // 收集传感器数据
    std::vector<float> observation = readSensors();
    
    // 使用AI模型进行决策
    std::vector<float> action = aiModel.predict(observation);
    
    // 执行动作
    executeAction(action);
    
    // 定期进行持续学习
    if (shouldUpdateModel()) {
        aiModel.continualLearningEWC(collectRecentExperiences());
    }
    
    delay(100);
}

void loop2() {
    // 读取传感器状态
    int health = digitalRead(32);     // 0/1
    int ac_target = digitalRead(33);  // 0/1
    int dehum_target = digitalRead(25); // 0/1
    int light = analogRead(34) / 341; // 0/1/2
    int humidity = analogRead(35) / 341; // 0/1/2

    int state[STATE_DIM] = {health, ac_target, dehum_target, light, humidity};

    // 选择动作
    int action_id = select_action(state);

    int action_bits[2];
    get_action_bits(action_id, action_bits);

    digitalWrite(5, action_bits[0]);
    digitalWrite(18, action_bits[1]);
 

    delay(1000);
}
