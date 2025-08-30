#include "hvac_q_agent.h"


void loop() {
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
