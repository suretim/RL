#include <vector>
#include <array>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>

#include "PlantHVACEnv.h"
#include "fl_client.h"   
#include "hvac_q_agent.h"
#include "config_wifi.h"
#include "version.h"
std::vector<float> health_result;
std::array<int,PORT_CNT> plant_action= {0, 0, 0,0, 0,0, 0,0, 0};

std::vector<std::vector<float>> state_history; // 存储每步的状态
std::vector<float> reward_history;
extern "C" void plant_env_step() {
    PlantHVACEnv env(20, 3, 25.0f, 0.5f, 64);
    
    env.set_seq_fetcher([](int t) -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> seq_input;
        char task_str[32]="seq_input";
        char task_url[128];
        sprintf(task_url, "http://%s:%s/%s", BASE_URL,BASE_PORT,task_str);
        
        

        //sprintf(task_url, "http://%s/seq_input", BASE_URL);
        std::string url =task_url;// "http://192.168.0.57:5000/seq_input";
        bool ok = fetch_seq_from_server(seq_input, url); // 只传 seq_input 和 url
        if(!ok){
            // 默认填充
            seq_input = std::vector<std::vector<float>>(20, std::vector<float>(3, 0.0f));
        }
         
        return seq_input;
    }); 
      
        auto  result = env.step(plant_action);
        std::vector<float> new_state = result.state;   // 当前新的状态
        float reward = result.reward;                   // 当前的奖励值
        bool done = result.done;                        // 是否任务完成
 
        state_history.push_back(new_state);  // 存储当前状态
        reward_history.push_back(reward);    // 存储当前奖励

        // 如果任务完成，可能需要重新初始化环境
        if (done) { 
            std::cout << "Task finished. Resetting environment." << std::endl; 
            env.reset();
        }

        // 你还可以在这里进行其他操作，如更新策略、打印日志等
        std::cout << "New state: ";
        for (auto s : new_state) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Reward: " << reward << std::endl;
        //printf("Step %d, Temp: %.2f, Humid: %.2f, VPD: %.2f, Reward: %.3f\n",
        //       t, result.temp, result.humid, result.vpd, result.reward);
        //if(result.done) break;
        //} 
        std::vector<float> health_result={
            result.reward ,
            result.flower_prob,
            result.temp,
            result.humid ,
            result.light ,
            result.co2 ,
            result.vpd
        };
}
 