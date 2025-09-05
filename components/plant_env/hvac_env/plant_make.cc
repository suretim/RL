#include <vector>
#include <array>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>

#include "PlantHVACEnv.h"
#include "fl_client.h"   
 
std::vector<float> health_result;
std::array<int,4> action= {0, 0, 0,0};


extern "C" void plant_env_step() {
    PlantHVACEnv env(20, 3, 25.0f, 0.5f, 64);
 
    env.set_seq_fetcher([](int t) -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> seq_input;
        std::string url = "http://192.168.0.57:5000/seq_input";
        bool ok = fetch_seq_from_server(seq_input, url); // 只传 seq_input 和 url
        if(!ok){
            // 默认填充
            seq_input = std::vector<std::vector<float>>(20, std::vector<float>(3, 0.0f));
        }
         
        return seq_input;
    }); 
     
    //for(int t=0; t<20; t++){
        //std::array<int,4> action = {t%2, (t+1)%2, t%2, (t+1)%2};
        auto  result = env.step(action);
        //printf("Step %d, Temp: %.2f, Humid: %.2f, VPD: %.2f, Reward: %.3f\n",
        //       t, result.temp, result.humid, result.vpd, result.reward);
        //if(result.done) break;
    //} 
    std::vector<float> health_result={result.reward ,
                                result.flower_prob,
                                result.temp,
                                result.humid ,
                                result.vpd
                            };
}
 