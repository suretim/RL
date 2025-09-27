 
# TFLite 模型文件
 
set(MODEL_META_TFLITE   "../../fed_server/flask/models/meta_model.tflite")
set(MODEL_OPTI_TFLITE   "../../fed_server/flask/models/esp32_optimized_model.tflite")
set(MODEL_ACTOR_TFLITE  "../../fed_server/flask/ppo_model/actor_task0.tflite")
set(MODEL_CRITIC_TFLITE "../../fed_server/flask/ppo_model/critic_task0.tflite")
 
  

set(APPLICATION_SRCS
    "hvac_q_agent.cc"
    "sensor_module.cc"
    "model_pb_handler.cc"
    "config_mqtt.cc"
    "config_wifi.cc"
    "config_spiffs.cc"
    "classifier_storage.cc"
    "infer_esp32_lstm_lll.cc"
    "model.pb.c"
)

# 主组件注册 
idf_component_register(
    SRCS 
        "${APPLICATION_SRCS}"

    INCLUDE_DIRS 
        "." 
        "../nanopb"

    PRIV_INCLUDE_DIRS 
        "."

    EMBED_FILES 
        "${MODEL_META_TFLITE}"
        "${MODEL_OPTI_TFLITE}"
        "${MODEL_ACTOR_TFLITE}"
        "${MODEL_CRITIC_TFLITE}"
    REQUIRES
        spiffs
        esp_http_client
        mbedtls
        ml
        func
        nanopb
    PRIV_REQUIRES 
        tflite-micro 
        nvs_flash
        esp_wifi
        esp_netif
        mqtt        
        console
        json
        esp_timer
            
)

# C++ 特定设置
target_compile_options(${COMPONENT_LIB} PRIVATE 
    "-Wno-unused-variable"
    "-Wno-unused-but-set-variable"
    "-std=c++11"
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE 
    CONFIG_SPIRAM_USE=y
    CONFIG_SPIRAM_IGNORE_NOTFOUND=y
)