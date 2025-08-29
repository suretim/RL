#ifndef SENSOR_MODULE_H
#define SENSOR_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>  

#include "types.h"
#include "define.h"
#include "device_type.h"
#include "type_struct.h"
#include "ai_out.h"

#pragma pack(1)  // 

typedef struct{
    // --------------------- FOR AI insight ---------------------------
	union{
		uint16_t data;
		struct{
			uint16_t high_temp_flag				:1;	//高温状态标志
			uint16_t high_humid_flag 			:1;	//高湿状态标志
			uint16_t grow_light_flag			:1;	//植物灯异常处理标志
			uint16_t clip_fan_flag				:1;	//夹扇异常处理标志
			uint16_t sun_start_flag 			:1; //
			uint16_t low_co2_flag 				:1; //CO2浓度过低, < 400PPM,理想范围:400PPM~1200PPM
			uint16_t smart_co2_monitor_flag 	:1; //CO2浓度和各环境条件都在target范围内，AI光比管道风机以维持现有环境
			uint16_t co2_on_timer_flag			:1; //CO2生成器运行定时时间到，CO2浓度未达到目标值，检查CO2 Regulator或做相应调整
			uint16_t co2_regulator_paused_flag	:1;	//co2浓度超出范围，AI停止运行CO2生成器		
			uint16_t co2_safety_shutoff_flag	:1;	//CO2生成器运行最小时间结束，CO2浓度未达到目标值，检查CO2 Regulator或做相应调整
			uint16_t high_co2_flag	 			:1;	//co2浓度>5000PPM
			uint16_t high_co2_regulator_paused_flag	:1; //co2浓度>5000PPM,停止CO2 Regulator
			uint16_t soil_moisture_alert_flag	:1;	//水泵定时运行时间结束，土壤湿度未变化，请检查洒水系统
			uint16_t water_detect_alert_flag	:1; //水泵定时运行时间结束，未检测到有水，请检查洒水系统
			uint16_t revered					:2;
		}bit;
	}abnormal;	//异常
}ai_insight_run_data_st;

typedef struct{
	uint32_t app_reset_action_bit;
}ai_insight_out_st;

enum{
	INSIGHT_TemperatureWarning	=	0,
	INSIGHT_HumidityWarning,
	INSIGHT_GrowLightLevelDimmed,
	INSIGHT_ClipFanLevelRaised,
	INSIGHT_sunStartcollecting,
	INSIGHT_LowCo2Warning,
	INSIGHT_SmartCo2Monitoring,
	INSIGHT_co2AlertOnTimer,	//
	INSIGHT_co2RegulatorPaused,	
	INSIGHT_co2SafetyShutoff,
	INSIGHT_HighCo2Warning,		//CO2 > 5000ppm
	INSIGHT_Highco2RegulatorPaused,	//
	INSIGHT_SoilMoistureAlert,
	INSIGHT_WaterDetectAlert,	//
	INSIGHT_MAX,
};

typedef struct{
	uint32_t utc[INSIGHT_MAX];
}ai_insight_com_st;

#pragma pack() 

extern ai_insight_run_data_st ai_insight_run_data;
extern ai_insight_com_st ai_insight_com;
extern void ai_insight_set_light_is_ruuning_flag();

extern bool get_aiinsight_be_reset_flag(uint8_t insight_id);
 
void read_all_sensor();
#endif


// 传感器数据结构
typedef struct {
    // 相机数据
    int camera_width;
    int camera_height;
     uint8_t *camera_frame;  // 指向图像数据的指针，数据格式依项目定义
    //esp_image_metadata_t *camera_frame;
    // 温湿度数据
    float temperature ;  // 摄氏度
    float humidity;     // 相对湿度百分比

    // 光照数据
    float light_lux;            // 光照强度，单位lux
} rl_sensor_data_t;

// 初始化传感器模块（相机 + 温湿度 + 光照）
bool sensor_module_init(void);

// 读取所有传感器数据，写入 sensor_data 结构体
bool sensor_module_read(rl_sensor_data_t *data);

// 释放传感器模块资源
void sensor_module_deinit(void);

extern void ml_set_sw(uint8_t port,uint8_t on_off);
extern s16 ml_get_cur_light();
extern s16 ml_get_cur_co2();
extern void ml_read_sensor(void);	
extern s16 ml_get_cur_temp();
extern s16 ml_get_cur_humid();
extern s16 ml_get_outside_temp();
extern s16 ml_get_outside_humid();
extern s16 ml_get_outside_vpd();
extern s16 ml_get_cur_vpd();
extern s16 ml_get_target_temp();
extern s16 ml_get_target_humid();
#ifdef __cplusplus
}
#endif
 