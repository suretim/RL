#ifndef _ML_RULE_H_
#define _ML_RULE_H_

#include "types.h"
#include "define.h"
#include "device_type.h"
#include "type_struct.h"
#include "ml_pid.h"
#define PID_RULE_EN
extern pid_run_input_st pid_run_input;



#pragma pack(1)
#define DEV_STA_NO_PROBLEM	0
#define DEV_STA_ERR			1
#define DEV_STA_OFF_LINE	2

#define env_dev_origin_cnt	3	//0-原始 1-开关
typedef enum ML_DEV_TYPE
{
	env_dev_type_none = 0,
	env_dev_type_humid,
	env_dev_type_dehumi,
	env_dev_type_heater,
	env_dev_type_ac,
	env_dev_type_inlinefan,
	env_dev_origin_switch,

	ml_dev_type_growlight,
	ml_dev_type_fan,
	ml_dev_type_pump,
	ml_dev_type_water_pump,
	ml_dev_type_co2_generator,
	ml_dev_type_switch,
	ml_dev_type_max,
}ml_dev_type_e;
#define env_dev_type_cnt	(env_dev_type_inlinefan+1)

enum{
	ml_env_temp,
	ml_env_humid,
	ml_env_vpd,
	ml_env_light,
	ml_env_co2,
	ml_env_soil,
	ml_env_water,
	ml_env_max,
};
#define ml_env_base_cnt	(ml_env_vpd+1)

typedef struct{
	struct{
		uint8_t water_pum_over_time		: 1;
		uint8_t co2_genarator_over_time	: 1;
		uint8_t inlinefan_smart_co2		: 1;
		uint8_t co2_off_outside_range	: 1;
		uint8_t co2_over_5000_off		: 1;
	}ai_inisight_bit;
	int8_t inline_fan_env_action[ml_env_base_cnt];	//-1 降低 0-无影响 1-提高
	uint8_t inline_fan_action_info;


	bool temp_help;
	bool humid_help;
	bool vpd_help;
}ml_out_info_t;
extern ml_out_info_t ml_out_info;

// 最小开关时间
typedef struct ST_SW_LIMIT_MG{
	uint8_t dev_keep_sta;			//	设备强行保持状态：0-no  1-off_wait  2-on_wait
	uint16_t dev_wait_sec_cnt;		//最小开关时间计时
}sw_limit_data_t;

typedef struct ST_ai_env_dev_run_data
{
	// run data
	struct{
		uint8_t init_flag		:1;
		uint8_t center_rule_run	:1;
		uint8_t dev_lev_driction:1;
		uint8_t is_center_lev	:1;
		uint8_t is_prf_run		:1;
		uint8_t	reserved		:3;
	}config;
	sw_limit_data_t sw_limit_data;	//设备开关处理

	// 处理挡位变化的参数
	float speed_dev_sec_cnt;		//升降档计时
	uint8_t speed_pref;				//运行prf时的挡位
	uint8_t	speed_of_dev;			//设备当前挡位 - mem
	uint8_t	last_speed_of_dev;		//设备的非中心记忆挡位 - mem
	uint8_t	center_speed_of_dev;	//设备记忆的中心保持挡位 - mem
	uint8_t out_speed;				//设备输出挡位
}ml_env_dev_run_data_t;	//设备运行时的数据

typedef struct ST_env_co2_run_data_t{
	struct{
		uint8_t init_flag	:1;
		uint8_t co2_err 	:1;
		uint8_t user_need_on:1;
		uint8_t	reserved	:6;
	}config;
	uint8_t onoff_sta;		//0-关闭 1-开启
	uint16_t mem_delay;
	uint16_t dev_keep_delay;
	uint8_t speed;
	// 用作计算Co2上升差
	uint16_t run_sec;
	uint16_t co2_increase;
}ml_env_co2_run_data_t;

typedef struct ST_env_water_run_data_t{
	struct{
		uint8_t init_flag	:1;
		uint8_t soil_trig		:1;
		uint8_t water_trig	:1;	//超过最大运行时间
		uint8_t	reserved	:5;
	}config;
	//	水传感器
	uint8_t user_need_on;
	
	uint8_t water_over_lock ;
	uint16_t water_running_cnt ;
	//  土壤传感器
	uint8_t soli_lock ;
	uint16_t soli_running_cnt ;
	uint8_t speed;
}ml_env_water_run_data_t;

typedef struct ST_growlight_rundata{
	struct{
		uint8_t init_flag	:1;
		uint8_t reserve		:7;
	}config;
	sun_t sun;
	uint8_t sun_speed;
}growlight_rundata_t;

//	风扇运行参数
typedef struct ST_fan_run_data{
	uint8_t fan_speed;	//风扇运行参数
}fan_run_data_t;

typedef struct 
{
	uint8_t type;
	uint8_t origin;
	union 
	{
		ml_env_dev_run_data_t env_dev;
		ml_env_co2_run_data_t co2_dev;
		ml_env_water_run_data_t water_dev;
		growlight_rundata_t light_dev;
		fan_run_data_t	fan;
		uint8_t data[30];	//最大数据量
	}dev;

}ml_dev_run_data_t;


typedef struct ST_env_run_data{
	uint8_t is_reach_target		:1;
	uint8_t target_sta_change	:1;
	uint8_t is_reach_pref		:1;

	uint8_t emerg_data;

	uint16_t env_change_delay;		//环境期望修改前 等待确认延时 - init
	uint8_t	env_change_wait_sta;	//在等待环境回到中心值 重启设备的标志 - init
	int16_t env_need_add;			//环境期望的运行设备功效-(偏离target后重置|第一次运行) - init
	int16_t env_diff;				//环境差值 - init
	uint8_t	target_adjust_step;		//环境调节所处步骤 - init
}ml_env_run_data_t;


typedef struct ST_ml_range_data{
	int16_t min;
	int16_t max;
}ml_range_data_t;

typedef struct ST_ml_run_temp_data
{
	struct{
		uint8_t mode_en		:1;
		uint8_t emerg_en	:1;
		uint8_t pref_en		:1;
		uint8_t sensor_sta	:1;
		uint8_t sensor_selct:1;
		uint8_t reseve		:3;
	}config;	//相关环境配置
	int16_t inside_value;
	int16_t outside_value;
	ml_range_data_t set_range;
	ml_range_data_t emerg_range;
	ml_range_data_t pref_range;
}ml_sensor_config_data_t;

typedef struct ST_ml_env_running_data{
	
	//	------ running data ------
	ml_env_run_data_t env_run_data[ml_env_base_cnt];

}ml_env_data_t;


typedef struct ST_ml_dev_run_setting{
	uint8_t type;
	uint8_t origin;
	uint8_t init;
	ml_port_setting_st setting;
}ml_dev_run_setting_t;	//记录用户设置 处理数据重置逻辑

#define MAX_DEV_PARAM_CNT	PORT_CNT
typedef struct ST_ml_running_data
{
	//	--------------------- FOR ML RUN CTR---------------------------
	struct
	{
		uint8_t sw		:1;
		uint8_t is_new		:1;
		uint8_t delt 	:1;
		uint8_t reserve	:5;
	}config;

	//	---------------------- FOR DEV setting ------------------------
	ml_dev_run_setting_t dev_setting_data[MAX_DEV_PARAM_CNT];	//最多只有 PORT CNT 组参数

	//	--------------------- FOR DEV RULER	---------------------------
	ml_dev_run_data_t dev_run_data[MAX_DEV_PARAM_CNT];	//环境通用规则运行数据

	
	ml_sun_param_t* p_ml_sun_param;

	//	--------------------- FOR ENV RULER	---------------------------
	uint8_t	env_dev_have_flag;		//有环境类设备
	uint8_t	env_dev_have_running;	//有环境类设备运行
	uint8_t env_in_range;			//户内环境在设置范围内

	ml_env_data_t env_data;

	//	--------------------	OUT SPEED	--------------------------
	uint8_t out_speed[MAX_DEV_PARAM_CNT];		//对外输出速度
}ml_running_data_t;

typedef struct ST_ml_input_dev_info{
	uint8_t ml_type_of_port;
	uint8_t origin_of_port;
	uint8_t port_sta;	//端口在线状态
	uint8_t min_lev;
	uint8_t max_lev;
	ml_port_setting_st dev_rule_setting;
}ml_input_dev_info_t;

#pragma pack()

extern void ml_set_speed(ml_output_port_t* output_port_list,uint8_t port,uint8_t speed);

extern ml_running_data_t g_ml_running_data;

extern void ml_rest_run_sta( ml_running_data_t *running_data );

extern void ml_rule(ml_input_dev_info_t* input_dev_list, ml_sensor_config_data_t* input_sensor_list, \
						uint8_t sec_flag, Time_Typedef *sys_time, ml_running_data_t *running_data, ml_output_port_t* output_port_list);

#endif
