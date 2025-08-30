
#include <stdio.h>
#include <string.h>

#include "includes.h"

#include "rom/rtc.h"
#include "esp_log.h"
#include "math.h"

#include "ai.h"
#include "ai_out.h" 

#include <stdlib.h>
#include <time.h>

#include "ai_log.h"
#include "ml_grade.h"
#include "ml_dynamic_sun.h"
#include "app_sensor_rule.h"

#include "ai_insight.h"
#include "ml_rule.h"

extern bool smart_co2_monitor_trigger(bool run_minlev,uint8_t cur_lev,uint8_t min_lev,int16_t co2_ppm);

#define TAG     " ml  rule "

// #define ML_RULE_DEBUG_EN
enum{
	INLINE_FAN_IDEL = 0,
	INLINE_FAN_RAISE_TEMP,
	INLINE_FAN_LOWER_TEMP,
	INLINE_FAN_RAISE_HUMID,
	INLINE_FAN_LOWER_HUMID,
	INLINE_FAN_RAISE_VPD,
	INLINE_FAN_LOWER_VPD,
	INLINE_FAN_MAINTAIN_CO2,
	INLINE_FAN_IMPROVE_ENV_DEV,
	INLINE_FAN_RAISE_TEMP_HUMID,
	INLINE_FAN_LOWER_TEMP_HUMID,
	INLINE_FAN_RAISE_TEMP_LOWER_HUMID,
	INLINE_FAN_LOWER_TEMP_RAISE_HUMID,
};

bool ml_sec_flag = 0;

#define log_ml_info(format, ...)		do{ESP_LOGI(TAG, format, ##__VA_ARGS__);}while(0)
#define log_ml_warn(format, ...)		do{ESP_LOGW(TAG, format, ##__VA_ARGS__);}while(0)
#define log_ml_err(format, ...)			do{ESP_LOGE(TAG, format, ##__VA_ARGS__);}while(0)

#ifdef ML_RULE_DEBUG_EN
#define log_ml_debug_i(en_1s,format, ...)		if( ml_sec_flag || en_1s==false ){ESP_LOGI(TAG, format, ##__VA_ARGS__);}
#define log_ml_debug_w(en_1s,format, ...)		if( ml_sec_flag || en_1s==false ){ESP_LOGW(TAG, format, ##__VA_ARGS__);}
#define log_ml_debug_e(en_1s,format, ...)		if( ml_sec_flag || en_1s==false ){ESP_LOGE(TAG, format, ##__VA_ARGS__);}

#else
#define log_ml_debug_i(en_1s,format, ...)
#define log_ml_debug_w(en_1s,format, ...)
#define log_ml_debug_e(en_1s,format, ...)

#endif

enum{
	KEEP_STA_NONE = 0,
	KEEP_STA_ON,
	KEEP_STA_OFF,
};

typedef enum{
	DEV_ACTION_CLOSE = 0,
	DEV_ACTION_DM_LEV,
	DEV_ACTION_UP_LEV,
	DEV_ACTION_KEEP,
}dev_action_e;

const char* dev_name_list[ml_dev_type_max]={
	"none",
	"humi",
	"dehumi",
	"heater",
	"A_C",
	"inlinefan",
	"light",
	"fan",
	"pump",
	"water_pump",
	"co2_generator",
};

const char* env_dev_name_list[env_dev_type_cnt]={
	"none",
	"humi",
	"dehumi",
	"heater",
	"A_C",
	"inlinefan",
};

const char* env_name_list[ml_env_max]={
	"temp",
	"humid",
	"vpd",
	"light",
	"co2",
	"soil",
	"water",
};

enum ENV_EFFECT_EM
{
   ENV_POSITIVE,
   ENV_NEGTIVE,
   ENV_MEDIUM,
};

const char* dev_origin_list[env_dev_origin_cnt]={
	"self",
	"switch",
	"second",
};

typedef enum{
	ml_dev_step_adjust = 0,	//调整过程中
	ml_dev_step_away,	//
	ml_dev_step_reach,	//到达target时 挡位一半
	ml_dev_step_keep,	//到达taregt保持中
	ml_dev_step_wait,	//等待
}ml_adj_step_e;


#pragma pack(1)  // 

typedef struct
{
bool set;
uint8_t val;
uint8_t env_type;

}ai_ret_speed_t;


typedef struct ST_ai_device
{
	// output
	uint8_t action;				//实际的升降挡位控制
	uint8_t action_trend;		//设备运行趋势：0-关闭 1-升档 -1-降档
	uint8_t is_speed_updated;

	// input
	uint8_t type;				//
	uint8_t adj_env_bit_type;	// bit
	uint8_t env_prior_bit;		// 高优先级使能
	uint8_t env_action_step;
	bool multi_effective;
	bool exist;
	bool can_action;
	bool pref_en;
	bool prohibit;			//emergency rule

	int8_t env_factor[ml_env_base_cnt];

	// running change
	uint8_t priority;

}ai_device_t;


#pragma pack() 


//	env_dev_origin_self	, env_dev_origin_switch	, env_dev_origin_seconed
//	设备正常升档间隔表: 处于 未达到Taregt 的升档调节时间
const uint16_t SPEED_RISE_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*60,	1*60,	1*60},
	[env_dev_type_dehumi]		= {1*60,	1*60,	2*60},
	[env_dev_type_heater]		= {2*60,	2*60,	2*60},
	[env_dev_type_ac]			= {5*60,	5*60,	5*60},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
	 
};

//	设备最快调整时间表：到达Target 后的最快调整时间
const uint16_t SPEED_RISE_MAX_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*60,	1*60,	1*60},
	[env_dev_type_dehumi]		= {1*60,	1*60,	2*60},
	[env_dev_type_heater]		= {2*60,	2*60,	2*60},
	[env_dev_type_ac]			= {5*60,	5*60,	5*60},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
};

//	设备最慢调节时间表：到达Target 后的最慢调整时间
const uint16_t SPEED_RISE_MIN_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*30,	2*60,	1*30},
	[env_dev_type_dehumi]		= {1*30,	2*60,	1*30},
	[env_dev_type_heater]		= {1*30,	2*60,	1*30},
	[env_dev_type_ac]			= {1*30,	2*60,	1*30},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
};

//	设备最慢降档间隔表：没有达到Tareget 时的调节时间
const uint16_t SPEED_FALL_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*60,	1*60,	1*60},
	[env_dev_type_dehumi]		= {1*60,	1*60,	1*60},
	[env_dev_type_heater]		= {1*60,	1*60,	1*60},
	[env_dev_type_ac]			= {1*60,	1*60,	1*60},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
};

const uint16_t SPEED_FALL_MAX_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*60,	1*60,	1*60},
	[env_dev_type_dehumi]		= {1*60,	1*60,	1*60},
	[env_dev_type_heater]		= {1*60,	1*60,	1*60},
	[env_dev_type_ac]			= {1*60,	1*60,	1*60},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
};

//	设备最快降档时间表：达到target后的 最快调节时间
const uint16_t SPEED_FALL_MIN_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*30,	1*60,	1*30},
	[env_dev_type_dehumi]		= {1*30,	1*60,	1*30},
	[env_dev_type_heater]		= {1*30,	1*60,	1*30},
	[env_dev_type_ac]			= {1*30,	1*60,	1*30},
	[env_dev_type_inlinefan]	= {2*60,	2*60,	2*60},
};

//	设备关闭最小时间间隔表
const uint16_t SPEED_ON_OFF_INTERVAL[env_dev_type_cnt][env_dev_origin_cnt]={
	[env_dev_type_none]			= {0,		0	,	0	},
	[env_dev_type_humid]		= {1*30,	1*30,	1*30},
	[env_dev_type_dehumi]		= {1*30,	1*60,	1*30},
	[env_dev_type_heater]		= {1*30,	1*60,	1*30},
	[env_dev_type_ac]			= {1*30,	1*60,	1*30},
	[env_dev_type_inlinefan]	= {1*30,	1*60,	1*30},
};


/// @brief 对外输出
ml_out_info_t ml_out_info;


#ifdef __cplusplus
extern "C" {
#endif

extern s16 float_round_up_to_integer(float raw_num);

//	保留精度 0.1
s16 ml_get_cur_temp()
{
	if (is_temp_unit_f())
	    return	float_round_up_to_integer(g_sensors.temp_f.real_val_float)/10;
	else
		return float_round_up_to_integer(g_sensors.temp_c.real_val_float)/10;
}
//	保留精度 0.1
s16 ml_get_target_humid()
{
	return ( pid_run_input.env_target[ENV_HUMID]);
}
//	保留精度 0.1
s16 ml_get_target_temp()
{
	
	
	return ( pid_run_input.env_target[ENV_TEMP]);
}



s16 ml_get_cur_humid()
{
	return float_round_up_to_integer(g_sensors.humid.real_val_float)/10;
}

//	1PPM
s16 ml_get_cur_co2()
{
	return get_sensor_val( CO2 );
}

s16 ml_get_cur_light()
{
	return get_sensor_val( LIGHT );
}

//单位 0.01
s16 ml_get_cur_vpd()
{
	return g_sensors.vpd.read_val/10;
}

s16 ml_get_outside_temp()
{
	return (get_real_zonetemp(is_temp_unit_f())/10);
}

s16 ml_get_outside_humid()
{
	return (get_real_zonehumid()/10);
}

s16 ml_get_outside_vpd()
{
	return get_real_zonevpd();
}

void ml_read_sensor()
{
	for(uint16 i=0; i<50; i++){
		ReadSensor();
	}
}

void ml_set_sw(uint8_t port,uint8_t on_off)
{
	port_switch_onoff(port, on_off > 0 );
}

#ifdef __cplusplus
}
#endif

void ml_run_clean_flag()
{
	ml_out_info.inline_fan_action_info = 0;
	for( uint8_t env_cnt=0; env_cnt<ml_env_base_cnt; env_cnt++ )
	{
		ml_out_info.inline_fan_env_action[env_cnt] = 0;
	}
}

bool get_humid_temp_run_sta(uint8_t* new_action)
{
	uint8_t temp_action = ml_out_info.inline_fan_env_action[ml_env_temp];
	uint8_t humid_action = ml_out_info.inline_fan_env_action[ml_env_humid];
	uint8_t action = *new_action;
	if( 0 == temp_action || 0 == humid_action ){
		return 0;
	}
	if( temp_action == INLINE_FAN_RAISE_TEMP ){
		if( humid_action == INLINE_FAN_LOWER_HUMID ){
			action = INLINE_FAN_RAISE_TEMP_LOWER_HUMID;
		}else if( humid_action == INLINE_FAN_RAISE_HUMID ){
			action = INLINE_FAN_RAISE_TEMP_HUMID;
		}
	}
	if( temp_action == INLINE_FAN_LOWER_TEMP ){
		if( humid_action == INLINE_FAN_LOWER_HUMID ){
			action = INLINE_FAN_LOWER_TEMP_HUMID;
		}else if( humid_action == INLINE_FAN_RAISE_HUMID ){
			action = INLINE_FAN_LOWER_TEMP_RAISE_HUMID;
		}
	}
	*new_action = action;
	return 1;
}

void updata_inline_fan_run_sta(int8_t* env_factor_list,uint8_t env_type,uint8_t useful,uint8_t set_fan_sta)
{
	uint8_t inlin_fan_action_info = INLINE_FAN_IDEL;
	if( env_factor_list != NULL ){
		int8_t wish_factor = env_factor_list[env_type];
		if( false == useful ){
			wish_factor = 0-wish_factor;
		}
		if( env_type == ml_env_temp ){
			inlin_fan_action_info = wish_factor>0 ? INLINE_FAN_RAISE_TEMP:INLINE_FAN_LOWER_TEMP;
		}else if( env_type == ml_env_humid ){
			inlin_fan_action_info = wish_factor>0 ? INLINE_FAN_RAISE_HUMID:INLINE_FAN_LOWER_HUMID;
		}else if( env_type == ml_env_vpd ){
			inlin_fan_action_info = wish_factor>0 ? INLINE_FAN_RAISE_VPD:INLINE_FAN_LOWER_VPD;
		}
		if( env_type < ml_env_base_cnt ){
			ml_out_info.inline_fan_env_action[env_type] = inlin_fan_action_info;
		}
		get_humid_temp_run_sta(&inlin_fan_action_info);
	}
	
	if( set_fan_sta != INLINE_FAN_IDEL ){
		inlin_fan_action_info = set_fan_sta;
	}
	ml_out_info.inline_fan_action_info = inlin_fan_action_info;
}

void updata_inline_fan_action_bit(uint8_t env_type_bit,uint8_t action)
{
	ml_out_info.humid_help = 0;
	ml_out_info.temp_help = 0;
	ml_out_info.vpd_help = 0;

	if( action != DEV_ACTION_UP_LEV ){
		return;
	}
	// if(ml_sec_flag)
	// 	ESP_LOGI(TAG,"action:%d",env_type_bit);
	if( env_type_bit& (1<<ml_env_temp) ){
		ml_out_info.temp_help = 1;	
	}
	if( env_type_bit& (1<<ml_env_humid) ){
		ml_out_info.humid_help = 1;	
	}
	if( env_type_bit& (1<<ml_env_vpd) ){
		ml_out_info.vpd_help = 1;	
	}
}

void clean_ai_insight_bit_info()
{
	memset(&ml_out_info.ai_inisight_bit, 0x00, sizeof(ml_out_info.ai_inisight_bit) );
}

inline void insight_set_water_pump_over_time(bool en)
{
	ml_out_info.ai_inisight_bit.water_pum_over_time = en;
}

inline void insight_set_co2_genarator_over_time(bool en)
{
	ml_out_info.ai_inisight_bit.co2_genarator_over_time = en;
}

inline void insight_set_inlinefan_off_co2_run(bool en)
{
	ml_out_info.ai_inisight_bit.inlinefan_smart_co2 = en;
}

inline void insight_set_co2_off_outside_range(bool en)
{
	ml_out_info.ai_inisight_bit.co2_off_outside_range = en;
}

inline void insight_set_co2_over_5000_off(bool en)
{
	ml_out_info.ai_inisight_bit.co2_over_5000_off = en;
}

static bool water_pupm_lock_is_cleaned()
{
	return get_aiinsight_be_reset_flag(INSIGHT_WaterDetectAlert);
}

static bool soil_pupm_lock_is_cleaned()
{
	return get_aiinsight_be_reset_flag(INSIGHT_SoilMoistureAlert);
}

static bool co2_lock_is_cleaned()
{
	return get_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff);
}


//	获取当前和最大最小区间的差值
static int16_t ai_cal_diff_leave_target(int16_t cur, int16_t min, int16_t max)
{
    if(min > max)
		return 0;

	if(min == cur || max == cur || 0 == max)
		return 0;
	
    if(cur < min)
		return min - cur;
	else if(cur > max)
		return max - cur;
	else
		return 0;
}

//	返回中心差值
static int16_t ai_cal_diff_to_target(int16_t cur, int16_t min, int16_t max)
{
    int16_t center;
	
    if(min > max)
		return 0;

    center = min+ ((max - min) >> 1);
	
	return center - cur;
}

/// center - cur
static int16_t ai_cal_diff(bool has_reached_target, int16_t cur, int16_t min, int16_t max)
{   
   if(has_reached_target)
       return ai_cal_diff_leave_target(cur, min, max);
   else
   	   return ai_cal_diff_to_target(cur, min, max);
}


const int8_t DEV_ENV_FACTOR_INIT[env_dev_type_inlinefan+1][ml_env_base_cnt] =
{
	//  temp   humid   vpd  
	{	0,        0,     0	},	// env_dev_type_none
	{	0,		  1,    -1	},	// env_dev_type_humid
	{	0,	     -1,     1	},	// env_dev_type_dehumi
	{	1,	      0,     1	},	// env_dev_type_heater
	{	-1,	      0,    -1	},	// env_dev_type_ac
	{	0,        0,     0	},	// env_dev_type_inlinefan
};


ml_running_data_t g_ml_running_data;

inline uint8_t env_dev_type_check(uint8_t loadtype)
{
	if(  loadtype >env_dev_type_none && loadtype < env_dev_type_cnt ){
		return loadtype;
	}
	return env_dev_type_none;
}

static void outdoor_affects_inlinefan_factor(int8_t *env_factor,ml_sensor_config_data_t* input_sensor_list)
{
	for( uint8_t i = 0; i < ml_env_base_cnt; i++ ){
		env_factor[i] = 0;
		int16_t out_val = input_sensor_list[i].outside_value;
		int16_t in_val = input_sensor_list[i].inside_value;
		if( out_val > in_val)
			env_factor[i] = 2;
		else if( out_val < in_val)
			env_factor[i] = -2;
	 }
}


void update_devices_env_factor(ai_device_t *ai_dev, ml_sensor_config_data_t* input_sensor_list, bool *is_reach_target)
{
    uint8_t i,j;

	for(i=0; i<ml_dev_type_max; i++)
	{
	    if(!ai_dev[i].can_action)
			continue;
		
		if(env_dev_type_inlinefan == i)
		{
			outdoor_affects_inlinefan_factor(ai_dev[i].env_factor,input_sensor_list);
			//target_affects_inlinefan_factor(ai_dev[i].env_factor, is_reach_target);
		}else{
			for(j=0; j<ml_env_light; j++) 
			{
				if( env_dev_type_none != env_dev_type_check(i) ){
					ai_dev[i].env_factor[j] = DEV_ENV_FACTOR_INIT[i][j];
				}
			}
	    }
	}
}

//	对应环境有调节的设备
bool env_single_have_dev_control(uint8_t env_type,u32 dev_online_bit)
{
	for(uint8_t i = 0; i < ml_dev_type_max; i++)
	{
		if( (dev_online_bit&(1<<i)) == 0 )
			continue;

		if( env_dev_type_none == env_dev_type_check(i) ){
			continue;
		}

		if( DEV_ENV_FACTOR_INIT[i][env_type] != 0 )
			return 1;
	}
	return 0;
}

bool env_have_dev_control(uint8_t env_type_bits,u32 dev_online_bit)
{
	for(uint8_t env_type=0; env_type<ml_env_base_cnt; env_type++){
		if(!((1<<env_type) & env_type_bits))  // skip env_type not relevant to this mode
		   continue;
		if( env_single_have_dev_control(env_type,dev_online_bit) ){
			return 1;
		}
	}
	return 0;
}

//	对应环境有正向调节的设备
bool have_dev_help_single_env(uint8_t env_type,int16_t env_diff,u32 dev_online_bit )
{
	for(uint8_t i = 0; i < ml_dev_type_max; i++)
	{
		if( (dev_online_bit&(1<<i)) == 0 )
			continue;
		if( env_dev_type_none == env_dev_type_check(i) ){
			continue;
		}
		if( env_diff* DEV_ENV_FACTOR_INIT[i][env_type] > 0 )
			return 1;
	}
	return 0;
}

bool dev_helpful_env(uint8_t dev_type,uint8_t env_type,int16_t env_diff)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	if( env_diff* DEV_ENV_FACTOR_INIT[dev_type][env_type] > 0 )
		return 1;
	return 0;
}

// void update_devices_priority(ai_device_t *ai_dev, ml_running_data_t *running_data, uint8_t env_type)
// {
//      uint8_t i;
//      bool inlinefan_higher_priority=false;
 
// 	 for(i=0; i<ml_dev_type_max; i++)
// 	 {
// 	     if(!ai_dev[i].exist)
// 		 	continue;
		 
//          if(ai_dev[env_dev_type_inlinefan].exist && inlinefan_higher_priority)         
// 		     ai_dev[i].priority = (env_dev_type_inlinefan == i);
// 		 else
// 		 	 ai_dev[i].priority = 1;
// 	 }
// }

_unused_ static int16_t cal_target_diff(uint8_t env_type, bool is_reach_target, ml_sensor_config_data_t* input_sensor_list )
{
    int16_t cur_value, target_min, target_max;
	int16_t diff;
	if( env_type >= ml_env_base_cnt ){
		ESP_LOGE(TAG,"%s ml type err!", __func__ );
		return 0;
	}
	ml_sensor_config_data_t* cur_sensor = input_sensor_list+env_type;

	cur_value = cur_sensor->inside_value;
	target_min = cur_sensor->set_range.min;
	target_max = cur_sensor->set_range.max;

    if(0 == target_min && 0 == target_max)
		return 0;
	
	diff = ai_cal_diff(is_reach_target, cur_value, target_min, target_max);

	return diff;
}

bool get_ai_reach_target( ml_running_data_t *running_data, ml_sensor_config_data_t* input_sensor_list )
{
	bool reach_flag = 1;
	
	for(uint8_t env_cnt=ml_env_temp; env_cnt<ml_env_base_cnt; env_cnt++ ){
		ml_sensor_config_data_t* cur_sensor = input_sensor_list+env_cnt;
		ml_env_run_data_t* cur_env_run_data = &(running_data->env_data.env_run_data[env_cnt]);
		if( cur_sensor->config.mode_en == false )
			continue;
		if( cur_env_run_data->is_reach_target == false ){
			reach_flag = 0;
		}
	}

	return reach_flag;
}

//	-1 只有减小的设备	0-都有/没有	1-只有增加的设备
int8_t get_env_is_signal_dev(uint8_t env,ai_device_t* ai_dev)
{
	int8_t ret = 0;
	switch( env )
	{
		case ml_env_temp:
			if( ai_dev[env_dev_type_ac].can_action ){
				ret -= 1;
			}
			if( ai_dev[env_dev_type_heater].can_action ){
				ret += 1;
			}
			break;

		case ml_env_humid:
			if( ai_dev[env_dev_type_dehumi].can_action ){
				ret -= 1;
			}
			if( ai_dev[env_dev_type_humid].can_action ){
				ret += 1;
			}
			break;

		case ml_env_vpd:
			if( ai_dev[env_dev_type_humid].can_action || ai_dev[env_dev_type_ac].can_action ){
				ret -= 1;
			}
			if( ai_dev[env_dev_type_dehumi].can_action || ai_dev[env_dev_type_heater].can_action ){
				ret += 1;
			}
			break;
		default:
			break;
	}
	return ret;
}

#define ENV_DEV_CHANGE_DELAY	2*60
void ai_flash_target_step( ml_running_data_t *running_data, ml_sensor_config_data_t* input_sensor_list, ai_device_t* ai_dev, uint8_t sec_flag )
{
	bool in_range_flag = true;
	for(uint8_t env_cnt=0; env_cnt < ml_env_base_cnt; env_cnt++ )
	{
		ml_env_run_data_t* env_run_data = &(running_data->env_data.env_run_data[env_cnt]);
		uint8_t* p_env_wait_sta	= &(env_run_data->env_change_wait_sta);
		uint8_t* p_target_step 	= &(env_run_data->target_adjust_step);
		int16_t* p_env_wish		= &(env_run_data->env_need_add);
		uint16_t* p_env_delay 	= &(env_run_data->env_change_delay);
		int16_t	 target_diff 	= env_run_data->env_diff;

		ml_sensor_config_data_t* cur_sensor = input_sensor_list+env_cnt;
		int16_t cur_value,target_min,target_max;
		cur_value = cur_sensor->inside_value;
		target_min = cur_sensor->set_range.min;
		target_max = cur_sensor->set_range.max;

		int16_t	 range_diff	 	= ai_cal_diff(true, cur_value, target_min, target_max);	//inside_range_diff_list[env_cnt];

		if( 0 == cur_sensor->config.mode_en ){
			(*p_target_step) = 0;
			(*p_env_wish) = 0;
			(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
			(*p_env_wait_sta) = 0;
			continue;
		}

		if( range_diff != 0 ){
			in_range_flag = false;
		}

		if( env_run_data->is_reach_target ){
			(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
			(*p_env_wait_sta) = 0;
			if( (*p_target_step) != ml_dev_step_keep ){
				(*p_target_step) = ml_dev_step_reach;	//执行中心挡位减半的触发
			}
			continue;
		}
		if( (*p_target_step) != ml_dev_step_adjust ){
			(*p_target_step) = ml_dev_step_away;	//执行退出中心挡位的触发
		}

		//	环境需求切换的处理
		s8 ret = get_env_is_signal_dev(env_cnt,ai_dev);	//当前处于单环境设备状态
		if( ret != 0 ){		//单环境设备场景
			//	对立设备工作时掉线处理
			if( (*p_env_wish)*ret < 0 ){
				(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
				(*p_env_wait_sta) = 1;
			}
			(*p_env_wish) = ret;
		}

		if( (*p_env_wish) == 0 ){
			(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
			(*p_env_wish) = target_diff;
			(*p_env_wait_sta) = 0;
		}
		
		if( (*p_env_wish)*range_diff < 0 ){
			if( (*p_env_wait_sta) == 0 ){
				(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
			}
			(*p_env_wait_sta) = 1;
		}else{
			(*p_env_delay) = ENV_DEV_CHANGE_DELAY;
		}

		if( (*p_env_wait_sta) != 0 ){
			//	超时恢复 - 切换环境需求
			if( (*p_env_delay) ){
				if( sec_flag ){
					(*p_env_delay)--;
					// ESP_LOGI(TAG,"%s wait:%d",env_name_list[env_cnt],(*p_env_delay));
				}
			}else{
				//	立马切换
				if( ret == 0 ){	//非单环境设备场景允许切换
					(*p_env_wish) = target_diff;
					(*p_env_wait_sta) = 0;
				}
			}
		}

		if( (*p_env_wait_sta) != 0 ){
			//	恢复环境需求 退出等待
			if( (*p_env_wish)*target_diff >= 0 ){
				//	回到中点之下
				(*p_env_wait_sta) = 0;
				(*p_env_wish) = target_diff;
			}
		}
		// if((*p_env_wait_sta)) 
		{ log_ml_debug_i(1, "ret(%d) wait[%d] wish[%d]:%d delay:%d", ret, env_cnt, (*p_env_wait_sta), (*p_env_wish), *p_env_delay); }
	}
	running_data->env_in_range = in_range_flag;
}

void ai_target_step_auto(ml_running_data_t *running_data)
{
	for(uint8_t env_cnt=0; env_cnt < ml_env_base_cnt; env_cnt++ )
	{
		ml_env_run_data_t* env_run_data = &(running_data->env_data.env_run_data[env_cnt]);
		if( env_run_data->target_adjust_step == ml_dev_step_reach )
			env_run_data->target_adjust_step = ml_dev_step_keep;

		if( env_run_data->target_adjust_step == ml_dev_step_away )
			env_run_data->target_adjust_step = ml_dev_step_adjust;
	}
}

// inline uint8_t get_env_target_step(ml_running_data_t *running_data,uint8_t env_type)
// {
// 	if(  env_type >= ml_env_base_cnt )
// 		env_type = 0;
// 	return running_data->env_data.env_run_data[env_type].target_adjust_step;
// }

// inline void set_env_target_step(ml_running_data_t *running_data,uint8_t env_type,uint8_t step)
// {
// 	running_data->env_data.env_run_data[env_type].target_adjust_step = step;
// }

// uint8_t get_dev_target_step(ai_device_t* ai_dev,ml_running_data_t *running_data,uint16_t env_bit,uint8_t* get_wait_sta)
// {
// 	// uint16_t env_bit = 0;
// 	for( uint8_t env_cnt=0; env_cnt < ml_env_base_cnt; env_cnt++ )
// 	{
// 		ml_env_run_data_t* env_run_data = &(running_data->env_data.env_run_data[env_cnt]);
// 		if( 0 == (env_bit & (1<<env_cnt)) )
// 			continue;
// 		if( ai_dev->env_factor[env_cnt] != 0 ){
// 			*get_wait_sta  = env_run_data->env_change_wait_sta;

// 			return env_run_data->target_adjust_step;
// 		}
// 	}
// 	return 0;
// }

void get_ai_outside_range_diff(int16_t *outside_diff_list,ml_sensor_config_data_t* input_sensor_list,ml_running_data_t *running_data)
{
	ml_sensor_config_data_t* cur_sensor;

	cur_sensor = input_sensor_list+ml_env_temp;
	outside_diff_list[ml_env_temp] = ai_cal_diff(true,cur_sensor->inside_value,\
													cur_sensor->set_range.min,cur_sensor->set_range.max);

	cur_sensor = input_sensor_list+ml_env_humid;
	outside_diff_list[ml_env_humid] = ai_cal_diff(true,cur_sensor->inside_value,\
													cur_sensor->set_range.min,cur_sensor->set_range.max);

	cur_sensor = input_sensor_list+ml_env_vpd;
	outside_diff_list[ml_env_vpd] = ai_cal_diff(true,cur_sensor->inside_value,\
													cur_sensor->set_range.min,cur_sensor->set_range.max);
}

//	close up dowm keep
// 
static uint8_t get_device_adjust_action(uint8_t env_type_bits, int8_t *env_factor, int16_t* env_wish, \
										int16_t *target_diff_list,bool is_check_range ) 
{
    uint8_t env_type,action_type;
	bool dev_take_effect;

	action_type = DEV_ACTION_KEEP;
	dev_take_effect = false;
	
	for(env_type=0; env_type<ml_env_base_cnt; env_type++)
	{
		if(!((1<<env_type) & env_type_bits))  // skip env_type not relevant to this mode
			continue;

		if(0 == env_factor[env_type])	// skip env_type the device cannot take effect on
			continue;
			
		if( 0 == env_wish[env_type] )	// skip env rule need not dev
			continue;
		// log_ml_debug_i(1," wish[%d]:%d", env_type, env_wish[env_type] );
		dev_take_effect = true;
		if( env_factor[env_type] * env_wish[env_type] < 0 /*|| env_wish[env_type] == 0*/ )
		{
			action_type = DEV_ACTION_CLOSE;
			break;
		}
		else if(target_diff_list[env_type] * env_factor[env_type] > 0)  // priority mid, open action: device helpful to target
		{
			action_type = DEV_ACTION_UP_LEV;
		}
		else if(target_diff_list[env_type] * env_factor[env_type] < 0)	// priority high, close action: device deviating from target
		{
			action_type = DEV_ACTION_DM_LEV;
			break;
		}
		else 														  // priority low, close/no action:  diff=0,  [target_min, target_max] 
			action_type *= 1;
	}

	if(dev_take_effect){
	    return action_type;
	}
	else if( is_check_range ){
		return DEV_ACTION_KEEP;
	}
	return DEV_ACTION_CLOSE;
}

static void updata_inlinefan_run_sta(uint8_t env_type_bits, uint8_t action, int8_t *env_factor, int16_t *target_diff_list, uint8_t* env_help_bit)
{
	uint8_t env_type = 0;
	*env_help_bit = 0;
	for(env_type=0; env_type<ml_env_base_cnt; env_type++)
	{
	   if(!((1<<env_type) & env_type_bits))  // skip env_type not relevant to this mode
		   continue;
	   
	   if(0 == env_factor[env_type])	// skip env_type the device cannot take effect on
		   continue;
		
		//	更新 挡位变化原因
		if( target_diff_list[env_type] * env_factor[env_type] >= 0 ){
			*env_help_bit |= (1<<env_type);
			if( action == DEV_ACTION_UP_LEV || action == DEV_ACTION_KEEP ){
				updata_inline_fan_run_sta(env_factor,env_type,1,INLINE_FAN_IDEL);
			}
		}else{
			if( action == DEV_ACTION_DM_LEV || action == DEV_ACTION_CLOSE ){
				updata_inline_fan_run_sta(env_factor,env_type,0,INLINE_FAN_IDEL);
			}
		}
	}
}

/*
*	int16_t *target_outside_diff_list: data of senseor exceed the setting range
*/
static uint8_t get_inlinefan_adjust_action(uint8_t env_type_bits, int8_t *env_factor, int16_t *target_diff_list,
										int16_t* inside_range_diff_list, uint8_t env_have_dev_flag) 
{
    uint8_t env_type,action_type;
	_unused_ bool env_single_flag;
	bool dev_take_effect;

	action_type = DEV_ACTION_CLOSE;
	dev_take_effect = false;
	env_single_flag = (env_type_bits&(env_type_bits-1))==0?1:0;
	for(env_type=0; env_type<ml_env_base_cnt; env_type++)
	{
	   if(!((1<<env_type) & env_type_bits))  // skip env_type not relevant to this mode
		   continue;
	   
	   if(0 == env_factor[env_type])	// skip env_type the device cannot take effect on
		   continue;

	   dev_take_effect = true;
	   
	   log_ml_debug_i(1,"inlinFan env:%d factor:%d diff:%d", env_type, env_factor[env_type], target_diff_list[env_type] );
	   
	   if(target_diff_list[env_type] * env_factor[env_type] >= 0)  // open action: device helpful to target
	   {
			action_type = DEV_ACTION_UP_LEV;
			if( env_have_dev_flag == false ){
				break;
			}
	   }else if(target_diff_list[env_type] * env_factor[env_type] < 0)	// close action: device deviating from target
	   {
			if( env_have_dev_flag /*env_single_have_dev_control(env_type,target_diff_list[env_type],dev_bit)*/ ){
				// ESP_LOGW(TAG,"env:%d is_single:%d range:%d",env_type,env_single_flag,inside_range_diff_list[env_type]);
				if( env_single_flag == false && 0 == inside_range_diff_list[env_type] ){
					if( action_type != DEV_ACTION_UP_LEV ){
						action_type = DEV_ACTION_CLOSE;
					}
				}else{
					action_type = DEV_ACTION_CLOSE;
					break;
				}
			}else{
				action_type = DEV_ACTION_DM_LEV;
			}
	   }
	   else 														  // priority low, close/no action:  diff=0,  [target_min, target_max] 
		   action_type *= 1;
	}

	if(dev_take_effect)
	    return action_type;
	else
		return DEV_ACTION_KEEP;
}

// step 1 -- target 环境的需求处理 
// 输出 target 环境规则的需求
void sensor_target_rule_run(ml_sensor_config_data_t* input_sensor_list,ml_running_data_t *running_data, \
							ai_device_t* ai_dev, int16_t* env_need_list,uint8_t sec_flag)
{
	int16_t target_diff[ml_env_base_cnt];
	for( uint8_t env=0; env<ml_env_base_cnt; env++ ){
		// 1、环境 target 需求逻辑处理
		ml_env_run_data_t* cur_env_run_data = &(running_data->env_data.env_run_data[env]);
		uint8_t new_target_sta = cur_env_run_data->is_reach_target;
		ml_sensor_config_data_t* cur_sens_data = input_sensor_list+env;
		if( !cur_sens_data->config.mode_en ){
			continue;
		}

		target_diff[env] = ai_cal_diff(cur_env_run_data->is_reach_target, cur_sens_data->inside_value, \
											cur_sens_data->set_range.min, cur_sens_data->set_range.max);
		if( cur_env_run_data->env_diff !=0 && cur_env_run_data->env_diff*target_diff[env]<=0 ){
			if( ai_is_setting_updata() == 0 ){
				new_target_sta = 1;
				target_diff[env] = 0;
			}
		}
		if( target_diff[env] != 0 ) {
			new_target_sta = 0;
		}
		cur_env_run_data->target_sta_change = 0;
		if( new_target_sta != cur_env_run_data->is_reach_target ){
			cur_env_run_data->is_reach_target = new_target_sta;
			cur_env_run_data->target_sta_change = 1;
			ESP_LOGI(TAG,"sense[%d] target change:%d",env, new_target_sta );
		}
		cur_env_run_data->env_diff = target_diff[env];
	}

	ai_flash_target_step(running_data,input_sensor_list,ai_dev,sec_flag);

	for( uint8_t env=0; env<ml_env_base_cnt; env++ ){
		ml_env_run_data_t* env_run_data = &(running_data->env_data.env_run_data[env]);
		env_need_list[env] = env_run_data->env_need_add;
		if( env_run_data->env_change_wait_sta ){
			env_need_list[env] = 0;
		}
	}
	// running_data->env_dev_have_flag = env_have_dev_control( env_type_bits, dev_bit );	//有没有对应的环境设备控制
}

// step2 - 环境偏好规则处理
// 在VPD模式下 到达 vpd 目标后 检查有设置偏好选项 触发
// 这个时候会以 偏好为目标 直到达到偏好目标范围
// out 
void sensor_pref_rule_run( ml_sensor_config_data_t* input_sensor_list,ml_running_data_t *running_data, int16_t* env_need_list )
{
	int16_t target_diff[ml_env_base_cnt];
	bool vpd_reach_taregt = running_data->env_data.env_run_data[ml_env_vpd].is_reach_target;
	for( uint8_t env=0; env<ml_env_base_cnt; env++ ){
		ml_env_run_data_t* cur_env_run_data = &(running_data->env_data.env_run_data[env]);
		ml_sensor_config_data_t* cur_sens_data = input_sensor_list+env;

		if( !cur_sens_data->config.pref_en || !vpd_reach_taregt ){
			continue;
		}
		target_diff[env] = ai_cal_diff(true, cur_sens_data->inside_value, \
										cur_sens_data->pref_range.min, cur_sens_data->pref_range.max);
		cur_env_run_data->is_reach_pref = (target_diff[env] == 0);
		if( target_diff[env] != 0 ){
			env_need_list[env] = target_diff[env];
		}
	}
}

/// 环境紧急关闭规则
/// 在达到设置紧急关闭值时 关闭环境对应需求
void sensor_emerge_rule_run( ml_sensor_config_data_t* input_sensor_list,ml_running_data_t *running_data, int16_t* env_need_list )
{
	int16_t target_diff[ml_env_base_cnt];
	for( uint8_t env=0; env<ml_env_base_cnt; env++ ){
		ml_env_run_data_t* cur_env_run_data = &(running_data->env_data.env_run_data[env]);
		ml_sensor_config_data_t* cur_sens_data = input_sensor_list+env;
		cur_env_run_data->emerg_data = 0;
		if( !cur_sens_data->config.emerg_en || input_sensor_list[ml_env_vpd].config.mode_en == false ){
			continue;
		}
		target_diff[env] = ai_cal_diff(true, cur_sens_data->inside_value, \
											cur_sens_data->emerg_range.min, cur_sens_data->emerg_range.max);
		
		cur_env_run_data->emerg_data = target_diff[env];
		log_ml_debug_w(1,"emerg[%d] %d min:%d max:%d",env,cur_env_run_data->emerg_data,cur_sens_data->emerg_range.min,cur_sens_data->emerg_range.max );
	}
}

/// @brief 环境规则运行逻辑
/// @param 输出
void sensor_param_rule(ml_sensor_config_data_t* input_sensor_list,ml_running_data_t *running_data, ai_device_t* ai_dev, int16_t* env_need_list,uint8_t sec_flag)
{
	// 1、目标环境需求 处理逻辑
	sensor_target_rule_run(input_sensor_list, running_data, ai_dev, env_need_list, sec_flag );
	// 2、环境 偏好 需求逻辑处理
	sensor_pref_rule_run(input_sensor_list, running_data, env_need_list);
	// 3、环境 紧急制动逻辑处理
	sensor_emerge_rule_run(input_sensor_list, running_data, env_need_list);
	// 环境需求更新完毕
}

/// 设备生效环境更新逻辑 根据设定 确定设备生效的环境
// adj_env_bit_type
void dev_env_ctr_updata( ai_device_t *ai_dev_list, ml_running_data_t *running_data, ml_sensor_config_data_t* input_sensor_list )
{
	bool vpd_reach_target = running_data->env_data.env_run_data[ml_env_vpd].is_reach_target;
	bool is_temp_humid_mode = ( input_sensor_list[ml_env_temp].config.mode_en && input_sensor_list[ml_env_humid].config.mode_en );

	for(uint8_t dev = 0; dev < env_dev_type_cnt; dev++){
		ai_device_t* cur_dev = ai_dev_list + dev;
		uint16_t dev_env_type = 0;
		uint16_t dev_pref_type = 0;
		uint8_t prohibit = 0;
		for( uint8_t env=0; env< ml_env_base_cnt; env++ ){
			ml_sensor_config_data_t* cur_sens_data = input_sensor_list+env;
			ml_env_run_data_t* cur_sens_run = running_data->env_data.env_run_data+env;
			if( cur_dev->env_factor[env] == 0 ){
				continue;
			}

			if( cur_sens_data->config.mode_en ){
				dev_env_type |= (1<<env);
			}

			if( dev == env_dev_type_inlinefan ){
				continue;
			}
			
			if( (cur_sens_run->emerg_data)*(cur_dev->env_factor[env]) < 0 ){
				prohibit = 1;
			}

			if( !cur_sens_data->config.pref_en || !vpd_reach_target || cur_sens_run->is_reach_pref ){
				continue;
			}
			dev_pref_type |= (1<<env);
		}
		cur_dev->prohibit = prohibit;
		cur_dev->pref_en = ( dev_pref_type != 0? 1: 0);
		
		if( dev == env_dev_type_inlinefan && is_temp_humid_mode ){
			cur_dev->adj_env_bit_type = ( cur_dev->env_prior_bit != 0? cur_dev->env_prior_bit : dev_env_type); 
		}else{
			cur_dev->adj_env_bit_type = ( dev_pref_type != 0? dev_pref_type: dev_env_type);
		}
		log_ml_debug_i(1,"dev[%s] prf[%d] value:%d prohb[%d]", dev_name_list[dev], dev_pref_type, cur_dev->adj_env_bit_type,prohibit);
	}
}

//	通过输入设备 找到对立设备
uint8_t get_opposition_dev(uint8_t dev_type)
{
	uint8_t ret = env_dev_type_none;
	switch(dev_type)
	{
		case env_dev_type_humid:
			ret = env_dev_type_dehumi;
			break;
		case env_dev_type_dehumi:
			ret = env_dev_type_humid;
			break;
		case env_dev_type_heater:
			ret = env_dev_type_ac;
			break;
		case env_dev_type_ac:
			ret = env_dev_type_heater;
			break;
		default:
			break;
	}
	return ret;
}

bool opposition_dev_is_running( uint8_t dev_type, ml_dev_run_data_t *dev_run_data_num_list )
{
	extern bool get_dev_type_is_constrain_on(ml_dev_run_data_t *dev_run_data_num_list,uint8_t ml_dev_type);
	uint8_t opps_dev = get_opposition_dev(dev_type);
	if( opps_dev == env_dev_type_none ){
		return false;
	}
	return get_dev_type_is_constrain_on( dev_run_data_num_list, opps_dev );
}

uint8_t get_dev_env_adj_step(uint8_t env_bit, ml_running_data_t *running_data )
{
	uint8_t env_action = ml_dev_step_adjust;
	for(uint8_t env =0; env<ml_env_base_cnt; env++){
		if( 0 == (env_bit&(1<<env)) ){
			continue;
		}
		env_action = running_data->env_data.env_run_data[env].target_adjust_step;
	}
	return env_action;
}

//	设备动作处理逻辑
void dev_action_rule(ai_device_t *ai_dev,ml_running_data_t *running_data,ml_sensor_config_data_t* input_sensor_list,int16_t* env_need_list,uint8_t sec_flag)
{
	int16_t target_diff[ml_env_base_cnt];	//环境差值
	int16_t inside_range_diff[ml_env_base_cnt];		//户内值和设置范围的差
	for( uint8_t env=0; env<ml_env_base_cnt; env++ ){
		// 1、环境 target 需求逻辑处理
		// ml_env_run_data_t* cur_env_run_data = &(running_data->env_data.env_run_data[env]);
		ml_sensor_config_data_t* cur_sens_data = input_sensor_list+env;
		target_diff[env] = 0;
		if( cur_sens_data->config.mode_en ){
			target_diff[env] = ai_cal_diff( false, cur_sens_data->inside_value, \
											cur_sens_data->set_range.min, cur_sens_data->set_range.max);
		}else if( cur_sens_data->config.pref_en ){
			target_diff[env] = ai_cal_diff( true, cur_sens_data->inside_value, \
				cur_sens_data->pref_range.min, cur_sens_data->pref_range.max );
		}
		inside_range_diff[env] = ai_cal_diff( true, cur_sens_data->inside_value, \
			cur_sens_data->set_range.min, cur_sens_data->set_range.max);
	}

	// 1、 环境类设备规则
	// 设备&环境 处理逻辑
	// 设备互斥  处理逻辑
	bool reach_target = get_ai_reach_target( running_data, input_sensor_list );	//确定控制的环境都达到 target
	// running_data->env_dev_have_flag = env_have_dev_control( env_type_bits, dev_bit );	//有没有对应的环境设备控制
	log_ml_debug_w( 1, "target: %d", reach_target );

	for( uint8_t dev_type=0; dev_type< env_dev_type_cnt; dev_type++ ){
		ai_device_t* cur_dev = ai_dev+dev_type;
		uint8_t action_type = 0;
		if( !cur_dev->exist ){	// skip non-existed devices
			if( dev_type == env_dev_type_inlinefan ){
				updata_inline_fan_action_bit(0,DEV_ACTION_CLOSE);
			}
			continue;
		}

		// if( !ai_dev[dev_type].can_action )	// skip non-existed devices
		// 	continue;

		//	设备 环境处理逻辑
		if( dev_type == env_dev_type_inlinefan ){
			uint8_t help_env_bit = 0;
			action_type =get_inlinefan_adjust_action( cur_dev->adj_env_bit_type, cur_dev->env_factor, target_diff, inside_range_diff, running_data->env_dev_have_running );
			// if( reach_target && running_data->env_dev_have_running ){
			// 	action_type = DEV_ACTION_CLOSE;
			// }
			updata_inlinefan_run_sta(cur_dev->adj_env_bit_type, action_type, cur_dev->env_factor, target_diff, &help_env_bit );
			//	只有环境设备能触发关闭
		#if 0
			if( action_type == DEV_ACTION_CLOSE && running_data->env_dev_have_running ){
				updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_IMPROVE_ENV_DEV);
			}
		#endif
			updata_inline_fan_action_bit( help_env_bit, action_type );
		}else{
			action_type = get_device_adjust_action( cur_dev->adj_env_bit_type, cur_dev->env_factor, env_need_list, target_diff, 0 );	//设备
		}

		cur_dev->is_speed_updated = true;
		cur_dev->action_trend = action_type;

		//	互斥设备逻辑
		if( opposition_dev_is_running(dev_type,running_data->dev_run_data) || cur_dev->prohibit ){
			action_type = DEV_ACTION_CLOSE;
		}

		cur_dev->action = action_type;
		cur_dev->env_action_step = get_dev_env_adj_step( cur_dev->adj_env_bit_type, running_data );
		if( cur_dev->pref_en ){
			cur_dev->env_action_step = ml_dev_step_adjust;
		}
		
		log_ml_debug_w( 1, "%s dev action %d, step %d", dev_name_list[dev_type], cur_dev->action, cur_dev->env_action_step);
	}
}


// only_check_range：device will turn off if needed(out of range) / can not open device
static void ai_sensor_adjust_rule( ml_sensor_config_data_t* input_sensor_list, ai_device_t *ai_dev, 
									ml_running_data_t *running_data, uint8_t sec_flag, bool only_check_range)
{
	int16_t target_wish[ml_env_base_cnt];					//与到达taregt有关系的差

	//	vpd模式下更行 运行的参数
	sensor_param_rule(input_sensor_list, running_data, ai_dev, target_wish, sec_flag);

	dev_env_ctr_updata(ai_dev, running_data, input_sensor_list );
	
	// get_ai_outside_range_diff(outside_range_diff,outside_sensor_val_list,running_data);

	dev_action_rule( ai_dev, running_data, input_sensor_list, target_wish, sec_flag );

}


bool dev_is_environment_dev(uint8_t env_dev_type)
{
	switch(env_dev_type){
		case env_dev_type_humid:
		case env_dev_type_dehumi:
		case env_dev_type_heater:
		case env_dev_type_ac:
			return true;
			break;
		default: return false;
			break;
	}
	return false;
}

//	是否是需要控制的环境指标
bool get_env_setting_need_ctr(uint8_t env, ml_sensor_config_data_t* input_sensor_list)
{
	bool env_need = false;
	if( input_sensor_list[ml_env_vpd].config.mode_en ){
		if( env == ml_env_vpd )
			env_need = true;
	}
	if( input_sensor_list[ml_env_temp].config.mode_en ){
		if( env == ml_env_temp )
			env_need = true;
	}
	if( input_sensor_list[ml_env_humid].config.mode_en ){
		if( env == ml_env_humid )
			env_need = true;
	}
	return env_need;
}

//	判断当前设备对 当前使能的那个环境指标 有效
uint8_t get_env_is_dev_effect(uint8_t dev_type,ml_sensor_config_data_t* input_sensor_list)
{
	for(uint8_t i=0; i<ml_env_base_cnt; i++)
	{
		if( env_dev_type_none == env_dev_type_check(dev_type) ){
			return 0;
		}
		if( DEV_ENV_FACTOR_INIT[dev_type][i] != 0 && get_env_setting_need_ctr(i,input_sensor_list) )
			return i;
	}
	return ml_env_base_cnt;
}

void get_env_target_data_param(uint8_t env,ml_sensor_config_data_t* input_sensor_list,int16_t* p_cur,int16_t* p_min,int16_t* p_max)
{
	int16_t target_min=0,target_max=0;
	if( env >= ml_env_max ){
		ESP_LOGE(TAG,"%s env err %d", __func__, env );
		return;
	}

	ml_sensor_config_data_t* cur_sensor = input_sensor_list+env;
	if( cur_sensor->config.mode_en ){
		target_min = cur_sensor->set_range.min;
		target_max = cur_sensor->set_range.max;
	}else if( cur_sensor->config.pref_en ){
		target_min = cur_sensor->pref_range.min;
		target_max = cur_sensor->pref_range.max;
	}

	*p_min = target_min;
	*p_max = target_max;
	*p_cur = cur_sensor->inside_value;
}


float Proportional_time_calculation(uint16_t cur,uint16_t trg_min,uint16_t trg_max,uint16_t tim_min,uint16_t tim_max)
{
	float v_min,v_max;
	v_min = 1.0/tim_max;
	v_max = 1.0/tim_min;
	if( v_min == v_max || trg_min == trg_max )
		return 1;

	uint16_t t_center = (trg_min + trg_max)/2;
	float v = (float)abs(t_center - cur)/(float)abs(t_center - trg_min)*(v_max - v_min)+v_min;

	return v*tim_max;
}


/////////////////////////////////////////////////////////////////////////////////////////
//	DEV 接口 
//	on_off_wait: 0-no  1-off_wait  2-on_wait
inline void set_limit_sw_sta(sw_limit_data_t *limit_data,uint8_t on_off_wait)
{
	limit_data->dev_keep_sta = on_off_wait;
}

inline uint8_t get_limit_sw_sta(sw_limit_data_t *limit_data)
{
	return limit_data->dev_keep_sta;
}

// 0是挡位加计时 1是挡位减计时 
inline uint8_t get_origin_dev_timer_cnt_direction(ml_env_dev_run_data_t *basic_running_data)
{
	return basic_running_data->config.dev_lev_driction;
}

inline void set_origin_dev_timer_cnt_direction(ml_env_dev_run_data_t *basic_running_data,bool dt)
{
	basic_running_data->config.dev_lev_driction = dt;
}

inline uint8_t get_origin_dev_have_center_lev(ml_env_dev_run_data_t *basic_running_data)
{
	return basic_running_data->config.center_rule_run;
}

inline void set_origin_dev_have_center_lev(ml_env_dev_run_data_t *basic_running_data,bool dt)
{
	basic_running_data->config.center_rule_run = dt;
}

//	针对单一种类的环境清除
void ml_clear_env_dev_run_data( ml_env_dev_run_data_t *cur_env_dev_run_data, uint8_t init_lev, uint8_t env_dev_type )
{
	if( env_dev_type >= env_dev_type_cnt ){
		ESP_LOGE(TAG,"env type oversize(%d)",env_dev_type);
		return;
	}
	cur_env_dev_run_data->speed_dev_sec_cnt = 0;
	cur_env_dev_run_data->speed_of_dev = init_lev;
	if( false == dev_is_environment_dev(env_dev_type) ){
		cur_env_dev_run_data->speed_of_dev = 0;
	}
	cur_env_dev_run_data->last_speed_of_dev = 0;
	cur_env_dev_run_data->center_speed_of_dev = 0;
	cur_env_dev_run_data->sw_limit_data.dev_wait_sec_cnt = 0;
	cur_env_dev_run_data->config.is_center_lev = 0;
	set_origin_dev_timer_cnt_direction(cur_env_dev_run_data,0);
	set_limit_sw_sta( &(cur_env_dev_run_data->sw_limit_data), KEEP_STA_NONE );
	set_origin_dev_have_center_lev(cur_env_dev_run_data,0);
}

uint16_t get_dev_general_rise_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_RISE_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_general_down_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_FALL_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_rise_min_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_RISE_MIN_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_rise_max_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_RISE_MAX_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_down_min_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_FALL_MIN_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_down_max_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	return SPEED_FALL_MAX_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

// ----------- 强制开关时间 ------------
uint16_t get_dev_limit_off_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	if( origin != dev_type && origin == env_dev_type_ac ){
		dev_type = origin;
	}
	return SPEED_ON_OFF_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

uint16_t get_dev_limit_on_tim(uint8_t dev_type, uint8_t origin)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return 0;
	}
	if( origin != dev_type && origin == env_dev_type_ac ){
		dev_type = origin;
	}
	return SPEED_ON_OFF_INTERVAL[dev_type][origin==ml_dev_type_switch];
}

//	keep_sta: KEEP_STA_NONE
void set_dev_onoff_keep_delay(ml_env_dev_run_data_t *env_run_data,uint8_t env_dev, uint16_t min_on, uint16_t min_off, uint8_t keep_sta )
{
	//	没有过最小关闭延时
	uint16_t cnt;
	if( keep_sta == KEEP_STA_ON ){
		cnt = min_on;	//SPEED_ON_OFF_INTERVAL[env_dev][dev_origin];
	}else if( keep_sta == KEEP_STA_OFF ){
		cnt = min_off;	//SPEED_ON_OFF_INTERVAL[env_dev][dev_origin];	
	}else{
		keep_sta = 0;
		cnt = 0;
	}
	//	上一个等待时间结束 才能开启新的等待时间
	if( env_run_data->sw_limit_data.dev_wait_sec_cnt != 0 )
		return;

	if( keep_sta != get_limit_sw_sta(&(env_run_data->sw_limit_data)) ){
		set_limit_sw_sta(&(env_run_data->sw_limit_data),keep_sta);
		env_run_data->sw_limit_data.dev_wait_sec_cnt =  cnt;
		// log_ml_debug_i(0,"new keep sta :%d", keep_sta);

		// running_data->speed_dev_sec_cnt[dev_type][dev_origin] = 0;
	}
}

uint8_t speed_get_constrain_wait_sta(ml_env_dev_run_data_t *cur_dev_run )
{
	if( cur_dev_run->sw_limit_data.dev_wait_sec_cnt ){
		if( KEEP_STA_ON == get_limit_sw_sta(&(cur_dev_run->sw_limit_data)) ){		//	强制关闭阶段
			return KEEP_STA_ON;
		}else if( KEEP_STA_OFF == get_limit_sw_sta(&(cur_dev_run->sw_limit_data)) ){	//	强制开启阶段
			return KEEP_STA_OFF;
		}
	}
	return KEEP_STA_NONE;
}

//	获取设备是否在强制打开的状态
bool get_dev_type_is_constrain_on(ml_dev_run_data_t *ml_dev_run_data_list,uint8_t dev_type)
{
	if( env_dev_type_none == env_dev_type_check(dev_type) ){
		return false;
	}
	
	for( uint8_t num =0; num < MAX_DEV_PARAM_CNT; num++ ){
		ml_dev_run_data_t* cur_dev_data = ml_dev_run_data_list+num;
		if( dev_type != cur_dev_data->type ){
			continue;
		}
		if( KEEP_STA_ON == speed_get_constrain_wait_sta( &(cur_dev_data->dev.env_dev) ) ){
			return true;
		}
	}
	return false;
}

//	判断设备是否是 被选中运行的
bool get_dev_is_env_select(ai_device_t* ai_dev,ml_running_data_t *running_data)
{
	for(uint8_t env=0; env<ml_env_base_cnt; env++){
		if( ai_dev->env_factor[env] * running_data->env_data.env_run_data[env].env_need_add > 0 ){
			return 1;
		}
	}
	return 0;
}

/// @brief 获取对应设备的运行数据
/// @param dev_run_list 设备的列表
/// @param cur_type 	当前作为运行的设备类型
/// @param origin_type 	原始设备类型
/// @return 
ml_dev_run_data_t* get_ml_dev_type_run_data(ml_dev_run_data_t* dev_run_list,uint8_t cur_type,uint8_t origin_type)
{
	if( cur_type >= ml_dev_type_max ){
		ESP_LOGE(TAG,"%s type[%d] err", __func__, cur_type );
		return NULL;
	}
	for(uint8_t i =0; i<MAX_DEV_PARAM_CNT ; i++ ){
		if( dev_run_list[i].type == cur_type && \
			(dev_run_list[i].origin == origin_type )) {
			return dev_run_list+i;
		}
	}
	return NULL;
}

void ml_dev_type_run_data_sync(ml_dev_run_data_t* dev_run_list,uint8_t cur_type,uint8_t origin_type)
{
	ml_dev_run_data_t* cur_run_data = NULL;

	if( cur_type >= ml_dev_type_max || origin_type >= ml_dev_type_max ){
		ESP_LOGE(TAG,"%s type[%d] origin[%d] err", __func__, cur_type, origin_type );
		return ;
	}
	for( uint8_t i =0; i<MAX_DEV_PARAM_CNT ; i++ ){
		if( dev_run_list[i].type == cur_type && \
			( dev_run_list[i].origin==origin_type ) ) {
			if( cur_run_data == NULL ){
				cur_run_data = dev_run_list+i;
			}else{
				memcpy( dev_run_list+i, cur_run_data, sizeof(ml_dev_run_data_t) );
			}
		}
	}
}

ml_dev_run_setting_t* get_ml_dev_type_setting_data( ml_dev_run_setting_t *setting_list, uint8_t dev_type, uint8_t origin )
{
	for(uint8_t num=0; num<MAX_DEV_PARAM_CNT; num++){
		ml_dev_run_setting_t* p_sev_setting = setting_list+num;
		if(p_sev_setting->type == dev_type && p_sev_setting->origin == origin && p_sev_setting->init == true ){
			return p_sev_setting;
		}
	}
	return NULL;
}

bool ml_inline_fan_is_running(ml_running_data_t* running_data)
{
	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++){
		ml_dev_run_data_t* cur_param = running_data->dev_run_data+i;
		if( cur_param->type != env_dev_type_inlinefan ){
			continue;
		}
		if( cur_param->dev.env_dev.out_speed ){
			return true;
		}
	}
	return false;
}

bool ml_list_have_type_dev(ml_running_data_t* running_data,uint8_t dev_type)
{
	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++){
		ml_dev_run_data_t* cur_param = running_data->dev_run_data+i;
		if( cur_param->type != dev_type ){
			continue;
		}
		return true;
	}
	return false;
}

bool ml_get_dev_is_running(ml_running_data_t* running_data,uint8_t dev_type)
{
	extern uint8_t get_port_speed( uint8_t port, ml_running_data_t *running_data );
	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++){
		ml_dev_run_data_t* cur_param = running_data->dev_run_data+i;
		if( cur_param->type != dev_type ){
			continue;
		}
		if( 0 != get_port_speed( i, running_data ) ){
			return true;
		}
	}
	return false;
}

////////////////////////////////////////////////////////////////////////////////////
//	运行规则
//	重置中心挡位值逻辑 - 判断运行设备 判断 增加/减小
//	说明是工作在设备调节流程 --- 更新中心挡位逻辑 + 更新开启挡位逻辑
void ai_middle_rule_run(ml_env_dev_run_data_t *running_data,uint8_t env_target_step,
						uint8_t dev_origin,uint8_t dev_lev_min,
						uint8_t dev_lev_max,int8_t trend_action )
{
	//	中心挡位 偏离Target处理逻辑 恢复之前挡位
	if( ( env_target_step == ml_dev_step_away || env_target_step == ml_dev_step_adjust )
		 && get_origin_dev_have_center_lev(running_data) ){
		if( running_data->config.is_center_lev == 1 ){
			running_data->config.is_center_lev = 0;

			if( trend_action == DEV_ACTION_CLOSE || trend_action == DEV_ACTION_DM_LEV ){			//中心挡位要减小
				if(running_data->center_speed_of_dev){
					running_data->center_speed_of_dev--;
				}
			}else if( trend_action == DEV_ACTION_UP_LEV || trend_action == DEV_ACTION_KEEP ){	//中心挡位要增加
				if(running_data->center_speed_of_dev<10){
					running_data->center_speed_of_dev++;
				}
				running_data->speed_of_dev = running_data->last_speed_of_dev;
				running_data->speed_dev_sec_cnt = 0;
			}
			if( running_data->center_speed_of_dev < dev_lev_min ){
				running_data->center_speed_of_dev = dev_lev_min;
			}
			if( running_data->center_speed_of_dev > dev_lev_max ){
				running_data->center_speed_of_dev = dev_lev_max;
			}
			log_ml_debug_i(0,"dev center away set:%d",running_data->center_speed_of_dev);
		}
	}

	//	中心挡位 到达中心点处理逻辑
	if( env_target_step == ml_dev_step_reach || env_target_step == ml_dev_step_keep ){
		if( running_data->config.is_center_lev == 0 ){
			running_data->config.is_center_lev = 1;

			running_data->last_speed_of_dev = running_data->speed_of_dev;		//记录target挡位
			if( 0 == get_origin_dev_have_center_lev(running_data) ){
				log_ml_debug_i(0,"new center set");
				running_data->center_speed_of_dev = running_data->speed_of_dev>>1;	//更新中心值挡位
				set_origin_dev_have_center_lev(running_data,1);
			}
			if( running_data->center_speed_of_dev < dev_lev_min ){
				running_data->center_speed_of_dev = dev_lev_min;
			}
			if( running_data->center_speed_of_dev > dev_lev_max ){
				running_data->center_speed_of_dev = dev_lev_max;
			}
			
			running_data->speed_of_dev = running_data->center_speed_of_dev;		//更新当前挡位
			running_data->speed_dev_sec_cnt = 0;
			log_ml_debug_i( 0, "dev center in set:%d", running_data->speed_of_dev );
		}
	}
	log_ml_debug_i( 1, "dev is center lev? %d ", running_data->config.is_center_lev );
}

void ai_dev_wait_rule_run( ml_env_dev_run_data_t *dev_run_data, bool sec_flag)
{
	if(0 ==  dev_run_data->sw_limit_data.dev_wait_sec_cnt ){
		return;
	}
	if( sec_flag ){
		dev_run_data->sw_limit_data.dev_wait_sec_cnt--;
		if( dev_run_data->sw_limit_data.dev_wait_sec_cnt == 0 ){
			//	退出等待的操作
			if( get_limit_sw_sta(&dev_run_data->sw_limit_data) ){

			}
		}
	#if 1
		log_ml_debug_i(1,"wait cnt:%d,%d",dev_run_data->sw_limit_data.dev_keep_sta,dev_run_data->sw_limit_data.dev_wait_sec_cnt);
	#endif
	}
}

void updata_dev_keep_onOff_sta( uint8_t env_dev_type, uint8_t dev_origin, uint8_t cur_speed, uint8_t action, \
								uint16_t min_on, uint16_t min_off, ml_env_dev_run_data_t *cur_dev_data )
{
	//	强制关闭
	if( cur_speed == 0 /*|| action == DEV_ACTION_CLOSE*/ ){
		set_dev_onoff_keep_delay(cur_dev_data,env_dev_type,min_on,min_off,KEEP_STA_OFF);
	}else{
	//	强制打开
		set_dev_onoff_keep_delay(cur_dev_data,env_dev_type,min_on,min_off,KEEP_STA_ON);
	}
}

bool cur_dev_is_running( uint8_t env_dev_type, uint8_t cur_speed, ml_env_dev_run_data_t *cur_dev_data )
{
	bool ret = false;
	if( env_dev_type != env_dev_type_inlinefan ){
		if( KEEP_STA_ON ==speed_get_constrain_wait_sta(cur_dev_data) \
			|| (cur_speed && KEEP_STA_OFF != speed_get_constrain_wait_sta(cur_dev_data) ) ){
				ret = 1;
		}
	}
	return ret;
}

// 
void update_dev_min_max_lev(uint8_t env_dev_type, uint8_t origin_type,ml_port_setting_st* port_setting, bool env_dev_running,uint8_t* min_lev,uint8_t* max_lev)
{
	uint8_t dev_lev_max = 0, dev_lev_min = 0;
	dev_lev_max = port_setting->device_else.lev.max_lev;
	dev_lev_min = port_setting->device_else.lev.min_lev;
	// if( origin_type == env_dev_type_heater && env_dev_type == env_dev_type_humid){
	// 	dev_lev_max = port_setting->humid.humid_lev.max_lev;
	// 	dev_lev_min = port_setting->humid.humid_lev.min_lev;
	// }
	
	if( env_dev_running && env_dev_type == env_dev_type_inlinefan && origin_type != ml_dev_type_switch ){
		if( dev_lev_max != 1 ){
			dev_lev_max = dev_lev_max/2;
		}
		if( dev_lev_max < dev_lev_min )
			dev_lev_max = dev_lev_min;
	}

	if( origin_type == ml_dev_type_switch ){
		dev_lev_max = 1;
		dev_lev_min = 0;
	}
	
	*min_lev = dev_lev_min;
	*max_lev = dev_lev_max;
	log_ml_debug_i(1,"lev:%d %d",dev_lev_min,dev_lev_max);
}

void update_dev_action(uint8_t env_dev_type, bool env_dev_running, bool co2_dev_running, uint8_t opposite_running, uint8_t* p_action)
{
	uint8_t action = *p_action;
	//	在对立设备 强制打开情况下 先关闭
	if( opposite_running )
	{ 
		action = DEV_ACTION_CLOSE;
		log_ml_debug_i(1,"%s wait constrain",env_dev_name_list[env_dev_type]);
	}
	if( env_dev_type == env_dev_type_inlinefan ){
		if( env_dev_running && action == DEV_ACTION_DM_LEV ){
			action = DEV_ACTION_CLOSE;
			updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_IMPROVE_ENV_DEV);
		}
		if( co2_dev_running ){
			action = DEV_ACTION_CLOSE;
			updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_MAINTAIN_CO2);
		}
	}
	*p_action = action;
}

void updata_dev_lev_direction(uint8_t env_loadtype,uint8_t dev_action,ml_env_dev_run_data_t *running_data)
{
	if( dev_action == DEV_ACTION_CLOSE )	//强制关闭 - 超过范围/不合适运行
	{
		
	}else if( dev_action == DEV_ACTION_DM_LEV)	//挡位减小
	{
		if( 0 == get_origin_dev_timer_cnt_direction(running_data) ){
			set_origin_dev_timer_cnt_direction(running_data,1);
			running_data->speed_dev_sec_cnt = 0;
		}
	}else if( dev_action == DEV_ACTION_UP_LEV ){	//挡位增加
		if( 1 == get_origin_dev_timer_cnt_direction(running_data) ){
			set_origin_dev_timer_cnt_direction(running_data,0);
			running_data->speed_dev_sec_cnt = 0;
		}
	}
}

void updata_dev_lev_gap_min_max(uint8_t env_step,uint8_t env_dev_type, uint8_t dev_origin,bool lev_down,uint16_t* gap_min,uint16_t* gap_max)
{
	uint16_t temp_min,temp_max = 0;
	if( lev_down == false )
	{
		if( env_step == ml_dev_step_away || env_step == ml_dev_step_adjust ){
			temp_max = get_dev_general_rise_tim( env_dev_type, dev_origin );
			temp_min = get_dev_general_rise_tim( env_dev_type, dev_origin );
		}else{
			temp_max = get_dev_rise_max_tim( env_dev_type, dev_origin );
			temp_min = get_dev_rise_min_tim( env_dev_type, dev_origin );
		}
	}else{
		temp_max = get_dev_down_max_tim( env_dev_type, dev_origin );
		temp_min = get_dev_down_min_tim( env_dev_type, dev_origin );
	}
	*gap_min = temp_min;
	*gap_max = temp_max;
}

void updata_dev_speed_cnt(uint8_t env_dev_type, uint8_t dev_origin, uint8_t env_step, 
                        uint8_t dev_action, uint8_t min_lev, uint8_t max_lev, uint8_t cur_keep_sta, 
                        uint8_t sec_flag, float add_sec, uint16_t gap_max, float* p_speed_cnt,uint8_t* p_speed)
{
	//	在强制开启/关闭阶段
	float cur_cnt = 0;
	uint8_t cur_speed = *p_speed;
	cur_cnt = *p_speed_cnt;
	//	---- 强制开关 只影响生效挡位 不影响设备当前单位
	if( ml_dev_step_keep != env_step || env_dev_type == env_dev_type_inlinefan ){
		add_sec = 1;
	}
	if( KEEP_STA_OFF == cur_keep_sta ){
		if( cur_speed != 0 ){
			add_sec = 0;
			cur_cnt = 0;			//	为0档可以变为1档 不为0档 不允许升档和降档（维持原挡位）
		}
	}
	
	log_ml_debug_i(1,"keep_sta:%d - add sec=%f",cur_keep_sta,add_sec); 
	if( cur_cnt < gap_max ){
		if( sec_flag ){
			cur_cnt += add_sec;
		}
	}else{
		cur_cnt = add_sec;
		if( dev_action == DEV_ACTION_DM_LEV ){
			if( cur_speed ){
				if( KEEP_STA_ON == cur_keep_sta && cur_speed==1){
					//	在强制开启阶段 最小挡位为 1 
				}else{
					cur_speed--;
				}
			}
		}else{
			cur_speed+=	1;
		}
	}


	//	强行改变该运行的环境设备初始挡位
	if( env_dev_type != env_dev_type_inlinefan && cur_speed == 0 && dev_action == DEV_ACTION_UP_LEV ){
		if( env_step == ml_dev_step_adjust || env_step == ml_dev_step_away ){
			cur_speed = 1;
		}
	}

	if( min_lev > cur_speed )
		cur_speed = min_lev;
	if( max_lev < cur_speed )
		cur_speed = max_lev;

	*p_speed = cur_speed;
	*p_speed_cnt = cur_cnt;
#if 1
	log_ml_debug_i(1,"on cnt:%d,%f,%d",cur_speed,cur_cnt,gap_max);
#endif
}

void inlinfan_deal_env_dev_sta(uint8_t reset, uint8_t last_sta, uint8_t run_sta, uint8_t min,ml_env_dev_run_data_t* run_data)
{
	if( run_sta != last_sta ){
		if(run_sta){ log_ml_debug_i(0,"env on");} else { log_ml_debug_i(0,"env off"); }
		if( reset == true ){
			return;
		}
		if( run_sta ){
			run_data->speed_of_dev = run_data->speed_of_dev/2;	//挡位一半
			run_data->speed_dev_sec_cnt = 0;
			if( run_data->speed_of_dev < min ){
				run_data->speed_of_dev = min;
			}
			updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_IMPROVE_ENV_DEV);
			ESP_LOGW(TAG,"env dev run inlinfan lev 1/2 ");
		}
	}
}

//
void env_run_sta_change_rule(ai_device_t* ai_dev_list,ml_running_data_t *running_data, uint8_t env_dev_running)
{
	//	环境类设备运行时 管道风机 处理逻辑
	if( env_dev_running != running_data->env_dev_have_running ){
		if(env_dev_running){ log_ml_debug_i(0,"env on");} else { log_ml_debug_i(0,"env off"); }
		running_data->env_dev_have_running = env_dev_running;
		if( running_data->config.is_new == true ){
			return;
		}
		if( env_dev_running && ai_dev_list[env_dev_type_inlinefan].can_action ){
			for( uint8_t dev_origin = 0; dev_origin<ml_dev_type_max; dev_origin++ )
			{
				ml_dev_run_data_t* run_data = get_ml_dev_type_run_data( running_data->dev_run_data, env_dev_type_inlinefan, dev_origin );
				if( run_data == NULL ){
					continue;
				}
				ml_dev_run_setting_t* setting_data = get_ml_dev_type_setting_data( running_data->dev_setting_data, env_dev_type_inlinefan, dev_origin );
				if( setting_data == NULL ){
					continue;
				}

				ml_env_dev_run_data_t* p_inlinfn_run_data = &(run_data->dev.env_dev);

				uint8_t setting_min = setting_data->setting.device_else.lev.min_lev;
				uint8_t setting_max = setting_data->setting.device_else.lev.max_lev;

				if( dev_origin == ml_dev_type_switch ){
					setting_min = 0;
					setting_max = 1;
				}

				p_inlinfn_run_data->speed_of_dev = p_inlinfn_run_data->speed_of_dev/2;	//挡位一半
				p_inlinfn_run_data->speed_dev_sec_cnt = 0;
				if( p_inlinfn_run_data->speed_of_dev < setting_min ){
					p_inlinfn_run_data->speed_of_dev = setting_min;
				}
				p_inlinfn_run_data->out_speed = p_inlinfn_run_data->speed_of_dev;
				updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_IMPROVE_ENV_DEV);
				ESP_LOGW(TAG,"env dev run inlinfan lev 1/2 ");
				//	不利条件
				if( ai_dev_list[env_dev_type_inlinefan].action != DEV_ACTION_UP_LEV ){
					ai_dev_list[env_dev_type_inlinefan].action = DEV_ACTION_CLOSE;	//强制关闭 - 超过范围/不合适运行
					p_inlinfn_run_data->out_speed = setting_min;
					uint16_t min_on 	= get_dev_limit_on_tim(env_dev_type_inlinefan,dev_origin);
					uint16_t min_off 	= get_dev_limit_off_tim(env_dev_type_inlinefan,dev_origin);
					set_dev_onoff_keep_delay(p_inlinfn_run_data,env_dev_type_inlinefan,min_on,min_off,KEEP_STA_OFF);
				}
				ml_dev_type_run_data_sync( running_data->dev_run_data, env_dev_type_inlinefan, dev_origin );	//运行完 同步其它设备
			}
		}
	}
}

uint8_t updata_env_run_speed_rule(uint8_t env_dev_type, uint8_t dev_action, uint8_t dev_keep_sta, bool is_switch, uint8_t cur_speed, uint8_t min_lev, uint8_t max_lev )
{
	uint8_t obj_speed_of_port;
	obj_speed_of_port = cur_speed;

	//	close action
	if( ( DEV_ACTION_CLOSE == dev_action && KEEP_STA_ON != dev_keep_sta )
		|| ( KEEP_STA_OFF == dev_keep_sta ) )
	{
		// inlinefan should run at least at min level set from App
		if(env_dev_type == env_dev_type_inlinefan && !is_switch ) 
			obj_speed_of_port = min_lev;
		else
			obj_speed_of_port = 0;

		return obj_speed_of_port;
	}

	// speed rises every interval, and it will restore from the last obj-speed once deviating from target again
	if( DEV_ACTION_CLOSE == dev_action || obj_speed_of_port == 0 )
	{
		if( KEEP_STA_ON == dev_keep_sta ){
			obj_speed_of_port = 1;	//强制开操作
		}
	}
	//the following will be update per second if ai setting is modified from APP
	if(obj_speed_of_port > max_lev)
		obj_speed_of_port = max_lev;
	// inline fan has min level
	if(obj_speed_of_port < min_lev && env_dev_type == env_dev_type_inlinefan)
		obj_speed_of_port = min_lev; 

	// actual loadtype is switch
	if( is_switch && obj_speed_of_port )
		obj_speed_of_port = 1;

	return obj_speed_of_port;
}

// 1、设置同步 --- 初始化处理
// 2、运行逻辑
// 3、运行完同步
void all_dev_general_rules_run( uint8_t cur_type,ai_device_t* ai_dev_cur, ml_sensor_config_data_t* input_sensor_list,\
							ml_running_data_t *running_data,uint8_t sec_flag, uint8_t* p_env_dev_running_flag)
{
	int16_t sensor_cur=0,target_min=0,target_max=0;

	// scan by cur type
	if( ai_dev_cur->exist == false ){
		return;
	}

	uint8_t env_target_step = ai_dev_cur->env_action_step;

	for( uint8_t i=0; i<ml_env_base_cnt; i++ ){
		if( 0 == (ai_dev_cur->adj_env_bit_type & (1<<i)) ){
			continue;
		}
		get_env_target_data_param( i, input_sensor_list, &sensor_cur, &target_min, &target_max );
		// log_ml_debug_i(1,"cur env %s", env_name_list[i] );
		break;
	}

	uint8_t dev_run_target = get_dev_is_env_select(ai_dev_cur, running_data );
	// scan by origin type
	for( uint8_t origin_type=0; origin_type<ml_dev_type_max; origin_type++ ){

		ml_dev_run_data_t* cur_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, cur_type, origin_type );
		ml_dev_run_setting_t* cur_setting = get_ml_dev_type_setting_data( running_data->dev_setting_data, cur_type, origin_type );
		if( cur_run_data == NULL || cur_setting == NULL ){
			continue;
		}
		// log_ml_debug_i(1,"origin type %s", env_name_list[origin_type] );
		ml_env_dev_run_data_t* env_run_data = &(cur_run_data->dev.env_dev);

		// 更新运行设置
		uint8_t* p_dev_speed  = &(env_run_data->speed_of_dev);
		float* p_dev_cnt = &(env_run_data->speed_dev_sec_cnt);

		uint8_t dev_lev_min=0,dev_lev_max=0;

		update_dev_min_max_lev( cur_type, origin_type, &(cur_setting->setting), 
				running_data->env_dev_have_running, &dev_lev_min, &dev_lev_max );
		
		// Forced shutdown rule

		// rule run 
		ai_dev_wait_rule_run( env_run_data, sec_flag );		//	最小开关时间计时
		uint8_t dev_forced_sta = speed_get_constrain_wait_sta( env_run_data );	//强制设备状态: KEEP_STA_NONE KEEP_STA_ON KEEP_STA_OFF
		bool opposition_dev_limit = opposition_dev_is_running( cur_type, running_data->dev_run_data );	//环境设备互斥处理
		bool co2_device_running = ml_get_dev_is_running( running_data, ml_dev_type_co2_generator );

		if( !ai_dev_cur->is_speed_updated ){	//与环境无关设备直接退出循环
			// log_ml_debug_i(1,"%s have no action",dev_name_list[cur_type]);
			ai_dev_cur->action = DEV_ACTION_CLOSE;
			goto _dev_on_off_rule;
		}

		if( cur_type == env_dev_type_inlinefan ){
			//	1/2 挡位逻辑
			inlinfan_deal_env_dev_sta(running_data->config.is_new, running_data->env_dev_have_running, 
										*p_env_dev_running_flag, dev_lev_min, env_run_data);
		}

		// ------------- opposition dev rule -------------
		update_dev_action( cur_type, *p_env_dev_running_flag, \
			co2_device_running, opposition_dev_limit, &(ai_dev_cur->action) );	//互斥设备强行关闭关闭逻辑

		updata_dev_lev_direction(cur_type, ai_dev_cur->action, env_run_data);

		/// prf 切换后 挡位切换的逻辑
		if( ai_dev_cur->pref_en != env_run_data->config.is_prf_run ){
			env_run_data->config.is_prf_run  = ai_dev_cur->pref_en;
			*p_dev_cnt = 0;
			if( env_run_data->config.is_prf_run == true ){
				env_run_data->speed_pref = env_run_data->speed_of_dev;
			}
			ESP_LOGI(TAG,"dev[%s] prf change->%d", dev_name_list[cur_type],env_run_data->config.is_prf_run );
		}
		if( env_run_data->config.is_prf_run ){
			p_dev_speed = &( env_run_data->speed_pref );
		}

		if( cur_type != env_dev_type_inlinefan 
			&& dev_run_target 
			&& ai_dev_cur->pref_en == false )
		{
			ai_middle_rule_run(env_run_data,env_target_step,origin_type,dev_lev_min,dev_lev_max,ai_dev_cur->action_trend);
		}

		if( ai_dev_cur->action == DEV_ACTION_CLOSE ){
			if( cur_type == env_dev_type_inlinefan ){
				//	reset run data
				*p_dev_speed = dev_lev_min;
				*p_dev_cnt = 0;
			}
			goto _dev_on_off_rule;
		}

		// dynamic speed rule
		uint16_t cur_speed_gap_max = 0;
		uint16_t cur_speed_gap_min = 0;
		//	动态升档逻辑
		bool is_lev_dowm = get_origin_dev_timer_cnt_direction( env_run_data );
		// 
		updata_dev_lev_gap_min_max(env_target_step,cur_type,origin_type,is_lev_dowm,&cur_speed_gap_min,&cur_speed_gap_max);
	#if 1
		log_ml_debug_i(1,"sensor:%d,%d,%d,%d,%d,%d,%d,%d",cur_type,ai_dev_cur->adj_env_bit_type,
					sensor_cur,target_min,target_max,cur_speed_gap_min,cur_speed_gap_max,is_lev_dowm);
	#endif

		float add_sec = Proportional_time_calculation( sensor_cur, target_min, target_max, cur_speed_gap_min, cur_speed_gap_max );

		// uint8_t keep_sta = speed_get_constrain_wait_sta( env_run_data );

		updata_dev_speed_cnt( cur_type, origin_type, env_target_step, ai_dev_cur->action, dev_lev_min,\
							dev_lev_max, dev_forced_sta, sec_flag, add_sec, cur_speed_gap_max, p_dev_cnt, p_dev_speed );

_dev_on_off_rule:
		uint8_t out_speed = *p_dev_speed;	//只改变输出挡位 改变设备记忆挡位
		if( ai_dev_cur->action == DEV_ACTION_CLOSE ){
			out_speed = dev_lev_min;
		}

		if( cur_type == env_dev_type_inlinefan && co2_device_running ){
			out_speed = 0;
			*p_dev_cnt = 0;
		}

		if( cur_type == env_dev_type_inlinefan && input_sensor_list[ml_env_co2].config.sensor_selct ){
			bool smart_en = cur_setting->setting.inlinefan.config.smart_co2_en;
			int16_t co2_ppm = input_sensor_list[ml_env_co2].inside_value;
			uint8_t co2_sta = input_sensor_list[ml_env_co2].config.sensor_sta;
			bool is_run_min = false;
			bool smart_trigger = false;
			if( ai_dev_cur->action == DEV_ACTION_CLOSE || ai_dev_cur->action == DEV_ACTION_DM_LEV ){
				is_run_min = true;
				// ESP_LOGW(TAG,"ml smart trgger 2");
			}
			if( smart_en && co2_sta ){
				smart_trigger = smart_co2_monitor_trigger(is_run_min,out_speed,dev_lev_min,co2_ppm);
				// ESP_LOGW(TAG,"ml smart trgger 3");
			}
			// ESP_LOGW(TAG,"ml smart is_min(%d) cur_lev(%d) min_lev(%d) trigger(%d)", is_run_min, out_speed, dev_lev_min, smart_trigger);

			if( smart_trigger ){
				out_speed = 0;
				*p_dev_cnt = 0;
				dev_lev_min = 0;
				insight_set_inlinefan_off_co2_run(1);
				updata_inline_fan_run_sta(NULL,0,0,INLINE_FAN_MAINTAIN_CO2);
			}
		}

		// 强制的情况处理
		if( dev_forced_sta == KEEP_STA_ON ){	//强行变为 1 档
			if( ai_dev_cur->action == DEV_ACTION_CLOSE ){
				out_speed = 1;
			}
		}else if( dev_forced_sta == KEEP_STA_OFF ){
			out_speed = 0;
		}

		// Forced rule flush
		uint16_t limit_on = get_dev_limit_on_tim(cur_type, origin_type);
		uint16_t limit_off = get_dev_limit_off_tim(cur_type, origin_type);
		updata_dev_keep_onOff_sta( cur_type, origin_type, out_speed, ai_dev_cur->action, limit_on, limit_off, env_run_data );

		env_run_data->out_speed = out_speed;
		if( cur_type != env_dev_type_inlinefan ){
			if( env_run_data->out_speed ){
				*p_env_dev_running_flag = 1;
			}
		}

		ml_dev_type_run_data_sync( running_data->dev_run_data, cur_type, origin_type );	//运行完 同步其它设备
	}
}

void ml_sun_clear_stop_flag(sun_t * p_sun,uint8* p_speed)
{
	sun_clear_stopped(p_sun);
	*p_speed = 0;
}


static void ai_sched(Time_Typedef * sys_time, u8 start_hour, u8 start_min, u8 end_hour, u8 end_min, u8 on_speed, u8 off_speed, sun_t * p_sun, u8 * p_speed)
{
    u8 sw_sta = SW_ALL_OPEN; // sched 模式，且 start 和 end 时间值都有
    u8 to;
    u32 time_cur, time_on, time_off, time_spare;

    // sun[run_port].duration = sun_duration;
    // sun[run_port].en       = 1;

    get_sched_using_times(sys_time, start_hour, start_min, end_hour, end_min, &time_cur, &time_on, &time_off);
    get_sched_running_param(sw_sta, 0, 0, time_cur, time_on, time_off, &to, &time_spare);

    if (!get_sched_sun_param(sw_sta, 0, 0, time_cur, time_on, time_off, on_speed, off_speed, p_sun) || !sun_process(p_sun, p_speed))
    {
        ml_sun_clear_stop_flag(p_sun,p_speed);
		
        set_sched_level_without_sun(on_speed, off_speed, to, time_spare, p_speed);
    }
}

static void ai_growlight_init( growlight_rundata_t* growlight_rundata )
{
	sun_t* sun = &(growlight_rundata->sun);
	uint8_t* sun_speed = &(growlight_rundata->sun_speed);

	// if( growlight_rundata->config.init_flag == false )
	{
		growlight_rundata->config.init_flag = true;
		ml_sun_clear_stop_flag(sun,sun_speed);
	}
}

static void ai_rule_growlight( bool is_sw, Time_Typedef *sys_time, ml_sun_param_t* p_dynamic, ml_dev_run_setting_t* setting, ml_dev_run_data_t *run_data, s16 cur_temp)
{
	uint8_t speed   = 0;
	Time_Typedef cur_time = ml_sun_get_tim(sys_time);	//SECONDS_OF_TIME(*sys_time);
	// 使用参数
	growlight_set* p_set = &(setting->setting.growlight);

	sun_t* sun = &(run_data->dev.light_dev.sun);
	uint8_t* sun_speed = &(run_data->dev.light_dev.sun_speed);

	if( p_set == NULL ){
		ESP_LOGE(TAG,"%s ptr is null",__func__);
		return;
	}

    uint16_t start_time    = p_set->start_hour * 60 + p_set->start_min;
    uint16_t stop_time     = p_set->end_hour * 60 + p_set->end_min;
    uint16_t sun_duration  = p_set->duration_hour * 60 + p_set->duration_min; 

    uint8_t on_speed = p_set->lev.max_lev = p_set->lev.max_lev > 0 ? p_set->lev.max_lev : 10;

	//	使用自动调节的灯光参数
	if( p_set->config.auto_light_en ){
        start_time = p_dynamic->start_minute;
        stop_time = (start_time + p_set->period_hour*60 + p_set->period_min)% (60 * 24);
	}
	if( is_sw ){
		on_speed = 1;
		sun_duration = 0;
	}
	sun->duration = sun_duration;
	sun->en = 1;
	ai_sched( &cur_time, start_time / 60, start_time % 60, stop_time / 60, stop_time % 60, on_speed, 0, sun, sun_speed);
	speed = *sun_speed;

	bool is_high_temp;
	s16 limit_data = (is_temp_unit_f()?GROWLIGHT_HIGH_TEMP_F:GROWLIGHT_HIGH_TEMP_C);
	is_high_temp = (cur_temp >  limit_data* 10);
    // ESP_LOGI(TAG, "%s(%d):%d", __func__, run_port, speed);
    if( speed ){
        ai_insight_set_light_is_ruuning_flag();
    }
    if (speed > 0 && is_high_temp) // temp protection
    {
		speed = 1; 
		if( is_sw )
			speed = 0;
        //  log_ml_debug_i(1,TAG, "%s:temp protection!:%d", __func__,is_in_range); 
    }
	*sun_speed = speed;
}

#define SECONDS_OF_TIME(_tm_)  ((_tm_).hour * 60 + (_tm_).min)
static void ai_rule_clipfan( Time_Typedef *tm, ml_dev_run_setting_t* setting, ml_dev_run_data_t *run_data, s16 cur_humid)
{
    uint16_t duration;
	int16_t runTime;
    uint16_t start_time, stop_time;
	u8 speed;
	fan_set* p_set = &(setting->setting.fan);
	fan_run_data_t* p_run_data = &(run_data->dev.fan);

	if( p_set == NULL ){
		ESP_LOGE(TAG,"%s ptr is null",__func__);
		return;
	}

	start_time = p_set->start_hour*60 + p_set->start_min; 
	stop_time = p_set->end_hour*60 + p_set->end_min;
	
    speed = p_set->lev.max_lev;
    // new20230822 : double clip fan speed if humid > 90%
	duration = (stop_time + (60*24) - start_time) % (60*24);
	runTime = (SECONDS_OF_TIME(*tm) + (60*24) - start_time) % (60*24);
	if (runTime < duration || 0 == duration ){
	/* if((start_time <= cur_time || cur_time <= stop_time) || (start_time+stop_time==0)) */
	}else{
		speed = 0;
	}
	
	if(cur_humid > CLIPFAN_HIGH_HUMID*10)   // on high humid , set clip fan level to stronger
	{
		// temp1010
	    speed <<= 1;
	    if(speed > 10){
			speed = 10;
		}

		if(0 == speed)   // ensure device's running on high humid state
			speed = 2;
	}
	p_run_data->fan_speed = speed;
}

static void ai_water_pump_init(ml_env_water_run_data_t* water_run_data )
{
	// if( water_run_data->config.init_flag == false )
	{
		water_run_data->config.init_flag 	= true;
		water_run_data->config.soil_trig 	= false;
		water_run_data->config.water_trig	= false;

		water_run_data->user_need_on = false;

		water_run_data->water_over_lock = 0;
		water_run_data->water_running_cnt = 0;

		water_run_data->soli_running_cnt = 0;
		water_run_data->soli_lock = 0;

		insight_set_water_pump_over_time( 0 );
	}
}

bool soil_target_rule( uint16_t cur_val, uint16_t target_min, uint16_t target_max, uint8_t safe_sw, 
					uint16_t max_sec, bool sec, uint8_t* p_speed, ml_env_water_run_data_t* water_run_data )
{
	uint8_t low_trig = 0, high_trig = 0;
	uint8_t out_speed = 0;
	if( cur_val < target_min ){
		low_trig = 1;
	}
	if( cur_val > target_max ){
		high_trig = 1;
	}

	if( low_trig ){
		water_run_data->config.soil_trig = 1;
	}else if( high_trig ){
		water_run_data->config.soil_trig = 0;
	}
	if( safe_sw == 0 ){
		out_speed = water_run_data->config.soil_trig;
		*p_speed = out_speed;
		return false;
	}

	uint8_t time_trig = 0;
	if( water_run_data->soli_lock == false ){	//	设备正常
		if( water_run_data->config.soil_trig ){	//	设备正在靠近高值
			if( water_run_data->soli_running_cnt < max_sec ){
				if( sec ){
					water_run_data->soli_running_cnt++;
				}
				if( water_run_data->soli_running_cnt == max_sec ){
					if( low_trig ){
						water_run_data->soli_lock = true;
						log_ml_warn("soil dev err lock!");
					}
				}
				time_trig = 1;
			}else{
				time_trig = 0;
				if( low_trig ){
					water_run_data->soli_running_cnt = 0;
					log_ml_info("soil below lock!");
				}
			}
			out_speed = water_run_data->soli_lock?0:time_trig;
		}else{	//设备已经到达高值
			water_run_data->user_need_on = 0;
			water_run_data->soli_running_cnt = 0;
			out_speed = 0;
		}
	}else{	//设备出错
		water_run_data->soli_running_cnt = 0;
		out_speed = 0;
	#if 0
		if( water_run_data->user_need_on ){
			if( high_trig ){	//达到触发高 接触保护
				water_run_data->user_need_on = 0;
				water_run_data->soli_running_cnt = 0;
				water_run_data->soli_lock = false;
				log_ml_info("soil high exit lock!");
			}
		}
		out_speed = water_run_data->user_need_on;
	#endif
	}
	*p_speed = out_speed;
	return 0;
}

// 
static void ai_rule_water_pump( ml_dev_run_setting_t* setting, ml_dev_run_data_t *run_data,\
							int16_t water_val,uint8_t w_sta, int16_t soil_val, uint8_t s_sta,uint8_t sec)
{
	uint8_t speed = 0;
	water_pump_set* p_set = &(setting->setting.water_pump);
	ml_env_water_run_data_t* water_run_data = &(run_data->dev.water_dev);
	bool sefe_sw = p_set->safeShutOff;
	bool water_generate_insight= 0;

	if( p_set == NULL ){
		ESP_LOGE(TAG,"%s ptr is null",__func__);
		return;
	}
#if 0
	static uint8_t val = 0;
	if( val != water_val ){
		val = water_val;
		log_ml_debug_w(0,"water: sta(%d) v(%d) mode(%d) waterMax(%d) safeSet(%d) soil: sta(%d) v(%d) MaxOn(%d) tragetMin(%d)",
			w_sta, water_val, p_set->mode,p_set->max_on_time,p_set->safeShutOff, s_sta, soil_val, p_set->soil_run_max,p_set->target_min);
	}

	log_ml_debug_i(1,"water: sta(%d) v(%d) mode(%d) waterMax(%d) safeSet(%d) soil: sta(%d) v(%d) MaxOn(%d) tragetMin(%d)",
		w_sta, water_val, p_set->mode,p_set->max_on_time,p_set->safeShutOff, s_sta, soil_val, p_set->soil_run_max,p_set->target_min);
#endif

	switch(p_set->mode){
		case 0:	//water mode
			if( w_sta == 0 ){
				break;
			}
			if( water_run_data->water_over_lock && water_pupm_lock_is_cleaned() ){
				// water_run_data->user_need_on = true;
				ai_water_pump_init( water_run_data );
				log_ml_info("water be app reset!");
			}
			uint16_t max_sec_water = p_set->max_on_time;
			if( sefe_sw == false ){
				if( water_val == 0 ){
					speed = 1;
				}else{
					speed = 0;
				}
				water_run_data->water_running_cnt = 0;
				water_run_data->water_over_lock = false;
				water_run_data->user_need_on = false;
			}else{
				if( water_run_data->water_over_lock == false ){
					if( water_val == 0 ){
						if( water_run_data->water_running_cnt < max_sec_water ){
							if(sec){
								water_run_data->water_running_cnt++;
							}
						}else{
							water_run_data->water_over_lock = true;
							log_ml_warn("water be lock!");
						}
						speed=water_run_data->water_over_lock?0:1;
					}else{
						water_run_data->water_running_cnt = 0;
						speed = 0;
					}
				}else{
					water_run_data->water_running_cnt = 0;
					speed = 0;
				#if 0
					if( water_run_data->user_need_on ){
						speed = 1;
						if( water_val == 1 ){
							water_run_data->water_running_cnt = 0;
							water_run_data->water_over_lock = false;
							water_run_data->user_need_on = false;
							speed = 0;
						}
					}
				#endif
				}
			} 
			water_generate_insight = water_run_data->water_over_lock;
			if( water_val == 1 ){
				water_generate_insight = false;
			}
			break;

		case 1:	//soil mode
			if( s_sta == 0 ){
				break;
			}
			if( water_run_data->soli_lock && soil_pupm_lock_is_cleaned() ){
				// water_run_data->user_need_on = true;
				ai_water_pump_init( water_run_data );
				log_ml_info("soil be app reset!");
			}
			uint16_t max_sec = p_set->soil_run_max;
			uint16_t target_min = p_set->target_min;
			uint16_t target_max = p_set->target_max;
			target_min = target_min*100;
			target_max = target_max*100;
			// ESP_LOGW(TAG,"soil val:%d,set val:%d",soil_val,target_min);
			soil_target_rule( soil_val, target_min, target_max, sefe_sw, max_sec, sec, &speed, water_run_data );
			water_generate_insight = water_run_data->soli_lock;
			if( soil_val > target_min ){
				water_generate_insight = false;
			}
			break;
	}
	water_run_data->speed = speed;

	if( water_run_data->user_need_on ){
		water_generate_insight = false;
	}
	insight_set_water_pump_over_time( water_generate_insight );

	log_ml_debug_i(1,"water speed:%d",speed);
}

static void ai_co2_gennerator_init( ml_env_co2_run_data_t* co2_run_data )
{
	// if( co2_run_data->config.init_flag == false )
	{
		co2_run_data->dev_keep_delay = 0;
		co2_run_data->onoff_sta = 0;
		co2_run_data->config.init_flag = true;
		co2_run_data->config.co2_err = 0;
		co2_run_data->config.user_need_on = 0;
		co2_run_data->run_sec = 0;
		co2_run_data->co2_increase = 0xffff;
		co2_run_data->speed = 0;
		co2_run_data->mem_delay = co2_run_data->dev_keep_delay;
	}
}

/// 要获取 target 范围决定运行
static void ai_rule_co2_generator_rule(Time_Typedef* tm, ml_dev_run_setting_t* setting, ml_dev_run_data_t *run_data,\
										uint8_t in_range_sta,s16 co2_val,s16 light_val,u8 sec)
{
	uint8_t speed = 0,sensor_off=0;
	// uint8_t light_force = 0;	//
	co2_generator_set* p_set = &(setting->setting.co2_generator);
	uint16_t co2_target 	= p_set->target;
	// uint16_t co2_target_min = (p_set->target>p_set->accept_buff)?(p_set->target-p_set->accept_buff):0;
	uint16_t co2_run_sec = p_set->on_duration;
	uint16_t co2_off_sec = p_set->interval_duration;
	ml_env_co2_run_data_t* co2_run_data = &(run_data->dev.co2_dev);

	if( p_set == NULL ){
		ESP_LOGE(TAG,"%s ptr is null",__func__);
		return;
	}

	//  异常保护功能
	if( light_val < 10 || co2_val >= 5000 ){
		if( co2_val >= 5000 ){
			insight_set_co2_over_5000_off(1);
		}
		sensor_off = 1;
	}

	//	在达到 range 范围的处理
	uint8_t env_need_on = false;
	if( co2_val < co2_target ){
		speed = 1;
		env_need_on = 1;
	}else{
		speed = 0;
	#if 0
		if( co2_run_data->config.user_need_on ){
			co2_run_data->config.user_need_on = 0;
		}
	#endif
	}

	if( co2_run_data->config.co2_err ){
		speed = 0;
		if( co2_lock_is_cleaned() ){
			ai_co2_gennerator_init( co2_run_data );
			log_ml_info("co2 be app reset!");
		#if 0
			// co2_run_data->config.user_need_on = 1;
			co2_run_data->config.co2_err = 0;

			if( co2_run_data->onoff_sta != 1 ){
				co2_run_data->dev_keep_delay = 0;	//强制设置
				in_range_sta = 1;
			}
			speed = 1;
		#endif
		}
	}

	if( sensor_off ){
		speed = 0;
		//	强制切换
		if( co2_run_data->onoff_sta != 0 ){
			co2_run_data->dev_keep_delay = 0;	//强制设备
		}
	}
	
#if 0
	if( co2_run_data->config.user_need_on ){
		if( co2_run_data->onoff_sta != 1 ){
			co2_run_data->dev_keep_delay = 0;	//强制设置
			in_range_sta = 1;
		}
		speed = 1;
	}
#endif

	if( co2_run_data->dev_keep_delay /*&& light_force == 0*/ ){
		if(sec){
			co2_run_data->dev_keep_delay--;
		}
		// log_ml_debug_i(1,"co2 keep %d,cnt %d", co2_run_data->onoff_sta, co2_run_data->dev_keep_delay);
	}else{
		if( co2_run_data->onoff_sta != speed ){
			if( (speed == 1 && in_range_sta) || speed == 0 ){
				co2_run_data->onoff_sta = speed;
				co2_run_data->dev_keep_delay = co2_run_data->onoff_sta?co2_run_sec:co2_off_sec;
				co2_run_data->mem_delay = co2_run_data->dev_keep_delay;
				log_ml_info("co2 dev sta chage:%d cnt:%d",co2_run_data->onoff_sta, co2_run_data->dev_keep_delay);
			}
		}
		if( co2_run_data->onoff_sta == 0 && env_need_on && in_range_sta == false ){
			insight_set_co2_off_outside_range(1);
		}
		if( co2_run_data->config.co2_err ){
			insight_set_co2_off_outside_range(0);
		}
	}
	
	co2_run_data->speed = co2_run_data->onoff_sta;
#if 1
	log_ml_debug_i( 1,"co2 targ :%d val :%d err:%d speed:%d on:%d off:%d delay:%d",
			co2_target, co2_val, co2_run_data->config.co2_err, co2_run_data->speed, 
			co2_run_sec, co2_off_sec, co2_run_data->dev_keep_delay );
#endif

	//	在开启状态 判断浓度是否正常增加 - 异常判断
	#define CO2_JUDGE_GAP	(3*60)
	#define CO2_JUDGE_DLT	(20)

	if( co2_run_data->co2_increase == 0xffff ){	//初始化参数
		co2_run_data->co2_increase = co2_val;
	}

	if( co2_run_data->config.co2_err /*|| co2_run_data->config.user_need_on*/ || co2_run_data->speed == 0 || (co2_run_data->dev_keep_delay >= CO2_JUDGE_GAP) ){
		co2_run_data->co2_increase = co2_val;
		co2_run_data->run_sec = 0;
	}else{
		if( sec ){
			co2_run_data->run_sec++;
		}
		if( co2_run_data->run_sec >= CO2_JUDGE_GAP ){
			co2_run_data->run_sec = 0;
			// 数据异常
			if( co2_val < co2_run_data->co2_increase + CO2_JUDGE_DLT && env_need_on ){
				co2_run_data->config.co2_err = 1;
				log_ml_warn("co2 run err (last:%d cur:%d)",co2_run_data->co2_increase,co2_val);
			}
			co2_run_data->co2_increase = co2_val;
		}
	}
	if( co2_run_data->config.co2_err ){
		insight_set_co2_genarator_over_time(1);
	}
}

void ml_device_init( uint8_t loadtype, uint8_t orgin, ml_dev_run_data_t *run_data, ml_dev_run_setting_t* setting )
{
	switch( loadtype ){
		case env_dev_type_humid:
		case env_dev_type_dehumi:
		case env_dev_type_heater:
		case env_dev_type_ac:
		case env_dev_type_inlinefan:
			uint8_t max_lev = setting->setting.device_else.lev.max_lev;
			ml_clear_env_dev_run_data( &(run_data->dev.env_dev), max_lev/2, loadtype );
			break;

		case ml_dev_type_fan:

			break;

		case ml_dev_type_growlight:
			ai_growlight_init( &(run_data->dev.light_dev) );
			break;

		case ml_dev_type_co2_generator:
			ai_co2_gennerator_init( &(run_data->dev.co2_dev) );
			break;
			
		case ml_dev_type_water_pump:
			ai_water_pump_init( &(run_data->dev.water_dev) );
			break;
	}
	log_ml_debug_w(0, "dev[%s] init", dev_name_list[loadtype] );
}

void device_speed_rule_run(uint8_t sec_flag, Time_Typedef *sys_time,ai_device_t* ai_dev,  \
							ml_running_data_t *running_data, ml_sensor_config_data_t* input_sensor_list )
{
	uint8_t loadtype;
	uint8_t env_dev_running_flag = 0;

    // update speed per x min 
	//	按设备进行挡位规则更新
	log_ml_debug_w(1,"\r\n" );
	clean_ai_insight_bit_info();
	memset(running_data->out_speed, 0x00, sizeof(running_data->out_speed) );
	for( loadtype=0; loadtype<ml_dev_type_max; loadtype++ )	//按当前设备类型进行规则运行	-	设备规则里面去判断
	{
		if(!ai_dev[loadtype].exist){
			continue;
		}
		if(!ai_dev[loadtype].can_action){
			continue;
		}
		log_ml_debug_w(1,"dev:%s", dev_name_list[loadtype] );
		switch( loadtype ){
			case env_dev_type_humid:
			case env_dev_type_dehumi:
			case env_dev_type_heater:
			case env_dev_type_ac:
			case env_dev_type_inlinefan:
				all_dev_general_rules_run( loadtype, ai_dev+loadtype, input_sensor_list, running_data, sec_flag, &env_dev_running_flag );
				if( loadtype == env_dev_type_inlinefan ){
					running_data->env_dev_have_running = env_dev_running_flag;
					// env_run_sta_change_rule(ai_dev, running_data, env_dev_running_flag );
				}
				break;

			case ml_dev_type_fan:
				for( uint8_t origin_type=0; origin_type<ml_dev_type_max; origin_type++ ){
					ml_dev_run_data_t* cur_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, loadtype, origin_type );
					ml_dev_run_setting_t* cur_setting = get_ml_dev_type_setting_data( running_data->dev_setting_data, loadtype, origin_type );
					if( cur_run_data == NULL ){
						continue;
					}
					ai_rule_clipfan( sys_time, cur_setting, cur_run_data, input_sensor_list[ml_env_humid].inside_value );
					ml_dev_type_run_data_sync( running_data->dev_run_data, loadtype, origin_type );	//运行完 同步其它设备
				}
				break;

			case ml_dev_type_growlight:		//植物灯运行规则
				for( uint8_t origin_type=0; origin_type<ml_dev_type_max; origin_type++ ){
					ml_dev_run_data_t* cur_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, loadtype, origin_type );
					ml_dev_run_setting_t* cur_setting = get_ml_dev_type_setting_data( running_data->dev_setting_data, loadtype, origin_type );
					if( cur_run_data == NULL ){
						continue;
					}
					ai_rule_growlight( (origin_type==ml_dev_type_switch), sys_time, running_data->p_ml_sun_param, \
										cur_setting, cur_run_data, input_sensor_list[ml_env_temp].inside_value);
					ml_dev_type_run_data_sync( running_data->dev_run_data, loadtype, origin_type );	//运行完 同步其它设备
				}
				break;

			case ml_dev_type_co2_generator:	//CO2生成器运行规则
				for( uint8_t origin_type=0; origin_type<ml_dev_type_max; origin_type++ ){
					ml_dev_run_data_t* cur_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, loadtype, origin_type );
					ml_dev_run_setting_t* cur_setting = get_ml_dev_type_setting_data( running_data->dev_setting_data, loadtype, origin_type );
					if( cur_run_data == NULL ){
						continue;
					}
					int16_t grow_light_v = input_sensor_list[ml_env_light].inside_value;
					if( ml_list_have_type_dev(running_data, ml_dev_type_growlight) ){
						if( false == ml_get_dev_is_running(running_data, ml_dev_type_growlight) ){
							grow_light_v = 0;
						}else{
							grow_light_v = 1000;
						}
					}
					uint8_t is_in_range = running_data->env_in_range;
					if( false == ml_list_have_type_dev(running_data,env_dev_type_inlinefan) ){
						is_in_range = true;
					}
					if( input_sensor_list[ml_env_co2].config.sensor_sta && input_sensor_list[ml_env_light].config.sensor_sta ){
						ai_rule_co2_generator_rule(sys_time, cur_setting,cur_run_data,is_in_range,\
							input_sensor_list[ml_env_co2].inside_value, grow_light_v, sec_flag );
					}else{
						ai_co2_gennerator_init( & cur_run_data->dev.co2_dev );
					}
					
					ml_dev_type_run_data_sync( running_data->dev_run_data, loadtype, origin_type );	//运行完 同步其它设备
				}
				break;

			case ml_dev_type_water_pump:
				for( uint8_t origin_type=0; origin_type<ml_dev_type_max; origin_type++ ){
					ml_dev_run_data_t* cur_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, loadtype, origin_type );
					ml_dev_run_setting_t* cur_setting = get_ml_dev_type_setting_data( running_data->dev_setting_data, loadtype, origin_type );
					if( cur_run_data == NULL ){
						continue;
					}
					ai_rule_water_pump(cur_setting, cur_run_data,
						input_sensor_list[ml_env_water].inside_value, input_sensor_list[ml_env_water].config.sensor_sta,
						input_sensor_list[ml_env_soil].inside_value, input_sensor_list[ml_env_soil].config.sensor_sta, sec_flag );
					
					ml_dev_type_run_data_sync( running_data->dev_run_data, loadtype, origin_type );	//运行完 同步其它设备
				}
				break;
			default: break;
		}
		log_ml_debug_w(1,"\r\n" );
	}
	ai_target_step_auto(running_data);
}

uint8_t get_port_speed( uint8_t port, ml_running_data_t *running_data )
{
	ml_dev_run_data_t* cur_run_data = running_data->dev_run_data+port;
	uint8_t speed = 0;
	switch( cur_run_data->type )
	{
		case env_dev_type_humid:
		case env_dev_type_dehumi:
		case env_dev_type_heater:
		case env_dev_type_ac:
		case env_dev_type_inlinefan:
			speed = cur_run_data->dev.env_dev.out_speed;
			break;

		case ml_dev_type_growlight:
			speed = cur_run_data->dev.light_dev.sun_speed;
			break;
		case ml_dev_type_fan:
			speed = cur_run_data->dev.fan.fan_speed;
			break;
		case ml_dev_type_water_pump:
		case ml_dev_type_pump:
			speed = cur_run_data->dev.water_dev.speed;
			break;

		case ml_dev_type_co2_generator:
			speed = cur_run_data->dev.co2_dev.speed;
			break;
	}
	return speed;
}

void update_speed(ml_input_dev_info_t* input_dev_list, uint8_t port_num, ai_device_t* ai_dev, ml_running_data_t *running_data, ml_output_port_t* output_port_list )
{
    uint8_t i, loadtype;
	uint8_t dev_origin,port_sta;

	//将设备信息 实现到每个端口
	for(i=0; i<port_num; i++)
	{
	    loadtype = input_dev_list[i].ml_type_of_port;
		dev_origin = input_dev_list[i].origin_of_port;
		port_sta = input_dev_list[i].port_sta;

		bool is_switch_port = (dev_origin == ml_dev_type_switch );

		if( loadtype == env_dev_type_none || port_sta == DEV_STA_OFF_LINE )
		    continue;

		if( port_sta == DEV_STA_ERR ){
			ml_set_speed(output_port_list, i, 0);
			continue;
		}

		//	端口限制中心单位速度
		uint8_t speed = get_port_speed(i,running_data);	
		if( is_switch_port ){
			speed = (speed>0);
		}
		ml_set_speed(output_port_list, i, speed); 
	}
}


bool ml_grow_light_setting_change_need_reset(ml_port_setting_st* old_setting,ml_port_setting_st* new_setting)
{
	// if( 0 == memcmp( &old_setting->growlight, &new_setting->growlight, sizeof(new_setting->growlight) ) )
	growlight_set* old = &old_setting->growlight;
	growlight_set* new = &new_setting->growlight;
	if( old->config.auto_light_en != new->config.auto_light_en ||
		old->duration_hour != new->duration_hour ||
		old->duration_min != new->duration_min	||
		old->start_hour != new->start_hour ||
		old->start_min != new->start_min ||
		old->end_hour != new->end_hour ||
		old->end_min != new->end_min ||
		old->period_hour != new->period_hour ||
		old->period_min != new->period_min
	){
		return true;
	}
	return false;
}

bool ml_co2_setting_change_need_reset(ml_port_setting_st* old_setting,ml_port_setting_st* new_setting)
{
	// if( 0 == memcmp( &old_setting->growlight, &new_setting->growlight, sizeof(new_setting->growlight) ) )
	co2_generator_set* old = &old_setting->co2_generator;
	co2_generator_set* new = &new_setting->co2_generator;
	if( old->target != new->target || old->on_duration != new->on_duration 
		|| old->interval_duration != new->interval_duration
	){
		return true;
	}
	return false;
}

bool ml_water_setting_change_need_reset(ml_port_setting_st* old_setting,ml_port_setting_st* new_setting)
{
	// if( 0 == memcmp( &old_setting->growlight, &new_setting->growlight, sizeof(new_setting->growlight) ) )
	water_pump_set* old = &old_setting->water_pump;
	water_pump_set* new = &new_setting->water_pump;
	if( old->mode != new->mode || old->max_on_time != new->max_on_time 
		|| old->target_max != new->target_max || old->target_min != new->target_min
		|| old->soil_run_max != new->soil_run_max || old->safeShutOff != new->safeShutOff
	){
		return true;
	}
	return false;
}

// 更新 ai_device_t + ml_dev_run_setting_t
void update_devices_information(ml_input_dev_info_t* input_dev_list, uint8_t port_num,\
								ml_dev_run_setting_t *dev_setting_list, ai_device_t *ai_dev )
{
	uint8_t i, loadtype, origin, port_sta;

	for(i=0; i<port_num; i++)
	{
		ml_input_dev_info_t* cur_input_dev_info = &(input_dev_list[i]);
		ml_port_setting_st* port_dev_setting = &(cur_input_dev_info->dev_rule_setting);

		loadtype = cur_input_dev_info->ml_type_of_port;
		origin = cur_input_dev_info->origin_of_port;
		port_sta = cur_input_dev_info->port_sta;

		//	setting info updata
		// port_dev_setting->type = loadtype;
		// port_dev_setting->origin = origin;
		// memcpy( &(port_dev_setting->setting), &(cur_input_dev_info->dev_rule_setting), sizeof(ml_port_setting_st) );

		//	清除端口的运行数据
		if( loadtype == env_dev_type_none ){
			continue;
		}
		if(loadtype >= ml_dev_type_max)
			continue;
		if( port_sta == DEV_STA_OFF_LINE ){
			continue;
		}
		ai_dev[loadtype].exist = true;
		ai_dev[loadtype].type = loadtype;

		if( loadtype ==  env_dev_type_inlinefan && port_dev_setting->inlinefan.config.enPro ){
			ai_dev[loadtype].env_prior_bit = (port_dev_setting->inlinefan.config.humidPro<<ml_env_humid);
			ai_dev[loadtype].env_prior_bit |= (port_dev_setting->inlinefan.config.tempPro<<ml_env_temp);
		}

		//	提前在端口关闭 异常端口
		if( port_sta != DEV_STA_NO_PROBLEM ){
			continue;
		}
		ai_dev[loadtype].can_action = true;
	}
}

void printf_run_data_list(ml_running_data_t *running_data)
{
	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++)
	{
		ml_dev_run_data_t* dev_run_data = &(running_data->dev_run_data[i]);
		uint8_t loadtype = dev_run_data->type;
		uint8_t origin = dev_run_data->origin;
		log_ml_debug_w(1,"port%d run data type[%s] origin[%s]",i,dev_name_list[loadtype],dev_name_list[origin] );
	}

	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++)
	{
		ml_dev_run_setting_t* port_run_setting = &(running_data->dev_setting_data[i]);
		uint8_t loadtype = port_run_setting->type;
		uint8_t origin = port_run_setting->origin;
		log_ml_debug_w(1,"port%d setting type[%s] origin[%s]",i,dev_name_list[loadtype],dev_name_list[origin] );
	}
}

//	负责运行过程中的 设备重置逻辑
//	设备 拷贝 重置逻辑 
// 	查找有没类型匹配 
//		不匹配则重置
//		匹配 是新增加设备类型则拷贝
//	 		不是 去判断设置项的改变 没改变-不做处理 改变则针对初始化 
void ml_dev_run_reinit_scan(ml_input_dev_info_t* input_dev_list, ml_running_data_t *running_data )
{
	//	不存在的设备要进行初始化 - 存在的设备判断数据改变进行初始话
	for(uint8_t i=0; i<MAX_DEV_PARAM_CNT; i++)
	{
		bool port_rest = false;
		ml_input_dev_info_t* cur_input_dev_info = &(input_dev_list[i]);

		uint8_t loadtype = cur_input_dev_info->ml_type_of_port;
		uint8_t origin = cur_input_dev_info->origin_of_port;

		// //	对于出错设备处理 不运行
		// if( cur_input_dev_info->port_sta != DEV_STA_NO_PROBLEM ){
		// 	loadtype = env_dev_type_none;
		// 	origin = env_dev_type_none;
		// }
		// uint8_t port_sta = cur_input_dev_info->port_sta;
		// uint8_t env_dev_type = env_dev_type_check(loadtype);

		//	setting 检测处理逻辑 复位运行
		//  setting 代表同一设备类型的 修改都针对 setting 
		ml_dev_run_setting_t* dev_setting = get_ml_dev_type_setting_data(running_data->dev_setting_data,loadtype,origin);	//获取相关设备的运行数据 - 如何检测设置修改
		ml_dev_run_setting_t* port_run_setting = &(running_data->dev_setting_data[i]);		//当前的端口设置 - 最终修改同步的

		ml_port_setting_st* dev_port_new_setting = &(cur_input_dev_info->dev_rule_setting);	//新的设置
		
		if( port_run_setting->init == false ){
			// init
			port_run_setting->init = true;
			port_run_setting->type = loadtype;
			port_run_setting->origin = origin;
			port_run_setting->setting = *dev_port_new_setting;
			dev_setting = port_run_setting;
		}

		// 新的设备 初始化 当前端口 更新配置
		if( dev_setting == NULL ){
			dev_setting = port_run_setting;
			dev_setting->type = loadtype;
			dev_setting->origin = origin;
			port_rest = true;
			log_ml_debug_w(0, "port[%d] dev_setting is null", i );
		}

		// rst check
		if( loadtype == ml_dev_type_growlight ){
			if( true == ml_grow_light_setting_change_need_reset( &(dev_setting->setting), dev_port_new_setting ) ){
				//	执行复位处理
				port_rest = true;
				log_ml_debug_w(0, "light setting chg");
			}
		}
		if( loadtype == ml_dev_type_co2_generator ){
			if( true == ml_co2_setting_change_need_reset( &(dev_setting->setting), dev_port_new_setting ) ){
				//	执行复位处理
				port_rest = true;
				log_ml_debug_w(0, "co2 setting chg");
			}
		}
		if( loadtype == ml_dev_type_pump || loadtype == ml_dev_type_water_pump ){
			if( true == ml_water_setting_change_need_reset( &(dev_setting->setting), dev_port_new_setting ) ){
				//	执行复位处理
				port_rest = true;
				log_ml_debug_w(0, "water setting chg");
			}
		}

		//	更新配置
		dev_setting->setting = *dev_port_new_setting;

		// run data check
		ml_dev_run_data_t* dev_run_data = get_ml_dev_type_run_data( running_data->dev_run_data, loadtype, origin );
		ml_dev_run_data_t* port_run_data = running_data->dev_run_data+i;
		// 无相关运行数据
		if( dev_run_data == NULL ){
			// 执行设备初始化处理
			port_rest = true;
			dev_run_data = port_run_data;
			dev_run_data->type = loadtype;
			dev_run_data->origin = origin;
			log_ml_debug_w(0, "port[%d] dev_run_data is null", i);
		}
		
		if( port_rest ){
			ml_device_init( loadtype, origin, dev_run_data, dev_setting );
		}
		//	sync
		*port_run_setting = *dev_setting;

		*port_run_data = *dev_run_data;

		// //	设置数据同步
		// memcpy( port_run_setting, dev_setting, sizeof(ml_port_setting_st) );	
		// // 	运行数据拷贝
		// memcpy( port_run_data, dev_run_data, sizeof(ml_dev_run_data_t) );
	}
	// printf_run_data_list(running_data);
}

// switch type or not,  inlinefan or not, clip fan or not, environment dev or not
void ml_rule(ml_input_dev_info_t* input_dev_list, ml_sensor_config_data_t* input_sensor_list, \
			uint8_t sec_flag, Time_Typedef *sys_time, \
			ml_running_data_t *running_data, ml_output_port_t* output_port_list)
{
	ai_device_t ai_dev[ml_dev_type_max];

	ml_sec_flag = sec_flag;	// debug 节拍

    // 2. control temp/humid/vpd to target, and generate ret speeds of used devices - 参数转换 不涉及规则运行与速度改变

	// generator run data 
	ml_run_clean_flag();

	memset(ai_dev, 0, ml_dev_type_max*sizeof(ai_device_t));
	update_devices_information( input_dev_list, MAX_DEV_PARAM_CNT, running_data->dev_setting_data, ai_dev );   // ports classified as devices types, managed with devices conception
	update_devices_env_factor( ai_dev, input_sensor_list, NULL );
	ml_dev_run_reinit_scan( input_dev_list, running_data );
	if( running_data->config.delt == 1  ){
		clean_ai_insight_bit_info();
	}
	if( running_data->config.sw == 0 ){
		updata_inline_fan_action_bit(0,DEV_ACTION_CLOSE);
		return;
	}
	
	ai_sensor_adjust_rule( input_sensor_list, ai_dev, running_data, sec_flag, false);
	
	//	输出运行逻辑
	//	根据运行逻辑 结合 速度规则 得出运行挡位
	device_speed_rule_run(sec_flag, sys_time, ai_dev, running_data, input_sensor_list);

	// 3. update devices output speed to port speeds, with time delay, within setting range, and under special device limitation rule
	update_speed(input_dev_list, MAX_DEV_PARAM_CNT, ai_dev, running_data, output_port_list);
	running_data->config.is_new = 0;
}

// ------------------------------------------- 对外接口 -------------------------------------------
void ml_rest_run_sta( ml_running_data_t *running_data )
{
	memset( running_data,0x00,sizeof(ml_running_data_t) );
	running_data->config.is_new = 1;
	// for( uint8_t env_type=0; env_type < ml_env_base_cnt; env_type++ )
	// {
	// 	ml_env_run_data_t* cur_run_data = &( running_data->env_data.env_run_data[env_type] );
	// 	cur_run_data->is_reach_target = 0;
	// 	cur_run_data->env_diff = 0;
	// 	cur_run_data->target_adjust_step = 0;
	// 	cur_run_data->env_need_add = 0;
	// }
}

uint16_t ml_sava_running_data(uint8_t* p_buf)
{
	uint16_t len = 0;
	//	dev g_ml_running_data
	write_atom_unit( &(g_ml_running_data.env_dev_have_running), 		p_buf,  &len);
	for(uint8_t num=0; num<MAX_DEV_PARAM_CNT; num++){
		ml_dev_run_data_t* cur_dev_run = ( g_ml_running_data.dev_run_data + num );
		
		ml_dev_run_data_t save_dev_run_data;
		memset(&save_dev_run_data,0x00,sizeof(ml_dev_run_data_t));
		ml_dev_run_data_t* save_run_data = &save_dev_run_data;
		
		save_run_data->type = cur_dev_run->type;
		save_run_data->origin = cur_dev_run->origin;

		switch(cur_dev_run->type){
			case env_dev_type_humid:
			case env_dev_type_dehumi:
			case env_dev_type_heater:
			case env_dev_type_ac:
			case env_dev_type_inlinefan:
				save_run_data->dev.env_dev.center_speed_of_dev 	= cur_dev_run->dev.env_dev.center_speed_of_dev;
				save_run_data->dev.env_dev.last_speed_of_dev 	= cur_dev_run->dev.env_dev.last_speed_of_dev;
				save_run_data->dev.env_dev.speed_of_dev 		= cur_dev_run->dev.env_dev.speed_of_dev;
				save_run_data->dev.env_dev.config.center_rule_run= cur_dev_run->dev.env_dev.config.center_rule_run;
				save_run_data->dev.env_dev.config.is_center_lev	= cur_dev_run->dev.env_dev.config.is_center_lev;
				break;
			case ml_dev_type_growlight:

				break;
			case ml_dev_type_water_pump:
			case ml_dev_type_pump:
				save_run_data->dev.water_dev.soli_lock 		= cur_dev_run->dev.water_dev.soli_lock;
				save_run_data->dev.water_dev.water_over_lock= cur_dev_run->dev.water_dev.water_over_lock;
				save_run_data->dev.water_dev.user_need_on 	= cur_dev_run->dev.water_dev.user_need_on;
				save_run_data->dev.water_dev.speed			= cur_dev_run->dev.water_dev.speed;
				save_run_data->dev.water_dev.config.soil_trig = cur_dev_run->dev.water_dev.config.soil_trig;
				save_run_data->dev.water_dev.config.water_trig = cur_dev_run->dev.water_dev.config.water_trig;
				break;
			case ml_dev_type_co2_generator:
				save_run_data->dev.co2_dev.config.co2_err 	= cur_dev_run->dev.co2_dev.config.co2_err;
				save_run_data->dev.co2_dev.config.user_need_on 	= cur_dev_run->dev.co2_dev.config.user_need_on;
				save_run_data->dev.co2_dev.onoff_sta 		= cur_dev_run->dev.co2_dev.onoff_sta;
				save_run_data->dev.co2_dev.speed			= cur_dev_run->dev.co2_dev.speed;
				save_run_data->dev.co2_dev.mem_delay		= cur_dev_run->dev.co2_dev.mem_delay;
				break;
		}
		write_atom_unit( &(save_dev_run_data), p_buf,  &len);
	}
	return len;
}

uint16_t ml_read_running_data(uint8_t* p_buf)
{
	uint16_t len = 0;
	read_atom_unit( &(g_ml_running_data.env_dev_have_running), 		p_buf,  &len);
	for(uint8_t num=0; num<MAX_DEV_PARAM_CNT; num++){
		ml_dev_run_data_t* save_run_data = ( g_ml_running_data.dev_run_data + num );
		memset(save_run_data,0x00,sizeof(ml_dev_run_data_t));

		ml_dev_run_data_t read_run_data;
		read_atom_unit( &(read_run_data), p_buf,  &len);
		ml_dev_run_data_t* cur_dev_run = &read_run_data;
		
		save_run_data->type = cur_dev_run->type;
		save_run_data->origin = cur_dev_run->origin;

		switch(cur_dev_run->type){
			case env_dev_type_humid:
			case env_dev_type_dehumi:
			case env_dev_type_heater:
			case env_dev_type_ac:
			case env_dev_type_inlinefan:
				save_run_data->dev.env_dev.center_speed_of_dev 	= cur_dev_run->dev.env_dev.center_speed_of_dev;
				save_run_data->dev.env_dev.last_speed_of_dev 	= cur_dev_run->dev.env_dev.last_speed_of_dev;
				save_run_data->dev.env_dev.speed_of_dev 		= cur_dev_run->dev.env_dev.speed_of_dev;
				save_run_data->dev.env_dev.config.center_rule_run= cur_dev_run->dev.env_dev.config.center_rule_run;
				break;
			case ml_dev_type_growlight:

				break;
			case ml_dev_type_water_pump:
			case ml_dev_type_pump:
				save_run_data->dev.water_dev.soli_lock 		= cur_dev_run->dev.water_dev.soli_lock;
				save_run_data->dev.water_dev.water_over_lock= cur_dev_run->dev.water_dev.water_over_lock;
				save_run_data->dev.water_dev.user_need_on 	= cur_dev_run->dev.water_dev.user_need_on;
				save_run_data->dev.water_dev.speed			= cur_dev_run->dev.water_dev.speed;
				save_run_data->dev.water_dev.config.soil_trig = cur_dev_run->dev.water_dev.config.soil_trig;
				save_run_data->dev.water_dev.config.water_trig = cur_dev_run->dev.water_dev.config.water_trig;
				break;
			case ml_dev_type_co2_generator:
				save_run_data->dev.co2_dev.config.co2_err 	= cur_dev_run->dev.co2_dev.config.co2_err;
				save_run_data->dev.co2_dev.config.user_need_on 	= cur_dev_run->dev.co2_dev.config.user_need_on;
				save_run_data->dev.co2_dev.onoff_sta 		= cur_dev_run->dev.co2_dev.onoff_sta;
				save_run_data->dev.co2_dev.speed			= cur_dev_run->dev.co2_dev.speed;
				save_run_data->dev.co2_dev.mem_delay		= cur_dev_run->dev.co2_dev.mem_delay;
				save_run_data->dev.co2_dev.dev_keep_delay	= save_run_data->dev.co2_dev.mem_delay;
				break;
		}
	}
	return len;
}

