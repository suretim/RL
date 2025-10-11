#ifndef _AI_OUT_H_
#define _AI_OUT_H_

#include "types.h"
#include "define.h"
#include "device_type.h"
#include "type_struct.h"

#pragma pack(1)  // 

enum ai_workmode
{
AI_WORKMODE_PAUSE,
AI_WORKMODE_ON,
AI_WORKMODE_TENTWORK,
AI_WORKMODE_CNT,
};

enum port_workmode
{
WORKMODE_NORMAL,
WORKMODE_ADV,
WORKMODE_AI,
};

enum e_ai_pause_reason
{
	AI_PAUSE_DEFAULT = 0,
	AI_PAUSE_BY_KEY_SETTING,
	AI_PAUSE_SENSOR_CHANGE,
	AI_PAUSE_DEV_REMOVE,
	AI_PAUSE_REASON_MAX,
};

enum ENV_INDEX
{
	ENV_LIGHT,
	ENV_TEMP,
	ENV_HUMID,
	ENV_VPD,
	ENV_CO2,
	ENV_SOIL,
	ENV_WATTER,
	ENV_CNT,
};
#define ENV_BASE_CNT	(ENV_VPD+1)

#define ONE_DYA_MINUTE	(24*60)
typedef struct{
	bool refresh_start_time;	//刷新开始时间上传	- runnig
	bool refresh_24hour_sta;	//等待使能位更新上传 更新标记 - running
	uint8_t last_min;
	uint16_t setting_minute;	//刷新日出日落的时刻	- mem
	uint16_t setting_period;	//周期	- running

	bool beyoned_24_hour;		//满24小时 标记
	bool on_off_sw;				//重置状态		mem

	uint16_t start_minute;		//正在生效的开始时间	mem
	uint16_t min_temp_minute;	//最低温度对应的分钟	unuse
	int16_t  min_temp_f;		//对应最低温度值(华氏度F)-0.1	unuse

	uint16_t effective_data_cnt;	// 记录一个周期内(24h) 有效的数据个数(min)	running
	uint32_t sunStartCollectingUtc;	// 动态植物灯开始收集时间	mem
	uint32_t start_utc_sec;				// 日出开始时间对应的UTC (24小时内的)
	uint16_t cacalate_cnt;			// 更新日出日落的次数	unuse
	int16_t temp_f_tab[ONE_DYA_MINUTE];		//收集温度数据	running [0:00] -> [24:00]	:	0 -> ONE_DYA_MINUTE
}ml_sun_param_t;

extern ml_sun_param_t ml_sun_param;

typedef struct
{
	// 2bytes
	u8 targetTemp_C_max;
	u8 targetTemp_C_min;

	// 2bytes
	u8 targetTemp_F_max;
	u8 targetTemp_F_min;

	// 2bytes
	u8 targetHumid_max;
	u8 targetHumid_min;
} auto_mode_set_t; 

typedef struct
{
	u16 highVpd;  // 1.23Kpa  * 100
	u16 lowVpd;   // 1.23Kpa  * 100
	struct{
		u8 reserved 	: 4;
		u8 humid_emg	: 1;
		u8 humid_frist	: 1;
		u8 temp_emg		: 1;
		u8 temp_frist	: 1;
	}config;
	u8 temp_max_c;
	u8 temp_min_c;
	u8 temp_max_f;
	u8 temp_min_f;
	u8 temp_emerg_max_c;
	u8 temp_emerg_min_c;
	u8 temp_emerg_max_f;
	u8 temp_emerg_min_f;
	u8 humid_max;
	u8 humid_min;
	u8 humid_emerg_max;
	u8 humid_emerg_min;
} vpd_mode_set_t;

typedef struct{
	u8 max_lev	:	4;
	u8 min_lev	:	4;
}lev_set;


typedef struct{
	lev_set lev;
	struct{
		u8 reserved	:	6;
		u8 auto_light_data_collection_completed	: 1;
		u8 auto_light_en  	: 1;  // 0: disable, 1: enable 
	}config;
	u8 duration_hour;
	u8 duration_min;
	u8 start_hour;
	u8 start_min;
	u8 end_hour;
	u8 end_min;
	u8 period_hour;
	u8 period_min;
	u8 dynamicGrowLightStartHour;
	u8 dynamicGrowLightStartMin;
}growlight_set;

typedef struct 
{
	lev_set lev;
	lev_set humid_lev;	//暖雾挡位
}humid_set;

typedef struct{
	u8 reserve	:	7;
	u8 dynamicEn:	1;
} fan_config_t;

typedef struct{
	u8 reserve 		:	4;
	u8 enPro 		:	1;
	u8 humidPro 	:	1;
	u8 tempPro 		:	1;
	u8 smart_co2_en :	1;
} inlinefan_config_t;

typedef struct{
	lev_set lev;
	fan_config_t config;
	u8 start_hour;
	u8 start_min;
	u8 end_hour;
	u8 end_min;
	u8 degree;
}fan_set;

typedef struct{
	lev_set lev;
	inlinefan_config_t config;
}inlinefan_set;

typedef struct{
	u8 mode;
	u16 max_on_time;
	u8 target_max;
	u8 target_min;
	u16 soil_run_max;
	u8 safeShutOff;
}water_pump_set;

typedef struct{
	u16 target;
	u16 on_duration;
	u16 interval_duration;
}co2_generator_set;

typedef struct{
	lev_set lev;
}device_else_set;

typedef struct ST_ml_port_setting_st
{
	u8 origin_type;
	union
	{
		growlight_set 		growlight;
		humid_set			humid;
		fan_set 			fan;
		inlinefan_set		inlinefan;
		water_pump_set		water_pump;
		co2_generator_set	co2_generator;
		device_else_set		device_else;	//决定port数据长度
		u8 reserve[13];
	};
	u8 type;
}ml_port_setting_st;

typedef struct
{
	//------the first sent byte-------
	
    // 1byte 	- 	mlCtrByte
	// the lowest bit
    u8 reserved        : 4;
	u8 is_easy_mode    : 1;  // ai adv mode=0 / easy mode = 1
	u8 is_ai_deleted   : 1;  // 
	u8 ai_workmode     : 2;  // 0: stop, 1: on, 2: tent work(15min back to on)
	// the highest bit

	// 1byte
	u8 tentwork_period;

	// 1bytes	-	mlInfoByte
	// the lowest bit
	u8 temp_unit             : 1;   // 0: F,  1: C
	u8 xxx_reserved          : 1;  // 0: disable, 1: enable
	u8 is_night_run			 : 1;  // 当前运行 night 模式
	u8 pause_reason          : 4;  // 0: disable, 1: enable
	u8 switch_zone_position  : 1;  // 0 : not switch  1 : switch
	// the highest bit
	
	auto_mode_set_t autoMode;
    vpd_mode_set_t vpdMode;

    // 6bytes
    u8 reserved_bytes[6];  // co2, light

	struct
	{
	u8 reserved    : 5;
	u8 vpd_en      : 1;	
	u8 humid_en    : 1;
	u8 temp_en     : 1;
	}ai_mode_sel_bits;
	
	// 2bytes
	u16 tentwork_sparetime;

    // 3bytes
	struct 
	{
		uint32_t utc;	//ML设置时间点
	}start_time;

	uint8_t ml_grade;
	// 2bytes
    u16 ai_work_days;  // days

	// 4byes
	u32 ai_sensor_sel_bits; 

    // 1byte
	u8 ai_port_sel_bits;

	u16 plant_type;
	u8 growth_period;
	u32 update_time;
	u32 sunStartCollectingTime;
	// 7*PORT_CNT(9) bytes
	ml_port_setting_st port_ctrl[PORT_CNT];

	struct
	{
		u8 start_hour;
		u8 start_min;

		u8 reserve    				: 7;
		u8 collection_completed     : 1;
		
	}dynamic_sun;

	u8 reserv2_bytes[24];
	//------the last sent byte--------
}ai_setting_t;


typedef struct{
	lev_set lev;
	fan_config_t config;
	uint8_t degree;
}fan_night_set;

typedef struct{
	lev_set lev;
	inlinefan_config_t config;
}inlinefan_night_set;

typedef struct{
	lev_set lev;
}device_else_night_set;

typedef struct
{
	u8 origin_type;
	union
	{
		humid_set				humid;
		fan_night_set 			fan;
		inlinefan_night_set		inlinefan;
		device_else_night_set	device_else;
		uint8_t data[5];
	};
	u8 type;
}ml_port_nightsetting_st;

typedef struct{
	struct{
		u8 res:7;
		u8 en:1;   // 0:night mode disable  1:enable
	}config;

	u8 startHour;
	u8 startMin;
	u8 endHour;
	u8 endMin;

	auto_mode_set_t autoMode;
	vpd_mode_set_t vpdMode;
	ml_port_nightsetting_st port_ctrl[PORT_CNT];
}ai_night_mode_setting_t;

//------------------- ML LOG	---------------
typedef struct{
	union 
	{
		u32 data;
		struct
		{
			u32 ml_end_flag : 1; 
			u32 temp_100_flag : 1; 
			u32 temp_130_growlight_flag : 1; 
			u32 humid_90_without_fan_flag : 1; 
			u32 humid_fan_90_flag : 1; 
			u32 temp_130_flag : 1; 
			u32 humid_90_flag : 1; 
			u32 co2_200_flag  : 1;
			u32 soil_low_flag : 1;
			u32 water_no_flag : 1;
			u32 co2_reg_safety_shutoff_flag : 1;
			u32 high_co2_without_reg_flag : 1;
			u32 high_co2_with_reg_flag : 1;
		} bit;
	}flag;

	struct{
		u8 start_hour;
		u8 start_min;

		u8 period_hour;
		u8 period_min;
	}dynamic_sun;
	
}ml_log_mem_data_st;

extern ml_log_mem_data_st log_mem_data;	//x相关运行数据
//------------------- ML LOG	---------------
typedef struct ML_OUTPUT_PORT_T{
	struct 
	{
		uint8_t	speed_updata	: 1;
		uint8_t	mode_updata		: 1;
		uint8_t	reserved		: 6;
	}flag;
	
	uint8_t speed;
	uint8_t mode;
}ml_output_port_t;

typedef struct{
	struct{
		uint8_t match			:1;
		uint8_t can_restore		:1;
		uint8_t reserved		:6;
	}flag;
	uint8_t dev_type;
	uint8_t dev_origin;
	uint8_t orgin_delay;
	uint8_t dev_delay;
}dev_info_t;

typedef struct{
	uint8_t sensor_restore_err;	//传感器自动恢复
	uint8_t dev_restore_err;		//设备自动恢复
	uint8_t pause_can_restore;	//模式可以自动恢复
	uint8_t mode_restore;		//自动恢复的模式
}ai_mode_restore_t;

typedef struct {
	struct
	{
		u8 reserved    : 4;
		u8 is_ai_deleted:1;
		u8 vpd_en      : 1;	
		u8 humid_en    : 1;
		u8 temp_en     : 1;
	}ai_mode_sel_bits;

	dev_info_t 			port_dev_info[PORT_CNT];	
	ai_mode_restore_t	ai_restore;
	uint32_t tent_start_utc;
}ai_running_data_t;

#define ML_NAME_LEN	20
#define ML_APP_INFO_LEN  (130-ML_NAME_LEN)
typedef struct 
{
	u8 ml_name[ML_NAME_LEN];
	u8 ml_app_info[ML_APP_INFO_LEN];
}ai_app_info_t;

#define ML_45_TOTAL_LEN		120

#define ML_45_TARGET_LEN	3
#define ML_45_TAB_DEV_NUM	(loadType_pump-loadType_growLight+1)
#define ML_45_TAB_ENV_NUM	(4/*ENV_CNT*/)
#define ML_45_TAB_LEN		(ML_45_TAB_ENV_NUM*ML_45_TAB_DEV_NUM)
#define ML_45_REASV_LEN		(ML_45_TOTAL_LEN-ML_45_TARGET_LEN-ML_45_TAB_LEN)
typedef struct 
{
	u8 target[ML_45_TARGET_LEN];
	//s8 tab[ML_45_TAB_DEV_NUM][ML_45_TAB_ENV_NUM];
	//u8 reserved[ML_45_REASV_LEN];
}ai_ml_45_info_t;

extern ai_ml_45_info_t g_ai_ml_45_info;
extern ai_app_info_t g_ai_app_info;
extern ai_setting_t g_ai_setting;
extern ai_running_data_t g_ai_running_data;
extern ai_night_mode_setting_t g_ai_night_mode_setting;

extern bool g_is_ai_setting_changed;

extern bool g_ai_con_setting_changed;

extern curLoad_t old_ai_curLoad[PORT_CNT];
extern curLoad_t old_curLoad[PORT_CNT];

extern void ai_com_mutex_init(void);
extern void ai_com_mutex_wait(void);
extern void ai_com_mutex_give(void);

extern uint8_t ai_get_port_setting_type(ai_setting_t* app,uint8_t port);
extern uint8_t ai_get_port_setting_lev(ai_setting_t* p_ai_setting,uint8_t port,bool is_min_lev);
extern void ai_deal_before_rule_run(Time_Typedef * p_cur_time, ai_night_mode_setting_t * p_ai_night_setting, ai_setting_t * p_ai_setting_in, ai_setting_t * p_ai_setting_out);
extern void ai_deal_after_rule_run(ai_night_mode_setting_t * p_ai_night_setting, ai_setting_t * p_ai_setting_sys, ai_setting_t * p_ai_setting_run);
extern void aci_ai_entry( ai_setting_t* cur_ai_setting, Time_Typedef sys_time,dev_type_t* dev_type_list, s16* sensor_data_list, 
							u8* sen_sta_list, uint8_t* mode_list, rule_speed_t *dev_speeds, rule_port_set_t *dev_angles );
extern void ai_clr_setting_data(void);
extern bool ai_port_runnig(ai_setting_t* cur_setting_t,uint8_t port_num);
extern bool is_ai_port(uint8_t port_num);
extern void ai_pause_set(ai_setting_t* cur_ai_setting,enum e_ai_pause_reason reas);
extern bool ai_is_setting_updata(void);

extern bool ai_get_sensor_is_selected(ai_setting_t* app,uint8_t sensor_id);

extern uint16_t save_ai_run_data(uint8_t* p_buf);
extern uint16_t read_ai_run_data(uint8_t* p_buf);
extern uint16_t ai_save_night_data(uint8_t* p_buf);
extern uint16_t ai_read_night_data(uint8_t* p_buf);

//	AI SYNC
extern void ai_param_wait_app_set(void);
extern bool ai_param_wait_app_sync_sta(void);

extern uint16_t ml_sava_running_data(uint8_t* p_buf);
extern uint16_t ml_read_running_data(uint8_t* p_buf);

extern bool ai_cur_port_inabnormal_state(uint8_t port);

//	AI LOG
extern void ml_log_default_setting(void);

extern void ai_force_exit_deal(ai_setting_t* cur_ai_setting);

extern uint16_t ai_save_insight_data(uint8_t* p_buf);
extern uint16_t ai_read_insight_data(uint8_t* p_buf);

#pragma pack() 

#endif
