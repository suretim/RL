#include <stdio.h>
#include <string.h>

#include "includes.h"

//#include "Comm.h"
#include "rom/rtc.h"
// #include "esp_adc_cal.h"
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
#include "ml_pid.h"
#include "ml_rule.h"
#include "ai_insight.h"


extern bool is_second_func_port(u8 port);
extern bool port_is_hot_humid_run(uint8_t port);
extern bool port_is_hot_humid(uint8_t port);

ai_ml_45_info_t g_ai_ml_45_info;
ai_app_info_t g_ai_app_info;
ai_setting_t g_ai_setting;
ai_night_mode_setting_t g_ai_night_mode_setting;

bool ai_setting_changed_local=0;
bool g_ai_con_setting_changed=0;

bool ai_change_need_sync = false;
bool g_is_ai_setting_changed=true;
typedef struct{
	uint8_t pause_wait_restore	:	1;
	uint8_t wait_sensor_restore	:	1;	//sensor拔出进入Pause 置1，
	uint8_t wait_dev_restore	:	1;	//设备拔出进入Pause 置1，
	uint8_t reserved			:	5;
}ai_restore_ctr_t;
ai_restore_ctr_t ai_restore_ctr;

static uint8_t ai_port_err[PORT_CNT]={0};

const char* ml_mode_string[AI_WORKMODE_CNT]={
	"ml pause",
	"ml on",
	"ml tent",
};

#ifdef ANT_EXPLORE
uint8_t DropTroop[NumCities] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // power save
#endif

/////////////////////////////////porting layer///////////////////////////////

// ai debug 
#define TAG "    AI    "

// #define AI_RUN_DEBUG	//所有运行信息打印
// #define AI_INFO_BUG		//一些简单信息打印

#ifdef	AI_RUN_DEBUG
#define ai_run_info(format, ...)		ESP_LOGI(TAG, format, ##__VA_ARGS__)
#define ai_print(format, ...)  			esp_log_write(ESP_LOG_INFO,TAG, format, ##__VA_ARGS__)
#define ai_debug(TAG, format, ...)  	ESP_LOGI(TAG, format, ##__VA_ARGS__)

#ifdef AI_INFO_BUG
#undef AI_INFO_BUG
#endif

#else
#define ai_run_info(format, ...)
#define ai_print(format, ...)
#define ai_debug(TAG, format, ...)
#endif

#ifdef  AI_INFO_BUG
#define ai_debug(TAG, format, ...)  ESP_LOGI(TAG, format, ##__VA_ARGS__)
#else

#endif


extern u8 _getTargetDummyLoadType_port(u8 port_num);

//--------------------- usr struct start ------------------------------

//--------------------- usr struct end ------------------------------
ai_running_data_t g_ai_running_data;
static uint8_t ai_param_wait_sync = false;


bool ml_get_outside_temp_sensor_sta()
{
	return  (0 != g_sensors.zonetemp_c.dectected);
}

const char* pause_reason_string[AI_PAUSE_REASON_MAX]={
	"app set",
	"key set",
	"sensor change",
	"dev remove",
};

 
void ai_pause_set(ai_setting_t* cur_ai_setting,enum e_ai_pause_reason reas)
{
	if( cur_ai_setting->ai_workmode != AI_WORKMODE_PAUSE ){
		if( reas != AI_PAUSE_DEFAULT && reas != AI_PAUSE_BY_KEY_SETTING ){
			g_ai_running_data.ai_restore.mode_restore = g_ai_setting.ai_workmode;
			g_ai_running_data.ai_restore.pause_can_restore = true;
		}else{
			g_ai_running_data.ai_restore.pause_can_restore = false;
		}
	}
	cur_ai_setting->ai_workmode = AI_WORKMODE_PAUSE;
	cur_ai_setting->pause_reason = reas;
	ai_change_need_sync	= true;
	ESP_LOGE(TAG, "pause_reason=%s",pause_reason_string[reas]);
}

void port_enter_off(uint8_t port)
{
	curLoad[port].workMode = MODE_OFF;
}

///	通信互斥锁 处理通信异步
static SemaphoreHandle_t mutex_ai_com = NULL;
void ai_com_mutex_init()
{
	if( mutex_ai_com == NULL ){
		mutex_ai_com = xSemaphoreCreateMutex();
	}
}

void ai_com_mutex_wait()
{
	if( mutex_ai_com)
		xSemaphoreTake(mutex_ai_com, portMAX_DELAY);
}

void ai_com_mutex_give()
{
	if( mutex_ai_com)
		xSemaphoreGive(mutex_ai_com);
}

//	当前属于被AI选中
bool is_ai_port_bit(ai_setting_t* cur_setting_t,uint8_t port_num)
{
	if(port_num < 1 || port_num >= PORT_CNT )
		return false;
	return (cur_setting_t->ai_port_sel_bits&(1<<(port_num-1)));
}

//	当前被AI选中切端口类型匹配
bool is_ai_port(uint8_t port_num)
{
	if( false == is_ai_port_bit( &g_ai_setting , port_num ) )
		return false;
	// if( ai_port_err[port_num] ){
	// 	return false;
	// }
    return g_ai_running_data.port_dev_info[port_num].flag.match;
}

void ai_clear_port(ai_setting_t* cur_setting_t,uint8_t port_num)
{
	if( is_ai_port_bit(cur_setting_t,port_num) == false )
		return;
	cur_setting_t->ai_port_sel_bits &=~(1<<(port_num-1));
	ai_change_need_sync	= true;
}

bool ai_port_runnig(ai_setting_t* cur_setting_t,uint8_t port_num)
{
	if( AI_WORKMODE_PAUSE != cur_setting_t->ai_workmode && is_ai_port(port_num) )
	{
		return 1;
	}
	return 0;
}

inline void ml_set_mode(ml_output_port_t* output_port_list,uint8_t port,uint8_t mode)
{
	output_port_list[port].flag.mode_updata = true;
	output_port_list[port].mode = mode;
}

inline void ml_set_speed(ml_output_port_t* output_port_list,uint8_t port,uint8_t speed)
{
	output_port_list[port].flag.speed_updata = true;
	output_port_list[port].speed = speed;
}

void ml_set_port_off(uint8_t port,ml_output_port_t* output_port_list)
{
	ml_set_mode( output_port_list, port, MODE_OFF );
	ml_set_speed( output_port_list, port, 0 );
}

//	ml result sync dev setting
void ai_sync_angle( ai_setting_t* cur_ai_setting, rule_port_set_t *dev_angles)
{
	for(uint8_t i=0; i<PORT_CNT; i++){
		if( false == ai_port_runnig(cur_ai_setting,i) ){
			continue;
		}
		uint8_t set_type = ai_get_port_setting_type(cur_ai_setting,i);
		if( set_type == loadType_fan ){
			dev_angles[i].is_set = true;
			dev_angles[i].angle = cur_ai_setting->port_ctrl[i].fan.degree/9;
		}else if( set_type == loadType_humi ){
			dev_angles[i].warm_set = true;
			dev_angles[i].warm_lev = cur_ai_setting->port_ctrl[i].humid.humid_lev.max_lev;
			// ESP_LOGW(TAG,"port%d warm lev max_lev:%d", i,dev_angles[i].warm_lev);
		}
	}
}

void ai_sync_speed( ml_output_port_t* output_port_list, rule_speed_t *dev_speeds, dev_type_t* dev_type_list )
{
	for(uint8_t i=0; i<PORT_CNT; i++){
		if(output_port_list[i].flag.speed_updata){
			dev_speeds[i].is_set = true;
			dev_speeds[i].speed = output_port_list[i].speed;
			dev_type_list[i].using_type = loadType_nomatter;
		}
		if(output_port_list[i].flag.mode_updata){
			//mode_list[i] = output_port_list[i].mode;
		}
	}
}

inline uint8_t ai_get_port_setting_is_sw(ai_setting_t* app,uint8_t port)
{
	if(port >= PORT_CNT){
		return 0;
	}
	return (app->port_ctrl[port].type&0x80)?1:0;
}

inline uint8_t ai_get_port_setting_type(ai_setting_t* app,uint8_t port)
{
	if(port >= PORT_CNT){
		return loadType_nomatter;
	}
	return (app->port_ctrl[port].type&0x7f);
}

ml_port_setting_st* ai_get_dev_setting(ai_setting_t* app,uint8_t type)
{
	for(uint8_t i=0; i<PORT_CNT; i++){
		if( type != ai_get_port_setting_type(app,i) ){
			continue;
		}
		return app->port_ctrl+i;
	}
	return NULL;
}

/// @brief 
/// @param app 
/// @param sensor_id TEMP_F ... CO2 ... SEN_TYPE_CNT
/// @return 
bool ai_get_sensor_is_selected(ai_setting_t* app,uint8_t sensor_id)
{
	return (app->ai_sensor_sel_bits&(1<<sensor_id)) != 0;
}

uint8_t ai_get_port_setting_lev(ai_setting_t* p_ai_setting,uint8_t port,bool is_min_lev)
{
	if(port >= PORT_CNT){
		return 0;
	}
	uint8_t type = ai_get_port_setting_type(p_ai_setting,port);
	uint8_t is_sw = ai_get_port_setting_is_sw(p_ai_setting,port);
	uint8_t ret = 0;
	switch(type){
		case loadType_growLight:
		case loadType_humi:
		case loadType_dehumi:
		case loadType_heater:
		case loadType_A_C:
		case loadType_fan:
		case loadType_inlinefan:
		// case loadType_pump:
			if( is_min_lev ){
				ret = p_ai_setting->port_ctrl[port].device_else.lev.min_lev;
			}else{
				ret = p_ai_setting->port_ctrl[port].device_else.lev.max_lev;
			}
			break;
		default:
			if( is_min_lev ){ ret = 0; }else{ ret = 1; }
			break;
	}
	if(is_sw){
		if( is_min_lev ){ ret = 0; }else{ ret = 1; }
	}
	return ret;
}

static void SetLoad_OFF_tentwork(ai_setting_t* p_ai_setting, u8 ai_port_num, dev_type_t dev_type,ml_output_port_t* output_port_list) // 将负载设置为 OFF 状态
{
    u8 speed;
    u8 load_type = dev_type.using_type;
	
    if(loadType_growLight == load_type || loadType_fan == load_type )
		speed = 1;
	else
		speed = 0;
	if( loadType_inlinefan == load_type )	//管道风机 Tent运行用户设置最小档
	{
		speed = ai_get_port_setting_lev( p_ai_setting, ai_port_num, true);
	}
	if( dev_type.is_outlet && loadType_growLight != load_type )	//挡位虚拟类也为开关
		speed = 0;
	
	ml_set_speed(output_port_list, ai_port_num, speed );
}

void ml_log_get_loadtype(ai_setting_t* cur_setting_t,uint8_t *port_load_type,const dev_type_t* dev_type_list)
{
	for (uint8_t n = 0; n < PORT_CNT; n++){
		if (is_ai_port_bit(cur_setting_t, n)){
			port_load_type[n] = dev_type_list[n].using_type;
		}else{
			port_load_type[n] = loadType_nomatter;
		}
	}
}

void ai_clr_setting_data(void)
{
    memset(&g_ai_setting, 0, sizeof(g_ai_setting));
	memset(&g_ai_app_info, 0, sizeof(g_ai_app_info));
	memset(&g_ai_night_mode_setting, 0, sizeof(g_ai_night_mode_setting));
	g_ai_setting.is_ai_deleted = 1;
	//----	需要与系统设备同步的异常操作
	extern void sys_setting_sync_ml_smart_co2();
	sys_setting_sync_ml_smart_co2();
}

static uint8_t update_port_workmode0( ai_setting_t* cur_ai_setting, const dev_type_t* dev_type_list )
{
	uint8_t i;
	uint8_t cur_ai_workmode = cur_ai_setting->ai_workmode;

    // if(cur_ai_workmode!=AI_WORKMODE_ON && cur_ai_workmode!=AI_WORKMODE_TENTWORK)
	// 	return cur_ai_workmode;

    // set advMode[i] to ai mode or backup mode when ai port select is changed
	uint8_t have_ai_dev = 0;
	for(i=1; i<PORT_CNT; i++)
	{
	    if(is_ai_port(i)){
			if( loadType_nomatter != dev_type_list[i].using_type )
			{
				have_ai_dev = 1;
			}
		}
	}
	if( have_ai_dev == 0 ){
		g_ai_running_data.ai_restore.dev_restore_err = true;	
	}else{
		g_ai_running_data.ai_restore.dev_restore_err = false;	
	}
	if( have_ai_dev == 0 && ( cur_ai_workmode != AI_WORKMODE_PAUSE && g_ai_con_setting_changed == false ) ){
		ai_pause_set( cur_ai_setting, AI_PAUSE_DEV_REMOVE );
		ESP_LOGW(TAG, "all ai port offline");
	}
	return cur_ai_workmode;
}


void update_port_select(uint8_t new_port_sel_bits,uint8_t cui_ai_mode,ml_output_port_t* output_port_list)
{
    static u8 last_port_sel_bits=0;
	static u8 power_on_sync_ok = 0;
	uint8_t i;
	uint8_t sel_bit=0;
	
    if(last_port_sel_bits != new_port_sel_bits && power_on_sync_ok == 1  )
    {
        for(i=1; i<PORT_CNT; i++)
        {
            sel_bit = 1<<(i-1);
			// set to speed 0 if old sel port is cancled in new selection
            if((last_port_sel_bits&sel_bit) && !(new_port_sel_bits&sel_bit))	//设备移除
			{
			    ai_debug(TAG, "ai port%d cancel", i);

				if( cui_ai_mode != AI_WORKMODE_PAUSE )
				{
					ml_set_port_off( i, output_port_list );
				}
            }
	
		    if(!(last_port_sel_bits&sel_bit) && (new_port_sel_bits&sel_bit)){	//设备添加

			}
        }
    }
	power_on_sync_ok = 1;
	last_port_sel_bits = new_port_sel_bits;
}


// called per sec
static void tent_work_process(ai_setting_t* cur_ai_setting, ai_running_data_t* ai_running,const dev_type_t* dev_type_list, uint8_t sec_flag,ml_output_port_t* output_port_list)
{
    static u16 tentwork_secs_cnt=0xffff;	//非法值 用作倒计时上电更新
    uint8_t i;
	
	if( tentwork_secs_cnt == 0xffff ){
		tentwork_secs_cnt = get_utc_time() - ai_running->tent_start_utc;
		if( ai_running->tent_start_utc > get_utc_time() ){
			tentwork_secs_cnt = 0;
		}
		if( (ai_running->tent_start_utc+15*60) < get_utc_time() ){
			tentwork_secs_cnt = 15*60;
		}
		ai_change_need_sync = true;
	}
	// reset tent work time downcounter when tent work mode is enabled.
	if(cur_ai_setting->ai_workmode != AI_WORKMODE_TENTWORK  )
	{
		if( ai_running->ai_restore.pause_can_restore == false ){
			tentwork_secs_cnt = 0;
			cur_ai_setting->tentwork_sparetime = 15*60 - tentwork_secs_cnt;
		}
	    return;
	}

	// 15 min
	if(tentwork_secs_cnt < 15*60){
		if( sec_flag ){
			if(tentwork_secs_cnt == 0){ 
				ai_change_need_sync = true; 
				ai_running->tent_start_utc = get_utc_time();
			}
			tentwork_secs_cnt++;
		}
	} 
	else{
	    if(cur_ai_setting->ai_workmode != AI_WORKMODE_ON)
			ai_change_need_sync = true;
		cur_ai_setting->ai_workmode = AI_WORKMODE_ON;
	}

	cur_ai_setting->tentwork_sparetime = 15*60 - tentwork_secs_cnt;
	//ai_change_need_sync = true;
	
	ai_debug(TAG, "_____tent work%d:%d___", cur_ai_setting->tentwork_sparetime/60,cur_ai_setting->tentwork_sparetime%60);
	
	for(i=1; i<PORT_CNT; i++)
	{
	    if(is_ai_port(i))
		    SetLoad_OFF_tentwork( cur_ai_setting, i, dev_type_list[i], output_port_list );
	}
}

static void easy_mode_day_run(Time_Typedef *p_sys_time)
{
	//	Easy 模式暂停
	// if( g_ai_setting.ai_workmode == AI_WORKMODE_PAUSE )
	// 	return;

	// if( g_ai_setting.is_easy_mode ){
	// 	// if( true == isExpired( &g_ai_setting, p_sys_time ) ){
	// 	// 	ai_pause_set(g_ai_setting.pause_reason);
	// 	// 	ai_debug(TAG,"easy mode over\n");
	// 	// }
	// }
}

//	num:0-not zone  1-zone
static uint8_t sensor_port[2] = {0,0};	//value: 0-offline 1-port1 2-port2 3-zone
static uint8_t sensor_delay[2] = {0,0};
uint8_t get_side_sensor_port(uint8_t is_zone)
{
	uint8_t detel = 0;
	if( is_zone == true ){
		detel = sensor_port[0];
	}else{
		detel = sensor_port[1];
	}
	return detel;
}

bool get_outside_inside_sensor_in_delay_sta(bool is_outside)
{
	uint8_t ret = 0;
	if( g_ai_setting.switch_zone_position == 0){
		ret = sensor_delay[is_outside];
	}else{
		ret = sensor_delay[!is_outside];
	}
	return (ret>0);
}

// @brief 逻辑：传感器掉线后 延时5S 
void ai_sensor_det_deal(ai_setting_t* cur_ai_setting,ai_running_data_t* cur_run_data,uint8_t sec_flag)
{
#if 1
	//	如果 zone 传感器 切换到板载 说明传感器位置改变
	//	0-板载	1-sensor1	2-sensor2
	static uint8_t last_mode = 0xff;
	
	//0: zone sensor 位置	1:sensor 位置
	uint8_t sensor_port_new[2] = {0,0};	

	if( last_mode == 0xff ){
		last_mode = cur_ai_setting->ai_workmode;
	}

	sensor_port_new[0] = g_sensors.zone_humid.dectected;
	sensor_port_new[1] = g_sensors.humid.dectected;

	for(uint8_t i=0; i<2; i++ ){
		//	发生状态改变	之前有设备在线	并且	不是板载传感器
		if(sensor_port[i] != sensor_port_new[i] && sensor_port[i] != 0 && sensor_port[i] != 7  ){
			if(sec_flag)	sensor_delay[i]++;
			if( sensor_delay[i] > 5 ){
				sensor_delay[i] = 0;
				sensor_port[i] = sensor_port_new[i];
			}
			//	使用 delay 作为 判断掉线的逻辑
			if( sensor_delay[i] == 0 ){
				sensor_delay[i] = 1;
			}
		}else{
			sensor_delay[i] = 0;
			sensor_port[i] = sensor_port_new[i];
		}
		if( (ai_setting_changed_local == true || g_ai_con_setting_changed )&& \
			(cur_ai_setting->ai_workmode != last_mode && cur_ai_setting->ai_workmode == AI_WORKMODE_ON ) ){
			sensor_delay[i] = 0;
			sensor_port[i] = sensor_port_new[i];
			// ESP_LOGW(TAG,"sensor[%d]:%d",i,sensor_port[i]);
		}
		// if( sec_flag ){
		// 	ESP_LOGW(TAG,"sensor[%d]:%d",i,sensor_port[i]);
		// }
	}

	// if(sec_flag){
	// ESP_LOGE(TAG,"%d %d %d %d",g_sensors.zone_humid.dectected,g_sensors.humid.dectected,g_sensors2.zone_humid.dectected,g_sensors2.zone_humid.dectected);
	// }

	if( g_ai_con_setting_changed == false && last_mode != cur_ai_setting->ai_workmode ){
		last_mode = cur_ai_setting->ai_workmode;
	}

	//-------------------------------    传感器掉线处理    -------------------------------
	//	

	if( AI_WORKMODE_PAUSE != cur_ai_setting->ai_workmode )
	{
		//	传感器掉线 || 传感器切换
		if( !sensor_port[1] || !sensor_port[0]  )  //fix:AI运行中拔掉外接温湿度传感器，AI运行暂停
		{
			ai_pause_set(cur_ai_setting,AI_PAUSE_SENSOR_CHANGE);
			ai_debug(TAG,"sensor cg 2\n");
		}
	}
	if( !sensor_port[1] || !sensor_port[0]  ) {
		cur_run_data->ai_restore.sensor_restore_err = true;	//传感器错误状态
	}else{
		cur_run_data->ai_restore.sensor_restore_err = false;	//传感器正常状态
	}


	// if( (sensor_bit_new&sensor_bit) != sensor_bit ){
	// 	//	要确保数据同步
	// 	if( AI_WORKMODE_PAUSE != g_ai_setting.ai_workmode && last_mode == g_ai_setting.ai_workmode ){
	// 		ESP_LOGE(TAG,"old sta:0x%02x, new sta:0x%02x",sensor_bit,sensor_bit_new);
	// 		ai_pause_set(AI_PAUSE_SENSOR_CHANGE);
	// 		ai_debug(TAG,"sensor cg 2\n");
	// 	}
	// }
#endif
}

void clean_ai_restore()
{
	g_ai_running_data.ai_restore.pause_can_restore 	= false;
	// g_ai_running_data.ai_restore.dev_restore_err 	= false;
	// g_ai_running_data.ai_restore.sensor_restore_err 	= false;
	// g_ai_running_data.ai_restore.mode_restore 		= 0;
}

//	AI自动恢复处理
void ai_restore_deal(ai_setting_t* cur_ai_setting)
{
	if( g_ai_running_data.ai_restore.pause_can_restore == false ){
		return;
	}
	if( g_ai_running_data.ai_restore.dev_restore_err == true ){
		return;
	}
	if( g_ai_running_data.ai_restore.sensor_restore_err == true ){
		return;
	}
	if( g_ai_running_data.ai_restore.mode_restore != cur_ai_setting->ai_workmode ){
		cur_ai_setting->ai_workmode = g_ai_running_data.ai_restore.mode_restore;
		ai_change_need_sync	= true;
		ESP_LOGW(TAG,"ai auto restore [%d]",g_ai_running_data.ai_restore.mode_restore);
	}
}

int16_t ml_temp_tran_unit_f( uint16_t data, uint8_t is_unit_f )
{
	float temp = data/100.0f;
	if( false == is_unit_f ){
		temp = C_to_F( temp );
	}
	return temp*100;
}

//	在内存结构改变时 需要同步动态光设置
void ai_param_wait_app_set()
{
	 ai_param_wait_sync = true;
}

bool ai_param_wait_app_sync_sta()
{
	return ai_param_wait_sync;
}


/// @brief 升级完成 同步设置数据
/// @param app 
/// @param p_ml_sun_param 
void ai_param_app_sync(ai_setting_t* app,ml_sun_param_t* p_ml_sun_param)
{
	if( false == ai_param_wait_app_sync_sta() )
		return;

	if( false == g_ai_con_setting_changed )	//等待指令下发
		return;

	ai_param_wait_sync = false;
	//ml_sun_param;
	uint8_t start_hour=0,start_min=0,collection_completed =0;

	start_hour = app->dynamic_sun.start_hour;
	start_min = app->dynamic_sun.start_min;
	collection_completed = app->dynamic_sun.collection_completed;
#if 0
	for(uint8_t i=0; i<PORT_CNT; i++){
		if( loadType_growLight != ai_get_port_setting_type(app,i) ){
			continue;
		}
		if( app->port_ctrl[i].growlight.config.auto_light_data_collection_completed == 1 ){
			collection_completed = 1;
			start_hour = app->port_ctrl[i].growlight.dynamicGrowLightStartHour;
			start_min = app->port_ctrl[i].growlight.dynamicGrowLightStartMin;
			break;
		}
	}
#endif

	p_ml_sun_param->beyoned_24_hour = 0;
	if( collection_completed ){
		p_ml_sun_param->beyoned_24_hour = 1;
		p_ml_sun_param->on_off_sw = 1;
		
		p_ml_sun_param->sunStartCollectingUtc = app->sunStartCollectingTime;
		
		Time_Typedef temp;
		temp = *get_cur_time();
		temp.hour = start_hour;
		temp.min = start_min;
		temp.sec = 0;
		p_ml_sun_param->start_utc_sec = rtc_to_real_utc(temp);
	}
	ESP_LOGW(TAG,"%s() start_collect[%ld] start[%d:%d] flag[%d]",__func__,
			p_ml_sun_param->sunStartCollectingUtc,start_hour,start_min,p_ml_sun_param->beyoned_24_hour);
}

uint16_t ai_sun_get_setting_period(ai_setting_t *ai_setting,ml_sun_param_t* p_ml_sun_param)
{
	uint16_t new_period = 0;
	for(uint8_t i=0; i<PORT_CNT; i++){
		if( 0 == is_ai_port(i) || loadType_growLight != ai_get_port_setting_type(ai_setting,i) ){
			continue;
		}
		if( ai_setting->port_ctrl[i].growlight.config.auto_light_en == 0 ){
			continue;
		}
		new_period = ai_setting->port_ctrl[i].growlight.period_hour*60 + ai_setting->port_ctrl[i].growlight.period_min;
	}
	return new_period;
}

void ai_sun_dynamic_sync_before( ai_setting_t *ai_setting, ml_sun_param_t* p_ml_sun_param)
{
		//	设置参数改变
	uint16_t new_period = ai_sun_get_setting_period(ai_setting, p_ml_sun_param);
	if( new_period != p_ml_sun_param->setting_period ){
		p_ml_sun_param->setting_period = new_period;
	}
}


void ai_sun_dynamic_sync_setting(ai_setting_t *ai_setting, ml_sun_param_t* p_ml_sun_param )
{
	bool sync_flag = 0;
	uint8_t start_hour =0, start_min = 0;
	start_hour = p_ml_sun_param->start_minute/60;
	start_min = p_ml_sun_param->start_minute%60;
	for(uint8_t i=0; i<PORT_CNT; i++){
		if( 0 == is_ai_port_bit(ai_setting, i) || loadType_growLight != ai_get_port_setting_type(ai_setting,i) ){
			continue;
		}
		if( p_ml_sun_param->beyoned_24_hour != ai_setting->port_ctrl[i].growlight.config.auto_light_data_collection_completed ){
			ai_setting->port_ctrl[i].growlight.config.auto_light_data_collection_completed = p_ml_sun_param->beyoned_24_hour;
			sync_flag = 1;
		}
		ai_setting->port_ctrl[i].growlight.dynamicGrowLightStartHour = start_hour;
		ai_setting->port_ctrl[i].growlight.dynamicGrowLightStartMin = start_min;
	}
	if( p_ml_sun_param->sunStartCollectingUtc !=  ai_setting->sunStartCollectingTime ){
		ai_run_info("sunStartCollectingUtc:%ld",p_ml_sun_param->sunStartCollectingUtc);
		ai_setting->sunStartCollectingTime = p_ml_sun_param->sunStartCollectingUtc;
		sync_flag = 1;
	}

	if( ai_setting->dynamic_sun.start_hour != start_hour ||
		ai_setting->dynamic_sun.start_min != start_min ||
		ai_setting->dynamic_sun.collection_completed != p_ml_sun_param->beyoned_24_hour
	 ){
		ai_setting->dynamic_sun.start_hour = start_hour;
		ai_setting->dynamic_sun.start_min = start_min;
		ai_setting->dynamic_sun.collection_completed = p_ml_sun_param->beyoned_24_hour;
		ESP_LOGI(TAG,"dynamic h(%d) min(%d) over(%d)", start_hour, start_min, p_ml_sun_param->beyoned_24_hour);
		sync_flag = 1;
	}

	if( p_ml_sun_param->refresh_start_time ){
		p_ml_sun_param->refresh_start_time = 0;
		ai_change_need_sync = 1;	//上报改变
		ai_run_info("light start refresh");
	}

	// sun_debug("sun sync:%d", p_ml_sun_param->beyoned_24_hour);
	if( sync_flag /*|| p_ml_sun_param->refresh_24hour_sta*/ ){
		ai_change_need_sync = 1;	//上报改变
		p_ml_sun_param->refresh_24hour_sta = 0;
		ai_run_info("light en refresh");
	}
}

void ai_app_45_rset(ai_ml_45_info_t* p_ml_45_info)
{
	memset(p_ml_45_info,0x00,sizeof(ai_ml_45_info_t));
	setChgUpload(TYP_ML_AI_INFO, 0);
}

void ai_app_45_flash_target_sta(ai_ml_45_info_t* p_ml_45_info,const ml_running_data_t *running_data)
{
	bool target_change = 0;

	for(uint8_t i=1; i<ENV_BASE_CNT; i++)
	{
		// if( p_ml_45_info->target[i-1] != running_data->env_data.env_run_data[i].is_reach_target )
		// 	target_change = 1;
		// p_ml_45_info->target[i-1] = running_data->env_data.env_run_data[i].is_reach_target;
	}
	if( target_change == 0 )
		return;

	setChgUpload(TYP_ML_AI_INFO, 0);
}

//	检测AI运行状态，是否有模式改变，设置改变，重置运行状态
void ai_detect_change_reset(ai_running_data_t* p_record,ai_setting_t *ai_setting,ml_running_data_t* p_running_data)
{
	if(	ai_setting->ai_mode_sel_bits.humid_en != p_record->ai_mode_sel_bits.humid_en || 
		ai_setting->ai_mode_sel_bits.temp_en != p_record->ai_mode_sel_bits.temp_en ||
		ai_setting->ai_mode_sel_bits.vpd_en != p_record->ai_mode_sel_bits.vpd_en ||
		ai_setting->is_ai_deleted != p_record->ai_mode_sel_bits.is_ai_deleted
	)
	{
		p_record->ai_mode_sel_bits.humid_en = ai_setting->ai_mode_sel_bits.humid_en;
		p_record->ai_mode_sel_bits.temp_en =ai_setting->ai_mode_sel_bits.temp_en;
		p_record->ai_mode_sel_bits.vpd_en = ai_setting->ai_mode_sel_bits.vpd_en;
		p_record->ai_mode_sel_bits.is_ai_deleted = ai_setting->is_ai_deleted;

		clean_ai_restore();
		ml_rest_run_sta( p_running_data ); 

		ai_app_45_rset( &g_ai_ml_45_info );
		ESP_LOGI(TAG,"ai setting rst!");
	}
}

#define DELAY_PORT_DEV		30
#define DELAY_PORT_ORRGIN	30
/// @brief 端口处于识别状态
/// @param port 
/// @return 
bool dev_is_being_recognized(uint8_t port)
{
	if( false == is_ai_port(port) ){
		return true;
	}
	if( g_ai_running_data.port_dev_info[port].dev_delay< DELAY_PORT_DEV ){
		return true;
	}
	if( g_ai_running_data.port_dev_info[port].orgin_delay < DELAY_PORT_ORRGIN ){
		return true;
	}
	return false;
}

uint8_t ai_get_dev_origin_type(dev_type_t dev_type,uint8_t port)
{
	uint8_t dev_origin = loadType_nomatter ;	//get dev from
	// if( dev_type.real_type != dev_type.using_type && is_second_func_port(port) ){
	// 	dev_origin = env_dev_origin_seconed;
	// }
	if( dev_type.real_type == loadType_switch ){
		dev_origin = loadType_switch;
	}else if( dev_type.real_type == loadType_A_C && is_second_func_port(port) ){
		dev_origin = loadType_A_C;
	}else if( dev_type.using_type == loadType_humi && port_is_hot_humid(port) ){
		dev_origin = loadType_heater;
	}
	return dev_origin;
}

bool ai_get_dev_sensor_ok( ai_setting_t* cur_ai_setting, uint8_t dev_type,uint8_t* sensor_sta)
{
	bool ret = true;
	ml_port_setting_st* port_set = NULL;
	port_set = ai_get_dev_setting(cur_ai_setting,dev_type);
	if( port_set == NULL ){
		return false;
	}
	switch( dev_type )
	{
		case loadType_water_pump:
		case loadType_pump:
			if( port_set->water_pump.mode == 0 )	//水
			{
				if( false == ai_get_sensor_is_selected(cur_ai_setting, WATER_LEVEL)  ||
					0 == sensor_sta[WATER_LEVEL] ){
					ret = false;
				}
			}else{
				if( false == ai_get_sensor_is_selected(cur_ai_setting, SOIL_HUMID)  ||
					0 == sensor_sta[SOIL_HUMID] ){
					ret = false;
				}
			}
			break;
		case loadType_co2_generator:
			if( false == ai_get_sensor_is_selected(cur_ai_setting, CO2)  ||
				0 == sensor_sta[CO2] ){
				ret = false;
			}
			break;
	}
	return ret;
}

/// @brief 管理设备类型改变 清除AI选中端口
/// @param dev_type_list 
/// @param sec_flag 
void ai_port_dev_manage( ai_setting_t* cur_ai_setting,const dev_type_t* dev_type_list, uint8_t sec_flag, 
						uint8_t* mode_list,uint8_t* sensor_sta_list,ai_running_data_t* ai_running_data)
{
	if( true == ai_setting_changed_local ){
		ESP_LOGW(TAG,"dev updata");
		for(uint8_t i=1; i<PORT_CNT; i++ ){
			dev_info_t* cur_port_dev = &(ai_running_data->port_dev_info[i]);
			if( is_ai_port_bit(cur_ai_setting,i) ){
				// uint8_t last_type = cur_port_dev->dev_type;
				cur_port_dev->dev_type = ai_get_port_setting_type( cur_ai_setting, i );//dev_type_list[i].using_type;
				// if( last_type != cur_port_dev->dev_type || dev_type_list[i].using_type != loadType_nomatter ){
					cur_port_dev->dev_origin = ai_get_dev_origin_type( dev_type_list[i], i );	//更新逻辑
				// }
				cur_port_dev->flag.match = true;
				cur_port_dev->flag.can_restore = true;
			}else{
				cur_port_dev->dev_type = loadType_nomatter; 
				cur_port_dev->dev_origin = 0;
				cur_port_dev->orgin_delay = 0;
				cur_port_dev->dev_delay = 0;
				cur_port_dev->flag.match = false;
				cur_port_dev->flag.can_restore = false;
			}
		}
		return;
	}

	for( uint8_t i=1; i<PORT_CNT; i++ ){
		dev_info_t* cur_port_dev = &(ai_running_data->port_dev_info[i]);
		if( is_ai_port_bit(cur_ai_setting,i) ){
			uint8_t dev_origin = ai_get_dev_origin_type(dev_type_list[i], i);
			uint8_t dev_type = dev_type_list[i].using_type;

			//	----------- 掉线重新判断 -----------
			// if( curLoad[i].plug_in == false || dev_type == loadType_nomatter ){
			// 	cur_port_dev->orgin_delay = 0;
			// 	cur_port_dev->dev_delay = 0;
			// 	continue;
			// }

			// if( cur_port_dev->flag.can_restore && mode_list[i] != MODE_OFF ){
			// 	cur_port_dev->flag.can_restore = 0;	
			// 	ai_clear_port(cur_ai_setting, i);
			// 	ESP_LOGI(TAG,"port[%d] cannot try again",i);			
			// }

			if( cur_port_dev->dev_type == dev_type && \
				cur_port_dev->dev_origin == dev_origin &&
				ai_get_dev_sensor_ok( cur_ai_setting, cur_port_dev->dev_type, sensor_sta_list ) ){
				if( cur_port_dev->flag.can_restore ){
					cur_port_dev->flag.match = true;
				}
			}else{
				if( cur_port_dev->flag.match == true ){
					cur_port_dev->flag.match = false;
					ESP_LOGW(TAG,"port[%d] dev origin=[%d]->[%d]", i, cur_port_dev->dev_origin, dev_origin );
					ESP_LOGW(TAG,"port[%d] dev type=[%d]->[%d]", i, cur_port_dev->dev_type, dev_type );
				}
			}
		}else{
			cur_port_dev->orgin_delay = 0;
			cur_port_dev->dev_delay = 0;
			cur_port_dev->flag.match = false;
			cur_port_dev->flag.can_restore = false;
		}
	}
}

//	设备处于异常状态 
bool ai_cur_port_inabnormal_state(uint8_t port)
{
	extern bool dev_is_being_recognized(uint8_t port);

	if( port >= PORT_CNT)
		return true;

	if( true == m_port_dev_err_run(port) || dev_is_being_recognized(port) ){
		return true;
	}
	return false;
}

void ai_force_exit_deal(ai_setting_t* cur_ai_setting)
{
	if( cur_ai_setting->ai_workmode != AI_WORKMODE_PAUSE ){
		for(uint8_t i=1; i<PORT_CNT; i++)
		{
			if(is_ai_port_bit(cur_ai_setting,i)){
				port_enter_off(i);
			}
		}
	}
	cur_ai_setting->ai_workmode = AI_WORKMODE_PAUSE;
}

void ai_online_port_enter_off(ai_setting_t* cur_ai_setting,ml_output_port_t* ml_output_port_list)
{
	for(uint8_t i=1; i<PORT_CNT; i++)
	{
		if(is_ai_port(i) && cur_ai_setting->ai_workmode != AI_WORKMODE_PAUSE ){
			ml_set_port_off(i,ml_output_port_list);
		}
	}
}

/// @brief 处理 ml 删除 模式改变的 端口处理
/// @param ml_output_port_list 
void ai_mode_change_manage(ai_setting_t* cur_ai_setting,ml_output_port_t* ml_output_port_list)
{
	static uint8_t last_ai_workmode=AI_WORKMODE_PAUSE;

	if(cur_ai_setting->is_ai_deleted){   //fixbug-v40: ai删除时要同步
		cur_ai_setting->ai_workmode = AI_WORKMODE_PAUSE;
		clean_ai_restore();
	}

	if( cur_ai_setting->ai_workmode != AI_WORKMODE_PAUSE ){
		clean_ai_restore();
	}

	// ai paras will be reset when ai closed
    if( cur_ai_setting->ai_workmode != last_ai_workmode ){
		if( AI_WORKMODE_PAUSE == cur_ai_setting->ai_workmode ){
			for(uint8_t i=1; i<PORT_CNT; i++){
				if(is_ai_port_bit(cur_ai_setting,i)){
					ml_set_port_off(i,ml_output_port_list);
				}
			}
		}
		//	从Pause 切到 on/tent 模式也切换到OFF for UI display
		if( last_ai_workmode == AI_WORKMODE_PAUSE ){
			for(uint8_t i=1; i<PORT_CNT; i++){
				if(is_ai_port_bit(cur_ai_setting,i)){
					ml_set_mode( ml_output_port_list, i, MODE_OFF );
				}
			}
		}
		last_ai_workmode = cur_ai_setting->ai_workmode;
		ESP_LOGI(TAG,"cur mode: %s",ml_mode_string[last_ai_workmode]);
    }

	if( cur_ai_setting->is_ai_deleted ){
		cur_ai_setting->ai_port_sel_bits = 0;
	}
}

void ai_clean_pause_rason(ai_setting_t* cur_ai_setting)
{
	//	恢复运行时 清除 pause reason
	if( cur_ai_setting->pause_reason != 0 && cur_ai_setting->ai_workmode != AI_WORKMODE_PAUSE ){
		cur_ai_setting->pause_reason = 0;
	}
}

void ai_refresh_port_err_sta(const dev_type_t* dev_type_list, uint8_t* port_err_list)
{
	for(uint8_t i=0; i<PORT_CNT; i++ ){
		port_err_list[i] = dev_type_list[i].has_err;
	}
}

//	-------------------------------- 对外接口 start --------------------------------
uint16_t save_u16_to_data(uint16_t value,uint8_t* data)
{
	uint16_t i= 0;
	data[i++] = (value >> 8) & 0xff;
	data[i++] = (value) & 0xff;
	return i;
}

uint16_t read_u16_from_data(uint16_t* p_value,uint8_t* data)
{
	uint16_t t = 0;
	uint16_t i= 0;

	t |= data[i++] << 8;	// 45
	t |= data[i++] << 0;
	*p_value = t;
	return i;
}


uint16_t save_ai_run_data(uint8_t* p_buf)
{
	uint16_t len = 0;
	write_atom_unit( &g_ai_running_data, p_buf+len, &len );
	return len;
}

uint16_t read_ai_run_data(uint8_t* p_buf)
{
	uint16_t len = 0;

	read_atom_unit( &g_ai_running_data, p_buf+len, &len );
	
	return len;
}


uint16_t ai_save_night_data(uint8_t* p_buf)
{
	uint16_t len = 0;

	memcpy(&(p_buf[len]), &g_ai_night_mode_setting, sizeof(ai_night_mode_setting_t) );
	len += sizeof(ai_night_mode_setting_t);

	return len;
}

uint16_t ai_read_night_data(uint8_t* p_buf)
{
	uint16_t len = 0;

	memcpy(&g_ai_night_mode_setting, &(p_buf[len]), sizeof(ai_night_mode_setting_t) );
	len += sizeof(ai_night_mode_setting_t);
	
	return len;
}
//	-------------------------------- 对外接口 end --------------------------------

void ai_updata_setting_operate()
{
	if( true == ai_setting_changed_local ){
		ai_setting_changed_local = false;
	}
	if( true == g_ai_con_setting_changed ){
		ai_setting_changed_local = true;
		g_ai_con_setting_changed = false;
	}
}

bool ai_is_setting_updata()
{
	return ai_setting_changed_local;
}

void check_ai_change_need_updata()
{
	if( ai_change_need_sync ){
		g_is_ai_setting_changed = 1;
		ESP_LOGW(TAG,"ai setting updata!!!");
	}
	ai_change_need_sync = 0;
}

uint8_t ai_dev_type_2_ml_dev_type(uint8_t ai_dev_type)
{
	uint8_t env_type = env_dev_type_none;
	switch( ai_dev_type ){
		case loadType_growLight:
			env_type = ml_dev_type_growlight;
			break;
		case loadType_humi:
		case loadType_dehumi:
		case loadType_heater:
		case loadType_A_C:
			env_type = (ai_dev_type-loadType_humi)+env_dev_type_humid;
			break;
		case loadType_inlinefan:
			env_type = env_dev_type_inlinefan;
			break;
		case loadType_fan:
			env_type = ml_dev_type_fan;
			break;
		case loadType_pump:
			env_type = ml_dev_type_pump;
			break;
		case loadType_water_pump:
			env_type = ml_dev_type_water_pump;
			break;
		case loadType_co2_generator:
			env_type = ml_dev_type_co2_generator;
			break;
		case loadType_switch:
			env_type = ml_dev_type_switch;
			break;
		default: break;
	}
	return 	env_type;
}

void ai_night_mode_log(uint8_t is_night_mode)
{
	static uint8_t cur_mode = 0xff;
	if( cur_mode == 0xff ){
		cur_mode = is_night_mode;
	}
	if( cur_mode != is_night_mode ){
		cur_mode = is_night_mode;
		ai_change_need_sync = 1;
		if( cur_mode == 0 ){
			ESP_LOGW(TAG,"exit night mode !!!");
		}else{
			ESP_LOGW(TAG,"enter night mode !!!");
		}
	}
}

bool ai_get_light_run_time( ai_setting_t * cur_ai_info,uint8_t* start_hour,uint8_t* start_min,uint8_t* end_hour,uint8_t* end_min)
{
	growlight_set* light_setting = NULL;
	for( uint8_t port=0; port<PORT_CNT; port++ ){
		if( false == is_ai_port(port) ){
			continue;
		}
		if( ai_get_port_setting_type(cur_ai_info, port) == loadType_growLight ){
			light_setting = &(cur_ai_info->port_ctrl[port].growlight);
			break;
		}
	}
	if( light_setting == NULL ){
		return false;
	}
	//	动态灯光使能
	if( light_setting->config.auto_light_en ){
		uint16_t dynamic_end_min = 0;
		dynamic_end_min = light_setting->dynamicGrowLightStartHour*60 + light_setting->dynamicGrowLightStartMin + light_setting->period_hour*60 +light_setting->period_min ;
		dynamic_end_min = dynamic_end_min%(24*60);
		*start_hour = light_setting->dynamicGrowLightStartHour;
		*start_min  = light_setting->dynamicGrowLightStartMin;
		*end_hour   = dynamic_end_min/60;
		*end_min   	= dynamic_end_min%60;
	}else{
		*start_hour = light_setting->start_hour;
		*start_min  = light_setting->start_min;
		*end_hour   = light_setting->end_hour;
		*end_min   	= light_setting->end_min;
	}
	return true;
}

void ai_deal_before_rule_run(Time_Typedef * p_cur_time, ai_night_mode_setting_t * p_ai_night_setting, ai_setting_t * p_ai_setting_in, ai_setting_t * p_ai_setting_out)
{
	ai_com_mutex_wait();

	ai_param_app_sync(p_ai_setting_in, &ml_sun_param );

	/* 先复制整个 ai_setting */
	bool have_grow_light = false;
	memcpy(p_ai_setting_out, p_ai_setting_in, sizeof(ai_setting_t));
	
    if (!p_ai_night_setting->config.en){
		ai_night_mode_log(0);
		p_ai_setting_out->is_night_run = false;
		return;
	}

    uint8_t start_hour,start_min,end_hour,end_min;	//夜间模式时间
	// 植物灯参数替换逻辑
	if( false == ai_get_light_run_time(p_ai_setting_in,&end_hour,&end_min,&start_hour,&start_min) ){
		start_hour = p_ai_night_setting->startHour;
		start_min = p_ai_night_setting->startMin;
		end_hour = p_ai_night_setting->endHour;
		end_min = p_ai_night_setting->endMin;
	}else{
		//	主动更行 night 时间
		if( p_ai_night_setting->startHour != start_hour || \
			p_ai_night_setting->startMin != start_min || \
			p_ai_night_setting->endHour != end_hour || \
			p_ai_night_setting->endMin != end_min){
			
			p_ai_night_setting->startHour = start_hour;
			p_ai_night_setting->startMin = start_min;
			p_ai_night_setting->endHour = end_hour;
			p_ai_night_setting->endMin = end_min;
			ai_change_need_sync	= true;
		}
		have_grow_light = true;
	}
	static uint8_t l_start_hour=0,l_start_min=0,l_end_hour=0,l_end_min=0xff;
	if( l_start_hour != start_hour || l_start_min != start_min || 
		l_end_hour != end_hour || l_end_min != end_min ){
		l_start_hour = start_hour;
		l_start_min = start_min;
		l_end_hour = end_hour;
		l_end_min = end_min;
		ESP_LOGW(TAG,"light(%d) new night time:%d %d %d %d",have_grow_light, start_hour,start_min,end_hour,end_min);
	}

	u16 start_time = start_hour * 60 + start_min;
    u16 end_time   = end_hour * 60 + end_min;

    u16 cur_time   = p_cur_time->hour * 60 + p_cur_time->min;
	u16 cur_start_duration = ((cur_time + 24*60) - start_time)%(24*60);
	u16 end_start_duration = ((end_time + 24*60) - start_time)%(24*60);

	if( end_start_duration == 0 && have_grow_light == true ){
		if( p_ai_night_setting->config.en ){
			p_ai_night_setting->config.en = 0;
			ai_change_need_sync	= true;	
		}
	}

    if ( ( cur_start_duration > end_start_duration && end_start_duration != 0 ) || p_ai_night_setting->config.en == 0 ){
		ai_night_mode_log(0);
		p_ai_setting_out->is_night_run = false;
		return;
	}

	ai_night_mode_log(1);
	p_ai_setting_out->is_night_run = true;

    /* 再修改对应参数为 night_setting */
    memcpy(&p_ai_setting_out->autoMode, &p_ai_night_setting->autoMode, sizeof(auto_mode_set_t));
    memcpy(&p_ai_setting_out->vpdMode, &p_ai_night_setting->vpdMode, sizeof(vpd_mode_set_t));

    for (u8 i = 0; i < PORT_CNT; i++)
    {
        switch (p_ai_night_setting->port_ctrl[i].type & 0x7f )
        {
            /* ml_port_setting_st 中有定义， ml_port_nightsetting_st 中无定义 */
            case loadType_growLight :
            case loadType_water_pump :
            case loadType_co2_generator : break;

            /* ml_port_setting_st 、 ml_port_nightsetting_st 中都有定义 */
            case loadType_humi :
                p_ai_setting_out->port_ctrl[i].humid.lev       = p_ai_night_setting->port_ctrl[i].humid.lev;
                p_ai_setting_out->port_ctrl[i].humid.humid_lev = p_ai_night_setting->port_ctrl[i].humid.humid_lev;
                break;
            case loadType_fan :
                p_ai_setting_out->port_ctrl[i].fan.lev    = p_ai_night_setting->port_ctrl[i].fan.lev;
                p_ai_setting_out->port_ctrl[i].fan.config = p_ai_night_setting->port_ctrl[i].fan.config;
				p_ai_setting_out->port_ctrl[i].fan.degree = p_ai_night_setting->port_ctrl[i].fan.degree;
                break;
            case loadType_inlinefan :
                p_ai_setting_out->port_ctrl[i].inlinefan.lev    = p_ai_night_setting->port_ctrl[i].inlinefan.lev;
                p_ai_setting_out->port_ctrl[i].inlinefan.config = p_ai_night_setting->port_ctrl[i].inlinefan.config;
                break;

			/* device_else */
            default : p_ai_setting_out->port_ctrl[i].device_else.lev = p_ai_night_setting->port_ctrl[i].device_else.lev; break;
        }
    }
}

void ai_deal_after_rule_run(ai_night_mode_setting_t * p_ai_night_setting, ai_setting_t * p_ai_setting_sys, ai_setting_t * p_ai_setting_run)
{
	p_ai_setting_run->autoMode = p_ai_setting_sys->autoMode;
	p_ai_setting_run->vpdMode = p_ai_setting_sys->vpdMode;
	for (u8 i = 0; i < PORT_CNT; i++)
    {
		switch (p_ai_setting_run->port_ctrl[i].type & 0x7f )
        {
            /* ml_port_setting_st 中有定义， ml_port_nightsetting_st 中无定义 */
            case loadType_growLight :
            case loadType_water_pump :
            case loadType_co2_generator : break;
			default: p_ai_setting_run->port_ctrl[i] = p_ai_setting_sys->port_ctrl[i];	break;
		}
	}
	//	APP 设置的为最新数据
	*p_ai_setting_sys = *p_ai_setting_run;
	check_ai_change_need_updata();
	ai_com_mutex_give();
}

static uint8_t ai_updata_ml_dev_setting( ai_running_data_t* p_ai_runnig, ai_setting_t* ai_setting, dev_type_t* dev_type_list, ml_input_dev_info_t* dev_info_list )
{
	uint8_t n,cnt;

    cnt = 0;
	// get all ai distLoad_type, and find grow light type first
	for (n = 1; n < PORT_CNT; n++)
	{
		dev_info_t* cur_dev_info = &(p_ai_runnig->port_dev_info[n]);
		if ( is_ai_port_bit(ai_setting,n) ){
			if( cur_dev_info->flag.match == false ){
				dev_info_list[n].port_sta = DEV_STA_OFF_LINE;
			}else if( true == m_port_dev_err_run(n) ){
				dev_info_list[n].port_sta = DEV_STA_ERR;
			}else{
				dev_info_list[n].port_sta = DEV_STA_NO_PROBLEM;
			}
			
			dev_info_list[n].origin_of_port		= ai_dev_type_2_ml_dev_type(cur_dev_info->dev_origin);
			dev_info_list[n].ml_type_of_port 	= ai_dev_type_2_ml_dev_type(cur_dev_info->dev_type);	//dev_type_list[n].using_type;
			dev_info_list[n].max_lev = ai_get_port_setting_lev(ai_setting,n,1);
			dev_info_list[n].min_lev = ai_get_port_setting_lev(ai_setting,n,0);
			dev_info_list[n].dev_rule_setting = ai_setting->port_ctrl[n];
			cnt++;
		}else{
			dev_info_list[n].ml_type_of_port = ai_dev_type_2_ml_dev_type(loadType_nomatter);
			dev_info_list[n].origin_of_port = ai_dev_type_2_ml_dev_type(loadType_nomatter);
			dev_info_list[n].port_sta = DEV_STA_OFF_LINE;
		}
	}

	return cnt;
}

const uint8_t env_to_sensor_tab[ENV_CNT]={
	LIGHT,
	TEMP_F,
	HUMID,
	VPD,
	CO2,
	SOIL_HUMID,
	WATER_LEVEL,
};
uint8_t ai_env_to_sensor_id(uint8_t env)
{
	uint8_t sensor_id = 0;
	sensor_id = env_to_sensor_tab[env];
	if( env == ENV_TEMP ){
		sensor_id = (is_temp_unit_f()?TEMP_F:TEMP_C);
	}
	return sensor_id;
}

//sensor realtime data of temp, humid, vpd
static void get_sensor_val_list(s16* sensor_data_list, u8* sen_sta_list, int16_t *env_val_list,int16_t * outside_env_val_list,uint8_t* env_sensor_sta_list)
{
#if 0	//传感器采集数据输出
	static s16 sol_val_last[ENV_CNT]={0};
	const char* snesor[ENV_CNT]={
		"light",
		"temp",
		"humid",
		"vpd",
		"co2",
		"soil",
		"water",
	};
	for( uint8_t env =0; env<ENV_CNT; env++ ){
		uint8_t sensor_id = ai_env_to_sensor_id(env);
		if( sol_val_last[env] != sensor_data_list[sensor_id] ){
			sol_val_last[env] = sensor_data_list[sensor_id];
			ESP_LOGW(TAG,"sensor[%s] sta:%d v:%d", snesor[env],sen_sta_list[sensor_id], sensor_data_list[sensor_id] );
		}
	}
#endif

	for( uint8_t env =0; env<ENV_CNT; env++ ){
		switch( env ){
			case ENV_LIGHT:
				env_val_list[env] 	= 	sensor_data_list[LIGHT]; 	//ai_get_cur_light();
				env_sensor_sta_list[env]	= sen_sta_list[LIGHT];	//g_sensor[LIGHT].dectected;
				break;
			case ENV_TEMP:
				uint8_t temp_unit = (is_temp_unit_f()?TEMP_F:TEMP_C);
				env_val_list[env] 	= sensor_data_list[temp_unit]/10;	//ai_get_cur_temp();
				outside_env_val_list[env] = sensor_data_list[temp_unit+ZONE_TEMP_F-TEMP_F]/10;
				env_sensor_sta_list[env]	= sen_sta_list[temp_unit];	//g_sensor[TEMP_C].dectected;
				break;
			case ENV_HUMID:
				env_val_list[env] 	= sensor_data_list[HUMID]/10;		//ai_get_cur_humid();
				outside_env_val_list[env] = sensor_data_list[ZONE_HUMID]/10;	//ai_get_outside_humid();
				env_sensor_sta_list[env]	= sen_sta_list[HUMID];		//g_sensor[HUMID].dectected;
				break;
			case ENV_VPD:
				env_val_list[env] 	= sensor_data_list[VPD];		//ai_get_cur_vpd();
				outside_env_val_list[env] = sensor_data_list[ZONE_VPD];	//ai_get_outside_vpd();
				env_sensor_sta_list[env]	= sen_sta_list[VPD];	//g_sensor[VPD].dectected;
				// ESP_LOGI(TAG,"vpd:%d zone:%d",env_val_list[env], outside_env_val_list[env] );
				break;
			case ENV_CO2:
				env_val_list[env] 	= sensor_data_list[CO2];		//ai_get_cur_co2();
				env_sensor_sta_list[env]	= sen_sta_list[CO2];	//g_sensor[CO2].dectected;
				break;
			case ENV_SOIL:
				env_val_list[env] 	= sensor_data_list[SOIL_HUMID]; 	//ai_get_cur_soil();
				env_sensor_sta_list[env]	= sen_sta_list[SOIL_HUMID];	//g_sensor[SOIL_HUMID].dectected;
				break;
			case ENV_WATTER:
				env_val_list[env] 	= sensor_data_list[WATER_LEVEL];	//ai_get_cur_water();
				env_sensor_sta_list[env]	= sen_sta_list[WATER_LEVEL];//g_sensor[WATER_LEVEL].dectected;
				break;
		}
	}
}

#ifdef PID_RULE_EN

uint16_t get_ai_env_target_bit(ai_setting_t *ai_setting)
{
	uint16_t ai_env_ctl_bit = 0;
	if( ai_setting->ai_mode_sel_bits.vpd_en )
		ai_env_ctl_bit |=(1<<ENV_VPD);

	if( ai_setting->ai_mode_sel_bits.humid_en )
		ai_env_ctl_bit |=(1<<ENV_HUMID);

	if( ai_setting->ai_mode_sel_bits.temp_en )
		ai_env_ctl_bit |=(1<<ENV_TEMP);

	return ai_env_ctl_bit;
}

void pid_get_env_target_data_param(uint8_t env,ai_setting_t* ai_setting,int16_t* target,int16_t* p_min,int16_t* p_max)
{
	int16_t target_min=0,target_max=0;
	switch( env )
	{
		case ENV_TEMP:
			if (is_temp_unit_f())
			{
				target_min = to_s16x10(ai_setting->autoMode.targetTemp_F_min);
				target_max = to_s16x10(ai_setting->autoMode.targetTemp_F_max);
			}
			else
			{
				target_min = to_s16x10(ai_setting->autoMode.targetTemp_C_min);
				target_max = to_s16x10(ai_setting->autoMode.targetTemp_C_max);
			}
			break;
			
		case ENV_HUMID: 
			target_min = to_s16x10(ai_setting->autoMode.targetHumid_min);
			target_max = to_s16x10(ai_setting->autoMode.targetHumid_max);
			break;

		case ENV_VPD:
			target_min = ai_setting->vpdMode.lowVpd;
			target_max = ai_setting->vpdMode.highVpd;
			break;
			
		default: return;	
	}
	*p_min = target_min;
	*p_max = target_max;
	*target = (target_min+target_max)/2;
}

pid_run_input_st pid_run_input;
void pid_param_get(ai_setting_t *ai_setting, uint8_t* load_type_list, uint8_t* dev_origin_list, int16_t* env_value_list, pid_run_input_st* param)
{
// Before line 1520, add null checks
if (param == NULL) {
    ESP_LOGE("PID", "NULL parameters pointer!");
    return  ; // or handle appropriately
}
 
	memset(param,0x00,sizeof(pid_run_input_st));
	//tim modify 
	param->ml_run_sta = 1;
	#if 0
	if( ai_setting->ai_workmode != AI_WORKMODE_PAUSE){
		param->ml_run_sta = 1;
	}else{
		return;
	}
	#endif

	for(uint8_t i=0; i<PORT_CNT; i++ ){
		param->dev_type[i] 	= load_type_list[i];
		param->is_switch[i] = (dev_origin_list[i]==env_dev_origin_switch);
	}
	param->env_en_bit = get_ai_env_target_bit(ai_setting);
	for(uint8_t env=0; env<ENV_BASE_CNT; env++ ){
		param->env_value_cur[env] = env_value_list[env];
		pid_get_env_target_data_param(env,ai_setting,&(param->env_target[env]),&(param->env_min[env]),&(param->env_max[env]));
	}
}

void pid_rule_output_set_speed(pid_run_output_st out,uint8_t* load_type_list,ml_output_port_t* output_port_list)
{
	for(uint8_t i=0; i<PORT_CNT; i++ ){
		if( load_type_list[i] != loadType_nomatter){
			ml_set_speed( output_port_list, i, out.speed[i]);
		}
	}
}

#endif

	// ENV_LIGHT,
	// ENV_TEMP,
	// ENV_HUMID,
	// ENV_VPD,
	// ENV_CO2,
	// ENV_SOIL,
	// ENV_WATTER,
	// ENV_CNT,
const uint8_t ai_env_tans_ml_env_tab[ENV_CNT]={
	ml_env_light,
	ml_env_temp,
	ml_env_humid,
	ml_env_vpd,
	ml_env_co2,
	ml_env_soil,
	ml_env_water,
};

inline uint8_t get_ml_env_for_ai_env(uint8_t ai_env){
	if( ai_env >= ENV_CNT ){
		return ml_env_max;
	}
	return ai_env_tans_ml_env_tab[ai_env];
}

void get_ai_set_target_data(ai_setting_t *ai_setting,uint8_t ai_env_type,int16_t* p_min,int16_t* p_max,uint8_t* env_en)
{
	int16_t target_min = 0, target_max = 0, mode_en = 0;
	switch( ai_env_type )
	{
		case ENV_TEMP:
			if( ai_setting->ai_mode_sel_bits.temp_en ){
				mode_en = 1;
			}
			if (is_temp_unit_f()){
				target_min = to_s16x10(ai_setting->autoMode.targetTemp_F_min);
				target_max = to_s16x10(ai_setting->autoMode.targetTemp_F_max);
				if( target_min == target_max ){
					target_min -= 20;
					target_max += 20;	//x10
				}
			}else{
				target_min = to_s16x10(ai_setting->autoMode.targetTemp_C_min);
				target_max = to_s16x10(ai_setting->autoMode.targetTemp_C_max);
				if( target_min == target_max ){
					target_min -= 10;
					target_max += 10;	//x10
				}
			}
			break;
		case ENV_HUMID:
			//	humid 3%
			if( ai_setting->ai_mode_sel_bits.humid_en ){
				mode_en = 1;
			}
			target_min = to_s16x10(ai_setting->autoMode.targetHumid_min);
			target_max = to_s16x10(ai_setting->autoMode.targetHumid_max);
			if( target_min == target_max ){
				target_min -= 30;
				target_max += 30;	//x100
			}
			break;
		case ENV_VPD:
			//	*100 0.1 kp
			if( ai_setting->ai_mode_sel_bits.vpd_en ){
				mode_en = 1;
			}
			target_min = ai_setting->vpdMode.lowVpd;
			target_max = ai_setting->vpdMode.highVpd;
			if( target_min == target_max ){
				target_min -= 10;
				target_max += 10;	//
			}
			break;
	}
	*p_min = target_min;
	*p_max = target_max;
	*env_en = mode_en;
}

void get_ai_set_emerge_data(ai_setting_t *ai_setting,uint8_t ai_env_type,int16_t* p_min,int16_t* p_max,uint8_t* env_en)
{
	int16_t set_min = 0, set_max = 0, mode_en = 0;
	switch( ai_env_type )
	{
		case ENV_TEMP:
			if( ai_setting->vpdMode.config.temp_emg ){
				mode_en = 1;
				if (is_temp_unit_f()){
					set_min = ai_setting->vpdMode.temp_emerg_min_f;
					set_max = ai_setting->vpdMode.temp_emerg_max_f;
				}else{
					set_min = ai_setting->vpdMode.temp_emerg_min_c;
					set_max = ai_setting->vpdMode.temp_emerg_max_c;
				}
			}
			break;
		case ENV_HUMID:
			if( ai_setting->vpdMode.config.humid_emg ){
				mode_en = 1;
				set_min = ai_setting->vpdMode.humid_emerg_min;
				set_max = ai_setting->vpdMode.humid_emerg_max;
			}
			break;
	}
	*p_min = set_min*10;
	*p_max = set_max*10;
	*env_en = mode_en;
}

void get_ai_set_pref_data(ai_setting_t *ai_setting,uint8_t ai_env_type,int16_t* p_min,int16_t* p_max,uint8_t* env_en)
{
	int16_t set_min = 0, set_max = 0, mode_en = 0;
	switch( ai_env_type )
	{
		case ENV_TEMP:
			if( ai_setting->vpdMode.config.temp_frist ){
				mode_en = 1;
				if (is_temp_unit_f()){
					set_min = ai_setting->vpdMode.temp_min_f;
					set_max = ai_setting->vpdMode.temp_max_f;
				}else{
					set_min = ai_setting->vpdMode.temp_min_c;
					set_max = ai_setting->vpdMode.temp_max_c;
				}
			}
			break;
		case ENV_HUMID:
			if( ai_setting->vpdMode.config.humid_frist ){
				mode_en = 1;
				set_min = ai_setting->vpdMode.humid_min;
				set_max = ai_setting->vpdMode.humid_max;
			}
			break;
	}
	*p_min = set_min*10;
	*p_max = set_max*10;
	*env_en = mode_en;
}

void ai_updata_sensor_selected(ai_setting_t *ai_setting,uint8_t* sensor_sta_list)
{
	for( uint8_t env =0; env<ENV_CNT; env++ ){
		if( 0 == ai_get_sensor_is_selected( ai_setting, env ) ){
			sensor_sta_list[env] = 0;
		}
	}
}

void ai_updata_ml_sensor_param(ai_setting_t *ai_setting,ml_sensor_config_data_t* sensor_data_list,\
							int16_t *sensor_val_list,int16_t * outside_sensor_val_list,uint8_t* sensor_sta_list )
{
	for( uint8_t ai_env=0; ai_env<ENV_CNT; ai_env++ ){
		uint8_t ml_env = get_ml_env_for_ai_env(ai_env);
		if( ml_env >= ml_env_max ){
			continue;
		}
		ml_sensor_config_data_t* cur_ml_sensor_set = sensor_data_list+ml_env;

		uint8_t sensor_id = ai_env_to_sensor_id(ai_env);
		cur_ml_sensor_set->config.sensor_selct = ai_get_sensor_is_selected( ai_setting, sensor_id );
		cur_ml_sensor_set->inside_value = sensor_val_list[ai_env];
		cur_ml_sensor_set->outside_value = outside_sensor_val_list[ai_env];
		cur_ml_sensor_set->config.sensor_sta = (sensor_sta_list[ai_env] != 0);

		uint8_t en = 0;
		get_ai_set_target_data(ai_setting, ai_env, &cur_ml_sensor_set->set_range.min, \
								&cur_ml_sensor_set->set_range.max, &en );
		cur_ml_sensor_set->config.mode_en = en;

		get_ai_set_emerge_data(ai_setting, ai_env, &cur_ml_sensor_set->emerg_range.min, \
								&cur_ml_sensor_set->emerg_range.max, &en );
		cur_ml_sensor_set->config.emerg_en = en;

		get_ai_set_pref_data(ai_setting, ai_env, &cur_ml_sensor_set->pref_range.min, \
								&cur_ml_sensor_set->pref_range.max, &en );
		cur_ml_sensor_set->config.pref_en = en;
	}
}

void ai_setting2_ml_running_data(ai_setting_t *ai_setting,ml_running_data_t* running_data)
{
	if( ai_setting->is_ai_deleted ){
		running_data->config.delt = 1;
	}else{
		running_data->config.delt = 0;
	}
	if( ai_setting->is_ai_deleted || ai_setting->ai_workmode != AI_WORKMODE_ON ){
		running_data->config.sw = 0;
	}else{	running_data->config.sw = 1; }
}

//
// 
void ai_running_process(ai_setting_t *ai_setting, ai_running_data_t* p_ai_runnig, s16* sensor_data_list, u8* sen_sta_list, dev_type_t* dev_type_list,  \
						ml_running_data_t *running_data,ml_sun_param_t* p_ml_sun_param,uint8_t sec_flag, Time_Typedef* p_sys_time, ml_output_port_t* output_port_list)
{
	// running ml rule
#ifdef PID_RULE_EN
	int16_t sensor_val_list[ENV_CNT];
	uint8_t cur_load_type[PORT_CNT];
	uint8_t port_dev_origin[PORT_CNT];

	pid_run_output_st pid_run_output;
	pid_param_get(ai_setting, cur_load_type, port_dev_origin, sensor_val_list, &pid_run_input );
	pid_run_output = pid_run_rule( &pid_run_input );
	pid_rule_output_set_speed(pid_run_output, cur_load_type, output_port_list );
#else
	int16_t env_val_list[ENV_CNT];
	int16_t outside_env_val_list[ENV_CNT];
	uint8_t env_sensor_sta_list[ENV_CNT];
	ml_input_dev_info_t ml_iput_info[PORT_CNT];
	ml_sensor_config_data_t sensor_config_data[ml_env_max];

	memset( ml_iput_info, 0x00, sizeof(ml_iput_info) );
	memset( sensor_config_data, 0x00, sizeof(sensor_config_data) );

	ai_detect_change_reset(p_ai_runnig,ai_setting,running_data);
	running_data->p_ml_sun_param = p_ml_sun_param;
	// input loadtype of ports
	ai_updata_ml_dev_setting(p_ai_runnig, ai_setting, dev_type_list, ml_iput_info);

	// input realtime data of relevant sensors 
	get_sensor_val_list( sensor_data_list, sen_sta_list, env_val_list, outside_env_val_list, env_sensor_sta_list);
	ai_updata_sensor_selected(ai_setting, env_sensor_sta_list);
	ai_updata_ml_sensor_param(ai_setting, sensor_config_data, env_val_list, outside_env_val_list, env_sensor_sta_list);

	ai_setting2_ml_running_data(ai_setting, running_data);
	ml_rule( ml_iput_info, sensor_config_data, sec_flag,\
			 p_sys_time, running_data, output_port_list );
#endif
	// ai_run_info("\n");
}


void aci_ai_entry( ai_setting_t* cur_ai_setting, Time_Typedef sys_time,dev_type_t* dev_type_list, s16* sensor_data_list, 
					u8* sen_sta_list, uint8_t* mode_list, rule_speed_t *dev_speeds, rule_port_set_t *dev_angles )
{
	extern u8 is_outlet_sw_on(void);

	static uint8_t last_sec = 0xff;
	ml_output_port_t ml_output_port_list[PORT_CNT]={0};

	uint8_t sec_flag = 0;

	ai_refresh_port_err_sta(dev_type_list, ai_port_err);
	memset(ml_output_port_list,0x00,sizeof(ml_output_port_list));
	if( last_sec != sys_time.sec ){
		sec_flag = 1;
		last_sec = sys_time.sec;
	}
	if( false == is_sensor_poweron_det_ok() ){
		ai_online_port_enter_off(cur_ai_setting, ml_output_port_list);
		goto _lev_sync_return;
	}
#if (_TYPE_of(VER_HARDWARE) == _TYPE_(_OUTLET))
	static uint8_t outlet_sys_sw = 0xff;
	uint8_t outlet_sw_temp = is_outlet_sw_on();
	if( outlet_sys_sw != outlet_sw_temp ){
		outlet_sys_sw = outlet_sw_temp;
		ai_change_need_sync	= true;
	}
    if (!outlet_sys_sw)
    {
        return;
    }
#endif
#if	(_TYPE_of(VER_HARDWARE) == _TYPE_(_CTRLER)|| (_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX)))
	extern uint16_t port_get_wait_poweron();
	if( 0 != port_get_wait_poweron() )	//运行AI 等待端口检测完成
		return;
#endif
	ai_updata_setting_operate();

    // dev type change deal
	ai_port_dev_manage( cur_ai_setting, dev_type_list, sec_flag, mode_list, sen_sta_list, &g_ai_running_data );
	//	port dev lost effact run mode
	//update_port_workmode( cur_ai_setting, dev_type_list );
	// dev online offline deal
	update_port_select(cur_ai_setting->ai_port_sel_bits, cur_ai_setting->ai_workmode, ml_output_port_list);
	//	sensor lost deal
	ai_sensor_det_deal( cur_ai_setting, &g_ai_running_data, sec_flag );

	// tent work rule
	tent_work_process( cur_ai_setting, &g_ai_running_data, dev_type_list, sec_flag, ml_output_port_list );  

	ai_restore_deal(cur_ai_setting);

	easy_mode_day_run(&sys_time);

	ai_mode_change_manage(cur_ai_setting,ml_output_port_list);

	// ESP_LOGI(TAG, "pause reson=%d", g_ai_setting.pause_reason);
	ai_clean_pause_rason(cur_ai_setting);

	/* 比对生成日志 */
	uint8_t distLoad_temp[PORT_CNT];
	ml_log_get_loadtype( cur_ai_setting, distLoad_temp, dev_type_list );
	compare_generate_mL_log( sys_time, distLoad_temp );
#if(_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))
	s16 grade_sensor_list[ENV_CNT];
	s16 outside_sensor_val_list[ENV_CNT];
	get_sensor_val_list(grade_sensor_list,outside_sensor_val_list);
	ml_grade_run(cur_ai_setting,&sys_time,grade_sensor_list);
	//	评级变化 上传
	if( cur_ai_setting->ml_grade !=  get_ml_grade() ){
		cur_ai_setting->ml_grade = get_ml_grade();
		ai_change_need_sync	= true;
	}
#endif
	ai_sun_dynamic_sync_before( cur_ai_setting, &ml_sun_param );
	int16_t zone_temp_f = is_temp_unit_f()?sensor_data_list[ZONE_TEMP_F]:ml_temp_tran_unit_f(sensor_data_list[ZONE_TEMP_C],false);
	// if(sec_flag) ESP_LOGW(TAG,"sun zone temp:%d sta:%d",zone_temp_f,sen_sta_list[ZONE_TEMP_F] );
	ai_sun_dynamic_rule_run( cur_ai_setting->is_ai_deleted, &sys_time, &ml_sun_param,  zone_temp_f, sen_sta_list[ZONE_TEMP_F] != 0 );
	ai_sun_dynamic_sync_setting( cur_ai_setting, &ml_sun_param);

	// return when ai off
	// ml logic start 
    // ai running
	// g_ml_running_data.p_ml_sun_param = &ml_sun_param;

	ai_running_process( cur_ai_setting, &g_ai_running_data, sensor_data_list, sen_sta_list,  dev_type_list, &g_ml_running_data, &ml_sun_param, sec_flag, &sys_time, ml_output_port_list );
	ai_app_45_flash_target_sta(&g_ai_ml_45_info,&g_ml_running_data);
	ai_insight_rule(&sys_time, cur_ai_setting, &ai_insight_run_data, dev_type_list, sensor_data_list );

_lev_sync_return:
	ai_sync_angle( cur_ai_setting, dev_angles );
	ai_sync_speed( ml_output_port_list, dev_speeds, dev_type_list );
}

