#include <stdio.h>
#include <string.h>

#include "includes.h"

#include "esp_log.h"
#include "math.h"

#include "ai.h"
#include "ai_out.h" 
#include "ml_dynamic_sun.h"

#define TAG "dynamic_sun"

// #define AI_DYNAMIC_SUN_DEBUG		//调试总开关
// #define AI_DYNAMIC_SUN_LOG_DEBUG	//动态光调试信息

#ifdef AI_DYNAMIC_SUN_DEBUG
#define AI_DYNAMIC_SUN_FAST_TEST	//快速测试版本
#endif

#define sun_err(format, ...)		ESP_LOGE(TAG, format, ##__VA_ARGS__)
#define sun_warn(format, ...)		ESP_LOGW(TAG, format, ##__VA_ARGS__)
#define sun_info(format, ...)		ESP_LOGI(TAG, format, ##__VA_ARGS__)

#ifdef AI_DYNAMIC_SUN_LOG_DEBUG
#define	sun_debug(format, ...)		ESP_LOGI(TAG, format, ##__VA_ARGS__)
#else
#define	sun_debug(format, ...)
#endif

EXT_RAM_BSS_ATTR ml_sun_param_t ml_sun_param = {0};

#define MAX_TEMP_F		INT16_MAX	//32767

#ifdef AI_DYNAMIC_SUN_FAST_TEST
	Time_Typedef ml_tim;
#endif

uint16_t ml_get_tim_24h_minute(const Time_Typedef *sys_time)
{
	return (sys_time->hour*60+sys_time->min);
}

uint32_t ml_get_utc_tim()
{
#ifdef AI_DYNAMIC_SUN_FAST_TEST
	return rtc_to_real_utc(ml_tim);
#else
	return get_utc_time();
#endif
}

Time_Typedef *ml_get_cur_time()
{
#ifdef AI_DYNAMIC_SUN_FAST_TEST
	return &ml_tim;
#else
	return get_cur_time();
#endif
}

Time_Typedef ml_sun_get_tim(Time_Typedef *sys_time)
{
#ifdef AI_DYNAMIC_SUN_FAST_TEST
	static uint8_t last_sec = 0xff;
	static uint16_t cnt = 0;
	if( last_sec == 0xff ){
		cnt = (sys_time->hour*60+sys_time->min);
	}
	if( last_sec != sys_time->sec ){
		cnt ++;
		last_sec = sys_time->sec;
		cnt = cnt%ONE_DYA_MINUTE;
	}
	ml_tim = *sys_time;

	ml_tim.hour = cnt/60;
	ml_tim.min  = cnt%60;
	ml_tim.sec	 = 30;

	return ml_tim;
#else
	return *sys_time;
#endif
}

/// @brief 将当前UTC时间，转化为rtc对应24小时内的分钟数
/// @param utc 
/// @return utc 对应 RTC 距0点 分钟数
uint16_t get_utc_min_in_24hours_compare_rtc0(uint32_t utc)
{
	Time_Typedef* cur_tim = ml_get_cur_time();
	uint32_t cur_utc = ml_get_utc_tim();

	cur_utc = ( cur_utc + 24*60*60- ((cur_tim->hour*60 + cur_tim->min)*60 + cur_tim->sec) )%(24*60*60);	//rtc - 0点
	uint32_t sec_24h = (utc + 24*60*60 - cur_utc )%(24*60*60);

	return (sec_24h/60);
}

/// @brief 以utc时间等于0 循环计算24小时
/// @param utc 当前UTC使时间
/// @return 相对 utc 0 点的 24小时内的分钟数 --- 统一计算标准,避免时区问题
uint16_t get_utc_min_in_24hours_compare_utc0(uint32_t utc)
{
	return ((utc/60)%(24*60));
}

//new - ref
uint16_t get_minute_difference_in_24hour(uint16_t ref_point,uint16_t new_point)
{
	uint16_t differ_cur = ((new_point+ONE_DYA_MINUTE) - ref_point)%ONE_DYA_MINUTE;
	return differ_cur;
}

void ml_dynamic_sun_clean_temp_tab( ml_sun_param_t* p_ml_sun_param )
{
	for(uint16_t i=0; i< ONE_DYA_MINUTE; i++ ){
		p_ml_sun_param->temp_f_tab[i] = MAX_TEMP_F;
	}
}

/// @brief 以utc-0点 为第一个点 填充数据
/// @param p_ml_sun_param 
/// @param min_utc0 
/// @param temp_f 
void ai_sun_input_temp(ml_sun_param_t* p_ml_sun_param, uint16_t min_utc0, int16_t temp_f)
{
	if( min_utc0 < ONE_DYA_MINUTE ){
		if( temp_f != MAX_TEMP_F && p_ml_sun_param->effective_data_cnt < ONE_DYA_MINUTE ){
			p_ml_sun_param->effective_data_cnt++;
		}
		p_ml_sun_param->temp_f_tab[min_utc0] = temp_f;
	}else{
		sun_info("data over size");
	}
}


uint16_t ai_sun_dynamic_interval_minutes_max(Time_Typedef *sys_time, ml_sun_param_t* p_ml_sun_param)
{
	static uint8_t last_min = 0xff;
	uint16_t max_minutes = 0;
	uint16_t gap_minutes = 0;
	if( last_min == sys_time->min || 30 != sys_time->sec){
		return 0;
	}
	last_min = sys_time->min;

	uint16_t cur_24h_min_utc = get_utc_min_in_24hours_compare_utc0( ml_get_utc_tim() );
	uint16_t set_24h_min_utc = get_utc_min_in_24hours_compare_utc0(p_ml_sun_param->sunStartCollectingUtc);
	for( uint16_t min=set_24h_min_utc; min != cur_24h_min_utc; min = (min+1)%ONE_DYA_MINUTE )
	{
		if( p_ml_sun_param->temp_f_tab[min] != MAX_TEMP_F ){
			gap_minutes = 0;
			continue;
		}
		gap_minutes++;
		if( gap_minutes > max_minutes ){
			max_minutes = gap_minutes;
		}
	}
#ifdef AI_DYNAMIC_SUN_DEBUG
	static uint16_t last_max_minut = 0;
	if( max_minutes != last_max_minut ){
		last_max_minut = max_minutes;
		sun_debug("lose data max:%d", max_minutes );
	}
#endif
	return max_minutes;
}

float ai_sun_get_tab_avrage(int16_t* tab, uint16_t size)
{
	float average = 0.0;
	uint16_t useful_cnt = 0;
	for( uint16_t i=0; i<size; i++ ){
		if( MAX_TEMP_F == tab[i]){
			continue;
		}
		useful_cnt++;
		average+= tab[i];
	}
	if( useful_cnt != 0 ){
		average =average/(float)useful_cnt;
	}
	//	保留一位小数 精确到 0.1F
	average = (uint16_t)((uint16_t)average+5)/10;
	average *= 10;
	return average;
}

/// @brief 从start_point往后找最靠近的点
/// @param tab 输入比较数组
/// @param start_point 开始往后查询的点
/// @param size 总共比较点数
/// @param compare_data 比较的数据
/// @param is_foward 向前查找/向后查找
/// @return 最靠近的点
uint16_t ai_sun_get_closest_to_data(int16_t* tab, uint16_t start_point, uint16_t size, float compare_data, bool is_foward)
{
	float temp_diff_min = MAX_TEMP_F;	//差值计算
	uint16_t temp_cnt_min = 0;
	
	for( uint16_t i=0; i<size; i++ ){
		uint16_t num = 0;
		if( is_foward ){
			num = start_point + i;
		}else{
			num = start_point + size - i;
		}
		if( num >= size ){
			num -= size;
		}
		if( MAX_TEMP_F == tab[num]){
			continue;
		}
		float temp = (tab[num] > compare_data)?(tab[num]-compare_data):(compare_data-tab[num]);
		if( temp < temp_diff_min ){
			temp_diff_min = temp;
			temp_cnt_min = num;
		}
	}
	sun_info("sun[%d->%d]:%f minimum to %f",get_utc_min_in_24hours_compare_rtc0(start_point*60),get_utc_min_in_24hours_compare_rtc0(temp_cnt_min*60),temp_diff_min,compare_data);
	return temp_cnt_min;
}

int16_t get_temp_rtc_min(ml_sun_param_t* p_ml_sun_param, uint16_t rtc_min)
{
	uint16_t min_utc = get_utc_min_in_24hours_compare_rtc0(0);	//utc 为0 对应的 rtc min
	min_utc = (rtc_min + ONE_DYA_MINUTE - min_utc )%ONE_DYA_MINUTE;
	return p_ml_sun_param->temp_f_tab[min_utc];
}

#ifdef AI_DYNAMIC_SUN_DEBUG
/// @brief 输出动态光 运行数据
/// 开始收集时间 - 开始时间 - 运行数据 - 
/// @param p_ml_sun_param 
void ai_sun_dynamic_output_run_data(ml_sun_param_t* p_ml_sun_param)
{
	float average = ai_sun_get_tab_avrage( p_ml_sun_param->temp_f_tab, ONE_DYA_MINUTE );
	uint16_t min_utc = get_utc_min_in_24hours_compare_utc0(p_ml_sun_param->start_utc_sec);
	uint16_t new_min_temp_minute = ai_sun_get_closest_to_data( p_ml_sun_param->temp_f_tab, min_utc, ONE_DYA_MINUTE, average, 1);
	Time_Typedef sun_tim = real_utc_to_rtc(p_ml_sun_param->sunStartCollectingUtc);
	sun_debug("avg[%4d] minute[%2d-%2d] flag[%d] CollectingUtc[%2d-%2d-%2d-%2d-%2d-%2d] start[%2d-%2d]",(uint16_t)average, \
			get_utc_min_in_24hours_compare_rtc0(new_min_temp_minute*60)/60, get_utc_min_in_24hours_compare_rtc0(new_min_temp_minute*60)%60, \
			p_ml_sun_param->beyoned_24_hour, sun_tim.year, sun_tim.month, sun_tim.date, sun_tim.hour,sun_tim.min,sun_tim.sec, \
			p_ml_sun_param->start_minute/60, p_ml_sun_param->start_minute%60 );

	uint16_t i=0;
	min_utc = get_utc_min_in_24hours_compare_rtc0(0);	//utc 为0 对应的 rtc min
	while( i<ONE_DYA_MINUTE ){
		uint16_t min = (i + ONE_DYA_MINUTE - min_utc )%ONE_DYA_MINUTE;
		sun_debug("data[%4d-%4d]= %5d %5d %5d %5d %5d %5d",i,i+6,p_ml_sun_param->temp_f_tab[min],p_ml_sun_param->temp_f_tab[(min+1)%ONE_DYA_MINUTE],
			p_ml_sun_param->temp_f_tab[(min+2)%ONE_DYA_MINUTE],p_ml_sun_param->temp_f_tab[(min+3)%ONE_DYA_MINUTE],
			p_ml_sun_param->temp_f_tab[(min+4)%ONE_DYA_MINUTE],p_ml_sun_param->temp_f_tab[(min+5)%ONE_DYA_MINUTE] );
		i += 6;
	}
}
#endif

//	end  -  start in 12hour
uint16_t get_tim_minut_distance(uint16_t start,uint16_t end)
{
	uint16_t diff = ( end + ONE_DYA_MINUTE - start )%ONE_DYA_MINUTE;
	if( diff > ONE_DYA_MINUTE/2 ){
		diff = ONE_DYA_MINUTE - diff;
	}
	return diff;
}

//	固定点更新处理
bool ai_sun_param_caculate(Time_Typedef *sys_time,ml_sun_param_t* p_ml_sun_param,int16_t temp_f)
{
	uint8_t ret = false;
	if(sys_time->sec != 30 ){	//	保证数据的获取点 和 历史数据同步, 计算参数统一
		return false;
	}
	if( sys_time->min == p_ml_sun_param->last_min ){
		return false;
	}
	p_ml_sun_param->last_min = sys_time->min;

	uint16_t cur_24h_min 	= ml_get_tim_24h_minute(sys_time);	//get_utc_min_in_24hours_compare_rtc0( ml_get_utc_tim() );
	uint16_t start_minute_utc = get_utc_min_in_24hours_compare_utc0(p_ml_sun_param->start_utc_sec);

	if( ( cur_24h_min == p_ml_sun_param->setting_minute && (p_ml_sun_param->effective_data_cnt > 0) ) ){	//计算时间点
#ifdef AI_DYNAMIC_SUN_DEBUG
		ai_sun_dynamic_output_run_data(p_ml_sun_param);
#endif
		float average = ai_sun_get_tab_avrage( p_ml_sun_param->temp_f_tab, ONE_DYA_MINUTE );
		uint16_t new_min_temp_minute_utc = ai_sun_get_closest_to_data( p_ml_sun_param->temp_f_tab, start_minute_utc, ONE_DYA_MINUTE, average, 1);

		uint16_t new_min_temp_minute_utc_back = ai_sun_get_closest_to_data( p_ml_sun_param->temp_f_tab, start_minute_utc, ONE_DYA_MINUTE, average, 0);
		if( get_tim_minut_distance(start_minute_utc, new_min_temp_minute_utc_back) < get_tim_minut_distance(start_minute_utc, new_min_temp_minute_utc) ){
			new_min_temp_minute_utc = new_min_temp_minute_utc_back;
		}

		sun_info("old sun start:%d-%d", p_ml_sun_param->start_minute/60, p_ml_sun_param->start_minute%60);
		if( p_ml_sun_param->beyoned_24_hour == 0 ){
			p_ml_sun_param->start_utc_sec = new_min_temp_minute_utc*60;
		}else{
			uint16_t differ_cur = ( (new_min_temp_minute_utc+ONE_DYA_MINUTE) - start_minute_utc)%ONE_DYA_MINUTE;
			int8_t add_minute = 0;
			
			if( differ_cur == 0 ){
				add_minute = 0;
			}else if( differ_cur < ONE_DYA_MINUTE/2 ){	//	新的时间点 在当前后面
				add_minute = 1;
			}else{
				add_minute = -1;
			}
			p_ml_sun_param->start_utc_sec = ( start_minute_utc + ONE_DYA_MINUTE + add_minute )%(ONE_DYA_MINUTE)*60;
		}

		p_ml_sun_param->effective_data_cnt = 0;
		p_ml_sun_param->start_minute = get_utc_min_in_24hours_compare_rtc0(p_ml_sun_param->start_utc_sec);;
		sun_info("sun average:%f,point:%d", average, get_utc_min_in_24hours_compare_rtc0(new_min_temp_minute_utc*60));
		sun_info("new sun start:%d-%d", p_ml_sun_param->start_minute/60, p_ml_sun_param->start_minute%60);
		ret = true;
	}

	uint16_t cur_24h_min_utc = get_utc_min_in_24hours_compare_utc0( ml_get_utc_tim() );

	ai_sun_input_temp(p_ml_sun_param, cur_24h_min_utc, temp_f);
	sun_debug("set[%2d-%2d],start[%2d-%2d],tim[%2d-%2d],caculate[%4d],f[%5d],flag[%d]",\
				p_ml_sun_param->setting_minute/60,	p_ml_sun_param->setting_minute%60,\
				p_ml_sun_param->start_minute/60,	p_ml_sun_param->start_minute%60,\
				cur_24h_min/60,	cur_24h_min%60, 	p_ml_sun_param->effective_data_cnt,temp_f,p_ml_sun_param->beyoned_24_hour);

	return ret;
}

inline void ai_sun_dynamic_reset(ml_sun_param_t* p_ml_sun_param)
{
	p_ml_sun_param->on_off_sw = 0;
}

bool ai_sun_dynamic_reset_deal( ml_sun_param_t* p_ml_sun_param,Time_Typedef *sys_time,int16_t outside_temp_f,bool sw)
{
	if( p_ml_sun_param->on_off_sw == sw )
		return 0;
	p_ml_sun_param->on_off_sw = sw;

	uint16_t cur_24h_min = ml_get_tim_24h_minute( sys_time );
	p_ml_sun_param->refresh_24hour_sta = 1;

	p_ml_sun_param->beyoned_24_hour = 0;
	p_ml_sun_param->setting_minute = cur_24h_min;

	p_ml_sun_param->refresh_start_time = 0;
	p_ml_sun_param->effective_data_cnt = 0;
	p_ml_sun_param->start_utc_sec = rtc_to_real_utc(*sys_time);
	p_ml_sun_param->sunStartCollectingUtc = rtc_to_real_utc(*sys_time);
	p_ml_sun_param->start_minute = get_utc_min_in_24hours_compare_rtc0(p_ml_sun_param->start_utc_sec);

	p_ml_sun_param->last_min = 0xff;
	ml_dynamic_sun_clean_temp_tab( p_ml_sun_param );

	sun_info("init run data:%d",p_ml_sun_param->on_off_sw);
	return 1;
}

void ai_sun_dynamic_rule_run(uint8_t ml_deleted,Time_Typedef *sys_time, ml_sun_param_t* p_ml_sun_param , int16_t outside_temp_f, bool sensor_is_ok)
{
	bool rule_sw = 1;
	//	重置规则
	Time_Typedef ml_sun_tim = ml_sun_get_tim(sys_time);

	if( ml_deleted ){
		rule_sw = 0;
	}
	// FOR debug
#ifdef AI_DYNAMIC_SUN_DEBUG
	if( g_sys_setting.key_sound_en == false ){
		sensor_is_ok = false;
	}
	static uint8_t last_port = 0;
	if( g_sys_setting.backlight_rank != last_port ){
		last_port = g_sys_setting.backlight_rank;
		if( 3 == g_sys_setting.backlight_rank ){
			ai_sun_dynamic_output_run_data(p_ml_sun_param);
		}
	}
#endif

	if( sensor_is_ok == false ){
		outside_temp_f = MAX_TEMP_F;
	}
#define SUN_DYNAMIC_LOSE_DATA_MAX	(2*60)
	if( p_ml_sun_param->on_off_sw == true && p_ml_sun_param->beyoned_24_hour == false \
		&& ai_sun_dynamic_interval_minutes_max( &ml_sun_tim, p_ml_sun_param ) >= SUN_DYNAMIC_LOSE_DATA_MAX ){	//超过2hour
		rule_sw = 0;
		sun_warn("lose temp data >= %dmin",SUN_DYNAMIC_LOSE_DATA_MAX);
	}
	if( p_ml_sun_param->on_off_sw == true && p_ml_sun_param->beyoned_24_hour == true ){
		if( p_ml_sun_param->start_minute >= 24*60 ){
			rule_sw = 0;
			sun_err("sun data err!!! (%d)", p_ml_sun_param->start_minute );
		}
	}

	ai_sun_dynamic_reset_deal( p_ml_sun_param, &ml_sun_tim, outside_temp_f, rule_sw );
	if( p_ml_sun_param->on_off_sw == 0 ){
		return;
	}
	// 随着 utc 动态改变 刷新
	p_ml_sun_param->start_minute 	= get_utc_min_in_24hours_compare_rtc0(p_ml_sun_param->start_utc_sec);
	p_ml_sun_param->setting_minute 	= get_utc_min_in_24hours_compare_rtc0(p_ml_sun_param->sunStartCollectingUtc);

	//	计算规则
	if( true == ai_sun_param_caculate( &ml_sun_tim, p_ml_sun_param, outside_temp_f ) ){
		p_ml_sun_param->beyoned_24_hour = 1;
		p_ml_sun_param->refresh_start_time = 1;
		sun_debug("get caculate data");
	}
	//	更新时间
}


////////////////////////////////////////////// 对外接口 ///////////////////////////////////////////////////
/// @brief 开机清除
/// @param p_ml_sun_param 
void ai_sun_dynamic_poweron_data_init()
{
	ml_dynamic_sun_clean_temp_tab(&ml_sun_param);
}

extern int16_t ml_temp_tran_unit_f( uint16_t data, uint8_t is_unit_f );
void ai_sun_dynamic_history_data_input(uint32_t utc,int16_t temp,uint8_t is_unit_f)
{
	// 填充近24小时内数据
	if( ml_sun_param.on_off_sw == 0 || utc < ml_sun_param.sunStartCollectingUtc || utc + 24*60*60 < get_utc_time() ){
		return;
	}
	int16_t temp_f = ml_temp_tran_unit_f( temp, is_unit_f );
	uint16_t temp_min = get_utc_min_in_24hours_compare_utc0(utc);
#if 0
	uint16_t log_minute = get_utc_min_in_24hours_compare_rtc0(temp_min*60);
	ESP_LOGI(TAG, "history:time[%d-%d]-temp:%d ", log_minute/60, log_minute%60, temp_f);
#endif
	ai_sun_input_temp(&ml_sun_param, temp_min, temp_f);
}


void save_ai_sun_dynamic_sava_param(ml_sun_param_t* p_ml_sun_param,u8* buf,u16* position)
{
	u16 i = *position;
	write_atom_unit(&(p_ml_sun_param->setting_minute), buf, &i) ;
	write_atom_unit(&(p_ml_sun_param->setting_period), buf, &i) ;
	write_atom_unit(&(p_ml_sun_param->beyoned_24_hour), buf, &i) ;
	write_atom_unit(&(p_ml_sun_param->on_off_sw), buf, &i) ;
	write_atom_unit(&(p_ml_sun_param->start_minute), buf, &i) ;
	write_atom_unit(&(p_ml_sun_param->sunStartCollectingUtc), buf, &i);
	write_atom_unit(&(p_ml_sun_param->start_utc_sec), buf, &i) ;
	*position =  i;
}

void read_ai_sun_dynamic_sava_param(ml_sun_param_t* p_ml_sun_param,u8* buf,u16* position)
{
	u16 i = *position;
	read_atom_unit(&(p_ml_sun_param->setting_minute), buf, &i) ;
	read_atom_unit(&(p_ml_sun_param->setting_period), buf, &i) ;
	read_atom_unit(&(p_ml_sun_param->beyoned_24_hour), buf, &i) ;
	read_atom_unit(&(p_ml_sun_param->on_off_sw), buf, &i) ;
	read_atom_unit(&(p_ml_sun_param->start_minute), buf, &i) ;
	read_atom_unit(&(p_ml_sun_param->sunStartCollectingUtc), buf, &i);
	read_atom_unit(&(p_ml_sun_param->start_utc_sec), buf, &i);
	*position =  i;
}
