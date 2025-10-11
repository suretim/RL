#include <stdio.h>
#include <string.h>

#include "includes.h"

#include "rom/rtc.h"
#include "esp_log.h"
#include "math.h"

#include "ai.h"
#include "ai_out.h" 
#include "ml_grade.h"

#define TAG "ml grade"

// #define AI_ML_GRADE_DEBUG

#ifdef AI_ML_GRADE_DEBUG
#define	grade_info(format, ...)		ESP_LOGI("grade ", format, ##__VA_ARGS__)
#else
#define	grade_info(format, ...)
#endif
//-------------------- ML Grade Calculation -------------------------
#if(_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))

ml_grade_t ml_grade={ 0 };

#define ML_CALCULATING_ONE_DAY_MIN	(24*60*60)

const char* grade_char[]={
	"CALCULATING",
	"IDEAL",
	"GOOD",
	"FAIR",
	"POOR",
	"MAX",
};

float ml_grade_generate(float cur,float min,float max)
{
	if( cur <= max && cur >= min ){
		return 100;
	}
	float delt = (cur<min)?(min-cur):(cur-max);
	if( delt > 100 ){
		delt = 100;
	}
	return (100-delt);
}

bool ml_grade_update_time(Time_Typedef* time)
{
	if( (time->hour % 6) == 0 && time->min == 0 ){
		return 1;
	}
	return 0;
}

void ml_grade_clean()
{
	ml_grade.grade = ML_GRADE_CALCULATING;
	ml_grade.sum = 0;
	ml_grade.cnt = 0;
}

ml_grade_e get_ml_grade()
{
	return ml_grade.grade;
}



//	每分钟计算一次
void ml_grade_calculation(ai_setting_t* ai_setting,Time_Typedef* time,int16_t* sensor_list)
{
	//	未满24小时
	uint8_t cnt = 0;
	float grade = 0;
	uint8_t temp_min,temp_max;
	if( time->sec != 0 ){
		return;
	}
	
	grade_info("cur utc:%ld,set utc:%ld,",rtc_to_real_utc(*time),ai_setting->start_time.utc);
	if( ai_setting->ai_mode_sel_bits.vpd_en ){
		grade += ml_grade_generate( sensor_list[ENV_VPD]/100.0, ai_setting->vpdMode.lowVpd/100.0, ai_setting->vpdMode.highVpd/100.0 );
		cnt++;
	}else{
		if( ai_setting->ai_mode_sel_bits.humid_en ){
			grade += ml_grade_generate( sensor_list[ENV_HUMID]/10.0, ai_setting->autoMode.targetHumid_min, ai_setting->autoMode.targetHumid_max );
			cnt++;
		}
		if( ai_setting->ai_mode_sel_bits.temp_en ){
			if( is_temp_unit_f() ){
				temp_min = ai_setting->autoMode.targetTemp_F_min;
				temp_max = ai_setting->autoMode.targetTemp_F_max;
			}else{
				temp_min = ai_setting->autoMode.targetTemp_C_min;
				temp_max = ai_setting->autoMode.targetTemp_C_max;
			}
			grade += ml_grade_generate( sensor_list[ENV_TEMP]/10.0, temp_min, temp_max );
			cnt++;
		}
	}
	grade = grade/cnt;
	ml_grade.sum += grade;
	ml_grade.cnt++;
	grade_info("grade:%4f,cnt:%ld,sum:%4f,",grade,ml_grade.cnt,ml_grade.sum);
	if( (rtc_to_real_utc(*time)-ml_grade.utc) < ML_CALCULATING_ONE_DAY_MIN ){
		return;
	}
	
	if( ml_grade_update_time(time) /*|| ml_grade.cnt >= 10*/ ){
		grade = ml_grade.sum/ml_grade.cnt;
		if( ai_setting->ai_mode_sel_bits.vpd_en ){
			grade = (100-grade)/0.2;
		}else{
			grade = (100-grade)/5;
		}
		if( grade >= ML_GRADE_POOR ){
			grade = ML_GRADE_POOR;
		}else{
			grade += 1;
		}
		ml_grade.sum = 0;
		ml_grade.cnt = 0;
		ml_grade.grade = grade;
		grade_info("grade lev:%s",grade_char[ml_grade.grade]);
	}
}

void ml_grade_run(ai_setting_t* ai_setting,Time_Typedef* time,s16* sensor_list)
{
	if( ai_setting->is_ai_deleted ){
		if( ml_grade.utc != 0 ){
			ml_grade.utc = 0;
			ml_grade_clean();
		}
		return;
	}
	if( ai_setting->start_time.utc != ml_grade.utc ){
		ml_grade.utc = ai_setting->start_time.utc;
		ml_grade_clean();
	}
    if( ai_setting->ai_workmode != AI_WORKMODE_ON ){
        return;
    }
	ml_grade_calculation( ai_setting, time, sensor_list );
}
#else
ml_grade_e get_ml_grade()
{
	return 0;
}
#endif
//-------------------- ML Grade Calculation -------------------------
