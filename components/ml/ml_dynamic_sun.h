#ifndef _ML_DYNAMIC_SUN_H_
#define _ML_DYNAMIC_SUN_H_

#include "ai_out.h" 

extern void ai_sun_dynamic_rule_run(uint8_t ml_deleted,Time_Typedef *sys_time, ml_sun_param_t* p_ml_sun_param , int16_t outside_temp_f, bool sensor_is_ok);
extern uint16_t get_utc_min_in_24hours_compare_rtc0(uint32_t utc);
extern Time_Typedef ml_sun_get_tim(Time_Typedef *sys_time);

#endif
