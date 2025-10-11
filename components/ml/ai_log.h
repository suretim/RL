#ifndef _AI_LOG_H_
#define _AI_LOG_H_

extern void compare_generate_mL_log(Time_Typedef sys_time, uint8_t* load_type_list );
extern bool isExpired(ai_setting_t *newSetting,Time_Typedef* p_sys_time);

#endif
