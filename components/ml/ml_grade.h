#ifndef _ML_GRADE_H_
#define _ML_GRADE_H_
#include "ai_out.h"

typedef enum{
	ML_GRADE_CALCULATING = 0,
	ML_GRADE_IDEAL,
	ML_GRADE_GOOD,
	ML_GRADE_FAIR,
	ML_GRADE_POOR,
	ML_GRADE_MAX,
}ml_grade_e;

typedef struct{
    uint32_t cnt;
    uint32_t utc;
    uint8_t grade;
    double 	sum;
}ml_grade_t;

extern ml_grade_t ml_grade;
extern ml_grade_e get_ml_grade(void);	//在GrowBox获取评级，其它设备返回 0
extern void ml_grade_run(ai_setting_t* ai_setting,Time_Typedef* time,int16_t* sensor_list);

#endif
