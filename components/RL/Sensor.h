#ifndef __Sensor_H
#define __Sensor_H

#include "define.h"

#pragma pack(1)  // 

#define SensorhumiArraySize 20
#define		ERR_MAX		3  // 最大的温湿度读取错误次数 ，0.5s一次


typedef enum
{
	TEMP_F,           //from sh40t temp+humid sensor
    TEMP_C,           //from sh40t temp+humid sensor
	HUMID,            //from sh40t temp+humid sensor
	VPD,              //from sh40t temp+humid sensor
	ZONE_TEMP_F,      //from shtc3 temp+humid zone sensor
	ZONE_TEMP_C,      //from shtc3 temp+humid zone sensor
	ZONE_HUMID,       //from shtc3 temp+humid zone sensor
	ZONE_VPD,         //from shtc3 temp+humid zone sensor
	LEAFTEMP_F,       //from leaf  sensor
	LEAFTEMP_C,       //from leaf  sensor
	SOIL_HUMID,       //from soil  sensor
	CO2,              //from co2   sensor
	LIGHT,            //from light sensor
	PH,               //from ph	sensor
	EC_us,            //from ph	sensor
	EC_ms,            //from ph	sensor
	TDS_ppm,          //from ph	sensor
	TDS_ppt,          //from ph	sensor
	WATERTEMP_F,      //from ph	sensor
	WATERTEMP_C,      //from ph	sensor
	WATER_LEVEL,      //from water level sensor
	SEN_TYPE_CNT,
}SEN_LIST_T;

// sensor id type without unit
typedef enum
{
	_TEMP,             //from sh40t temp+humid sensor
	_HUMID,            //from sh40t temp+humid sensor
	_VPD,              //from sh40t temp+humid sensor
	_ZONE_TEMP,        //from shtc3 temp+humid zone sensor
	_ZONE_HUMID,       //from shtc3 temp+humid zone sensor
	_ZONE_VPD,         //from shtc3 temp+humid zone sensor
	_LEAFTEMP,         //from leaf  sensor
	_SOIL_HUMID,       //from soil  sensor
	_CO2,              //from co2   sensor
	_LIGHT,            //from light sensor
	_PH,               //from ph	sensor
	_EC_TDS,           //from ph	sensor
	_WATERTEMP,        //from ph	sensor
	_WATER_LEVEL,      //from water level sensor
	_SEN_TYPE_CNT,
}_SEN_LIST_T;


#define Bit(n)      (1UL << (n))

typedef struct
{
	float temp;
	float humid;
	bool  dectected;  // 是否正常
} zone_sensor_t;				// 与 传感器的 读取和显示 有关的变量


typedef struct 
{
//    s16 new;      // corrected_val = ()(read_val + cali_para)
//	s16 current;  // current display value
	u8 change_trends; //value changed direction 0: no change 1：falling, 2: rising
	s16 real_val;   // 校准后的实际读数：real_val = read_val + cali_para
	s16 current_val;  //当前显示值：有可能是上一次读数的值，与这次的real_val有差别
}smooth_disp_t;



/*
typedef struct
{
    float read_val;   // raw data read from sensor
	s16 cali_para;    // calibration para
	s16 corrected_val;  // corrected_val = ()(read_val + cali_para)
	s16 disp_val;  // current display value
	u8 change_trends; //value changed direction 0: no change 1：falling, 2: rising
    bool dectected;
}sensor_val_t;*/

typedef struct
{  
	smooth_disp_t disp_val;  // for smooth display of sensor data
    float read_val;   // raw data read from sensor 
    float real_val_float; // float type data after calibration
    uint8_t dectected;  // 0-undectected, 1-on sensor port1, 2-on sensor port2
}sensor_val_t;

typedef struct
{
sensor_val_t temp_f;       //from sh40t temp+humid sensor
sensor_val_t temp_c;       //from sh40t temp+humid sensor
sensor_val_t humid;        //from sh40t temp+humid sensor
sensor_val_t vpd;          //from sh40t temp+humid sensor
sensor_val_t zonetemp_f;   //from shtc3 temp+humid zone sensor
sensor_val_t zonetemp_c;   //from shtc3 temp+humid zone sensor
sensor_val_t zone_humid;   //from shtc3 temp+humid zone sensor
sensor_val_t zone_vpd;   //from shtc3 temp+humid zone sensor
sensor_val_t leaftemp_f;   //from leaf  sensor
sensor_val_t leaftemp_c;   //from leaf  sensor
sensor_val_t soil_humid;   //from soil  sensor
sensor_val_t co2;          //from co2   sensor
sensor_val_t light;        //from light sensor
sensor_val_t ph;		   //from ph	sensor
sensor_val_t ec_us;		   //from ph	sensor
sensor_val_t ec_ms;			//from ph	sensor
sensor_val_t tds_ppm;		//from ph	sensor
sensor_val_t tds_ppt;		//from ph	sensor
sensor_val_t watertemp_f;  //from ph	sensor
sensor_val_t watertemp_c;  //from ph	sensor
sensor_val_t water_level;  //from water level sensor
sensor_val_t ph_mv; 		//from ph	sensor
}sensors_t;



typedef struct sensor_type
{
   // the lowest bit
   u8 sub_type  : 4;  // temp+humid sensor-[0: temp, 1: humid, 2: vpd]  / PH sensor-[0:EC, 1:TDS, 2: watertemp, 3: ph]
   u8 dev_type  : 4;  // 0: temp+humid sensor, 1: zone temp+humid sensor, 2: leaf, 3: soil,  4: co2+light, 5: ph
   // the highest bit
}sensor_type_t;	

typedef struct
{
    //------the first sent byte-------
    
    // >>1byte
    struct config_bit
    {
        // the lowest bit
        u8 high_trigger_en   : 1;
		u8 low_trigger_en    : 1;
		u8 target_en         : 1;
		u8 trans_en          : 1;
		u8 buffer_en         : 1;
		u8 auto_or_target    : 1;  // 0:auto, 1: target
		u8 reseverd          : 2;
		// the highest bit
        
		// the lowest bit
		u8 adjust_precision  : 2;  // (actual trans/buffer) = (trans/buffer) x (0: 10, 1: 1, 2: 0.1, 3: 0.01), 
		u8 value_precision   : 2;  // (high/low/target) = (actual high/low/target) * (0:1, 1:10, 2:100, 3: 1000),  for PH,VPD =1, 
//		u8 reseverd1         : 4;
		u8 high_close_type   : 2;  // 
		u8 low_close_type	 : 2;  // 

		// the highest bit
	}config;

    // >>4byte
    u8 trans;
	u8 buff;

	// >>6byte
	s16 target;
	s16 high;
	s16 low;

    // >>2byte
	// u8 on_max;  //turn to off mode when reaches on_max time, and auto adjust will be ended up.
	// u8 on_min;

	//------the last sent byte--------
}sensor_autoctrl_setting_t;

// sensor precision
// sensor unit
// sensor cali


// const data
//   1->about sensor[sen_i] : sen_i = temp_c,tempf,humid,co2,ph,ec,tds....
//      1.x ->precision of trans_and_buff, data_for_app, data_for_lcd, data_for_storage
//      1.4 ->max cali para
//      1.5 ->max trans
//      1.6 ->max buff
//      1.7 ->max unit
//      1.8 ->max data
//   2->about port rule[port] :  port=1-8, 9(all)

// gloable data interacted between disp(lcd,app), set(key,app), and storage(flash)

// realtime data to lcd/app
//   1->sensor data
//      1.x-> s16 data,cali, precision,unit
//      2.x-> sensor data cnt
//      3.x-> sensor type, port, 
//      4.x-> has new log

// sys setting from/to dev/app
//   1->clock
//   2->backlight
//   3->sensors settings[sen_i] :  sen_i = temp_c,tempf,humid,co2,ph,ec,tds....
//      3.1->cali
//      3.2->unit

// port-rule-setting[port]: port=1-8, 9(all)
//   1->sensors autoctrl settings[sen_i] : sen_i = temp_c,tempf,humid,co2,ph,ec,tds....
//      1.x->high,low,trans,buff, + config_en_bits 
//      
//   2->time ctrl settings
//      2.1->timer on/off settings
//      2.2->cycle on/off
//      2.3->schedule on/off 
//      2.4->config_en_bits 
//   3->mix_ctrl settings
//      2.5->on_max = autoctrl rule & timer off rule(higher priority)
//      2.6->on_min = autoctrl rule | timer off rule(higher priority)
//   3->dose time settings = cycle on/off rule & autoctrl rule(higher priority)
//      3.x->dose on/off
//   4->advance mode settings: based on elements collections from 1. 2. 3, each with running period set
//   


typedef enum
{
PRECISION_10,
PRECISION_1,
PRECISION_0_1,
PRECISION_0_0_1,
}PRESION_VAL_T;

typedef enum
{
UNIT_F,
UNIT_C,
UNIT_TEMP_CNT,
}TEMP_UNIT_T;

typedef enum
{
INDOOR,
OUTDOOR,
SENSOR_SRC_CNT,
}SENSOR_SRC_T;
extern sensors_t g_sensors, g_sensors2;
extern sensor_val_t *g_sensor;
extern sensor_val_t *g_sensor2;

extern void refresh_zone_calib(void);
extern void RefreshCalib(void);  //当设置校准值时，刷新当前温湿度数值
extern void refresh_sensors_data_after_calib(u8 sensortype);

extern bool is_zone_sensor_lost(void);
extern bool is_no_water(void);

extern void ReadSensor(void);
extern void delayMs(uint32 num);

//#define SENSOR_DATABLOCK_CNT  (SEN_TYPE_CNT)	// 最大的 传感器数据块个数
#define SENSOR_DATABLOCK_CNT  (12)	// 最大的 传感器数据块个数

#define CO2_VAL_MAX 5000
#define LIGHT_VAL_MIN   10  // 0.1 % * LIGHT_VAL_MIN
#define AUTO_CALIB_TIME 1800   // 180s
#define CALIB_SAVE_TIME	300       //  30s

typedef struct
{
    u8 cnt;                        // 实际有效的 传感器数据块的个数
    u8 from[SENSOR_DATABLOCK_CNT];
    u8 type[SENSOR_DATABLOCK_CNT]; // 某个有效传感器数据块，其数据块的类型 （ SEN_LIST_T ）
} sensor_datablock_t;

// 4 bytes
typedef struct 
{
    //------the first sent byte-------
 
	// >>1byte	   
	u8 sensor_type;  // defined by enum SEN_LIST_T

	// >>1byte
    struct sensor_state  
    {
        // the lowest bit
        u8 sensor_port      : 3; // 0:not dectected, 1:from sensor port1, 2: from sensor port2, 7:from inside pcb
		u8 change_trends    : 2;   // 0: stay, 1: down, 2: up
		u8 precision        : 2;   // 0: actual_val/10, 1: actual_val, 2: actual_val*10, 3:actual_val*100
		u8 unit             : 1;   // [0:F,1:C], [0: us/cm, 1:ms/cm], [0: ppm, 1:ppt]
		// the highest bit
	}state;   

	// >>2byte, higher byte sent first
	s16 disp_val;   
	
    //------the last sent byte--------
}sensor_data_t;  

typedef struct{
	u8 temp      : 1;  // 0:f, 1:c
	u8 EC        : 1;  // 0:us/cm, 1:mc/cm
	u8 TDS       : 1;  // 0:ppm, 1:ppt
	u8 EC_or_TDS : 1;  // 0:EC, 1:TDS
	u8 reserved  : 4;
	//	8 bit
}sensor_unit_t;

typedef struct
{
	sensor_unit_t unit;
	struct
	{
		u8 switch_zone_pos         : 1;   
	    u8 reserved                : 7;   // switch zone sensor posistion 
	}config;  // zone sensor
	u8 sensor_type_cnt;
	s16 cali_para[SEN_TYPE_CNT];
	s16 cali_para_second[SEN_TYPE_CNT];
}sensor_setting_t;

typedef struct
{
	struct
	{
		u8 sensor_port: 4;
		u8 sensor_type: 4;		
	}sensor_info;
	struct
	{
		u8 cal_status: 4;    // 0:校准完成       1:校准中     2:待保存      3:校准取消
		u8 cal_step: 4;		
	}cal_info;
	u16 cal_val;
	u32 time_stamp;
}sensor_self_cal_data_t;

typedef struct
{
	u32 ec_low_time;
	u32 ec_middle_time;
	u32 ec_high_time;
}sensor_self_cal_time_t;

typedef struct 
{
	u8 sensor_type : 4;
	u8 sensor_port : 4;
	u8 cal_status:   4;
	u8 cal_step:	  4;
 }sensor_cal_state_t;

extern sensor_self_cal_data_t sensor_self_cal;
extern sensor_self_cal_time_t sensor_self_cal_time[2];
extern sensor_cal_state_t ph_cal_state[2];
extern sensor_cal_state_t ec_cal_state[2];
extern bool co2_auto_self_calib_flag;	
extern bool co2_setting_reset_flag;
extern bool ph_auto_self_calib_flag; 
extern bool ph_setting_reset_flag;
extern bool ec_auto_self_calib_flag;
extern bool ec_setting_reset_flag;
extern uint16_t auto_self_calib_time;
extern uint8_t auto_self_calib_retrytimes;
extern u8 co2_cali_step;
extern uint8_t ph_ec_cali_step;

bool is_sensor_value_invalid(u8 sensor_type);
extern u8 get_sensor_type_from_workmode(u8 workmode);
extern s16 get_real_zonetemp(bool is_unit_f);
extern s16 get_real_zonehumid(void);
extern s16 get_real_zonevpd(void);
extern s16 get_sensor_val(u8 sensor_type);
extern s16 get_real_soil_humid(void);
extern s16 get_real_co2(void);
extern s16 get_real_water(void);
extern s16 get_real_ph(void);
extern u8 get_highlow_precision_multi(u8 sensor_type);
extern u8 get_transbuff_precision_multi(u8 sensor_type);

extern s16 GetRealTemp(bool is_unit_f);
extern s16 GetRealHumid(void);
extern s16 GetRealVpd(void);
s16 to_s16x100(u8 raw_num);
s16 to_s16x10(u8 raw_num);

extern bool is_zone_pos_switched(void);
extern bool is_tds_unit_ppm(void);
extern bool is_ec_unit_us(void);
extern bool is_temp_unit_f(void);
extern bool is_cur_ec_en(void);
extern u8 to_sensor_id_with_unit(u8 sensor_id_no_unit);
extern double get_svp(double t);
extern void get_vpd(void);

extern void GetTempHumidByte(u8 * data);  //将 设备显示屏 显示的 温度 和 湿度 转换为 4个字节
extern void GetVpdByte(u8 * data);
extern void sensor_getDataBlockArg(sensor_datablock_t * sensor_datablock, bool flag);
extern void sensor_fillDataBlockValue(u8 from, u8 sensor_type, sensor_data_t * p);
extern void sensor_cali_fillDataBlockValue(u8 from, u8 sensor_type, sensor_data_t * p);
bool is_sensor_plug_in(void);
bool is_zone_sensor_lost(void);
bool is_sensor_on_cali(u8 sensor_type_id);
extern void auto_calibration_check(void); 
#pragma pack() 

#endif

