#include "Sensor.h"
#include "Function.h"
#include "bsp.h"
#include "define.h"
#include "math.h"
#include "variable.h"
#include <stdio.h>
#include "includes.h"
#include "esp_log.h"
#include "sensor_driver.h"

#define TAG "sensor    " // 10个字符

u16 SensorhumiArray[SensorhumiArraySize];
u8 SensorhunArrayCnt = 0;
float t_sht30[2], h_sht30[2];
sensors_t g_sensors,g_sensors2;
sensor_self_cal_data_t sensor_self_cal;
sensor_self_cal_time_t sensor_self_cal_time[2];
sensor_cal_state_t ph_cal_state[2];
sensor_cal_state_t ec_cal_state[2];
bool co2_auto_self_calib_flag = false;	
bool co2_setting_reset_flag = false;
bool ph_auto_self_calib_flag = false; 
bool ph_setting_reset_flag = false;
bool ec_auto_self_calib_flag = false;
bool ec_setting_reset_flag = false;
uint16_t auto_self_calib_time = 0;
uint8_t auto_self_calib_retrytimes = 0;
u8 co2_cali_step = 0;
uint8_t ph_ec_cali_step = 0;


static u8 g_replace_pcb_zone_sensor=false;

sensor_val_t *g_sensor = &g_sensors.temp_f;
sensor_val_t *g_sensor2 = &g_sensors2.temp_f;




//EM_SENSOR_STA ext_sensor_sta = SENSOR_NULL;
/*
****************************************************************************************************
* 函数名称：
* 函数功能： 将 double 类型数  转换为 int 类型数
* 输入：
* 输出：
* 返回值：
****************************************************************************************************
*/
s16 Double_2_Short(double n)
{
    s16 i = 0;

    if (n > 32767 || n < -32768)
    {
        return 0;
    }

    if (n > -1 && n < 1)
    {
        if (n <= -0.5)
        {
            i = -1;
        }
        else if (n < 0.5)
        {
            i = 0;
        }
        else
        {
            i = 1;
        }
    }
    else
    {
        if (n > 0)
        {
            i = (s16)(n + 0.5);
        }
        else
        {
            i = (s16)(n - 0.5);
        }
    }
    return i;
}

void delayMs(uint32 num)
{
    uint32 i = 0;
    uint32 j = 0;

    for (i = 0; i < num; i++)
    {
        for (j = 0; j < 100; j++)
        {
            ;
        }
    }
}

s16 float_to_int(float val_float, u8 save_precision)
{
    s16 val;

	switch(save_precision)
	{
		case PRECISION_10:    val = val_float/10; break;
		case PRECISION_1:     val = val_float;    break;
		case PRECISION_0_1:   val = val_float*10; break;
		case PRECISION_0_0_1: val = val_float*100;break;
		default:              val = val_float;    break;
	}

	return val;
}


bool is_zone_pos_switched(void)
{
   return g_sys_setting.sensor_set.config.switch_zone_pos;
}

bool is_cur_ec_en()
{
	return (0 == g_sys_setting.sensor_set.unit.EC_or_TDS);
}

bool is_ec_unit_us(void)
{
    return (0 == g_sys_setting.sensor_set.unit.EC);
}
bool is_tds_unit_ppm(void)
{
    return (0 == g_sys_setting.sensor_set.unit.TDS);
}

bool is_temp_unit_f(void)
{
    return (0 == g_sys_setting.sensor_set.unit.temp);
}

u8 to_sensor_id_with_unit(u8 sensor_id_no_unit)
{
    switch(sensor_id_no_unit)
    {
    case _TEMP : if(is_temp_unit_f()) { return TEMP_F;} else {return TEMP_C;}
	case _HUMID : return HUMID;
	case _VPD : return VPD;
	case _ZONE_TEMP : if(is_temp_unit_f()) { return ZONE_TEMP_F;} else {return ZONE_TEMP_C;}
	case _ZONE_HUMID : return ZONE_HUMID;
	case _ZONE_VPD : return ZONE_VPD;
	case _LEAFTEMP : if(is_temp_unit_f()) { return LEAFTEMP_F;} else {return LEAFTEMP_C;}
	case _SOIL_HUMID : return SOIL_HUMID;
	case _CO2 : return CO2;
	case _LIGHT : return LIGHT;
	case _PH : return PH;
	case _EC_TDS : 
		if(g_sys_setting.sensor_set.unit.EC_or_TDS) 
		{
		    if(is_tds_unit_ppm()) { return TDS_ppm;} else {return TDS_ppt;}
		}
		else
		{
		    if(is_ec_unit_us()) { return EC_us;} else {return EC_ms;}
		}
		break;
	case _WATERTEMP : if(is_temp_unit_f()) { return WATERTEMP_F;} else {return WATERTEMP_C;}
	case _WATER_LEVEL : return WATER_LEVEL;
	default : return SEN_TYPE_CNT;
    }
}

/*
****************************************************************************************************
*    函数名称：
*    功能说明：
*    参   数：
*    返 回 值：
*
****************************************************************************************************
*/
double get_svp(double t)
{
    double svp, power;

    // power = t / (t + 238.3) * 17.2694;
    power = t / (t + 237.3) * 17.2694;

    svp   = 610.78 * pow(2.71828, power);

    return svp;
}

void get_vpd(void)
{
    double air_svp, leaf_svp;

    // s16 air_t, leaf_t;
    // s8 t_offset;

    float air_t, leaf_t;
    float t_offset;

#if 0
	if (!g_sensors.vpd.dectected)
		return;
#else
    if (sensor.readError)
        return;
#endif

    if (is_temp_unit_f()) // 华氏度
    {
        // t_offset = g_sensors.leaftemp_f.cali_para >> 1;
        t_offset = g_sys_setting.sensor_set.cali_para[VPD] * 0.5;
    }
    else
    {
        t_offset = g_sys_setting.sensor_set.cali_para[VPD]* 0.5;;
    }
	if(is_zone_pos_switched())
		t_offset = g_sys_setting.sensor_set.cali_para[ZONE_VPD] * 0.5;

    float humid = GetRealHumid()/100.0f;
    float temp_c = GetRealTemp( is_temp_unit_f() )/100.0f;
	if( is_temp_unit_f() ){
        s16 i=0;
		temp_c = F_to_C( temp_c );
        i = (temp_c*10 + 0.5);
        temp_c = i/10.0f;
	}
    // ESP_LOGW(TAG, "humid:%f || temo:%f", humid, temp_c );

    air_t              = temp_c;
    leaf_t             = temp_c + t_offset;

    air_svp            = get_svp(air_t);
    leaf_svp           = get_svp(leaf_t);

    sensor.vpdComputer = leaf_svp - (humid * air_svp / 100);

    // sensor.showVpd = DoubleToShort(sensor.vpdComputer);
    if(sensor.vpdComputer < 0)
		sensor.vpdComputer = 0;
    sensor.showVpd     = (sensor.vpdComputer);
}


float calied_temp(float raw_temp_c, s16 cali_para_c, s16 cali_para_f, u8 cur_unit) 
{
    float calied_temp;
	
    if(cur_unit == UNIT_F)
		calied_temp = raw_temp_c +  cali_para_f/1.8;
	else
		calied_temp = raw_temp_c + cali_para_c;

    if(cur_unit == UNIT_F)
		calied_temp = C_to_F(calied_temp);

    return calied_temp;
}


void cali_temp_humid_after_read(void)
{
	if(is_zone_pos_switched()) //zone sensor and external sensor swapped, use zone calib value
	{
		 if (is_temp_unit_f()) // 华氏度
	        sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_F] / 1.8;
	    else               // 摄氏度
	        sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_C]; 
	}
	else
	{
	    if (is_temp_unit_f()) // 华氏度
	        sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[TEMP_F] / 1.8;
	    else               // 摄氏度
	        sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[TEMP_C]; 
	}

    // sensor.ObjTempC = Double_2_Short(sensor.readTemp);//这个函数实现四舍五入
    // sensor.ObjTempF = Double_2_Short(sensor.readTemp*1.8+32);
    sensor.ObjTempC  = (s16)sensor.readTemp;
    sensor.ObjTempF  = (s16)C_to_F(sensor.readTemp);
	
	if(is_zone_pos_switched())   //zone sensor and external sensor swapped, use zone calib value
		sensor.readHumid = h_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_HUMID];
	else
    	sensor.readHumid = h_sht30[0] + g_sys_setting.sensor_set.cali_para[HUMID];

    if (sensor.readHumid > 100)
        sensor.readHumid = 100;

    if (sensor.readHumid < 0)
        sensor.readHumid = 0;

    // sensor.ObjHumid = Double_2_Short(sensor.readHumid);
    sensor.ObjHumid = (s16)sensor.readHumid; // frank 不要四舍五入
    if (sensor.readError)                    // 上次读取数据出错，则直接显示读取的数
    {
        sensor.showTempC = sensor.ObjTempC;
        sensor.showTempF = sensor.ObjTempF;
        sensor.showHumid = sensor.ObjHumid;
    }

    if (sensor.firstReadFlag) // 单片机上电后第一次读取温湿度数据，则直接显示读取的数据
    {
        sensor.firstReadFlag = 0;
        sensor.showTempC     = sensor.ObjTempC;
        sensor.showTempF     = sensor.ObjTempF;
        sensor.showHumid     = sensor.ObjHumid;
    }
	if(switch_zone_pos_delay_time)
	{
		sensor.showTempF = sensor.ObjTempF;
		sensor.showTempC = sensor.ObjTempC;
		sensor.showHumid = sensor.ObjHumid;
	}
}

void read_sht40_sht30(void)
{
    u16 detected_list = 0;
	u8 detected[SENSOR_PORT_CNT];
	
	detected_list = sensor_temp_humid_sht40_read(t_sht30, h_sht30);
	//if(g_sensors.temp_c.dectected)
	//	sht40_detect_ok = detected_list;
	
    if(!detected_list)
    {
		//fix : 解决插上传感器，APP的温湿度数据会闪烁一下的情况		
		g_sensors.temp_c.dectected = false;//&& !g_sys_settings.temp_unit_F;  
		g_sensors.temp_f.dectected = false;//&& g_sys_settings.temp_unit_F;
		g_sensors.humid.dectected  = false;  
	    g_sensors.vpd.dectected = false;
		
		g_sensors2.temp_c.dectected = false;//&& !g_sys_settings.temp_unit_F;  
		g_sensors2.temp_f.dectected = false;//&& g_sys_settings.temp_unit_F;
		g_sensors2.humid.dectected  = false;  
		g_sensors2.vpd.dectected = false;
		g_replace_pcb_zone_sensor = false;
		
        sensor.readError = 1; // 数据读取错误，屏幕显示 --C--%
		return;  
    }
	
	//fix : 解决插上传感器，APP的温湿度数据会闪烁一下的情况
    detected[0] = detected_list & (0x7);
	detected_list >>= 3;
	detected[1] = detected_list & (0x7);
	
	g_sensors.temp_c.dectected = detected[0]; 
	g_sensors.temp_f.dectected = detected[0];
	g_sensors.humid.dectected  = detected[0];  
    g_sensors.vpd.dectected = detected[0];

    g_replace_pcb_zone_sensor = detected[1];
	g_sensors2.temp_c.dectected = false;//detected[1];  
	g_sensors2.temp_f.dectected = false;//detected[1];
	g_sensors2.humid.dectected  = false;//detected[1];  
    g_sensors2.vpd.dectected = false;//detected[1]; 

	g_sensors.temp_f.real_val_float = C_to_F(t_sht30[0]) ;
	g_sensors.temp_c.real_val_float = t_sht30[0]; 

	g_sensors.humid.real_val_float = h_sht30[0];

    if (g_sensors.humid.real_val_float > 100)
        g_sensors.humid.real_val_float = 100;

    if (g_sensors.humid.real_val_float < 0)
        g_sensors.humid.real_val_float = 0;
}

/*
****************************************************************************************************
* 函数名称：
* 函数功能： 当设置校准值时， 刷新当前zone温湿度数据（传感器的数值 + 设置的校准值）
* 输入：
* 输出：
* 返回值：
****************************************************************************************************
*/
void refresh_zone_calib(void)
{
	static u8 poweronflag = 1;
	
    if ( g_sensors.zonetemp_c.dectected )
    {
        //fixbug-v40: zone temp调节cali值的时候显示问题
        if (is_temp_unit_f())
		{
			if(is_zone_pos_switched())  //zone sensor and external sensor swapped, use ext-sensor calib value
				g_sensors.zonetemp_f.real_val_float = C_to_F(g_sensors.zonetemp_c.read_val) + g_sys_setting.sensor_set.cali_para[TEMP_F];
			else
            	g_sensors.zonetemp_f.real_val_float = C_to_F(g_sensors.zonetemp_c.read_val) + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_F];
			g_sensors.zonetemp_f.disp_val.real_val =   g_sensors.zonetemp_f.real_val_float; 
			g_sensors.zonetemp_c.disp_val.real_val = F_to_C(g_sensors.zonetemp_f.real_val_float);
			if(abs((s32)(g_sensors.zonetemp_f.real_val_float - g_sensors.zonetemp_f.disp_val.current_val)) > 1)
				g_sensors.zonetemp_f.real_val_float = g_sensors.zonetemp_f.disp_val.current_val;
		}
        else
        {
        	if(is_zone_pos_switched())  //zone sensor and external sensor swapped, use ext-sensor calib value
				g_sensors.zonetemp_c.real_val_float = g_sensors.zonetemp_c.read_val + g_sys_setting.sensor_set.cali_para[TEMP_C];
			else
          	  	g_sensors.zonetemp_c.real_val_float = g_sensors.zonetemp_c.read_val + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_C];
			g_sensors.zonetemp_c.disp_val.real_val = g_sensors.zonetemp_c.real_val_float;
			g_sensors.zonetemp_f.disp_val.real_val  = C_to_F(g_sensors.zonetemp_c.real_val_float);
			if(abs((s32)(g_sensors.zonetemp_c.real_val_float - g_sensors.zonetemp_c.disp_val.current_val)) > 1)
				g_sensors.zonetemp_c.real_val_float = g_sensors.zonetemp_c.disp_val.current_val;
        }

		if(is_zone_pos_switched())	//zone sensor and external sensor swapped, use ext-sensor calib value
			g_sensors.zone_humid.real_val_float = g_sensors.zone_humid.read_val + g_sys_setting.sensor_set.cali_para[HUMID];
		else
			g_sensors.zone_humid.real_val_float = g_sensors.zone_humid.read_val + g_sys_setting.sensor_set.cali_para[ZONE_HUMID];
        g_sensors.zone_humid.disp_val.real_val = g_sensors.zone_humid.real_val_float;
		if(abs((s32)(g_sensors.zone_humid.real_val_float - g_sensors.zone_humid.disp_val.current_val)) > 1)
			g_sensors.zone_humid.real_val_float = g_sensors.zone_humid.disp_val.current_val;

        if (g_sensors.zone_humid.disp_val.real_val > 100)
        {
         	g_sensors.zone_humid.disp_val.real_val = 100;
        }

        if (g_sensors.zone_humid.disp_val.real_val < 0)
        {
            g_sensors.zone_humid.disp_val.real_val = 0;
        }
		if (poweronflag == 1)   // 上电第一次读数
		{
			poweronflag = 0;
			g_sensors.zonetemp_c.disp_val.current_val = g_sensors.zonetemp_c.disp_val.real_val;
			g_sensors.zonetemp_f.disp_val.current_val = g_sensors.zonetemp_f.disp_val.real_val;
			g_sensors.zone_humid.disp_val.current_val = g_sensors.zone_humid.disp_val.real_val;
		}
		if(switch_zone_pos_delay_time)
		{
			g_sensors.zonetemp_c.disp_val.current_val = g_sensors.zonetemp_c.disp_val.real_val;
			g_sensors.zonetemp_f.disp_val.current_val = g_sensors.zonetemp_f.disp_val.real_val;
			g_sensors.zone_humid.disp_val.current_val = g_sensors.zone_humid.disp_val.real_val;
		}
    }
}

/*
****************************************************************************************************
* 函数名称：
* 函数功能： 当设置校准值时， 刷新当前温湿度数据（传感器的数值 + 设置的校准值）
* 输入：
* 输出：
* 返回值：
****************************************************************************************************
*/
void RefreshCalib(void)
{
    if (sensor.readError == 0)
    {
        if (is_temp_unit_f())
        {
        	if(is_zone_pos_switched())
				sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_F] / 1.8;
			else
            	sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[TEMP_F] / 1.8;
        }
        else
        {
        	if(is_zone_pos_switched())
				sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_TEMP_C];
			else
          		sensor.readTemp = t_sht30[0] + g_sys_setting.sensor_set.cali_para[TEMP_C];
        }

        sensor.ObjTempC  = (sensor.readTemp);
        sensor.ObjTempF  = C_to_F(sensor.readTemp);//(sensor.readTemp * 1.8 + 32);

		if(is_zone_pos_switched())
			sensor.readHumid = h_sht30[0] + g_sys_setting.sensor_set.cali_para[ZONE_HUMID];
		else
        	sensor.readHumid = h_sht30[0] + g_sys_setting.sensor_set.cali_para[HUMID];

        if (sensor.readHumid > 100)
        {
            sensor.readHumid = 100;
        }

        if (sensor.readHumid < 0)
        {
            sensor.readHumid = 0;
        }

        sensor.ObjHumid  = (sensor.readHumid);
        sensor.showTempC = sensor.ObjTempC;
        sensor.showTempF = sensor.ObjTempF;
        sensor.showHumid = sensor.ObjHumid;
    }
	if(is_zone_pos_switched())
		switch_zone_pos_delay_time = 20;
}

typedef struct
{
	u8 sensortype;
//	sensor_val_t *sensor_data;
	u16 value_max;
	u16 value_min;
}sensor_cali_data_t;

const sensor_cali_data_t sensor_cali_data[SEN_TYPE_CNT]=
{
	{TEMP_F, 194, 32},
	{TEMP_C, 90,   0},
	{HUMID,  100,  0},
	{VPD,	 99,   0},
	{ZONE_TEMP_F, 194,	32},
	{ZONE_TEMP_C, 90,    0},
	{ZONE_HUMID,  100,   0},
	{ZONE_VPD,    99,    0},
	{LEAFTEMP_F,  194,  32},
	{LEAFTEMP_C,  90,    0},
	{SOIL_HUMID,  100,   0},
	{CO2,     20000,   0},
	{LIGHT,   9999,   0},
	{PH,      140,    0},
	{EC_us,   30000,   0},
	{EC_ms,   200,   0},
	{TDS_ppm, 30000,   0},
	{TDS_ppt, 100,   0},
	{WATERTEMP_F, 140, 32},
	{WATERTEMP_C, 60,  0},
	{WATER_LEVEL,   1,	0},
};



void refresh_sensors_data_after_calib(u8 sensortype)
{
//	if(g_sensor[sensortype].dectected )
	{
		g_sensor[sensortype].real_val_float = g_sensor[sensortype].read_val + g_sys_setting.sensor_set.cali_para[sensortype];
		g_sensor[sensortype].disp_val.real_val = g_sensor[sensortype].read_val + g_sys_setting.sensor_set.cali_para[sensortype];
	}
	if(g_sensor[sensortype].disp_val.real_val > sensor_cali_data[sensortype].value_max && sensortype != EC_ms)
	{
		g_sensor[sensortype].disp_val.real_val = sensor_cali_data[sensortype].value_max;
		g_sensor[sensortype].real_val_float = g_sensor[sensortype].disp_val.real_val;
	}
	else if(g_sensor[sensortype].disp_val.real_val < sensor_cali_data[sensortype].value_min)
	{
		g_sensor[sensortype].disp_val.real_val = sensor_cali_data[sensortype].value_min;
		g_sensor[sensortype].real_val_float = g_sensor[sensortype].disp_val.real_val;
	}
}

void refresh_sensors2_data_after_calib(u8 sensortype)
{
//	if(g_sensor2[sensortype].dectected )
	{
		g_sensor2[sensortype].real_val_float = g_sensor2[sensortype].read_val + g_sys_setting.sensor_set.cali_para_second[sensortype];
		g_sensor2[sensortype].disp_val.real_val = g_sensor2[sensortype].read_val + g_sys_setting.sensor_set.cali_para_second[sensortype];
	}
	if(g_sensor2[sensortype].disp_val.real_val > sensor_cali_data[sensortype].value_max)
	{
		g_sensor2[sensortype].disp_val.real_val = sensor_cali_data[sensortype].value_max;
		g_sensor2[sensortype].real_val_float = g_sensor2[sensortype].disp_val.real_val;
	}
	else if(g_sensor2[sensortype].disp_val.real_val < sensor_cali_data[sensortype].value_min)
	{
		g_sensor2[sensortype].disp_val.real_val = sensor_cali_data[sensortype].value_min;
		g_sensor2[sensortype].real_val_float = g_sensor2[sensortype].disp_val.real_val;
	}
}

float get_vpd_zone(void)
{
	double air_svp, leaf_svp;
    s16 temp_c;
	// s16 air_t, leaf_t;
	// s8 t_offset;

	float air_t, leaf_t;
	float t_offset;
	float zone_vpd_computer;

	if (is_temp_unit_f()) // å华氏度
	{
		//t_offset = g_sensors.leaftemp_f.cali_para >> 1;
		t_offset = g_sys_setting.sensor_set.cali_para[ZONE_VPD] * 0.5;
	}
	else
	{
		t_offset = g_sys_setting.sensor_set.cali_para[ZONE_VPD] * 0.5;
	}

	if(is_zone_pos_switched())
		t_offset = g_sys_setting.sensor_set.cali_para[VPD] * 0.5;
	temp_c = g_sensors.zonetemp_c.disp_val.current_val;	
	air_t = temp_c;
	leaf_t = temp_c + t_offset;

	air_svp = get_svp(air_t);
	leaf_svp = get_svp(leaf_t);
	
	zone_vpd_computer = leaf_svp - (g_sensors.zone_humid.disp_val.current_val * air_svp / 100);
	
	if(zone_vpd_computer < 0)
		zone_vpd_computer = 0;
	return zone_vpd_computer;
}

bool is_sensor_on_cali(u8 sensor_type_id)
{
	if(sensor_type_id == CO2)
    	return co2_auto_self_calib_flag;
	else if(sensor_type_id >= PH && sensor_type_id <= WATERTEMP_C)
		return (ph_auto_self_calib_flag || ec_auto_self_calib_flag);
	else
		return false;
}

bool is_sensor_value_invalid(u8 sensor_type)
{
    return get_sensor_val(sensor_type) >= _CONST.SENSOR_RANGE[sensor_type].max;
}

void call_co2_alert(bool co2_exceed)
{
   	// Alarm_msglog_Co2Over5000(co2_exceed);
}

void check_sensors_value_valid(void)
{
    u8 i;

	for(i=0; i<SEN_TYPE_CNT; i++)
	{
		if(i == CO2)
		  	call_co2_alert(is_sensor_value_invalid(CO2));
	}
}

bool is_sensor_plug_in(void)
{
	if(g_sensors.temp_c.dectected)  //温湿度传感器
		return true;
	else if(g_sensors.co2.dectected || g_sensors.light.dectected)  //co2+光感传感器
		return true;
	else if(g_sensors.ph.dectected)   //ph传感器 
		return true;
	else if(g_sensors.soil_humid.dectected)  //土壤湿度传感器
		return true;
	else if(g_sensors.water_level.dectected)  //水检测传感器
		return true;
	else
		return false;
}

void swap_indoor_and_outdoor(void)
{
	float tempTemp;
	float tempHumid;
		
	if(!g_sensors.temp_c.dectected)
		return;
	
	tempTemp = g_sensors.zonetemp_c.read_val;
	tempHumid = g_sensors.zone_humid.read_val;
	g_sensors.zonetemp_c.read_val = t_sht30[0];
	g_sensors.zone_humid.read_val = h_sht30[0];
	t_sht30[0] = tempTemp;
	h_sht30[0] = tempHumid;
	cali_temp_humid_after_read();
}

void refresh_temp_humid_data(void)
{
	swap_indoor_and_outdoor();		// 交换内外温湿度数据
	cali_temp_humid_after_read();
	refresh_zone_calib();  //当设置校准值时，刷新当前温湿度数值
	get_vpd();
	g_sensors.zone_vpd.read_val = get_vpd_zone();
}

#if ((_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX)))
bool is_zone_sensor_lost(void)
{
	if(!g_sensors.zonetemp_c.dectected && is_sensor_poweron_det_ok())
	{
		err_flash_flag |= (1 << ERR_NO_TEMP_SENSOR);
		return true;
	}
	else
	{
		err_flash_flag &= ~(1 << ERR_NO_TEMP_SENSOR);
		return false;
	}
}

bool is_no_water(void)
{
	return (err_flash_flag & (1 << ERR_NO_WATER));
}

static void no_water_check(void)
{
	static u32 no_water_time = 0;

	if(g_sensors.water_level.read_val == 0)
		no_water_time++;
	else
		no_water_time = 0;
	if(no_water_time >= 540000)   //60*60*24*5*5/4
//	if(no_water_time >= 540)
		err_flash_flag |= (1 << ERR_NO_WATER);
	else 
		err_flash_flag &= ~(1 << ERR_NO_WATER);
}
#endif
/*
****************************************************************************************************
* 函数名称：
* 函数功能：
* 输入：
* 输出：
* 返回值：
****************************************************************************************************
*/
void ReadSensor(void)
{
    static uint8_t i=0;
	float param1[SENSOR_PORT_CNT]={0.0f,0.0f},param2[SENSOR_PORT_CNT]={0.0f,0.0f},param3[SENSOR_PORT_CNT]={0.0f,0.0f},param4[SENSOR_PORT_CNT]={0.0f,0.0f},param5[SENSOR_PORT_CNT]={0.0f,0.0f};
    u16 detected; 
	switch(i)
	{
	case 0: i = 1; read_sht40_sht30();  break;
		
    case 1: i = 2; 
	    if(g_replace_pcb_zone_sensor)
	    {
	        detected = g_replace_pcb_zone_sensor;
			g_sensors.zonetemp_c.read_val = t_sht30[1];
			g_sensors.zone_humid.read_val = h_sht30[1];
	    }
		else
		    detected = sensor_temp_humid_zone_read(&g_sensors.zonetemp_c.read_val, &g_sensors.zone_humid.read_val); 

	    g_sensors.zonetemp_f.read_val = C_to_F(g_sensors.zonetemp_c.read_val); // temp_f is a simulated sensor that outputs temperature by unit F
		g_sensors.zonetemp_c.dectected = detected ;//&& !g_sys_settings.temp_unit_F;  
		g_sensors.zonetemp_f.dectected = detected ;//&& g_sys_settings.temp_unit_F;
		g_sensors.zone_humid.dectected  = detected;     
		g_sensors.zone_vpd.dectected = detected;   
		
		if(is_zone_pos_switched() && g_sensors.temp_c.dectected && g_sensors.zonetemp_c.dectected)
		 	swap_indoor_and_outdoor();		// 交换内外温湿度数据
		else if(is_zone_pos_switched() && !g_sensors.temp_c.dectected && is_sensor_poweron_det_ok())
			g_sys_setting.sensor_set.config.switch_zone_pos = 0;
//		if(g_sensors.temp_c.dectected)
//			cali_temp_humid_after_read();
		refresh_zone_calib();  //当设置校准值时，刷新当前温湿度数值
		g_sensors.zone_vpd.read_val = get_vpd_zone();
		break;	
		
    case 2: 
		i = 3; 
		static u8 soil_dualdetect_bak;
		detected =  sensor_soil_read(param1);	
	    g_sensors.soil_humid.dectected = detected & (0x7);
		detected >>= 3;
		g_sensors2.soil_humid.dectected = detected & (0x7);
		
		if(g_sensors.soil_humid.dectected)
		{
		    g_sensors.soil_humid.read_val = param1[0];
			refresh_sensors_data_after_calib(SOIL_HUMID);
		}
		if(g_sensors2.soil_humid.dectected)
		{
		    g_sensors2.soil_humid.read_val = param1[1];
			refresh_sensors2_data_after_calib(SOIL_HUMID);
		}
		else if(soil_dualdetect_bak)  // 上一次是双soil humid传感器
		{
			if(g_sensors.soil_humid.dectected == 2)  //拔掉第l路，留第2路时,将第2路的cali值copy到第一路设置中
				g_sys_setting.sensor_set.cali_para[SOIL_HUMID] = g_sys_setting.sensor_set.cali_para_second[SOIL_HUMID];
		}
		soil_dualdetect_bak = g_sensors2.soil_humid.dectected;
	    break;	
		
    case 3: 
		i = 4; 
		static u8 co2_dualdetect_bak;
		detected =  sensor_co2_read(param1);
	    g_sensors.co2.dectected = detected & (0x7);
		detected >>= 3;
		g_sensors2.co2.dectected = detected & (0x7);
		
//		if(g_sensors.co2.dectected)
		{
		    g_sensors.co2.read_val = param1[0];
			refresh_sensors_data_after_calib(CO2);
		}
		if(!g_sensors.co2.dectected)
		{
			if(co2_auto_self_calib_flag)
			{
				co2_auto_self_calib_flag = false;
				sensor_self_cal.cal_info.cal_status = 3;
			}
		}
		if(g_sensors2.co2.dectected)
		{
		    g_sensors2.co2.read_val = param1[1];
			refresh_sensors2_data_after_calib(CO2);
		}
		else if(co2_dualdetect_bak)  // 上一次是双CO2传感器
		{
			if(g_sensors.co2.dectected == 2)  //拔掉第l路，留第2路时,将第2路的cali值copy到第一路设置中
				g_sys_setting.sensor_set.cali_para[CO2] = g_sys_setting.sensor_set.cali_para_second[CO2];
		}
		co2_dualdetect_bak = g_sensors2.co2.dectected;
		break;	
	
    case 4: 
		i = 5; 
		static u8 light_dualdetect_bak;
		detected =  sensor_light_read(param1);
	    g_sensors.light.dectected = detected & (0x7);
		detected >>= 3;
		g_sensors2.light.dectected = detected & (0x7);
		
		if(g_sensors.light.dectected)
		{
			g_sensors.light.read_val = param1[0];
			refresh_sensors_data_after_calib(LIGHT);
		}
		if(g_sensors2.light.dectected)
		{
			g_sensors2.light.read_val = param1[1];
			refresh_sensors2_data_after_calib(LIGHT);
		}
		else if(light_dualdetect_bak)  // 上一次是双LIGHT传感器
		{
			if(g_sensors.light.dectected == 2)  //拔掉第l路，留第2路时,将第2路的cali值copy到第一路设置中
				g_sys_setting.sensor_set.cali_para[LIGHT] = g_sys_setting.sensor_set.cali_para_second[LIGHT];
		}
		light_dualdetect_bak = g_sensors2.light.dectected;		
		//	if (g_sensors.light.read_val >= g_sys_setting.sensor_set.cali_para[LIGHT])	 //cali_para[LIGHT]:保存下发的光照强度临界值
	    break;	

	case 5:
		i = 6;
		static u8 ph_dualdetect_bak;
		static u8 ph_detected_bak;
		static u8 pwr_on_delay = 2;

		if(pwr_on_delay > 0)
			pwr_on_delay--;
		detected = sensor_PH_EC_read(param1, param2, param3, param4, param5);
		g_sensors.ph.dectected = detected & 0x07;
		detected >>= 3;
		g_sensors2.ph.dectected = detected & 0x07;
		
		g_sensors.watertemp_c.dectected = g_sensors.ph.dectected;
		g_sensors.watertemp_f.dectected = g_sensors.ph.dectected;
		g_sensors.ec_us.dectected = g_sensors.ph.dectected;
		g_sensors.ec_ms.dectected = g_sensors.ph.dectected;
		g_sensors.tds_ppm.dectected = g_sensors.ph.dectected;
		g_sensors.tds_ppt.dectected = g_sensors.ph.dectected;
		g_sensors2.watertemp_c.dectected = g_sensors2.ph.dectected;
		g_sensors2.watertemp_f.dectected = g_sensors2.ph.dectected;
		g_sensors2.ec_us.dectected = g_sensors2.ph.dectected;
		g_sensors2.ec_ms.dectected = g_sensors2.ph.dectected;
		g_sensors2.tds_ppm.dectected = g_sensors2.ph.dectected;
		g_sensors2.tds_ppt.dectected = g_sensors2.ph.dectected;
		if(g_sensors.ph.dectected)
		{
			g_sensors.ph.read_val = param1[0] * 10; 
			g_sensors.ph_mv.read_val = param5[0] * 10;   //ph mv读数取一位小数
			g_sensors.ec_ms.read_val = param2[0];    // 毫西
			g_sensors.ec_us.read_val = param2[0] * 1000;   // 微西
			if(g_sensors.ec_us.read_val > 20000)
				g_sensors.ec_us.read_val = 20000;
			g_sensors.tds_ppm.read_val = param3[0];   // 微克
			g_sensors.tds_ppt.read_val = param3[0] / 1000;  //ppt 毫克
			if(g_sensors.tds_ppt.read_val > 20000)
				g_sensors.tds_ppt.read_val = 20000;
			g_sensors.watertemp_c.read_val = param4[0]/10;
			g_sensors.watertemp_f.read_val = C_to_F(g_sensors.watertemp_c.read_val);
			refresh_sensors_data_after_calib(PH);
			if(is_temp_unit_f())
				refresh_sensors_data_after_calib(WATERTEMP_F);
			else
				refresh_sensors_data_after_calib(WATERTEMP_C);
			if(is_ec_unit_us())
				refresh_sensors_data_after_calib(EC_us);
			else
				refresh_sensors_data_after_calib(EC_ms);
			if(is_tds_unit_ppm())
				refresh_sensors_data_after_calib(TDS_ppm);
			else
				refresh_sensors_data_after_calib(TDS_ppt);	

			if(ph_detected_bak != g_sensors.ph.dectected && pwr_on_delay == 0)
			{
				if(!g_sensors2.ph.dectected && g_sensors.ph.dectected == 1)
					memset((u8 *)&sensor_self_cal_time[1], 0, sizeof(sensor_self_cal_time_t));
				if(!g_sensors2.ph.dectected && g_sensors.ph.dectected == 2)
					memset((u8 *)&sensor_self_cal_time[0], 0, sizeof(sensor_self_cal_time_t));
			}
		}
		else
		{	
			if(ph_auto_self_calib_flag)
			{
				ph_auto_self_calib_flag = false;
				ph_ec_cali_step = 0;
				sensor_self_cal.cal_info.cal_status = ph_cal_state[0].cal_status = 3;
			}
			else if(ec_auto_self_calib_flag)
			{
				ec_auto_self_calib_flag = false;
				ph_ec_cali_step = 0;
				sensor_self_cal.cal_info.cal_status = ec_cal_state[0].cal_status = 3;
			}
			if(ph_detected_bak != g_sensors.ph.dectected && pwr_on_delay == 0)
			{
				memset((u8 *)&sensor_self_cal_time[0], 0, sizeof(sensor_self_cal_time_t) * 2);
			}
		}
		if(g_sensors2.ph.dectected)
		{			
			g_sensors2.ph.read_val = param1[1] * 10;
			g_sensors2.ec_ms.read_val = param2[1];    // 毫西
			g_sensors2.ec_us.read_val = param2[1] * 1000;   // 微西
			if(g_sensors2.ec_us.read_val > 20000)
				g_sensors2.ec_us.read_val = 20000;
			g_sensors2.tds_ppm.read_val = param3[1];   // 微克
			g_sensors2.tds_ppt.read_val = param3[1] / 1000;  //ppt 毫克
			if(g_sensors2.tds_ppt.read_val > 20000)
				g_sensors2.tds_ppt.read_val = 20000;
			g_sensors2.watertemp_c.read_val = param4[1]/10;
			g_sensors2.watertemp_f.read_val = C_to_F(g_sensors2.watertemp_c.read_val);
			refresh_sensors2_data_after_calib(PH);
			if(is_temp_unit_f())
				refresh_sensors2_data_after_calib(WATERTEMP_F);
			else
				refresh_sensors2_data_after_calib(WATERTEMP_C);
			if(is_ec_unit_us())
				refresh_sensors2_data_after_calib(EC_us);
			else
				refresh_sensors2_data_after_calib(EC_ms);
			if(is_tds_unit_ppm())
				refresh_sensors2_data_after_calib(TDS_ppm);
			else
				refresh_sensors2_data_after_calib(TDS_ppt);	
		}
//		else if(ph_dualdetect_bak)  // 双PH传感器,拔掉1路PH传感器
//		{
//			if(g_sensors.ph.dectected == 2)  //拔掉第l路，留第2路时,将第2路的cali值copy到第一路设置中
//			{
//				for(u8 type=PH; type<=WATERTEMP_C; type++)
//					g_sys_setting.sensor_set.cali_para[type] = g_sys_setting.sensor_set.cali_para_second[type];
//			}
//		}
		ph_dualdetect_bak = g_sensors2.ph.dectected;
		ph_detected_bak = g_sensors.ph.dectected;
		break;
		
	case 6:
		i = 7;
		detected = sensor_water_level_read(param1);
		g_sensors.water_level.dectected = detected & 0x07;
		g_sensors2.water_level.dectected = (detected >> 3) & 0x07;
		if(g_sensors.water_level.dectected)
		{	
			g_sensors.water_level.real_val_float = param1[0];
			g_sensors.water_level.read_val = param1[0];
		#if ((_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX)))
			no_water_check();
		#endif
		}
		if(g_sensors2.water_level.dectected)
		{
			g_sensors2.water_level.real_val_float = param1[1];
			g_sensors2.water_level.read_val = param1[1];
		}
		break;

	case 7:
		i = 0;
		detected = sensor_leaf_read(param1);
		g_sensors.leaftemp_c.dectected = detected & 0x07;
		g_sensors2.leaftemp_c.dectected = (detected >> 3) & 0x07;		
		g_sensors.leaftemp_f.dectected = g_sensors.leaftemp_c.dectected;		
		g_sensors2.leaftemp_f.dectected = g_sensors2.leaftemp_c.dectected;
		if(g_sensors.leaftemp_c.dectected)
		{
			g_sensors.leaftemp_c.read_val = param1[0];
			g_sensors.leaftemp_f.read_val = C_to_F(g_sensors.leaftemp_c.read_val);
			refresh_sensors_data_after_calib(LEAFTEMP_C);
			refresh_sensors_data_after_calib(LEAFTEMP_F);
		}
		if(g_sensors2.leaftemp_c.dectected)
		{
			g_sensors2.leaftemp_c.read_val = param1[1];
			g_sensors2.leaftemp_f.read_val = C_to_F(g_sensors2.leaftemp_c.read_val);
			refresh_sensors2_data_after_calib(LEAFTEMP_C);
			refresh_sensors2_data_after_calib(LEAFTEMP_F);
		}
		
		check_sensors_value_valid();
		break;
		
	default: break;
	
	}
}

void auto_calibration_check(void)
{
	if (auto_self_calib_time)  // 校准超时判断	   3分钟倒计时
	{
		auto_self_calib_time--;
		if(auto_self_calib_time == 0)
		{
			auto_self_calib_retrytimes++;
			if(auto_self_calib_retrytimes >= 3)
				auto_self_calib_retrytimes = 0;
			if(ph_auto_self_calib_flag)
			{
				ph_auto_self_calib_flag = false;
				if(g_sensors.ph.dectected == 1)
				{
					if(auto_self_calib_retrytimes == 0)  //重复次数达到3次，校准取消
						ph_cal_state[0].cal_status = 3;  // 超时次数达到3次，校准取消
					else
						ph_cal_state[0].cal_status = 4;  // 超时失败
				}
				else if(g_sensors.ph.dectected == 2)
				{
					if(auto_self_calib_retrytimes == 0)  //重复次数达到3次，校准取消
						ph_cal_state[1].cal_status = 3;  // 超时次数达到3次，校准取消
					else
						ph_cal_state[1].cal_status = 4;  // 超时失败
				}
			}
			else if(ec_auto_self_calib_flag)
			{
				ec_auto_self_calib_flag = false;
				if(g_sensors.ph.dectected == 1)
				{
					if(auto_self_calib_retrytimes == 0)  //重复次数达到3次，校准取消
						ec_cal_state[0].cal_status = 3;
					else
						ec_cal_state[0].cal_status = 4;
				}
				else if(g_sensors.ph.dectected == 2)
				{
					if(auto_self_calib_retrytimes == 0)  //重复次数达到3次，校准取消
						ec_cal_state[1].cal_status = 3;
					else
						ec_cal_state[1].cal_status = 4;
				}
			}
			else if(co2_auto_self_calib_flag)
			{
				co2_auto_self_calib_flag = false;
				if(auto_self_calib_retrytimes == 0)  //重复次数达到3次，校准取消
					sensor_self_cal.cal_info.cal_status = 3;
				else
					sensor_self_cal.cal_info.cal_status = 4;
			}
		}
	}
}

/*
void ui_smooth_num(smooth_disp_t *disp_val, uint8_t max_dot_num, bool b_disp_err, void (*disp_num)(uint16_t), void (*disp_trends)(uint16_t), void (*disp_unit)(void))
{
    if(b_disp_err)
    {
        disp_num(ICON_DATA_ERR);
		disp_unit();
		disp_trends(ICON_TRENDS_NO);
		return;
    }

	smooth_data_for_disp(disp_val);
	
	disp_num(disp_val->current);
	disp_unit();
	disp_trends(disp_val->change_trends);
}


void ui_sensors(void)
{
    // the following sensor data area will display err '--' if relevant sensor is not dectected
    if(is_temp_unit_f())//(g_sys_settings.temp_unit_f)
	{
	    ui_smooth_num(&g_sensors.temp_f.disp_val, disp_data_temp_f, 0, false==g_sensors.temp_f.dectected);		
	    ui_smooth_num(&g_sensors.zonetemp_f.disp_val, disp_data_zonetemp_f, 0, false==g_sensors.zonetemp_f.dectected);
	    ui_smooth_num(&g_sensors.vpd.disp_val, disp_data_vpd, 2, false==g_sensors.vpd.dectected); // disp err if the sensor is not detected.
    }
	else
	{
	    ui_smooth_num(&g_sensors.temp_c.disp_val, disp_data_temp_c, 0, false==g_sensors.temp_c.dectected);		
	    ui_smooth_num(&g_sensors.zonetemp_c.disp_val, disp_data_zonetemp_c, 0, false==g_sensors.zonetemp_c.dectected);			
	    ui_smooth_num(&g_sensors.vpd.disp_val, disp_data_vpd, 2, false==g_sensors.vpd.dectected); // disp err if the sensor is not detected.
	}

    // the following sensor data area will display blank if relevant sensor is not dectected
    
	// co2
	if(g_sensors.co2.dectected) 
		ui_smooth_num(&g_sensors.co2.disp_val, disp_data_co2, 4, true);
	// light
	if(g_sensors.light.dectected)
		ui_smooth_num(&g_sensors.light.disp_val, disp_data_light, 4, true);
	// ph
	if(g_sensors.ph.dectected) 
		ui_smooth_num(&g_sensors.ph.disp_val, disp_data_ph, 4, true);
	// soil
	if(g_sensors.soil_humid.dectected) 
		ui_smooth_num(&g_sensors.soil_humid.disp_val, disp_data_soil_humid, 4, true);
	// leaf temp
	if(g_sensors.leaftemp_c.dectected && !g_sys_settings.temp_unit_f)
        ui_smooth_num(&g_sensors.leaftemp_c.disp_val, disp_data_leaftemp_f, 0, true);
	if(g_sensors.leaftemp_f.dectected && g_sys_settings.temp_unit_f) 
		ui_smooth_num(&g_sensors.leaftemp_f.disp_val, disp_data_leaftemp_f, 0, true);
	
}

*/



s16 float_round_up_to_integer(float raw_num)
{
    s16 lowest_num;
    s16 ret_num;

	ret_num = raw_num*100;
	lowest_num = ret_num % 10;
	ret_num -= lowest_num;
    if(lowest_num >= 5)
		ret_num += 10;

	return ret_num;
}

s16 to_s16x100(u8 raw_num)
{
    s16 num;
	
    num = raw_num;
	num *= 100;

	return num;
}

s16 to_s16x10(u8 raw_num)
{
    s16 num;
	
    num = raw_num;
	num *= 10;

	return num;
}

s16 get_real_zonetemp(bool is_unit_f)
{
	if (is_unit_f)
	    return	float_round_up_to_integer(g_sensors.zonetemp_f.real_val_float);
	else
		return float_round_up_to_integer(g_sensors.zonetemp_c.real_val_float);
}

s16 get_real_zonehumid(void)
{
	return float_round_up_to_integer(g_sensors.zone_humid.real_val_float);
}


s16 get_real_zonevpd(void)
{
	return g_sensors.zone_vpd.read_val/10;
}

s16 get_real_soil_humid(void)
{
    return g_sensors.soil_humid.disp_val.real_val;
}

s16 get_real_co2(void)
{
    return g_sensors.co2.disp_val.real_val;
}

s16 get_real_water(void)
{
	return g_sensors.water_level.read_val;
}

s16 get_real_ph(void)
{
	return float_round_up_to_integer(g_sensors.ph.real_val_float);
}




/*
****************************************************************************************************
*    函数名称：
*    功能说明： 实时的 温度 和 湿度 数据 转换为 4个字节
*    参   数：
*    返 回 值：
*
****************************************************************************************************
*/
void GetTempHumidByte(u8 * data)
{
    double temp, humid;
    s16 temp_1, humid_1;

    if (sensor.readError == 0)
    {
        if (is_temp_unit_f())
        {
            if (sensor.showTempF == sensor.ObjTempF)
            {
                temp = sensor.readTemp;
            }
            else
            {
                temp = sensor.showTempF;
                temp = F_to_C(temp);
            }
        }
        else
        {
            if (sensor.showTempC == sensor.ObjTempC)
            {
                temp = sensor.readTemp;
            }
            else
            {
                temp = sensor.showTempC;
            }
        }

        if (sensor.showHumid == sensor.ObjHumid)
        {
            humid = sensor.readHumid;
        }
        else
        {
            humid = sensor.showHumid;
        }
        //temp    = temp * 100;
        //humid   = humid * 100;

      
        //temp_1  = (temp);
        //humid_1 = (humid);
        temp_1 = float_round_up_to_integer(temp);
		humid_1 = float_round_up_to_integer(humid);

        data[0] = (temp_1 & 0xff00) >> 8;  // 温度
        data[1] = (temp_1 & 0x00ff);
        data[2] = (humid_1 & 0xff00) >> 8; // 湿度
        data[3] = (humid_1 & 0x00ff);
    }
    else
    {
        data[0] = 0x80;
        data[1] = 0x00;
        data[2] = 0x80;
        data[3] = 0x00;
    }
}

void GetVpdByte(u8 * data) // 将 Vpd 转换为 2个字节
{
    double temp;

    s16 temp_1;                // 16位有符号整形数

    if (sensor.readError == 0) // 温湿度数据 读取正常
    {
        temp    = sensor.vpdComputer / 10;

        // temp_1 = DoubleToShort(temp); //转换为 16位有符号整数型
        temp_1  = (temp);
        data[0] = (temp_1 & 0xff00) >> 8;
        data[1] = (temp_1 & 0x00ff);
    }
    else // 当 读取 温湿度数据失败时（即 外置传感器探头未插入），温湿度数据4个字节 置为 无效值
    {
        data[0] = 0x80;
        data[1] = 0x00;
    }
}


s16 GetRealTemp(bool is_unit_f)
{
	double temp;
	s16 temp_1;

	if (is_unit_f)
	{
		if (abs(sensor.showTempF - sensor.ObjTempF) < 2)
		{
		    temp = sensor.readTemp;
			temp = C_to_F(temp);
		}
		else
		{
			temp = sensor.showTempF;	
			//temp = F_to_C(temp);
		}
	}
	else
	{
		if (abs(sensor.showTempC - sensor.ObjTempC) < 2)
			temp = sensor.readTemp;
		else
			temp = sensor.showTempC;
	}
	
	//temp = temp * 100;
	//temp_1 = (temp);
	temp_1 = float_round_up_to_integer(temp);
	
	return temp_1;
	
}

s16 GetRealHumid(void)
{
	double humid;
	s16 humid_1;

	if (abs(sensor.showHumid - sensor.ObjHumid) < 2)
		humid = sensor.readHumid;
	else
		humid = sensor.showHumid;
	
	//humid = humid * 100;
	//humid_1 = (humid);
	humid_1 = float_round_up_to_integer(humid);

	return humid_1;
}

s16 GetRealVpd(void)
{	
	return sensor.vpdComputer / 10;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：  获取当前有多少个传感器数据块 以及 其对应的类型
*	参    数：flag=0,ec/tds各获取1个传感器数据块,  flag=1,ec/tds各获取2个传感器数据块
*	返 回 值：
****************************************************************************************************
*/
static void sensor_getDataBlockArg_from(sensors_t * src_sensors, sensor_datablock_t * sensor_datablock, bool flag)
{
    // temp/humid/vpd
    if (is_temp_unit_f())
    {    if (src_sensors->temp_f.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TEMP_F;}
    else
    {    if (src_sensors->temp_c.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TEMP_C;}
	
    if (src_sensors->humid.dectected) sensor_datablock->type[sensor_datablock->cnt++] = HUMID;
    if (src_sensors->vpd.dectected) sensor_datablock->type[sensor_datablock->cnt++] = VPD;
	
    // zone temp/humid/vpd
    if (is_temp_unit_f())
    {    if (src_sensors->zonetemp_f.dectected) sensor_datablock->type[sensor_datablock->cnt++] = ZONE_TEMP_F;}
    else
    {    if(src_sensors->zonetemp_c.dectected) sensor_datablock->type[sensor_datablock->cnt++] = ZONE_TEMP_C;}
	
    if (src_sensors->zone_humid.dectected) sensor_datablock->type[sensor_datablock->cnt++] = ZONE_HUMID;
    if (src_sensors->zone_vpd.dectected) sensor_datablock->type[sensor_datablock->cnt++] = ZONE_VPD;

	// solid humid
	if (src_sensors->soil_humid.dectected) sensor_datablock->type[sensor_datablock->cnt++] = SOIL_HUMID;

	// co2 + light
	if (src_sensors->co2.dectected) sensor_datablock->type[sensor_datablock->cnt++] = CO2;
	if (src_sensors->light.dectected) sensor_datablock->type[sensor_datablock->cnt++] = LIGHT;
	
    // clang-format on
	if (src_sensors->ph.dectected) sensor_datablock->type[sensor_datablock->cnt++] = PH;
	if(flag)
	{
		if (src_sensors->ec_us.dectected) sensor_datablock->type[sensor_datablock->cnt++] = EC_us;
		if (src_sensors->ec_ms.dectected) sensor_datablock->type[sensor_datablock->cnt++] = EC_ms;
		if (src_sensors->tds_ppm.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TDS_ppm;
		if (src_sensors->tds_ppt.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TDS_ppt;
	}
	else
	{
		if(is_tds_unit_ppm())
	    {   if (src_sensors->tds_ppm.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TDS_ppm;}
		else
		{   if (src_sensors->tds_ppt.dectected) sensor_datablock->type[sensor_datablock->cnt++] = TDS_ppt;}
		if(is_ec_unit_us())
		{   if (src_sensors->ec_us.dectected) sensor_datablock->type[sensor_datablock->cnt++] = EC_us;}
		else
		{   if (src_sensors->ec_ms.dectected) sensor_datablock->type[sensor_datablock->cnt++] = EC_ms;}
	}
   	if (is_temp_unit_f())
	{	if(src_sensors->watertemp_f.dectected) sensor_datablock->type[sensor_datablock->cnt++] = WATERTEMP_F;}
	else
	{   if(src_sensors->watertemp_c.dectected) sensor_datablock->type[sensor_datablock->cnt++] = WATERTEMP_C;}

	//water level	
	if (src_sensors->water_level.dectected) sensor_datablock->type[sensor_datablock->cnt++] = WATER_LEVEL;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：  获取当前有多少个传感器数据块 以及 其对应的类型
*	参    数：flag=0,ec/tds各获取1个传感器数据块,  flag=1,ec/tds各获取2个传感器数据块
*	返 回 值：
****************************************************************************************************
*/
void sensor_getDataBlockArg(sensor_datablock_t * sensor_datablock, bool flag)
{
    u8 i,j;
    sensors_t *src_sensor[SENSOR_SRC_CNT];

    // clang-format off
    sensor_datablock->cnt = 0;

	src_sensor[0] = &g_sensors;
	src_sensor[1] = &g_sensors2;
    j = 0;
    for(i=0; i<SENSOR_SRC_CNT; i++)
	{
	    sensor_getDataBlockArg_from(src_sensor[i], sensor_datablock, flag);
		for(j=j; j<sensor_datablock->cnt; j++)
			sensor_datablock->from[j] = i;
    }
}

s16 get_sensor_val(u8 sensor_type)
{
    switch (sensor_type)
    {
        case TEMP_F : return GetRealTemp(is_temp_unit_f());
		case TEMP_C : return GetRealTemp(is_temp_unit_f());
		case HUMID  : return GetRealHumid();
		case VPD    : return GetRealVpd();

		case LEAFTEMP_C :
		case LEAFTEMP_F :
		case ZONE_TEMP_F :
		case ZONE_TEMP_C : 		
		case ZONE_HUMID  :
		case SOIL_HUMID  :  
		case WATERTEMP_F : 
		case WATERTEMP_C :  
		case EC_ms   :
		case TDS_ppt :		return g_sensor[sensor_type].real_val_float*100;

		case ZONE_VPD    :  return g_sensor[sensor_type].read_val/10;

		case PH      :	return g_sensor[sensor_type].real_val_float*10;
        case LIGHT:  return g_sensor[sensor_type].read_val*10;
		
		case WATER_LEVEL :
		case CO2     :  
	   	case EC_us   :
		case TDS_ppm :	    return g_sensor[sensor_type].real_val_float;

		default: return 0;
    }
}

u8 get_highlow_precision_multi(u8 sensor_type)
{
    s16 multi = 1;
	
    switch (sensor_type)
    {
        case TEMP_F : 
		case TEMP_C : 
		case HUMID  :
		case LEAFTEMP_C :
		case LEAFTEMP_F :
		case ZONE_TEMP_F :
		case ZONE_TEMP_C : 		
		case ZONE_HUMID  :
		case SOIL_HUMID  :  
		case WATERTEMP_F : 
		case WATERTEMP_C :  
		case EC_ms   :
		case TDS_ppt :		multi = 100; break;	

		case PH      :
		case ZONE_VPD    :	
		case VPD    :       multi = 10; break;

		case WATER_LEVEL :
		case CO2     :  	   
		case LIGHT   :  
		case EC_us   :
		case TDS_ppm :		multi = 1; break;

		default: return 1;
    }

	return multi;
}


u8 get_transbuff_precision_multi(u8 sensor_type)
{
    s16 multi = 1;
	
    switch (sensor_type)
    {
        case TEMP_F : 
		case TEMP_C : 
		case HUMID  :
		case LEAFTEMP_C :
		case LEAFTEMP_F :
		case ZONE_TEMP_F :
		case ZONE_TEMP_C : 		
		case ZONE_HUMID  :
		case SOIL_HUMID  :  
		case WATERTEMP_F : 
		case WATERTEMP_C :   
		case EC_ms   :
		case TDS_ppt :		multi = 100; 	break;			

		case PH      :
		case TDS_ppm :
		case EC_us   :
		case ZONE_VPD :	
		case VPD    : 
		case CO2     :  	 multi = 10; break;
		
		case WATER_LEVEL :	    
		case LIGHT   :  	multi = 1; break;

		default: return 1;
    }

	return multi;
}



/*
****************************************************************************************************
*	函数名称：
*	功能说明：  填充 某个类型数据块的 相关数据
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
static void sensor_fillDataBlockValue_from(sensors_t * src_sensors, u8 sensor_type, sensor_data_t * p)
{
    switch (sensor_type)
    {
        case TEMP_F :
            // if (g_sensors.temp_f.dectected)
            {
                p->sensor_type         = TEMP_F;

                p->state.sensor_port   = src_sensors->temp_f.dectected;
//                p->state.change_trends = src_sensors->temp_f.disp_val.change_trends;
				p->state.change_trends = change_temp.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;
				
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(src_sensors->zonetemp_f.disp_val.current_val*100);
					else
	                	p->disp_val            = _s16_(sensor.showTempF * 100);
				}
				else
				{
					if(is_zone_pos_switched())
//						p->disp_val            = _s16_(src_sensors->zonetemp_f.real_val_float*100);
						p->disp_val            = _s16_(get_real_zonetemp(is_temp_unit_f()));
					else
	                	p->disp_val            = _s16_(GetRealTemp(is_temp_unit_f())); // g_sensors.temp_f.disp_val.current;
				}
            }
            break;

        case TEMP_C :
            // if (g_sensors.temp_c.dectected)
            {
                p->sensor_type         = TEMP_C;

                p->state.sensor_port   = src_sensors->temp_c.dectected;
//                p->state.change_trends = src_sensors->temp_c.disp_val.change_trends;
				p->state.change_trends = change_temp.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(src_sensors->zonetemp_c.disp_val.current_val*100);
					else
	                	p->disp_val            = _s16_(sensor.showTempC * 100);
				}
				else
				{
					if(is_zone_pos_switched())
//						p->disp_val            = _s16_(src_sensors->zonetemp_c.real_val_float*100);
						p->disp_val            = _s16_(get_real_zonetemp(is_temp_unit_f()));
					else
	              	    p->disp_val            = _s16_(GetRealTemp(is_temp_unit_f())); // g_sensors.temp_c.disp_val.current;
				}
            }
            break;
        
        case HUMID :
            // if (g_sensors.humid.dectected)
            {
                p->sensor_type         = HUMID;

                p->state.sensor_port   = src_sensors->humid.dectected;
//                p->state.change_trends = src_sensors->humid.disp_val.change_trends;
				p->state.change_trends = change_humid.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = 0;
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(src_sensors->zone_humid.disp_val.current_val*100);
					else
	                	p->disp_val            = _s16_(sensor.showHumid * 100);
				}
				else
				{
					if(is_zone_pos_switched())
//						p->disp_val            = _s16_((s16)(src_sensors->zone_humid.real_val_float*100));
						p->disp_val            = _s16_(get_real_zonehumid());
					else
	                	p->disp_val            = _s16_(GetRealHumid()); // g_sensors.humid.disp_val.current;
				}
            }
            break;

        case VPD :
            // if (g_sensors.vpd.dectected)
            {
                p->sensor_type         = VPD;

                p->state.sensor_port   = src_sensors->vpd.dectected;
//                p->state.change_trends = src_sensors->vpd.disp_val.change_trends;
				p->state.change_trends = change_vpd.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = 0;
				if(is_zone_pos_switched())
					p->disp_val            = _s16_((s16)(src_sensors->zone_vpd.read_val/10));
				else
                	p->disp_val            = _s16_(GetRealVpd()); 
            }
            break;

        case ZONE_TEMP_F :
            // if (g_sensors.zonetemp_f.dectected)
            {
                p->sensor_type         = ZONE_TEMP_F;

                p->state.sensor_port   = src_sensors->zonetemp_f.dectected;
//                p->state.change_trends = src_sensors->zonetemp_f.disp_val.change_trends;
				p->state.change_trends = change_zone_temp.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(sensor.showTempF*100);
					else
	                	p->disp_val            = _s16_(src_sensors->zonetemp_f.disp_val.current_val*100);
				}
				else
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(GetRealTemp(is_temp_unit_f()));
					else
//						p->disp_val            = _s16_(src_sensors->zonetemp_f.real_val_float*100);  //fixbug-v40: zone的F单位显示偏差问题
						p->disp_val            = _s16_(get_real_zonetemp(is_temp_unit_f()));
				}
            }
            break;

        case ZONE_TEMP_C :
            // if (g_sensors.zonetemp_c.dectected)
            {
                p->sensor_type         = ZONE_TEMP_C;

                p->state.sensor_port   = src_sensors->zonetemp_c.dectected;
//                p->state.change_trends = src_sensors->zonetemp_c.disp_val.change_trends;
				p->state.change_trends = change_zone_temp.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(sensor.showTempC*100);
					else
	                	p->disp_val            = _s16_(src_sensors->zonetemp_c.disp_val.current_val*100);
				}
				else
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(GetRealTemp(is_temp_unit_f()));
					else
//	                	p->disp_val            = _s16_(src_sensors->zonetemp_c.real_val_float*100);
						p->disp_val            = _s16_(get_real_zonetemp(is_temp_unit_f()));
				}
            }
            break;

        case ZONE_HUMID :
            // if (g_sensors.zone_humid.dectected)
            {
                p->sensor_type         = ZONE_HUMID;

                p->state.sensor_port   = src_sensors->zone_humid.dectected;
//                p->state.change_trends = src_sensors->zone_humid.disp_val.change_trends;
				p->state.change_trends = change_zone_humid.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = 0;
				if(switch_zone_pos_delay_time)
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(sensor.showHumid*100);
					else
	                	p->disp_val            = _s16_(src_sensors->zone_humid.disp_val.current_val*100);
				}
				else
				{
					if(is_zone_pos_switched())
						p->disp_val            = _s16_(GetRealHumid());
					else
//						p->disp_val            = _s16_((s16)(src_sensors->zone_humid.real_val_float*100));
						p->disp_val            = _s16_(get_real_zonehumid());
				}
            }
            break;

        case ZONE_VPD :
            // if (g_sensors.zonetemp_f.dectected)
            {
                p->sensor_type         = ZONE_VPD;

                p->state.sensor_port   = src_sensors->zone_vpd.dectected;
//                p->state.change_trends = src_sensors->zone_vpd.disp_val.change_trends;
				p->state.change_trends = 0;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = 0;
				if(is_zone_pos_switched())
					p->disp_val            = _s16_(GetRealVpd());
				else
                	p->disp_val            = _s16_((s16)(src_sensors->zone_vpd.read_val/10));
            }
            break;

		case SOIL_HUMID :
            {
                p->sensor_type         = SOIL_HUMID;

                p->state.sensor_port   = src_sensors->soil_humid.dectected;
//                p->state.change_trends = src_sensors->soil_humid.disp_val.change_trends;
				p->state.change_trends = change_humid.change;
                p->state.precision     = PRECISION_0_0_1;
                p->state.unit          = 0;

                p->disp_val            = _s16_((s16)(src_sensors->soil_humid.real_val_float*100));
            }
            break;

		case CO2 :
            {
                p->sensor_type         = CO2;

                p->state.sensor_port   = src_sensors->co2.dectected;
//                p->state.change_trends = src_sensors->co2.disp_val.change_trends;
				p->state.change_trends = change_co2.change;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

                p->disp_val            = _s16_((s16)(src_sensors->co2.real_val_float));
            }
            break;

		case LIGHT :
            {
                p->sensor_type         = LIGHT;

                p->state.sensor_port   = src_sensors->light.dectected;
                p->state.change_trends = change_light.change;
                p->state.precision     = PRECISION_0_1;    //XX.X%
                p->state.unit          = 0;

                p->disp_val            = _s16_((s16)(src_sensors->light.real_val_float * 10));
            }
            break;
		
		case PH :
            {
                p->sensor_type         = PH;

                p->state.sensor_port   = src_sensors->ph.dectected;
//                p->state.change_trends = src_sensors->ph.disp_val.change_trends;
				p->state.change_trends = change_ph.change;
				p->state.precision 		= PRECISION_0_0_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_(get_real_ph());
				p ->disp_val			= _s16_(src_sensors->ph.real_val_float*10);
            }
            break;

		case EC_us :
            {
                p->sensor_type         = EC_us;

                p->state.sensor_port   = src_sensors->ec_us.dectected;
//                p->state.change_trends = src_sensors->ec.disp_val.change_trends;
				p->state.change_trends = change_ec_tds.change;
				p->state.unit		   = g_sys_setting.sensor_set.unit.EC;
				if(src_sensors->ec_us.real_val_float < 100)
				{
					p->state.precision     = PRECISION_0_0_1;
                    p->disp_val            = _s16_((s16)(src_sensors->ec_us.real_val_float * 100));
				}
				else if(src_sensors->ec_us.real_val_float < 1000)
				{
					p->state.precision     = PRECISION_0_1;
                    p->disp_val            = _s16_((s16)(src_sensors->ec_us.real_val_float * 10));
				}	
				else
				{
              		p->state.precision     = PRECISION_1;
					p->disp_val            = _s16_((s16)(src_sensors->ec_us.real_val_float));
				}
            }
            break;
		
		case EC_ms :
            {
                p->sensor_type         = EC_ms;

                p->state.sensor_port   = src_sensors->ec_ms.dectected;
//                p->state.change_trends = src_sensors->ec.disp_val.change_trends;
				p->state.change_trends = change_ec_tds.change;				
                p->state.unit          = g_sys_setting.sensor_set.unit.EC;
				if(src_sensors->ec_ms.real_val_float < 100)
				{
	                p->state.precision     = PRECISION_0_0_1;
	                p->disp_val            = _s16_((s16)(src_sensors->ec_ms.real_val_float*100));
				}
				else if(src_sensors->ec_ms.real_val_float < 1000)
				{
					p->state.precision     = PRECISION_0_1;
	                p->disp_val            = _s16_((s16)(src_sensors->ec_ms.real_val_float*10));
				}
				else
				{
              		p->state.precision     = PRECISION_1;
					p->disp_val            = _s16_((s16)(src_sensors->ec_ms.real_val_float));
				}
            }
            break;

		case TDS_ppm :
            {
                p->sensor_type         = TDS_ppm;

                p->state.sensor_port   = src_sensors->tds_ppm.dectected;
//                p->state.change_trends = src_sensors->tds.disp_val.change_trends;
				p->state.change_trends = change_ec_tds.change;
				p->state.unit          = g_sys_setting.sensor_set.unit.TDS;
				if(src_sensors->tds_ppm.real_val_float < 100)
				{
					p->state.precision     = PRECISION_0_0_1;
					p->disp_val            = _s16_((s16)(src_sensors->tds_ppm.real_val_float * 100));
				}
				else if(src_sensors->tds_ppm.real_val_float < 1000)
				{
					p->state.precision     = PRECISION_0_1;
					p->disp_val            = _s16_((s16)(src_sensors->tds_ppm.real_val_float * 10));
				}
				else
				{
	                p->state.precision     = PRECISION_1;
					p->disp_val            = _s16_((s16)(src_sensors->tds_ppm.real_val_float));
				}
            }
            break;

		case TDS_ppt :
            {
                p->sensor_type         = TDS_ppt;

                p->state.sensor_port   = src_sensors->tds_ppt.dectected;
//                p->state.change_trends = src_sensors->tds.disp_val.change_trends;
				p->state.change_trends = change_ec_tds.change;				
                p->state.unit          = g_sys_setting.sensor_set.unit.TDS;
				if(src_sensors->tds_ppt.real_val_float < 100)
				{
	                p->state.precision     = PRECISION_0_0_1;
	                p->disp_val            = _s16_((s16)(src_sensors->tds_ppt.real_val_float * 100));
				}
				else if(src_sensors->tds_ppt.real_val_float < 1000)
				{
					p->state.precision     = PRECISION_0_1;
	                p->disp_val            = _s16_((s16)(src_sensors->tds_ppt.real_val_float * 10));
				}
				else
				{
					p->state.precision     = PRECISION_1;
	                p->disp_val            = _s16_((s16)(src_sensors->tds_ppt.real_val_float));
				}
            }
            break;
	
		case WATERTEMP_F:
			{
				p->sensor_type			= WATERTEMP_F;
				
				p->state.sensor_port	= src_sensors->watertemp_f.dectected;
//				p->state.change_trends	= src_sensors->watertemp_f.disp_val.change_trends;
				p->state.change_trends	= change_water_temp.change;
				p->state.precision		= PRECISION_0_0_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.temp;	  // 0

				p->disp_val			= _s16_((s16)(src_sensors->watertemp_f.real_val_float*100));
			}
			break;

		case WATERTEMP_C:
			{
				p->sensor_type			= WATERTEMP_C;
				
				p->state.sensor_port	= src_sensors->watertemp_c.dectected;
//				p->state.change_trends	= src_sensors->watertemp_c.disp_val.change_trends;
				p->state.change_trends	= change_water_temp.change;
				p->state.precision		= PRECISION_0_0_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.temp;	// 1
				
				p->disp_val			= _s16_((s16)(src_sensors->watertemp_c.real_val_float*100));
			}
			break;

		case WATER_LEVEL :
            {
                p->sensor_type         = WATER_LEVEL;

                p->state.sensor_port   = src_sensors->water_level.dectected;
				p->state.change_trends = 0; //change_humid.change;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

                p->disp_val            = _s16_((s16)(src_sensors->water_level.read_val));
            }
            break;
            // TODO
            // 。。。 待完成
    }
}

void sensor_fillDataBlockValue(u8 from, u8 sensor_type, sensor_data_t * p)
{
    sensors_t *src_sensors;

	if(0 == from)
	    src_sensors = &g_sensors;
	if(1 == from)
	    src_sensors = &g_sensors2;

     sensor_fillDataBlockValue_from(src_sensors, sensor_type, p);
}

void sensor_cali_fillDataBlockValue(u8 from, u8 sensor_type, app_sensor_data_t * p)
{
	sensors_t *src_sensors = NULL;
	s16 *cali = NULL;

	if (from == 0)
	{
		src_sensors = (sensors_t *)&g_sensors;
		cali = &g_sys_setting.sensor_set.cali_para[sensor_type];
	}
	else if(from == 1)
	{
		src_sensors = (sensors_t *)&g_sensors2;
		cali = &g_sys_setting.sensor_set.cali_para_second[sensor_type];
	}
	p->disp_val = _s16_(*cali);
	
    switch (sensor_type)
    {
        case TEMP_F :
            // if (g_sensors.temp_f.dectected)
            {
                p->sensor_type         = TEMP_F;

                p->state.sensor_port   = src_sensors->temp_f.dectected;
                p->state.change_trends = src_sensors->temp_f.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;

//                p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[TEMP_F]); // g_sensors.temp_f.disp_val.current;
            }
            break;

        case TEMP_C :
            // if (g_sensors.temp_c.dectected)
            {
                p->sensor_type         = TEMP_C;

                p->state.sensor_port   = src_sensors->temp_c.dectected;
                p->state.change_trends = src_sensors->temp_c.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;

//                p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[TEMP_C]); // g_sensors.temp_c.disp_val.current;
            }
            break;

        case HUMID :
            // if (g_sensors.humid.dectected)
            {
                p->sensor_type         = HUMID;

                p->state.sensor_port   = src_sensors->humid.dectected;
                p->state.change_trends = src_sensors->humid.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[HUMID]); // g_sensors.humid.disp_val.current;
            }
            break;

        case VPD :
            // if (g_sensors.vpd.dectected)
            {
                p->sensor_type         = VPD;

                p->state.sensor_port   = src_sensors->vpd.dectected;
                p->state.change_trends = src_sensors->vpd.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;
				
//                p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[VPD]); 
            }
            break;

        case ZONE_TEMP_F :
            // if (g_sensors.zonetemp_f.dectected)
            {
                p->sensor_type         = ZONE_TEMP_F;

                p->state.sensor_port   = 7;//g_sensors.zonetemp_f.dectected;
                p->state.change_trends = src_sensors->zonetemp_f.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;

//				p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[ZONE_TEMP_F]);  //fixbug-v40: zone0Š40ƒ604F0‘20‚30ƒ10Š10¹10‚30Œ40ƒ40—60Š4¡è0”60‘2010‚50‘2¡¦0—5¨¦0ƒ30—5¨¦0¸70ƒ4
            }
            break;

        case ZONE_TEMP_C :
            // if (g_sensors.zonetemp_c.dectected)
            {
                p->sensor_type         = ZONE_TEMP_C;

                p->state.sensor_port   = 7;//g_sensors.zonetemp_c.dectected;
                p->state.change_trends = src_sensors->zonetemp_c.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = g_sys_setting.sensor_set.unit.temp;

//                p->disp_val            = _s16_(g_sys_setting.sensor_set.cali_para[ZONE_TEMP_C]);
            }
            break;

        case ZONE_HUMID :
            // if (g_sensors.zone_humid.dectected)
            {
                p->sensor_type         = ZONE_HUMID;

                p->state.sensor_port   = 7;//g_sensors.zone_humid.dectected;
                p->state.change_trends = src_sensors->zone_humid.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//				p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[ZONE_HUMID]));
            }
            break;

        case ZONE_VPD :
            // if (g_sensors.zonetemp_f.dectected)
            {
                p->sensor_type         = ZONE_VPD;

                p->state.sensor_port   = 7;//g_sensors.zone_vpd.dectected;
                p->state.change_trends = src_sensors->zone_vpd.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[ZONE_VPD]));
            }
            break;

		case SOIL_HUMID :
            {
                p->sensor_type         = SOIL_HUMID;

                p->state.sensor_port   = src_sensors->soil_humid.dectected;
                p->state.change_trends = src_sensors->soil_humid.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[SOIL_HUMID]));
            }
            break;

		case CO2 :
            {
                p->sensor_type         = CO2;

                p->state.sensor_port   = src_sensors->co2.dectected;
                p->state.change_trends = src_sensors->co2.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[CO2]));
            }
            break;

		case LIGHT :
            {
                p->sensor_type         = LIGHT;

                p->state.sensor_port   = src_sensors->light.dectected;
                p->state.change_trends = src_sensors->light.disp_val.change_trends;
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[LIGHT]));
            }
            break;

		case PH:
			{
				p->sensor_type			= PH;
				
				p->state.sensor_port	= src_sensors->ph.dectected;
				p->state.change_trends	= src_sensors->ph.disp_val.change_trends;
				p->state.precision		= PRECISION_0_1;
				p->state.unit			= 0;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[PH]));
			}
			break;

		case EC_us:
			{
				p->sensor_type			= EC_us;
				
				p->state.sensor_port	= src_sensors->ec_us.dectected;
				p->state.change_trends	= src_sensors->ec_us.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.EC;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[EC_us]));
			}
			break;

		case EC_ms:
			{
				p->sensor_type			= EC_ms;
				
				p->state.sensor_port	= src_sensors->ec_ms.dectected;
				p->state.change_trends	= src_sensors->ec_ms.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.EC;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[EC_ms]));
			}
			break;

		case TDS_ppm:
			{
				p->sensor_type			= TDS_ppm;
				
				p->state.sensor_port	= src_sensors->tds_ppm.dectected;
				p->state.change_trends	= src_sensors->tds_ppm.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.TDS;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[TDS_ppm]));
			}
			break;

		case TDS_ppt:
			{
				p->sensor_type			= TDS_ppt;
				
				p->state.sensor_port	= src_sensors->tds_ppt.dectected;
				p->state.change_trends	= src_sensors->tds_ppt.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.TDS;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[TDS_ppt]));
			}
			break;

		case WATERTEMP_F:
			{
				p->sensor_type			= WATERTEMP_F;
				
				p->state.sensor_port	= src_sensors->watertemp_f.dectected;
				p->state.change_trends	= src_sensors->watertemp_f.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.temp;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[WATERTEMP_F]));
			}
			break;

		case WATERTEMP_C:
			{
				p->sensor_type			= WATERTEMP_C;
				
				p->state.sensor_port	= src_sensors->watertemp_c.dectected;
				p->state.change_trends	= src_sensors->watertemp_c.disp_val.change_trends;
				p->state.precision		= PRECISION_1;
				p->state.unit			= g_sys_setting.sensor_set.unit.temp;
				
//				p->disp_val				= _s16_((s16)(g_sys_setting.sensor_set.cali_para[WATERTEMP_C]));
			}
			break;

		case WATER_LEVEL :
            {
                p->sensor_type         = WATER_LEVEL;

                p->state.sensor_port   = src_sensors->water_level.dectected;
                p->state.change_trends = src_sensors->water_level.disp_val.change_trends;  // 0
                p->state.precision     = PRECISION_1;
                p->state.unit          = 0;

//                p->disp_val            = _s16_((s16)(g_sys_setting.sensor_set.cali_para[WATER_LEVEL]));
            }
            break;
    }
}


