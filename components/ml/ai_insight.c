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

#include "ai_insight.h"
#include "ml_rule.h"


#define AI_INSIGHT_DEBUG_EN

#define TAG "ai insight"

#ifdef AI_INSIGHT_DEBUG_EN
#define ai_log(format, ...)		ESP_LOGI(TAG, format, ##__VA_ARGS__)
#else
#define ai_log(format, ...)	
#endif

ai_insight_com_st ai_insight_com, ai_insight_com_bak;
ai_insight_run_data_st ai_insight_run_data;
ai_insight_out_st ai_insight_out;
bool sun_light_running_flag = false;

/* 高温高湿提示 */
static void clean_ai_insights(void)
{
	memset((u8 *)&ai_insight_com, 0, sizeof(ai_insight_com_st));
	memset((u8 *)&ai_insight_com_bak, 0, sizeof(ai_insight_com_st));
	memset((u8 *)&ai_insight_run_data, 0, sizeof(ai_insight_run_data_st));
	memset((u8 *)&ai_insight_out, 0, sizeof(ai_insight_out));
}

static void updateHighStartTime(const Time_Typedef *currentTime, uint32_t *high_start_time) {
	extern bool ai_change_need_sync;
    // Calculate the total seconds from the beginning of the year
	uint32_t cur_utc = 0;
	if( currentTime != NULL){
		cur_utc = rtc_to_real_utc(*currentTime);
	}
	if( cur_utc != *high_start_time ){
		*high_start_time = cur_utc;
		ai_change_need_sync = true;
	}
}

bool get_aiinsight_be_reset_flag(uint8_t insight_id)
{
	bool ret = 0;
	if( insight_id < 32 ){
		ret = ( (ai_insight_out.app_reset_action_bit&(1<<insight_id)) != 0 );
	}
	return ret;
}

static void set_aiinsight_be_reset_flag(uint8_t insight_id,bool app_clean)
{
	if( app_clean != get_aiinsight_be_reset_flag(insight_id) ){
		ai_insight_out.app_reset_action_bit = (ai_insight_out.app_reset_action_bit&~(1<<insight_id));
		ai_insight_out.app_reset_action_bit |= ((app_clean!=0)<<insight_id);
		if( app_clean ){
			ESP_LOGI(TAG,"insight[%d] app reset",insight_id);
		}else{
			ESP_LOGI(TAG,"insight[%d] app clean",insight_id);
		}
	}
}

//	单位 t-1 h-1 v-0.1
bool env_data_is_in_target_range(const ai_setting_t *ai_setting,uint8_t env,s16 env_data)
{
	bool ret = 0;
	switch( env ){
		case ENV_TEMP:
			if( ai_setting->ai_mode_sel_bits.temp_en ){
				s16 max = (is_temp_unit_f()? ai_setting->autoMode.targetTemp_F_max:ai_setting->autoMode.targetTemp_C_max );
				if(  max > env_data ){
					ret = 1;
				}
			}
			break;

		case ENV_HUMID:
			if( ai_setting->ai_mode_sel_bits.humid_en ){
				s16 max = ai_setting->autoMode.targetHumid_max;
				if(  max > env_data ){
					ret = 1;
				}
			}
			break;

		case ENV_VPD:
			if( ai_setting->ai_mode_sel_bits.vpd_en ){
				s16 max = ai_setting->vpdMode.highVpd;
				if(  max > env_data ){
					ret = 1;
				}
			}
			break;
	}
	return ret;
}

//  ------------------------- sensor -------------------------
static void ai_sensor_data_abnormal_alarm_rule(Time_Typedef *tm, ai_setting_t *ai_setting, ai_insight_run_data_st *running_data, int16_t* sensor_list)
{
	if(ai_setting ->is_ai_deleted  == 1)
		return;
    //单位 0.01
	if(sensor_list[HUMID] > SENSOR_HIGH_HUMID_WARN*100)   // on high humid , set clip fan level to stronger
	{
		if( running_data->abnormal.bit.high_humid_flag == 0 ){
			running_data->abnormal.bit.high_humid_flag = 1;
			updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_HumidityWarning]) );
			ai_log("%s: high humid!! ", __func__);
		}
	}else{
		if( running_data->abnormal.bit.high_humid_flag == 1 ){
			running_data->abnormal.bit.high_humid_flag = 0;
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_HumidityWarning]) );
			ai_log("%s: high humid cancel!! ", __func__);
		}
	}

	//单位0.01
	if( (is_temp_unit_f()?sensor_list[TEMP_F]:sensor_list[TEMP_C]) > (is_temp_unit_f()?SENSOR_HIGH_TEMP_F:SENSOR_HIGH_TEMP_C)*100 )   // on high humid , set clip fan level to stronger
	{
		if( running_data->abnormal.bit.high_temp_flag == 0 ){
			running_data->abnormal.bit.high_temp_flag = 1;
			updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_TemperatureWarning]) );
			ai_log("%s: high temp!! ", __func__);
		}
	}else{
		if( running_data->abnormal.bit.high_temp_flag == 1){
			running_data->abnormal.bit.high_temp_flag = 0;
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_TemperatureWarning]) );
			ai_log("%s: high temp cancel!! ", __func__);
		}
	}

	//CO2 单位 1
	if(ai_get_sensor_is_selected(ai_setting, CO2)){
		if( sensor_list[CO2] > 0 && sensor_list[CO2] < 200 )
		{
			if(running_data->abnormal.bit.low_co2_flag == 0){
				running_data->abnormal.bit.low_co2_flag = 1;
				updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_LowCo2Warning]));
				ai_log("%s: low co2 warning!! ", __func__);
			}
		}
		else{
			if(running_data->abnormal.bit.low_co2_flag == 1){
				running_data->abnormal.bit.low_co2_flag = 0;
				updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_LowCo2Warning]));
				ai_log("%s: low co2 warning cancel!! ", __func__);
				set_aiinsight_be_reset_flag(INSIGHT_LowCo2Warning, 0);
			}
		}
		if( sensor_list[CO2] >= 5000){
			uint8_t co2_regulator_flag = 0;
			for(uint8_t i=1; i<PORT_CNT; i++) {
				if(!is_ai_port(i))
					continue;
				if(get_using_devtype(i) != loadType_co2_generator)
					continue;
				co2_regulator_flag = 1;
			}
			if(!co2_regulator_flag)
			{				
				if(running_data->abnormal.bit.high_co2_flag == 0){
					running_data->abnormal.bit.high_co2_flag = 1;
					updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_HighCo2Warning]));
					ai_log("%s: high co2 warning!! ", __func__);
				}
			}
			else{
				if(running_data->abnormal.bit.high_co2_regulator_paused_flag == 0){
					running_data->abnormal.bit.high_co2_regulator_paused_flag = 1;
					updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_Highco2RegulatorPaused]));
					ai_log("%s: co2 regulator stop!!!! ", __func__);
					set_aiinsight_be_reset_flag(INSIGHT_Highco2RegulatorPaused, 0);
				}
			}
		}
		else{
			if(ai_insight_com.utc[INSIGHT_HighCo2Warning] != 0){
				updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_HighCo2Warning]));
				set_aiinsight_be_reset_flag(INSIGHT_HighCo2Warning, 0);
			}				
			running_data->abnormal.bit.high_co2_flag = 0;
			if(ai_insight_com.utc[INSIGHT_Highco2RegulatorPaused] != 0){
				updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_Highco2RegulatorPaused]));				
				set_aiinsight_be_reset_flag(INSIGHT_Highco2RegulatorPaused, 0);
			}	
			running_data->abnormal.bit.high_co2_regulator_paused_flag = 0;
		}
	}
	else{
		if(ai_insight_com.utc[INSIGHT_LowCo2Warning] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_LowCo2Warning]));
			running_data->abnormal.bit.low_co2_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_LowCo2Warning, 0);
		}
		if(ai_insight_com.utc[INSIGHT_HighCo2Warning] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_HighCo2Warning]));
			running_data->abnormal.bit.high_co2_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_HighCo2Warning, 0);
		}
	}
}

void ai_insight_set_light_is_ruuning_flag()
{
    sun_light_running_flag = 1;
}

static void ai_device_data_abnormal_alarm_rule(Time_Typedef *tm, ai_setting_t *ai_setting, ai_insight_run_data_st *running_data, dev_type_t *dev_type_list, int16_t* sensor_list)
{
	uint8_t dev_type;
	uint8_t dev_water_bump_flag = 0;
	uint8_t dev_co2_regulator_flag = 0;
	uint8_t dev_inline_fan_flag = 0;
	for(uint8_t i=1; i<PORT_CNT; i++ ){
		if(!is_ai_port(i)){
			continue;
		}
		dev_type = dev_type_list[i].using_type;
		if(dev_type == loadType_water_pump)
			dev_water_bump_flag = 1;
		else if(dev_type == loadType_co2_generator)
			dev_co2_regulator_flag = 1;
		else if(dev_type == loadType_inlinefan && !dev_type_list[i].is_outlet)
			dev_inline_fan_flag = 1;
        switch( dev_type )
        {
            case loadType_growLight:
                //  grow light
                bool is_high_temp;
                int16_t limit_data = (is_temp_unit_f()?GROWLIGHT_HIGH_TEMP_F:GROWLIGHT_HIGH_TEMP_C);
				int16_t cur_data = (is_temp_unit_f()?sensor_list[TEMP_F]:sensor_list[TEMP_C]);
                is_high_temp = ( cur_data >  limit_data* 100 );
                if ( is_high_temp){
                    /* 植物灯提示，只有在当前超过130℉的时候才会显示。 */
                    bool is_in_range = env_data_is_in_target_range(ai_setting,ENV_TEMP,limit_data);

                    if( sun_light_running_flag && running_data->abnormal.bit.grow_light_flag == 0 && is_in_range == false ){
                        running_data->abnormal.bit.grow_light_flag = 1;
                        updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_GrowLightLevelDimmed]) );
                    }
                }else{
                	if( running_data->abnormal.bit.grow_light_flag == 1 ){
                    	running_data->abnormal.bit.grow_light_flag = 0;
						updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_GrowLightLevelDimmed]) );
                	}
                }
                break;

            case loadType_fan:
                //  ------------------------- device -------------------------
                //  clip fan
                if(sensor_list[HUMID] > CLIPFAN_HIGH_HUMID*100){   // on high humid , set clip fan level to stronger
                    bool is_in_range = env_data_is_in_target_range(ai_setting,ENV_HUMID,CLIPFAN_HIGH_HUMID);    
                    // ai_log(TAG, "%s: humid>90!! double speed:%d", __func__,is_in_range);
                    if( running_data->abnormal.bit.clip_fan_flag == 0 && is_in_range==false ){
                        running_data->abnormal.bit.clip_fan_flag = 1;
                        updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_ClipFanLevelRaised]) );
                    }
                }else{
                 	if( running_data->abnormal.bit.clip_fan_flag == 1 ){
                    	running_data->abnormal.bit.clip_fan_flag = 0;
						updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_ClipFanLevelRaised]) );
                 	}
                }
                break;

			case loadType_inlinefan:
				// inlinefan
				if( dev_type_list[i].is_outlet )
					break;
				if(ai_get_sensor_is_selected(ai_setting, CO2)){
					if(ai_setting->port_ctrl[i].inlinefan.config.smart_co2_en){
						if(ml_out_info.ai_inisight_bit.inlinefan_smart_co2)
						{
							if( running_data->abnormal.bit.smart_co2_monitor_flag == 0)
							{
								running_data->abnormal.bit.smart_co2_monitor_flag = 1;
								updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_SmartCo2Monitoring]) );
								ai_log("%s: INSIGHT_SmartCo2Monitoring!! ", __func__);
							}
						}				
						else{
							if( running_data->abnormal.bit.smart_co2_monitor_flag == 1 ){
								running_data->abnormal.bit.smart_co2_monitor_flag = 0;
								updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_SmartCo2Monitoring]) );
								ai_log("%s: INSIGHT_SmartCo2Monitoring cancel!! ", __func__);
								set_aiinsight_be_reset_flag(INSIGHT_SmartCo2Monitoring, 0);
							}
						}
					}else if( running_data->abnormal.bit.smart_co2_monitor_flag == 1 ){
						running_data->abnormal.bit.smart_co2_monitor_flag = 0;
						updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_SmartCo2Monitoring]) );
						ai_log("%s: INSIGHT_SmartCo2Monitoring cancel!! ", __func__);
						set_aiinsight_be_reset_flag(INSIGHT_SmartCo2Monitoring, 0);
					}
				}
				break;

			case loadType_water_pump:
				if(ml_out_info.ai_inisight_bit.water_pum_over_time)
				{
					if(ai_setting->port_ctrl[i].water_pump.mode == 0)
					{
						if( running_data->abnormal.bit.water_detect_alert_flag == 0 ){
							running_data->abnormal.bit.water_detect_alert_flag = 1;
							updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_WaterDetectAlert]) );
							if(ai_insight_com_bak.utc[INSIGHT_WaterDetectAlert] != 0)
								ai_insight_com.utc[INSIGHT_WaterDetectAlert] = ai_insight_com_bak.utc[INSIGHT_WaterDetectAlert];
							ai_insight_com_bak.utc[INSIGHT_WaterDetectAlert] = ai_insight_com.utc[INSIGHT_WaterDetectAlert];
							ai_log("%s: water detect alert!! ", __func__);
						}
						if( ai_insight_com.utc[INSIGHT_WaterDetectAlert] == 0 ){
							set_aiinsight_be_reset_flag(INSIGHT_WaterDetectAlert, 1);
						}
					}
					else if(ai_setting->port_ctrl[i].water_pump.mode == 1)
					{
						if( running_data->abnormal.bit.soil_moisture_alert_flag == 0 ){
							running_data->abnormal.bit.soil_moisture_alert_flag = 1;
							updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_SoilMoistureAlert]) );
							if(ai_insight_com_bak.utc[INSIGHT_SoilMoistureAlert] != 0)
								ai_insight_com.utc[INSIGHT_SoilMoistureAlert] = ai_insight_com_bak.utc[INSIGHT_SoilMoistureAlert];
							ai_insight_com_bak.utc[INSIGHT_SoilMoistureAlert] = ai_insight_com.utc[INSIGHT_SoilMoistureAlert];
							ai_log("%s: soil alert!! ", __func__);
						}
						if( ai_insight_com.utc[INSIGHT_SoilMoistureAlert] == 0 ){
							set_aiinsight_be_reset_flag(INSIGHT_SoilMoistureAlert, 1);
						}
					}
				}
				else
				{
					if(ai_setting->port_ctrl[i].water_pump.mode == 0)
					{
						if( running_data->abnormal.bit.water_detect_alert_flag == 1 ){
							running_data->abnormal.bit.water_detect_alert_flag = 0;							
							updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_WaterDetectAlert]) );
							ai_insight_com_bak.utc[INSIGHT_WaterDetectAlert] = 0;
							ai_log("%s: water detect alert cancel!! ", __func__);
						}
						set_aiinsight_be_reset_flag(INSIGHT_WaterDetectAlert, 0);
					}
					else if(ai_setting->port_ctrl[i].water_pump.mode == 1)
					{
						if( running_data->abnormal.bit.soil_moisture_alert_flag == 1 ){
							running_data->abnormal.bit.soil_moisture_alert_flag = 0;
							updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_SoilMoistureAlert]) );
							ai_insight_com_bak.utc[INSIGHT_SoilMoistureAlert] = 0;
							ai_log("%s: soil alert cancel!! ", __func__);
						}
						set_aiinsight_be_reset_flag(INSIGHT_SoilMoistureAlert, 0);
					}
				}
				break;

			case loadType_co2_generator:
				if(ml_out_info.ai_inisight_bit.co2_genarator_over_time)
				{
					if( running_data->abnormal.bit.co2_safety_shutoff_flag == 0)
					{
						running_data->abnormal.bit.co2_safety_shutoff_flag = 1;
						updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_co2SafetyShutoff]) );
						if(ai_insight_com_bak.utc[INSIGHT_co2SafetyShutoff] != 0)
							ai_insight_com.utc[INSIGHT_co2SafetyShutoff] = ai_insight_com_bak.utc[INSIGHT_co2SafetyShutoff];
						ai_insight_com_bak.utc[INSIGHT_co2SafetyShutoff] = ai_insight_com.utc[INSIGHT_co2SafetyShutoff];
						ai_log("%s: co2 safety shutoff alert!! ", __func__);
					}
					if( ai_insight_com.utc[INSIGHT_co2SafetyShutoff] == 0 ){
						set_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff, 1);
					}
					if(sensor_list[CO2] >= (ai_setting->port_ctrl[i].co2_generator.target * 10))
					{
						if(ai_insight_com.utc[INSIGHT_co2SafetyShutoff] != 0)
						{
							updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2SafetyShutoff]) );
							running_data->abnormal.bit.co2_safety_shutoff_flag = 0;
							set_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff, 0);
						}
					}						
				}
				else
				{
					if( running_data->abnormal.bit.co2_safety_shutoff_flag == 1 && g_sensors.co2.dectected)
					{
						running_data->abnormal.bit.co2_safety_shutoff_flag = 0;
						updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2SafetyShutoff]) );
						ai_insight_com_bak.utc[INSIGHT_co2SafetyShutoff] = 0;
						ai_log("%s: co2 safety shutoff alert cancel!! ", __func__);
					}
					set_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff, 0);
				}
				if(ml_out_info.ai_inisight_bit.co2_off_outside_range)
				{
					if( running_data->abnormal.bit.co2_regulator_paused_flag == 0)
					{
						running_data->abnormal.bit.co2_regulator_paused_flag = 1;
						updateHighStartTime(tm, &(ai_insight_com.utc[INSIGHT_co2RegulatorPaused]) );
						if(ai_insight_com_bak.utc[INSIGHT_co2RegulatorPaused] != 0)
							ai_insight_com.utc[INSIGHT_co2RegulatorPaused] = ai_insight_com_bak.utc[INSIGHT_co2RegulatorPaused];
						ai_insight_com_bak.utc[INSIGHT_co2RegulatorPaused] = ai_insight_com.utc[INSIGHT_co2RegulatorPaused];
						ai_log("%s: co2 alert outside range!! ", __func__);
					}
				}
				else{
					if( running_data->abnormal.bit.co2_regulator_paused_flag == 1 )
					{
						running_data->abnormal.bit.co2_regulator_paused_flag = 0;
						updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2RegulatorPaused]) );
						ai_insight_com_bak.utc[INSIGHT_co2RegulatorPaused] = 0;
						ai_log("%s: co2 alert outside range cancel!! ", __func__);
					}
					set_aiinsight_be_reset_flag(INSIGHT_co2RegulatorPaused, 0);
				}
				break;
        }
    }
	if(!dev_water_bump_flag ){
		 if( ai_insight_com.utc[INSIGHT_SoilMoistureAlert] != 0 )
		 {
		 	updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_SoilMoistureAlert]));
			running_data->abnormal.bit.soil_moisture_alert_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_SoilMoistureAlert, 0);
		 }
		 if( ai_insight_com.utc[INSIGHT_WaterDetectAlert] != 0 )
		 {
		 	updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_WaterDetectAlert]));
			running_data->abnormal.bit.water_detect_alert_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_WaterDetectAlert, 0);
		 }
	}
	if(!dev_co2_regulator_flag){
		if(ai_insight_com.utc[INSIGHT_co2SafetyShutoff] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2SafetyShutoff]));
			running_data->abnormal.bit.co2_safety_shutoff_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff, 0);
		}
		if( ai_insight_com.utc[INSIGHT_Highco2RegulatorPaused] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_Highco2RegulatorPaused]));
			running_data->abnormal.bit.high_co2_regulator_paused_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_Highco2RegulatorPaused, 0);
		}
	}
	else{
		if( ai_insight_com.utc[INSIGHT_HighCo2Warning] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_HighCo2Warning]));
			running_data->abnormal.bit.high_co2_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_HighCo2Warning, 0);
		}
	}
	if(!g_sensors.co2.dectected){
		if(ai_insight_com.utc[INSIGHT_co2SafetyShutoff] != 0){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2SafetyShutoff]));
			running_data->abnormal.bit.co2_safety_shutoff_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_co2SafetyShutoff, 0);
		}
	}
	if(!dev_co2_regulator_flag || !ml_out_info.ai_inisight_bit.co2_off_outside_range){
		if(ai_insight_com.utc[INSIGHT_co2RegulatorPaused] != 0 ){
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_co2RegulatorPaused]));
			running_data->abnormal.bit.co2_regulator_paused_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_co2RegulatorPaused, 0);
		}
	}
	if(!dev_inline_fan_flag || !g_sensors.temp_c.dectected || !ai_get_sensor_is_selected(ai_setting, CO2) || !ml_out_info.ai_inisight_bit.inlinefan_smart_co2){
		if(ai_insight_com.utc[INSIGHT_SmartCo2Monitoring] != 0){			
			updateHighStartTime(NULL, &(ai_insight_com.utc[INSIGHT_SmartCo2Monitoring]) );
			ai_log("%s: INSIGHT_SmartCo2Monitoring cancel!! ", __func__);
			running_data->abnormal.bit.smart_co2_monitor_flag = 0;
			set_aiinsight_be_reset_flag(INSIGHT_SmartCo2Monitoring, 0);
			ml_out_info.ai_inisight_bit.inlinefan_smart_co2 = 0;
		}
	}
}

uint16_t ai_save_insight_data(uint8_t* p_buf)
{
	uint16_t len = 0;

	memcpy(&(p_buf[len]), (uint8_t *)&ai_insight_com.utc, sizeof(ai_insight_com_st) );
	len += sizeof(ai_insight_com_st);
	memcpy(&(p_buf[len]), (uint8_t *)&ai_insight_com_bak.utc, sizeof(ai_insight_com_st) );
	len += sizeof(ai_insight_com_st);
	memcpy(&(p_buf[len]), (uint8_t *)&ai_insight_run_data, sizeof(ai_insight_run_data_st) );
	len += sizeof(ai_insight_run_data_st);
	memcpy(&(p_buf[len]), (uint8_t *)&ai_insight_out, sizeof(ai_insight_out) );
	len += sizeof(ai_insight_out);

	return len;
}

uint16_t ai_read_insight_data(uint8_t* p_buf)
{
	uint16_t len = 0;

	memcpy((uint8_t *)&ai_insight_com.utc, &(p_buf[len]), sizeof(ai_insight_com_st) );
	len += sizeof(ai_insight_com_st);
	memcpy((uint8_t *)&ai_insight_com_bak.utc, &(p_buf[len]), sizeof(ai_insight_com_st) );
	len += sizeof(ai_insight_com_st);
	memcpy((uint8_t *)&ai_insight_run_data, &(p_buf[len]),  sizeof(ai_insight_run_data_st) );
	len += sizeof(ai_insight_run_data_st);
	memcpy((uint8_t *)&ai_insight_out, &(p_buf[len]),  sizeof(ai_insight_out) );
	len += sizeof(ai_insight_out);
	
	return len;
}

void ai_insight_rule(  Time_Typedef *tm,ai_setting_t *ai_setting, ai_insight_run_data_st *ai_insight_data, 
					dev_type_t *dev_type_list, int16_t* sensor_list)
{
	static u8 is_ai_deleted_bak = 0;
    ai_sensor_data_abnormal_alarm_rule(tm,ai_setting,&ai_insight_run_data,sensor_list);
    ai_device_data_abnormal_alarm_rule(tm,ai_setting,&ai_insight_run_data,dev_type_list,sensor_list);
	if(g_ai_setting.is_ai_deleted == 1 && is_ai_deleted_bak == 0)
		clean_ai_insights();
	is_ai_deleted_bak = g_ai_setting.is_ai_deleted;
    sun_light_running_flag = 0;
}
