
#include <esp_log.h>
#include <string.h>  
#include <esp_system.h>
#include <sys/param.h> 
#include "sensor_module.h" 
//#include "app_time_rule.h"
#include "math.h"

//#include "Sensor.h"
#include "ml_pid.h"
#include "ml_rule.h"
#define TAG "RLSensor  " // 10个字符

    

void rl_sync_speed( ml_output_port_t* output_port_list, rule_speed_t *dev_speeds, dev_type_t* dev_type_list )
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


static double pid_exp(double x)
{
	if(x > 708) x = 708;
	else if(x < -708) x = -708;
	return exp(x);	
}
float pid_cal_vpd(float t, float rh)
{
	float dx;
	dx = (17.27f * t) / (t + 237.3f);
	dx = pid_exp(dx);
	dx = dx * 0.61078 * (1.0f - rh / 100.0f);
	return dx;
}

void read_all_sensor(void )
{
      ml_read_sensor();
	  for (uint8_t i = 1; i < 9; i++)
	  {
		ml_set_sw( i,0);
	  }
	  ml_set_sw( 1,1);
	  
		bp_pid_th.t_feed    = ml_get_cur_temp() ;//input->env_value_cur[ENV_TEMP]/10.0f;
		bp_pid_th.t_target  = ml_get_target_temp() ;
		bp_pid_th.t_outside = ml_get_outside_temp() ;//input->env_value_cur[ENV_TEMP]/10.0f;
		//if(is_temp_unit_f())
		//{	//C = (F - 32) × 5/9
		//	bp_pid_th.t_target = (bp_pid_th.t_target - 32) * 5 / 9;
		//	bp_pid_th.t_feed   = (bp_pid_th.t_feed   - 32) * 5 / 9;
		//	bp_pid_th.t_outside= (bp_pid_th.t_outside- 32) * 5 / 9;
		//}
		bp_pid_th.h_target = ml_get_target_humid() ;
		bp_pid_th.h_feed   = ml_get_cur_humid() ;//nput->env_value_cur[ENV_HUMID]/10.0f;
		bp_pid_th.h_outside =   ml_get_outside_humid() ;//nput->env_value_cur[ENV_HUMID]/10.0f;
	  	bp_pid_th.l_feed =ml_get_cur_light();
	  	bp_pid_th.c_feed =ml_get_cur_co2(); 
		
		//if(input->env_en_bit & (1 << ENV_VPD))
		//{
		//	bp_pid_th.v_target = input->env_target[ENV_VPD]/100.0;
   		//}
		//else
		//{
			bp_pid_th.v_target =  pid_cal_vpd(bp_pid_th.t_target, bp_pid_th.h_target) ; 
   			//bp_pid_dbg(" cal v_target=%.2f,t_target= %.2f, h_target= %.2f  \r\n", pid_arg.v_target, pid_arg.t_target,pid_arg.h_target  );
   			//bp_pid_dbg(" cal v_target=%.2f,v_feed= %.2f, v_inside= %.2f  \r\n", pid_arg.v_target, pid_arg.v_feed,pid_arg.v_inside  );
		//}			
		bp_pid_th.v_feed    = ml_get_cur_vpd()/100.0;//input->env_value_cur[ENV_VPD]/10.0f;
		bp_pid_th.v_outside = ml_get_outside_vpd()/100.0;//in_side_vpd;
		ESP_LOGI(TAG, "inisde  t=%f, h=%f, v=%f", bp_pid_th.t_feed, bp_pid_th.h_feed, bp_pid_th.v_feed);
		ESP_LOGI(TAG, "outside t=%f, h=%f, v=%f", bp_pid_th.t_outside, bp_pid_th.h_outside, bp_pid_th.v_outside);
		ESP_LOGW(TAG, "light:%f co2:%f ",bp_pid_th.l_feed,bp_pid_th.c_feed);

		ESP_LOGI(TAG, "傳感器正常运行");

}

float read_temperature_sensor(void)
{
    return bp_pid_th.t_feed;
}

float read_humidity_sensor(void)
{
    return bp_pid_th.h_feed;
}

float read_light_sensor(void)
{
    return 28.0;
}
#define MODEL_INPUT_SIZE 64
 
uint8_t img_0[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*3];
 
//extern pid_run_output_st pid_run_rule(pid_run_input_st* input);
// 传感器数据采集函数
rl_sensor_data_t *get_sensor_data(uint8_t *img) {
    //pid_run_output_st pid_run_output;
	//pid_param_get(ai_setting, cur_load_type, port_dev_origin, env_val_list, &pid_run_input );
	//pid_run_output = pid_run_rule( &pid_run_input );
	//pid_rule_output_set_speed(pid_run_output, cur_load_type, output_port_list );

    static rl_sensor_data_t data; 
    // 获取湿度数据
    data.temperature = read_temperature_sensor();
    data.humidity = read_humidity_sensor();
    // 获取光感器数据
    data.light_lux = read_light_sensor();
    // 获取图像数据
    data.camera_frame= img;
    //capture_image( data.camera_frame); // 从相机模块获取 64x64 的图像数据
    return &data;
} 
 
 

