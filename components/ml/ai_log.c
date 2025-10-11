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
#include "ml_rule.h"
// --------------- ML LOG --------------- //
#define MAX_LOG_ENTRIES 1000  // 哈希表大小
#define MAX_LOG_AGE 60  // 日志有效时间（秒）

EXT_RAM_BSS_ATTR ai_setting_t g_old_ai_setting;
EXT_RAM_BSS_ATTR ai_night_mode_setting_t g_old_ai_night_mode_setting;
EXT_RAM_BSS_ATTR sys_setting_t old_sys_setting;
EXT_RAM_BSS_ATTR sensors_t g_sensors_old;

EXT_RAM_BSS_ATTR curLoad_t old_curLoad[PORT_CNT] = {0}; 	//负载的相关变量 风扇档位，插座开关状态历史数据
EXT_RAM_BSS_ATTR curLoad_t old_ai_curLoad[PORT_CNT] = {0}; 	//负载的相关变量 风扇档位，插座开关状态历史数据

uint8_t distLoad_type[PORT_CNT] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // 记录AI端口的工作设备类型

ml_log_mem_data_st log_mem_data;
// --------------- ML LOG --------------- //


#define TAG "  AI_LOG  "


// 哈希表节点
typedef struct LogEntry {
    u32 timestamp;
    u8 type;
    u8 data[12];
    struct LogEntry* next;
} LogEntry;

// log哈希表数组
EXT_RAM_BSS_ATTR LogEntry* hashLog[MAX_LOG_ENTRIES];
// msg哈希表数组
EXT_RAM_BSS_ATTR LogEntry* hashMsg[MAX_LOG_ENTRIES];

// 计算哈希值
unsigned int hash(unsigned char *str)
{
    unsigned int hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % MAX_LOG_ENTRIES;
}

// 清理过期日志
void cleanExpiredLogs() {
    u32 current_time = (u32)time(NULL);

    for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
        LogEntry* prev = NULL;
        LogEntry* current = hashLog[i];
        
        while (current != NULL) {
            if (current_time - current->timestamp > MAX_LOG_AGE) {
                // 当前日志已过期，从哈希表中删除
                if (prev == NULL) {
                    hashLog[i] = current->next;
                } else {
                    prev->next = current->next;
                }
                heap_caps_free(current);
                current = prev == NULL ? hashLog[i] : prev->next;
            } else {
                prev = current;
                current = current->next;
            }
        }
    }
}

void clearAllLogs() {
    for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
        LogEntry* current = hashLog[i];
        while (current != NULL) {
            LogEntry* next = current->next;
            heap_caps_free(current);
            current = next;
        }
        hashLog[i] = NULL; // 将哈希表的当前条目置为空
    }
}

// 检查是否存在重复日志
int checkDuplicateLog(u8 type, u8 data[12]) {
    //unsigned int index = hash(data);
    //LogEntry* current = hashLog[index];
	for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
		LogEntry* current = hashLog[i];
		while (current != NULL) {
			if (current->type == type && memcmp(current->data, data, sizeof(msg_MLInfo_t)) == 0) {
				return 1;  // 不同端口相同设备类型重复日志
			}
			// ESP_LOGI(TAG, "checkDuplicateLog current_port[%d]  port[%d], current->type : [%d], type : [%d] , LoadType : %d , LoadType : %d !", current->data[0], data[0], current->type, type, _getTargetDummyLoadType_port(data[0]), _getTargetDummyLoadType_port(current->data[0]));
			// ESP_LOGI(TAG, "checkDuplicateLog current->data : [%d], data : [%d], data : [%d] !", (current->data[6] & 0x2), (data[6] & 0x2), data[6] & 0x3C);
			if (current->type == type && current->type != AI_MODE_CONTROL ) {
				if (current->type == AI_CONTROL){
					if( (_getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0]))){
						if( (current->data[6] & 0x2) == (data[6] & 0x2) 
							&& ((current->data[6] & 0x3C) == (data[6] & 0x3C) || ((data[6] & 0x3C) == 0 && _getTargetDummyLoadType_port(data[0]) != loadType_growLight))) {
							return 1;  // 设备类型相同重复日志 植物灯turn on/off 保留，区分日出日落
						}
					}
				}else if(current->type == USER_CONTROL && data[1] == current->data[1] ) {
					if( data[1] == USER_CONTROL_SENSOR ){
						if( memcmp(current->data, data, sizeof(msg_MLInfo_t)) == 0 )
							return 1;  // 设备类型相同重复日志
					}else if( data[1] == USER_CONTROL_DYNAMIC_SUN ){
						if( memcmp(current->data + 1, data + 1, sizeof(msg_MLInfo_t) - 1) == 0 )
							return 1;
					}else if( data[1] == USER_CONTROL_SUNRISE_SUNSET ){
						if( _getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0]) && 
							current->data[1] == data[1] && current->data[2] == data[2] &&
							memcmp(current->data + 8, data + 8, sizeof(msg_MLInfo_t) - 8) == 0 ){
							//	过滤当中5字节 不区分原始设备类型
							return 1;
						}
					}else{
						if( (_getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0])) && memcmp(current->data + 1, data + 1, sizeof(msg_MLInfo_t) - 1) == 0 ){
							return 1;
						}
					}
				}
			}
			current = current->next;
		}
	}
    return 0;  // 不是重复日志
}

// 添加日志到哈希表
void addLogToHashLog(u8 type, u8 data[12]) {
    unsigned int index = hash(data);
    LogEntry* newNode = (LogEntry*)heap_caps_malloc(sizeof(LogEntry), MALLOC_CAP_SPIRAM);
    if (newNode == NULL)
	{
		ESP_LOGE(TAG, "**** %s() malloc addLogToHashLog ram err1", __func__);
		return;
	}
	newNode->timestamp = (u32)time(NULL);
    newNode->type = type;
    memcpy(newNode->data, data, sizeof(msg_MLInfo_t));
    newNode->next = hashLog[index];
    hashLog[index] = newNode;
} 

// 清理过期Msg
void cleanExpiredMsgs() {
    u32 current_time = (u32)time(NULL);

    for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
        LogEntry* prev = NULL;
        LogEntry* current = hashMsg[i];
		
        while (current != NULL) {
            if (current_time - current->timestamp > MAX_LOG_AGE) {
                // 当前Msg已过期，从哈希表中删除
                if (prev == NULL) {
                    hashMsg[i] = current->next;
                } else {
                    prev->next = current->next;
                }
                heap_caps_free(current);
                current = prev == NULL ? hashMsg[i] : prev->next;
            } else {
                prev = current;
                current = current->next;
            }
        }
    }
}

void clearAllMsgs() {
    for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
        LogEntry* current = hashMsg[i];
        while (current != NULL) {
            LogEntry* next = current->next;
            heap_caps_free(current);
            current = next;
        }
        hashMsg[i] = NULL; // 将哈希表的当前条目置为空
    }
}

// 检查是否存在重复Msg
int checkDuplicateMsg(u8 type, u8 data[12]) {
    // unsigned int index = hash(data);
    // LogEntry* current = hashMsg[index];
	for (int i = 0; i < MAX_LOG_ENTRIES; ++i) {
		LogEntry* current = hashLog[i];
		while (current != NULL) {
			if (current->type == type && memcmp(current->data, data, sizeof(msg_MLInfo_t)) == 0) {
				return 1;  // 不同端口相同设备类型重复日志
			}
			// ESP_LOGI(TAG, "checkDuplicateLog current_port[%d]  port[%d], current->type : [%d], type : [%d] , LoadType : %d , LoadType : %d !", current->data[0], data[0], current->type, type, _getTargetDummyLoadType_port(data[0]), _getTargetDummyLoadType_port(current->data[0]));
			// ESP_LOGI(TAG, "checkDuplicateLog current->data : [%d], data : [%d], data : [%d] !", (current->data[6] & 0x2), (data[6] & 0x2), data[6] & 0x3C);
			if (current->type == type && current->type != AI_MODE_CONTROL ) {
				if (current->type == AI_CONTROL){
					if( (_getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0]))){
						if( (current->data[6] & 0x2) == (data[6] & 0x2) 
							&& ((current->data[6] & 0x3C) == (data[6] & 0x3C) || ((data[6] & 0x3C) == 0 && _getTargetDummyLoadType_port(data[0]) != loadType_growLight))) {
							return 1;  // 设备类型相同重复 植物灯turn on/off 保留，区分日出日落
						}
					}
				}else if(current->type == USER_CONTROL ) {
					if( data[1] != current->data[1] ){

					}else if( data[1] == USER_CONTROL_SENSOR ){
						if( memcmp(current->data, data, sizeof(msg_MLInfo_t)) == 0 )
							return 1;  // 设备类型相同重复
					}else if( data[1] == USER_CONTROL_DYNAMIC_SUN ){
						if( memcmp(current->data + 1, data + 1, sizeof(msg_MLInfo_t) - 1) == 0 )
							return 1;
					}else if( data[1] == USER_CONTROL_SUNRISE_SUNSET ){
						if( _getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0]) && 
							current->data[1] == data[1] && current->data[2] == data[2] &&
							memcmp(current->data + 8, data + 8, sizeof(msg_MLInfo_t) - 8) == 0 ){
							//	过滤当中5字节 不区分原始设备类型
							return 1;
						}
					}else{
						if( (_getTargetDummyLoadType_port(data[0]) == _getTargetDummyLoadType_port(current->data[0])) ){
							if( memcmp(current->data + 1, data + 1, sizeof(msg_MLInfo_t) - 1) == 0 ){
								return 1;
							}
						}
					}
				}
			}
			current = current->next;
		}
	}
    return 0;  // 不是重复
}

// 添加Msg到哈希表
void addLogToHashMsg(u8 type, u8 data[12]) {
    unsigned int index = hash(data);
	LogEntry* newNode = (LogEntry*)heap_caps_malloc(sizeof(LogEntry), MALLOC_CAP_SPIRAM);
    if (newNode == NULL)
	{
		ESP_LOGE(TAG, "**** %s() malloc addLogToHashMsg ram err1", __func__);
		return;
	}
    newNode->timestamp = (u32)time(NULL);
    newNode->type = type;
    memcpy(newNode->data, data, sizeof(msg_MLInfo_t));
    newNode->next = hashMsg[index];
    hashMsg[index] = newNode;
}

//生成一条ML log信息
static void Generate_ML_log(u8 ml_log_type, u8 data[12])
{
	// 清理过期日志
    cleanExpiredLogs();
    
    // 如果当前日志与哈希表中的日志重复，则不生成新日志
    if (!checkDuplicateLog(ml_log_type, data)) {
		// printf("ML Log Type: %u\n", ml_log_type);
		// for (int i = 0; i < 12; ++i) {
		// 	printf("%u ", data[i]);
		// }
		// printf("\n");
		if(ml_log_type == USER_CONTROL && (data[1] & 0x0f) >= USER_CONTROL_MAX){
			return;
		}
		if(ml_log_type == AI_MODE_CONTROL && data[4] >= AI_MODE_LOG_MAX){
			return;
		}
		log_payload_t log = {0};
		log.ml.num = 0;
		log.ml.type = ml_log_type;
		memcpy(log.ml.info.data, data, sizeof(msg_MLInfo_t));
		GenerateLog_AI(&log);
						
		// 添加日志到哈希表
		addLogToHashLog(ml_log_type, data);
    }
}

static void Generate_ML_Msg(u8 ml_log_type, u8 data[12])
{
	// 清理过期日志
    cleanExpiredMsgs();
    
    // 如果当前日志与哈希表中的日志重复，则不生成新日志
    if (!checkDuplicateMsg(ml_log_type, data)) {	
		// printf("ML Log Type: %u\n", ml_log_type);
		// for (int i = 0; i < 12; ++i) {
		// 	printf("%u ", data[i]);
		// }
		// printf("\n");
		if(ml_log_type == USER_CONTROL && (data[1] & 0x0f) >= USER_CONTROL_MAX){
			return;
		}
		if(ml_log_type == AI_MODE_CONTROL && data[4] >= AI_MODE_LOG_MAX){
			return;
		}
		log_payload_t log = {0};
		log.ml.num = 0;
		log.ml.type = ml_log_type;
		memcpy(log.ml.info.data, data, sizeof(msg_MLInfo_t));
		CreateMsg_AI(msgStyle_pp, &log);
								
		// 添加日志到哈希表
		addLogToHashMsg(ml_log_type, data);
	}
}

s16 ai_log_get_cur_vpd()
{
	return GetRealVpd();//return g_ai_setting.switch_zone_position ? get_real_zonevpd() : GetRealVpd();
}

s16 ai_log_get_cur_humid()
{
	return GetRealHumid();//return g_ai_setting.switch_zone_position ? get_real_zonehumid() : GetRealHumid();
}

s16 ai_log_get_cur_temp()
{
	return GetRealTemp(is_temp_unit_f());//return g_ai_setting.switch_zone_position ? get_real_zonetemp(is_temp_unit_f()) : GetRealTemp(is_temp_unit_f());
}

s16 ai_log_get_outside_vpd()
{
	return get_real_zonevpd();//return (g_ai_setting.switch_zone_position==0) ? get_real_zonevpd() : GetRealVpd();
}

s16 ai_log_get_outside_humid()
{
	return get_real_zonehumid();//return (g_ai_setting.switch_zone_position==0) ? get_real_zonehumid() : GetRealHumid();
}

s16 ai_log_get_outside_temp()
{
	return get_real_zonetemp(is_temp_unit_f());//return (g_ai_setting.switch_zone_position==0) ? get_real_zonetemp(is_temp_unit_f()) : GetRealTemp(is_temp_unit_f());
}

s16 ai_log_get_cur_co2()
{
	return get_real_co2();
}

s16 ai_log_get_cur_soil()
{
	return get_real_soil_humid();
}

s16 ai_log_get_cur_water()
{
	return get_real_water();
}

//	传入单位 1 f
int16_t temp_to_C(uint16 temp)
{
    if (is_temp_unit_f())
        return C_to_F(temp);
    else
        return temp;
}

// Function to check if ai_setting_t is empty
int is_ai_setting_empty(const ai_setting_t *setting) {
    ai_setting_t empty_setting = {0};

	if (setting == NULL) return 0;
    //memset(&empty_setting, 0, sizeof(ai_setting_t));
    return memcmp(setting, &empty_setting, sizeof(ai_setting_t)) == 0;
}

// Function to check if ai_night_mode_setting_t is empty
int is_ai_night_mode_setting_empty(const ai_night_mode_setting_t *setting) {
    ai_night_mode_setting_t empty_setting = {0};

	if (setting == NULL) return 0;
    return memcmp(setting, &empty_setting, sizeof(ai_night_mode_setting_t)) == 0;
}

int is_curLoad_empty(const curLoad_t *curLoad) {
    curLoad_t empty_curLoad;
    memset(&empty_curLoad, 0, sizeof(curLoad_t));
    return memcmp(curLoad, &empty_curLoad, sizeof(curLoad_t)) == 0;
}

int is_sensor_setting_empty(const sys_setting_t *sensor_setting) {
    sys_setting_t empty_sensor_setting;
    memset(&empty_sensor_setting, 0, sizeof(sys_setting_t));
    return memcmp(sensor_setting, &empty_sensor_setting, sizeof(sys_setting_t)) == 0;
}

int is_g_sensors_empty(const sensors_t *g_sensors) {
    sensors_t empty_g_sensors;
    memset(&empty_g_sensors, 0, sizeof(sensors_t));
    return memcmp(g_sensors, &empty_g_sensors, sizeof(sensors_t)) == 0;
}

bool is_during_night_mode_time()
{
	Time_Typedef cur_time = *get_cur_time();
	u16 current_min,start_min,end_min;

	current_min = cur_time.hour * 60 + cur_time.min;
	start_min = g_ai_night_mode_setting.startHour * 60 + g_ai_night_mode_setting.startMin;
	end_min = g_ai_night_mode_setting.endHour * 60 + g_ai_night_mode_setting.endMin;
	if(start_min == end_min)
		return true;
	if(start_min > end_min)
		end_min += 24*60;
	if(current_min >= start_min && current_min <= end_min)
		return true;
	return false;
}

bool is_ai_night_mode()
{
	if( !g_ai_night_mode_setting.config.en )
		return false;
	if( is_during_night_mode_time( ) )
		return true;
	return false;
}

u8 get_target_type()
{
	u8 type = 0;
	
	if(g_ai_night_mode_setting.config.en)
		type = 1;
	return type;
}

u8 get_day_night_mode_type()
{
	u8 type = 0;
	
	if(!g_ai_night_mode_setting.config.en)
		type = 0;
	else if(is_ai_night_mode())
		type = 1;
	else
		type = 2;
	return type;
}

// checnk  sensor1/sensor2
bool sensor1_flag = false;	//sensor1 在线状态
bool sensor2_flag = false;	//sensor2 在线状态
bool sensor_change_status = false;
u8 sensor1_status;
u8 sensor2_status;
void checkInConnectionStatus(uint8_t old_sta, uint8_t new_sta, u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status) {
    u8 old_status = old_sta;
    u8 new_status = new_sta;
	if((old_status == new_status) && (new_status == 1)){
		sensor1_flag = true;
		sensor2_flag = false;
	}
	if((old_status == new_status) && (new_status == 2)){
		sensor1_flag = false;
		sensor2_flag = true;
	}
    if (old_status == 0 && new_status != 0) {
        *sensor_port = (new_status == 1) ? 1 : 2;
		(new_status == 1) ? (sensor1_flag = true) : (sensor2_flag = true);
        *sensor_status = 0;
		*new_sensor_status= 1;// sensor1 || sensor2 Acces
		sensor_change_status = true;
    } else if (old_status != 0 && new_status == 0) {
        *sensor_port = (old_status == 1) ? 1 : 2;
		(old_status == 1) ? (sensor1_flag = false) : (sensor2_flag = false);
        *sensor_status = 1;
		*new_sensor_status= 0; // sensor1 || sensor2 Lost
		sensor_change_status = true;
    } else if (old_status == 1 && new_status == 2) {
        *sensor_port = 1;
		sensor1_flag = false;
		sensor2_flag = true;
        *sensor_status = 1;
		*new_sensor_status= 0; // sensor2 Acces && sensor1 Lost
		sensor_change_status = true;
		// ESP_LOGW(TAG,"sensor inside change 1->2!!!!");
    } else if (old_status == 2 && new_status == 1) {	//切换新端口
        *sensor_port = 1;
		sensor1_flag = true;
		// sensor2_flag = true;
        *sensor_status = 0;
		*new_sensor_status= 1;// sensor1 || sensor2 Acces
		sensor_change_status = true;
		// ESP_LOGW(TAG,"sensor inside change 2->1!!!!");
    }
}

void checkOutConnectionStatus(uint8_t old_sta, uint8_t new_sta, u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status) {
	// zone
	u8 old_status = old_sta;
    u8 new_status = new_sta;
	if (old_status != 2 && new_status == 2 && sensor1_flag) {
		*sensor_port = 2;
		sensor2_flag = true;
		*sensor_status = 0; 
		*new_sensor_status= 1; // sensor1 Access && sensor2 Acces
		sensor_change_status = true;
		// ESP_LOGI(TAG, "Fai3 Find sensor_port[%d] : SensorStatus-- %d, %d !", *sensor_port, old_status, new_status);
	} else if (old_status == 2 && new_status != 2 && sensor1_flag) {
		*sensor_port = 2;
		sensor2_flag = false;
		*sensor_status = 1; 
		*new_sensor_status= 0; // sensor1 Access && sensor2 Lost
		sensor_change_status = true;
		// ESP_LOGI(TAG, "Fai4 Find sensor_port[%d] : SensorStatus-- %d, %d !", *sensor_port, old_status, new_status);
	} else if (old_status == 2 && new_status == 2 && sensor1_flag) {
		*sensor_port = 2;
		sensor2_flag = true;
		*sensor_status = 1; 
		*new_sensor_status= 1; // sensor1 Access && sensor2 Access
		// ESP_LOGI(TAG, "Fai5 Find sensor_port[%d] : SensorStatus-- %d, %d !", *sensor_port, old_status, new_status);
	} else{
		*sensor_port = 0;
		*sensor_status = 1; 
		*new_sensor_status= 1;
		// ESP_LOGI(TAG, "Fai6 Find sensor_port[%d] : SensorStatus-- %d, %d !", *sensor_port, old_status, new_status);
	}
}

void checkConnectionStatus(sensor_val_t *old_sensor, sensor_val_t *new_sensor, u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status) {
    u8 old_status = (*old_sensor).dectected;
    u8 new_status = (*new_sensor).dectected;
	if(sensor1_flag && sensor2_flag){
		return;
	}
    if (old_status == 0 && new_status != 0) {
        *sensor_port = (new_status == 1) ? 1 : 2;
        *sensor_status = 0;
		*new_sensor_status= 1;// Access
    } else if (old_status != 0 && new_status == 0) {
        *sensor_port = (old_status == 1) ? 1 : 2;
        *sensor_status = 1;
		*new_sensor_status= 0; // Lost
    }else{
		*sensor_port = 0;
		*sensor_status = 0;
		*new_sensor_status= 0;
	}
}

void Generate_ML_side_sensor_log(u8 *sensor_port, u8 *sensor_status, u8 inside_or_outside, u8 *new_sensor_status){
	u8 ml_log_type = USER_CONTROL;
	u8 data[12] = {0};
	data[0] = *sensor_port;
	data[1] = USER_CONTROL_SENSOR;
	if(*sensor_status != *new_sensor_status && *new_sensor_status == 0){
		if(*sensor_port == 2 && sensor1_flag){
			data[0] = 0;
		}
		// Sensor 2 (OUTSIDE) lost its connection. AI programming has been paused.
		data[2] = AI_CONTROL_SENSOR_LOST;
		data[3] = inside_or_outside;
		// Generate_ML_log(ml_log_type, data);
	}
	*sensor_port = 0;
	*sensor_status = 0;
	*new_sensor_status= 0;
}

// void Generate_ML_outside_sensor_log(u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status){
// 	u8 ml_log_type = USER_CONTROL;
// 	u8 data[12] = {0};
// 	data[0] = *sensor_port;
// 	data[1] = USER_CONTROL_SENSOR;
// 	if(*sensor_status != *new_sensor_status && *new_sensor_status == 0){
// 		// Sensor 2 (OUTSIDE) lost its connection. AI programming has been paused.
// 		data[2] = AI_CONTROL_SENSOR_LOST;
// 		data[3] = g_ai_setting.switch_zone_position == 0? 0 : 1;
// 		Generate_ML_log(ml_log_type, data);
// 	}
// 	*sensor_port = 0;
// 	*sensor_status = 0;
// 	*new_sensor_status= 0;
// }


void Generate_ML_sensor_log(u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status){
	u8 ml_log_type = USER_CONTROL;
	u8 data[12] = {0};
	data[0] = *sensor_port;
	data[1] = USER_CONTROL_SENSOR;
	if(*sensor_status != *new_sensor_status && *new_sensor_status == 0){
		// Sensor 2 (OUTSIDE) lost its connection. AI programming has been paused.
		data[2] = AI_CONTROL_SENSOR_LOST;
		data[3] = *new_sensor_status;
		// Generate_ML_log(ml_log_type, data);
	}
	*sensor_port = 0;
	*sensor_status = 0;
	*new_sensor_status= 0;
}

void Generate_ML_sensor_side_log(u8 *sensor_port, u8 *sensor_status, u8 *new_sensor_status){
	u8 ml_log_type = USER_CONTROL;
	u8 data[12] = {0};
	data[0] = *sensor_port;
	data[1] = USER_CONTROL_SENSOR;
	if(sensor_change_status || g_ai_setting.switch_zone_position != g_old_ai_setting.switch_zone_position){
		// ESP_LOGI(TAG, "Fai4 Find sensor_port[%d] : Sensor__Flag-- %d, %d !", *sensor_port, sensor1_flag, sensor2_flag);
		if(g_ai_setting.switch_zone_position == 1){
			if(sensor1_flag){
				data[0] = 1;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 1;
				Generate_ML_log(ml_log_type, data);
				// data[0] = sensor2_flag ? 2 : 0;
				data[0] = 0;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 0;
				Generate_ML_log(ml_log_type, data);
			}else if(!sensor1_flag){
				if(sensor2_flag){
					data[0] = 2;
					data[2] = AI_CONTROL_SENSOR_SIDE;
					data[3] = 1;
					Generate_ML_log(ml_log_type, data);
				}
				data[0] = 0;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 0;
				Generate_ML_log(ml_log_type, data);
			}
		}else if(g_ai_setting.switch_zone_position == 0){
			if(sensor1_flag){
				data[0] = 1;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 0;
				Generate_ML_log(ml_log_type, data);
				// data[0] = sensor2_flag ? 2 : 0;
				data[0] = 0;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 1;
				Generate_ML_log(ml_log_type, data);
			}else if(!sensor1_flag){
				if(sensor2_flag){
					data[0] = 2;
					data[2] = AI_CONTROL_SENSOR_SIDE;
					data[3] = 0;
					Generate_ML_log(ml_log_type, data);
				}
				data[0] = 0;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = 1;
				Generate_ML_log(ml_log_type, data);
			}
		}
	}
	// You updated Sensor 2’s position to OUTSIDE.
	*sensor_port = 0;
	*sensor_status = 0;
	*new_sensor_status= 0;
}

void compare_g_sensors(sensors_t *old_sensors, sensors_t *new_sensors){
	if(!g_ai_setting.is_ai_deleted){
		if(g_ai_setting.is_ai_deleted != g_old_ai_setting.is_ai_deleted){
			return;
		}
		if( g_ai_setting.ai_workmode == AI_WORKMODE_PAUSE && g_old_ai_setting.ai_workmode == AI_WORKMODE_PAUSE ) {
			return;
		}
		u8 sensor_port = 0;
		u8 sensor_status = 0;
		u8 new_sensor_status = 0;

		static uint8_t side_sensor_port_old[2]={0xff,0xff};	//0-zone 1-not zone
		uint8_t side_sensor_port_new[2]={0,0};
		extern uint8_t get_side_sensor_port(uint8_t is_zone);
		
		// if(g_ai_setting.switch_zone_position == 1){
		// 	side_sensor_port_new[0] = get_side_sensor_port(0);
		// 	side_sensor_port_new[1] = get_side_sensor_port(1);
		// }else
		{
			side_sensor_port_new[0] = get_side_sensor_port(1);
			side_sensor_port_new[1] = get_side_sensor_port(0);
		}
		
		//	开机同步 
		for(uint8_t i=0; i<2; i++ ){
			if( 0xff == side_sensor_port_old[i] ){
				side_sensor_port_old[i] = side_sensor_port_new[i];
			}
		}

		sensor_change_status = false;

		checkInConnectionStatus(side_sensor_port_old[1], side_sensor_port_new[1], &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 0, &new_sensor_status);
		// OutSide
		checkOutConnectionStatus(side_sensor_port_old[0], side_sensor_port_new[0], &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 1, &new_sensor_status);

		for(uint8_t i=0; i<2; i++ ){
			side_sensor_port_old[i] = side_sensor_port_new[i];
		}

	#if 0
		if (is_temp_unit_f()){
			// Inside
			checkInConnectionStatus(&old_sensors->temp_f, &new_sensors->temp_f, &sensor_port, &sensor_status, &new_sensor_status);
			Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 0, &new_sensor_status);
			// OutSide
			checkOutConnectionStatus(&old_sensors->zonetemp_f, &new_sensors->zonetemp_f, &sensor_port, &sensor_status, &new_sensor_status);
			Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 1, &new_sensor_status);
		}
    	else{
			// Inside
			checkInConnectionStatus(&old_sensors->temp_c, &new_sensors->temp_c, &sensor_port, &sensor_status, &new_sensor_status);
			Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 0, &new_sensor_status);
			// OutSide
			checkOutConnectionStatus(&old_sensors->zonetemp_c, &new_sensors->zonetemp_c, &sensor_port, &sensor_status, &new_sensor_status);
			Generate_ML_side_sensor_log(&sensor_port, &sensor_status, 1, &new_sensor_status);
		}
	#endif
		// // Inside
		// checkInConnectionStatus(&old_sensors->temp_c, &new_sensors->temp_c, &sensor_port, &sensor_status, &new_sensor_status);
		// checkInConnectionStatus(&old_sensors->temp_f, &new_sensors->temp_f, &sensor_port, &sensor_status, &new_sensor_status);
		// checkInConnectionStatus(&old_sensors->humid, &new_sensors->humid, &sensor_port, &sensor_status, &new_sensor_status);
		// checkInConnectionStatus(&old_sensors->vpd, &new_sensors->vpd, &sensor_port, &sensor_status, &new_sensor_status);
		// Generate_ML_side_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);
		// // OutSide
		// checkOutConnectionStatus(&old_sensors->zonetemp_f, &new_sensors->zonetemp_f, &sensor_port, &sensor_status, &new_sensor_status);
		// checkOutConnectionStatus(&old_sensors->zonetemp_c, &new_sensors->zonetemp_c, &sensor_port, &sensor_status, &new_sensor_status);
		// checkOutConnectionStatus(&old_sensors->zone_humid, &new_sensors->zone_humid, &sensor_port, &sensor_status, &new_sensor_status);
		// checkOutConnectionStatus(&old_sensors->zone_vpd, &new_sensors->zone_vpd, &sensor_port, &sensor_status, &new_sensor_status);
		// Generate_ML_side_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);

		//leaftemp
		checkConnectionStatus(&old_sensors->leaftemp_c, &new_sensors->leaftemp_c, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->leaftemp_f, &new_sensors->leaftemp_f, &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);
		//soil_humid
		checkConnectionStatus(&old_sensors->soil_humid, &new_sensors->soil_humid, &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);
		// co2
		checkConnectionStatus(&old_sensors->co2, &new_sensors->co2, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->light, &new_sensors->light, &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);
		// PH+watertemp_c
		checkConnectionStatus(&old_sensors->ph, &new_sensors->ph, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->ec_us, &new_sensors->ec_us, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->ec_ms, &new_sensors->ec_ms, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->tds_ppm, &new_sensors->tds_ppm, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->tds_ppt, &new_sensors->tds_ppt, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->watertemp_c, &new_sensors->watertemp_c, &sensor_port, &sensor_status, &new_sensor_status);
		checkConnectionStatus(&old_sensors->watertemp_f, &new_sensors->watertemp_f, &sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_sensor_log(&sensor_port, &sensor_status, &new_sensor_status);
		Generate_ML_sensor_side_log(&sensor_port, &sensor_status, &new_sensor_status);
	}
}

/* You updated Sensor 1's position to INSIDE */
/* You updated Sensor 2's position to OUTSIDE */
void compare_sensor_setting(sys_setting_t *old_sys_setting, sys_setting_t *g_sys_setting){ 
	if(g_sys_setting->sensor_set.config.switch_zone_pos != old_sys_setting->sensor_set.config.switch_zone_pos && !g_ai_setting.is_ai_deleted && g_ai_setting.ai_workmode != 0){
		u8 ml_log_type = 0;
		u8 data[12];
		for(uint8_t i=1; i<PORT_CNT; i++){
			u8 new_bit_value = g_sys_setting->sensor_set.config.switch_zone_pos;  
			u8 old_bit_value = old_sys_setting->sensor_set.config.switch_zone_pos; 
			if(new_bit_value != old_bit_value){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				data[1] = USER_CONTROL_SENSOR;
				data[2] = AI_CONTROL_SENSOR_SIDE;
				data[3] = new_bit_value;
				Generate_ML_log(ml_log_type,data);
			}
		}
	}
}

void compare_ai_curLoad(curLoad_t *curLoad, curLoad_t *old_curLoad, ml_out_info_t adjust_reason) {
	// AI deleted/normal mode, do not print logs
	if(g_ai_setting.is_ai_deleted == 1 ){
		return;
	}
	if( g_ai_setting.ai_workmode == AI_WORKMODE_PAUSE && g_old_ai_setting.ai_workmode == AI_WORKMODE_PAUSE ) {
		return;
	}
	u8 ml_log_type = 0;
	u8 data[12];
	// 为解决开关和挡位设备，触发多条 turned on 日志
	uint8_t dev_on_off_sta[loadType_cnt]={0};	// 0-无相关设置	1-关闭 - 2-打开
	memset(dev_on_off_sta,0x00,sizeof(dev_on_off_sta));
	for(uint8_t i=1; i<PORT_CNT; i++){
		if(!is_ai_port(i)) continue;
		u16 res = _s16_(curLoad[i].load_res);
		u16 port_dev_id = ((CanDever[i].Cantype&0x00FF)<<8 |(CanDever[i].Cantype&0xFF00)>>8);
		u8 user_loadtype = curLoad[i].user_load_type;
		/* "Port 1" was turned off / Port 1" was turned on */
		if((res == 0xffff || res == 0)) {
			if(curLoad[i].port_type == 0 || !curLoad[i].plug_in) {  // removed Port
				res = _s16_(old_curLoad[i].load_res);
				port_dev_id = ((CanDever[i].Cantype & 0x00FF) << 8 | (CanDever[i].Cantype & 0xFF00) >> 8);
				user_loadtype = old_curLoad[i].user_load_type;
			}
		}
		uint8_t dev_use_type = _getTargetDummyLoadType_port(i);
			
		/* Port 1" increased to level 8. / Port 1" decreased to level 2 */
		if(curLoad[i].port_type == portType_rank && curLoad[i].objSpeed != old_curLoad[i].objSpeed){
			ml_log_type = AI_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = i;
			data[1] = res & 0xFF;         			// Low byte
			data[2] = (res >> 8) & 0xFF;  			// High byte
			data[3] = port_dev_id & 0xFF;      		// Low byte
			data[4] = (port_dev_id >> 8) & 0xFF;	// High byte
			data[5] = user_loadtype;
			u8 change_value = (curLoad[i].objSpeed > old_curLoad[i].objSpeed) ? 2 : 1;
			if (old_curLoad[i].objSpeed == 0){
				u8 switch_value = 1;
				data[6] = change_value << 6 | old_curLoad[i].objSpeed << 2 | switch_value << 1;
				data[7] = adjust_reason.inline_fan_action_info;
				if( dev_on_off_sta[dev_use_type] != (switch_value + 1) )	//开关Log
					Generate_ML_log(ml_log_type,data); 			// 档位turn on (档位0  开关on)
				dev_on_off_sta[dev_use_type] = switch_value+1;	//记录 turn on 日志
			}
			u8 switch_value = (curLoad[i].objSpeed == 0) ? 0 : 1;
			data[6] = change_value << 6 | curLoad[i].objSpeed << 2 | switch_value << 1;
			//data[7]----增加ai adjust reason
			data[7] = adjust_reason.inline_fan_action_info;
			Generate_ML_log(ml_log_type,data);
		}
		/* You changed "Port 1's" outlet device type from FAN to GROW LIGHT. */
		if(curLoad[i].load_type == loadType_switch && curLoad[i].user_load_type != old_curLoad[i].user_load_type){
			if(g_old_ai_setting.ai_workmode == 0 && g_ai_setting.ai_workmode == 1 ){
				return;
			}
			ml_log_type = USER_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = i;
			data[1] = USER_CONTROL_DEVICE_TYPE;
			data[2] = AI_CONTROL_NONE;
			data[3] = res & 0xFF;         			// Low byte
			data[4] = (res >> 8) & 0xFF;  			// High byte
			data[5] = port_dev_id & 0xFF;      		// Low byte
			data[6] = (port_dev_id >> 8) & 0xFF;	// High byte
			data[7] = user_loadtype;
			data[8] = old_curLoad[i].user_load_type;
			Generate_ML_log(ml_log_type,data);
		}
	}
	for(uint8_t i=1; i<PORT_CNT; i++){
		if(!is_ai_port(i)) continue;
		u16 res = _s16_(curLoad[i].load_res);
		u16 port_dev_id = ((CanDever[i].Cantype&0x00FF)<<8 |(CanDever[i].Cantype&0xFF00)>>8);
		u8 user_loadtype = curLoad[i].user_load_type;
		/* "Port 1" was turned off / Port 1" was turned on */
		if((res == 0xffff || res == 0)) {
			if(curLoad[i].port_type == 0 || !curLoad[i].plug_in) {  // removed Port
				res = _s16_(old_curLoad[i].load_res);
				port_dev_id = ((CanDever[i].Cantype & 0x00FF) << 8 | (CanDever[i].Cantype & 0xFF00) >> 8);
				user_loadtype = old_curLoad[i].user_load_type;
			}
		}
		uint8_t dev_use_type = _getTargetDummyLoadType_port(i);
		
		if(curLoad[i].port_type == portType_plug && curLoad[i].switchSta != old_curLoad[i].switchSta){
			ml_log_type = AI_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = i;
			data[1] = res & 0xFF;         			// Low byte
			data[2] = (res >> 8) & 0xFF;  			// High byte
			data[3] = port_dev_id & 0xFF;         	// Low byte
			data[4] = (port_dev_id >> 8) & 0xFF; 	// High byte
			data[5] = user_loadtype;
			u8 switch_value = (curLoad[i].switchSta == 0) ? 0 : 1;
			u8 level_value = switch_value;
			data[6] = curLoad[i].change<< 6 | level_value << 2 | switch_value << 1; // Switch state
			//data[7]----增加ai adjust reason
			data[7] = adjust_reason.inline_fan_action_info;
			if( dev_on_off_sta[dev_use_type] != (switch_value + 1) )	//开关Log
				Generate_ML_log(ml_log_type,data);
			dev_on_off_sta[dev_use_type] = switch_value+1;	//记录 turn on 日志
		}
	}
}

bool isExpired(ai_setting_t *newSetting,Time_Typedef* p_sys_time) {
    // Convert start_time to seconds since some reference point
    uint32_t start_seconds = newSetting->start_time.utc;

    // Convert ai_work_days to seconds
    uint32_t ai_work_seconds = newSetting->ai_work_days * 86400;

	if(0 == start_seconds || 0 == ai_work_seconds){
		return false;
	}

    // Convert cur_time to seconds since some reference point
    _unused_ uint32_t sys_seconds = rtc_to_real_utc(*p_sys_time);

    // Compare the sum of start_time and ai_work_days with cur_time
    return ((start_seconds + ai_work_seconds) < sys_seconds);
}

// Function to compare two structures the differing fields
void compare_ai_setting(ai_setting_t *newSetting, ai_setting_t *oldSetting, curLoad_t *curLoad, curLoad_t *old_curLoad, curLoad_t *old_ai_curLoad,Time_Typedef* p_sys_time) {
	u8 ml_log_type = 0;
	u8 data[12] = {0};
    if (memcmp(newSetting, oldSetting, sizeof(ai_setting_t)) != 0) {
		if (newSetting->ai_workmode != oldSetting->ai_workmode && newSetting->is_ai_deleted == 0) {
			bool not_generate_log_flag = 0;
			ml_log_type = AI_MODE_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
			data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
			data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
			data[3] = (u8)(newSetting->growth_period & 0xFF);
			if(newSetting->ai_workmode == 0){
				data[4] = AI_PAUSED;	/* Seedling" Al paused. */
				if( newSetting->pause_reason != 0 ){
					not_generate_log_flag = 1;
				}
			}
			if(newSetting->ai_workmode == 1){
				if(oldSetting->is_ai_deleted != newSetting->is_ai_deleted){
					data[4] = AI_STARTED;  /* "Seedling" Al started running */
					if(get_target_type())
						not_generate_log_flag = 1;
				}else if (oldSetting->ai_workmode == 0 || (oldSetting->ai_workmode == 2 && newSetting->tentwork_sparetime > 0)){
					data[4] = AI_RESUMED;  /* "Seedling" Al resumed. */
					if( oldSetting->pause_reason != 0 ){
						not_generate_log_flag = 1;
					}
				}else{
					not_generate_log_flag = 1;	//tent work 自然结束
				}
			}
			if(newSetting->ai_workmode == 2 && 2 != oldSetting->ai_workmode){
				data[4] = TEND_WORK_ACTIVATED; 	/* Tent Work Mode was activated. */
				if( oldSetting->pause_reason != 0 ){
					not_generate_log_flag = 1;
				}
			}
			if( not_generate_log_flag == 0 ){
				Generate_ML_log(ml_log_type,data);
			}
		}

		if(newSetting->is_ai_deleted != oldSetting->is_ai_deleted){
			ml_log_type = AI_MODE_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
			data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
			data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
			data[3] = (u8)(newSetting->growth_period & 0xFF);
			if(newSetting->is_ai_deleted == 1){
				data[4] = AI_DELETED;		/* "Seedling" Al deleted. */
				Generate_ML_log(ml_log_type,data);
			}
		}

		if(2 == newSetting -> ai_workmode){
			ml_log_type = AI_MODE_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
			data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
			data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
			data[3] = (u8)(newSetting->growth_period & 0xFF);
			if(newSetting->tentwork_sparetime == 0 && oldSetting->tentwork_sparetime > 0 ){
				data[4] = TEND_WORK_ENDED;		/* Tent Work Mode ended. Al was resumed. */
				Generate_ML_log(ml_log_type,data);
			}
		}
		if (isExpired(&g_ai_setting,p_sys_time)){
			if( log_mem_data.flag.bit.ml_end_flag == false && newSetting->is_ai_deleted == 0 ){
				log_mem_data.flag.bit.ml_end_flag = true;
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = AI_GROW_STAGE_ENDED;		/*"Seedling" Al's recommended grow stage has ended. */
				Generate_ML_log(ml_log_type,data); 	/* log */
				// Generate_ML_Msg(ml_log_type,data); 	/* msg */
			}
		}else{
			log_mem_data.flag.bit.ml_end_flag = false;
		}

		/* You updated the target range from 67° to 75° */
		if(newSetting->is_ai_deleted != 1 && newSetting->is_ai_deleted == oldSetting->is_ai_deleted){
			if (is_temp_unit_f()){
				if((newSetting->autoMode.targetTemp_F_max != oldSetting->autoMode.targetTemp_F_max 
				|| newSetting->autoMode.targetTemp_F_min != oldSetting->autoMode.targetTemp_F_min
				|| newSetting->ai_mode_sel_bits.temp_en > oldSetting->ai_mode_sel_bits.temp_en
				|| newSetting->ai_mode_sel_bits.humid_en != oldSetting->ai_mode_sel_bits.humid_en)
				&& newSetting->ai_mode_sel_bits.temp_en
				&& get_day_night_mode_type() != 2){
					ml_log_type = USER_CONTROL;
					memset(data, 0, sizeof(data));
					data[1] = get_target_type() << 7 | USER_CONTROL_TARGET;
					data[2] = AI_CONTROL_TEMP_F;
					data[3] = newSetting->autoMode.targetTemp_F_max;
					data[4] = newSetting->autoMode.targetTemp_F_min;
					Generate_ML_log(ml_log_type,data);
				}
			}else {
				if((newSetting->autoMode.targetTemp_C_max != oldSetting->autoMode.targetTemp_C_max
					|| newSetting->autoMode.targetTemp_C_min != oldSetting->autoMode.targetTemp_C_min
					|| newSetting->ai_mode_sel_bits.temp_en > oldSetting->ai_mode_sel_bits.temp_en
					|| newSetting->ai_mode_sel_bits.humid_en > oldSetting->ai_mode_sel_bits.humid_en)
					&& newSetting->ai_mode_sel_bits.temp_en
					&& get_day_night_mode_type() != 2){
					ml_log_type = USER_CONTROL;
					memset(data, 0, sizeof(data));
					data[1] = get_target_type() << 7 | USER_CONTROL_TARGET;
					data[2] = AI_CONTROL_TEMP_C;
					data[3] = newSetting->autoMode.targetTemp_C_max;
					data[4] = newSetting->autoMode.targetTemp_C_min;
					Generate_ML_log(ml_log_type,data);
				}
			}
			/* You updated the target range from 35% to 50%. */
			if((newSetting->autoMode.targetHumid_max != oldSetting->autoMode.targetHumid_max
				|| newSetting->autoMode.targetHumid_min != oldSetting->autoMode.targetHumid_min
				|| newSetting->ai_mode_sel_bits.humid_en > oldSetting->ai_mode_sel_bits.humid_en
				|| newSetting->ai_mode_sel_bits.temp_en != oldSetting->ai_mode_sel_bits.temp_en) 
				&& newSetting->ai_mode_sel_bits.humid_en
				&& get_day_night_mode_type() != 2){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				data[1] = get_target_type() << 7 | USER_CONTROL_TARGET;
				data[2] = AI_CONTROL_HUMID;
				data[3] = newSetting->autoMode.targetHumid_max;
				data[4] = newSetting->autoMode.targetHumid_min;
				Generate_ML_log(ml_log_type,data);
			}
			/* You updated the target range from 0.8kPa to 1.2kPa */
			if((newSetting->vpdMode.highVpd != oldSetting->vpdMode.highVpd
				|| newSetting->vpdMode.lowVpd != oldSetting->vpdMode.lowVpd
				|| newSetting->ai_mode_sel_bits.vpd_en > oldSetting->ai_mode_sel_bits.vpd_en)
				&& get_day_night_mode_type() != 2
				&&  newSetting->ai_mode_sel_bits.vpd_en){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				data[1] = get_target_type() << 7 | USER_CONTROL_TARGET;
				data[2] = AI_CONTROL_VPD;
				data[3] = (u8)((newSetting->vpdMode.highVpd >> 8) & 0xFF); // High byte
				data[4] = (u8)(newSetting->vpdMode.highVpd & 0xFF);       // Low byte
				data[5] = (u8)((newSetting->vpdMode.lowVpd >> 8) & 0xFF);
				data[6] = (u8)(newSetting->vpdMode.lowVpd & 0xFF);
				Generate_ML_log(ml_log_type,data);
			}
		
			/* You added Port 1: Grow Light to your Aprogramming. */
			/* You added Port 2: Heater to your Al programming */
			/* You removed Port 1: Grow Light from your Alprogramming. */
			/* You removed Port 2: Heater from your Alprogramming. */
			if (newSetting->ai_port_sel_bits != oldSetting->ai_port_sel_bits && g_ai_setting.is_ai_deleted == g_old_ai_setting.is_ai_deleted) {
				for (uint8_t i = 1; i < PORT_CNT; i++) {
					u16 res = _s16_(curLoad[i].load_res);
					u16 port_dev_id = ((CanDever[i].Cantype & 0x00FF) << 8 | (CanDever[i].Cantype & 0xFF00) >> 8);
					u8 user_loadtype = curLoad[i].user_load_type;

					u8 new_bit_value = (newSetting->ai_port_sel_bits >> (i-1)) & 0x01;
					u8 old_bit_value = (oldSetting->ai_port_sel_bits >> (i-1)) & 0x01;
					// ESP_LOGI(TAG, "Fai Find curLoad[%d] : USER_CONTROL_AI_PROGRAMMING-- %d, %d, %d, %d!", i, _s16_(curLoad[i].load_res), curLoad[i].user_load_type, _s16_(old_ai_curLoad[i].load_res), old_ai_curLoad[i].user_load_type);
					if (new_bit_value != old_bit_value) {
						if((res == 0xffff || res == 0) && new_bit_value == 0) {
							res = _s16_(old_ai_curLoad[i].load_res);
							port_dev_id = ((CanDever[i].Cantype & 0x00FF) << 8 | (CanDever[i].Cantype & 0xFF00) >> 8);
							user_loadtype = old_ai_curLoad[i].user_load_type;
						}
						// ESP_LOGI(TAG, "Fai2 Find curLoad[%d] : USER_CONTROL_AI_PROGRAMMING-- %d, %d !", i, res, user_loadtype);
						ml_log_type = USER_CONTROL;
						memset(data, 0, sizeof(data));
						data[0] = i;
						data[1] = USER_CONTROL_AI_PROGRAMMING;
						data[2] = new_bit_value;
						data[3] = res & 0xFF;          // Low byte
						data[4] = (res >> 8) & 0xFF;   // High byte
						data[5] = port_dev_id & 0xFF;  // Low byte
						data[6] = (port_dev_id >> 8) & 0xFF;  // High byte
						data[7] = user_loadtype;
						Generate_ML_log(ml_log_type, data);
					}
				}
				memcpy(old_ai_curLoad, curLoad, sizeof(curLoad_t)* PORT_CNT);// ai update
			}

			for(uint8_t i=1; i<PORT_CNT; i++)
			{	
				if(!is_ai_port(i)){
					continue;
				}
				u16 res = _s16_(curLoad[i].load_res);
				u16 port_dev_id = ((CanDever[i].Cantype&0x00FF)<<8 |(CanDever[i].Cantype&0xFF00)>>8);
				u8 user_loadtype = curLoad[i].user_load_type;

				//	ai_get_port_setting_type( newSetting, i )
				if( IsPortLoadTypeX(i, loadType_fan) ){
					if(newSetting->port_ctrl[i].fan.start_hour != oldSetting->port_ctrl[i].fan.start_hour
					|| newSetting->port_ctrl[i].fan.start_min != oldSetting->port_ctrl[i].fan.start_min
					|| newSetting->port_ctrl[i].fan.end_hour != oldSetting->port_ctrl[i].fan.end_hour
					|| newSetting->port_ctrl[i].fan.end_min != oldSetting->port_ctrl[i].fan.end_min){
						ml_log_type = USER_CONTROL;
						memset(data, 0, sizeof(data));
						data[0] = i;
						data[1] = USER_CONTROL_START_END_TIME;
						data[2] = 0;
						data[3] = res & 0xFF;         			// Low byte
						data[4] = (res >> 8) & 0xFF;  			// High byte
						data[5] = port_dev_id & 0xFF;      		// Low byte
						data[6] = (port_dev_id >> 8) & 0xFF;	// High byte
						data[7] = user_loadtype;
						data[8] = newSetting->port_ctrl[i].fan.start_hour;
						data[9] = newSetting->port_ctrl[i].fan.start_min;
						data[10] = newSetting->port_ctrl[i].fan.end_hour;
						data[11] = newSetting->port_ctrl[i].fan.end_min;
						Generate_ML_log(ml_log_type,data);
					}
				}
				if( IsPortLoadTypeX(i, loadType_growLight) && false == newSetting->port_ctrl[i].growlight.config.auto_light_en ){
					/* You updated the start time to 8:30AM and end time to 8:30PM in "Port 1" */
					if(newSetting->port_ctrl[i].growlight.start_hour != oldSetting->port_ctrl[i].growlight.start_hour
					|| newSetting->port_ctrl[i].growlight.start_min != oldSetting->port_ctrl[i].growlight.start_min
					|| newSetting->port_ctrl[i].growlight.end_hour != oldSetting->port_ctrl[i].growlight.end_hour
					|| newSetting->port_ctrl[i].growlight.end_min != oldSetting->port_ctrl[i].growlight.end_min 
					|| newSetting->port_ctrl[i].growlight.config.auto_light_en != oldSetting->port_ctrl[i].growlight.config.auto_light_en ){
						ml_log_type = USER_CONTROL;
						memset(data, 0, sizeof(data));
						data[0] = i;
						data[1] = USER_CONTROL_SUNRISE_SUNSET;
						data[2] = AI_CONTROL_START_END_TIME;
						data[3] = res & 0xFF;         			// Low byte
						data[4] = (res >> 8) & 0xFF;  			// High byte
						data[5] = port_dev_id & 0xFF;      		// Low byte
						data[6] = (port_dev_id >> 8) & 0xFF;	// High byte
						data[7] = user_loadtype;
						data[8] = newSetting->port_ctrl[i].growlight.start_hour;
						data[9] = newSetting->port_ctrl[i].growlight.start_min;
						data[10] = newSetting->port_ctrl[i].growlight.end_hour;
						data[11] = newSetting->port_ctrl[i].growlight.end_min;
						Generate_ML_log(ml_log_type,data);
					}	
				}
				
				/* You updated the sunrise/sunset duration to 2 hour 30 minutes in "Port 1" */
				if (curLoad[i].load_type != loadType_switch && IsPortLoadTypeX(i, loadType_growLight)){
					if(newSetting->port_ctrl[i].growlight.duration_hour != oldSetting->port_ctrl[i].growlight.duration_hour
					|| newSetting->port_ctrl[i].growlight.duration_min != oldSetting->port_ctrl[i].growlight.duration_min){
						ml_log_type = USER_CONTROL;
						memset(data, 0, sizeof(data));
						data[0] = i;
						data[1] = USER_CONTROL_SUNRISE_SUNSET;
						data[2] = AI_CONTROL_DURATIOM_TIME;
						data[3] = res & 0xFF;         			// Low byte
						data[4] = (res >> 8) & 0xFF;  			// High byte
						data[5] = port_dev_id & 0xFF;      		// Low byte
						data[6] = (port_dev_id >> 8) & 0xFF;	// High byte
						data[7] = user_loadtype;
						data[8] = newSetting->port_ctrl[i].growlight.duration_hour; 	// High byte
						data[9] = newSetting->port_ctrl[i].growlight.duration_min;		// Low byte
						Generate_ML_log(ml_log_type,data);
					}
				}
			}

			//	DYNAMIC_GROW_LIGHT_SCHEDULE_UPDATE
			bool sun_collection_end_flag = false;
			bool sun_dynamic_setting_new_on = false;	//动态光开启时 生成动态光的周期时间
			u8 port_num = 0;
			u8 port_bit = 0;
			for(uint8_t i=1; i<PORT_CNT; i++)
			{	
				// if(!is_ai_port(i)){
				// 	continue;
				// }
				if(IsPortLoadTypeX(i, loadType_growLight) \
					&&  newSetting->port_ctrl[i].growlight.config.auto_light_en && newSetting->port_ctrl[i].growlight.config.auto_light_data_collection_completed ){
					if( i> 0 && i < 9 ){
						port_bit |= (1<<(i-1));
					} 
					port_num = i;
					sun_collection_end_flag = true;
				}
				if( IsPortLoadTypeX(i, loadType_growLight) && newSetting->port_ctrl[i].growlight.config.auto_light_en
					&& newSetting->port_ctrl[i].growlight.config.auto_light_en != oldSetting->port_ctrl[i].growlight.config.auto_light_en ){
					sun_dynamic_setting_new_on = true;
				}
			}
			bool generate_dynamic_sun_new_schedule = false;
			uint16_t dynamic_start_min=0,dynamic_end_min=0;

			if( sun_collection_end_flag == true ){
				if( true == sun_dynamic_setting_new_on ){
					generate_dynamic_sun_new_schedule = true;
					ESP_LOGI(TAG,"Dynamic on!!!");
				}
				if( newSetting->port_ctrl[port_num].growlight.config.auto_light_data_collection_completed != oldSetting->port_ctrl[port_num].growlight.config.auto_light_data_collection_completed ){
					generate_dynamic_sun_new_schedule = true;
					ESP_LOGI(TAG,"Dynamic 24hour over!!!");
				}
				if( newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartHour != log_mem_data.dynamic_sun.start_hour ||
					newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartMin != log_mem_data.dynamic_sun.start_min 	||
					newSetting->port_ctrl[port_num].growlight.period_hour 	!= log_mem_data.dynamic_sun.period_hour 	||
					newSetting->port_ctrl[port_num].growlight.period_min 	!= log_mem_data.dynamic_sun.period_min 
					) 
				{
					log_mem_data.dynamic_sun.start_hour		= newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartHour;
					log_mem_data.dynamic_sun.start_min		= newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartMin;
					log_mem_data.dynamic_sun.period_hour	= newSetting->port_ctrl[port_num].growlight.period_hour;
					log_mem_data.dynamic_sun.period_min		= newSetting->port_ctrl[port_num].growlight.period_min;
					generate_dynamic_sun_new_schedule = true;
					ESP_LOGI(TAG,"Dynamic updata!!!");
				}

				if( generate_dynamic_sun_new_schedule == true ){
					dynamic_start_min = newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartHour*60 + newSetting->port_ctrl[port_num].growlight.dynamicGrowLightStartMin;
					dynamic_end_min = ( dynamic_start_min + newSetting->port_ctrl[port_num].growlight.period_hour*60 + newSetting->port_ctrl[port_num].growlight.period_min );
					dynamic_end_min = dynamic_end_min%(24*60);
				}
			}

			// if( newSetting->ai_workmode == AI_WORKMODE_PAUSE && oldSetting->ai_workmode == AI_WORKMODE_PAUSE ) {
			// 	generate_dynamic_sun_new_schedule = false;
			// }
			if(generate_dynamic_sun_new_schedule == true){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				uint8_t i = 0;
				data[i++] = port_bit;
				data[i++] = USER_CONTROL_DYNAMIC_SUN;
				data[i++] = AI_USR_CONTROL_DYNAMIC_SUN_UAPDATE;

				data[i++] = (u8)(dynamic_start_min/60);
				data[i++] = (u8)(dynamic_start_min%60);

				data[i++] = (u8)(dynamic_end_min/60);
				data[i++] = (u8)(dynamic_end_min%60);
				Generate_ML_log(ml_log_type,data); 	/* log */
			}
			//	-------------------------------------------
		}

		if (is_ai_setting_empty(&g_old_ai_setting)) {
			ml_log_type = AI_MODE_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
			data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
			data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
			data[3] = (u8)(newSetting->growth_period & 0xFF);
			data[4] = AI_INSIGHT_UPDATED;		/*"Seedling" AI insight has been updated. View your AI page for more info. */
			Generate_ML_Msg(ml_log_type,data); 	/* msg */
		}
    } 
	// 环境温度/湿度告警
	bool ml_warnning_limit = 0;
	if(newSetting->ai_workmode == AI_WORKMODE_PAUSE || newSetting->is_ai_deleted == true ){
		ml_warnning_limit = 1;
	}
	if(g_ai_setting.is_ai_deleted != g_old_ai_setting.is_ai_deleted){
		ml_warnning_limit = 1;
	}
	if(  ml_warnning_limit == 0 ){
		bool loadType_growLight_flag = false;
		bool loadType_fan_flag = false;
		bool loadType_co2_generator_flag = false;
		for(uint8_t i=1; i<PORT_CNT; i++) 
		{
			u8 new_bit_value = (newSetting->ai_port_sel_bits >> (i-1)) & 0x01;
			if(1 == new_bit_value && loadType_growLight ==distLoad_type[i]){
				loadType_growLight_flag = true;
			}
			if(1 == new_bit_value && loadType_fan ==distLoad_type[i]){
				loadType_fan_flag = true;
			}
			if(1 == new_bit_value && loadType_co2_generator ==distLoad_type[i]){
				loadType_co2_generator_flag = true;
			}
		}
		if(newSetting->is_ai_deleted == 1)
			return;
		if( ai_log_get_cur_temp() > temp_to_C(38) * 100){
			if(!log_mem_data.flag.bit.temp_100_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				if(is_temp_unit_f()){
					data[4] = EXTREME_TEMP_F_MAX100;		/* Seedling"Al detected unfavorable grow environment conditions (Above 100°F) */
				}else{
					data[4] = EXTREME_TEMP_C_MAX38;		/* Seedling"Al detected unfavorable grow environment conditions (Above 38°C)  */
				}
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.temp_100_flag = true;
			}
		}else{
			log_mem_data.flag.bit.temp_100_flag = false;
		}
		if( ai_log_get_cur_temp() > temp_to_C(55) * 100 && loadType_growLight_flag){
			if(!log_mem_data.flag.bit.temp_130_growlight_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				if(is_temp_unit_f()){
					data[4] = EXTREME_TEMP_F_MAX130;		
				}else{
					data[4] = EXTREME_TEMP_C_MAX55;		
				}
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.temp_130_growlight_flag = true;
			}
		}else{
			log_mem_data.flag.bit.temp_130_growlight_flag = false;
		}
		if(ai_log_get_cur_humid() > 90*100 && !loadType_fan_flag){
			if(!log_mem_data.flag.bit.humid_90_without_fan_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = EXTREME_HUMID_EXCEPT_FAN_MAX90;		/* "Seedling" Al detected extreme conditions (Above 80%) */
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.humid_90_without_fan_flag = true;
			}
		}else{
			log_mem_data.flag.bit.humid_90_without_fan_flag = false;
		}
		if(ai_log_get_cur_humid() > 90*100 && loadType_fan_flag){
			if(!log_mem_data.flag.bit.humid_fan_90_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = EXTREME_HUMID_FAN_MAX90;		/* "Seedling" Al detected extreme conditions (Below 80%) */
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.humid_fan_90_flag = true;
			}
		}else{
			log_mem_data.flag.bit.humid_fan_90_flag = false;
		}
		
	#if(_TYPE_of(VER_HARDWARE) == _TYPE_(_GROWBOX))
		if(ai_log_get_cur_humid() > 90*100){
			if(!log_mem_data.flag.bit.humid_90_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = EXTREME_HUMID_MAX90_GROWBOX;		/* "Seedling" Al detected extreme conditions (Below 80%) */
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.humid_90_flag = true;
			}
		}else{
			log_mem_data.flag.bit.humid_90_flag = false;
		}
		if( ai_log_get_cur_temp() > 130 * 100){
			if(!log_mem_data.flag.bit.temp_130_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = EXTREME_TEMP_F_MAX130_GROWBOX;		/* Seedling"Al detected unfavorable grow environment conditions (Above 100°F) */
				
				Generate_ML_log(ml_log_type,data); 	/* log */
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.temp_130_flag = true;
			}
		}else{
			log_mem_data.flag.bit.temp_130_flag = false;
		}
	#endif
		if( loadType_co2_generator_flag ) {
			if(ml_out_info.ai_inisight_bit.co2_genarator_over_time){
				if(!log_mem_data.flag.bit.co2_reg_safety_shutoff_flag){
					ml_log_type = AI_MODE_CONTROL;
					memset(data, 0, sizeof(data));
					data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
					data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
					data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
					data[3] = (u8)(newSetting->growth_period & 0xFF);
					data[4] = CO2_SAFETY_SHUTOFF;		
					Generate_ML_Msg(ml_log_type,data); 	/* msg */
					log_mem_data.flag.bit.co2_reg_safety_shutoff_flag = true;
				}
			}else{
				log_mem_data.flag.bit.co2_reg_safety_shutoff_flag = false;
			}
		}
			
		for(uint8_t i=1; i<PORT_CNT; i++){
			if(!is_ai_port(i) )
				continue;
			if(get_using_devtype(i) != loadType_water_pump)
				continue;
			if(g_sensors.soil_humid.dectected && (ml_out_info.ai_inisight_bit.water_pum_over_time && newSetting->port_ctrl[i].water_pump.mode == 1)){
				if(!log_mem_data.flag.bit.soil_low_flag){
					ml_log_type = AI_MODE_CONTROL;
					memset(data, 0, sizeof(data));
					data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
					data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
					data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
					data[3] = (u8)(newSetting->growth_period & 0xFF);
					data[4] = SOIL_ALERT;		
					Generate_ML_Msg(ml_log_type,data); 	/* msg */
					log_mem_data.flag.bit.soil_low_flag = true;
				}
			}else{
				log_mem_data.flag.bit.soil_low_flag = false;
			}
			if(g_sensors.water_level.dectected && (ml_out_info.ai_inisight_bit.water_pum_over_time && newSetting->port_ctrl[i].water_pump.mode == 0)){
				if(!log_mem_data.flag.bit.water_no_flag){
					ml_log_type = AI_MODE_CONTROL;
					memset(data, 0, sizeof(data));
					data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
					data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
					data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
					data[3] = (u8)(newSetting->growth_period & 0xFF);
					data[4] = WATER_DETECT_ALERT;		
					Generate_ML_Msg(ml_log_type,data); 	/* msg */
					log_mem_data.flag.bit.water_no_flag = true;
				}
			}else{
				log_mem_data.flag.bit.water_no_flag = false;
			}
		}
	}
	if(newSetting->is_ai_deleted == 1)
		return;
	if(ai_get_sensor_is_selected(newSetting, CO2)){
		if((ai_log_get_cur_co2() < 200 && ai_log_get_cur_co2() > 0)){
			if(!log_mem_data.flag.bit.co2_200_flag){
				ml_log_type = AI_MODE_CONTROL;
				memset(data, 0, sizeof(data));
				data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
				data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
				data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
				data[3] = (u8)(newSetting->growth_period & 0xFF);
				data[4] = CO2_ALERT;		
				Generate_ML_Msg(ml_log_type,data); 	/* msg */
				log_mem_data.flag.bit.co2_200_flag = true;
			}
		}else{
			log_mem_data.flag.bit.co2_200_flag = false;
		}
		if(ai_log_get_cur_co2() >= 5000){
			uint8_t co2_regulator_flag = 0;
			for(uint8_t i=1; i<PORT_CNT; i++) {
				if(!is_ai_port(i))
					continue;
				if(get_using_devtype(i) != loadType_co2_generator)
					continue;
				co2_regulator_flag = 1;
			}
			if(co2_regulator_flag){
				if(!log_mem_data.flag.bit.high_co2_with_reg_flag){
					ml_log_type = AI_MODE_CONTROL;
					memset(data, 0, sizeof(data));
					data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
					data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
					data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
					data[3] = (u8)(newSetting->growth_period & 0xFF);
					data[4] = HIGH_CO2_WITH_REGULATOR;		
					Generate_ML_Msg(ml_log_type,data); 	/* msg */
					log_mem_data.flag.bit.high_co2_with_reg_flag = true;
				}	
			}
			else{
				if(!log_mem_data.flag.bit.high_co2_without_reg_flag){
					ml_log_type = AI_MODE_CONTROL;
					memset(data, 0, sizeof(data));
					data[0] = g_ai_night_mode_setting.config.en << 6 | (is_ai_night_mode() << 5) | newSetting->is_easy_mode << 4 | newSetting->reserved;
					data[1] = (u8)((_u16_(newSetting->plant_type) >> 8) & 0xFF);
					data[2] = (u8)(_u16_(newSetting->plant_type) & 0xFF);
					data[3] = (u8)(newSetting->growth_period & 0xFF);
					data[4] = HIGH_CO2_WITHOUT_REGULATOR;		
					Generate_ML_Msg(ml_log_type,data); 	/* msg */
					log_mem_data.flag.bit.high_co2_without_reg_flag = true;
				}
			}
		}
		else{
			log_mem_data.flag.bit.high_co2_without_reg_flag = false;
			log_mem_data.flag.bit.high_co2_with_reg_flag = false;
		}
	}
}

// Function to compare two structures the differing fields
void compare_ai_night_mode_setting(ai_night_mode_setting_t *newSetting, ai_night_mode_setting_t *oldSetting) 
{
	u8 ml_log_type = 0;
	u8 data[12] = {0};

	// AI deleted/normal mode, do not print logs
	if(g_ai_setting.is_ai_deleted == 1 ){
		return;
	}
	if( g_ai_setting.ai_workmode == AI_WORKMODE_PAUSE && g_old_ai_setting.ai_workmode == AI_WORKMODE_PAUSE ) {
		return;
	}
    if (memcmp(newSetting, oldSetting, sizeof(ai_night_mode_setting_t)) != 0) {
		if (newSetting->config.en != oldSetting->config.en ) {
			bool not_generate_log_flag = 0; 
			ml_log_type = AI_MODE_CONTROL;
			memset(data, 0, sizeof(data));
			data[0] = newSetting->config.en << 6 | (is_ai_night_mode() << 5) | g_ai_setting.is_easy_mode << 4 | g_ai_setting.reserved ;  // ????bit5: day/night mode
			data[1] = (u8)((_u16_(g_ai_setting.plant_type) >> 8) & 0xFF);
			data[2] = (u8)(_u16_(g_ai_setting.plant_type) & 0xFF);
			data[3] = (u8)(g_ai_setting.growth_period & 0xFF);
			if(newSetting->config.en == 0){
				data[4] = NIGHT_MODE_DISABLED;	
			}
			else if(newSetting->config.en == 1){
				data[4] = NIGHT_MODE_ENABLED;
			}
			if(!is_during_night_mode_time())
				not_generate_log_flag = 1;
			if(!not_generate_log_flag)
				Generate_ML_log(ml_log_type,data);
		}
			/* You updated the target range from 67° to 75° */
		if (is_temp_unit_f()){
			if((newSetting->autoMode.targetTemp_F_max != oldSetting->autoMode.targetTemp_F_max 
			|| newSetting->autoMode.targetTemp_F_min != oldSetting->autoMode.targetTemp_F_min)
			&& get_day_night_mode_type() != 1){
//			|| newSetting->ai_mode_sel_bits.temp_en > oldSetting->ai_mode_sel_bits.temp_en
//			|| newSetting->ai_mode_sel_bits.vpd_en > oldSetting->ai_mode_sel_bits.vpd_en) 
//			&& (newSetting->ai_mode_sel_bits.temp_en || newSetting->ai_mode_sel_bits.vpd_en)){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				data[1] = get_target_type() << 6 | USER_CONTROL_TARGET;
				data[2] = AI_CONTROL_TEMP_F;
				data[3] = newSetting->autoMode.targetTemp_F_max;
				data[4] = newSetting->autoMode.targetTemp_F_min;
				Generate_ML_log(ml_log_type,data);
			}
		}else {
			if((newSetting->autoMode.targetTemp_C_max != oldSetting->autoMode.targetTemp_C_max
				|| newSetting->autoMode.targetTemp_C_min != oldSetting->autoMode.targetTemp_C_min)
				&& get_day_night_mode_type() != 1){
//				|| newSetting->ai_mode_sel_bits.temp_en > oldSetting->ai_mode_sel_bits.temp_en
//				|| newSetting->ai_mode_sel_bits.vpd_en > oldSetting->ai_mode_sel_bits.vpd_en)
//				&& (newSetting->ai_mode_sel_bits.temp_en || newSetting->ai_mode_sel_bits.vpd_en)){
				ml_log_type = USER_CONTROL;
				memset(data, 0, sizeof(data));
				data[1] = get_target_type() << 6 | USER_CONTROL_TARGET;
				data[2] = AI_CONTROL_TEMP_C;
				data[3] = newSetting->autoMode.targetTemp_C_max;
				data[4] = newSetting->autoMode.targetTemp_C_min;
				Generate_ML_log(ml_log_type,data);
			}
		}
		/* You updated the target range from 35% to 50%. */
		if((newSetting->autoMode.targetHumid_max != oldSetting->autoMode.targetHumid_max
			|| newSetting->autoMode.targetHumid_min != oldSetting->autoMode.targetHumid_min)
			&& get_day_night_mode_type() != 1){
//			|| newSetting->ai_mode_sel_bits.humid_en > oldSetting->ai_mode_sel_bits.humid_en
//			|| newSetting->ai_mode_sel_bits.vpd_en > oldSetting->ai_mode_sel_bits.vpd_en) 
//			&& (newSetting->ai_mode_sel_bits.humid_en || newSetting->ai_mode_sel_bits.vpd_en)){
			ml_log_type = USER_CONTROL;
			memset(data, 0, sizeof(data));
			data[1] = get_target_type() << 6 | USER_CONTROL_TARGET;
			data[2] = AI_CONTROL_HUMID;
			data[3] = newSetting->autoMode.targetHumid_max;
			data[4] = newSetting->autoMode.targetHumid_min;
			Generate_ML_log(ml_log_type,data);
		}
		/* You updated the target range from 0.8kPa to 1.2kPa */
		if((newSetting->vpdMode.highVpd != oldSetting->vpdMode.highVpd
			|| newSetting->vpdMode.lowVpd != oldSetting->vpdMode.lowVpd)
			&& get_day_night_mode_type() != 1){
//			|| newSetting->ai_mode_sel_bits.vpd_en > oldSetting->ai_mode_sel_bits.vpd_en)
//			&&  newSetting->ai_mode_sel_bits.vpd_en){
			ml_log_type = USER_CONTROL;
			memset(data, 0, sizeof(data));
			data[1] = get_target_type() << 6 | USER_CONTROL_TARGET;
			data[2] = AI_CONTROL_VPD;
			data[3] = (u8)((newSetting->vpdMode.highVpd >> 8) & 0xFF); // High byte
			data[4] = (u8)(newSetting->vpdMode.highVpd & 0xFF);       // Low byte
			data[5] = (u8)((newSetting->vpdMode.lowVpd >> 8) & 0xFF);
			data[6] = (u8)(newSetting->vpdMode.lowVpd & 0xFF);
			Generate_ML_log(ml_log_type,data);
		}
    } 
}

void ml_log_default_setting()
{
	memset(&old_ai_curLoad[0], 0, sizeof(curLoad_t)* PORT_CNT);
	memset(&old_curLoad[0], 0, sizeof(curLoad_t)* PORT_CNT);
	memset(&g_old_ai_setting, 0, sizeof(ai_setting_t));
	memset(&g_old_ai_night_mode_setting, 0, sizeof(ai_night_mode_setting_t));
	memset(&old_sys_setting, 0, sizeof(sys_setting_t));
	memset(&g_sensors_old, 0, sizeof(sensors_t));
	log_mem_data.flag.bit.ml_end_flag = false;
	log_mem_data.flag.bit.temp_100_flag = false;
	log_mem_data.flag.bit.temp_130_growlight_flag = false;
	log_mem_data.flag.bit.humid_90_without_fan_flag = false;
	log_mem_data.flag.bit.humid_fan_90_flag = false;
	log_mem_data.flag.bit.temp_130_flag = false;
	log_mem_data.flag.bit.humid_90_flag = false;
	log_mem_data.flag.bit.co2_200_flag = false;
	log_mem_data.flag.bit.soil_low_flag = false;
	log_mem_data.flag.bit.water_no_flag = false;
	log_mem_data.dynamic_sun.start_hour = 0;
	log_mem_data.dynamic_sun.start_min = 0;
	log_mem_data.dynamic_sun.period_hour = 0;
	log_mem_data.dynamic_sun.period_min = 0;
	clearAllLogs();
}

void compare_generate_mL_log(Time_Typedef sys_time, uint8_t* load_type_list ){
	/* 比对生成日志 */
	for(uint8_t i=0; i<PORT_CNT; i++ ){
		distLoad_type[i] = load_type_list[i];
	}
	if (is_curLoad_empty(&old_ai_curLoad[0])) {
		memcpy(&old_ai_curLoad[0], &curLoad[0], sizeof(curLoad_t)* PORT_CNT);
	}

	if (is_g_sensors_empty(&g_sensors_old)) {
        // g_old_ai_setting is empty, copy g_ai_setting to it
        memcpy(&g_sensors_old, &g_sensors, sizeof(sensors_t));
    } else if(!g_ai_setting.is_ai_deleted && g_ai_setting.ai_sensor_sel_bits > 0) {
        // g_old_ai_setting is not empty, compare and print differences
        compare_g_sensors(&g_sensors_old, &g_sensors);
    }

	if (is_curLoad_empty(&old_curLoad[0])) {
        // g_old_ai_setting is empty, copy g_ai_setting to it
        memcpy(&old_curLoad[0], &curLoad[0], sizeof(curLoad_t)* PORT_CNT);
    } else {
        // g_old_ai_setting is not empty, compare and print differences
        compare_ai_curLoad(&curLoad[0], &old_curLoad[0], ml_out_info);
    }

	if (is_ai_setting_empty(&g_old_ai_setting)) {
        // g_old_ai_setting is empty, copy g_ai_setting to it
        memcpy(&g_old_ai_setting, &g_ai_setting, sizeof(ai_setting_t));
    } else {
        // g_old_ai_setting is not empty, compare and print differences
        compare_ai_setting(&g_ai_setting, &g_old_ai_setting, &curLoad[0], &old_curLoad[0], &old_ai_curLoad[0],&sys_time);
    }

	// 检测AI night mode设置参数的变化并生成日志
	if (is_ai_night_mode_setting_empty(&g_old_ai_night_mode_setting)) {
        // g_old_ai_night_mode_setting is empty, copy g_ai_night_mode_setting to it
        memcpy(&g_old_ai_night_mode_setting, &g_ai_night_mode_setting, sizeof(g_ai_night_mode_setting));
    } else {
        // g_old_ai_night_mode_setting is not empty, compare and print differences
        compare_ai_night_mode_setting(&g_ai_night_mode_setting, &g_old_ai_night_mode_setting);
    }
	// Copy new setting to old setting
	memcpy(&old_curLoad, &curLoad, sizeof(curLoad_t)* PORT_CNT);
	memcpy(&g_old_ai_setting, &g_ai_setting, sizeof(ai_setting_t));
	memcpy(&g_old_ai_night_mode_setting, &g_ai_night_mode_setting, sizeof(ai_night_mode_setting_t));
	memcpy(&old_sys_setting, &g_sys_setting, sizeof(sys_setting_t));
	memcpy(&g_sensors_old, &g_sensors, sizeof(sensors_t));
}

