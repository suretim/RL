/*
****************************************************************************************************
*
*	模块名称 ：
*	文件名称 ：
*	版    本 ： V1.0
*	说    明 ：
*
*	修改记录 ：
*		版本号  日期        作者     说明
*		V1.0    2023-05-20	Peter/Charles/Jeff
*
*	Copyright (C), 2023-2033, COOLTRON
*
****************************************************************************************************
*/
#include "esp_log.h"
#include "io_iic.h"
#include "types.h"
#include "sensor_driver.h"


typedef enum
{
    BH1750_PWR_OFF = 0x00, // 断电，无激活状态
    BH1750_PWR_ON  = 0x01, // 通电，等待测量(内部寄存器地址)
    BH1750_RST     = 0x07, // 重置数据寄存器，在通电时有效，用来清除之前的测量结果

    BH1750_CON_H   = 0x10, // 连续高分辨率模式，分辨率1lx,测量时间120ms
    BH1750_CON_H2  = 0x11, // 连续高分辨率模式，0.5lx,120ms
    BH1750_CON_L   = 0x13, // 连续低分辨率模式，4lx,16ms

    BH1750_ONE_H   = 0x20, // 一次高分辨率模式，1lx,120ms，测量后模块转为 PWR_DOWN
    BH1750_ONE_H2  = 0x21, // 一次高分辨率模式，0.5lx,120ms，测量后模块转为 PWR_DOWN
    BH1750_ONE_L   = 0x23, // 一次低分辨率模式，4lx,16ms，测量后模块转为 PWR_DOWN

} BH1750_CMD;

enum
{
    mode_con_h,
    mode_con_h2,
    mode_con_l,
    mode_one_h,
    mode_one_h2,
    mode_one_l,
};

//-----------------------------------
#define TAG "sensor_light_bh1750 "

#define bh1750_log(TAG, format, ...)  ESP_LOGI(TAG, format, ##__VA_ARGS__)

//-----------------------------------

static uint8_t s_MTReg[SENSOR_PORT_CNT] = {69,69};          /* 灵敏度倍率 ，默认值为 69 */
static uint8_t s_Mode[SENSOR_PORT_CNT]  = {mode_con_h2, mode_con_h2}; /* 测量模式 */

#define BH1750_I2C_ADDR 0x23

//static uint8_t iic_num = 0;
#define iic_num   sen_obj[SENSOR_BH1750].iic_num
//#define step      sen_obj[SENSOR_BH1750].init_step[iic_num]

static float Lux[SENSOR_PORT_CNT];

static bool bh1750_read_bytes(uint8_t * pbuffer, uint16_t length)
{
    iic_sel_port_speed(iic_num, 6);
    return iic_reads((BH1750_I2C_ADDR << 1), pbuffer, length);
}

static bool bh1750_write_bytes(uint8_t * pbuffer, uint16_t length)
{
    iic_sel_port_speed(iic_num, 6);
    return iic_writes((BH1750_I2C_ADDR << 1), pbuffer, length);
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
static bool bh1750_writecmd(uint8_t cmd)
{
    //
    return bh1750_write_bytes(&cmd, 1);
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明： 修改BH1750测量模式，决定测量分辨率
*	参    数： _ucMode : 测量模式 值域(1，2，3)
*	返 回 值：
****************************************************************************************************
*/
static bool bh1750_setmode(uint8_t _ucMode)
{
    uint8_t cmd = BH1750_CON_H;

    switch (_ucMode)
    {
        case mode_con_h : cmd = BH1750_CON_H; break;

        case mode_con_h2 : cmd = BH1750_CON_H2; break;

        case mode_con_l : cmd = BH1750_CON_L; break;

        case mode_one_h : cmd = BH1750_ONE_H; break;

        case mode_one_h2 : cmd = BH1750_ONE_H2; break;

        case mode_one_l : cmd = BH1750_ONE_L; break;

        default : cmd = BH1750_CON_H; break;
    }

    s_Mode[iic_num] = _ucMode;

    if (!bh1750_writecmd(cmd))
    {
        return false;
    }
    return true;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明： 调节BH1750测量灵敏度
*	参    数： _ucMTReg : 量程倍率.  值域【31，254】，值越大 灵敏度越高
*	返 回 值：
****************************************************************************************************
*/
static bool bh1750_adjustsensitivity(uint8_t _ucMTReg)
{
    if (_ucMTReg <= 31)
    {
        _ucMTReg = 31;
    }
    else if (_ucMTReg >= 254)
    {
        _ucMTReg = 254;
    }

    s_MTReg[iic_num] = _ucMTReg;

    /* Changing High bit of MTreg */
    if (!bh1750_writecmd(0x40 + (s_MTReg[iic_num] >> 5))) /* 更改高 3bit */
    {
        return false;
    }
    /* Changing Low bit of MTreg */
    if (!bh1750_writecmd(0x60 + (s_MTReg[iic_num] & 0x1F))) /* 更改低 5bit */
    {
        return false;
    }
    /*　更改量程范围后，需要重新发送命令设置测量模式　*/
    if (!bh1750_setmode(s_Mode[iic_num]))
    {
        return false;
    }
    return true;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：  读取BH1750测量结果, 连续测量模式下，之后主程序可以定时调用本函数读取光强度数据，间隔时间需要大于180ms
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
static bool bh1750_readdata(uint16_t * light)
{
    uint8_t buf[2] = {0};
    uint16_t Light = 0;

    if (!bh1750_read_bytes(buf, 2))
    {
        return false;
    }

    Light  = (buf[0] << 8) + buf[1];
    *light = Light;

    return true;
}

static bool bh1750_check(void)
{
    iic_sel_port_speed(iic_num, 6);
    return iic_check_device(BH1750_I2C_ADDR << 1);
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
bool bh1750_init(bool * iic_ok)
{
    static uint8_t step[SENSOR_PORT_CNT] = {0,0};

    *iic_ok             = true;

    switch (step[iic_num])
    {
        case 0 :
			if (!(*iic_ok = bh1750_check())) 
                return false;
            if (!(*iic_ok = bh1750_writecmd(BH1750_PWR_ON))) /* 芯片上电 */
                return false;
            step[iic_num] = 1;
            break;
        case 1 :
            if (!(*iic_ok = bh1750_setmode(mode_con_h2))) /* 高分辨率连续测量 */
                return false;
            step[iic_num] = 2;
            break;
        case 2 :
            if (!(*iic_ok = bh1750_adjustsensitivity(69))) /* 芯片缺省灵敏度倍率 = 69 */
                return false;
            step[iic_num] = 3;
            break;
        default : break;
    }

    if (3 == step[iic_num])
    {
        step[iic_num] = 0;
        Lux[iic_num]  = 0;
        return true;
    }

    return false;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明：
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
static bool bh1750_reinit_to_read(bool * iic_ok)
{
    *iic_ok = true;

    if (!(*iic_ok = bh1750_writecmd(BH1750_PWR_ON))) /* 芯片上电 */
    {
        return false;
    }
    if (!(*iic_ok = bh1750_setmode(mode_con_h2)))    /* 高分辨率连续测量 */
    {
        return false;
    }
    if (!(*iic_ok = bh1750_adjustsensitivity(69)))   /* 芯片缺省灵敏度倍率 = 69 */
    {
        return false;
    }

    return true;
}

/*
****************************************************************************************************
*	函数名称：
*	功能说明： 读取BH1750测量结果, 并转换为 Lux单位
*	参    数：
*	返 回 值：
****************************************************************************************************
*/
bool bh1750_read(float * lx, float * para2, float * para3, float * para4, float * para5)
{
    uint16_t usLight          = 0;
    float lux_temp            = 0;

    static uint8_t cnt_value0[SENSOR_PORT_CNT] = {0,0};
    static uint8_t step[SENSOR_PORT_CNT]       = {0,0};
    bool ret                  = false;
    bool iic_ok               = false;

    /*
        光照传感器 和 CO2 传感器在同一 PCB 板， CO2 开始读取后，光照传感器好像有被影响，
        直接进行数据的读取 会返回 0 ， 原因暂时不明

        解决办法：　每次都进行初始化之后再进行读取
    */
    if (step[iic_num] == 0)
    {
        if (bh1750_reinit_to_read(&iic_ok))
        {
            ret  = true;
            step[iic_num] = 1;
        }
    }
    else if (step[iic_num] == 1)
    {
        if (!bh1750_readdata(&usLight))
        {
            bh1750_log(TAG, "%s >>>> bh1750_readdata ERR", __func__);
        }
        else
        {
            bh1750_log(TAG, "%s >>>> usLight = %d", __func__, usLight);

            /* 计算光强度 = 16位寄存器值 / 1.2  * (69 / X) */

            // lux_temp = usLight * (((float)1 / 1.2) * ((float)69 / s_MTReg));
            lux_temp = (float)(usLight * 5 * 69) / (6 * s_MTReg[iic_num]);

            if (s_Mode[iic_num] == mode_con_h2 || s_Mode[iic_num] == mode_one_h2) /* 高分辨率测量模式2 */
            {
                lux_temp = lux_temp / 2;
            }
            else
            {
                ; /* 不必除2 */
            }

            /* 处理异常读取到的 0 值 */
            if (lux_temp == 0)
            {
                if (cnt_value0[iic_num] > 2) // 连续读取到多次 0 ， 才认为是真正读取到数据 0
                {
                    Lux[iic_num] = 0;
                }
                else
                {
                    cnt_value0[iic_num]++;
                }
            }
            else
            {
                cnt_value0[iic_num] = 0;
                Lux[iic_num]        = lux_temp;
            }
            ret = true;
        }
        step[iic_num] = 0;
    }

//    *lx = Lux[iic_num];
	*lx = Lux[iic_num] * 100 / 1138 + 0.05;  //计算百分比
	if(*lx > 100.0)  
		*lx = 100.0;
    return ret;
}
