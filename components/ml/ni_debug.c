#include "esp_log.h"
#include "string.h"
#include "math.h"  
  
#include <stdio.h>
#include <stdlib.h> 

 
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h" 
//#include "freertos/FreeRTOS.h"
//#include "freertos/task.h"
//#include <time.h>
//#include <sys/time.h>
#include "ml_pid.h"
#include "ni_debug.h"
#include "pso.h"
extern float hvac_margin[];       // 若是全域陣列
extern float gain_adj_ctrl;       // 若是全域變數
extern struct pso_optimizer pso ;


struct tent_optimizer
{
	  
	float			buf[60];
	unsigned int	buf_idx, buf_cnt  ; 
	
};
struct tent_optimizer tent = {0};
//extern SVDStruct svd;             // 若是結構體
union un_double
{
	unsigned char 	buf[8];
	double			dx;
	float		  	fx;
	unsigned int	wx;
};

struct st_name_addr_type {
	char 			*name;		
	unsigned int 	addr;		
	unsigned char	type;		
};

static float		test_t_feed = 0;
static float		test_h_feed = 0;
static unsigned int	test_feed_idx = 0;
static float		test_t_target = 0;
static float		test_h_target = 0;

// static float		test_temp_input_rate = 1.0f;
// static float		test_humi_input_rate = 1.0f;
static unsigned int	test_gear_mode = 0;			//0--多挡位,1--单9档
static unsigned int test_humi_cycle_num = 4;//2;

static unsigned int gain_adj_size = 8;	//30
static float 		gain_adj_rate_min = 1000;
static float 		gain_adj_rate_max = 10000;
static float 		gain_adj_rate_change = 1000;
const struct st_name_addr_type debug_name_addr_type[] = {
	{"test_t_feed",				(unsigned int)&test_t_feed,				1},
	{"test_h_feed",				(unsigned int)&test_h_feed,				1},
	{"test_feed_idx",			(unsigned int)&test_feed_idx,			32},
	{"test_t_target",			(unsigned int)&bp_pid_th.v_outside,			1},
	{"test_h_target",			(unsigned int)&bp_pid_th.v_feed,			1},

	{"pid_t_target",			(unsigned int)&bp_pid_th.t_target,		1},
	{"pid_h_target",			(unsigned int)&bp_pid_th.h_target,		1},
	{"pid_v_target",			(unsigned int)&bp_pid_th.v_target,		1},
	{"pid_t_rate",				(unsigned int)&pso.swarm[0].position[DEV_TU],	1},
	{"pid_h_rate",				(unsigned int)&pso.swarm[0].position[DEV_HU],	1},
	{"pid_update_rate",			(unsigned int)&bp_pid_th.update_rate,	1},
	{"pid_dt_rate",				(unsigned int)&pso.swarm[0].position[DEV_TD], 1},
	{"pid_dh_rate",				(unsigned int)&pso.swarm[0].position[DEV_HD],	1},
	{"pid_tmr",					(unsigned int)&bp_pid_th.tmr,			32},

	{"test_temp_input_rate",	(unsigned int)&hvac_margin[ENV_T],	1},
	{"test_humi_input_rate",	(unsigned int)&hvac_margin[ENV_H],	1},
	{"test_gear_mode",			(unsigned int)&test_gear_mode,			32},
	{"test_humi_cycle_num",		(unsigned int)&test_humi_cycle_num,		32},

	{"gain_adj_size",			(unsigned int)&gain_adj_size,			32},
	{"gain_adj_rate_min",		(unsigned int)&gain_adj_rate_min,		1},
	{"gain_adj_rate_max",		(unsigned int)&gain_adj_rate_max,		1},
	{"gain_adj_rate_change",	(unsigned int)&gain_adj_rate_change,	1},
	{"gain_adj_ctrl",			(unsigned int)&gain_adj_ctrl,			32},


	//{"pso_global_idx",			(unsigned int)&pso.global_idx,				32},
	{"global_fitness",			(unsigned int)&pso.global_bestval,	            1},
	{"global_best_t",			(unsigned int)&pso.global[0].pos[0],		1},
	{"global_best_dt",			(unsigned int)&pso.global[0].pos[1],		1},
	{"global_best_h",			(unsigned int)&pso.global[0].pos[2],		1},
	{"global_best_dh",			(unsigned int)&pso.global[0].pos[3],		1},

	//{"pso_test_req",			(unsigned int)&pso.test_req,				32},
	//{"pso_buf_cnt",				(unsigned int)&pso.buf_cnt,					32},
	{"pso_swarm_idx",			(unsigned int)&pso.swarm_idx,				32},
	{"swarm_fitness",			(unsigned int)&pso.swarm[0].best_mae,	1},
	{"swarm_best_t",			(unsigned int)&pso.swarm[1].best_mae,	1},
	{"swarm_best_dt",			(unsigned int)&pso.swarm[2].best_mae,	1},
	{"swarm_best_h",			(unsigned int)&pso.swarm[3].best_mae,	1},
	{"swarm_best_dh",			(unsigned int)&pso.swarm[0].best_pos[3],	1},
	{"swarm_pos_t",				(unsigned int)&pso.swarm[0].position[0],	1},
	{"swarm_pos_dt",			(unsigned int)&pso.swarm[0].position[1],	1},
	{"swarm_pos_h",				(unsigned int)&pso.swarm[0].position[2],	1},
	{"swarm_pos_dh",			(unsigned int)&pso.swarm[0].position[3],	1},
	{"swarm_vel_t",				(unsigned int)&pso.swarm[0].velocity[0],	1},
	{"swarm_vel_dt",			(unsigned int)&pso.swarm[0].velocity[1],	1},
	{"swarm_vel_h",				(unsigned int)&pso.swarm[0].velocity[2],	1},
	{"swarm_vel_dh",			(unsigned int)&pso.swarm[0].velocity[3],	1},

};


void  dbg_init(void)
{
	static const int RX_BUF_SIZE = 1024;

	#if dbg_serial_sel    ==  dbg_serial_uart0
	#define RXD_PIN (GPIO_NUM_44)
	#define TXD_PIN (GPIO_NUM_43)

    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(UART_NUM_0, RX_BUF_SIZE * 2, RX_BUF_SIZE * 2, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uart_config);
    uart_set_pin(UART_NUM_0, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
	#else
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .rx_buffer_size = RX_BUF_SIZE * 2,
        .tx_buffer_size = RX_BUF_SIZE * 2,
    };

    usb_serial_jtag_driver_install(&usb_serial_jtag_config);
	#endif
}

int  dbg_read(void *buf, unsigned int len)
{
	#if dbg_serial_sel == dbg_serial_uart0
	return uart_read_bytes(UART_NUM_0, buf, len, pdMS_TO_TICKS(50));
	#else
	return usb_serial_jtag_read_bytes(buf, len, pdMS_TO_TICKS(50));
	#endif
}

  

unsigned int str2num(char *buf, unsigned int *wx, double *dx)
{
	unsigned int 	i, base;
	char			c;

	if((buf == NULL)||((wx == NULL)&&(dx == NULL)))
		return c_ret_nk;

	for(base = 10, i = 0; buf[i] != 0; i++)
	{
		c = buf[i];

		if((c >= '0')&&(c <= '9'))
		{
			continue;
		}
		else if((c == 'x')||(c == 'X'))
		{
			if((i == 0)||(buf[0] != '0')||(base != 10))
				return c_ret_nk;
			base = 16;	
		}
		else if(((c >= 'a')&&(c <= 'f'))||((c >= 'A')&&(c <= 'A')))
		{
			if(base != 16)
				return c_ret_nk;
		}	
		else if(c == '.')
		{
			if((i == 0)||(buf[i+1] == 0)||(base != 10))
				return c_ret_nk;
			base = 15;	
		}
		else
		{
			return c_ret_nk;
		}
	}
	if(base == 15)
	{
		if(dx != NULL) 
		{
			*dx = strtod(buf, NULL);
			if(wx != NULL)
				*wx = (unsigned int)(*dx + 0.25); 
		}
		else
			*wx = (unsigned int)strtod(buf, NULL);
	}
	else
	{
		if(wx != NULL)
		{
			*wx = strtoul(buf, NULL, base);
			if(dx != NULL)
				*dx = (double)*wx;
		}
		else
			*dx = (double)strtoul(buf, NULL, base);
	}
	return c_ret_ok;
}

//name,val/?
unsigned int dbg_parse_var(char *buf)
{
	char 				name_str[32], val_str[16];
	unsigned int 		i, value, typ, addr;
	double 				dx;

    if(sscanf(buf, "%[^,],%s", name_str, val_str) != 2)
	{
        return c_ret_nk;
	}

	for(i = 0; i < sizeof(debug_name_addr_type)/sizeof(debug_name_addr_type[0]); i++) 
	{
		if(strcmp(debug_name_addr_type[i].name, name_str) != 0) 
		{
			continue;
		}
		
		addr = debug_name_addr_type[i].addr;
		typ = debug_name_addr_type[i].type;

		if((name_str[0] == 's')&&(strstr(name_str, "swarm_") != NULL))
		{
			addr += pso.swarm_idx * sizeof(struct pso_particle);
		}	
		// if((name_str[0] == 'g')&&(strstr(name_str, "global_") != NULL))
		// {
		// 	addr += pso.global_idx * sizeof(struct pso_global);
		// }	
		if(strcmp(name_str, "pid_v_target") == 0)
		{
			//VPD=0.61078*e^(17.27*Ta/(Ta+237.3))*(1-RH)	//摄氏度/kpa
			//dx = (17.27 * bp_pid_th.t_target) / (bp_pid_th.t_target + 237.3);
			//dx = pid_exp(dx);
			//dx = dx * 0.61078 * (1.0 - bp_pid_th.h_target / 100.0);
			dx=			 bp_pid_th.v_target;//	 bp_pid_th.v_inside;
			bp_pid_th.v_target =dx;
		}	
		if(val_str[0] == '?') 
		{
			switch(typ)
			{
				case 1:		bp_pid_dbg("var %s=%f OK\r\n",       debug_name_addr_type[i].name, *(float*)addr); break;
				case 2:		bp_pid_dbg("var %s=%lf OK\r\n",      debug_name_addr_type[i].name, *(double*)addr); break;				
				case 7:		bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(char*)addr, 			*(char*)addr); break;
				case 8:		bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(unsigned char*)addr, 	*(unsigned char*)addr); break;
				case 15:	bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(short*)addr, 			*(short*)addr); break;
				case 16:	bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(unsigned short*)addr, 	*(unsigned short*)addr); break;
				case 31:	bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(int*)addr, 			*(int*)addr); break;
				case 32:	bp_pid_dbg("var %s=0x%x(%d) OK\r\n", debug_name_addr_type[i].name, *(unsigned int*)addr, 	*(unsigned int*)addr); break;
				default:	return c_ret_nk;
			}
			return c_ret_ok;
		}		
		else 
		{
			if(str2num(val_str, &value, &dx) != c_ret_ok) 
				return c_ret_nk;			
			switch(typ)
			{
				case 1:		*(float*)addr = (float)dx; break;
				case 2:		*(double*)addr = dx; break;
				case 7:		*(char*)addr = (char)value; break;						
				case 8:		*(unsigned char*)addr = (unsigned char)value; break;
				case 15:	*(short*)addr = (short)value; break;
				case 16:	*(unsigned short*)addr = (unsigned short)value; break;
				case 31:	*(int*)addr = (int)value; break;
				case 32:	*(unsigned int*)addr = value; break;
				default: 	return c_ret_nk;
			}
			bp_pid_dbg("var %s,%s OK\r\n", name_str, val_str);
			return c_ret_ok;
		}
	}
	return c_ret_nk;
}
 

static void bp_pid_th_upload(int type)
{
	unsigned char 	*px;
	unsigned int	i=0,j=0;
	union un_double	uf;
	px = (unsigned char*)&bp_pid_th;
	int len=(c_nh_nodes0 * c_ni_nodes + c_no_nodes * c_nh_nodes1+ c_nh_nodes0 * c_nh_nodes1);
	if(type==1)
	{
		px = (unsigned char*)&pso;
		len=(NUM_ENV_TYPE*2+NUM_KEYN);
		for(i = 0; i <len * 8; i += 8)
		{
			uf.buf[0] = px[i + 0];
			uf.buf[1] = px[i + 1];
			uf.buf[2] = px[i + 2];
			uf.buf[3] = px[i + 3];
			uf.buf[4] = px[i + 4];
			uf.buf[5] = px[i + 5];
			uf.buf[6] = px[i + 6];
			uf.buf[7] = px[i + 7];
			bp_pid_dbg("argup %d=%lf\r\n", i / 8, uf.dx);
		}
	}
	else if(type==2)
		{
#if 0
			px = (unsigned char*)&svd.arisk;
			//len= (MN*MN); //(NUM_SPK+1)*(NUM_LATN+1);
			//len=  (NUM_SPK+1)*(NUM_KEYN+1);
			len =  ((MN)*(MN)*3 +MN);
			for(i = 0; i <NUM_SPK ; i ++)
			{
				for(  j = 0; j <NUM_KEYN ; j ++)
				{
					
					bp_pid_dbg(" as[%d][%d]=%lf\r\n", i,j, svd.arisk[i+base][j+base]);
				}
			}

			for(i = 0; i <NUM_KEYN ; i ++)
			{
				for(  j = 0; j <NUM_KEYN ; j ++)
				{
					
					bp_pid_dbg(" u[%d][%d]=%lf\r\n", i,j, svd.u_mat[i+base][j+base]);
				}
			}
			 
			for(i = 0; i <NUM_SPK   ; i ++)
			{
				for(  j = 0; j < NUM_SPK; j ++)
				{
					
					bp_pid_dbg(" v[%d][%d]=%lf\r\n", j, i,svd.v_mat[j+base][i+base]);
				}
			}  
			for(  j = 0; j <NUM_KEYN ; j ++)
				{					
					bp_pid_dbg(" w[%d]=%lf\r\n", j, svd.w_vec [j+base]);
				}
#endif
		}
	 
	
	//for(i = 0; i < (c_nh_nodes * c_ni_nodes + c_no_nodes * c_nh_nodes) * 3 * 8; i += 8)
	
}

static void print_dbg_var_name_addr(void)
{
	unsigned int i;

	bp_pid_dbg("******** dbg_var_name ********\r\n");
	for(i = 0; i < sizeof(debug_name_addr_type)/sizeof(debug_name_addr_type[0]); i++)
	{
		bp_pid_dbg("    %s,0x%x\r\n", debug_name_addr_type[i].name, debug_name_addr_type[i].addr);
	}
}
static void dbg_parse(char *buf)
{
    char *px;

    if((buf[0] == 'h')&&(strstr(buf, "help") != NULL))
    {
        bp_pid_dbg("********  cmd sets  ********\r\n");
        bp_pid_dbg("1. help\r\n");
        bp_pid_dbg("2. version\r\n");
        bp_pid_dbg("3. var?\r\n");
        bp_pid_dbg("4. var:name,?/val\r\n");
        return;
    }
    else if((buf[0] == 'v')&&(strstr(buf, "version") != NULL))
    {
        bp_pid_dbg("version:%u\r\n", bp_pid_th.version);
        return;
    }
    else if((buf[0] == 'v')&&((px = strstr(buf, "var?")) != NULL))
    {
        print_dbg_var_name_addr();
        return;
    }
    else if((buf[0] == 'v')&&((px = strstr(buf, "var:")) != NULL))
    {
        if(dbg_parse_var(px + 4) == c_ret_ok)
            return;
    }
    else if((buf[0] == 'a')&&((px = strstr(buf, "argup start")) != NULL))
    {
        bp_pid_th_upload(2);
        return;
    }
    else if((buf[0] == 'a')&&((px = strstr(buf, "argdn ")) != NULL))
    {
        unsigned int	addr, idx;
		double			dx;

		if(sscanf(px, "argdn %d=%lf", &idx, &dx) == 2)
		{
			addr = (unsigned int)&bp_pid_th;
			addr += idx * 8;
			*(double*)addr = dx;
        	return;
		}
    }

    bp_pid_dbg("%s NK\r\n", buf);
}

void dbg_task_loop(void *arg)
{
    unsigned int                len = 0, idx = 0;
    char                        buf[129] = {0};
    char                        dat = 0;

    while(1)
    {
		len = dbg_read( &dat, 1);
        if(len == 1)
        {
            if((dat == '\r')||(dat == '\n'))
            {
                if(idx != 0)
                {
                    buf[idx] = 0;
                    idx = 0;
                    dbg_parse(buf);
                }
            }
            else 
            {
                if(idx < sizeof(buf) - 1)
                    buf[idx++] = dat;
            }
        }
    }
}
 
 
 /// @brief 
 /// @param p_buf 
 /// @return 存儲數據總數
 uint16_t pid_sava_running_data(uint8_t* p_buf)
 {
	uint16_t len = 0,siz_pid=0;
	memcpy(p_buf+len, &bp_pid_th.update_rate, sizeof(float));len += sizeof(float);

	memcpy(p_buf+len, &bp_pid_th.ho_sigmoid_out[0], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &bp_pid_th.ho_sigmoid_out[1], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &bp_pid_th.ho_sigmoid_out[2], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.swarm[0].position[0], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.swarm[0].position[1], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.swarm[0].position[2], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.mae_buf[ENV_T], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.mae_buf[ENV_H], sizeof(double));len += sizeof(double);
	memcpy(p_buf+len, &pso.mae_buf[ENV_V], sizeof(double));len += sizeof(double);

    // siz_pid=sizeof(double)*c_no_nodes*c_nh_nodes;	memcpy(p_buf+len, &bp_pid_th.wt_no_nh,siz_pid);len += siz_pid;
    // siz_pid=sizeof(double)*c_nh_nodes*c_ni_nodes;	memcpy(p_buf+len, &bp_pid_th.wt_nh_ni,siz_pid);len += siz_pid;
    // siz_pid=sizeof(double)*c_nh_nodes*c_nh_nodes;	memcpy(p_buf+len, &bp_pid_th.wt_nh_nh,siz_pid);len += siz_pid;
	 
	return len;
 }

 uint16_t pid_read_running_data(uint8_t* p_buf)
 {
	uint16_t len = 0,siz_pid=0;
	memcpy(&bp_pid_th.update_rate, p_buf+len, sizeof(float ));len += sizeof(float);
	
	memcpy(&bp_pid_th.ho_sigmoid_out[0], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&bp_pid_th.ho_sigmoid_out[1], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&bp_pid_th.ho_sigmoid_out[2], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.swarm[0].position[0], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.swarm[0].position[1], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.swarm[0].position[2], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.mae_buf[ENV_T], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.mae_buf[ENV_H], p_buf+len, sizeof(double ));len += sizeof(double);
	memcpy(&pso.mae_buf[ENV_V], p_buf+len, sizeof(double ));len += sizeof(double); 
	return len;
 }
