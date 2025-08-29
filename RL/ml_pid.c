#include "esp_log.h"
#include "string.h"
#include "ml_pid.h"
#include "math.h"
#include "Sensor.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "beep.h"

// struct st_wc1c2_tab
// {
// 	float			w;
// 	float			c1;
// 	float			c2;
// };
 
// struct st_wc1c2_tab	wc1c2_tab[3]={{0.4, 1.0, 1.0},{0.7, 1.5, 1.5},{0.9, 2.0, 2.0}};


#define c_gain_ctrl_stop			0
#define c_gain_ctrl_req				1
#define c_gain_ctrl_auto			2
static unsigned int gain_adj_ctrl = 2;

#include <time.h>
#include <sys/time.h>

#define c_pso_step_init				0
#define c_pso_step_wait				1
#define c_pso_step_update			2
#define c_pso_step_check			3


#define dbg_serial_usb_serial_jtag		1
#define dbg_serial_uart0				2
#define dbg_serial_sel        			dbg_serial_uart0
//#define dbg_serial_sel        			dbg_serial_usb_serial_jtag


						//       t     dt   h     dh
const float pso_pos_max_tab[DIM]  = {30000,30000,20000,20000,10000,10000};
const float pso_pos_min_tab[DIM]  = {10000,10000,8000,8000,4000,4000};

 
//float pitch_limit[2][NUM_PTH_TYPE]={{20,80,300},{40,160,600}};
//float tmp_pitch[NUM_UPDOWN][NUM_PTH_TYPE]= {{0.99,0.99,0.99 },{1.01,1.01,1.01 }};
unsigned int tmp_pdx[DIM]={0};
struct st_bp_pid_th    bp_pid_th = {0};
struct pso_optimizer pso = {0};
struct svd_optimizer svd = {0};
static float hvac_margin[NUM_ENV_TYPE]={0.030,0.030,0.050};

// static float		test_temp_input_rate = 1.0f;
// static float		test_humi_input_rate = 1.0f;
static unsigned int	test_gear_mode = 0;			//0--多挡位,1--单9档
static unsigned int test_humi_cycle_num = 4;//2;

static unsigned int gain_adj_size = 8;	//30
static float 		gain_adj_rate_min = 1000;
static float 		gain_adj_rate_max = 10000;
static float 		gain_adj_rate_change = 1000;
    double          ni_dat[c_ni_nodes],  net_oh_dat[c_no_nodes],net_hh_dat[c_nh_nodes1],net_ih_dat[c_nh_nodes0];
	double          no_delta1d[c_no_nodes],d_hide0[c_nh_nodes0],d_hide_o[c_no_nodes],d_hide1[c_nh_nodes1]; 
	double          ih_sigmoid_out[c_nh_nodes0],hh_sigmoid_out[c_nh_nodes1] ;
    double          nh_delta1[c_nh_nodes1],nh_delta0[c_nh_nodes0] ;


struct tent_optimizer
{
	  
	float			buf[60];
	unsigned int	buf_idx, buf_cnt  ; 
	
};
struct tent_optimizer tent = {0};


void  dbg_init(void);
unsigned int dbg_parse_var(char *buf);
int  dbg_read(void *buf, unsigned int len);

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
static unsigned int  bp_pid_tick_tmr = 0;
	const int base=1;

#if 0
const unsigned short temp_gear_list[11]  ={0,277,360,455,517,586,665,744,834,913,1000};
const unsigned short humi_gear_list[11]  ={0,560,600,640,680,730,780,830,880,930,1000};
const unsigned short vpd_gear_list[11]   ={0,560,600,640,680,730,780,830,880,930,1000};
const unsigned short detemp_gear_list[11]={0,277,360,455,517,586,665,744,834,913,1000};
const unsigned short dehumi_gear_list[11]={0,560,600,640,680,730,780,830,880,930,1000};
const unsigned short devpd_gear_list[11]= {0,560,600,640,680,730,780,830,880,930,1000};

#else
const float gear_list[NUM_ENVDEV][11]        = {
	{10, 100, 200, 300, 400, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
};
// const unsigned short temp_gear_list[11]   = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
// const unsigned short humi_gear_list[11]   = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
// const unsigned short vpd_gear_list[11]    = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
// const unsigned short detemp_gear_list[11] = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
// const unsigned short dehumi_gear_list[11] = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
// const unsigned short devpd_gear_list[11]  = {0, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000};
#endif
// static float g_epid_tent_tmr =-1000;
// static float l_epid_tent_tmr = 1000;
// static float  t_mae=0;
// static float  h_mae=0 ;
/*
	1--float,
	2--double,
	7--s8,
	8--unsigned char,
	15--s16,
	16--unsigned short,
	31--s32,
	32--unsigned int,
 */
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


	{"pso_global_idx",			(unsigned int)&pso.global_idx,				32},
	{"global_fitness",			(unsigned int)&pso.global_bestval,	            1},
	{"global_best_t",			(unsigned int)&pso.global[0].pos[0],		1},
	{"global_best_dt",			(unsigned int)&pso.global[0].pos[1],		1},
	{"global_best_h",			(unsigned int)&pso.global[0].pos[2],		1},
	{"global_best_dh",			(unsigned int)&pso.global[0].pos[3],		1},

	//{"pso_test_req",			(unsigned int)&pso.test_req,				32},
	//{"pso_buf_cnt",				(unsigned int)&pso.buf_cnt,					32},
	{"pso_swarm_idx",			(unsigned int)&pso.swarm_idx,				32},
	{"swarm_fitness",			(unsigned int)&pso.swarm[0].best_mae,	1},
	{"swarm_best_t",			(unsigned int)&svd.pitchs[0],	1},
	{"swarm_best_dt",			(unsigned int)&svd.pitchs[1],	1},
	{"swarm_best_h",			(unsigned int)&svd.pitchs[2],	1},
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

static double update_particles (void);
void pso_init(void) ;
void svd_init(void);
unsigned int calculate_svd(double new_mae);
void svd_port_latent( double *profile ,int spkidx);


static unsigned int tick_get(void)
{  
	return bp_pid_tick_tmr;
}
static unsigned int tick_reset(void)
{  
	  bp_pid_tick_tmr=0;
	  return 0;
}


static unsigned int tick_cmp(int tmr, int tmo)
{  
	return (tick_get()  >= (tmo+tmr)) ? c_ret_ok : c_ret_nk; 
}
 

void bp_pid_th_init(void) 
{
    unsigned int i, h, o;

	memset(&bp_pid_th, 0, sizeof(struct st_bp_pid_th));
	 
	bp_pid_th.tmr = 4000;//60000;
	bp_pid_th.dev_token = 0;
	// bp_pid_th.u_min[GOV_TOUT] = -1000.0f;	
	// bp_pid_th.u_min[GOV_HOUT] = -1000.0f;	 
	// bp_pid_th.u_max[GOV_TOUT] = 1000.0f;	
	// bp_pid_th.u_max[GOV_HOUT] = 1000.0f; 	

	// bp_pid_th.u_gain[GOV_TOUT*NUM_ENVGAN_TYPEOUT+ENVGAN_UP]   = 7000.0f;
	// bp_pid_th.u_gain[GOV_TOUT*NUM_ENVGAN_TYPEOUT+ENVGAN_DOWN]   = 7000.0f;
	// bp_pid_th.u_gain[GOV_HOUT*NUM_ENVGAN_TYPEOUT+ENVGAN_UP]   = 7000.0f; 
	// bp_pid_th.u_gain[GOV_HOUT*NUM_ENVGAN_TYPEOUT+ENVGAN_DOWN]   = 7000.0f; 
	for(h= 0; h< NUM_ENV_TYPE; h++)
	{
			bp_pid_th.s[h]=0.0f;
			bp_pid_th.f[0][h]=0.0f;
			bp_pid_th.e[0][h]=0.0f;
	}
    for(h= 0; h< NUM_ENVDEV; h++)
	{
		bp_pid_th.u_gear_tmr[h]  =0 ; 
		bp_pid_th.pid_o[h]  =0 ; 
	}
    // for(h= 0; h< NUM_ENVDEV; h++)
	// {
	// 	//bp_pid_th.u_gain[h]   = 7000.0f;		
	// 	bp_pid_th.u_gain_tmr[h] = 0.0f;
	// }
	
		 

	pso_init();
	//bp_pid_th.ptch_tmr = 0;
	//bp_pid_th.u_gear_tmr = 0;
	bp_pid_th.update_rate = 0.001f;// 0.0001f;
    for(o = 0; o < c_nh_nodes0; o++)
    {
        for(h= 0; h< c_ni_nodes; h++)
        {
            bp_pid_th.wt_nh_ni  [o][h] =.05*(1.0- (0.5* (float)rand() / RAND_MAX)) ;//.02*(float)rand() / RAND_MAX ;// wi_init[0 * c_nh_nodes * c_ni_nodes + h * c_ni_nodes + i];

        }
    }

   for(o = 0; o < c_nh_nodes1; o++)
    {
        for(h = 0; h < c_nh_nodes0; h++)
        { 		
			bp_pid_th.dropout_hh[o][h] =  ((float)rand() / RAND_MAX )<0.9?false:true;
			//bp_pid_th.dropout_hh[o][h] = ((float)rand() / RAND_MAX )<0.5?false:true;		
            bp_pid_th.wt_nh_nh  [o][h] = .05*(1.0- (0.5* (float)rand() / RAND_MAX)) ;//(.05* (float)rand() / RAND_MAX) ;// wo_init[0 * c_no_nodes * c_nh_nodes + o * c_nh_nodes + h];  	 
        }
    }	


	// for(h = 0;h < c_nh_nodes1;h++)
	// {
		 
	// 	bp_pid_th.wt_no_nh[0][h] =bp_pid_th.wt_no_nh[3][h] =bp_pid_th.wt_no_nh[6][h] = .07*(1.0- (0.5* (float)rand() / RAND_MAX)) ;//.07;
	// 	bp_pid_th.wt_no_nh[1][h] =bp_pid_th.wt_no_nh[4][h] =bp_pid_th.wt_no_nh[7][h] = .02*(1.0- (0.5* (float)rand() / RAND_MAX)) ;// .02;
	// 	bp_pid_th.wt_no_nh[2][h] =bp_pid_th.wt_no_nh[5][h] =bp_pid_th.wt_no_nh[8][h] = .01*(1.0- (0.5* (float)rand() / RAND_MAX)) ;// .01;
	// }
    for(o = 0; o < c_no_nodes; o++)
    {		
		 for(h = 0; h < c_nh_nodes1; h++)
        {
        bp_pid_th.wt_no_nh  [o][h]  = .06*(1.0- (0.5* (float)rand() / RAND_MAX)) ;// (.5* (float)rand() / RAND_MAX) ;// wo_init[0 * c_no_nodes * c_nh_nodes + o * c_nh_nodes + h];
		//bp_pid_th.dropout_oh[o][h] = ((float)rand() / RAND_MAX )<0.9?false:true;	
        }
    }
    //pid_read_running_data();
}

 

static unsigned int float_chk(double fx)
{
	if((isinf (fx) != 0)||(isnan(fx) != 0))
		return c_ret_nk;
	return c_ret_ok;
}

static double pid_exp(double x)
{
	if(x > 708) x = 708;
	else if(x < -708) x = -708;
	return exp(x);	
}
 
static void sigmoid_o(double *dx,double *x,s16 len) 
{	
	for(s16 o = 0; o < len; o++) 
	{
		dx[o]=x[o]>0?x[o]:0;
		//dx[o]=1.0 / (1.0+ pid_exp(-x[o]));
		//dx[o]=(pid_exp(x[o]) - pid_exp(-x[o])) / (pid_exp(x[o]) + pid_exp(-x[o]));
		if(float_chk(dx[o]) == c_ret_nk)
			dx[o]=0.0f;  
	}
    return ;//(pid_exp(x) - pid_exp(-x)) / (pid_exp(x) + pid_exp(-x));
	//return 1.0 / (1.0+pid_exp(-x) );//sigmoid(x);

}
 
static void sigmoid_h(double *dx,double *x,s16 len) 
{	
	for(s16 o = 0; o < len; o++) 
	{
		dx[o]=x[o]>0?x[o]:0;		
		//dx[o]=1.0 / (1.0+ pid_exp(-x[o]));
		//dx[o]=(pid_exp(x[o]) - pid_exp(-x[o])) / (pid_exp(x[o]) + pid_exp(-x[o]));
		if(float_chk(dx[o]) == c_ret_nk)
			dx[o]=0.0f;  
	}
    return ;//(pid_exp(x) - pid_exp(-x)) / (pid_exp(x) + pid_exp(-x));
	//return 1.0 / (1.0+pid_exp(-x) );//sigmoid(x);

}
 

// 激活函数求导 

static void  derivative_o(double *dx,double *x,s16 len) 
{
	static double dvx[c_no_nodes];

	for(s16 o = 0; o < len; o++)
	{
		  
		dx[o]=x[o]>0?x[o]-dvx[o]:0;	
		dvx[o]=x[o];					
		//dx[o] = 2.0 / ((pid_exp(x[o]) + pid_exp(-x[o])) * (pid_exp(x[o]) + pid_exp(-x[o])));
		//dx[o] =  x[o] * (1.0-x[o] );
		//dx[o]=1.0-x[o]*x[o];
		if(float_chk(dx[o]) == c_ret_nk)
			dx[o]=0.0f;    	 
	}
    //return x / (1.0-x );
	return;
}

 
static void  derivative_tanh(double *dx,double *x,s16 len) 
{
	static double dvx[c_no_nodes];
	for(s16 o = 0; o < len; o++)
	{
		dx[o]=x[o]>0?x[o]-dvx[o]:0;	
		dvx[o]=x[o];			
		//dx[o] = 4.0 / ((pid_exp(x[o]) + pid_exp(-x[o])) * (pid_exp(x[o]) + pid_exp(-x[o]))); 
		//dx[o] =  x[o] * (1.0-x[o] );
		//dx[o]=1.0-x[o]*x[o];
		if(float_chk(dx[o]) == c_ret_nk)
			dx[o]=0.0f;    	 
	}
    return  ;
}



static float pid_map(float x, float in_min, float in_max, float out_min, float out_max)
{
    if(x < in_min) return out_min;
	if(x > in_max) return out_max;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min; 
}
// static unsigned int get_tent(  float tmr,float *tent_time_acc)
// {
// 		static float pre_tmr=0;
// #if 0
//  unsigned int len=4;
// 	unsigned int 		d;
// 	float		 		t_mae ;
	 
// 	//if(tmr-pre_tmr <=c_pid_tent_min) return c_ret_nk;
// 	tent.buf[tent.buf_idx]   = tmr-pre_tmr; 	
// 	pre_tmr=tmr;
// 	tent.buf_idx++;
// 	if(tent.buf_idx < len) return c_ret_nk;
// 	tent.buf_idx = 0;
// 	for(t_mae = 0, d = 0; d < len; d++)
// 		t_mae += tent.buf[d] ;
// 	//t_mae /= (float)d;
//     *tent_time_acc= t_mae;
// #else
// 	*tent_time_acc= tmr-pre_tmr;
// 	pre_tmr=tmr;
// #endif
// 	return c_ret_ok;
// }

const float fit_up_bound=8.0;
const float fit_low_bound=3.0;
double current_mae=0.0; 

static unsigned int bp_pid_th_exec(void)//float set0, float feed0, float set1, float feed1, float set2, float feed2 ) 
{
	double			dx=0; 
	unsigned int	i=0,  h=0, o=0;
    double          i_pid[NUM_ENV_IN];

	bp_pid_th.s[ENV_T]    = pid_map(bp_pid_th.t_target,  c_pid_temp_min, c_pid_temp_max, 0, 1);
	bp_pid_th.f[0][ENV_T] = pid_map(bp_pid_th.t_feed,    c_pid_temp_min, c_pid_temp_max, 0, 1);
	bp_pid_th.s[ENV_H]    = pid_map(bp_pid_th.h_target,  c_pid_humi_min, c_pid_humi_max, 0, 1);
	bp_pid_th.f[0][ENV_H] = pid_map(bp_pid_th.h_feed,    c_pid_humi_min, c_pid_humi_max, 0, 1);    
	bp_pid_th.s[ENV_V]    = pid_map(bp_pid_th.v_target,  c_pid_vpd_min,  c_pid_vpd_max , 0, 1);
	bp_pid_th.f[0][ENV_V] = pid_map(bp_pid_th.v_feed,    c_pid_vpd_min,  c_pid_vpd_max , 0, 1); 
	//bp_pid_dbg(" set_value_2=%f, feed_value_2=%f  \r\n",set_value_2,feed_value_2);
   
	//static float pre_gain[NUM_ENVDEV] ; 
	//static float pre_ptch[NUM_ENVPTCH] ;   
	
	for(h = 0; h <c_ni_nodes; h++)
	{
		ni_dat[h] =0.0f;
	}   
	for(h = 0; h < NUM_ENV_TYPE ; h++)
	{
		bp_pid_th.e[0][h] = bp_pid_th.s[h] - bp_pid_th.f[0][h];		
	  
		ni_dat[h*NUM_ENV_TYPEIN+PAR_SET_IN]   = bp_pid_th.s[h];
		ni_dat[h*NUM_ENV_TYPEIN+PAR_FED_IN]   = bp_pid_th.f[0][h];
		ni_dat[h*NUM_ENV_TYPEIN+PAR_DEL_IN]   = bp_pid_th.e[0][h];	 
	}
	
 	current_mae=update_particles(); 
	//wave
	
	// for(h = 0; h <( NUM_ENVDEV); h++)
	// {
	// 	ni_dat[NUM_ENV_IN+PAR_TIME_IN]  += pso.swarm[pso.swarm_idx].velocity[h]  ;
	// }
	// ni_dat[NUM_ENV_IN+PAR_TIME_IN]  +=   0.01*(1.0- (0.5* (float)rand() / RAND_MAX));	
	// ni_dat[NUM_ENV_IN+PAR_TIME_IN] = pid_map(ni_dat[NUM_ENV_IN+PAR_TIME_IN] ,  c_pid_ptch_min, c_pid_ptch_max, 0, 1);

	  
    	
	//ni_dat[NUM_ENV_TYPE*NUM_ENV_TYPEIN+PAR_GAIN_IN] =  pid_map(gain_error , 0,  8 , 0, 1);  
	//ni_dat[NUM_ENV_IN+NUM_LATN_TYPEIN+NUM_ENVPTCH] = 1;  	
 	for(h = 0; h <(NUM_ENVDEV); h++)
	{ 
		ni_dat[NUM_ENV_IN+h] =pid_map(pso.swarm[pso.swarm_idx].velocity[h]  + 100.0*(1.0- (0.5* (float)rand() / RAND_MAX)),  c_pid_ptch_min, c_pid_ptch_max, 0, 1);
		ni_dat[NUM_ENV_IN+NUM_ENVDEV+h]=(float)bp_pid_th.u_gear_tmr[h]/10.0f  + 0.1*(1.0- (0.5* (float)rand() / RAND_MAX)); 
		 
	}
			

	for(h = 0; h <( NUM_ENV_TYPE); h++)
	{	i_pid  [h*NUM_ENV_KPID+ENV_KP]   = bp_pid_th.e[0][h] -   bp_pid_th.e[1][h];
		i_pid  [h*NUM_ENV_KPID+ENV_KI]   = bp_pid_th.e[0][h];
		i_pid  [h*NUM_ENV_KPID+ENV_KD]   = bp_pid_th.e[0][h] - 2*bp_pid_th.e[1][h] + bp_pid_th.e[2][h];	
	}    
	 
	// for(h=0;h<NUM_ENVDEV;h++)
	// {
	// 	epid[NUM_DEV_KPID_OUT+h]= (ni_dat[NUM_ENV_IN+h]*ni_dat[NUM_ENV_IN+NUM_LATN_GAIN+h]);//-pre_gain[PAR_GAIN_TU_IN]; 
	// 	epid[NUM_DEV_KPID_OUT+h]= pso.swarm[pso.swarm_idx].position[h] ;
	// 	//pre_gain[h]  = (ni_dat[NUM_ENV_IN+h]*ni_dat[NUM_ENV_IN+h]);
	// 	//no_delta[NUM_DEV_KPID_OUT+h] = -(epid[NUM_DEV_KPID_OUT+h]) * (bp_pid_th.ho_sigmoid_out[NUM_DEV_KPID_OUT+h]);//-bp_pid_th.u_gain_tmr[h]) ;		
	// }	
	
	
	// epid[NUM_DEV_KPID_OUT+NUM_ENVDEV]  = pso.swarm[pso.swarm_idx].velocity[0] + pso.swarm[pso.swarm_idx].velocity[2];
	
	//no_delta[NUM_DEV_KPID_OUT+PAR_TIME_IN] =  epid[NUM_DEV_KPID_OUT+PAR_TIME_IN]-pre_ptch[ENVPTCH_UP]  ;  
	//pre_ptch[ENVPTCH_UP]      = epid[NUM_DEV_KPID_OUT+NUM_ENVDEV];
	
	for(h = 0; h < NUM_ENV_TYPE; h++){
		bp_pid_th.f [1][h]  =  bp_pid_th.f [0][h] ; 
		bp_pid_th.e [2][h]  =  bp_pid_th.e [1][h];
		bp_pid_th.e [1][h]  =  bp_pid_th.e [0][h]; 
	}

	
	sigmoid_o(bp_pid_th.ho_sigmoid_out,net_oh_dat,c_no_nodes);
	 
  	//float dygain[NUM_ENVGOV_TYPEOUT*NUM_ENVGAN_TYPEOUT];    
	//static double pre_dygain_out[NUM_ENVGOV_TYPEOUT*NUM_ENVGAN_TYPEOUT];    
	for(h = 0; h < NUM_ENV_TYPE; h++)
	{
		bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_UP]    =bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KP] *i_pid[h*NUM_ENV_KPID+ENV_KP]//*pso.swarm[pso.swarm_idx].position[h] //+pso.swarm[pso.swarm_idx].velocity[h]
		                					  +bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KI]*i_pid[h*NUM_ENV_KPID+ENV_KI]
						  					  +bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KD]*i_pid[h*NUM_ENV_KPID+ENV_KD];
		bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_DOWN]= - bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_UP];
		//no_delta[h*NUM_ENV_KPID+ENV_KP]= -(epid[h*NUM_ENV_KPID+ENV_KP] *(bp_pid_th.du[0][h*NUM_UPDOWN] - bp_pid_th.du[1][h*NUM_UPDOWN ]  )) ;
 	    //no_delta[h*NUM_ENV_KPID+ENV_KI]= -(epid[h*NUM_ENV_KPID+ENV_KI] * bp_pid_th.du[0][h*NUM_UPDOWN]) ;
	    //no_delta[h*NUM_ENV_KPID+ENV_KD]= -(epid[h*NUM_ENV_KPID+ENV_KD] *(bp_pid_th.du[0][h*NUM_UPDOWN] - 2*bp_pid_th.du[1][h*NUM_UPDOWN ] + bp_pid_th.du[2][h*NUM_UPDOWN ]  ));	
		//bp_pid_th.tmr     += ( pso.swarm[pso.swarm_idx].velocity[h]*1e-3);
	}
 	 
	 
	// g_epid_tent_tmr = (g_epid_tent_tmr < bp_pid_th.tent_tmr)?bp_pid_th.tent_tmr:g_epid_tent_tmr;
	// l_epid_tent_tmr = (l_epid_tent_tmr > bp_pid_th.tent_tmr)?bp_pid_th.tent_tmr:l_epid_tent_tmr;
	//float de= .06*(1.0- (0.5* (float)rand() / RAND_MAX)); 
	 
	// for(h = 0; h < NUM_ENV_TYPE; h++){ 
	// 	bp_pid_th.du[2][h]  =  bp_pid_th.du[1][h] ; 
	// 	bp_pid_th.du[1][h]  =  bp_pid_th.du[0][h] ; 
	// 	bp_pid_th.tu[1]     =  bp_pid_th.tu[0];
	// } 
	
	for(h = 0; h < c_nh_nodes0; h++)
	{
		net_ih_dat[h] = 0.0f;
		for(i = 0; i < c_ni_nodes; i++)
		{
			net_ih_dat[h] += ni_dat[i] * bp_pid_th.wt_nh_ni[h][i];
		}  
	}
    sigmoid_h(ih_sigmoid_out,net_ih_dat,c_nh_nodes0);
	for(h = 0; h < c_nh_nodes1; h++)
	{
		net_hh_dat[h] = 0.0f;
		for(i = 0; i <  c_nh_nodes0; i++)
		{
			if(bp_pid_th.dropout_hh[h][i] == false)
				net_hh_dat[h] +=  ih_sigmoid_out[i] * bp_pid_th.wt_nh_nh[h][i];	
			else
				bp_pid_th.wt_nh_nh[h][i]=0; 
		}
	}
	sigmoid_h(hh_sigmoid_out,net_hh_dat,c_nh_nodes1);
	for(h = 0; h < c_no_nodes; h++)
	{
		net_oh_dat[h] = 0.0f;
		for(i = 0; i <  c_nh_nodes1; i++)
		{
			//if(bp_pid_th.dropout_hh[h][i] == false)
				net_oh_dat[h] +=  hh_sigmoid_out[i] * bp_pid_th.wt_no_nh[h][i];	
			//else
			//	bp_pid_th.wt_no_nh[h][i]=0;

		}
	// 	bp_pid_dbg(" nh_dat= %.2f \r\n",net_hh_dat[i]);
	}
	  
  
	// for( j=0;j<3;j++){
	// 	for(h = j*c_nh_nodes/3; h < (j+1)* c_nh_nodes/3; h++)
	// 	{
	// 		net_hh_dat[h] = 0;
	// 		for(i = j*c_nh_nodes/3; i < (j+1)* c_nh_nodes/3; i++)
	// 		{
	// 			if(bp_pid_th.dropout_hh[h][i] == false)
	// 				net_hh_dat[h] += nih_dat[i] * bp_pid_th.wt_nh_nh[h][i];
	// 			else
	// 				bp_pid_th.wt_nh_nh[h][i]=0;
	// 		}			
	// 	} 
	// }
   
	derivative_o(d_hide_o,bp_pid_th.ho_sigmoid_out,c_no_nodes ); 
	derivative_tanh(d_hide1,hh_sigmoid_out,c_nh_nodes1 );
	derivative_tanh(d_hide0,ih_sigmoid_out,c_nh_nodes0 );
	double Etotal=0.0f;//pso.mae_buf[pso.swarm_idx][ENV_T]+pso.mae_buf[pso.swarm_idx][ENV_H];
	for(h=0;h<NUM_ENV_TYPE;h++)
    {
		Etotal +=bp_pid_th.e[0][h];
	}	
	
	for(o = 0; o < c_nh_nodes1  ; o++)
    {
		nh_delta1[o] = 0;
		for(h = 0; h < c_no_nodes; h++)
        {
          //nh_delta1[o] += no_delta[h] *bp_pid_th.wt_no_nh[h][o] * d_hide1[o];
          nh_delta1[o] += (Etotal *bp_pid_th.wt_no_nh[h][o] * d_hide1[o]);
        }
		//bp_pid_dbg(" nhh =(%.6f,%.3f,%.3f,%.3f)\r\n", nh_delta1[o], d_hide_o[o] ,net_oh_dat[o],bp_pid_th.wt_no_nh[0][o] );

    } 
	for(o = 0; o < c_nh_nodes0 ; o++)
	{
		nh_delta0[o] = 0;
		for(h = 0; h < c_nh_nodes1; h++)
		{ 
			nh_delta0[o] += nh_delta1[h] * bp_pid_th.wt_nh_nh[h][o] * d_hide0[o];
		}
		//bp_pid_dbg("nhi =(%.6f,%.6f,%.6f,%.6f)\r\n", nh_delta0[o], d_hide1[o] ,net_hh_dat[o],bp_pid_th.wt_nh_nh[0][o] );
	}  
  	 
	bp_pid_th.update_rate=0.001;
    //权值更新
    for(h = 0; h < c_no_nodes; h++) 
	{ 
        for(i = 0; i < c_nh_nodes1; i++) 
		{ 
		    //bp_pid_th.wt_no_nh[h][i] +=  bp_pid_th.update_rate * no_delta[h] * d_hide_o[h]  * hh_sigmoid_out[i];
		    bp_pid_th.wt_no_nh[h][i] +=  bp_pid_th.update_rate  * Etotal  * d_hide_o[h]  * hh_sigmoid_out[i];
		 
        }
    }
	for(h = 0; h < c_nh_nodes1; h++) 
	{
		for(i = 0; i < c_nh_nodes0; i++) 
		{	 
			//bp_pid_th.wt_nh_nh[h][i] +=   bp_pid_th.update_rate  * nh_delta1[h] * d_hide1[h]  * ih_sigmoid_out[i];		 
			bp_pid_th.wt_nh_nh[h][i] +=   bp_pid_th.update_rate  * nh_delta1[h]  * d_hide1[h]  * ih_sigmoid_out[i];		 
		} 
	}  
    for(h = 0; h < c_nh_nodes0; h++) 
	{     
       for(i = 0; i < c_ni_nodes; i++) 
		{ 
            //bp_pid_th.wt_nh_ni[h][i] +=  bp_pid_th.update_rate * nh_delta0[h] * d_hide0[h]* ni_dat[i];
            bp_pid_th.wt_nh_ni[h][i] +=  bp_pid_th.update_rate * nh_delta0[h]  * d_hide0[h]* ni_dat[i];
 
        }
    }
 	// bp_pid_dbg("wt_no_nh=(%.6f,%.6f,%.6f,%.6f)\r\n",bp_pid_th.wt_no_nh[0][0],no_delta[0],  d_hide_o[0], hh_sigmoid_out[0]);  
	// bp_pid_dbg("wt_nh_nh=(%.6f,%.6f,%.6f,%.6f)\r\n",bp_pid_th.wt_nh_nh[0][0],nh_delta1[0], d_hide1[0],  ih_sigmoid_out[0]); 
	// bp_pid_dbg("wt_nh_ni=(%.6f,%.6f,%.6f,%.6f)\r\n",bp_pid_th.wt_nh_ni[0][0],nh_delta0[0], d_hide0[0],  ni_dat[0]); 
	// bp_pid_dbg("sigmoid_dat=(%.4f,%.4f,%.4f) \r\n",ih_sigmoid_out[0],hh_sigmoid_out[0],bp_pid_th.ho_sigmoid_out[0] );
	// bp_pid_dbg(" net__dat=(%.4f,%.4f,%.4f,%.4f,%.4f)\r\n",ni_dat[NUM_ENV_TYPE*NUM_ENV_TYPEIN+PAR_GAIN_IN],ni_dat[NUM_ENV_TYPE*NUM_ENV_TYPEIN+PAR_GEAR_IN], net_ih_dat[0], net_hh_dat[1],net_oh_dat[0] );


    return c_ret_ok;
}

#if c_bp_pid_dbg_en == 1

#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"

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
#if 0
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

#endif


// 

// void get_target_diff(void)
// {
// 	target_diff[0][0]  = bp_pid_th.e[0][0];
// 	target_diff[0][1]  = -target_diff[0][0];
// 	target_diff[0][2]  = bp_pid_th.e[0][1];
// 	target_diff[0][3]  = -target_diff[0][2];
// 	target_diff[0][4]  = bp_pid_th.e[0][2];
// 	target_diff[0][5]  = -target_diff[0][4];
// }

unsigned int * pso_latin_permutation(void)
{
	int			 	i;
	unsigned int 	j, temp;
	static unsigned int perm[NUM_PARTICLES];
    for(i = 0; i < NUM_PARTICLES; i++)
	{
        perm[i] = i;
    }

    for(i = NUM_PARTICLES - 1; i >= 0; i--)
	{
        do 
		{
            j = rand();
        } while(j >= (RAND_MAX - RAND_MAX % (i + 1)));  // 得到均匀分布的[0, i]
        j %= (i + 1);

        temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
	return perm;
}

void pso_init(void) 
{
	  int i, d;
	pso.v_wight = 1e3; 
	unsigned int 	p, *perm ;
	float			interval, v_max; 
perm=pso_latin_permutation();
//bp_pid_dbg("perm[%d][%d][%d] \r\n",perm[0],perm[10],perm[20] );
//i=0x03&0x5&(1<<1);
//bp_pid_dbg("^(?!.*svd).+(\n|$)\r\n");
	//svd_init();
	// for(i = 0; i < 10; i++)
	// {
		perm=pso_latin_permutation();			
		for(d = 0; d < DIM; d++)
		{
			pso.mae_buf[0][d] =0.0f;
			pso.mae_buf[1][d] =0.0f;
			interval = (pso_pos_max_tab[d] - pso_pos_min_tab[d]) / NUM_PARTICLES;
			//v_max = (pso_pos_max_tab[d] - pso_pos_min_tab[d]) * 0.2f;
			for(p = 0; p < NUM_PARTICLES; p++)
			{
				pso.swarm[p].position[d] = pso_pos_min_tab[d] + (perm[p] + 0.5f * (float)rand() / RAND_MAX) * interval;
				//pso.swarm[p].velocity[d] = 2 * v_max * (float)rand()/RAND_MAX - v_max;				
			}
		}
	// 	for(p = 0; p < NUM_PARTICLES; p++)
	// 	{		
	// 		for(d = 0; d < NUM_ENV_TYPE; d++)
	// 		{	
	// 			if((pso.swarm[p].position[d*NUM_UPDOWN] >= pso_pos_max_tab[0] * 0.75f)&&(pso.swarm[p].position[d*NUM_UPDOWN+1] >= pso_pos_max_tab[1] * 0.75f)) 
	// 				break;
	// 			// if((pso.swarm[p].pos[2] >= pso_pos_max_tab[2] * 0.75f)&&(pso.swarm[p].pos[3] >= pso_pos_max_tab[3] * 0.75f)) 
	// 			// 	break;
	// 		}
	// 		if(d < NUM_ENV_TYPE) break;
	// 	}
	// 	if(p >= NUM_PARTICLES) 
	// 	{
	// 		for(p = 0; p < NUM_PARTICLES; p++)
	// 		{		
	// 			//pso.swarm[p].best_mae = 1e9; 
	// 			for(d = 0; d < DIM; d++) pso.swarm[p].best_pos[d] = pso.swarm[p].position[d]; 
	// 		}			
	// 		break;
	// 	}
	// }

	for(i = 0; i < NUM_PARTICLES; i++)
	{
         
		for(d = 0; d < DIM; d++){
			pso.swarm[i].v_idx[d]=0;
			//pso.swarm[i].position[d] = pso_pos_min_tab[d] + (i-1)*(pso_pos_max_tab[d]-pso_pos_min_tab[d])/NUM_PARTICLES;//pso.swarm[0].position[d];			
			pso.swarm[i].velocity[d] = 0.0f;// bp_pid_th.u_gain[d]/10.0;			
		}  
		
		//pso.buf_idx[0][i] = 0;
        pso.swarm[i].best_mae = 100.0f;
        memcpy(pso.swarm[i].best_pos, pso.swarm[i].position, sizeof(float) * DIM); 
    }
	pso.swarm_idx=perm[0];
	for(i = 0; i < NUM_GLOBAL; i++)	
    {
		pso.global_bestval = 100.0f;
		for(d = 0; d < DIM; d++){
			pso.global[i].pos[d] = pso.swarm[0].position[d];	
		}
	}
}





unsigned int pso_mae_val( double *fitness)
{
	unsigned int  	d=0;
	static unsigned char	du_status[DIM];
	double err=0;
	static unsigned int buf_idx=0;
	//pso.buf_cnt = 10 ; 
		*fitness=0;
		for(d=0;d<NUM_ENV_TYPE;d++)
		{
			// if(bp_pid_th.du_gain[d] < 0)  du_status[d] |= 1 << 0;
			// else  	du_status[d] |= 1 << 1; 	
		
			err =bp_pid_th.e[0][d];
			err = err + 0.01 * err * err +	0.01 * bp_pid_th.du_gain[d] ;/// pso_pos_max_tab[0]  ;
			pso.mae_buf[0][d] *=buf_idx;
			pso.mae_buf[0][d] += err; 
			pso.mae_buf[0][d] /=(buf_idx+1);
			pso.mae_buf[1][d]=(buf_idx==0  )?fabs(err):pso.mae_buf[1][d];	  
			pso.mae_buf[1][d]=(pso.mae_buf[1][d] < fabs(pso.mae_buf[0][d]))? pso.mae_buf[1][d]:fabs(pso.mae_buf[0][d]);
						
			// if((du_status[d] & 0x03) != 0x03) 
			// {
			// 	pso.mae_buf[pso.swarm_idx][d]  =50; 
			// 	du_status[d]=0;
			// }
			*fitness+= fabs(pso.mae_buf[0][d] );
		}
		
		buf_idx=buf_idx>=40? 0:buf_idx+1;

		//bp_pid_dbg("mae measaure[%d][%d] %.3f,%.3f,%.3f,%.3f\r\n",pso.swarm_idx,buf_idx,*fitness	,pso.mae_buf[1][ENV_T],pso.mae_buf[1][ENV_H],pso.mae_buf[1][ENV_V]);
		 
		 
	 
//*fitness=100;
	
	if(*fitness>50.0 ){
		bp_pid_dbg("fitness= %f reset[%d]:fit(%.2f,%.2f %.2f) \r\n", *fitness,pso.swarm_idx,pso.mae_buf[1][ENV_T],pso.mae_buf[1][ENV_H], pso.global_bestval);
	    return c_ret_nk;	 
	}
	// for(t_mae0 = 0, d = 0; d < pso.buf_cnt; d++)
	// 	t_mae0 += fabs(pso.t_buf[d] -  bp_pid_th.t_target);
	// t_mae0 /= d;

	// for(h_mae0 = 0, d = 0; d < pso.buf_cnt; d++)
	// 	h_mae0 += fabs(pso.h_buf[d] -  bp_pid_th.h_target);
	// h_mae0 /= d; 
 	

	return c_ret_ok;
}

unsigned int chex_swarm(double new_mae) 
{
	float r1=(float) rand() / RAND_MAX;	
	float r2=fabs(new_mae);
	//float r2=fabs(new_mae*1e1);
	static float max_mae=1;
	//static double min_mae=0;
	unsigned int pre_idx=pso.swarm_idx;
	if(r2 > max_mae) max_mae=r2;
	//if(r2 < min_mae) min_mae=r2;
	//r2=(float) (r2-min_mae)/(max_mae-min_mae);
	r2/=max_mae;
	if( r1  < r2)
	{
		
		unsigned int *perm= pso_latin_permutation();  
		//pso.swarm_idx =tmp_p% NUM_PARTICLES;//(unsigned int)  pid_map( (float) tmp_p ,  0,63,0, NUM_PARTICLES -1);
		//r1=(float) idx*NUM_PARTICLES / DIM;	
		unsigned int tmp_p =(unsigned int)  pid_map( r1 ,  0,r2,0, NUM_PARTICLES -1);	
		//pso.swarm_idx +=(unsigned int)  pid_map( bp_pid_th.u_gear_tmr[DEV_HU]  ,  0, 10,0, NUM_PARTICLES -1);	
		tmp_p=perm[tmp_p];
		pso.swarm_idx =perm[tmp_p]% NUM_PARTICLES;
		bp_pid_dbg("perm_swan[%d] to [%d] mae_rate=%f\r\n",pre_idx,pso.swarm_idx ,r2);
		//pso.swarm_idx +=	 (bp_pid_th.u_gear_tmr[DEV_VU]+bp_pid_th.u_gear_tmr[DEV_VD]) >0 ?1:0;
		return pre_idx;
	}
	return NUM_PARTICLES;
}
void pso_check(double new_mae)  //^(?!.*global_fit).+(\n|$)  
{
	
uint8 i=0,d=0;
	float fine_velocity=0;

	if(new_mae <  pso.swarm[pso.swarm_idx].best_mae)
	{ 
		
		bp_pid_dbg("swarm_fit=(%.3ft,%.3fh,%.3fn\r\n", pso.mae_buf[0][ENV_T], pso.mae_buf[0][ENV_H],pso.swarm[pso.swarm_idx].best_mae );
		pso.swarm[pso.swarm_idx].best_mae = new_mae;
		memcpy(pso.swarm[pso.swarm_idx].best_pos, pso.swarm[pso.swarm_idx].position, sizeof(float) * DIM);     
	}	
	if(new_mae <  pso.global_bestval)
	{
		bp_pid_dbg("global_fit=(%.3ft,%.3fh,%.3fg \r\n", pso.mae_buf[0][ENV_T], pso.mae_buf[0][ENV_H],pso.global_bestval );
		pso.global_bestval = new_mae;
		pso.global[0].swarm_idx = pso.swarm_idx;
		for(d = NUM_GLOBAL - 1; d > 0; d--){
				pso.global[d] = pso.global[d - 1];							
		}
		memcpy(pso.global[0].pos, pso.swarm[pso.swarm_idx].position, sizeof(float) * DIM);		
		for(i = 0; i <DIM; i++)
		{
			pso.global_position[i] =0.0f;										 
			for(d =0; d<NUM_GLOBAL; d++){
				pso.global_position[i] += (pso.global[d].pos[i]);
			}
		}
		pso.global_idx++;
		 		
		// if(pso.global_idx >=(NUM_GLOBAL/2)) {
		// 	pso.test_req = 0;
		// }
		if(pso.global_idx >=NUM_GLOBAL) {
			bp_pid_dbg("globlal reset [%d][%d]\r\n",pso.swarm_idx,pso.dev_token );
			pso.global_idx = 0;
			pso.dev_token=0;
			pso.global_bestval=100;
			pso.test_req = 1;
		}
	}	
	#if 1
	unsigned int pre_idx=chex_swarm(new_mae);
	if(pre_idx<NUM_PARTICLES)
	{
		float cur_pos=0;
		bp_pid_dbg("chex 0x%x,pso 0x%x,bp 0x%x\r\n",pso.dev_token&bp_pid_th.dev_token, pso.dev_token,bp_pid_th.dev_token);
		for(d = 0; d <DIM; d++)
		{
			uint8 x=pso.dev_token&bp_pid_th.dev_token&(1<<d);
			//float cur_pos= pso.swarm[pso.swarm_idx].position[d];
			if(x!=c_ret_ok)
			{
				cur_pos= pso.swarm[pso.swarm_idx].best_pos[d]+(pso.global_position[d]/NUM_GLOBAL)-pso.swarm[pso.swarm_idx].position[d]-pso.swarm[pso.swarm_idx].position[d];
				cur_pos=cur_pos*bp_pid_th.ho_sigmoid_out[NUM_DEV_KPID_OUT+d];
				fine_velocity =pid_map(cur_pos, c_pid_ptch_min,c_pid_ptch_max,c_pid_ptch_min,c_pid_ptch_max);							 
				pso.swarm[pso.swarm_idx].position[d]=pid_map(pso.swarm[pso.swarm_idx].position[d]+fine_velocity,  pso_pos_min_tab[d], pso_pos_max_tab[d],pso_pos_min_tab[d], pso_pos_max_tab[d]);
				bp_pid_dbg("chex [%d][%d](%fp, %.1fv,%.6fg glidx=%d\r\n",pso.swarm_idx,d,pso.swarm[pso.swarm_idx].position[d], fine_velocity ,pso.global_bestval,pso.global_idx  );
				 
			}
			else
			{
				
				pso.swarm[pso.swarm_idx].position[d]=pso.swarm[pre_idx].position[d];
			}
		}  	
	}	
	#endif
	return;
}
unsigned char   pso_get_val(void)
{	 
	static unsigned char check_time=0;
	int cur_time=tick_get() ; 
	//static unsigned char token=0 ;
	static unsigned char token[DIM];
	static int pre_tmr[2][DIM];
	static float target_diff[3][DIM];
	// double new_fitness;
	// float fine_velocity;
    //static float ptch_time_acc[DIM]; 
	unsigned int idx=0; 
    float tmp_hvac=0;
	pso.v_wight = 0.1;
	target_diff[0][DEV_TU]  = bp_pid_th.t_target-bp_pid_th.t_feed ;
	target_diff[0][DEV_TD]  = -target_diff[0][DEV_TU];
	target_diff[0][DEV_HU]  = (bp_pid_th.h_target-bp_pid_th.h_feed);
	target_diff[0][DEV_HD]  = -target_diff[0][DEV_HU];
	target_diff[0][DEV_VU]  = (bp_pid_th.v_target-bp_pid_th.v_feed)*10.0;
	target_diff[0][DEV_VD]  = -target_diff[0][DEV_VU];
	// target_diff[0][DEV_TU]  = bp_pid_th.s[ENV_T]-bp_pid_th.f[0][ENV_T] ;
	// target_diff[0][DEV_TD]  = -target_diff[0][0];
	// target_diff[0][DEV_HU]  =  bp_pid_th.s[ENV_H]-bp_pid_th.f[0][ENV_H];
	// target_diff[0][DEV_HD]  = -target_diff[0][2];
	// target_diff[0][DEV_VU]  =  bp_pid_th.s[ENV_V]-bp_pid_th.f[0][ENV_V];
	// target_diff[0][DEV_VD]  = -target_diff[0][4];
	for(idx=0;idx<DIM;idx++)
	{
		// unsigned int *perm= pso_latin_permutation(); 
		// bp_pid_th.ienerge[idx]=perm[idx%NUM_PARTICLES]*1e-2*((1.0- (0.5* (float)rand() / RAND_MAX)) );
		// bp_pid_th.pitch[0][idx]=perm[(idx+1)%NUM_PARTICLES]*1e-1*((1.0- (0.5* (float)rand() / RAND_MAX)) );
		// bp_pid_th.pitch[1][idx]=perm[(idx+2)%NUM_PARTICLES]*1e-1*((1.0- (0.5* (float)rand() / RAND_MAX)) );
		tmp_hvac=(target_diff[1][idx]-target_diff[0][idx])   ;
			
		if(target_diff[0][idx]> 0 &&bp_pid_th.u_gear_tmr[idx]==0)
		{
			target_diff[1][idx]=target_diff[0][idx];		
			pre_tmr[0][idx]=cur_time;
			token[idx]=(bp_pid_th.dev_token&(1<<idx))?1:0; 
			//token |= 1<<idx;
			//bp_pid_dbg("tag [%d][%d][0x%x] \r\n",idx,token[idx],bp_pid_th.dev_token );
			//bp_pid_dbg(" Start measure[%d][%d]=(%.1f,%.0f,%d,%.2f)\r\n",pso.swarm_idx,idx,pso.swarm[pso.swarm_idx].velocity[idx],pso.swarm[pso.swarm_idx].position[idx],(cur_time-pre_tmr[idx]),tmp_hvac); 
			//continue; 
		}
		//bp_pid_th.hvac_par[idx]=mae_tmr[0][idx]-mae_tmr[1][idx];
		 
		//else if(token[idx]==1 && bp_pid_th.du_gain[idx]<0.00f  &&  tmp_hvac >0.0f)
		else if(token[idx]==1 && bp_pid_th.du_gain[idx]<0.00f  &&bp_pid_th.pid_o[idx]>0 )
		{
			tmp_hvac=fabs(tmp_hvac)<1?1:tmp_hvac;
			uint8 tmp_token=(pso.mae_buf[1][idx>>1]> hvac_margin[idx>>1])? (1<<idx) :0;
			pso.dev_token |=tmp_token;	 
			svd.pitchs[idx][DEV_PTH_PW]=tmp_hvac*(float)(cur_time-pre_tmr[0][idx])/ bp_pid_th.tmr;
			svd.pitchs[idx][DEV_PTH_IN] =(float)(cur_time-pre_tmr[0][idx])*1e-3;
			svd.pitchs[idx][DEV_PTH_ON] =(float)(cur_time-pre_tmr[1][idx])*1e-3;
			for(int j=0;j<NUM_PTH_TYPE;j++)
			{						
				svd.avg_pitchs[idx][j]*=svd.avg_cnt[idx] ;
				svd.avg_pitchs[idx][j]+=svd.pitchs[idx][j];
				svd.avg_pitchs[idx][j]/=(svd.avg_cnt[idx] +1);
			}
			svd.avg_cnt[idx]=svd.avg_cnt[idx]>40?0:svd.avg_cnt[idx]+1;

			pso.swarm[pso.swarm_idx].position[idx]-=pso.swarm[pso.swarm_idx].velocity[idx];

			pso.swarm[pso.swarm_idx].velocity[idx] *=pso.swarm[pso.swarm_idx].v_idx[idx];			
			pso.swarm[pso.swarm_idx].velocity[idx] += ( (float) 1e3*pso.mae_buf[0][idx>>1]*svd.pitchs[idx][DEV_PTH_PW] ) ;
			pso.swarm[pso.swarm_idx].velocity[idx] /=(pso.swarm[pso.swarm_idx].v_idx[idx]+1);
			pso.swarm[pso.swarm_idx].v_idx[idx]=pso.swarm[pso.swarm_idx].v_idx[idx]>40?40:pso.swarm[pso.swarm_idx].v_idx[idx]+1;
			pso.swarm[pso.swarm_idx].velocity[idx] =pid_map(pso.swarm[pso.swarm_idx].velocity[idx],  c_pid_ptch_min, c_pid_ptch_max,c_pid_ptch_min, c_pid_ptch_max);
 
			pso.swarm[pso.swarm_idx].position[idx] += pso.swarm[pso.swarm_idx].velocity[idx];       				
				
			//bp_pid_dbg(" ENV velocity[%d][%d]=(%.1f,%.2f,%d,%.2f)\r\n",pso.swarm_idx,idx,pso.swarm[pso.swarm_idx].velocity[idx], tmp,(cur_time-pre_tmr[idx]),tmp_hvac); 
			//bp_pid_th.pid_o[idx]=-10.0f; 
			
			//token ^= (1<<idx);
			token[idx]=0;
   			pre_tmr[1][idx]=cur_time; 
			// int i=0;
			// for( i=0;i<NUM_PTH_TYPE;i++)
			// {
			// 	if(svd.pitchs[idx][i] >pitch_limit[1][i] )
			// 	{ 
			// 		break;
			// 	}
			// }
			//if(i>=NUM_PTH_TYPE)
			//{ 
				check_time|=(1<<idx);
			//}
bp_pid_dbg("check_idx[0x%x][%d][0x%x] ,pitch_energ=%.0f,%.0f,%.0f ,avg_pitchs=%.0f,%.0f,%.0f \r\n",check_time, idx,pso.dev_token
			,svd.pitchs[idx][DEV_PTH_PW],svd.pitchs[idx][DEV_PTH_IN] ,svd.pitchs[idx][DEV_PTH_ON],svd.avg_pitchs[idx][0] ,svd.avg_pitchs[idx][DEV_PTH_IN] ,svd.avg_pitchs[idx][DEV_PTH_ON] );
			
		} 
	} 
	unsigned char ck= check_time;
	 if(check_time!=0x00)
	 {	
	 	ck= ((check_time&0x0f) == (bp_pid_th.dev_token&0x0f));
	 	bp_pid_dbg("searching [%d][%d] [0x%x][0x%x]  \r\n", idx,ck,check_time,bp_pid_th.dev_token );
	 	check_time= ck  ?0:check_time;
	 }
	//bp_pid_dbg(" get feed=(%.2f,%.2f) tgt=(%.2f,%.2f) mae=(%.2f,%.2f,%.2f) \r\n" ,bp_pid_th.t_feed,bp_pid_th.h_feed,bp_pid_th.t_target,bp_pid_th.h_target, bp_pid_th.hvac_mae[0], bp_pid_th.hvac_mae[2] ,fitness);
	return ck;
}




static double update_particles (void)//float t_target, float t_feed, float h_target, float h_feed, float v_target, float v_feed) //^bp_pid_run.*$\r?\n
{
	static double  new_mae= 0;
	  float fine_velocity ; 
	unsigned char	check_idx=0;
	unsigned int d=0, i=0;
	float		 r1, r2; 
	switch(pso.step)
	{
		case c_pso_step_init:	 
			
			srand((unsigned int)time(NULL)); 
			pso.test_req = 1;
			// pso.buf_idx = 0;
			// pso.buf_cnt = 8;  
			//pso_init() ;
			for(d = 0; d < DIM; d++) 
			{		
				pso.swarm[pso.swarm_idx].position[d] =pso.swarm[pso.swarm_idx].best_pos[d] ;							 
			}
			
			bp_pid_dbg("new START:  fit=%.2f \r\n",   new_mae );  
			pso.global_bestval =100;
			pso.buf_cnt = 4;
			pso.swarm[pso.swarm_idx].best_mae=100;
			pso.step = c_pso_step_wait;   
			pso.dev_token=0;  
		break;
		
		case c_pso_step_wait:
			if(	pso_mae_val( &new_mae )==c_ret_ok)
			{
				pso.step = c_pso_step_update;	
			}
			else{
				bp_pid_dbg("reset START:  fit=%.2f \r\n",   new_mae ); 
				pso.step = c_pso_step_init;							
			}
		break;
		
		case c_pso_step_update:
		//update_particles_update:	 
	        
			
	    	//pso.step = __LINE__; return new_mae; case __LINE__: 
			check_idx=pso_get_val();
			if(check_idx==c_ret_ok)
			{ 
				pso.step = c_pso_step_wait;
			  	 // bp_pid_dbg("stay update %d check_idx %d new_mae=%f\r\n",pso.swarm_idx, check_idx ,new_mae);
				return new_mae;  
			}
			else{
				//bp_pid_dbg("finish update %d check_idx %d new_mae=%f\r\n",pso.swarm_idx, check_idx ,new_mae);				
				pso.step = c_pso_step_check; 
			}

		break;
		case c_pso_step_check :
			  	
				pso.step = c_pso_step_wait;	
				//calculate_svd(new_mae); 
				bp_pid_dbg("p00=%.4f,p01=%.4f,p02=%.4f,p20=%.4f,p21=%.4f,p22=%.4f,p40=%.4f,p41=%.4f,p42=%.4f;\r\n",svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2], svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2], svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2]);
			
				pso_check(new_mae);   
				//bp_pid_dbg("particle update[%.2f]: fit=%.2f,%.2f \r\n", new_mae, pso.swarm[pso.swarm_idx].best_mae,pso.global_bestval);

				//double chk_mae=fabs(pso.mae_buf[pso.swarm_idx][0]) + fabs(pso.mae_buf[pso.swarm_idx][1]);
		break;
			 
	
		default:   break;
	}	
	return  new_mae;
}
 
 
unsigned int  get_eco_update(void)
{
	int16 h=0,d=0;//,idx = 0;
	
 	for(h=0;h<NUM_ENVDEV;h++)
	{
	    float u_max= gear_list[h][10];
	    float u_min=-gear_list[h][0];
			//idx=h*NUM_ENVGOV_TYPEOUT+d;
			//bp_pid_th.pid_o[h] += (bp_pid_th.du[0][h]*pso.swarm[pso.swarm_idx].position[h]);//+bp_pid_th.u_gain_tmr[h]));
			bp_pid_th.pid_o[h] +=  (bp_pid_th.du_gain[h]  *pso.swarm[pso.swarm_idx].position[h]);
			if(  bp_pid_th.pid_o[h] >u_max)// bp_pid_th.u_max[h]) 
			{
				bp_pid_th.pid_o[h] = u_max;//bp_pid_th.u_max[h];
			}
			else if(bp_pid_th.pid_o[h] < u_min)//bp_pid_th.u_min[h]) 
			{
				bp_pid_th.pid_o[h]=u_min;// bp_pid_th.u_min[h];	
			}
		  	 
	}
    unsigned int on_tmr = tick_get();

	return on_tmr;
}

u_int8_t   find_gear_level( int  real_type ,int16 load_type,unsigned int on_tmr)
{
	u_int8_t idx = 1; 
	u_int8_t h_idx = 0; 
	u_int8_t max_gear=0;
	u_int8_t min_gear=0;
	float out=0; 
	float  p_out  =0 ; 
	u_int8_t *pgearout=NULL;
	unsigned int acctmr=  tick_get() - on_tmr; 

	// switch(  load_type  ) 
	// {
	// 	case loadType_heater:	h_idx=PAR_GAIN_TU_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_T+ENVGAN_UP);  p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_TOUT+ENVGAN_UP] ; max_gear= bp_pid_th.max_min_gear[0][0];min_gear=bp_pid_th.max_min_gear[1][0];break;
	// 	case loadType_A_C:		h_idx=PAR_GAIN_TD_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_T+ENVGAN_DOWN);p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_TOUT+ENVGAN_DOWN]; max_gear=bp_pid_th.max_min_gear[0][0];min_gear=bp_pid_th.max_min_gear[1][0];break;
	// 	case loadType_humi:		h_idx=PAR_GAIN_HU_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_H+ENVGAN_UP);  p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_HOUT+ENVGAN_UP] ; max_gear= bp_pid_th.max_min_gear[0][1];min_gear=bp_pid_th.max_min_gear[1][1];break;
	// 	case loadType_dehumi:	h_idx=PAR_GAIN_HD_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_H+ENVGAN_DOWN);p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_HOUT+ENVGAN_DOWN]; max_gear=bp_pid_th.max_min_gear[0][1];min_gear=bp_pid_th.max_min_gear[1][1];break;
	// 	case loadType_inlinefan:h_idx=PAR_GAIN_TU_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_V+ENVGAN_UP);  p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_HOUT+ENVGAN_UP] ; max_gear= bp_pid_th.max_min_gear[0][2];min_gear=bp_pid_th.max_min_gear[1][2];break;
	// 	case loadType_fan:      h_idx=PAR_GAIN_TU_OUT;pgearout=bp_pid_th.u_gear_tmr+(NUM_ENVGAN_TYPEOUT*ENV_V+ENVGAN_DOWN);p_out= bp_pid_th.pid_o[NUM_ENVGAN_TYPEOUT*GOV_HOUT+ENVGAN_DOWN]; max_gear=bp_pid_th.max_min_gear[0][2];min_gear=bp_pid_th.max_min_gear[1][2];break;
	// 	default:                h_idx=PAR_GAIN_TU_OUT;p_out= 0 ;  break;
	// }
	switch(  load_type  ) 
	{
		case loadType_heater:	h_idx=DEV_TU;  break;
		case loadType_A_C:		h_idx=DEV_TD;  break;
		case loadType_humi:		h_idx=DEV_HU;  break;
		case loadType_dehumi:	h_idx=DEV_HD;  break;
		case loadType_inlinefan:h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD; break;
		case loadType_fan:      h_idx=(bp_pid_th.v_outside- bp_pid_th.v_feed)>=0?DEV_VU:DEV_VD;   break;
		default:                return 0;
	}
	// if(load_type==loadType_inlinefan && (bp_pid_th.v_outside- bp_pid_th.v_feed)>=0 )
	// { 
	// 	h_idx = DEV_VU;
	// }
	//bp_pid_th.dev_token=(pso.mae_buf[1][h_idx>>1]>=hvac_margin[h_idx>>1])? (1<<h_idx) :bp_pid_th.dev_token; 
	bp_pid_th.dev_token|=(1<<h_idx);
	p_out= bp_pid_th.pid_o[h_idx] ;
	pgearout=bp_pid_th.u_gear_tmr+h_idx;
	if(p_out<=0){
		*pgearout= 0;
		return 0;
	}
	max_gear=bp_pid_th.port_setgear[1][h_idx];
	min_gear=bp_pid_th.port_setgear[0][h_idx];
	if( max_gear>10 ||max_gear==0) { max_gear=10;}
	if( min_gear>=10||min_gear<=1) { min_gear=1;}
	for(idx = min_gear; idx <= max_gear  ; idx++)
	{
		if( p_out <= gear_list[h_idx][idx])
		{
			break;
		}
	}
	unsigned int  on_tmo = (unsigned int)  pid_map( p_out, gear_list[h_idx][idx-1], gear_list[h_idx][idx], 0,(float) bp_pid_th.tmr);
	u_int8_t out_gear =(u_int8_t) (acctmr>= on_tmo) ? idx-1 :idx;

	if(  real_type == loadType_switch ) {
		 if( idx >=2)  //2
		    out_gear = idx ;
		else
		    out_gear = 0;
	}
	 
	 *pgearout= out_gear ;
	 //bp_pid_dbg("gear_level dev_type=(0x%x,%d),pid_o=%.2f,tmr(%d,%d,%d),out_gear=(%d,%d) ,max_mix=(%d,%d) \r\n", real_type,load_type,p_out,acctmr,on_tmo,bp_pid_th.tmr,  out_gear,idx,  max_gear,min_gear );
	 
	return out_gear ;
}  
 
static pid_run_output_st bp_pid_th_proc(short dev_type,uint8_t *input_dev_type  )  //^(?!.*svd).+(\n|$)
{		
    static unsigned int tmr = 0;//,geer_spk_tmr = 0;
	static unsigned int mode = 0, sec = 0;//, sn = 0;
	static unsigned int on_tmr=0;
	unsigned int	 idx;
	float				out;
	double				dx;
    static pid_run_output_st  output;
    dev_type_t devs_type_list[PORT_CNT];
	extern void get_devs_type_info(dev_type_t *devs_type_list);
	get_devs_type_info(devs_type_list);
    if(mode != bp_pid_th.mode)	
    {    
		mode = bp_pid_th.mode;    
        if(mode == 0)
        {
			for(idx=0;idx<NUM_ENVDEV;idx++)
			{
				bp_pid_th.pid_o[idx] = 0; 
			}
	    }
        else
        {
            tmr = tick_get() - bp_pid_th.tmr;
			bp_pid_tick_tmr=0;
			sec = tick_get() - 1000; 
		    //geer_spk_tmr=tick_get() - bp_pid_th.tmr;
			#if c_bp_pid_dbg_en == 1
			bp_pid_dbg("bp_pid_run tmr=%d bp_pid_th.tmr=%d sec=%d \r\n",tmr,bp_pid_th.tmr,sec);
			#endif
        }        
    }    
	if(mode != 0)
	{	   
    	if(tick_cmp(tmr, bp_pid_th.tmr  ) == c_ret_ok)
		{	 
			tmr += (bp_pid_th.tmr );
			bp_pid_th_exec(); 
			on_tmr=get_eco_update();  
 		 
			bp_pid_wave("t_feed=%.4f,h_feed=%.4f,vpd=%.4f\r\n",current_mae,bp_pid_th.t_feed,bp_pid_th.h_feed);	
			//bp_pid_dbg("p00=%.4f,p01=%.4f,p02=%.4f,p20=%.4f,p21=%.4f,p22=%.4f,p40=%.4f,p41=%.4f,p42=%.4f;\r\n",svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2], svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2], svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2]);
			//bp_pid_wave("t_feed=%f,h_feed=%f,vpd=%f\r\n", p_arg->t_feed, p_arg->h_feed, p_arg->v_feed);	
   			// bp_pid_dbg("sigmoid_dat=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f) \r\n",ni_dat[0],net_ih_dat[0],ih_sigmoid_dat[0],hh_sigmoid_dat[0],net_oh_dat[0],bp_pid_th.t_kp,nh_delta1[0],nh_delta[0],no_delta[0]);
 			if(pso.step==1)
			{
				char tmpstr='u';
				float tmp_v_val=pso.swarm[pso.swarm_idx].position[DEV_VU];
				float tmp_pid_o=bp_pid_th.pid_o[DEV_VU];
				float tmp_du_gain=bp_pid_th.du_gain[DEV_VU];
				if(bp_pid_th.dev_token&0x20)
				{   
					tmpstr='d';
					tmp_v_val=pso.swarm[pso.swarm_idx].position[DEV_VD];			
					tmp_pid_o=bp_pid_th.pid_o[DEV_VD];
					tmp_du_gain=bp_pid_th.du_gain[DEV_VD];
				}
				bp_pid_dbg("[%d][0x%x](%.0ftu %.0fhu %.0fv%c)tu(%.0f,%.3f)hu(%.0f,%.3f)v%c(%.0f,%.3f)kT(%.3f,%.3f,%.3f)mae(%.3f,%.3f,%.3f,%.3f)wave\r\n",pso.swarm_idx,bp_pid_th.dev_token
				,pso.swarm[pso.swarm_idx].position[DEV_TU], pso.swarm[pso.swarm_idx].position[DEV_HU],tmp_v_val,tmpstr   
				,bp_pid_th.pid_o[DEV_TU],bp_pid_th.du_gain[DEV_TU],bp_pid_th.pid_o[DEV_HU],bp_pid_th.du_gain[DEV_HU],tmpstr,tmp_pid_o,tmp_du_gain
				,bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KP],bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KI],bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KD]
				,current_mae,pso.mae_buf[0][ENV_T],pso.mae_buf[0][ENV_H],pso.mae_buf[0][ENV_V]);
			}
			 
		
		//} 
		

		//if(tick_cmp(geer_spk_tmr, bp_pid_th.tmr*2) == c_ret_ok)
		//{	
			 bp_pid_th.dev_token=0;//(pso.mae_buf[1][idx>>1]>=hvac_margin[idx>>1])? (1<<idx) :bp_pid_th.dev_token; 
			for(uint8_t port=1; port < PORT_CNT; port++ )
			{   
				output.speed[port] =find_gear_level(devs_type_list[port].real_type, input_dev_type[port],on_tmr );
				
			} 

			//bp_pid_dbg("speed tout(%.0f,%.0f,%.3f),hout(%.0f,%.0f,%.3f),vout(%.0f,%.0f,%.3f)\r\n",bp_pid_th.t_out,bp_pid_th.dt_out,bp_pid_th.t_du, bp_pid_th.h_out,bp_pid_th.dh_out,bp_pid_th.h_du, bp_pid_th.v_out,bp_pid_th.dv_out,bp_pid_th.v_du);
		  
			//geer_spk_tmr += (bp_pid_th.tmr*2); 
		} 	
		 
	}
	return output;
}




float pid_cal_vpd(float t, float rh)
{
	float dx;
	dx = (17.27f * t) / (t + 237.3f);
	dx = pid_exp(dx);
	dx = dx * 0.61078 * (1.0f - rh / 100.0f);
	return dx;
}


/// @brief 100ms call
/// @param input 
/// @return 
pid_run_output_st pid_run_rule(pid_run_input_st* input)
{
		
   short	 dev_type= 0;

    static pid_run_output_st 		output;
	//struct st_bp_pid_th_arg	pid_arg;
	static unsigned int 	en_bit = 0;
	//static unsigned int 	feed_idx = 0;
	//float in_side_vpd =ml_get_cur_vpd()/100.0; 
	//	memset(&output, 0x00, sizeof(output)); 
    bp_pid_tick_tmr += 100;

	if(bp_pid_th.version == 0)
	{
		//bp_pid_th_init();
		tick_reset();
		bp_pid_th.version = 2503061528;  
		pso.step=c_pso_step_init; 
		#if c_bp_pid_dbg_en == 1
		dbg_init();
		xTaskCreate(dbg_task_loop, 	"dbg_task", 2048*2, 	NULL, TASK_PRIO_MAIN, 	NULL);
		bp_pid_dbg("bp_pid init done! version:%u\r\n", bp_pid_th.version);  
		#endif
	}

    for( uint8_t env=0; env < ENV_CNT; env++  )
	{				
        if( input->env_en_bit &(1<<env) ){
            ESP_LOGD("ai pid","env[%d] cur_value[%d] min_value[%d] max_value[%d] target[%d]",
                env,input->env_value_cur[env],input->env_min[env],input->env_max[env], input->env_target[env] );
        }
    }
	if(en_bit != input->env_en_bit)
	{
		en_bit = input->env_en_bit;
		bp_pid_dbg("en_bit=0x%x\r\n", en_bit);
	}

	dev_type = 0;
    for(uint8_t port=1; port < PORT_CNT; port++ )
	{
        switch( input->dev_type[port] )
		{
            case loadType_heater:		dev_type |= (1 << loadType_heater);    bp_pid_th.port_setgear[1][DEV_TU] = input->max[port]; bp_pid_th.port_setgear[0][DEV_TU] = input->min[port]; break;
            case loadType_A_C:			dev_type |= (1 << loadType_A_C);	   bp_pid_th.port_setgear[1][DEV_TD] = input->max[port]; bp_pid_th.port_setgear[0][DEV_TD] = input->min[port]; break;
            case loadType_humi:			dev_type |= (1 << loadType_humi);      bp_pid_th.port_setgear[1][DEV_HU] = input->max[port]; bp_pid_th.port_setgear[0][DEV_HU] = input->min[port]; break;
            case loadType_dehumi:		dev_type |= (1 << loadType_dehumi);    bp_pid_th.port_setgear[1][DEV_HD] = input->max[port]; bp_pid_th.port_setgear[0][DEV_HD] = input->min[port]; break;
            case loadType_inlinefan:	dev_type |= (1 << loadType_inlinefan); bp_pid_th.port_setgear[1][DEV_VU] =bp_pid_th.port_setgear[1][DEV_VD] = input->max[port];
																		       bp_pid_th.port_setgear[0][DEV_VU] =bp_pid_th.port_setgear[0][DEV_VD] = input->min[port]; break; 
            case loadType_fan:	        dev_type |= (1 << loadType_fan);       bp_pid_th.port_setgear[1][DEV_VU] =bp_pid_th.port_setgear[1][DEV_VD] = input->max[port];
																		       bp_pid_th.port_setgear[0][DEV_VU] =bp_pid_th.port_setgear[0][DEV_VD] = input->min[port]; break;
        }		  	
    }
		// pid_arg.t_max_gear  =pid_arg.dt_max_gear  =pid_arg.h_max_gear  =pid_arg.dh_max_gear= pid_arg.v_max_gear  =pid_arg.dv_max_gear  = 10; 
		// pid_arg.t_min_gear  =pid_arg.dt_min_gear  =pid_arg.h_min_gear  =pid_arg.dh_min_gear= pid_arg.v_min_gear  =pid_arg.dv_min_gear  = 0; 
		
	if(dev_type != bp_pid_th.dev_type)
	{
		bp_pid_th.dev_type = dev_type;
		bp_pid_dbg("dev_type=0x%x\r\n", dev_type);
	}

   	bp_pid_th.mode = input->ml_run_sta;    
	
	//if(feed_idx != test_feed_idx)
	//{
	//	bp_pid_th.mode = 1;
	//	input->env_en_bit = (1 << ENV_TEMP) | (1 << ENV_HUMID)| (1 << ENV_VPD);
	//}	
    if((input->env_en_bit & (1 << ENV_TEMP))||(input->env_en_bit & (1 << ENV_HUMID))||(input->env_en_bit & (1 << ENV_VPD)))
    {
	//sensor_val_list[ENV_TEMP] = ml_get_cur_temp();
    //sensor_val_list[ENV_HUMID] = ml_get_cur_humid();
	//sensor_val_list[ENV_VPD] = ml_get_cur_vpd();
		extern s16 ml_get_cur_temp();
		extern s16 ml_get_cur_humid();
		extern s16 ml_get_outside_temp();
		extern s16 ml_get_outside_humid();
		extern s16 ml_get_outside_vpd();
		extern s16 ml_get_cur_vpd();

		bp_pid_th.t_target  = input->env_target[ENV_TEMP]/10.0f;
		bp_pid_th.t_feed    = ml_get_cur_temp()/10;//input->env_value_cur[ENV_TEMP]/10.0f;
		bp_pid_th.t_outside = ml_get_outside_temp()/10;//input->env_value_cur[ENV_TEMP]/10.0f;
		if(is_temp_unit_f())
		{	//C = (F - 32) × 5/9
			bp_pid_th.t_target = (bp_pid_th.t_target - 32) * 5 / 9;
			bp_pid_th.t_feed   = (bp_pid_th.t_feed   - 32) * 5 / 9;
			bp_pid_th.t_outside= (bp_pid_th.t_outside- 32) * 5 / 9;
		}
		bp_pid_th.h_target = input->env_target[ENV_HUMID]/10.0f;
		bp_pid_th.h_feed   = ml_get_cur_humid()/10;//nput->env_value_cur[ENV_HUMID]/10.0f;
		bp_pid_th.h_outside =   ml_get_outside_humid()/10;//nput->env_value_cur[ENV_HUMID]/10.0f;


		
		if(input->env_en_bit & (1 << ENV_VPD))
		{
			bp_pid_th.v_target = input->env_target[ENV_VPD]/100.0;
   			//bp_pid_dbg(" env v_target=%.2f,v_feed= %.2f, v_inside= %.2f  \r\n", pid_arg.v_target, pid_arg.v_feed,pid_arg.v_inside  );
		}
		else
		{
			bp_pid_th.v_target =  pid_cal_vpd(bp_pid_th.t_target, bp_pid_th.h_target) ; 
   			//bp_pid_dbg(" cal v_target=%.2f,t_target= %.2f, h_target= %.2f  \r\n", pid_arg.v_target, pid_arg.t_target,pid_arg.h_target  );
   			//bp_pid_dbg(" cal v_target=%.2f,v_feed= %.2f, v_inside= %.2f  \r\n", pid_arg.v_target, pid_arg.v_feed,pid_arg.v_inside  );
		}			
		bp_pid_th.v_feed    = ml_get_cur_vpd()/100.0;//input->env_value_cur[ENV_VPD]/10.0f;
		bp_pid_th.v_outside = ml_get_outside_vpd()/100.0;//in_side_vpd;
		// test_t_target	=bp_pid_th.v_outside;	
	    // test_h_target	=bp_pid_th.v_feed;	
	

	// if(feed_idx != test_feed_idx)
		// {
		// 	feed_idx = test_feed_idx;
		// 	pid_arg.t_feed = test_t_feed;
		// 	pid_arg.h_feed = test_h_feed;
		// 	pid_arg.t_target = test_t_target;
		// 	pid_arg.h_target = test_h_target;
		// }

		// bp_pid_th.t_target = pid_arg.t_target;
		// bp_pid_th.h_target = pid_arg.h_target;
		// bp_pid_th.v_target = pid_arg.v_target; 

		// bp_pid_th.t_feed   = pid_arg.t_feed;
		// bp_pid_th.h_feed   = pid_arg.h_feed;
        // bp_pid_th.v_feed   = pid_arg.v_feed; 
        output=	bp_pid_th_proc( dev_type,input->dev_type);		

	
    }
    return output;
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
	
	// siz_pid=sizeof(double)*c_no_nodes*c_nh_nodes;memcpy(&bp_pid_th.wt_no_nh, p_buf+len, siz_pid);len += siz_pid;
	// siz_pid=sizeof(double)*c_nh_nodes*c_ni_nodes;memcpy(&bp_pid_th.wt_nh_ni, p_buf+len, siz_pid);len += siz_pid;
	// siz_pid=sizeof(double)*c_nh_nodes*c_nh_nodes;memcpy(&bp_pid_th.wt_nh_nh, p_buf+len, siz_pid);len += siz_pid;


	return len;
 }



#define dbg_serial_usb_serial_jtag		1
#define dbg_serial_uart0				2
#define dbg_serial_sel        			dbg_serial_uart0
//#define dbg_serial_sel        			dbg_serial_usb_serial_jtag

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

//name,val/?
unsigned int dbg_parse_var(char *buf)
{
	char 				name_str[32], val_str[16];
	unsigned int 		i, val, typ, addr;
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
		if((name_str[0] == 'g')&&(strstr(name_str, "global_") != NULL))
		{
			addr += pso.global_idx * sizeof(struct pso_global);
		}	
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
			if(str2num(val_str, &val, &dx) != c_ret_ok) 
				return c_ret_nk;			
			switch(typ)
			{
				case 1:		*(float*)addr = (float)dx; break;
				case 2:		*(double*)addr = dx; break;
				case 7:		*(char*)addr = (char)val; break;						
				case 8:		*(unsigned char*)addr = (unsigned char)val; break;
				case 15:	*(short*)addr = (short)val; break;
				case 16:	*(unsigned short*)addr = (unsigned short)val; break;
				case 31:	*(int*)addr = (int)val; break;
				case 32:	*(unsigned int*)addr = val; break;
				default: 	return c_ret_nk;
			}
			bp_pid_dbg("var %s,%s OK\r\n", name_str, val_str);
			return c_ret_ok;
		}
	}
	return c_ret_nk;
}

#if 0


double **dmatrix(int nrl, int nrh, int ncl, int nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    int i,nrow=nrh-nrl+1,ncol=nch-ncl+1;
    double **m;
    /* allocate pointers to rows */
    m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
    m += NR_END;
    m -= nrl;
    /* allocate rows and set pointers to them */
    m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
    m[nrl] += NR_END;
    m[nrl] -= ncl;
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    /* return pointer to array of pointers to rows */
    return m;
}
 
double *dvector(int nl, int nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
    double *v;
    v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
    return v-nl+NR_END;
}
 
void free_dvector(double *v, int nl, int nh)
/* free a double vector allocated with dvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}
 
void free_dmatrix(double **v, int nrl, int nrh, int ncl, int nch)
/* free a double vector allocated with dvector() */
{
	for(int i=1;i<=nrl;i++) 
       free_dvector(v[i-NR_END], 1, nrh);//   free((FREE_ARG) v[i-NR_END]);
	free((FREE_ARG) v);	  
}
 #endif
 
double pythag(double a, double b)
/* compute (a2 + b2)^1/2 without destructive underflow or overflow */
{
    double absa,absb;
    absa=fabs(a);
    absb=fabs(b);
    if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
    else return (absb == 0.0 ? 1e-3 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}

struct svd_qsort
{
    double        vec  ; //[NUM_LAT+1][NUM_SPK+1]; 
    unsigned int  idx; //[NUM_SPK+1]
    
} ;
 
 void  vecnorm( double *u,double *a, int keyn) //latent dimension
{
     int i=0;
	 float sum=0;
	 for(i=base;i<keyn+base;i++)
	 {
		sum += a[i]  ;
	 }
	 for(i=base;i<keyn+base;i++)
	 {
		u[i] = a[i]/sum;
	 }
	 return ;
}
int cmpfunc(const void *a, const void *b) {
	//return (( (struct svd_qsort*)a)->vec < ((struct svd_qsort*)b)->vec);
	//return (( *(struct svd_qsort*)a).vec < (*(struct svd_qsort*)b).vec);
	return( (*(struct svd_qsort*)b).vec )> (*(struct svd_qsort*)a).vec?1:-1;
}

//const float epsilon0=5e-3; 

 static double a[MN][MN];
 static double v[MN][MN];
 static double w[MN];
 
void svdcmp0( int m, int n ) 
{
    int flag,i,iters,j,jj,k,l=0,nm;
    double anorm,c,f,g,h,s,scale,x,y,z, rv1[MN];
 	for (i = base; i <  m+base; i++) {  //M
		#if 1
			vecnorm(a[i],svd.arisk[i],NUM_KEYN);
		#else
			for (j = base; j < n+base; j++) { //N
				a[i][j] = svd.arisk[i][j];
			}
		#endif
	} 
	// for (k = 1; k <= m; k++)  
	// {
	// 	for(i=1;i<=n ;i++)
	// 	{ 
	// 		a[k][i]= svd.u_mat[k][i] ;
	// 	}
	// }
    //rv1=dvector(1,n);
    g=scale=anorm=0.0; /* Householder reduction to bidiagonal form */
    for (i=1;i<=n;i++) {
        l=i+1;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if (i <= m) {
            for (k=i;k<=m;k++) scale += fabs(a[k][i]);
            if (scale) {
                for (k=i;k<=m;k++) {
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
                }
                f=a[i][i];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                for (j=l;j<=n;j++) {
                    for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
                    f=s/h;
                    for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
                }
                for (k=i;k<=m;k++) a[k][i] *= scale;
            }
        }
        w[i]=scale *g;
        g=s=scale=0.0;
        if (i <= m && i != n) {
            for (k=l;k<=n;k++) scale += fabs(a[i][k]);
            if (scale) {
                for (k=l;k<=n;k++) {
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
                }
                f=a[i][l];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l]=f-g;
                for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
                for (j=l;j<=m;j++) {
                    for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
                    for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
                }
                for (k=l;k<=n;k++) a[i][k] *= scale;
            }
        }
        anorm = DMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
    }
    for (i=n;i>=1;i--) { /* Accumulation of right-hand transformations. */
        if (i < n) {
            if (g) {
                for (j=l;j<=n;j++) /* Double division to avoid possible underflow. */
                    v[j][i]=(a[i][j]/a[i][l])/g;
                for (j=l;j<=n;j++) {
                    for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
                    for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
                }
            }
            for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }
    for (i=IMIN(m,n);i>=1;i--) { /* Accumulation of left-hand transformations. */
        l=i+1;
        g=w[i];
        for (j=l;j<=n;j++) a[i][j]=0.0;
        if (g) {
            g=1.0/g;
            for (j=l;j<=n;j++) {
                for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
                f=(s/a[i][i])*g;
                for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
            }
            for (j=i;j<=m;j++) a[j][i] *= g;
        } else for (j=i;j<=m;j++) a[j][i]=0.0;
        ++a[i][i];
    }
 
    for (k=n;k>=1;k--) { /* Diagonalization of the bidiagonal form. */
        for (iters=1;iters<=4;iters++) {
            flag=1;
            for (l=k;l>=1;l--) { /* Test for splitting. */
                nm=l-1; /* Note that rv1[1] is always zero. */
                if ((double)(fabs(rv1[l])+anorm) == anorm) {
                    flag=0;
                    break;
                }
                if ((double)(fabs(w[nm])+anorm) == anorm) break;
            }
            if (flag) {
                c=0.0; /* Cancellation of rv1[l], if l > 1. */
                s=1.0;
                for (i=l;i<=k;i++) {
                    f=s*rv1[i];
                    rv1[i]=c*rv1[i];
                    if ((double)(fabs(f)+anorm) == anorm) break;
                    g=w[i];
                    h=pythag(f,g);
                    w[i]=h;
                    h=1.0/h;
                    c=g*h;
                    s = -f*h;
                    for (j=1;j<=m;j++) {
                        y=a[j][nm];
                        z=a[j][i];
                        a[j][nm]=y*c+z*s;
                        a[j][i]=z*c-y*s;
                    }
                }
            }
            z=w[k];
            if (l == k) { /* Convergence. */
                if (z < 0.0) { /* Singular value is made nonnegative. */
                    w[k] = -z;
                    for (j=1;j<=n;j++) v[j][k] = -v[j][k];
                }
                break;
            }
             
            x=w[l]; /* Shift from bottom 2-by-2 minor. */
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=pythag(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c=s=1.0; /* Next QR transformation: */
            for (j=l;j<=nm;j++) {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=pythag(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g = g*c-x*s;
                h=y*s;
                y *= c;
                for (jj=1;jj<=n;jj++) {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=pythag(f,h);
                w[j]=z; /* Rotation can be arbitrary if z = 0. */
                if (z) {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=c*g+s*y;
                x=c*y-s*g;
                for (jj=1;jj<=m;jj++) {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
struct svd_qsort q[NUM_SPK];  

	for(i=1;i<=n ;i++)
	{
		w[i]=(float_chk(w[i]) == c_ret_nk)?SIGN(1e-9,w[i]):w[i];  
		q[i-1].vec=w[i];
		q[i-1].idx=i;
		// for (k = 1; k <= n; k++)  
		// {
		// 	a[i][k]=(float_chk(a[i][k]) == c_ret_nk)?SIGN(1e-3,a[i][k]):a[i][k];   
		// 	//svd.v_mat[k][i]=(float_chk(v[k][i]) == c_ret_nk)?SIGN(1e-3,v[k][i]):v[k][i];   
		// }
	} 
	// for(i=1;i<=m ;i++)
	// {
	// 	//svd.w_vec[i]=(float_chk(w[i]) == c_ret_nk)?SIGN(1e-3,w[i]):w[i];  
	// 	for (k = 1; k <= m; k++)  
	// 	{
	// 		//svd.u_mat[i][k]=(float_chk(a[k][i]) == c_ret_nk)?SIGN(1e-3,a[k][i]):a[k][i];   
	// 		v[i][k]=(float_chk(v[i][k]) == c_ret_nk)?SIGN(1e-3,v[i][k]):v[i][k];   
	// 	}
	// } 
	 double t1[MN] ;
double t;

    qsort(q,m, sizeof(struct svd_qsort), cmpfunc);
	for (j =1; j <=  n; j++) {
		i=q[j-1].idx;
		w[j]=q[j-1].vec;		   
		for (k = 1; k <= m; k++) {
			t1[k] = a[k][i];
		}
		for (k = 1; k <= m; k++) {
			a[k][i] = a[k][j];
		}
		for (k = 1; k <= m; k++) {
			a[k][j] = t1[k];
		}
		for (k = 1; k <= n; k++) {
			t1[k] =v[i][k];
		}
		for (k = 1; k <= n; k++) {
			v[i][k] = v[j][k];
		}
		for (k = 1; k <= n; k++) {
			v[j][k] = t1[k];
		}
	}
	// for (i = 1; i <= n; i++) {
    //     for (j = i+1; j <= n; j++) {
    //         if (w[i] <w[j]) { /* 对特异值排序 */
    //             t = w[i];
    //             w[i] = w[j];
    //             w[j] = t;
    //             /* 同时也要把矩阵U,V的列位置交换 */
    //             /* 矩阵U */
    //             for (k = 1; k <= m; k++) {
    //                 t1[k] = a[k][i];
    //             }
    //             for (k = 1; k <= m; k++) {
    //                 a[k][i] = a[k][j];
    //             }
    //             for (k = 1; k <= m; k++) {
    //                 a[k][j] = t1[k];
    //             }
 
    //             /* 矩阵V */
    //             for (k = 1; k <= n; k++) {
    //                 t2[k] =v[i][k];
    //             }
    //             for (k = 1; k <= n; k++) {
    //                v[i][k] = v[j][k];
    //             }
    //             for (k = 1; k <= n; k++) {
    //                v[j][k] = t2[k];
    //             }
    //         }
    //     }
    // } 
   // free_dvector(rv1,1,n);
   for(i=1;i<=m ;i++)
	{
		 svd.w_vec[i]=  w[i];  
		for (k = 1; k <= m; k++)  
		{
			 svd.v_mat[i][k]=  (float_chk(a[i][k]) == c_ret_nk)?SIGN(1e-3,a[i][k]):a[i][k];   
			 svd.u_mat[i][k]=  (float_chk(v[i][k]) == c_ret_nk)?SIGN(1e-3,v[i][k]):v[i][k];   
		}
	} 
	
 }
 
 

void svdcmp( int m, int n ) 
{
    int flag,i,iters,j,jj,k,l=0,nm;
    double anorm,c,f,g,h,s,scale,x,y,z, rv1[MN];
 	for (i = base; i <  m+base; i++) {  //M
		#if 0
			vecnorm(a[i],svd.arisk[i],NUM_KEYN);
		#else
			for (j = base; j < n+base; j++) { //N
				a[i][j] = svd.arisk[i][j];
				//bp_pid_dbg(" a[%d][%d]=%.0f\r\n", i,j, a[i][j]);
		
			}
		#endif
	} 
	// for (k = 1; k <= m; k++)  
	// {
	// 	for(i=1;i<=n ;i++)
	// 	{ 
	// 		a[k][i]= svd.u_mat[k][i] ;
	// 	}
	// }
    //rv1=dvector(1,n);
    g=scale=anorm=0.0; /* Householder reduction to bidiagonal form */
    for (i=1;i<=n;i++) {
        l=i+1;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if (i <= m) {
            for (k=i;k<=m;k++) scale += fabs(a[k][i]);
            if (scale) {
                for (k=i;k<=m;k++) {
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
                }
                f=a[i][i];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                for (j=l;j<=n;j++) {
                    for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
                    f=s/h;
                    for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
                }
                for (k=i;k<=m;k++) a[k][i] *= scale;
            }
        }
        w[i]=scale *g;
        g=s=scale=0.0;
        if (i <= m && i != n) {
            for (k=l;k<=n;k++) scale += fabs(a[i][k]);
            if (scale) {
                for (k=l;k<=n;k++) {
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
                }
                f=a[i][l];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l]=f-g;
                for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
                for (j=l;j<=m;j++) {
                    for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
                    for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
                }
                for (k=l;k<=n;k++) a[i][k] *= scale;
            }
        }
        anorm = DMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
    }
    for (i=n;i>=1;i--) { /* Accumulation of right-hand transformations. */
        if (i < n) {
            if (g) {
                for (j=l;j<=n;j++) /* Double division to avoid possible underflow. */
                    v[j][i]=(a[i][j]/a[i][l])/g;
                for (j=l;j<=n;j++) {
                    for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
                    for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
                }
            }
            for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }
    for (i=IMIN(m,n);i>=1;i--) { /* Accumulation of left-hand transformations. */
        l=i+1;
        g=w[i];
        for (j=l;j<=n;j++) a[i][j]=0.0;
        if (g) {
            g=1.0/g;
            for (j=l;j<=n;j++) {
                for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
                f=(s/a[i][i])*g;
                for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
            }
            for (j=i;j<=m;j++) a[j][i] *= g;
        } else for (j=i;j<=m;j++) a[j][i]=0.0;
        ++a[i][i];
    }
 
    for (k=n;k>=1;k--) { /* Diagonalization of the bidiagonal form. */
        for (iters=1;iters<=4;iters++) {
            flag=1;
            for (l=k;l>=1;l--) { /* Test for splitting. */
                nm=l-1; /* Note that rv1[1] is always zero. */
                if ((double)(fabs(rv1[l])+anorm) == anorm) {
                    flag=0;
                    break;
                }
                if ((double)(fabs(w[nm])+anorm) == anorm) break;
            }
            if (flag) {
                c=0.0; /* Cancellation of rv1[l], if l > 1. */
                s=1.0;
                for (i=l;i<=k;i++) {
                    f=s*rv1[i];
                    rv1[i]=c*rv1[i];
                    if ((double)(fabs(f)+anorm) == anorm) break;
                    g=w[i];
                    h=pythag(f,g);
                    w[i]=h;
                    h=1.0/h;
                    c=g*h;
                    s = -f*h;
                    for (j=1;j<=m;j++) {
                        y=a[j][nm];
                        z=a[j][i];
                        a[j][nm]=y*c+z*s;
                        a[j][i]=z*c-y*s;
                    }
                }
            }
            z=w[k];
            if (l == k) { /* Convergence. */
                if (z < 0.0) { /* Singular value is made nonnegative. */
                    w[k] = -z;
                    for (j=1;j<=n;j++) v[j][k] = -v[j][k];
                }
                break;
            }
             
            x=w[l]; /* Shift from bottom 2-by-2 minor. */
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=pythag(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c=s=1.0; /* Next QR transformation: */
            for (j=l;j<=nm;j++) {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=pythag(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g = g*c-x*s;
                h=y*s;
                y *= c;
                for (jj=1;jj<=n;jj++) {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=pythag(f,h);
                w[j]=z; /* Rotation can be arbitrary if z = 0. */
                if (z) {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=c*g+s*y;
                x=c*y-s*g;
                for (jj=1;jj<=m;jj++) {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
	struct svd_qsort q[NUM_SPK];  

	for(i=1;i<=n ;i++)
	{
		w[i]=(float_chk(w[i]) == c_ret_nk)?SIGN(1e-9,w[i]):w[i];  
		q[i-1].vec=w[i];
		q[i-1].idx=i;
		// for (k = 1; k <= n; k++)  
		// {
		// 	a[i][k]=(float_chk(a[i][k]) == c_ret_nk)?SIGN(1e-3,a[i][k]):a[i][k];   
		// 	//svd.v_mat[k][i]=(float_chk(v[k][i]) == c_ret_nk)?SIGN(1e-3,v[k][i]):v[k][i];   
		// }
	} 
	// for(i=1;i<=m ;i++)
	// {
	// 	//svd.w_vec[i]=(float_chk(w[i]) == c_ret_nk)?SIGN(1e-3,w[i]):w[i];  
	// 	for (k = 1; k <= m; k++)  
	// 	{
	// 		//svd.u_mat[i][k]=(float_chk(a[k][i]) == c_ret_nk)?SIGN(1e-3,a[k][i]):a[k][i];   
	// 		v[i][k]=(float_chk(v[i][k]) == c_ret_nk)?SIGN(1e-3,v[i][k]):v[i][k];   
	// 	}
	// } 
	double t1[MN] ;
	double t;

    qsort(q,m, sizeof(struct svd_qsort), cmpfunc);
	for (j =1; j <=  n; j++) {
		i=q[j-1].idx;
		w[j]=q[j-1].vec;		   
		for (k = 1; k <= m; k++) {
			t1[k] = a[k][i];
		}
		for (k = 1; k <= m; k++) {
			a[k][i] = a[k][j];
		}
		for (k = 1; k <= m; k++) {
			a[k][j] = t1[k];
		}
		for (k = 1; k <= n; k++) {
			t1[k] =v[i][k];
		}
		for (k = 1; k <= n; k++) {
			v[i][k] = v[j][k];
		}
		for (k = 1; k <= n; k++) {
			v[j][k] = t1[k];
		}
	}
	 

   for(i=1;i<=m ;i++)
	{
		 svd.w_vec[i]=  w[i];  
		for (k = 1; k <= m; k++)  
		{
			 svd.u_mat[i][k]=  (float_chk(a[i][k]) == c_ret_nk)?SIGN(1e-3,a[i][k]):a[i][k];   
			 svd.v_mat[i][k]=  (float_chk(v[i][k]) == c_ret_nk)?SIGN(1e-3,v[i][k]):v[i][k];   
		}
	} 
 
	// for(i = 0; i <m   ; i ++)
	// 		{
	// 			for(  j = 0; j < m; j ++)
	// 			{
	// 				 dbg(" v[%d][%d]=%lf\r\n", j, i,svd.v_mat[i+base][j+base]);
	// 			}
	// 		}
	//int m=4;
	//int n=4;
	double W[MN][MN]={0};
	double Vt[MN][MN]={0};
	for (i = 1; i<= m; i++) {
        for (j =1; j <= n; j++) {
            if (i==j) {
                W[i][j] = svd.w_vec[i];
            } else {
                W[i][j] = 0.0;
            }
        }
    }
    for(i =base;i < m + base; i++){ 
		for(j =base;j <  m+base   ; j++){
			svd.uw[i][j] =0;
			for(k = base; k < n + base ; k++){ 
			    svd.uw[i][j] += ( svd.u_mat[i][k]*W[k][j])  ; 
			}
		} 
	}
	
double sum=0; 
 for(i = 1; i <= m; i++) {
        for(j = 1; j <= m; j++) {
            Vt[j][i] = svd.v_mat[i][j];
        }
    }
	for(i = 1; i <= m; i++) {
        for(j = 1; j <= m; j++) {
            svd.v_mat[i][j]=Vt[i][j]  ;
        }
    }
	for(i =base;i < m + base; i++){ 
		for(j =base;j <  m+base   ; j++){
			svd.wv[i][j] =0;
			for(k = base; k < n + base ; k++){ 
			    svd.wv[i][j] += (W[i][k]* svd.v_mat[k][j] )  ; 
			}
		} 
	} 
	for(i =base;i < m + base; i++){
	    for(j =base;j < m + base; j++){
			svd.latent_mat[i][j]  = 0;
			for(k = base; k < m + base ; k++){ 
			 	svd.latent_mat[i][j]  +=     svd.arisk[i][k] * svd.uw[k][j]  ;//profile[k];  
			}
		}
	   // dbg("latb[%d]=%.0f \r\n", i,svd.latent_vec[i]);
	}
#if 0
	double sumv[MN][MN]={0};
	double sumu[MN][MN]={0};
	for(i = 1; i <= n; i++){  
        for(j = 1; j <= m; j++){  
			for(k = 1; k <= m; k++){ 
				sumv[i][j] += svd.v_mat[k][i] * svd.v_mat[k][j];
			    sumu[i][j] += svd.u_mat[k][i] * svd.u_mat[k][j];
			}
		}
	}
	for(i = 1; i <= n; i++){  
        for(j = 1; j <= m; j++){  
			dbg("U*Ut[%d][%d] =%.0f V*Vt[%d][%d]=%.0f \r\n",i,j, sumu[i][j],i,j, sumv[i][j]); 
		}
	}
    for(i = 1; i <= n; i++){  
        for(j = 1; j <= m; j++){
            sum = 0;	 
            for(k = 1; k <= m; k++){
                sum += svd.uw[i][k] * svd.v_mat[k][j];
            }
            dbg("A[%d][%d]=%.0f \r\n", i,j,sum);
        } 
    }
	for(i = 1; i <= n; i++){  
        for(j = 1; j <= m; j++){
            sum = 0;	 
            for(k = 1; k <= m; k++){
                sum += svd.u_mat[i][k] * svd.wv[k][j];
            }
            dbg("AA[%d][%d]=%.0f \r\n", i,j,sum);
        } 
    }
	#endif
	
 }
    

 
void svd_port_latent( double *profile ,int spkidx) //latent dimension
 {
  
     int i, k,j; 
	int m=NUM_SPK;
	int n=NUM_KEYN;  
#if 0	
	for(i =base;i < n + base; i++){
		svd.latent_vec[i]  = 0;
		for(j = base; j < n + base ;j++){ 
			for(k = base; k < n + base ;k++){ 
				//	svd.latent_vec[i]    += svd.w_vec[i] * svd.u_mat[i][j]  *profile[j];
				
				svd.latent_vec[i] +=( j==k? svd.w_vec[k] * svd.v_mat[i][k] *profile[k]:0  );
				//sum[j] +=  svd.u_mat[j][k]  *profile[k] ; 
			}
		} 		
		//svd.latent_vec[i]=(float_chk(sum[i]) == c_ret_nk)?SIGN(1e-3,sum[i]):sum[i];   
		//svd.latent_vec[i]=(float_chk(acc) == c_ret_nk)?SIGN(1e-3,acc):acc;   
        //lat_in[i-1] = sum;  
		//  dbg("lata[%d]=%.0f \r\n", i,svd.latent_vec[i]);
	}
#else
	
	
	for(i =base;i < n + base; i++){
		svd.latent_vec[i]  = 0;
		for(k = base; k < m + base ; k++){ 
		 	svd.latent_vec[i]     +=  profile[k] * svd.uw[k][i]  ;  
		}
	 
	   // dbg("latb[%d]=%.0f \r\n", i,svd.latent_vec[i]);
	}
#endif  
	
    
	struct svd_qsort q[NUM_SPK];  

	double count0 = 0 , count1 = 0 , count2 = 0;
	static double ans_vec[NUM_SPK+1 ];
	 
	for(int k=base;k< m+base ;k++)
    {
		count2 += svd.latent_vec[k] *svd.latent_vec[k];
	}
	count2=sqrt(count2);
	for(int i=1;i<= m ;i++)
    {       
		 count0 = 0 ; count1 = 0;
		for (int k =base; k < L_GAIN+base ; k++)
		{  
#if 1			
            count0 += svd.latent_mat[i][k] * svd.latent_mat[i][k] ;
            count1 += svd.latent_vec[k]    * svd.latent_mat[i][k];
#else
            count0 += svd.v_mat[i][k] * svd.v_mat[i][k] ;
            count1 += svd.v_mat[i][k] * svd.latent_prj[k]; 
#endif	
		//	dbg("lat[%d][%d]=%.0f,%.0f, %lf \r\n",i,k,profile[k],svd.latent_vec[k],svd.v_mat[k][i]);
             
        }
		if (count0 ==0 || count2 ==0){
			ans_vec[i]=-1.0;			
		}
        else
        {  
		    //dbg("latent_prj[%d]=%f %f %f %f\r\n", i,svd.u_mat[spkidx+base][i],svd.latent_prj[i]/count2,svd.latent_vec[i],profile[i]);  
			//dbg("latent_prj[%d]=%f \r\n", i,svd.latent_prj[i]/count2);
			ans_vec[i]= count1/(sqrt(count0)*count2);
		}
		//q[idx].vec= svd.latent_prj[idx];//ans_vec[idx];
		q[i-1].vec=   ans_vec[i] ;//;
		q[i-1].idx=i-1;   
	}
	// size_t sz = sizeof(q[0]);
	// size_t num = sizeof(q) / sz;  struct svd_qsort q[NUM_SPK];  

    qsort(q,m, sizeof(struct svd_qsort), cmpfunc);
	// dbg("[0x%x][0x%x] lat(%.3f:%.3f:%.3f)vmat(%.3f:%.3f:%.3f)prf(%.0f:%.0f;%.0f:%.0f)\r\n",bp_pid_th.dev_token,spkidx 
	//	    ,svd.latent_vec[0+base],svd.latent_vec[1+base],svd.latent_vec[2+base] 
	//		,svd.v_mat[0+base][spkidx+base],svd.v_mat[1+base][spkidx+base],svd.v_mat[2+base][spkidx+base] 
	//		,profile[0*NUM_PTH_VP+0+base],profile[0*NUM_PTH_VP+1+base] 	,profile[1*NUM_PTH_VP+0+base],profile[1*NUM_PTH_VP+1+base]  
	//		);
	 bp_pid_dbg("[0x%x][0x%x] svd[0x(%x,%f)(%x,%f)(%x,%f)(%x,%f)] \r\n",bp_pid_th.dev_token,spkidx 
		    ,q[0].idx,q[0].vec,q[1].idx,q[1].vec,q[2].idx,q[2].vec,q[3].idx,q[3].vec 			
			);
	// bp_pid_dbg("[0x%x][0x%x] svd[(%d,%.2f)(%d,%.2f)(%d,%.2f)(%d,%.2f)(%d,%.2f)(%d,%.2f)(%d,%.2f)(%d,%.2f)] \r\n",bp_pid_th.dev_token,spkidx 
	// 	    ,q0[0].idx,q0[0].vec,q0[1].idx,q0[1].vec,q0[2].idx,q0[2].vec,q0[3].idx,q0[3].vec
	// 		,q0[4].idx,q0[4].vec,q0[5].idx,q0[5].vec,q0[6].idx,q0[6].vec,q0[7].idx,q0[7].vec  			
	// 		);
			//,profile[1] ,profile[2],profile[3]   );

	return  ;
	 
 }


void svd_init(void)
{
	unsigned int i=0,j=0,h=0,k=0;
	unsigned int tmp_spkidx=0;
	//unsigned int  tmp=0;
	for(i=base;i<NUM_SPK+base;i++)  //NUM_ENV_TH
	{		
		for(h=base;h<NUM_KEYN+base;h++)
		{
			svd.arisk[i ][h ]  += 5e-1*(1.0- (0.5* (float)rand() / RAND_MAX)) ;
		}
	}	 
	for(int tmp_spkidx  =0;tmp_spkidx <  NUM_SPK  ; tmp_spkidx++)			
	{	
		for(h = 0; h<NUM_ENV_TH ; h++) 
		{		
			for(i = 0; i < NUM_PTH_VP ; i++) 
			{				
				//unsigned char tf0 =(svd.pitchs[h+h][i]>=svd.avg_pitchs[h+h][i]?1:0); 
				unsigned char tf1=( (tmp_spkidx &   (1<<(h*NUM_PTH_VP+i)) )  ==0 ) ; 
				float tmp_f=( (   tf1 )?-1e-2: 1e-2); 
				//svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base] +=pid_map(  tmp_f,0,pitch_limit[1][i],0.01,1.0 );
				//svd.arisk[spkidx+base ][h*NUM_PTH_VP+i+base]  += (1e-1*tmp_f) ;
				  
				//bp_pid_dbg("tf=(%d %d %d)\r\n",tf0,tf1,tf0 ^ tf1);

				svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base]+= tmp_f;
			}			
		}
	}  
	svdcmp(  MN-1, MN-1 ); 
	
//JacobiSVD();
#if 0
	double profile[NUM_KEYN+1];
	int spkidx=0;
	for(spkidx  =0;spkidx <  NUM_SPK  ; spkidx++)			
	{
		for(h =0; h < NUM_ENV_TH ; h++)
		{
			for(i =0; i <  NUM_PTH_VP ; i++)
			{	 
				profile[h*NUM_PTH_TYPE+i+base] =  svd.pitchs[h+h][i];//pid_map(bp_pid_th.pitchs[h+h][i] ,0,pitch_limit[1][i],0.01,1.0 ) ;  
				//profile[h*NUM_PTH_VP+i+base] = svd.arisk[spkidx+base][h*NUM_PTH_VP+i+base] ;  
			}
		}	//double temp_res[NUM_SPK];
		svd_port_latent(profile,spkidx);
	}
#endif
	return;
}


unsigned int calculate_svd(double new_mae)  //  ^(?!.*svd).+(\n|$) 
{
  	
	double profile[MN];
	static unsigned int a_cnt =0;
	unsigned int  spkidx=0;
	int i=0,j=0,d=0,h=0; 
	 
	
	spkidx=0;
	unsigned int tf0 ;
	//static unsigned char spk_token [NUM_SPK+1] ;
	for(h = 0; h<NUM_ENV_TH  ; h++){ 
		unsigned int  tmpidx=0;
		for(i = 0; i < NUM_PTH_VP ; i++)
		{
			//tmp +=(svd.pitchs[h+h][i]>=pitch_limit[0][i]?(1<<i):0);  
			 tmpidx +=(svd.pitchs[h+h][i]>=svd.avg_pitchs[h+h][i]?(1<<i):0);  
			 bp_pid_dbg("spkidx=(%d %d %f %f)\r\n",tmpidx,spkidx,svd.pitchs[h+h][i],svd.avg_pitchs[h+h][i]);
	 
		} 		
		spkidx += (tmpidx  <<(h*NUM_PTH_VP))     ;	
	}	
	spkidx=spkidx%NUM_SPK;  
		
			//float tmp_f=( (spkidx &   (1<<(h*NUM_PTH_VP+i)) )  ==0 ? 0:1 ); 
			
			 
	for(h = 0; h<NUM_ENV_TH ; h++) 
		{		
			for(i = 0; i < NUM_PTH_VP ; i++) 
			{				
				unsigned char tf0 =(svd.pitchs[h+h][i]>=svd.avg_pitchs[h+h][i]?1:0); 
				unsigned char tf1=( (spkidx &   (1<<(h*NUM_PTH_VP+i)) )  ==0 ) ; 
				//float tmp_f=( (tf0 ^ tf1 )? tmp_pitch[1][i]:tmp_pitch[0][i]); 
				//svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base] +=pid_map(  tmp_f,0,pitch_limit[1][i],0.01,1.0 );
				//svd.arisk[spkidx+base ][h*NUM_PTH_VP+i+base]  += (1e-1*tmp_f) ;
				svd.arisk[spkidx+base][h*NUM_PTH_TYPE+i+base]*=svd.v_idx ; 
				svd.arisk[spkidx+base][h*NUM_PTH_TYPE+i+base]+=  ((tf0 ^ tf1 )?0:1);//=  svd.pitchs[h+h][i]  ;  
				svd.arisk[spkidx+base][h*NUM_PTH_TYPE+i+base]/=(svd.v_idx +1);  
				//bp_pid_dbg("tf=(%d %d %d)\r\n",tf0,tf1,tf0 ^ tf1);

				//svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base]= (tmp_spkidx+base)*NUM_KEYN+h*NUM_PTH_VP+i+base;
			}			
		}
#if 0
    for(int tmp_spkidx  =0;tmp_spkidx <  NUM_SPK  ; tmp_spkidx++)			
	{	
		for(h = 0; h<NUM_ENV_TH ; h++) 
		{		
			for(i = 0; i < NUM_PTH_VP ; i++) 
			{				
				unsigned char tf0 =(svd.pitchs[h+h][i]>=svd.avg_pitchs[h+h][i]?1:0); 
				unsigned char tf1=( (tmp_spkidx &   (1<<(h*NUM_PTH_VP+i)) )  ==0 ) ; 
				//float tmp_f=( (tf0 ^ tf1 )? tmp_pitch[1][i]:tmp_pitch[0][i]); 
				//svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base] +=pid_map(  tmp_f,0,pitch_limit[1][i],0.01,1.0 );
				//svd.arisk[spkidx+base ][h*NUM_PTH_VP+i+base]  += (1e-1*tmp_f) ;
				svd.arisk[tmp_spkidx+base][h*NUM_PTH_TYPE+i+base]*=svd.v_idx ; 
				svd.arisk[tmp_spkidx+base][h*NUM_PTH_TYPE+i+base]+=  ((tf0 ^ tf1 )?0:1);//=  svd.pitchs[h+h][i]  ;  
				svd.arisk[tmp_spkidx+base][h*NUM_PTH_TYPE+i+base]/=(svd.v_idx +1);  
				//bp_pid_dbg("tf=(%d %d %d)\r\n",tf0,tf1,tf0 ^ tf1);

				//svd.arisk[tmp_spkidx+base ][h*NUM_PTH_VP+i+base]= (tmp_spkidx+base)*NUM_KEYN+h*NUM_PTH_VP+i+base;
			}			
		}
	} 	
#endif
	svd.v_idx ++;

	  
	//svd.v_idx[spkidx]=svd.v_idx[spkidx]>=40?0:svd.v_idx[spkidx]+1;
	 if(a_cnt==0)
	{		
		svd_init();
		//for(i = 0; i <  NUM_SPK  ; i++)
        //   svd.v_idx[i]=1;
		//svd.flag =1;   
	}
	//unsigned int tmp_h=0;
	//for(h = 0; h < NUM_SPK ; h++)
	//{
	//    if(svd.v_idx[h]>1)tmp_h++;
	//}
	//svd.flag[0]=NUM_SPK;  
	bp_pid_dbg("spkidx=(0x%x %d)\r\n",spkidx,a_cnt);

	//if(svd.flag[0]>= (NUM_SPK>>1) && svd.flag[1]==0)  
	//{	 	   
		
		//svdcmp_1(NUM_SPK,NUM_KEYN); //spk,latent		
		 		
		//JacobiSVD(svd.u_mat,svd.v_mat,svd.w_vec);
	    //a_cnt =0;	  
		//bp_pid_dbg("svd=(%f: %f: %f: %f)  \r\n",pso.latent_vec[0],pso.latent_vec[1],pso.latent_vec[2],pso.latent_vec[3] );
  
		
		 
	//	svd.flag[1]=1; // a_cnt;
	//} 
		a_cnt++;
 
	//if(svd.flag ==1)
	{		
		for(h =0; h < NUM_ENV_TH ; h++)
		{
			for(i =0; i <  NUM_PTH_VP ; i++)
			{	 
				//profile[h*NUM_PTH_TYPE+i+base] =  svd.pitchs[h+h][i];//pid_map(svd.pitchs[h+h][i] ,0,pitch_limit[1][i],0.01,1.0 ) ;  
				profile[h*NUM_PTH_VP+i+base] = svd.arisk[spkidx+base][h*NUM_PTH_VP+i+base] ;  
			}
		}	//double temp_res[NUM_SPK];
		svd_port_latent(profile,spkidx);
		bp_pid_dbg("[0x%x]svd[0x%x] lat(%.3f:%.3f:%.3f)vmat(%lf:%lf:%lf)pitchs(%.0f:%.0f:%.0f ; %.0f:%.0f:%.0f ; %.0f:%.0f:%.0f)0x%x(%.0f:%.0f ; %.0f:%.0f )\r\n",bp_pid_th.dev_token,spkidx  
		    ,svd.latent_vec[0+base],svd.latent_vec[1+base],svd.latent_vec[2+base] 
			,svd.v_mat[0+base][spkidx+base],svd.v_mat[1+base][spkidx+base],svd.v_mat[2+base][spkidx+base] 
			,svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2]
			,svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2]
			,svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2] 
			,spkidx,svd.arisk[spkidx+base][0*NUM_PTH_VP+0+base],svd.arisk[spkidx+base][0*NUM_PTH_VP+1+base] 
			,svd.arisk[spkidx+base][1*NUM_PTH_VP+0+base],svd.arisk[spkidx+base][1*NUM_PTH_VP+1+base]  
			);
		  
		
	 
		if(a_cnt > ( NUM_SPK *4) )		 
		{
			svd.v_idx = 0;
			a_cnt=0;	
			for(i=base;i<NUM_SPK+base;i++)  //NUM_ENV_TH
			{		
				for(h=base;h<NUM_KEYN+base;h++)
				{
					svd.arisk[i ][h ]  = 5e-1*(1.0- (0.5* (float)rand() / RAND_MAX)) ;
				}
			}			
		}
	}
	 
	return a_cnt;
}				 

static unsigned int bp_pid_gain_write(void)
{
	nvs_handle_t my_handle;
	esp_err_t err;

	err = nvs_open("pid_gain", NVS_READWRITE, &my_handle);
	if (err != ESP_OK) {
		bp_pid_dbg("Error opening NVS namespace: %s\n", esp_err_to_name(err));
		return c_ret_ok;
	}
	nvs_set_i32(my_handle, "pid_rate_flag", (int)0xaaaa5555);
	nvs_set_i32(my_handle, "pid_t_rate", (int)bp_pid_th.du_gain[0]);
	nvs_set_i32(my_handle, "pid_dt_rate", (int)bp_pid_th.du_gain[1]);
	nvs_set_i32(my_handle, "pid_h_rate", (int)bp_pid_th.du_gain[2]);
	nvs_set_i32(my_handle, "pid_dh_rate", (int)bp_pid_th.du_gain[3]);
	nvs_commit(my_handle);
	nvs_close(my_handle);
	return c_ret_ok;
}
