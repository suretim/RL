#include "esp_log.h"
#include "string.h"
#include "ml_pid.h"
#include "math.h"
#include "Sensor.h" 
#include "nvs_flash.h"
#include "beep.h"
  
#include "ni_debug.h"
#define c_gain_ctrl_stop			0
#define c_gain_ctrl_req				1
#define c_gain_ctrl_auto			2

#include <time.h>
#include <sys/time.h>

#define c_pso_step_init				0
#define c_pso_step_wait				1
#define c_pso_step_update			2
#define c_pso_step_check			3

#define c_bp_pid_dbg_en 1
#define dbg_serial_usb_serial_jtag		1
#define dbg_serial_uart0				2
#define dbg_serial_sel        			dbg_serial_uart0
//#define dbg_serial_sel        			dbg_serial_usb_serial_jtag

static const char *TAG = "ML PID";
						//       t     dt   h     dh
const float pso_pos_max_tab[DIM]  = {30000,30000,20000,20000,10000,10000};
const float pso_pos_min_tab[DIM]  = {10000,10000,8000,8000,4000,4000};
unsigned int gain_adj_ctrl = 2;

dev_type_t devs_type_list[PORT_CNT];
pid_run_output_st lstm_pid_out_speed;

//float pitch_limit[2][NUM_PTH_TYPE]={{20,80,300},{40,160,600}};
//float tmp_pitch[NUM_UPDOWN][NUM_PTH_TYPE]= {{0.99,0.99,0.99 },{1.01,1.01,1.01 }};
unsigned int tmp_pdx[DIM]={0};
struct st_bp_pid_th bp_pid_th = {0};
struct pso_optimizer pso = {0};
//struct svd_optimizer svd = {0};
float hvac_margin[NUM_ENV_TYPE]={0.030,0.030,0.050};  
double          ni_dat[c_ni_nodes],  net_oh_dat[c_no_nodes],net_hh_dat[c_nh_nodes1],net_ih_dat[c_nh_nodes0];
double          no_delta1d[c_no_nodes],d_hide0[c_nh_nodes0],d_hide_o[c_no_nodes],d_hide1[c_nh_nodes1]; 
double          ih_sigmoid_out[c_nh_nodes0],hh_sigmoid_out[c_nh_nodes1] ;
double          nh_delta1[c_nh_nodes1],nh_delta0[c_nh_nodes0] ; 
static unsigned int  bp_pid_tick_tmr = 0;
const int base=1;
 
const float gear_list[NUM_ENVDEV][11]        = {
	{10, 100, 200, 300, 400, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
	{10, 220, 300, 380, 470, 550, 630, 710, 800, 880, 1000},
};  

const float fit_up_bound=8.0;
const float fit_low_bound=3.0;
double current_mae=0.0; 

static double update_particles (void);
void pso_init(void) ;
//void svd_init(void);
//unsigned int calculate_svd(double new_mae);
//void svd_port_latent( double *profile ,int spkidx);
float pid_map(float x, float in_min, float in_max, float out_min, float out_max)
{
    if(x < in_min) return out_min;
	if(x > in_max) return out_max;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min; 
}  
static unsigned int tick_get(void)
{  
	return bp_pid_tick_tmr;
}
static unsigned int tick_reset(void)
{  
	  bp_pid_tick_tmr=0;
	  return 0;
}
static unsigned int float_chk(double fx)
{
	if((isinf (fx) != 0)||(isnan(fx) != 0))
		return c_ret_nk;
	return c_ret_ok;
}

static void relu_derivative_from_net(double *dx, double *net, s16 len) {
    for (s16 i = 0; i < len; i++) {
        dx[i] = (net[i] > 0.0) ? 1.0 : 0.0;
        if (float_chk(dx[i]) == c_ret_nk) dx[i] = 0.0;
    }
}
static void tanh_derivative_from_act(double *dx, double *act, s16 len) {
	static double dvx[c_no_nodes];
    for (s16 i = 0; i < len; i++) {
        dx[i] = 1.0 - act[i] * act[i]; // act = tanh(net)
		//dx[i]=act[i]>0?act[i]-dvx[i]:0;	
		//dvx[i] = act[i];
        if (float_chk(dx[i]) == c_ret_nk) dx[i] = 0.0;
    }
}
static void sigmoid_o(double *dx,double *net,s16 len) 
{	
	for(s16 o = 0; o < len; o++) 
	{
		dx[o]=net[o]>0?1:0;
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
		dx[o]=x[o]>0?1:0;		
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


static unsigned int tick_cmp(int tmr, int tmo)
{  
	return (tick_get()  >= (tmo+tmr)) ? c_ret_ok : c_ret_nk; 
} 

void bp_pid_th_init(void) 
{
    unsigned int i, h, o;

	memset(&bp_pid_th, 0, sizeof(struct st_bp_pid_th));
	 
	bp_pid_th.tmr =  60000;
	bp_pid_th.dev_token = 0;
	 
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
     
	pso_init(); 
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


static double pid_exp(double x)
{
	if(x > 708) x = 708;
	else if(x < -708) x = -708;
	return exp(x);	
}
static unsigned int bp_pid_train(void)//float set0, float feed0, float set1, float feed1, float set2, float feed2 ) 
{
	double			dx=0; 
	unsigned int	i=0,  h=0, o=0;
    double          i_pid[NUM_ENV_IN];  
	bp_pid_th.s[ENV_T]    = pid_map(bp_pid_th.t_target,  plant_limit_params.temp_range.first, plant_limit_params.temp_range.second, 0, 1);
	bp_pid_th.f[0][ENV_T] = pid_map(bp_pid_th.t_feed,    plant_limit_params.temp_range.first, plant_limit_params.temp_range.second, 0, 1);
	bp_pid_th.s[ENV_H]    = pid_map(bp_pid_th.h_target,  plant_limit_params.humid_range.first, plant_limit_params.humid_range.second, 0, 1);
	bp_pid_th.f[0][ENV_H] = pid_map(bp_pid_th.h_feed,    plant_limit_params.humid_range.first, plant_limit_params.humid_range.second, 0, 1);    
	bp_pid_th.s[ENV_V]    = pid_map(bp_pid_th.v_target,  plant_limit_params.vpd_range.first, plant_limit_params.vpd_range.second , 0, 1);
	bp_pid_th.f[0][ENV_V] = pid_map(bp_pid_th.v_feed,    plant_limit_params.vpd_range.first, plant_limit_params.vpd_range.second , 0, 1); 
	
	//bp_pid_dbg(" bp_pid_th.t_target =%f bp_pid_th.h_target=%f, bp_pid_th.v_target=%f  \r\n",bp_pid_th.t_target,bp_pid_th.h_target,bp_pid_th.v_target);  
	//bp_pid_dbg(" bp_pid_th.t_feed =%f bp_pid_th.h_feed=%f, bp_pid_th.v_feed=%f  \r\n",bp_pid_th.t_feed,bp_pid_th.h_feed,bp_pid_th.v_feed);  
	//bp_pid_dbg(" plant_limit_params.first =%f plant_limit_params.second=%f \r\n",plant_limit_params.temp_range.first,plant_limit_params.temp_range.second);  
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
		//bp_pid_dbg(" ni_dat[%d]= %.2f \r\n",h*NUM_ENV_TYPEIN+PAR_SET_IN,ni_dat[h*NUM_ENV_TYPEIN+PAR_SET_IN]); 
	} 
	double prev_global_best = pso.global_bestval; 
 	current_mae=update_particles();  
	for(h = 0; h <(NUM_ENVDEV); h++)
	{ 
		ni_dat[NUM_ENV_IN+h] =pid_map(pso.swarm[pso.swarm_idx].velocity[h]  + 100.0*(1.0- (0.5* (float)rand() / RAND_MAX)),  c_pid_ptch_min, c_pid_ptch_max, 0, 1);
		ni_dat[NUM_ENV_IN+NUM_ENVDEV+h]=(float)bp_pid_th.u_gear_tmr[h]/10.0f  + 0.1*(1.0- (0.5* (float)rand() / RAND_MAX)); 
 	 	 
		//bp_pid_dbg(" ni_dat[%d]= %.2f \r\n",NUM_ENVDEV+h,ni_dat[h]);
	}
	for(h = 0; h <( NUM_ENV_TYPE); h++)
	{	i_pid  [h*NUM_ENV_KPID+ENV_KP]   = bp_pid_th.e[0][h] -   bp_pid_th.e[1][h];
		i_pid  [h*NUM_ENV_KPID+ENV_KI]   = bp_pid_th.e[0][h];
		i_pid  [h*NUM_ENV_KPID+ENV_KD]   = bp_pid_th.e[0][h] - 2*bp_pid_th.e[1][h] + bp_pid_th.e[2][h];	
	}    
	for(h = 0; h < NUM_ENV_TYPE; h++){
		bp_pid_th.f [1][h]  =  bp_pid_th.f [0][h] ; 
		bp_pid_th.e [2][h]  =  bp_pid_th.e [1][h];
		bp_pid_th.e [1][h]  =  bp_pid_th.e [0][h]; 
	} 
	
  
	for(h = 0; h < c_nh_nodes0; h++)
	{
		net_ih_dat[h] = 0.0f;
		for(i = 0; i < c_ni_nodes; i++)
		{
			net_ih_dat[h] += ni_dat[i] * bp_pid_th.wt_nh_ni[h][i];
		}  
		//bp_pid_dbg(" net_ih_dat= %.2f \r\n",net_ih_dat[h]);
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
	 	//bp_pid_dbg(" net_oh_dat= %.2f \r\n",net_oh_dat[h]);
	} 
    sigmoid_o(bp_pid_th.ho_sigmoid_out,net_oh_dat,c_no_nodes); 
	for(h = 0; h < NUM_ENV_TYPE; h++)
	{
		//*pso.swarm[pso.swarm_idx].position[h] //+pso.swarm[pso.swarm_idx].velocity[h]
		bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_UP]=bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KP] *i_pid[h*NUM_ENV_KPID+ENV_KP]
		                					  +bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KI] *i_pid[h*NUM_ENV_KPID+ENV_KI]
						  					  +bp_pid_th.ho_sigmoid_out[h*NUM_ENV_KPID+ENV_KD] *i_pid[h*NUM_ENV_KPID+ENV_KD];
		bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_DOWN]= - bp_pid_th.du_gain[h*NUM_UPDOWN+ENV_UP];
		
	}
	bp_pid_dbg(" du_gain= (%.2f,%.2f,%.2f)\r\n",bp_pid_th.du_gain[0],bp_pid_th.du_gain[2],bp_pid_th.du_gain[4]);
	derivative_o(d_hide_o,bp_pid_th.ho_sigmoid_out,c_no_nodes ); 
	derivative_tanh(d_hide1,hh_sigmoid_out,c_nh_nodes1 );
	derivative_tanh(d_hide0,ih_sigmoid_out,c_nh_nodes0 );
	
	bp_pid_th.update_rate=0.001;
	static double desired_out_static[c_no_nodes];
	double *desired_out = desired_out_static;
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
#if 1	
	if (true){//pso.global_bestval < prev_global_best) {
		// 1) 準備 desired_out（監督目標）——把 pso.global_position 映射到網路輸出長度
		//    假定 c_no_nodes 為網路輸出的大小；pso.global_position 長度為 DIM
		//    若 DIM >= c_no_nodes，採用前 c_no_nodes 元素；若 DIM < c_no_nodes，重複/填 0
		 	
		int i_map;
		for (i_map = 0; i_map < c_no_nodes; i_map++) {
			if (i_map < NUM_DEV_KPID_OUT) {
				desired_out[i_map] = Etotal;
			}
			else{
				// pso.global_position 可能範圍很大，需正規化到網路輸出 activation 範圍(例如 0..1)
				// 假設 pso.global_position 已是可用的尺度，否則用 pid_map 做 min-max mapping
				desired_out[i_map] =  pso.global_position[i_map-NUM_DEV_KPID_OUT]; // 若需要，可再 normalize
			}
		}
		bp_pid_dbg(" Etotal,desired_out[NUM_DEV_KPID_OUT] =(%.3f,%.6f)\r\n",Etotal, desired_out[NUM_DEV_KPID_OUT]);

		//pso.swarm[pso.swarm_idx].position[d]
		// OPTIONAL: 如果 desired_out 需要映射到 activation 範圍（0..1），請啟用下面的範例 normalizer
		for (i_map = 0; i_map < DIM; i_map++) {
		    desired_out[i_map+NUM_DEV_KPID_OUT] = pid_map((float)desired_out[i_map+NUM_DEV_KPID_OUT],pso_pos_min_tab[i_map],pso_pos_max_tab[i_map], 0.0f, 1.0f);
		}

		// 2) 正確計算導數（使用 pre-activation net_oh_dat、net_hh_dat、net_ih_dat）
		//    假設輸出層使用 ReLU（你的 sigmoid_o 其實是 ReLU）
		relu_derivative_from_net(d_hide_o, net_oh_dat, c_no_nodes);
		// 隱藏層使用 tanh/或你原本的 activation（此處用 act 儲存)
		tanh_derivative_from_act(d_hide1, hh_sigmoid_out, c_nh_nodes1);
		tanh_derivative_from_act(d_hide0, ih_sigmoid_out, c_nh_nodes0);

		// 3) 計算 output 層 delta (per-node)
		//    no_delta[h] = (target - output) * derivative(output_net)
		static double no_delta_static[c_no_nodes];
		double *no_delta = no_delta_static;
		sigmoid_o(desired_out,desired_out,c_no_nodes);
		for (unsigned int h = 0; h < (unsigned int)c_no_nodes; h++) {
			double y_pred   = bp_pid_th.ho_sigmoid_out[h];
			 
			double err_h = desired_out[h] - y_pred;
			no_delta[h] = err_h * d_hide_o[h];  // d_hide_o 为 ReLU/Sigmoid 导数  
			//double err_h = desired_out[h] - bp_pid_th.ho_sigmoid_out[h];
			//no_delta[h] = err_h * d_hide_o[h];
			if (float_chk(no_delta[h]) == c_ret_nk) no_delta[h] = 0.0;
		}

		// 4) 隱藏層 1 delta（從 output 反向）

		static double nh_delta1_static[c_nh_nodes1];
		double *nh_delta1 = nh_delta1_static;
		for (unsigned int o = 0; o < (unsigned int)c_nh_nodes1; o++) {
			double sum = 0.0;
			for (unsigned int h = 0; h < (unsigned int)c_no_nodes; h++) {
				sum += no_delta[h] * bp_pid_th.wt_no_nh[h][o];
			}
			nh_delta1[o] = sum * d_hide1[o];
			if (float_chk(nh_delta1[o]) == c_ret_nk) nh_delta1[o] = 0.0;
		}

		// 5) 隱藏層 0 delta
		static double nh_delta0_static[c_nh_nodes0];
		double *nh_delta0 = nh_delta0_static;
		for (unsigned int o = 0; o < (unsigned int)c_nh_nodes0; o++) {
			double sum = 0.0;
			for (unsigned int h = 0; h < (unsigned int)c_nh_nodes1; h++) {
				sum += nh_delta1[h] * bp_pid_th.wt_nh_nh[h][o];
			}
			nh_delta0[o] = sum * d_hide0[o];
			if (float_chk(nh_delta0[o]) == c_ret_nk) nh_delta0[o] = 0.0;
		}

		// 6) 權重更新（標準式： w += lr * delta * prev_activation ）
		double lr = bp_pid_th.update_rate; // 你已有的學習率
		// output <- hidden1 (wt_no_nh[h][i])
		for (unsigned int h = 0; h < (unsigned int)c_no_nodes; h++) {
			for (unsigned int i = 0; i < (unsigned int)c_nh_nodes1; i++) {
				double delta_w = lr * no_delta[h] * hh_sigmoid_out[i];
				if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
				bp_pid_th.wt_no_nh[h][i] += delta_w;
			}
		}
		// hidden1 <- hidden0 (wt_nh_nh)
		for (unsigned int h = 0; h < (unsigned int)c_nh_nodes1; h++) {
			for (unsigned int i = 0; i < (unsigned int)c_nh_nodes0; i++) {
				double delta_w = lr * nh_delta1[h] * ih_sigmoid_out[i];
				if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
				bp_pid_th.wt_nh_nh[h][i] += delta_w;
			}
		}
		// hidden0 <- input (wt_nh_ni)
		for (unsigned int h = 0; h < (unsigned int)c_nh_nodes0; h++) {
			for (unsigned int i = 0; i < (unsigned int)c_ni_nodes; i++) {
				double delta_w = lr * nh_delta0[h] * ni_dat[i];
				if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
				bp_pid_th.wt_nh_ni[h][i] += delta_w;
			}
		}
			
		// （可選）把部分更新寫入 log，或限制權重大小避免發散
		// for safety: clip weights to reasonable range
		// for (h...) for(i...) bp_pid_th.wt... = clamp(bp_pid_th.wt..., -WMAX, WMAX);
	}
#else	
	
	for(o = 0; o < c_nh_nodes0 ; o++)
	{
		nh_delta0[o] = 0;
		for(h = 0; h < c_nh_nodes1; h++)
		{ 
			nh_delta0[o] += nh_delta1[h] * bp_pid_th.wt_nh_nh[h][o] * d_hide0[o];
		}
		//bp_pid_dbg("nhi =(%.6f,%.6f,%.6f,%.6f)\r\n", nh_delta0[o], d_hide1[o] ,net_hh_dat[o],bp_pid_th.wt_nh_nh[0][o] );
	}   
    //权值更新
    for(h = 0; h < c_no_nodes; h++) 
	{ 
        for(i = 0; i < c_nh_nodes1; i++) 
		{ 
		    bp_pid_th.wt_no_nh[h][i] +=  bp_pid_th.update_rate  * Etotal  * d_hide_o[h]  * hh_sigmoid_out[i];
		   }
    }
	for(h = 0; h < c_nh_nodes1; h++) 
	{
		for(i = 0; i < c_nh_nodes0; i++) 
		{	 
			 bp_pid_th.wt_nh_nh[h][i] +=   bp_pid_th.update_rate  * nh_delta1[h]  * d_hide1[h]  * ih_sigmoid_out[i];		 
		} 
	}  
    for(h = 0; h < c_nh_nodes0; h++) 
	{     
       for(i = 0; i < c_ni_nodes; i++) 
		{  
            bp_pid_th.wt_nh_ni[h][i] +=  bp_pid_th.update_rate * nh_delta0[h]  * d_hide0[h]* ni_dat[i]; 
        }
    }  
#endif	
	
	return c_ret_ok;
}  
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
	static float max_mae=1; 
	unsigned int pre_idx=pso.swarm_idx;
	if(r2 > max_mae) max_mae=r2; 
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
		pso.swarm[pso.swarm_idx].best_mae = new_mae + rand() / (float) RAND_MAX * 0.01;
		memcpy(pso.swarm[pso.swarm_idx].best_pos, pso.swarm[pso.swarm_idx].position, sizeof(float) * DIM);     
	 	
		if(pso.swarm[pso.swarm_idx].best_mae <  pso.global_bestval)
		{
			bp_pid_dbg("global_fit=(%.3ft,%.3fh,%.3fg \r\n", pso.mae_buf[0][ENV_T], pso.mae_buf[0][ENV_H],pso.global_bestval );
			pso.global_bestval = pso.swarm[pso.swarm_idx].best_mae + rand() / (float) RAND_MAX * 0.01;
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
				pso.global_position[i]/=NUM_GLOBAL;
			}
			// pso.global_idx++;			
			// if(pso.global_idx >=NUM_GLOBAL) {
			// 	bp_pid_dbg("globlal reset [%d][%d]\r\n",pso.swarm_idx,pso.dev_token );
			// 	pso.global_idx = 0;
			// 	pso.dev_token=0;
			// 	pso.global_bestval=100;
			// 	pso.test_req = 1;
			// }
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
				bp_pid_dbg("chex [%d][%d](%fp, %.1fv,%.6fglobal \r\n",pso.swarm_idx,d,pso.swarm[pso.swarm_idx].position[d], fine_velocity ,pso.global_bestval);
				 
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
	static unsigned char token[DIM];
	static int pre_tmr[2][DIM];
	static float target_diff[3][DIM]; 
	unsigned int idx=0; 
    float tmp_hvac=0;
	pso.v_wight = 0.1;
	target_diff[0][DEV_TU]  =  bp_pid_th.s[ENV_T]-bp_pid_th.f[0][ENV_T] ; //bp_pid_th.t_target-bp_pid_th.t_feed ;
	target_diff[0][DEV_TD]  = -target_diff[0][DEV_TU];
	target_diff[0][DEV_HU]  = (bp_pid_th.s[ENV_H]-bp_pid_th.f[0][ENV_H]); //(bp_pid_th.h_target-bp_pid_th.h_feed);
	target_diff[0][DEV_HD]  = -target_diff[0][DEV_HU];
	target_diff[0][DEV_VU]  =  (bp_pid_th.s[ENV_V]-bp_pid_th.f[0][ENV_V]) ;//(bp_pid_th.v_target-bp_pid_th.v_feed)*10.0;
	target_diff[0][DEV_VD]  = -target_diff[0][DEV_VU];
	 
	for(idx=0;idx<DIM;idx++)
	{  
		tmp_hvac=(target_diff[1][idx]-target_diff[0][idx])   ;
			
		if(target_diff[0][idx]> 0 &&bp_pid_th.u_gear_tmr[idx]==0)
		{
			target_diff[1][idx]=target_diff[0][idx];		
			pre_tmr[0][idx]=cur_time;
			token[idx]=(bp_pid_th.dev_token&(1<<idx))?1:0; 
			 
		}
		//bp_pid_th.hvac_par[idx]=mae_tmr[0][idx]-mae_tmr[1][idx];
		 
		//else if(token[idx]==1 && bp_pid_th.du_gain[idx]<0.00f  &&  tmp_hvac >0.0f)
		else if(token[idx]==1 && bp_pid_th.du_gain[idx]<0.00f  &&bp_pid_th.pid_o[idx]>0 )
		{
			tmp_hvac=fabs(tmp_hvac)<1?1:tmp_hvac;
			uint8 tmp_token=(pso.mae_buf[1][idx>>1]> hvac_margin[idx>>1])? (1<<idx) :0;
			pso.dev_token |=tmp_token;	 
			// svd.pitchs[idx][DEV_PTH_PW]=tmp_hvac*(float)(cur_time-pre_tmr[0][idx])/ bp_pid_th.tmr;
			// svd.pitchs[idx][DEV_PTH_IN] =(float)(cur_time-pre_tmr[0][idx])*1e-3;
			// svd.pitchs[idx][DEV_PTH_ON] =(float)(cur_time-pre_tmr[1][idx])*1e-3;
			// for(int j=0;j<NUM_PTH_TYPE;j++)
			// {						
			// 	svd.avg_pitchs[idx][j]*=svd.avg_cnt[idx] ;
			// 	svd.avg_pitchs[idx][j]+=svd.pitchs[idx][j];
			// 	svd.avg_pitchs[idx][j]/=(svd.avg_cnt[idx] +1);
			// }
			// svd.avg_cnt[idx]=svd.avg_cnt[idx]>40?0:svd.avg_cnt[idx]+1;

			pso.swarm[pso.swarm_idx].position[idx]-=pso.swarm[pso.swarm_idx].velocity[idx];

			pso.swarm[pso.swarm_idx].velocity[idx] *=pso.swarm[pso.swarm_idx].v_idx[idx];			
			//pso.swarm[pso.swarm_idx].velocity[idx] += ( (float) 1e3*pso.mae_buf[0][idx>>1]*svd.pitchs[idx][DEV_PTH_PW] ) ;
			pso.swarm[pso.swarm_idx].velocity[idx] /=(pso.swarm[pso.swarm_idx].v_idx[idx]+1);
			pso.swarm[pso.swarm_idx].v_idx[idx]=pso.swarm[pso.swarm_idx].v_idx[idx]>40?40:pso.swarm[pso.swarm_idx].v_idx[idx]+1;
			pso.swarm[pso.swarm_idx].velocity[idx] =pid_map(pso.swarm[pso.swarm_idx].velocity[idx],  c_pid_ptch_min, c_pid_ptch_max,c_pid_ptch_min, c_pid_ptch_max);
 
			pso.swarm[pso.swarm_idx].position[idx] += pso.swarm[pso.swarm_idx].velocity[idx];       				
				
			//bp_pid_dbg(" ENV velocity[%d][%d]=(%.1f,%.2f,%d,%.2f)\r\n",pso.swarm_idx,idx,pso.swarm[pso.swarm_idx].velocity[idx], tmp,(cur_time-pre_tmr[idx]),tmp_hvac); 
			//bp_pid_th.pid_o[idx]=-10.0f; 
			
			//token ^= (1<<idx);
			token[idx]=0;
   			pre_tmr[1][idx]=cur_time; 
			check_time|=(1<<idx);
			 
			bp_pid_dbg("check_idx[0x%x][%d][0x%x] \r\n",check_time, idx,pso.dev_token);
		} 
	} 
	unsigned char ck= check_time;
	 if(check_time!=0x00)
	 {	
	 	ck= ((check_time&0x0f) == (bp_pid_th.dev_token&0x0f));
	 	bp_pid_dbg("searching [%d][%d] [0x%x][0x%x]  \r\n", idx,ck,check_time,bp_pid_th.dev_token );
	 	check_time= ck  ?0:check_time;
	 }
	bp_pid_dbg("pso_get_val feed=(%.2f,%.2f) tgt=(%.2f,%.2f) \r\n",bp_pid_th.t_feed,bp_pid_th.h_feed,bp_pid_th.t_target,bp_pid_th.h_target);
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
			  	bp_pid_dbg("stay update %d check_idx %d new_mae=%f\r\n",pso.swarm_idx, check_idx ,new_mae);
				pso_check(new_mae);
				return new_mae;  
			}
			else{
				bp_pid_dbg("finish update %d check_idx %d new_mae=%f\r\n",pso.swarm_idx, check_idx ,new_mae);				
				pso.step = c_pso_step_check; 
			} 
		break;
		case c_pso_step_check : 
				pso.step = c_pso_step_wait;	
				//calculate_svd(new_mae); 
				//bp_pid_dbg("p00=%.4f,p01=%.4f,p02=%.4f,p20=%.4f,p21=%.4f,p22=%.4f,p40=%.4f,p41=%.4f,p42=%.4f;\r\n",svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2], svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2], svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2]);
			    pso_check(new_mae);   
				bp_pid_dbg("particle update[%.2f]: fit=%.2f,%.2f \r\n", new_mae, pso.swarm[pso.swarm_idx].best_mae,pso.global_bestval);
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
    static unsigned int tmr = 0 ,geer_spk_tmr = 0;
	static unsigned int mode = 0, sec = 0;//, sn = 0;
	static unsigned int on_tmr=0;
	unsigned int	 idx=0;
	float				out=0;
	double				dx=0;
    static pid_run_output_st  output;
	extern void get_devs_type_info(dev_type_t *devs_type_list);
	get_devs_type_info(devs_type_list);
	//ESP_LOGI(TAG, "pid_run_output_st  dev_type=%d",dev_type);
    if(mode != bp_pid_th.mode)	
    {    
		mode = bp_pid_th.mode;    
        if(mode == 0)
        {
			for(int i=0;i<NUM_ENVDEV;i++)
			{
				bp_pid_th.pid_o[i] = 0; 
			}
	    }
        else
        {
            tmr = tick_get() - bp_pid_th.tmr;
			bp_pid_tick_tmr=0;
			sec = tick_get() - 1000; 
		    geer_spk_tmr= tick_get() - bp_pid_th.tmr;
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
			bp_pid_train(); 
			on_tmr=get_eco_update();  
 		 #if c_bp_pid_dbg_en == 1
			bp_pid_wave("t_feed=%.2f,h_feed=%.2f,vpd=%.2f\r\n",bp_pid_th.t_feed,bp_pid_th.h_feed*100,bp_pid_th.v_feed);	
			//bp_pid_dbg("p00=%.4f,p01=%.4f,p02=%.4f,p20=%.4f,p21=%.4f,p22=%.4f,p40=%.4f,p41=%.4f,p42=%.4f;\r\n",svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2], svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2], svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2]);
			// bp_pid_dbg("sigmoid_dat=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f) \r\n",ni_dat[0],net_ih_dat[0],ih_sigmoid_dat[0],hh_sigmoid_dat[0],net_oh_dat[0],bp_pid_th.t_kp,nh_delta1[0],nh_delta[0],no_delta[0]);
 		#endif	
			if(pso.step==c_pso_step_wait)
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
				bp_pid_dbg("[%d][0x%x](%.0ftu %.0fhu %.0fv%c)tu(%.0f,%.3f)hu(%.0f,%.3f)v%c(%.0f,%.3f)kT(%.3f,%.3f,%.3f)mae(%.3f,%.3f,%.3f,%.3f) \r\n",pso.swarm_idx,bp_pid_th.dev_token
				,pso.swarm[pso.swarm_idx].position[DEV_TU], pso.swarm[pso.swarm_idx].position[DEV_HU],tmp_v_val,tmpstr   
				,bp_pid_th.pid_o[DEV_TU],bp_pid_th.du_gain[DEV_TU],bp_pid_th.pid_o[DEV_HU],bp_pid_th.du_gain[DEV_HU],tmpstr,tmp_pid_o,tmp_du_gain
				,bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KP],bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KI],bp_pid_th.ho_sigmoid_out[NUM_ENV_KPID*ENV_V+ENV_KD]
				,current_mae,pso.mae_buf[0][ENV_T],pso.mae_buf[0][ENV_H],pso.mae_buf[0][ENV_V]);
			}  
		}   
		if(tick_cmp(geer_spk_tmr, bp_pid_th.tmr*2) == c_ret_ok)
		{	
			//bp_pid_th.dev_token= (pso.mae_buf[1][idx>>1]>=hvac_margin[idx>>1])? (1<<idx) :bp_pid_th.dev_token;
			bp_pid_th.dev_token= 0; 
			for(uint8_t port=1; port < PORT_CNT; port++ )
			{   
				output.speed[port] =find_gear_level(devs_type_list[port].real_type, input_dev_type[port],on_tmr ); 
			    bp_pid_dbg("  output.speed(%d)\r\n",output.speed[port]); 
			}
			geer_spk_tmr += (bp_pid_th.tmr*2); 
		} 	
		 
	}
	return output;
}  
 

// extern s16 ml_get_cur_temp();
// extern s16 ml_get_cur_humid();
// extern s16 ml_get_outside_temp();
// extern s16 ml_get_outside_humid();
// extern s16 ml_get_outside_vpd();
// extern s16 ml_get_cur_vpd();  
  
pid_run_output_st pid_run_rule(pid_run_input_st* input)
{
	extern pid_run_output_st nn_ppo_infer();	
    short	 dev_type= 0;
	static int init_flag = 0; 
    static pid_run_output_st  output ;
	//struct st_bp_pid_th_arg	pid_arg;
	static unsigned int 	en_bit = 0; 
    bp_pid_tick_tmr += 100;

	if(init_flag == 0)
	{
		init_flag = 1;
		bp_pid_th_init();
		tick_reset();
		bp_pid_th.version = 2503061528;  
		pso.step=c_pso_step_init; 
		#if c_bp_pid_dbg_en == 1
		dbg_init();
		xTaskCreate(dbg_task_loop, 	"dbg_task", 2048*2, 	NULL, TASK_PRIO_MAIN, 	NULL);
		bp_pid_dbg("bp_pid init done! version:%u\r\n", bp_pid_th.version);  
		#endif
	}
    //ESP_LOGI(TAG," pso.step %d ", pso.step);
    for( uint8_t env=0; env < ENV_CNT; env++  )
	{				
        if( input->env_en_bit &(1<<env) ){
            ESP_LOGD("ai pid","env[%d] cur_value[%f] min_value[%f] max_value[%f] target[%f]",
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
		    case loadType_fan:	        dev_type |= (1 << loadType_fan);       bp_pid_th.port_setgear[1][DEV_VU] =bp_pid_th.port_setgear[1][DEV_VD] = input->max[port];
	   }		  	
    } 
	if(dev_type != bp_pid_th.dev_type)
	{
		bp_pid_th.dev_type = dev_type;
		bp_pid_dbg("dev_type=0x%x\r\n", dev_type);
	} 
   	bp_pid_th.mode = input->ml_run_sta;    
	//ESP_LOGI(TAG,"input->env_en_bit %d ",(int) input->env_en_bit); 
	//ESP_LOGI(TAG,"t_feed %f ", bp_pid_th.t_feed); 
	pid_run_output_st output1=	bp_pid_th_proc( dev_type,input->dev_type);  
	for(int i=1;i<PORT_CNT;i++)
		output.speed[i]=  output1.speed[i];  
    return output;
} 
