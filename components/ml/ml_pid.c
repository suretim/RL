#include "esp_log.h"
#include "string.h"
#include "ml_pid.h"
#include "math.h"
#include "Sensor.h" 
#include "nvs_flash.h"
#include "beep.h"
  
#include "ni_debug.h"
#include "pso.h"
#define c_gain_ctrl_stop			0
#define c_gain_ctrl_req				1
#define c_gain_ctrl_auto			2

#include <time.h>
#include <sys/time.h>



#define dbg_serial_usb_serial_jtag		1
#define dbg_serial_uart0				2
#define dbg_serial_sel        			dbg_serial_uart0
//#define dbg_serial_sel        			dbg_serial_usb_serial_jtag

static const char *TAG = "ML PID";
		 
unsigned int gain_adj_ctrl = 2;

//dev_type_t devs_type_list[PORT_CNT];
pid_run_output_st lstm_pid_out_speed;
extern const float pso_pos_max_tab[DIM]  ;
extern const float pso_pos_min_tab[DIM] ;
extern curLoad_t curLoad[PORT_CNT] ;
//float pitch_limit[2][NUM_PTH_TYPE]={{20,80,300},{40,160,600}};
//float tmp_pitch[NUM_UPDOWN][NUM_PTH_TYPE]= {{0.99,0.99,0.99 },{1.01,1.01,1.01 }};
unsigned int tmp_pdx[DIM]={0};
struct st_bp_pid_th bp_pid_th = {0};
extern struct pso_optimizer pso ;
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

//const float fit_up_bound=8.0;
//const float fit_low_bound=3.0;
double current_mae=0.0; 


static double pid_exp(double x)
{
	if(x > 708) x = 708;
	else if(x < -708) x = -708;
	return exp(x);	
}

//void svd_init(void);
//unsigned int calculate_svd(double new_mae);
//void svd_port_latent( double *profile ,int spkidx);
float pid_map(float x, float in_min, float in_max, float out_min, float out_max)
{
    if(x < in_min) return out_min;
	if(x > in_max) return out_max;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min; 
}  
unsigned int tick_get(void)
{  
	return bp_pid_tick_tmr;
}
unsigned int tick_reset(void)
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
		//dx[o]=net[o]>0?1:0;
		dx[o]=1.0 / (1.0+ pid_exp(-net[o]));
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
static unsigned int tick_cmp(int tmr, int tmo)
{  
	return (tick_get()  >= (tmo+tmr)) ? c_ret_ok : c_ret_nk; 
} 



void bp_pid_th_init(void) 
{
    unsigned int i=0, h=0, o=0; 
	memset(&bp_pid_th, 0, sizeof(struct st_bp_pid_th)); 
	bp_pid_th.tmr =  10000;
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



// Improved/safer version of bp_pid_train
static unsigned int bp_pid_train(void)
{
    double dx = 0;
    unsigned int i = 0, h = 0, o = 0;

    // --- Input mapping (keep original semantics) ---
    bp_pid_th.s[ENV_T]    = pid_map(bp_pid_th.t_target, plant_limit_params.temp_range.first,  plant_limit_params.temp_range.second, 0.0f, 1.0f);
    bp_pid_th.f[0][ENV_T] = pid_map(bp_pid_th.t_feed,   plant_limit_params.temp_range.first,  plant_limit_params.temp_range.second, 0.0f, 1.0f);
    bp_pid_th.s[ENV_H]    = pid_map(bp_pid_th.h_target, plant_limit_params.humid_range.first, plant_limit_params.humid_range.second, 0.0f, 1.0f);
    bp_pid_th.f[0][ENV_H] = pid_map(bp_pid_th.h_feed,   plant_limit_params.humid_range.first, plant_limit_params.humid_range.second, 0.0f, 1.0f);
    bp_pid_th.s[ENV_V]    = pid_map(bp_pid_th.v_target, plant_limit_params.vpd_range.first,   plant_limit_params.vpd_range.second,   0.0f, 1.0f);
    bp_pid_th.f[0][ENV_V] = pid_map(bp_pid_th.v_feed,   plant_limit_params.vpd_range.first,   plant_limit_params.vpd_range.second,   0.0f, 1.0f);

    // zero input vector (faster & clearer)
    for (h = 0; h < c_ni_nodes; ++h) ni_dat[h] = 0.0f;

    // fill ni_dat with set/feed/delta for each env type
    for (h = 0; h < NUM_ENV_TYPE; ++h) {
        bp_pid_th.e[0][h] = bp_pid_th.s[h] - bp_pid_th.f[0][h];
        ni_dat[h * NUM_ENV_TYPEIN + PAR_SET_IN] = bp_pid_th.s[h];
        ni_dat[h * NUM_ENV_TYPEIN + PAR_FED_IN] = bp_pid_th.f[0][h];
        ni_dat[h * NUM_ENV_TYPEIN + PAR_DEL_IN] = bp_pid_th.e[0][h];
    }

    double prev_global_best = pso.global_bestval;
    current_mae = update_particles();

    // fill control-related inputs (clamping to [0,1] via pid_map)
    for (h = 0; h < NUM_ENVDEV; ++h) {
        double noisy_vel = pso.swarm[pso.swarm_idx].velocity[h] + 100.0 * (1.0 - (0.5 * (double)rand() / (double)RAND_MAX));
        ni_dat[NUM_ENV_IN + h] = pid_map((float)noisy_vel, c_pid_ptch_min, c_pid_ptch_max, 0.0f, 1.0f);

        double gear = (double)bp_pid_th.u_gear_tmr[h] / 10.0 + 0.1 * (1.0 - (0.5 * (double)rand() / (double)RAND_MAX));
        ni_dat[NUM_ENV_IN + NUM_ENVDEV + h] = gear;
    }

    // compute discrete-time derivative terms for PID basis
    double i_pid[NUM_ENV_TYPE * NUM_ENV_KPID];
    for (h = 0; h < NUM_ENV_TYPE; ++h) {
        i_pid[h * NUM_ENV_KPID + ENV_KP] = bp_pid_th.e[0][h] - bp_pid_th.e[1][h];  // approximate derivative
        i_pid[h * NUM_ENV_KPID + ENV_KI] = bp_pid_th.e[0][h];                     // integral term (simple)
        i_pid[h * NUM_ENV_KPID + ENV_KD] = bp_pid_th.e[0][h] - 2.0 * bp_pid_th.e[1][h] + bp_pid_th.e[2][h]; // 2nd diff
    }

    // shift history
    for (h = 0; h < NUM_ENV_TYPE; ++h) {
        bp_pid_th.f[1][h] = bp_pid_th.f[0][h];
        bp_pid_th.e[2][h] = bp_pid_th.e[1][h];
        bp_pid_th.e[1][h] = bp_pid_th.e[0][h];
    }

    // forward pass: input -> hidden0
    for (h = 0; h < c_nh_nodes0; ++h) {
        double sum = 0.0;
        for (i = 0; i < c_ni_nodes; ++i) sum += ni_dat[i] * bp_pid_th.wt_nh_ni[h][i];
        net_ih_dat[h] = sum;
    }
    sigmoid_h(ih_sigmoid_out, net_ih_dat, c_nh_nodes0); // or tanh depending on your design

    // hidden0 -> hidden1
    for (h = 0; h < c_nh_nodes1; ++h) {
        double sum = 0.0;
        for (i = 0; i < c_nh_nodes0; ++i) {
            if (!bp_pid_th.dropout_hh[h][i]) sum += ih_sigmoid_out[i] * bp_pid_th.wt_nh_nh[h][i];
            else bp_pid_th.wt_nh_nh[h][i] = 0.0;
        }
        net_hh_dat[h] = sum;
    }
    sigmoid_h(hh_sigmoid_out, net_hh_dat, c_nh_nodes1);

    // hidden1 -> output
    for (h = 0; h < c_no_nodes; ++h) {
        double sum = 0.0;
        for (i = 0; i < c_nh_nodes1; ++i) {
            sum += hh_sigmoid_out[i] * bp_pid_th.wt_no_nh[h][i];
        }
        net_oh_dat[h] = sum;
    }
    sigmoid_o(bp_pid_th.ho_sigmoid_out, net_oh_dat, c_no_nodes); // final activation (choose sigmoid or relu consistently)

    // produce PID du_gain from network outputs
    for (h = 0; h < NUM_ENV_TYPE; ++h) {
        double kp = bp_pid_th.ho_sigmoid_out[h * NUM_ENV_KPID + ENV_KP];
        double ki = bp_pid_th.ho_sigmoid_out[h * NUM_ENV_KPID + ENV_KI];
        double kd = bp_pid_th.ho_sigmoid_out[h * NUM_ENV_KPID + ENV_KD];

        double up = kp * i_pid[h * NUM_ENV_KPID + ENV_KP]
                  + ki * i_pid[h * NUM_ENV_KPID + ENV_KI]
                  + kd * i_pid[h * NUM_ENV_KPID + ENV_KD];
        bp_pid_th.du_gain[h * NUM_UPDOWN + ENV_UP] = up;
        bp_pid_th.du_gain[h * NUM_UPDOWN + ENV_DOWN] = -up;
    }

    bp_pid_dbg(" du_gain= (%.2f,%.2f,%.2f)\r\n", bp_pid_th.du_gain[0], bp_pid_th.du_gain[2], bp_pid_th.du_gain[4]);

    // compute derivatives for backprop (choose correct derivative functions)
    derivative_o(d_hide_o, bp_pid_th.ho_sigmoid_out, c_no_nodes);
    derivative_tanh(d_hide1, hh_sigmoid_out, c_nh_nodes1);
    derivative_tanh(d_hide0, ih_sigmoid_out, c_nh_nodes0);

    bp_pid_th.update_rate = 0.001;

    // desired_out: make sure buffer size and indexing are safe
    static double desired_out_static[/* ensure at least c_no_nodes */ 256];
    double *desired_out = desired_out_static;

    // Ensure we don't read/write past c_no_nodes
    for (o = 0; o < (unsigned int)c_no_nodes; ++o) desired_out[o] = 0.0;

    double Etotal = 0.0;
    for (h = 0; h < NUM_ENV_TYPE; ++h) Etotal +=  bp_pid_th.e[0][h]  ;

    // If PSO improved global best, perform supervised backprop towards pso.global_position
    if (true){//pso.global_bestval < prev_global_best) {
        // Build desired_out: first NUM_DEV_KPID outputs set to Etotal (as original logic wanted)
        for (i = 0; i < (int)c_no_nodes; ++i) {
            if (i < NUM_DEV_KPID_OUT) {
                desired_out[i] = Etotal;
            } else {
                int pos_idx = i - NUM_DEV_KPID_OUT;
                // only copy if PSO has that position entry
                if (pos_idx >= 0 && pos_idx < DIM) {
                    // Normalize PSO positions into activation range if necessary
                    double mapped = pid_map((float)pso.global_position[pos_idx], pso_pos_min_tab[pos_idx], pso_pos_max_tab[pos_idx], 0.0f, 1.0f);
                    desired_out[i] = mapped;
                } else {
                    desired_out[i] = 0.0;
                }
            }
        }

        bp_pid_dbg(" Etotal, desired_out[NUM_DEV_KPID_OUT] =(%.3f,%.6f)\r\n", Etotal, desired_out[NUM_DEV_KPID_OUT]);

        // --- BACKPROP ---
        // Compute output layer delta: no_delta = (target - y_pred) * d_hide_o
        static double no_delta_static[/* c_no_nodes */ 256];
        double *no_delta = no_delta_static;
        for (o = 0; o < (unsigned int)c_no_nodes; ++o) {
            double y_pred = bp_pid_th.ho_sigmoid_out[o];
            double err_h = desired_out[o] - y_pred;
            no_delta[o] = err_h * d_hide_o[o];
            if (float_chk(no_delta[o]) == c_ret_nk) no_delta[o] = 0.0;
        }

        // hidden1 delta: nh_delta1 = (Wt_no_nh^T * no_delta) .* d_hide1
        static double nh_delta1_static[/* c_nh_nodes1 */ 256];
        double *nh_delta1_local = nh_delta1_static;
        for (o = 0; o < (unsigned int)c_nh_nodes1; ++o) {
            double sum = 0.0;
            for (h = 0; h < (unsigned int)c_no_nodes; ++h) sum += no_delta[h] * bp_pid_th.wt_no_nh[h][o];
            nh_delta1_local[o] = sum * d_hide1[o];
            if (float_chk(nh_delta1_local[o]) == c_ret_nk) nh_delta1_local[o] = 0.0;
        }

        // hidden0 delta: nh_delta0 = (Wt_nh_nh^T * nh_delta1) .* d_hide0
        static double nh_delta0_static[/* c_nh_nodes0 */ 256];
        double *nh_delta0_local = nh_delta0_static;
        for (o = 0; o < (unsigned int)c_nh_nodes0; ++o) {
            double sum = 0.0;
            for (h = 0; h < (unsigned int)c_nh_nodes1; ++h) sum += nh_delta1_local[h] * bp_pid_th.wt_nh_nh[h][o];
            nh_delta0_local[o] = sum * d_hide0[o];
            if (float_chk(nh_delta0_local[o]) == c_ret_nk) nh_delta0_local[o] = 0.0;
        }

        // weight updates with clamping
        double lr = bp_pid_th.update_rate;

        // output <- hidden1
        for (unsigned int hh = 0; hh < (unsigned int)c_no_nodes; ++hh) {
            for (unsigned int ii = 0; ii < (unsigned int)c_nh_nodes1; ++ii) {
                double delta_w = lr * no_delta[hh] * hh_sigmoid_out[ii];
                if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
                bp_pid_th.wt_no_nh[hh][ii] += delta_w;

                // optional clipping
                if (bp_pid_th.wt_no_nh[hh][ii] > WMAX) bp_pid_th.wt_no_nh[hh][ii] = WMAX;
                else if (bp_pid_th.wt_no_nh[hh][ii] < -WMAX) bp_pid_th.wt_no_nh[hh][ii] = -WMAX;
            }
        }
        // hidden1 <- hidden0
        for (unsigned int hh = 0; hh < (unsigned int)c_nh_nodes1; ++hh) {
            for (unsigned int ii = 0; ii < (unsigned int)c_nh_nodes0; ++ii) {
                double delta_w = lr * nh_delta1_local[hh] * ih_sigmoid_out[ii];
                if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
                bp_pid_th.wt_nh_nh[hh][ii] += delta_w;
                if (bp_pid_th.wt_nh_nh[hh][ii] > WMAX) bp_pid_th.wt_nh_nh[hh][ii] = WMAX;
                else if (bp_pid_th.wt_nh_nh[hh][ii] < -WMAX) bp_pid_th.wt_nh_nh[hh][ii] = -WMAX;
            }
        }
        // hidden0 <- input
        for (unsigned int hh = 0; hh < (unsigned int)c_nh_nodes0; ++hh) {
            for (unsigned int ii = 0; ii < (unsigned int)c_ni_nodes; ++ii) {
                double delta_w = lr * nh_delta0_local[hh] * ni_dat[ii];
                if (float_chk(delta_w) == c_ret_nk) delta_w = 0.0;
                bp_pid_th.wt_nh_ni[hh][ii] += delta_w;
                if (bp_pid_th.wt_nh_ni[hh][ii] > WMAX) bp_pid_th.wt_nh_ni[hh][ii] = WMAX;
                else if (bp_pid_th.wt_nh_ni[hh][ii] < -WMAX) bp_pid_th.wt_nh_ni[hh][ii] = -WMAX;
            }
        }
    }

    return c_ret_ok;
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
u_int8_t find_gear_level( uint8_t  is_switch ,int16 load_type,unsigned int on_tmr)
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

	if(  is_switch==1 ) {
		 if( idx >=2)  //2
		    out_gear = idx ;
		else
		    out_gear = 0;
	}
	 
	 *pgearout= out_gear ;
	 //bp_pid_dbg("gear_level dev_type=(0x%x,%d),pid_o=%.2f,tmr(%d,%d,%d),out_gear=(%d,%d) ,max_mix=(%d,%d) \r\n", real_type,load_type,p_out,acctmr,on_tmo,bp_pid_th.tmr,  out_gear,idx,  max_gear,min_gear );
	 
	return out_gear ;
}  
 
pid_run_output_st pid_run_rule(pid_run_input_st* input)
{
	//extern pid_run_output_st nn_ppo_infer();	
    short	 dev_type= 0;
	static bool init_flag = false; 
    static pid_run_output_st  output ;
	//struct st_bp_pid_th_arg	pid_arg;
	//static unsigned int 	en_bit = 0; 
    bp_pid_tick_tmr += 100;

	if(init_flag == false)
	{ 
        init_flag=true;
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
    //for( uint8_t env=0; env < ENV_CNT; env++  )
	//{				
    //    if( input->env_en_bit &(1<<env) ){
    //        ESP_LOGD("ai pid","env[%d] cur_value[%f] min_value[%f] max_value[%f] target[%f]",
    //            env,input->env_value_cur[env],input->env_min[env],input->env_max[env], input->env_target[env] );
    //    }
    //}
	//if(en_bit != input->env_en_bit)
	//{
	//	en_bit = input->env_en_bit;
	//	bp_pid_dbg("en_bit=0x%x\r\n", en_bit);
	//} 
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
	//pid_run_output_st output1=	bp_pid_th_proc( dev_type,input );  
 

	static unsigned int tmr = 0 ,geer_spk_tmr = 0;
	static unsigned int mode = 0, sec = 0;//, sn = 0;
	static unsigned int on_tmr=0;
	 
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
				output.speed[port] =find_gear_level(input->is_switch[port], input->dev_type[port],on_tmr ); 			     
			}
			bp_pid_dbg("target(%f,%f) output.speed(%d,%d,%d,%d)\r\n",bp_pid_th.t_target,bp_pid_th.h_target,output.speed[1],output.speed[2],output.speed[3],output.speed[4]);
			geer_spk_tmr += (bp_pid_th.tmr*2); 
		} 	
		 
	}
	 
 
    return output;
} 
