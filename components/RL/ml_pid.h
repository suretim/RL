#ifndef _ML_PID_H_
#define _ML_PID_H_

#include "types.h"
#include "define.h"
#include "ai.h"
#include "ai_out.h"

#define NUM_PARTICLES 				24   //9		//20		//20 粒子数量
#define NUM_GLOBAL					8    //5
#define MAX_ITER 					2   //8

#define PID_RULE_EN
// #define FULL_NET 1
// #define NHH_NET 1
// #define BUF_UPDATE_NET 		0

#define NUM_UPDOWN	    2  //up,down 
#define ENV_UP			0 
#define ENV_DOWN	 	1

#define NUM_DEV_TYPE    2
#define DEV_GAIN		0
#define DEV_GEAR		1

#define NUM_PTH_TYPE    3   //vel,pth_inside,pitch_period
#define DEV_PTH_PW   	0
#define DEV_PTH_IN		1
#define DEV_PTH_ON		2 
#define NUM_PTH_VP      2   //vel,pth_inside 

#define NUM_ENVDEV		    6  //dev_tu,dev_td,dev_hu,dev_hd,dev_vu,dev_vd
#define DEV_TU  			0 
#define DEV_TD  			1 
#define DEV_HU  			2 
#define DEV_HD  			3 
#define DEV_VU  			4 
#define DEV_VD  			5

#define NUM_ENV_TYPE		3  //t,h,v
#define ENV_T				0 
#define ENV_H				1 
#define ENV_V				2  
#define NUM_ENV_TH		    2 //t,h  

#define NUM_DEV_FEA         (NUM_DEV_TYPE*NUM_ENVDEV)
#define NUM_KEY_FEA         (NUM_PTH_VP*NUM_ENV_TH)
#define NUM_SPK			    (1<<(NUM_PTH_VP+NUM_ENV_TH)) 

#define NUM_ENV_KPID	    3  //kp,ki,kd 
#define ENV_KP				0 
#define ENV_KI				1 
#define ENV_KD				2  

#define NUM_ENV_TYPEIN		3  //set,fed,delta
#define PAR_SET_IN			0 
#define PAR_FED_IN			1 
#define PAR_DEL_IN			2  



#define NR_END 1

#define NUM_ENV_IN		(NUM_ENV_TYPE*NUM_ENV_TYPEIN )   //tuset,tufed,tudel,huset,hufed,hudel,vset,vfed,vdel
//#define NUM_TH_UPDOWN_OUT		(NUM_ENVDEV_OUT*NUM_ENVGAN_TYPEOUT )   //tu,td,hu,hd

#define NUM_KEYN    	(NUM_KEY_FEA)   
#define L_GAIN          (3)  //(NUM_PTH_TYPE)    

#define PAR_GAIN_TU_IN		0   
#define PAR_GAIN_TD_IN		1   
#define PAR_GAIN_HU_IN		2   
#define PAR_GAIN_HD_IN		3   
#define PAR_GAIN_VU_IN		4   
#define PAR_GAIN_VD_IN		5 

#define PAR_GEAR_TU_IN		6   
#define PAR_GEAR_TD_IN		7   
#define PAR_GEAR_HU_IN		8   
#define PAR_GEAR_HD_IN		9   
#define PAR_GEAR_VU_IN		10   
#define PAR_GEAR_VD_IN		11    

#define PAR_PITCH_TU_IN		12   
#define PAR_PITCH_TD_IN		13   
#define PAR_PITCH_HU_IN		14   
#define PAR_PITCH_HD_IN		15   
#define PAR_PITCH_VU_IN		16   
#define PAR_PITCH_VD_IN		17    

#define PAR_PITCH_TU_ON		18   
#define PAR_PITCH_TD_ON		19   
#define PAR_PITCH_HU_ON		20   
#define PAR_PITCH_HD_ON		21   
#define PAR_PITCH_VU_ON		22   
#define PAR_PITCH_VD_ON		23    


// #define NUM_ENVGAIN_TYPEIN	3  //gain_in,gear_in,time_in
// #define PAR_GAIN_IN  		0  //gain_in  
// #define PAR_GEAR_IN	    	1  //gear_in  
// #define PAR_TENT_IN         2  //tent_in

//#define NUM_GAIN_TYPEIN	NUM_LATN_TYPEIN   
//#define NUM_GAIN_TYPEIN	NUM_ENVGAIN_TYPEIN   
//#if (NUM_GAIN_TYPEIN>NUM_ENVGAIN_TYPEIN)   
//#define PAR_TIME_IN NUM_LATN_TYPEIN
//#else
//    #define PAR_TIME_IN PAR_TENT_IN  //tent_in
//#endif

// #define NUM_PAR_TYPEOUT		6  //gain_tu,gain_td,gain_hu,gain_hd,gain_vu,gain_vd 
// #define PAR_GAIN_TU_OUT		0  
// #define PAR_GAIN_TD_OUT		1  
// #define PAR_GAIN_HU_OUT		2  
// #define PAR_GAIN_HD_OUT		3  
// #define PAR_GAIN_HU_OUT		4  
// #define PAR_GAIN_HD_OUT		5  
// #define PAR_TIME_OUT        4 
#define NUM_DEV_KPID_OUT		(NUM_ENV_TYPE*NUM_ENV_KPID )   //tkp,tki,tkd,hkp,hki,hkd,vkp,vki,vkd

#define c_ni_nodes         	(NUM_ENV_TYPE*NUM_ENV_TYPEIN+NUM_DEV_FEA+1)
//#define c_nh_nodes0        	(NUM_ENV_TYPE*NUM_ENV_TYPEIN+NUM_LATN_TYPEIN+3)
#define c_nh_nodes0        12

//#define c_nh_nodes1        	(NUM_DEV_KPID_OUT+NUM_ENVDEV+1)
#define c_nh_nodes1        12
#define c_no_nodes      	(NUM_DEV_KPID_OUT+NUM_ENVDEV)
#define DIM 				NUM_ENVDEV        
  

#define c_ret_ok            	0
#define c_ret_nk            	1


#define c_pid_temp_min			1
#define c_pid_temp_max			40
#define c_pid_humi_min			10
#define c_pid_humi_max			100
#define c_pid_vpd_min			0
#define c_pid_vpd_max			4
#define c_pid_ptch_min			-400
#define c_pid_ptch_max			400
// #define c_pid_gain_min			 2000
// #define c_pid_gain_max			 20000



#define FREE_ARG char*
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
static double dmaxarg1,dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ? (dmaxarg1) : (dmaxarg2))
static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ? (iminarg1) : (iminarg2))
 

#define M NUM_SPK     
#define N NUM_KEYN         // 输入的矩阵规模
#if (M>N)      // 取M、N中的较大数
    #define MN (M+NR_END)
#else
    #define MN (N+NR_END)
#endif

#define NM MN
#define NU (MN-1)

#define	c_bp_pid_dbg_en			1
#define bp_dbg(format, ...)			printf(" "format, ##__VA_ARGS__)
#define bp_pid_dbg(format, ...)			printf("bp_pid_d "format, ##__VA_ARGS__)
#define bp_pid_wave(format, ...)		printf("bp_pid_w "format, ##__VA_ARGS__)
#define c_diff_wc1c2  1
typedef struct{
    uint8_t ml_run_sta;
    uint8_t dev_type[PORT_CNT];
    uint8_t is_switch[PORT_CNT];
    int16_t max[PORT_CNT];
    int16_t min[PORT_CNT];

    int16_t env_value_cur[ENV_CNT];
    int16_t env_target[ENV_CNT];
    int16_t env_min[ENV_CNT];
    int16_t env_max[ENV_CNT];

    uint32_t env_en_bit;    //控制环境使能位
}pid_run_input_st;

typedef struct{
    uint8_t speed[PORT_CNT];
}pid_run_output_st;

extern pid_run_output_st pid_run_rule(pid_run_input_st* input);
extern struct st_bp_pid_th    bp_pid_th ;
 
 
// struct st_pos_val_arg
// {
// 	float	t_target;
// 	float	t_feed;
// 	float	h_target;
// 	float	h_feed; 
// };

// struct st_bp_pid_th_arg
// {
// 	float t_target;
// 	float t_feed;
// 	float t_outside;

// 	float h_target;
// 	float h_feed;
// 	float h_outside;

// 	float v_target;
// 	float v_feed;
// 	float v_outside; 

	
// 	unsigned short t_max_gear;
// 	unsigned short t_min_gear;
// 	unsigned short h_max_gear;
// 	unsigned short h_min_gear;
// 	unsigned short v_max_gear;
// 	unsigned short v_min_gear;
// 	unsigned short dt_max_gear;
// 	unsigned short dt_min_gear;
// 	unsigned short dh_max_gear;
// 	unsigned short dh_min_gear;
// 	unsigned short dv_max_gear;
// 	unsigned short dv_min_gear;
// }; 

struct pso_particle
{
	//float 				pos[DIM];    				// 当前增益值
   // float 				vel[DIM];    				// 速度向量
    //float 			best_val;    				// 速度向量
   
    float 			position[DIM];    		// 当前增益值
    float 			velocity[DIM];    		// 速度向量
    float 			best_pos[DIM];    		// 个体历史最优增益
    float 			best_mae;//[NUM_ENV_TYPE];     		// 个体最优适应度
	unsigned int    v_idx[DIM]; 
	float				val;
   
};
struct pso_global
{
    float 			pos[DIM];    		
    uint8           swarm_idx;     	
	 float 			best_val;     		// 个体最优适应度	
	 float best_pos[DIM];
};
struct pso_optimizer
{
	double			mae_buf[2][NUM_ENV_TYPE];

    //struct pso_particle	particle[NUM_PARTICLES];	// 分群粒子
    struct pso_particle 	swarm[NUM_PARTICLES];	// 群
	struct pso_global 		global[NUM_GLOBAL];
	 
	//float			h_buf[NUM_PARTICLES];
	// float			v_buf[60];
	unsigned int	 buf_cnt, test_req;
	unsigned int swarm_idx,global_idx , step;
	uint8 dev_token;	
	double   v_wight ;	// w:惯性权重
	float  global_bestval, global_position[DIM]; 
};

 

struct svd_optimizer
{
    double  w_vec[MN];  //[NUM_LAT+1]
    double  v_mat[NM][MN]; //[NUM_SPK+1][NUM_SPK+1];
    double  u_mat[MN][NM];  //[NUM_LAT+1][NUM_KEYN+1];
    double  uut_mat[MN][NM];  //[NUM_LAT+1][NUM_KEYN+1];
    double  arisk[MN][NM];  //k[NUM_SPK+1][NUM_KEYN+1]
    double  uw[NM][NM];	 //[L_GAIN+1]
    double  wv[NM][NM];	 //[L_GAIN+1]
    double  latent_mat[NUM_SPK+1][NM];
    double  latent_vec[NM];
     
    unsigned int   v_idx ; //[NUM_SPK+1]
    float pitchs[NUM_ENVDEV][NUM_PTH_TYPE];
    float avg_pitchs[NUM_ENVDEV][NUM_PTH_TYPE];
    unsigned int avg_cnt[NUM_ENVDEV];
    // unsigned int  flag ;
};
struct st_bp_pid_th
{

    double       	wt_nh_ni[c_nh_nodes0][c_ni_nodes]; 
    double      	wt_no_nh[c_no_nodes][c_nh_nodes1]; 
    double      	wt_nh_nh[c_nh_nodes1][c_nh_nodes0]; 
    unsigned int   	dropout_hh[c_nh_nodes1][c_nh_nodes0];
    unsigned int   	dropout_oh[c_no_nodes][c_nh_nodes1];  
	double 			ho_sigmoid_out[c_no_nodes];
	float           update_rate;
	double			e[3][NUM_ENV_TYPE],s[NUM_ENV_TYPE],f[2][NUM_ENV_TYPE]; 
	double 			pid_o[NUM_ENVDEV] ; 
    unsigned int  tmr; 
    double du_gain[NUM_ENVDEV];  //ugain delta
    unsigned char dev_token;
    //double tu ;  //tgain delta
    u_int8_t		u_gear_tmr[NUM_ENVDEV]; 
	//float 			u_gain_tmr[NUM_ENVDEV]; 
	float			t_target, t_feed, t_outside,h_target, h_feed,h_outside, v_target, v_feed,v_outside; 
    float l_feed,c_feed;
	unsigned char	mode; 
	unsigned int	dev_type,   version; 
	unsigned short port_setgear[2][NUM_ENVDEV];
    
    //double ienerge[NUM_ENVDEV];

};

#endif
