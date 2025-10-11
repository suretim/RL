#ifndef PSO_H
#define PSO_H

#define c_pso_step_init				0
#define c_pso_step_wait				1
#define c_pso_step_update			2
#define c_pso_step_check			3

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
	unsigned int swarm_idx , step;
    //unsigned int global_idx;
	uint8 dev_token;	
	double   v_wight ;	// w:惯性权重
	float  global_bestval, global_position[DIM]; 
};

extern void pso_check(double new_mae); 
extern double update_particles (void);
extern void pso_init(void) ;
#endif