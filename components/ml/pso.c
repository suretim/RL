
#include "esp_log.h"
#include "string.h"
#include "ml_pid.h"
#include "math.h"  
#include "pso.h"  

#include <time.h>
#include <sys/time.h>


#define MAE_BUF_MAX_SAMPLES 40
struct pso_optimizer pso = {0};
				//       t     dt   h     dh
const float pso_pos_max_tab[DIM]  = {30000,30000,20000,20000,10000,10000};
const float pso_pos_min_tab[DIM]  = {10000,10000,8000,8000,4000,4000};

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

unsigned int chex_swarm(double new_mae) 
{
	float r1=(float) rand() / RAND_MAX;	
	float r2=0.5+  fabs(new_mae); 
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
 
void pso_swarm_update(double new_mae)
{
	unsigned int pre_idx=chex_swarm(new_mae);
	float cur_pos=0;
	float fine_velocity=0;
	if(pre_idx<NUM_PARTICLES)
	{
		
		bp_pid_dbg("chex_swarm 0x%x,pso 0x%x,bp 0x%x\r\n",pso.dev_token&bp_pid_th.dev_token, pso.dev_token,bp_pid_th.dev_token);
		for(int d = 0; d <DIM; d++)
		{
			uint8 x=pso.dev_token&bp_pid_th.dev_token&(1<<d);
			cur_pos= pso.swarm[pso.swarm_idx].position[d];
			if(x!=0)
			{
				//cur_pos= pso.swarm[pso.swarm_idx].best_pos[d]+(pso.global.pos[d])-pso.swarm[pso.swarm_idx].position[d]-pso.swarm[pso.swarm_idx].position[d];
				cur_pos -=  pso.global.pos[d] ;				 
				fine_velocity =pid_map(cur_pos, -pso.global.pos[d] ,pso.global.pos[d] ,c_pid_ptch_min,c_pid_ptch_max);							 
				//pso.swarm[pso.swarm_idx].position[d]=pid_map(pso.swarm[pso.swarm_idx].position[d]+fine_velocity,  pso_pos_min_tab[d], pso_pos_max_tab[d],pso_pos_min_tab[d], pso_pos_max_tab[d]);
				pso.swarm[pso.swarm_idx].position[d]=pso.swarm[pso.swarm_idx].position[d]+fine_velocity;
				bp_pid_dbg("pso_swarm_update x=%x change_dim=%d swan_idx=%d (%fpos, %.1fvel \r\n",x,d,pso.swarm_idx,pso.swarm[pso.swarm_idx].position[d], fine_velocity );
			}
			//else
			//{ 
			//	pso.swarm[pso.swarm_idx].position[d]=pso.swarm[pre_idx].position[d];
			//}
			
		}  	
		pso.dev_token=0;
	}	
}

void pso_path_search(double new_mae)  //^(?!.*global_fit).+(\n|$)  
{ 
    uint8 i=0,d=0;
	// 
	bp_pid_dbg("pso_path_search %.3f,%.3f,%.3f \r\n",new_mae,pso.swarm[pso.swarm_idx].best_mae  , pso.global_bestval);
		
	if(new_mae < pso.swarm[pso.swarm_idx].best_mae)
	{  
		bp_pid_dbg("swarm_tune=(%.3f t,%.3f h,%.3f best_mae \r\n", pso.mae_buf[0][ENV_T], pso.mae_buf[0][ENV_H],pso.swarm[pso.swarm_idx].best_mae );
		pso.swarm[pso.swarm_idx].best_mae = new_mae + rand() / (float) RAND_MAX * 0.1;
		memcpy(pso.swarm[pso.swarm_idx].best_pos, pso.swarm[pso.swarm_idx].position, sizeof(float) * DIM);     
	}
	if(new_mae <  pso.global_bestval)
	{
			bp_pid_dbg("global_tune=(%.3f t,%.3f h,%.3f g_best_mae \r\n", pso.mae_buf[0][ENV_T], pso.mae_buf[0][ENV_H],pso.global_bestval );
			pso.global_bestval = new_mae + rand() / (float) RAND_MAX * 0.1;
			//pso.global[0].idx = pso.swarm_idx;
			// for(d = NUM_GLOBAL - 1; d > 0; d--){
			// 		pso.global[d] = pso.global[d - 1];							
			// }
					
			if(pso.global.idx== 0)
			{
				pso.global.idx = 0;
				memcpy(pso.global.pos, pso.swarm[pso.swarm_idx].position, sizeof(float) * DIM);
			}
			pso.global.idx ++;
			for(i = 0; i <DIM; i++)
			{ 
				//pso.global_position[i] =0.0f;										 				
				pso.global.pos[i] *= NUM_GLOBAL;
				pso.global.pos[i] += (pso.swarm[pso.swarm_idx].position[i]);				
				pso.global.pos[i]/=(NUM_GLOBAL+1);
			} 	

	}	
	 
	
	 
	return;
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
		//for(d = 0; d < DIM; d++){
			pso.global.pos[d] = pso.swarm[0].position[d];	
		//}
	}
}

unsigned int pso_mae_val(double *mae)
{
    unsigned int d;
    static unsigned char du_status[DIM] = {0};
    static unsigned int buf_idx = 0;            // current sample index (0..MAE_BUF_MAX_SAMPLES-1)
    static unsigned int sample_count = 0;       // how many samples accumulated (capped to MAE_BUF_MAX_SAMPLES)
    double err = 0.0;

    if (mae == NULL) return c_ret_nk;
    *mae = 0.0;

    for (d = 0; d < NUM_ENV_TYPE; d++)
    {
        // compute per-d error (preserve original formula but ensure types correct)
        double e = bp_pid_th.e[0][d];
        double du = (double)bp_pid_th.du_gain[d];
        err = e + 0.01 * e * e + 0.01 * du;   // consider clamping if needed

        // maintain running average in pso.mae_buf[0][d]
        // we'll store moving average using sample_count up to MAE_BUF_MAX_SAMPLES
        if (sample_count == 0) {
            pso.mae_buf[0][d] = err;
            pso.mae_buf[1][d] = fabs(err); // maybe the "min observed" metric
        } else {
            // previous_average * sample_count + new_value, then divide by (sample_count+1)
            double prev = pso.mae_buf[0][d];
            double new_avg = (prev * (double)sample_count + err) / (double)(sample_count + 1);
            pso.mae_buf[0][d] = new_avg;
            // pso.mae_buf[1][d] keep minimum absolute average seen (original intent seemed min)
            double abs_avg = fabs(new_avg);
            if (pso.mae_buf[1][d] == 0.0 || pso.mae_buf[1][d] > abs_avg) {
                pso.mae_buf[1][d] = abs_avg;
            }
        }

        *mae += fabs(pso.mae_buf[0][d]);
    }

    // update indices safely
    if (sample_count < MAE_BUF_MAX_SAMPLES) sample_count++;
    buf_idx = (buf_idx + 1) % MAE_BUF_MAX_SAMPLES;

    // sanity check
    if (*mae > 50.0) {
        bp_pid_dbg("mae too large (%.3f) — resetting.\r\n", *mae);
        return c_ret_nk;
    }

    return c_ret_ok;
}

 
unsigned char pso_capture_start_token(void)
{
    //static unsigned char check_time = 0;
    int cur_time = tick_get();
    static uint8_t token[DIM] = {0};
    static int pre_tmr[2][DIM] = {0};         // initialize to 0
    static float target_diff[3][DIM] = {{0}}; // [0] current, [1] previous, [2] spare (if needed)
    unsigned int idx;
    //float tmp_hvac = 0.0f;
	//unsigned char ck = pso.dev_token;
    pso.v_wight = 0.1f;

    // compute current target diff
    target_diff[0][DEV_TU] = bp_pid_th.s[ENV_T] - bp_pid_th.f[0][ENV_T];
    target_diff[0][DEV_TD] = -target_diff[0][DEV_TU];
    target_diff[0][DEV_HU] = bp_pid_th.s[ENV_H] - bp_pid_th.f[0][ENV_H];
    target_diff[0][DEV_HD] = -target_diff[0][DEV_HU];
    target_diff[0][DEV_VU] = bp_pid_th.s[ENV_V] - bp_pid_th.f[0][ENV_V];
    target_diff[0][DEV_VD] = -target_diff[0][DEV_VU];

    for (idx = 0; idx < DIM; idx++)
    {
        // compute change (delta) safely using previous sample (target_diff[1])
        //tmp_hvac = target_diff[0][idx] - target_diff[1][idx];

        // when device just activated (gear timer == 0) capture start time and token
        if (target_diff[0][idx] > 0.0f && bp_pid_th.u_gear_tmr[idx] == 0)
        {
            target_diff[1][idx] = target_diff[0][idx];
            pre_tmr[0][idx] = cur_time;
            token[idx] = (bp_pid_th.dev_token & (1 << idx)) ? 1 : 0;
        }

        if (token[idx] == 1 && bp_pid_th.du_gain[idx] < 0.00f && bp_pid_th.pid_o[idx] > 0.00f)
        {
            //tmp_hvac = fabsf(tmp_hvac) < 1.0f ? 1.0f : tmp_hvac;

            // careful: original used idx>>1 — preserve if intentional (grouping), otherwise use idx
            //uint8_t hvac_index = (uint8_t)(idx >> 1);
            //uint8_t tmp_token = (1 << idx) ;//(pso.mae_buf[1][hvac_index] > hvac_margin[hvac_index]) ? (1 << idx) : 0;
            pso.dev_token |= (1 << idx);
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
            // velocity/position update (kept original math)
            pso.swarm[pso.swarm_idx].position[idx] -= pso.swarm[pso.swarm_idx].velocity[idx];
            pso.swarm[pso.swarm_idx].velocity[idx] *= pso.swarm[pso.swarm_idx].v_idx[idx];
            pso.swarm[pso.swarm_idx].velocity[idx] /= (float)(pso.swarm[pso.swarm_idx].v_idx[idx] + 1);
            pso.swarm[pso.swarm_idx].v_idx[idx] = (pso.swarm[pso.swarm_idx].v_idx[idx] > 40) ? 40 : pso.swarm[pso.swarm_idx].v_idx[idx] + 1;

            // clamp pitch using pid_map; ensure values passed are correct
            pso.swarm[pso.swarm_idx].velocity[idx] = pid_map(pso.swarm[pso.swarm_idx].velocity[idx],
                                                            c_pid_ptch_min, c_pid_ptch_max,
                                                            c_pid_ptch_min, c_pid_ptch_max);

            pso.swarm[pso.swarm_idx].position[idx] += pso.swarm[pso.swarm_idx].velocity[idx];

            token[idx] = 0;
            pre_tmr[1][idx] = cur_time;
            //check_time |= (1 << idx);

            bp_pid_dbg("check_idx set: check_time=0x%x idx=%u dev_token=0x%x\r\n", pso.dev_token, idx, pso.dev_token);
        }
    }

    // evaluate check_time validity
    
    //if (pso.dev_token != ck)
    //{
        // consider only lower nibble? mimic original logic but clearer
        //unsigned char mask = (unsigned char)(bp_pid_th.dev_token & 0x3f);
        //ck = (((pso.dev_token & 0x3f) == mask) ? 0 : pso.dev_token);
		 
    //    bp_pid_dbg("searching idx=%u ck=%u pso.dev_token=0x%x dev_token=0x%x\r\n", idx, ck, pso.dev_token, bp_pid_th.dev_token);
        //pso.dev_token = ck ? pso.dev_token : 0;
		
    //}

    bp_pid_dbg("pso_capture_start_token feed=(%.2f,%.2f) tgt=(%.2f,%.2f)\r\n", bp_pid_th.t_feed, bp_pid_th.h_feed, bp_pid_th.t_target, bp_pid_th.h_target);

    // shift current to previous for next call (so target_diff[1] becomes previous)
    for (idx = 0; idx < DIM; idx++) {
        target_diff[1][idx] = target_diff[0][idx];
    }
	//if(ck==0)
	    return (unsigned char)pso.dev_token;
	//else
	//	return ck;
}



double particles_state_machine (void)//float t_target, float t_feed, float h_target, float h_feed, float v_target, float v_feed) //^bp_pid_run.*$\r?\n
{
	static double  new_mae= 0;
	float fine_velocity ; 
	unsigned char	check_pitch_idx=0;
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
		//particles_state_machine:	  
	    	//pso.step = __LINE__; return new_mae; case __LINE__: 
			check_pitch_idx=pso_capture_start_token(); 
			if(check_pitch_idx==0)
			{ 
				pso.step = c_pso_step_wait;
				pso_path_search(new_mae);  
			  	bp_pid_dbg("c_pso_step_update pso.swarm_idx %d check_pitch_idx %d new_mae=%f\r\n",pso.swarm_idx, check_pitch_idx ,new_mae);
				return new_mae;  
			}
			else{
				bp_pid_dbg("c_pso_step_update pso.swarm_idx %d check_pitch_idx %d new_mae=%f\r\n",pso.swarm_idx, check_pitch_idx ,new_mae);				
				pso.step = c_pso_step_check; 
			} 
		break;
		case c_pso_step_check : 
				pso.step = c_pso_step_wait;	
				pso_swarm_update(  new_mae);
				//pso.dev_token=0;
				//calculate_svd(new_mae); 
				//bp_pid_dbg("p00=%.4f,p01=%.4f,p02=%.4f,p20=%.4f,p21=%.4f,p22=%.4f,p40=%.4f,p41=%.4f,p42=%.4f;\r\n",svd.pitchs[0][0],svd.pitchs[0][1],svd.pitchs[0][2], svd.pitchs[2][0],svd.pitchs[2][1],svd.pitchs[2][2], svd.pitchs[4][0],svd.pitchs[4][1],svd.pitchs[4][2]);
			      
				bp_pid_dbg("particle update : pso.dev_token =%d \r\n", pso.dev_token );
				//double chk_mae=fabs(pso.mae_buf[pso.swarm_idx][0]) + fabs(pso.mae_buf[pso.swarm_idx][1]);
		break; 
		default:   break;
	}	
	return  new_mae;
}  
 