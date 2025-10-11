

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
 
 static double a[MN][MN];
 static double v[MN][MN];
 static double w[MN];
  

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


