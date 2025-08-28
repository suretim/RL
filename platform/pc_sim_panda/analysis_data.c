#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include <math.h>
#include "gl.h"
#include "types.h"
#include "ui.h"

#define X_11(_v_)          (((_v_)<<3)+((_v_)<<1)+(_v_))
#define X_12(_v_)          (((_v_)<<3)+((_v_)<<2))
#define HZ(_v_)            ((ushort)(2000000/(_v_)))      

FILE *fsrc,*fdes;

#define uart_putch(_v_)  fputc(_v_,fdes)//putchar(_v_)

bool bNegative;   
uchar flag_inc  ;
uchar flag_dec  ;
uchar flag_stable ,g_unstable_cnt,stable_cnt   ;
ushort Delt_Count,Delt_Count_Stable,Delt_Count_Old,SubVal,g_freq_buf[128],g_sum_h,g_sum_l,g_int0_cnt;
ushort freq_array[]={502,503,501,502,503,502,502,503,502,503};
ulong spend_time;
ushort Fin;
uchar g_scale,g_melody,g_wire_num,g_mode;



uchar hex2val(uchar ascii_code)
{
       if('9'>=ascii_code &&  ascii_code>='0') 
       	       return (ascii_code - '0');
       if('f'>=ascii_code &&  ascii_code>='a')	   
	       return (ascii_code - 'a' + 10);
       if('F'>=ascii_code &&  ascii_code>='A')
	       return (ascii_code - 'A' + 10);

       return 0xff;	   
}

ushort str2val(uchar *hex_h, uchar *hex_l)
{
   uchar hbyte,lbyte;
   ushort val;

   hbyte = hex2val(hex_h[0]);
   hbyte = (hbyte<<4) + hex2val(hex_h[1]);
   lbyte = hex2val(hex_l[0]);
   lbyte = (lbyte<<4) + hex2val(hex_l[1]);

   val = hbyte<<8;
   val += lbyte;
  
   return  val;
}

void acor(float *src, int count, float *des)
{
    int i,j;

    
	for(i=0; i<=count-1; i++)
    { 
        des[i] = 0.0;
        for(j=0; j<=count-1; j++)
            if(j-i>=0 && j-i<=count-1)
				des[i] = des[i] + src[j]*src[j-i];
    }
    /*
	for(i=0; i<count; i++){
		des[i] = 0;
	    for(j=0; j<count-i; j++){
	        des[i] += src[j]*src[i+j];
	    }
        des[i]=des[i]/(j+1.000);
    }*/
}

#define COUNT    50
#define STARTID  100
void draw_lines()
{
    ushort i;
    uchar hbyte[4],lbyte[4];
	ushort time_start=0;
    //ushort base;
	float src[COUNT];
	float des[COUNT];
	
	// open the file for read and write operation
	if((fsrc=fopen("gl_data.txt","r"))==NULL){
		//if the file does not exist print the string
		printf("Cannot open the file src...");
		exit(1);
	}
	if((fdes=fopen("output-xcor.txt","w+"))==NULL){
		//if the file does not exist print the string
		printf("Cannot open the file des...");
		exit(1);
	}

	//for(i=10; i<1000; i+=10)
	//   fprintf(fdes,"\n%dHZ-%ld-delT=%ld", i, HZ(i), HZ(i)-HZ(i+1));
	//goto _lExit;

	
	fseek(fsrc,0,SEEK_SET);
    i = 0;	
	g_int0_cnt = 0;
	spend_time = 0;
    /*
	while(!feof(fsrc))
	{
		if(fscanf(fsrc,"%s%s",hbyte,lbyte) == 2)	
		    fprintf(fdes,"\n%d",str2val(hbyte,lbyte));
	}*/


	glBegin(GL_LINE_STRIP);
	
	glColor3f(0,0,0); //设置当前颜色
	  
	while(!feof(fsrc))
	{
		if(fscanf(fsrc,"%s%s",hbyte,lbyte) == 2)	
		{
		    Delt_Count = str2val(hbyte,lbyte);
			g_int0_cnt++;
			spend_time += Delt_Count/2000;
			//if(find_stable())
			{
				Fin = Delt_Count;//Delt_Count_Stable;
				//cal_melody();
				//disp();
			}
			if(STARTID == g_int0_cnt)
			    time_start = spend_time;
            if(STARTID<=g_int0_cnt && g_int0_cnt < STARTID+COUNT)
			{
			    src[g_int0_cnt-STARTID] = (float)Delt_Count;
			    glVertex3f((float)(g_int0_cnt-STARTID)/(100),(float)(Delt_Count)/5000,0);
            }
		}
		    
	}

    glEnd();

	acor(src, COUNT, des);
	for(i=0; i<COUNT; i++)
	{
	     fprintf(fdes,"\n%f",des[i]);
         
	}
	glBegin(GL_LINE_STRIP);	
	
    glColor3f(1,0,0); //设置当前颜色
	
    for(i=0; i<COUNT; i++)
       glVertex3f((float)(i)/(100),(float)(des[i])/5000,0);

	glEnd();

	
//_lExit:	
	fclose(fsrc);
	fclose(fdes);
}
