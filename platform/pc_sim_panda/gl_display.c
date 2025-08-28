#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include <windows.h>
#include <math.h>
#include <GL\gl.h>
#include "types.h"
#include "ui.h"
#include "utc_fun.h"
#include "..\..\components\rule\time_rule\time_rules.h"

uchar is_gl_finished = FALSE;  


void ui_set_background(float *color)
{
    glClearColor(color[0], color[1], color[2], 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

void draw_level_texts()  // draw level0-level10 lines
{
       uchar level;
	   
	   glTranslatef(-0.035, -0.006, 0);      
       for(level = 0; level < UI_MAX_LEVEL; level++)
		    gl_printf(x_of_sec(0), y_of_level(level), "level-%02d", level);
	   glTranslatef(0.035, +0.006, 0);  	    
}

void set_cross_line(float * color) //在当前OCS的中心画一个十字
{      
       uchar level;
  
       glTranslatef(-0.93,-1+CROSS_SHIFT_Y, 0);  // 移动坐标到左下角，放大一倍
       glScalef(2,2,1);   
	       
       glBegin(GL_LINES);

       glColor3fv(color); //设置当前颜色

       glVertex2f(0,0);  // (0,0) -> (1,0) , 坐标横线
       glVertex2f(1,0);

       glVertex2f(0,0);  // (0,0) -> (0,1)  , 坐标竖线
       glVertex2f(0,1);                       
                 
       for(level = 0; level < UI_MAX_LEVEL; level++)
       {
            glVertex2f(x_of_sec(0),y_of_level(level));  
		    glVertex2f(x_of_sec(UI_MAX_SEC),y_of_level(level));
	   }
	   
	   glEnd();
	   
	   draw_level_texts();

}

#if 0
////////////////////////////////////////////////aci rule library test function/////////////////////////////////////////////////

void input_timer_on_setting_data(timer_on_ctrl_data_t *setting)
{
    setting->on_hour = 10;
    setting->on_min = 18;
}

void print_init_para(u32 start_time,  timer_on_ctrl_data_t timer_setting)
{
	Time_Typedef rtc_time,rtc_end_time;
	
	rtc_time = utc_to_rtc(start_time);
	rtc_end_time = utc_to_rtc(start_time + UI_MAX_SEC);
	log("start_time = %ld secs, [%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]\n", start_time, rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
	                                                                      rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	// 打印初始数据
	glColor3f(0,0,0); 	//设置当前颜色
	gl_printf(x_of_col(2), y_of_line(0), ".........timer on rule.........");
    gl_printf(x_of_col(0), y_of_line(2),"[%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]", rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
		                                                                      rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	gl_printf(x_of_col(0), y_of_line(4), "timer on setting: %dH:%dM", timer_setting.on_hour, timer_setting.on_min);	
	
}

void disp_timer_on_rule(void)
{
	double point_x=0;
	timer_on_ctrl_data_t timer_on_setting; 
	timer_running_data_t running_data;
	Time_Typedef rtc_time;
	
	u32 utc_sec, start_time;  			// x
	uchar cur_level = 6, new_level = 0;	// y  
	
	input_timer_on_setting_data(&timer_on_setting); 		// 模拟从APP设置获得的规则设置数据
    
	// 从 2024-7-29 16：08：00 开始运行，持续时长-UI_MAX_SEC
	start_time = to_utc_sec(2024,7,30, 0, 0, 0); 			// 设定起始时间
    print_init_para(start_time, timer_on_setting); 			// 打印 控制参数和画图时间范围
    
    
	glBegin(GL_LINE_STRIP);		// 画线开始 
	glColor3f(255,0,0); 		//设置当前颜色
		
	for(utc_sec = start_time; utc_sec < start_time + UI_MAX_SEC; utc_sec++)
	{
		rtc_time = utc_to_rtc(utc_sec);   // 模拟rtc_read，utc 秒数转成年月日时分秒
		
		new_level = rule_timer_on(&timer_on_setting, 10, 1, cur_level, &rtc_time, &running_data);
		
		point_x = x_of_sec((utc_sec+UI_MAX_SEC-start_time)%UI_MAX_SEC);
		
		if(new_level != cur_level)  // 档位变化时，在附近打印时间
		{
			glVertex2f(point_x, y_of_level(new_level));  // 设定要连线的点
			
			glEnd(); 	// 暂时退出画线模式
		    gl_printf(point_x, y_of_level(new_level)+LINE_MARGIN, "%d:%d:%d", rtc_time.hour, rtc_time.min, rtc_time.sec);  // 打印变化档位的时间
		    glBegin(GL_LINE_STRIP);  // 回到画线模式
		    
		    log("(%d:%d:%d,level=%d)", rtc_time.hour, rtc_time.min, rtc_time.sec, new_level); 
		}
		
		cur_level = new_level;
		
		glVertex2f(point_x, y_of_level(cur_level));  // 设定要连线的点
	}
	
	glEnd();  // 画线结束
}
#endif

/*
****************************************************************************************************


****************************************************************************************************
*/
void print_init_param_timer(u32 start_time,  timer_ctrl_data_t timer_setting)
{
	Time_Typedef rtc_time,rtc_end_time;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	 
	// 设置文本颜色为红色
	SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
	//printf("红色文字\n");
	
	rtc_time = utc_to_rtc(start_time);
	rtc_end_time = utc_to_rtc(start_time + UI_MAX_SEC);
	log("start_time = %ld secs, [%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]\n", start_time, rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
	                                                                      rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	// 打印初始数据
	glColor3f(0,0,0); 	//设置当前颜色
	gl_printf(x_of_col(2), y_of_line(0), ".........timer on rule.........");
    gl_printf(x_of_col(0), y_of_line(2),"[%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]", rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
		                                                                      rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	gl_printf(x_of_col(0), y_of_line(4), "timer on setting: %dH:%dM", timer_setting.hour, timer_setting.min);
	
}


void disp_timer_rule(void)
{
	double point_x=0;
	Time_Typedef rtc_time;	
	u32 utc_sec, start_time;  			// x
	u8 cur_level = 6, new_level = 0;	// y  	
	
	u8 max_level = 0;
	u8 min_level = 0;
	
	timer_ctrl_data_t timer_on_setting = {0}; 	// timer_on
//	timer_ctrl_data_t timer_off_setting = {0};	// timer_off	
	time_running_data_t running_data = {0};
	
	// 模拟从APP设置获得的规则设置数据
	default_timer_on(&timer_on_setting, &running_data);
	set_timer_on_ctrl_data(&timer_on_setting, 1, 0);
	set_timer_on_running_data(&running_data, &timer_on_setting, 1);
	max_level = 9;
	min_level = 2;	
    
	// 从 2024-7-29 16：08：00 开始运行，持续时长-UI_MAX_SEC
	start_time = to_utc_sec(2024, 7, 30, 0, 0, 0); 			// 设定起始时间
    print_init_param_timer(start_time, timer_on_setting); 	// 打印控制参数和画图时间范围
    
	rtc_time = utc_to_rtc(start_time);
    set_time_rule_restart(&running_data, &rtc_time, 1, 0);

	glBegin(GL_LINE_STRIP);		// 画线开始
	glColor3f(255, 0, 0); 		// 设置当前颜色
	
	for(utc_sec = start_time; utc_sec < start_time + UI_MAX_SEC; utc_sec++)
	{
		point_x = x_of_sec((utc_sec+UI_MAX_SEC-start_time)%UI_MAX_SEC);		
		rtc_time = utc_to_rtc(utc_sec);  // 模拟rtc_read，utc 秒数转成年月日时分秒
		
		
		rule_timer_on(&timer_on_setting, max_level, min_level, &cur_level, &rtc_time, &running_data);

		if(new_level != cur_level)  // 档位变化时，在附近打印时间
		{
			log("(%d:%d:%d, new_level = %d, cur_level = %d)\n", rtc_time.hour, rtc_time.min, rtc_time.sec, new_level, cur_level); 			
			new_level = cur_level;
			glVertex2f(point_x, y_of_level(new_level));  // 设定要连线的点
			
			glEnd(); // 暂时退出画线模式
		    gl_printf(point_x, y_of_level(new_level)+LINE_MARGIN, "%d:%d:%d", rtc_time.hour, rtc_time.min, rtc_time.sec);  // 打印变化档位的时间
		    glBegin(GL_LINE_STRIP);  // 回到画线模式
		}		
		
		glVertex2f(point_x, y_of_level(cur_level));  // 设定要连线的点
	}
	
	glEnd();  // 画线结束
}

/*
****************************************************************************************************


****************************************************************************************************
*/
void print_init_param_cycle(u32 start_time,  cycle_ctrl_data_t cycle_setting)
{
	Time_Typedef rtc_time,rtc_end_time;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	 
	// 设置文本颜色为红色
	SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
	//printf("红色文字\n");
	
	rtc_time = utc_to_rtc(start_time);
	rtc_end_time = utc_to_rtc(start_time + UI_MAX_SEC);
	log("start_time = %ld secs, [%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]\n", 
		start_time, 
		rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
	 	rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	 	
	// 打印初始数据
	glColor3f(0,0,0); //设置当前颜色
	gl_printf(x_of_col(2), y_of_line(0), ".........cycle rule.........");
    gl_printf(x_of_col(0), y_of_line(2),"[%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]", 
		rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
		rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	gl_printf(x_of_col(0), y_of_line(4), "cycle setting : ");		
	gl_printf(x_of_col(0), y_of_line(5), "ON ( %2dh:%2dm )", cycle_setting.on_hour, cycle_setting.on_min);	
	gl_printf(x_of_col(0), y_of_line(6), "OFF ( %2dh:%2dm )", cycle_setting.off_hour, cycle_setting.off_min);	
	
}

void disp_cycle_rule(void)
{	
	double point_x=0;
	Time_Typedef rtc_time;	
	u32 utc_sec, start_time;  			// x
	u8 cur_level = 6, last_level = 0;	// y  	
	u8 dummy_level = 0;

	u8 max_level = 0;
	u8 min_level = 0;
	
	cycle_ctrl_data_t cycle_setting = {0};	
	time_running_data_t running_data = {0};
	sun_t sun = {0};
	u8 run_sun = 0;

	// 模拟从APP设置获得的规则设置数据
	max_level = 9;
	min_level = 2;
	run_sun = 0;
	default_cycle(&cycle_setting, &running_data);
	set_cycle_ctrl_data(&cycle_setting, 0, 10, 0, 10);
    set_cycle_running_data(&running_data, &cycle_setting, 1, 1);
    sun_set_data(&sun, 1, hm_to_min(0, 10));    
    
	// 从 2024-7-29 16：08：00 开始运行，持续时长-UI_MAX_SEC
	start_time = to_utc_sec(2024, 7, 30, 0, 0, 0); 		// 设定起始时间
    print_init_param_cycle(start_time, cycle_setting); 	// 打印控制参数和画图时间范围  
    
    rtc_time = utc_to_rtc(start_time);
    set_time_rule_restart(&running_data, &rtc_time, 1, 0);

	glBegin(GL_LINE_STRIP);		// 画线开始
	glColor3f(255, 0, 0); 		// 设置当前颜色
		
	last_level = cur_level;
	for(utc_sec = start_time; utc_sec < start_time + UI_MAX_SEC; utc_sec++)
	{			
		point_x = x_of_sec((utc_sec+UI_MAX_SEC-start_time)%UI_MAX_SEC);		
		rtc_time = utc_to_rtc(utc_sec);   // 模拟rtc_read，utc 秒数转成年月日时分秒
		
		
		rule_cycle(&cycle_setting, max_level, min_level, &cur_level, &dummy_level, &rtc_time, &running_data, &sun, run_sun);	
			
		
		if(last_level != cur_level)  // 档位变化时，在附近打印时间
		{
			log("(%d:%d:%d, last_level = %d, cur_level = %d)\n\n", rtc_time.hour, rtc_time.min, rtc_time.sec, last_level, cur_level); 			
			last_level = cur_level;
			glVertex2f(point_x, y_of_level(last_level));  // 设定要连线的点
			
			glEnd(); // 暂时退出画线模式
		    gl_printf(point_x, y_of_level(last_level)+LINE_MARGIN, "%d:%d:%d", rtc_time.hour, rtc_time.min, rtc_time.sec);  // 打印变化档位的时间
		    glBegin(GL_LINE_STRIP);  // 回到画线模式
		}
		
		glVertex2f(point_x, y_of_level(cur_level));  // 设定要连线的点
	}
	
	glEnd();  // 画线结束
}



/*
****************************************************************************************************


****************************************************************************************************
*/
void print_init_param_sched(u32 start_time, sched_ctrl_data_t sched_setting)
{
	Time_Typedef rtc_time,rtc_end_time;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	 
	// 设置文本颜色为红色
	SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
	//printf("红色文字\n");
	
	rtc_time = utc_to_rtc(start_time);
	rtc_end_time = utc_to_rtc(start_time + UI_MAX_SEC);
	log("start_time = %ld secs, [%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]\n", 
		start_time, 
		rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
	 	rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
	 	
	// 打印初始数据
	glColor3f(0,0,0); //设置当前颜色
	gl_printf(x_of_col(2), y_of_line(0), ".........sched rule.........");
    gl_printf(x_of_col(0), y_of_line(2),"[%d-%d-%d %d:%d:%d] to [%d-%d-%d %d:%d:%d]", 
		rtc_time.year, rtc_time.month, rtc_time.date, rtc_time.hour, rtc_time.min, rtc_time.sec,
		rtc_end_time.year, rtc_end_time.month, rtc_end_time.date, rtc_end_time.hour, rtc_end_time.min, rtc_end_time.sec);
		
	gl_printf(x_of_col(0), y_of_line(4), "sched setting : ");
	if (sched_setting.start_hour == 0xff && sched_setting.start_min == 0xff)
		gl_printf(x_of_col(0), y_of_line(5), "start ( --:-- )");	
	else	
		gl_printf(x_of_col(0), y_of_line(5), "start ( %2dh:%2dm )", sched_setting.start_hour, sched_setting.start_min);	
				
	if (sched_setting.end_hour == 0xff && sched_setting.end_min == 0xff)	
		gl_printf(x_of_col(0), y_of_line(6), "end ( --:-- )");
	else 
		gl_printf(x_of_col(0), y_of_line(6), "end ( %2dh:%2dm )", sched_setting.end_hour, sched_setting.end_min);	
	
}

void disp_sched_rule(void)
{
	double point_x=0;
	Time_Typedef rtc_time;	
	u32 utc_sec, start_time;  			// x
	u8 cur_level = 6, last_level = 0;	// y
	
	u8 max_level = 0;
	u8 min_level = 0;
	
	sched_ctrl_data_t sched_setting = {0};	
	sched_ctrl_run_data_t sched_run = {0};
	time_running_data_t running_data = {0};
	sun_t sun = {0};
	u8 run_sun = 0;

	// 模拟从APP设置获得的规则设置数据
	max_level = 9;
	min_level = 0;
	run_sun = 0;
	default_sched(&sched_setting, &sched_run, &running_data);
	set_sched_ctrl_data(&sched_setting, 0, 30, 1, 30);
    sun_set_data(&sun, 1, hm_to_min(0, 40));
    
	// 从 2024-7-29 16：08：00 开始运行，持续时长-UI_MAX_SEC
	start_time = to_utc_sec(2024, 7, 30, 0, 0, 0); 		// 设定起始时间
    print_init_param_sched(start_time, sched_setting); 	// 打印控制参数和画图时间范围
    
    rtc_time = utc_to_rtc(start_time);
    set_time_rule_restart(&running_data, &rtc_time, 1, 0);

	glBegin(GL_LINE_STRIP);		// 画线开始
	glColor3f(255, 0, 0); 		// 设置当前颜色
	
	last_level = cur_level;
	for(utc_sec = start_time; utc_sec < start_time + UI_MAX_SEC; utc_sec++)
	{
		point_x = x_of_sec((utc_sec+UI_MAX_SEC-start_time)%UI_MAX_SEC);
		rtc_time = utc_to_rtc(utc_sec);   // 模拟rtc_read，utc 秒数转成年月日时分秒
		
		
		rule_sched(&sched_setting, &sched_run, max_level, min_level, &cur_level, &rtc_time, &running_data, &sun, run_sun);		
		
		
		if(last_level != cur_level)  // 档位变化时，在附近打印时间
		{
			log("(%d:%d:%d, last_level = %d, cur_level = %d)\n\n", rtc_time.hour, rtc_time.min, rtc_time.sec, last_level, cur_level); 			
			last_level = cur_level;
			glVertex2f(point_x, y_of_level(last_level));  // 设定要连线的点
			
			glEnd(); // 暂时退出画线模式
		    gl_printf(point_x, y_of_level(last_level)+LINE_MARGIN, "%d:%d:%d", rtc_time.hour, rtc_time.min, rtc_time.sec);  // 打印变化档位的时间
		    glBegin(GL_LINE_STRIP);  // 回到画线模式
		}
		
		glVertex2f(point_x, y_of_level(cur_level));  // 设定要连线的点
	}
	
	glEnd();  // 画线结束
}

/*
****************************************************************************************************


****************************************************************************************************
*/
void gl_main(void)
{	 
    float darkgrey[3]={vcolor(100),vcolor(100),vcolor(100)};
    float grey[3]={vcolor(120),vcolor(120),vcolor(120)};
    
    //if(is_gl_finished)
    //    return;
          
    log("gl start..\n");
    ui_set_background(grey);  // set image background
    glLoadIdentity();  // reinit , disregard old changes
    set_cross_line(darkgrey);  // 画坐标线
    
    {
	    //disp_timer_on_rule();  // 绘制时间档位变化曲线
	    
		disp_timer_rule();
		// disp_cycle_rule();
		// disp_sched_rule();
	}
    
    glFlush();          //更新窗口
    	
    log("gl end..\n");  
	is_gl_finished = TRUE;  

}