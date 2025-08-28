#ifndef UI_H
#define UI_H

#include <stdio.h>
extern unsigned char is_gl_finished;
#define log(...)   			if(!is_gl_finished){ printf(__VA_ARGS__);} else {;}
//#define log(format, ...) if(!is_gl_finished){printf("[%s-%d]:"format,__FILE__,__LINE__,##__VA_ARGS__);}else {;}

extern void gl_printf(float x, float y, const char* format, ...);
extern void gl_string(float x, float y, const char* str);


#define CROSS_SHIFT_Y  		0.1
#define FONT_WIDTH  		24
#define LINE_MARGIN   		((double)FONT_WIDTH/4/(256*2))
// 打印区显示范围设定
#define DEBUG_X_WIDTH   	((double)1/5)  // 窗口右侧1/5用于打印调试文本
#define x_of_debug(_v_)		((double)(_v_) + (1-DEBUG_X_WIDTH))
#define y_of_debug(_v_)		(1 - CROSS_SHIFT_Y - (double)(_v_) )
#define y_of_line(_l_)  	y_of_debug((double)(_l_)*FONT_WIDTH/(256*2))
#define x_of_col(_c_)   	x_of_debug((double)(_c_)*FONT_WIDTH/(256*6))


// 画时间和档位曲线相关范围设定
#define LINE_X_WIDTH    	((double)1 -  DEBUG_X_WIDTH)  // 展示曲线的x宽度
//#define UI_MAX_SEC    		(90*60*60 + 0*60 + 0)  // 展示曲线的总时间秒数 hh : mm : ss
#define UI_MAX_SEC    		(2*60*60 + 0*60 + 0)  // 展示曲线的总时间秒数 hh : mm : ss
#define UI_MAX_LEVEL  		(10 + 1)   // 最大显示10档
#define x_of_sec(_v_)  		((double)(_v_)/UI_MAX_SEC*LINE_X_WIDTH)     
#define y_of_level(_v_)		((double)(_v_)/UI_MAX_LEVEL)

#define vcolor(_x_)  		((double)(_x_)/255)   // x = 0-255,  rgb?
#define pi  				3.1415926


#endif
