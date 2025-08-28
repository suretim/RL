#ifndef  TYPES_H
#define TYPES_H

typedef unsigned char u8;
typedef unsigned char uchar;
typedef unsigned char bool;
typedef unsigned short ushort;
typedef unsigned short u16;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long u32;


typedef struct
{
	// 当前时钟芯片的时间， 十进制  （定时、历史数据、历史日志等功能 会用到）
	u8 year;  // 年     20xx
	u8 month; // 月     1~12
	u8 week;  // 星期   1~7
	u8 date;  // 日     1~31
	u8 hour;  // 小时   0~23
	u8 min;	  // 分钟   0~59
	u8 sec;	  // 秒钟   0~59
} Time_Typedef;

#define TRUE    1
#define FALSE   0
#endif
