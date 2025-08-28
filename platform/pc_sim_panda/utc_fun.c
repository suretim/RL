#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include "types.h"

// 当前 RTC 时间 距离 2000-1-1  0：00：00 的秒数
u32 rtc_to_utc(Time_Typedef time)
{
	static u16 CommYear[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
	static u16 LeapYear[13] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
	
	u32 utc = 0;
	u16 *tmpArray;
	u8 year, month, data, hour, min, sec;

	year = time.year;
	month = time.month;
	data = time.date;
	hour = time.hour;
	min = time.min;
	sec = time.sec;

	// calculate time by year
	utc = (year / 4) * (366 + 365 + 365 + 365) * 24 * 60 * 60;
	year = year % 4;
	switch (year)
	{
	case 1:
		utc += 366 * 24 * 60 * 60;
		break;

	case 2:
		utc += (366 + 365) * 24 * 60 * 60;
		break;

	case 3:
		utc += (366 + 365 + 365) * 24 * 60 * 60;
		break;

	default:
		break;
	}

	// calculate time by month
	if (year)
	{
		tmpArray = CommYear;
	}
	else
	{
		tmpArray = LeapYear;
	}
	utc += tmpArray[month - 1] * 24 * 60 * 60;

	// calculate time by day
	utc += (data - 1) * 24 * 60 * 60;

	// calculate time by hour
	utc += hour * 60 * 60;

	// calcualte time by minute
	utc += min * 60;

	// calcualte time by second
	utc += sec;

	return utc;
}

//1-- 函数：秒数转tm时间 
Time_Typedef utc_to_rtc(u32 ulSecond)
{
	/* 一个月有多少天 */
	#define DAYS_OF_MONTH(lMonth)     (alDaysOfMonth[(lMonth) - 1])  
	         
	/* 是否为闰年 */
	#define LEAP_YEAR(lYear)    (0 == ((lYear) % 400)  || (0 != ((lYear) % 100) && 0 == ((lYear) % 4) ) )
	 
	/* 一年有几天 */
	#define DAYS_IN_YEAR(lDays)       (LEAP_YEAR(lDays) ? 366 : 365)   
    
	u32 alDaysOfMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	
    u32 lLoopIndex;
    u32 lHMS;                         
    u32 lDays;
    Time_Typedef stTime;
 
    /* 补时区(根据实际情况，自行处理是否需要补时区) */
    //ulSecond+= (Uint)gstMng.lTimeZoneOff;
    lHMS = ulSecond;
    lDays = ulSecond/ 86400;  //86400 一天的秒数
 
    /* 获取小时、分、秒 */
    stTime.hour = (u8)((lHMS / 3600) %24);
    stTime.min = (u8)((lHMS % 3600) / 60);
    stTime.sec = (u8)((lHMS % 3600) % 60);
 
    /* 现在处于哪一年 */
    for (lLoopIndex = 2000; lDays >= DAYS_IN_YEAR(lLoopIndex) ; lLoopIndex++)
    {
        lDays -= DAYS_IN_YEAR(lLoopIndex);
    }
 
    stTime.year = (int)lLoopIndex - 2000;
    //stTime.d = (int)lDays;
    
    /* 如果是闰年，则2月为29天 */
    if (LEAP_YEAR(stTime.year))
    {
        DAYS_OF_MONTH(2) = 29;
    }
    
    for (lLoopIndex = 1; lDays>=DAYS_OF_MONTH(lLoopIndex) ; lLoopIndex++)
    {
        lDays -= DAYS_OF_MONTH(lLoopIndex);
    }
 
    /* 属于哪个月 */
    stTime.month = (int)lLoopIndex;
 
    /* 恢复2月28天作为默认值 */
    DAYS_OF_MONTH(2) = 28;
 
    /* 一个月的第几天 */
    stTime.date = (int)lDays + 1;
 
    return stTime;
}
 
u32 to_utc_sec(u16 year, u8 month, u8 date, u8 hour, u8 min, u8 sec)
{
    Time_Typedef start_time;
    
    if(year < 2000)
        return 0;
        
    start_time.year = year - 2000;
    start_time.month = month;
    start_time.date = date;
    start_time.hour = hour;
    start_time.min = min;
    start_time.sec = sec;
    
    return rtc_to_utc(start_time);
}
 
void utc_convert_test(void)
{ 
	//2-- 年月日时分秒
	//char szTime[64] = {0};
	//struct tm stTime = SecToTimeVal((Uint)tTime);
	
    //snprintf(szBeginTime, sizeof(szBeginTime), "%d/%.2d/%.2d %.2d:%.2d:%.2d", stBeginTime.tm_year, stBeginTime.tm_mon, stBeginTime.tm_mday, stBeginTime.tm_hour, stBeginTime.tm_min, stBeginTime.tm_sec); 
}
