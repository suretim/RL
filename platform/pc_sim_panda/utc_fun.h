#ifndef _UTC_FUN_H_
#define _UTC_FUN_H_

extern Time_Typedef utc_to_rtc(u32 ulSecond);
extern u32 rtc_to_utc(Time_Typedef time);
extern u32 to_utc_sec(u16 year, u8 month, u8 date, u8 hour, u8 min, u8 sec);

#endif
