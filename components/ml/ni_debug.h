#ifndef _NI_DEBUG_H_
#define _NI_DEBUG_H_

void dbg_task_loop(void *arg);

void  dbg_init(void);
unsigned int dbg_parse_var(char *buf);
int  dbg_read(void *buf, unsigned int len);
//#define c_bp_pid_dbg_en 1
#define dbg_serial_usb_serial_jtag		1
#define dbg_serial_uart0				2
#define dbg_serial_sel        			dbg_serial_uart0
//#define dbg_serial_sel        			dbg_serial_usb_serial_jtag
#endif