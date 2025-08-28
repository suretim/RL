# coding=utf-8
# -*- coding: utf-8 -*-
import serial
import threading
import serial.tools.list_ports
import time
import datetime
import os
import signal
import platform
import multiprocessing

from colorama import Fore, Back, Style

'''
Environment Install Step

0. Confirm your Python/Pip installation path.
1. pip install colorama
2. pip install pyserial
'''


bps = 115200

log_empty_line = "\033[0m\r\n"
log_color_head = "\033[0;"
log_color_tail = "\033[0m"
empty_line_byte = {'\r', '\n'}



# 终端颜色
def termcolor_init():
    from colorama import init
    #from termcolor import colored
    # use Colorama to make Termcolor work on Windows too
    init(autoreset=True)


# 输出串口列表
def print_com_list():
    port_list = list(serial.tools.list_ports.comports())

    if len(port_list) == 0:
        print(Fore.RED + "无可用串口！" + Fore.RESET)
        exit(0)
    else:
        print(Fore.GREEN +"\n可用串口列表: " + Fore.RESET)
        for i in range(0, len(port_list)):
            print(port_list[i])
    port = input(Fore.GREEN +"输入串口号(例如: 选择COM3, 请输入'3'): " + Fore.RESET)
    print(Fore.RESET)
    return port


# 发送输入的字符串
def serial_send_string(port, input_queue:multiprocessing.Queue, exit_event:threading.Event=threading.Event()):
    while True:
        data = input_queue.get()
        if data == "exit":
            exit_event.set()
            break
        else:
            # data = bytes(data + '\r\n', 'ascii')
            # port.write(data.encode('ascii'))
            pass
        time.sleep(0.2)

    # print(Fore.YELLOW + "退出发送线程" + Fore.RESET)
    if (platform.system() == 'Linux'):
        exit(0)

# 如果使用assic解码，过滤非 ascii 字符的数据, 转换成16进制字符串
def filter_no_assic_bytes(data):
    filtered_data = b''  # 初始化一个空的字节串

    for byte in data:
        if byte <= 0x7F:
            filtered_data += bytes([byte])  # 将小于等于0x7F的字节添加到新的数据流
        else:
            str_byte = "0x"
            str_byte += bytes([byte]).hex()
            filtered_data += str_byte.encode('ascii')
    return filtered_data


# 接受串口字符串
def recive_massage(port:serial.Serial, log_file_path="", exit_event:threading.Event=threading.Event()):    
    all_out, myout, log_color_cmd = "", "", ""
    log_content = ""

    while not exit_event.is_set():
        if port.inWaiting() > 0:
            # newline = filter_no_assic_bytes(port.readline())
            # myout = newline.decode('ascii')
            try:
                myout = port.readline().decode('utf-8')
                
                # 获取当前时间（datetime对象）
                # current_time = datetime.datetime.now()
                # 获取毫秒部分
                # current_time_ms = current_time.microsecond
                # print("当前时间（微秒）：", current_time_ms)

                # time_now = current_time.time().__str__()
                time_now = datetime.datetime.now().time().__str__()
                time_now = time_now[:-3] + ' '
                # print("当前时间: ", time_now)
                
                # color_str_idx = myout.find(log_color_head) + len(log_color_head)
                if myout.find(log_color_head) == 0 and myout[6] == "m":
                    # color = myout[color_str_idx : color_str_idx + 2]
                    # print("color = ", color)
                    # all_out = "".join([myout[:7], time_now, myout[7:]])

                    if myout.find(log_color_tail) > 7:
                        log_color_cmd = ""
                        log_content = time_now + myout[7:len(myout)-6]
                        # print("log_content =", log_content)
                        all_out = myout[0:7] + log_content + log_color_tail + '\n'
                        log_content += '\n'
                    else:
                        log_color_cmd = myout[0:7]
                        log_content = time_now + myout[7:]
                        # print("log_content_notail =", log_content)
                        all_out = log_color_cmd + log_content
                        

                elif myout.find(log_color_tail) == 0:
                    log_color_cmd = ""
                    if myout == log_empty_line:
                        print(myout, end='')
                        # if log_file_path != "":
                        #     with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        #         log_file.write('\n')
                        continue
                elif all(c in empty_line_byte for c in myout):
                    print(myout, end='')
                    if log_file_path != "":
                        with open(log_file_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(myout)
                    continue
                else:
                    ''' # for debug
                    if len(myout) < 5:
                        hexs = [hex(ord(c)) for c in myout]
                        print(hexs)
                    '''
                    all_out = log_color_cmd + time_now + myout

            except Exception as e:
                print("Serial Error:", e)
                if log_file_path != "":
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write("Serial Error:" + str(e) + "\n")
                continue
            # newline = filter_no_utf8_bytes(port.readline())
            # myout = newline.decode('utf-8')
            print(all_out, end='')
            if log_file_path != "":
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(log_content)
        else:
            time.sleep(0.002)

    # print(Fore.YELLOW + "退出接收线程" + Fore.RESET)
    


# 打开串口
def open_seri(portx, bps = 115200, timeout = None, log_file = "", exit_event:threading.Event=threading.Event()):
    ret = False
    try:  # 打开串口，并得到串口对象
        ser = serial.Serial(portx, bps, timeout=timeout)
        if(ser.is_open):  # 判断是否成功打开
            ret = True
            print(f"{portx} 串口打开成功!\n")
            th = threading.Thread(target=recive_massage, args=(
                ser, log_file, exit_event), name="recive")  # 创建一个子线程去等待读数据
            th.start()
        return ser, ret
    except Exception as e:
        print("打开串口错误! \n原因:\n", e)
        return serial, ret
    
def open_log_file_requeset():

    log_flg = input(Fore.GREEN + "是否开启日志记录功能？(y/n): " + Fore.RESET)
    if log_flg == 'y':
        log_path = input(Fore.GREEN +"请输入日志文件的保存路径：" + Fore.RESET)
        while log_path == "":
            log_path = input("请输入日志有效的文件路径：")
        log_name = input(Fore.GREEN + "给日志文件起个名字吧(直接按下回车可跳过): " + Fore.RESET)
        if log_name == "":
            log_name = "log_"
        else:
            log_name += "_"
        return True, log_path, log_name
    else:
        return False, "", ""
    
def serial_worker(log_flg=False, log_path="", log_name="", com=0, 
                  input_queue:multiprocessing.Queue=multiprocessing.Queue(), main_pid=0):

    termcolor_init()
    
    exit_event = threading.Event()
    exit_event.clear()

    # 串口接收
    # if os.uname()[0] == 'Windows':
    # elif os.uname()[0] == 'Linux':
    # os.uname()[0]
    print(f"当前系统为：{platform.system()}", end='\n\n')

    if platform.system() == 'Windows':
        if log_flg:
            if log_path[-1] != '\\':
                log_path += '\\'
            log_file_path = log_path + log_name + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            print(Fore.GREEN + f"日志文件路径为：{log_file_path}"+ Fore.RESET)
        else:
            log_file_path = ""
        
        com, ret = open_seri('COM' + str(com), bps, None, log_file_path, exit_event)
    
    elif platform.system() == 'Linux':
        if log_flg:
            if log_path[-1] != '/':
                log_path += '/'
            log_file_path = log_path + log_name + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            print(Fore.GREEN + f"日志文件路径为：{log_file_path}" + Fore.RESET)
        else:
            log_file_path = ""
            
        com, ret = open_seri('/dev/ttyUSB' + str(com), bps, None, log_file_path, exit_event)
        
    else:
        print(Fore.YELLOW + f"未知的系统类型, 无法处理: {platform.system()}" + Fore.RESET)

    if ret == False:
        print(Fore.RED + "串口打开失败，请检查串口连接!" + Fore.RESET)
        if main_pid != 0:
            os.kill(main_pid, signal.SIGINT)    # 发送信号给主进程，终止主进程
        exit(0)
    
# '''
    if input_queue:
        t2 = threading.Thread(target=serial_send_string, args=(
            com, input_queue, exit_event), name="send")  # 线程2：不断地去接收数据
        t2.start()  # 开启线程1
# '''  

if __name__ == '__main__':

    input_queue = multiprocessing.Queue()
    log_flg, log_path, log_name = open_log_file_requeset()
    com = print_com_list()

    task = multiprocessing.Process(target=serial_worker, 
                                   args=(log_flg, log_path, log_name, com, input_queue, os.getpid()))
    task.start()

    try:
        while True:
            user_input = input("")
            input_queue.put(user_input)  # 将输入数据放入队列
            if user_input == 'exit':
                break
    except KeyboardInterrupt:
        pass
    finally:
        task.join()  # 确保子进程退出
