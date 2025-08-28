import sys
import os
import platform
from pathlib import Path
import re
from colorama import Fore, Back, Style

''' [文件注释]
    -
    - 用途：遇到ESP32芯片的backtrace，使用该工具快速定位到源码位置。
    - 使用方法：
    -   打开一个配置了ESP-IDF环境的终端，进入[项目的根目录]，执行以下命令：
    -   python ./backtrace_tool.py
    - 
    - 人工检查运行环境项：
    - 1. 检查当前目录是否为项目根目录，且存在CMakeLists.txt和build文件夹
    - 2. 检查环境变量IDF_PATH和IDF_TOOLS_PATH是否设置
    - 3. 检查系统环境变量PATH中是否包含xtensa-esp32-elf-addr2line.exe
    - 
'''



pwd = Path(__file__).parent.absolute()

pj_top_cmke = os.path.join(pwd, "CMakeLists.txt")
cmake_floder = os.path.join(pwd, "build")

if not os.path.exists(pj_top_cmke) or not os.path.exists(cmake_floder):
    print(Fore.RED + "Please run this script in the project root directory"
          " (with project cmake file)."+ Fore.RESET)
    exit(1)

idf_path = os.getenv("IDF_PATH")
idf_tools = os.getenv("IDF_TOOLS_PATH")

if idf_path is None:
    print(Fore.RED + "IDF_PATH or IDF_TOOLS_PATH is not set, "
          "please run this script after idf.py environment setup."+ Fore.RESET)
    exit(1)


def find_parser_path(target ="") -> str:
    parser_path = ""
    system_path = os.getenv("PATH")

    if system_path is None:
        print(Fore.RED + "PATH is not set, "
              "please run this script after IDF source ./export environment setup."+ Fore.RESET)
        exit(1)

    path_list = system_path.split(os.pathsep)
    
    for p in path_list:
        # print(p)
        if target in p:
            parser_path = p
            break

    return parser_path


def find_project_name(cmke_file_path = ""):
    if not os.path.exists(cmke_file_path):
        print(Fore.RED + "CMakeLists.txt file not found."+ Fore.RESET)
        exit(1)
    with open(cmke_file_path, "r") as f:
        for line in f.readlines():
            if re.match(r"project\(", line):
                pj_name = line.split("(")[1].split(")")[0]
                break
        return pj_name


if platform.system() == 'Windows':
    parser_name = "xtensa-esp32-elf-addr2line.exe"
elif platform.system() == 'Linux':
    parser_name = "xtensa-esp32-elf-addr2line"
else:
    parser_name = "xtensa-esp32-elf-addr2line"


pj_name = find_project_name(pj_top_cmke)
parser_path = find_parser_path("xtensa-esp32-elf")
elf_name = pj_name + ".elf"


path_lin = os.path.join(parser_path, parser_name) + " -e "
path_elf = os.path.join(cmake_floder, elf_name)



def esp_backtrace(err_str=''):
    if err_str == '':
        return
    find = re.findall(r"0x4(.+?):", err_str)

    print(Fore.YELLOW + "--- Backtrace result:" + Fore.RESET) 
    for sss in find:
        addr = " 0x4"+sss
        os.system(path_lin + path_elf+addr)



if __name__ == "__main__":

    backtrace_line = input(Fore.GREEN + Style.BRIGHT + 
                           "Please input backtrace line: \n" + 
                           Style.RESET_ALL + Fore.RESET)
    
    esp_backtrace(backtrace_line)