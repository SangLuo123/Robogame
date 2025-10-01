import os
import sys

# --- 保证能导入 src/ 与 package/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from src.new_main1 import main as new_main1
from src.new_main2 import main as new_main2


# 全局变量存储参数
dart1_num = 3
dart2_num = 5
config_file = './config.txt'  # 替换为你的文件路径

# 读取配置文件
def read_config():
    global dart1_num, dart2_num
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith('dart1_num:'):
                        dart1_num = int(line.split(':')[1].strip())
                    elif line.startswith('dart2_num:'):
                        dart2_num = int(line.split(':')[1].strip())
            print(f'读取配置: dart1_num={dart1_num}, dart2_num={dart2_num}')
        else:
            print('配置文件不存在，使用默认值')
            write_config()  # 创建默认配置文件
    except Exception as e:
        print(f'读取配置失败: {e}，使用默认值')

# 写入配置文件
def write_config():
    try:
        with open(config_file, 'w') as file:
            file.write(f'dart1_num: {dart1_num}\n')
            file.write(f'dart2_num: {dart2_num}\n')
        print(f'保存配置: dart1_num={dart1_num}, dart2_num={dart2_num}')
    except Exception as e:
        print(f'保存配置失败: {e}')

# 正常启动函数
def normal_startup():
    print('执行正常启动流程')
    new_main1()  # 调用你的主函数

# 异常恢复函数
def exception_recovery():
    print('执行异常恢复流程')
    new_main2()  # 调用你的异常恢复函数

# 更新参数的函数
def update_parameters(new_dart1, new_dart2):
    global dart1_num, dart2_num
    dart1_num = new_dart1
    dart2_num = new_dart2
    write_config()  # 立即保存到文件

# 主函数
if __name__ == "__main__":
    # 程序启动时读取配置
    read_config()
    print(f'当前参数: dart1_num={dart1_num}, dart2_num={dart2_num}')
    
    # 根据参数决定执行哪个函数
    if dart1_num == 3 and dart2_num == 5:
        normal_startup()
    else:
        exception_recovery()
    
    # 示例：在需要时更新参数并保存
    # update_parameters(7, 8)