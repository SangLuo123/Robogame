#!/bin/sh

# 记录启动日志
echo "=== 飞镖检测程序启动: $(date) ===" >> /home/orangepi/Robogame/start.log

# 等待系统服务完全启动
sleep 15

# 切换到程序所在目录
cd /home/orangepi/Robogame/opencv/src
echo "当前目录: $(pwd)" >> /home/orangepi/Robogame/start.log

# 设置环境变量
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
export DISPLAY=:0

# 加入 Python 本地用户库路径 (确认你的版本号，比如 python3.10)
export PYTHONPATH=/home/orangepi/.local/lib/python3.10/site-packages:$PYTHONPATH

# 检查Python和numpy
echo "检查Python环境..." >> /home/orangepi/Robogame/start.log
/usr/bin/python3 --version >> /home/orangepi/Robogame/start.log 2>&1
/usr/bin/python3 -c "import sys; print('sys.path=', sys.path)" >> /home/orangepi/Robogame/start.log 2>&1
/usr/bin/python3 -c "import numpy; print('numpy检查通过')" >> /home/orangepi/Robogame/start.log 2>&1

# 执行主程序 - 使用绝对路径
echo "开始执行主程序: $(date)" >> /home/orangepi/Robogame/start.log
/usr/bin/python3 -u new_main4.py >> /home/orangepi/Robogame/program_output.log 2>/dev/null



echo "程序退出: $(date)" >> /home/orangepi/Robogame/start.log
