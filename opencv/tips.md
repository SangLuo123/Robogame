# 一些注意事项
## 相机标定
标定时未白平衡，貌似影响不大
去畸变不准确，边缘部分还是有畸变，可以只取中间部分进行处理？
<span style="color:red">注意</span>
## Github
``` python
git add <file>
git commit -m "commit message"
git push
git log
# git remote add origin https://github.com/SangLuo123/Robogame 
```
## vnc
每次使用vnc时需要先打开，使用命令`vncserver`
注意python代码如果有展示图片，但在纯命令行中运行会出错，故调试时需要使用vnc，但是实际比赛时用不上vnc，故需要特殊处理？
<span style="color:red">注意</span>
## 相机参数设置
不知道需不需要重新设置相机，比如自动白平衡、曝光时间等。
<span style="color:red">注意</span>
## 飞镖上的铁片
宽 18.8mm
长 28.5mm