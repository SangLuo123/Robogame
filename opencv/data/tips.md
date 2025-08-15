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
## 手眼标定
若摄像头固定在小车上不动，只需测出偏移量就行，虽然偏移量也有点麻烦
若摄像头固定在机器臂上随着机器臂移动，需要进行标定，很麻烦？\
目前转换的原理
$$
T_{world \leftarrow robot} = T_{world \leftarrow tag} \cdot (T_{cam \leftarrow tag})^{-1} \cdot (T_{robot \leftarrow cam})^{-1}
$$
## @staticmethod
定义静态方法的装饰器。
## 协议格式
$：帧头\
#：帧尾\
速度：$V<v_m_s>,<w_rad_s># （eg：V0.250,0.100）\
急停：$S#\
查询编码器：$Q# ；回 $E...#（查询轮子转了多少圈等）\
ACK：$AOK# / $AERR,code#（可选）（应答：下位机收到信号后进行回复是否准确收到）\
发射器转速：$P<rpm># （eg：$P3200#）\
触发发射：$T#\
机械臂预设动作：$C<name>#（eg：$CGRAB#、$CREL#）
## 速度控制
差速驱动运动学（unicycle model）\
懂得 v/w 是如何转化成左右轮速度的\
关键词：differential drive kinematics

P 控制导航到点\
距离差控制 v\
角度差控制 w

里程计（Odometry）\
学会用编码器数据估计位置\
关键词：wheel encoder odometry

Pure Pursuit 路径跟踪\
如果要走多个点/轨迹，用这个很方便
## 一些语法
Optional[X] 其实就是 Union[X, None] 的简写，意思是这个变量 可以是类型 X，也可以是 None。

Callable[[参数类型1, 参数类型2, ...], 返回类型] 用来表示 函数类型。