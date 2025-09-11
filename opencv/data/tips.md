# 一些注意事项
## 相机标定
标定时未白平衡，貌似影响不大
去畸变不准确，边缘部分还是有畸变，可以只取中间部分进行处理？
<span style="color:red">注意</span>
## Github
``` python
git add <file>
git commit -m "commit message"
git push <remote> <branch>
git log
git branch <name>
git branch
git checkout
git status
git log
git reset --hard HEAD^
git checkout -- file
git reset HEAD <file>
git checkout -b dev
git branch dev
git checkout dev
git branch
git checkout master
git switch -c dev
git switch dev
git merge <name>
git branch -d <name>
git merge --no-ff -m "merge with no-ff" dev
git stash
git stash list
git stash apply
git stash drop
git stash pop
git cherry-pick 4c805e2
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

<span style="color:red">注意</span>\
填入tag_map的四个点对应的是实际四个点
## 一些语法
Optional[X] 其实就是 Union[X, None] 的简写，意思是这个变量 可以是类型 X，也可以是 None。

Callable[[参数类型1, 参数类型2, ...], 返回类型] 用来表示 函数类型。

@dataclass 是 Python 3.7 引入的一个 装饰器，定义在 dataclasses 模块里，用来简化类的编写。\
如果你写一个普通的类，需要经常手动实现 __init__、__repr__、__eq__ 等方法，很啰嗦。
用了 @dataclass，Python 会自动帮你生成这些方法。
## 坐标角度表示
世界系W：赛场平面右手系，x 向右、y 向上（俯视图），z 朝上。

原点可取你方启动区左下角（或裁判统一原点）。赛场长约 10 m、宽约 5 m，可据此设定地图尺度。

机体系 B：车体坐标，x 指向车头前方，y 指向车体左侧，z 向上。

发射/相机系 C：相机/发射器自身坐标，用于瞄准/位姿标定。\
相机坐标系：X轴相机右方，y轴相机下方，Z轴相机前方\
tag坐标系：

场地是平面运动，只用航向角 yaw（θ） 就够了：
θ 为车体 x 轴相对世界系 x 轴的逆时针角度（弧度）。\
统一内部用弧度，显示/调参可用度。\
角度差一律做wrap到 [-π， π)：
wrap(α) = (α + π) mod 2π - π\
> 如果以后要做 3D 发射角（仰角），再引入 pitch；但底盘定位与路径规划还是以 SE(2) = (x, y, θ) 为主。

yaw(偏航): 绕Z轴旋转
pitch(俯仰): 绕Y轴旋转
roll(横滚): 绕X轴旋转
## 一些尺寸
tag 码边长12cm，纸边长14.9cm\
打击区 高15cm，边长120cm\
常规弹药区 高20cm，长180cm，宽15.4cm（？忘了），共五个孔，边缘孔离边缘40cm，每个孔8cm*8cm，两个孔最近两边相距15cm\
战略弹药区 边缘孔离边缘66.5cm，每个孔相隔32cm，总宽45.6cm，高29.7cm，底部还要高出15cm

打击区的tag_id是2
常规弹药区的tag_id是3
## 传输文件
前提能ping通（开启ssh）

香橙派传本机`scp -r orangepi@192.168.184.200:/home/orangepi/Robogame "D:\Desktop\robogame"`

本机传香橙派: 后面两个参数反过来即可
## 双摄像头
/dev/cam_up 对应上面个usb连的摄像头
/dev/cam_down 对应下面个usb连的摄像头
现暂时规定上面个连接新买的摄像头（112r），下面个连接旧的摄像头（158r）
下面的摄像头（cam1）用于取飞镖，上面的摄像头（cam0）用于发射

## 1
yaw朝y是90°，所以是与x轴的夹角

世界坐标系是右手坐标系 

相机坐标系：X轴相机右方，y轴相机下方，Z轴相机前方

机体系：车体坐标，x 指向车头前方，y 指向车体左侧，z 向上。

发送的转角度指令，默认逆时针为正