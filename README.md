# Robogame2025
This project is used for USTC Robogame2025.
## Files
```
opencv/
├── calib/
| ├── img/                  --标定相机用的图片
| ├── calc.py               --计算内参矩阵和畸变参数并调用摄像头根据计算结果实时测算棋盘格位姿
| ├── calib.npz             --标定得到的数据
| ├── findChess.py          --测试找棋盘格
| ├── take_photos.py        --拍标定相机的图片
| └── undistort.py          --测试去畸变后的相机
├── data/
| ├── config.json           --一些配置参数
| ├── hsv_range.json        --飞镖的HSV参数
| ├── tips.md               --一些注意事项
├── package/
| ├── __pycache__/
| ├── __init__.py           --使class文件夹被识别为python包
| ├── load_calib.py         --加载内参和畸变参数的代码
| ├── preprocess.py         --预处理函数
| └── undistored_camera.py  --调用摄像头并去畸变
├── src/
| ├── __pycache__/
| ├── car.py                --小车的类
| ├── comm.py               --与下位机交流的类以及通信协议
| ├── detector.py           --检测器的类
| ├── host_main.py          --测试通信
| ├── main.py               --主函数
| ├── TODO.md               --未完成的任务
| └── transform             --坐标变换时用到的一些矩阵相关的函数
├── tools/                  --一些测试和工具代码
| ├── find_obj.py           --使用掩码识别飞镖
| ├── get_param.py          --通过窗口找到飞镖的HSV参数
| ├── try.py                --AI写的识别Apriltag代码
| └── use_camera.py         --直接调用摄像头以及一些预处理结果
```
