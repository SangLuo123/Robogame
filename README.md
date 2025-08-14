# Robogame2025
This project is used for USTC Robogame2025.
## Files
├──
└──
```
opencv/
├── img/                    --标定相机用的图片
├── mtx_dist/
| ├── calc.py               --计算内参矩阵和畸变参数并调用摄像头根据计算结果实时测算棋盘格位姿
| ├── calib.npz             --标定得到的数据
| ├── take_photos.py        --拍标定相机的图片
| └── undistort             --测试去畸变后的相机
├── package/
| ├── __pycache__/          --略
| ├── __init__.py           --使class文件夹被识别为python包
| ├── load_calib.py         --加载内参和畸变参数的代码
| ├── preprocess.py         --预处理函数
| └── undistored_camera.py  --调用摄像头并去畸变
├── find_param.py           --使用掩码识别飞镖
├── get_param.py            --通过窗口找到飞镖的HSV参数
├── hsv_range.json          --飞镖的HSV参数
├── tips.md                 --一些注意事项
├── try.py                  --AI写的识别Apriltag代码
└── use_camera              --直接调用摄像头以及一些预处理结果
```