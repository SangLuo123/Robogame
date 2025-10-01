# Robogame2025
This project is used for USTC Robogame2025.
文件结构暂未更新...
## Files
```
流程/
├── 比赛流程.drawio         --比赛流程
├── 点位.md                 --一些移动的点位
opencv/
├── calib/
| ├── imgdown/              --标定旧相机用的图片
| ├── imgup/                --标定新相机用的图片
| ├── calc.py               --计算内参矩阵和畸变参数并调用摄像头根据计算结果实时测算棋盘格位姿
| ├── calibdown.npz         --标定得到的数据
| ├── calibup.npz           --标定得到的数据
| ├── findChess.py          --测试找棋盘格
| ├── take_photos.py        --拍标定相机的图片
| └── undistort.py          --测试去畸变后的相机
├── data/
| ├── config.json           --配置参数
| ├── config1.json          --以一个相机为中心配置参数
| ├── hsv_range.json        --飞镖的HSV参数
| ├── tips.md               --注意事项
├── img/                    --图片
├── package/
| ├── __pycache__/
| ├── __init__.py           
| ├── load_calib.py         --加载内参和畸变参数的代码
| ├── preprocess.py         --预处理函数
| └── undistored_camera.py  --调用摄像头并去畸变
├── src/
| ├── __pycache__/
| ├── car.py                --小车的类
| ├── comm.py               --与下位机交流的类以及通信协议
| ├── demo_twocam.py        --测试两个相机
| ├── detect_dart.py        --根据传入的ROI自动分析飞镖的HSV范围并计算ROI中目标飞镖的面积占比决定是否有飞镖
| ├── detector.py           --检测器的类
| ├── load.py               --加载配置的函数
| ├── main1.py              --主流程函数
| ├── main2.py              --三审2分流程
| ├── main3.py              --以一个相机为中心三审2分流程x
| ├── main4.py              --以一个相机为中心
| ├── main5.py              --只开一个相机，三审，全新流程，飞镖贴tag
| ├── main7.py              --只开一个相机，三审，拿全部常规
| ├── main8.py              --只开一个相机，三审，新流程
| ├── multicam.py           --多相机管理函数
| ├── new_main1.py          --三审撞墙流程
| ├── new_main2.py          --预赛撞墙流程（保下限）
| ├── one_cam_tag.py        --一个相机测试主函数
| ├── sim_main.py           --调用一个摄像头实时扫描tag
| ├── sim_main2.py          --调用两个摄像头实时扫描tag
| ├── simu_stm32.py         --模拟stm32进行通信
| ├── test_main.py          --测试通信
| ├── test.py               --测试通信
| ├── TODO.md               --测试待完成事项
| ├── transform.py          --坐标变换时用到的一些矩阵相关的函数
| └── two_cam_tag.py        --两个相机测试主函数
├── tools/      
| ├── img/                  --临时图片        
| ├── find_obj.py           --使用掩码识别飞镖
| ├── get_param.py          --通过窗口找到飞镖的HSV参数（窗口中图像已经去畸变以及预处理）
| ├── mark.py               --提取ROI且根据掩码提取
| ├── ori_photo             --使用原相机拍照并存入img文件夹
| ├── photo.py              --去畸变后的相机并拍照
| ├── roi.py                --找ROI工具
| ├── scan_tag.py           --识别tag
| ├── scan_tag1.py          --识别tag，并重投影自动排序计算误差
| └── use_camera.py         --直接调用摄像头以及一些预处理结果
```
