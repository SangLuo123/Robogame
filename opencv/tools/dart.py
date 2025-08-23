import os
import sys
import cv2
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector
import json


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from src.detector import Detector
from package.load_calib import load_calib
from src.transform import rt_to_T, T_inv

def load_config(path):
    """
    读取项目配置（若没有就给默认值）
    内容建议包含：串口、tag_size、T_robot_cam、tag_map、HSV、相机ID等
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认配置（可在 data/config.json 里覆盖）
    return {
        "serial_port": "/dev/ttyUSB0",      # 用 udev 固定后的名字；没有就 /dev/ttyACM0
        "baud": 115200,
        "camera_index": 0,                # /dev/video0
        "tag_size": 120,                 # 米，内黑框边长
        "hsv_range": {"lower": [0,120,100], "upper": [10,255,255]},
        # 机器人←相机 外参（示例：正前且抬高 0.25m）
        "T_robot_cam": {
            "R": [[1,0,0],[0,1,0],[0,0,1]],
            "t": [0.0, 0.0, 0.25]
        },
        # # tag 地图：世界←tag（示例 2 个）
        # "tag_map": {
        #     "0": { "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 0.0, 0.0] },
        #     "1": { "R": rotz_deg(90).tolist(),     "t": [2.0, 0.0, 0.0] }
        # },
        # ---- 这里改为四角坐标 ----
        "tag_map": {
            "0": {  # id=0 这张码四角在世界系下的 3D 坐标（示例：贴墙，z 为高度）
                "tl": [0.00, 0.80, 0.80],
                "tr": [0.16, 0.80, 0.80],
                "br": [0.16, 0.96, 0.80],
                "bl": [0.00, 0.96, 0.80]
            },
            "1": {
                "tl": [2.00, 0.80, 0.80],
                "tr": [2.16, 0.80, 0.80],
                "br": [2.16, 0.96, 0.80],
                "bl": [2.00, 0.96, 0.80]
            }
        },
        "goal": [1.5, 0.5],               # 目标点
        "reach_tol_m": 0.10,              # 到点阈值
        "kv": 0.6,                         # 线速度比例
        "ktheta": 1.2                      # 角速度比例（deg→rad 在 car 里处理）
    }
    
def dart_cal():
    config = load_config(os.path.join(ROOT, "data", "config.json"))
    mtx, dist = load_calib(os.path.join(ROOT, "calib", "calib.npz"))
    detect = Detector(mtx, dist, 120, hsv_params=config["hsv_range"])
    img_path = os.path.join(ROOT, "img", "feibiao1.png")
    frame = cv2.imread(img_path)
    if frame is None:
        print("图片读取失败，请检查路径：", img_path)
        return
    center, cnt = detect.detect_dart(frame)
    if center is None:
        print("未检测到飞镖靶")
        return
    frame_draw = detect.draw_results(frame, [], (center, cnt))
    cv2.imshow("dart cal", frame_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    dart_cal()