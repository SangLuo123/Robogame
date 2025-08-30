# src/main.py
import os
import sys
import time
import json
import math
import numpy as np
import cv2

# --- 保证能导入 src/ 与 package/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.car import RobotCar
from src.detector import Detector
from src.comm import SerialLink
from src.load import load_calib, load_config
from src.transform import build_T_robot_cam, build_tag_map

# ========== 任务状态机 ==========

class State:
    INIT = "INIT"
    LOCATE = "LOCATE"       # 找到任意可用tag并更新位姿
    NAVIGATE = "NAVIGATE"   # 去目标点
    GRAB = "GRAB"           # 抓取（留空）
    DONE = "DONE"

# ========== 可按需实现/替换的占位函数 ==========

def choose_best_tag(detections):
    """
    从 Detector.detect_tags() 的结果里选一个最靠谱的。
    这里先选“距离最近”的那个。
    detections: [{tag_id, pose_R, pose_t, corners}, ...]
    返回: 最近的标签
    """
    if not detections:
        return None
    dets = sorted(detections, key=lambda d: float(np.linalg.norm(np.array(d["pose_t"]).reshape(3))))
    return dets[0]

def main():
    config_path = os.path.join(ROOT, "data", "config.json")
    cfg = load_config(config_path)                     # 可换成 data/config.json
    calib_path = os.path.join(ROOT, "calib", "calib.npz")

    # 1) 相机标定参数
    K, dist = load_calib(calib_path)        # 你已有的函数，返回 (mtx, dist)

    # 2) 组装外参与地图
    T_robot_cam = build_T_robot_cam(cfg) # 机器人←相机 外参
    tag_map = build_tag_map(cfg) # tag_map

    # 3) 模块初始化
    det = Detector(
        camera_matrix=K,
        dist_coeffs=dist,
        tag_size=cfg["tag_size"],
        hsv_params=cfg["hsv_range"]
    )
    car = RobotCar(T_robot_cam, tag_map)
    # link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"], binary=True)
    # link.open()

    # cap = open_camera(cfg["camera_index"])
    state = State.INIT
    goal = cfg["goal"]
    kv, ktheta = cfg["kv"], cfg["ktheta"]

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")
    t0 = time.time()
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        tags = det.detect_tags(frame)
        best = choose_best_tag(tags)
        if best:
            car.update_pose_from_tag(best)
        x, y, yaw = car.get_pose()
        cv2.putText(frame, f"POSE x:{x:.2f} y:{y:.2f} yaw:{yaw:.1f}", (10,30), 0, 0.7, (0,255,0), 2)
        if tags:
            for dete in tags:
                pts = dete["corners"].astype(int)
                cv2.polylines(frame, [pts], True, (0,255,0), 2)
                c = np.mean(pts, axis=0).astype(int)
                cv2.putText(frame, f"id:{dete['tag_id']}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    

# ====== 入口 ======
if __name__ == "__main__":
    main()
