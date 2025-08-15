# src/main.py
import os
import sys
import time
import json
import numpy as np
import cv2

# --- 保证能导入 src/ 与 package/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.car import RobotCar
from src.detector import Detector
from src.comm import SerialLink
from package.load_calib import load_calib  # 你已有
from src.transform import rt_to_T
# ========== 配置 & 初始化相关 ==========

def load_config(path="data/config.json"):
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
        "tag_size": 0.16,                 # 米，内黑框边长
        "hsv_range": {"lower": [0,120,100], "upper": [10,255,255]},
        # 机器人←相机 外参（示例：正前且抬高 0.25m）
        "T_robot_cam": {
            "R": [[1,0,0],[0,1,0],[0,0,1]],
            "t": [0.0, 0.0, 0.25]
        },
        # tag 地图：世界←tag（示例 2 个）
        "tag_map": {
            "0": { "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 0.0, 0.0] },
            "1": { "R": rotz_deg(90).tolist(),     "t": [2.0, 0.0, 0.0] }
        },
        "goal": [1.5, 0.5],               # 目标点
        "reach_tol_m": 0.10,              # 到点阈值
        "kv": 0.6,                         # 线速度比例
        "ktheta": 1.2                      # 角速度比例（deg→rad 在 car 里处理）
    }

def rotz_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def build_T_robot_cam(cfg):
    R = cfg["T_robot_cam"]["R"]
    t = cfg["T_robot_cam"]["t"]
    return rt_to_T(R, t)

def build_tag_map(cfg):
    tag_map = {}
    for k, v in cfg["tag_map"].items():
        tag_id = int(k)
        T = rt_to_T(v["R"], v["t"])
        tag_map[tag_id] = T
    return tag_map

def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头 {index} 打不开")
    return cap

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

def reached_goal(car, goal, tol_m):
    x, y, _ = car.get_pose()
    return np.hypot(goal[0]-x, goal[1]-y) < tol_m

def do_grab(detector, frame):
    """
    抓取阶段的视觉动作（示例：找飞镖）
    实际抓取控制由 STM32 实现，这里只负责目标检测/定位 → 返回抓取点
    """
    # 示例：先留空
    return None

# ========== 主程序 ==========

def main():
    cfg = load_config()                     # 可换成 data/config.json
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
    link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"], binary=True)
    link.open()

    cap = open_camera(cfg["camera_index"])
    state = State.INIT
    goal = cfg["goal"]
    kv, ktheta = cfg["kv"], cfg["ktheta"]

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")
    t0 = time.time()

# TODO: 需要更新
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # ========== 视觉检测 ==========
            tags = det.detect_tags(frame)   # list of dict
            best = choose_best_tag(tags)
            if best:
                # 假装一个“对象”传给 car（兼容之前的 update_pose_from_tag 接口）
                dummy = type("D", (), {})() # 一个空对象，“D”是一个类
                dummy.tag_id = best["tag_id"]
                dummy.pose_R = best["pose_R"]
                dummy.pose_t = best["pose_t"]
                car.update_pose_from_tag(dummy)

            # ========== 状态机 ==========
            if state == State.INIT:
                state = State.LOCATE

            elif state == State.LOCATE:
                # 看到任意 tag 即可完成定位
                if best:
                    state = State.NAVIGATE

            elif state == State.NAVIGATE:
                # 去目标点
                v, w = car.compute_control_to_target(goal[0], goal[1], kv=kv, ktheta=ktheta)
                # 简单限幅（可移到 comm 里）
                v = float(max(-0.6, min(0.6, v)))
                w = float(max(-2.5, min(2.5, w)))
                link.send_vel(v, w)

                if reached_goal(car, goal, cfg["reach_tol_m"]):
                    link.send_vel(0.0, 0.0)
                    state = State.GRAB
            
            elif state == State.GRAB:
                # 视觉定位抓取对象（占位）
                grab_target = do_grab(det, frame)
                # TODO: 通过串口发抓取命令/位姿给 STM32（未实现）
                # link.send_command("GRAB ...")
                state = State.DONE

            elif state == State.DONE:
                link.send_vel(0.0, 0.0)
                # 这里可以进入下一任务或者直接停
                pass

            # 兜底心跳（主循环版）
            link.heartbeat(interval_s=0.2)

            # ========== HUD 可视化 ==========
            x, y, yaw = car.get_pose()
            cv2.putText(frame, f"STATE:{state}", (10,30), 0, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"POSE x:{x:.2f} y:{y:.2f} yaw:{yaw:.1f}", (10,60), 0, 0.7, (0,255,0), 2)
            if goal:
                cv2.circle(frame, (30,30), 3, (0,255,255), -1)
            cv2.imshow("Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        link.send_vel(0.0, 0.0)
        link.close()
        cap.release()
        cv2.destroyAllWindows()

# ====== 入口 ======
if __name__ == "__main__":
    main()
