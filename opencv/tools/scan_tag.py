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

def tag(img, calib_path, tag_families="tag36h11", hsv_params=None, tag_size=120):
    mtx, dist_coeffs = load_calib(calib_path)
    detector = Detector(mtx, dist_coeffs, tag_size, tag_families, hsv_params)
    detections = detector.detect_tags(img)
    if not detections:
        return []
    frame = detector.draw_results(img, detections)
    frame = cv2.resize(frame, (640, 480))  # 调整显示大小
    cv2.imshow("Detected Tags", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections

import math
import numpy as np

def euler_zyx_from_R(R: np.ndarray):
    """
    将 3x3 旋转矩阵 R 转成 ZYX 欧拉角 (yaw, pitch, roll)，单位：弧度
    约定（OpenCV相机坐标）：X右、Y下、Z前
      yaw   = 绕 Z 轴（前）旋转（左右转头）
      pitch = 绕 Y 轴（下）旋转（点头）
      roll  = 绕 X 轴（右）旋转（歪头）
    """
    # 防数值噪声
    R = np.asarray(R, dtype=float)
    # ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # 参考推导： 
    # pitch = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
    # yaw   = atan2(R[1,0], R[0,0])
    # roll  = atan2(R[2,1], R[2,2])
    sy = -R[2, 0]
    cy = math.sqrt(max(0.0, 1.0 - sy*sy))  # = sqrt(R[2,1]^2 + R[2,2]^2)

    yaw   = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(sy, cy)
    roll  = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll

def rad2deg_tuple(t):
    return tuple(np.degrees(ti) for ti in t)

#####################################################

def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def tag_T_from_world_corners(tl, tr, br, bl):
    """
    输入四个角点在世界坐标系下的 3D 坐标（单位 m），顺序：tl, tr, br, bl
    Tag 局部系约定:
      - 原点: Tag 中心
      - x 轴: tl->tr（上边向右）
      - y 轴: tl->bl（左边向下）
      - z 轴: x × y（右手系，指向相机/法线）
    返回 世界<-Tag 的 4x4 齐次矩阵
    """
    tl = np.asarray(tl, float)
    tr = np.asarray(tr, float)
    br = np.asarray(br, float)
    bl = np.asarray(bl, float)

    center = 0.25 * (tl + tr + br + bl)

    x_w = _normalize(tr - tl)
    y0  = _normalize(bl - tl)
    # 正交化 y，防止量测误差导致不垂直
    y_w = _normalize(y0 - np.dot(y0, x_w) * x_w)
    z_w = _normalize(np.cross(x_w, y_w))  # 右手系

    R_wt = np.column_stack((x_w, y_w, z_w))  # world <- tag
    T_wt = np.eye(4, dtype=float)
    T_wt[:3, :3] = R_wt
    T_wt[:3, 3]  = center
    return T_wt

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

def build_tag_map(cfg):
    tag_map = {}
    for k, v in cfg["tag_map"].items():
        tag_id = int(k)
        T = tag_T_from_world_corners(v["tl"], v["tr"], v["br"], v["bl"])
        tag_map[tag_id] = T
    return tag_map

def to_T(R, t):
    R = np.asarray(R, float).reshape(3,3)
    t = np.asarray(t, float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def yaw_of_T(T):
    R = T[:3,:3]
    return math.atan2(R[1,0], R[0,0])  # rad

def world_robot_from_detection(det: dict, tag_map: dict, T_robot_cam: np.ndarray):
    """
    根据一次 tag 检测 + 地图 + 外参，求 机器人在世界系的位姿。
    输入:
      det: {'tag_id','pose_R','pose_t', ...}  (camera<-tag)
      tag_map: {id: T_wt}  世界<-tag
      T_robot_cam: 机器人<-相机 的 4x4
    返回:
      T_wr: 世界<-机器人 的 4x4
      (x, y, yaw_deg): 机器人在世界系下的平面位姿
    """
    tag_id = int(det["tag_id"])
    if tag_id not in tag_map:
        raise KeyError(f"tag_map 缺少 id={tag_id}")

    # 相机<-Tag
    T_ct = to_T(det["pose_R"], det["pose_t"])
    # Tag<-相机
    T_tc = T_inv(T_ct)
    # 相机<-机器人  (T_cr = (T_rc)^-1)
    T_cam_robot = T_inv(T_robot_cam)
    # Tag<-机器人
    T_tr = T_tc @ T_cam_robot
    # 世界<-Tag
    T_wt = tag_map[tag_id]
    # 世界<-机器人
    T_wr = T_wt @ T_tr

    x, y = float(T_wr[0,3]), float(T_wr[1,3])
    yaw_deg = math.degrees(yaw_of_T(T_wr))
    return T_wr, (x, y, yaw_deg)

if __name__ == "__main__":
    img = cv2.imread(os.path.join(ROOT, "img", "changgui.jpg"))
    calib_path = os.path.join(ROOT, "calib", "calib.npz")
    det = tag(img, calib_path)
    if det:
        print("Detections found:")
        for d in det:
            print(d)  # 原始字典
            # ---- 取出旋转矩阵并转欧拉角（度） ----
            R = np.array(d["pose_R"], dtype=float)
            yaw, pitch, roll = euler_zyx_from_R(R)
            yaw_d, pitch_d, roll_d = rad2deg_tuple((yaw, pitch, roll))
            print(f"Euler ZYX (deg): yaw={yaw_d:.2f}, pitch={pitch_d:.2f}, roll={roll_d:.2f}")

            # 可选：距离（米）
            t = np.array(d["pose_t"], dtype=float).reshape(3)
            dist = np.linalg.norm(t)
            print(f"Distance: {dist:.3f} m\n")
    else:
        print("No detections found.")
    cfg = load_config(os.path.join(ROOT, "data", "config.json"))
    tag_map = build_tag_map(cfg)
    # 机器人←相机 外参（4x4）
    T_robot_cam = rt_to_T(cfg["T_robot_cam"]["R"], cfg["T_robot_cam"]["t"])
    print(tag_map)

    # 若刚才检测到了 det，这里计算“机器人在世界系”的位姿
    if det:
        print("=== Robot pose from detections ===")
        for d in det:
            try:
                T_wr, (rx, ry, ryaw) = world_robot_from_detection(d, tag_map, T_robot_cam)
                print(f"tag {d['tag_id']}:  x={rx:.3f}  y={ry:.3f}  yaw={ryaw:.1f}°")
            except KeyError as e:
                print(f"[WARN] {e}")