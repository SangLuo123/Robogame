import os
import sys
import cv2
import time
import math
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.detector import Detector
from package.load_calib import load_calib
from src.transform import rt_to_T, T_inv

# -----------工具与几何-----------------

def _objp_from_tag_size(tag_size):
    """以Tag中心为原点的四角3D坐标（单位与tag_size一致），顺序: tl,tr,br,bl"""
    s = float(tag_size) / 2.0
    return np.array([[-s,-s,0], [s,-s,0], [s,s,0], [-s,s,0]], dtype=np.float32)

def reprojection_stats_for_detection(det, K, dist, tag_size):
    """
    计算一次检测的重投影误差统计（像素）。
    约定：若帧已去畸变，则传 dist=None 且 K=K_new。
    """
    R = np.asarray(det["pose_R"], float).reshape(3,3)
    t = np.asarray(det["pose_t"], float).reshape(3,)
    imgp = np.asarray(det["corners"], np.float32).reshape(-1,2)
    objp = _objp_from_tag_size(tag_size)

    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(objp, rvec, t.reshape(3,1), K, dist)
    proj = proj.reshape(-1,2)

    per = np.linalg.norm(proj - imgp, axis=1)
    mean_err = float(per.mean())
    rms_err  = float(np.sqrt((per**2).mean()))
    max_err  = float(per.max())
    return {
        "per_point_err": per,
        "mean_err": mean_err,
        "rms_err": rms_err,
        "max_err": max_err,
        "proj": proj
    }

def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def tag_T_from_world_corners(tl, tr, br, bl):
    """
    输入四个角点在世界坐标系下的 3D 坐标（单位 m），顺序：tl, tr, br, bl
    Tag 局部系：
      - 原点: Tag 中心
      - x 轴: tl->tr（上边向右）
      - y 轴: tl->bl（左边向下）
      - z 轴: x × y（右手系）
    返回 世界<-Tag 的 4x4 齐次矩阵
    """
    tl = np.asarray(tl, float)
    tr = np.asarray(tr, float)
    br = np.asarray(br, float)
    bl = np.asarray(bl, float)

    center = 0.25 * (tl + tr + br + bl)

    x_w = _normalize(tr - tl)
    y0  = _normalize(bl - tl)
    y_w = _normalize(y0 - np.dot(y0, x_w) * x_w)  # 正交化
    z_w = _normalize(np.cross(x_w, y_w))          # 右手系

    R_wt = np.column_stack((x_w, y_w, z_w))  # world <- tag
    T_wt = np.eye(4, dtype=float)
    T_wt[:3, :3] = R_wt
    T_wt[:3, 3]  = center
    return T_wt

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

# ========== 配置 & 初始化相关 ==========

def load_config(path="data/config.json"):
    """
    读取项目配置（若没有就给默认值）
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认配置（可在 data/config.json 里覆盖）
    return {
        "serial_port": "/dev/ttyUSB0",
        "baud": 115200,
        "camera_index": 0,                # /dev/video0
        "tag_size": 120,                  # 毫米，内黑框边长（与你 Detector 保持一致）
        "hsv_range": {"lower": [0,120,100], "upper": [10,255,255]},
        # 机器人←相机 外参（示例）
        "T_robot_cam": {
            "R": [[1,0,0],[0,1,0],[0,0,1]],
            "t": [0.0, 0.0, 0.0]
        },
        # tag 地图（世界<-tag）用四角坐标构造
        "tag_map": {
            "0": {"tl":[0.00,0.80,0.80], "tr":[0.16,0.80,0.80],
                  "br":[0.16,0.96,0.80], "bl":[0.00,0.96,0.80]},
            "1": {"tl":[2.00,0.80,0.80], "tr":[2.16,0.80,0.80],
                  "br":[2.16,0.96,0.80], "bl":[2.00,0.96,0.80]}
        },
        # 实时融合参数
        "px_thresh": 2.5                 # 每个 tag 的平均重投影误差阈值（像素）
    }

def build_tag_map(cfg):
    tag_map = {}
    for k, v in cfg["tag_map"].items():
        tag_id = int(k)
        T = tag_T_from_world_corners(v["tl"], v["tr"], v["br"], v["bl"])
        tag_map[tag_id] = T
    return tag_map

# ========== 实时检测与显示 ==========

def _format_pose_text(x, y, yaw_deg):
    return f"x={x:.3f} m  y={y:.3f} m  yaw={yaw_deg:.1f}deg"

def _draw_lines(img, lines, org=(10,30), lh=24, color=(50,220,50)):
    x0, y0 = org
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x0, y0 + i*lh), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def _fuse_poses_by_weight(pose_list):
    """
    pose_list: list of (x, y, yaw_deg, w)
    返回融合 (x,y,yaw_deg)，若为空返回 None
    """
    if not pose_list:
        return None
    wx = wy = 0.0
    cx = cy = 0.0
    csum = 0.0
    ssum = 0.0
    for x,y,yaw_deg,w in pose_list:
        wx += w * x
        wy += w * y
        csum += w * math.cos(math.radians(yaw_deg))
        ssum += w * math.sin(math.radians(yaw_deg))
        cx += w
        cy += w
    if cx <= 1e-9:
        return None
    x_f = wx / cx
    y_f = wy / cy
    yaw_f = math.degrees(math.atan2(ssum, csum))
    return (x_f, y_f, yaw_f)

def run_live_from_camera():
    # --- 加载配置与标定 ---
    cfg = load_config(os.path.join(ROOT, "data", "config.json"))
    tag_size = cfg["tag_size"]          # 与 Detector 的期望保持一致（你之前用的是“内黑框边长（mm）”）
    tag_map  = build_tag_map(cfg)
    T_robot_cam = rt_to_T(cfg["T_robot_cam"]["R"], cfg["T_robot_cam"]["t"])

    calib_path = os.path.join(ROOT, "calib", "calib.npz")
    K, dist = load_calib(calib_path)

    # --- 打开摄像头 ---
    cam = cv2.VideoCapture(cfg.get("camera_index", 0))
    if not cam.isOpened():
        print("ERROR: 无法打开摄像头")
        return

    # 先读一帧以获尺寸
    ok, frame0 = cam.read()
    if not ok:
        print("ERROR: 读取首帧失败")
        return
    h, w = frame0.shape[:2]

    # --- 去畸变矩阵与映射 ---
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w,h), cv2.CV_16SC2)

    # --- 用去畸变后的内参构建 Detector（dist 在检测阶段不使用）---
    detector = Detector(K_new, None, tag_size, tag_families="tag36h11", hsv_params=None)

    # --- 主循环 ---
    t_last = time.time()
    fps = 0.0
    px_thresh = float(cfg.get("px_thresh", 2.5))

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        # 去畸变
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # 检测 tags
        detections = detector.detect_tags(undist)

        # 可视化 tag 轮廓/坐标轴（如果你的 Detector 有 draw_results）
        vis = detector.draw_results(undist.copy(), detections) if hasattr(detector, "draw_results") else undist.copy()

        # 为每个 tag 计算重投影误差并转世界位姿，筛选与加权
        fused_inputs = []   # (x, y, yaw_deg, w)
        debug_lines = []
        best_single = None  # (mean_err, (x,y,yaw), tag_id)

        for d in detections or []:
            try:
                stats = reprojection_stats_for_detection(d, K_new, None, tag_size)
            except Exception:
                # 如果你的 tag_size 单位与 pose_t 单位不一致，这里可能会爆
                continue

            try:
                _, (rx, ry, ryaw) = world_robot_from_detection(d, tag_map, T_robot_cam)
            except KeyError:
                # 地图没有这个 tag_id
                debug_lines.append(f"[id {d.get('tag_id')}] SKIP: not in map")
                continue

            mean_err = stats["mean_err"]
            debug_lines.append(f"[id {d.get('tag_id')}] mean_err={mean_err:.2f}px  pose=({_format_pose_text(rx,ry,ryaw)})")

            # 记录用于“若全被筛掉时的回退”
            if (best_single is None) or (mean_err < best_single[0]):
                best_single = (mean_err, (rx, ry, ryaw), int(d.get("tag_id", -1)))

            if mean_err <= px_thresh:
                wgt = 1.0 / (mean_err + 1e-6)
                fused_inputs.append((rx, ry, ryaw, wgt))

        # 融合/回退
        if fused_inputs:
            pose = _fuse_poses_by_weight(fused_inputs)
            pose_src = f"fused {len(fused_inputs)} tags (th={px_thresh:.1f}px)"
        elif best_single is not None:
            pose = best_single[1]
            pose_src = f"best single id={best_single[2]}  err={best_single[0]:.2f}px"
        else:
            pose = None
            pose_src = "no valid detection"

        # 叠加文本
        lines = []
        # FPS
        t_now = time.time()
        dt = t_now - t_last
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt)
        t_last = t_now
        lines.append(f"FPS: {fps:.1f}")

        if pose is not None:
            x,y,yaw = pose
            lines.append(_format_pose_text(x,y,yaw))
            lines.append("source: " + pose_src)
        else:
            lines.append("(---)")
            lines.append("source: " + pose_src)

        # 每个 tag 的调试行
        if debug_lines:
            lines.append("---- tags ----")
            lines.extend(debug_lines[:6])  # 避免太占屏

        _draw_lines(vis, lines, org=(10, 28), lh=24, color=(40, 240, 40))

        cv2.imshow("AprilTag Live (undistorted)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # Esc / q
            break

    cam.release()
    cv2.destroyAllWindows()

# ========== 入口 ==========

if __name__ == "__main__":
    run_live_from_camera()
