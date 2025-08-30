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
from src.multicam import MultiCam

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

def fit_width(im, width):
    h, w = im.shape[:2]
    scale = float(width) / float(w)
    new_size = (width, max(1, int(h * scale)))
    return cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)

def wait_ready(mc, names, timeout_s=3.0):
    t0 = time.time()
    ready = {n: False for n in names}
    while time.time() - t0 < timeout_s:
        for n in names:
            if not ready[n] and mc.latest(n) is not None:
                print(f"[OK] {n} produced first frame")
                ready[n] = True
        if all(ready.values()):
            return True
        time.sleep(0.02)
    for n, ok in ready.items():
        if not ok:
            print(f"[ERR] {n} no frame within {timeout_s}s")
    return False

def undistort_crop_via_cache(det, frame_bgr):
    cache = getattr(det, "_ud_cache", None)
    if cache is None:
        det.detect_tags(frame_bgr, estimate_pose=False)
        cache = det._ud_cache
    map1, map2 = cache["map1"], cache["map2"]
    x0, y0, ww, hh = cache["valid_roi"]
    undist_full = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)
    return undist_full[y0:y0+hh, x0:x0+ww]

def main():
    config_path = os.path.join(ROOT, "data", "config.json")
    cfg = load_config(config_path)                     # 可换成 data/config.json
    calibup_path = os.path.join(ROOT, "calib", "calibup.npz")
    calibdown_path = os.path.join(ROOT, "calib", "calibdown.npz")

    # 1) 相机标定参数
    K0, dist0 = load_calib(calibup_path)        # 你已有的函数，返回 (mtx, dist)
    K1, dist1 = load_calib(calibdown_path)      # 你已有的函数，返回 (mtx, dist)

    # 2) 组装外参与地图
    T_robot_cam_0 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam0"])
    T_robot_cam_1 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam1"])
    tag_map = build_tag_map(cfg) # tag_map

    # 3) 模块初始化
    det0 = Detector(
        camera_matrix=K0,
        dist_coeffs=dist0,
        tag_size=cfg["tag_size"],
        hsv_params=cfg["hsv_range"]
    )
    det1 = Detector(
        camera_matrix=K1,
        dist_coeffs=dist1,
        tag_size=cfg["tag_size"],
        hsv_params=cfg["hsv_range"]
    )
    car = RobotCar(tag_map)
    car.set_camera_extrinsic("cam0", T_robot_cam_0)
    car.set_camera_extrinsic("cam1", T_robot_cam_1)
    # link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"], binary=True)
    # link.open()

    # cap = open_camera(cfg["camera_index"])
    state = State.INIT
    goal = cfg["goal"]
    kv, ktheta = cfg["kv"], cfg["ktheta"]

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")
    t0 = time.time()

    mc = MultiCam()
    backend = cv2.CAP_DSHOW  # Windows使用DirectShow后端 # linux上直接使用默认赋值即可
    mc.add_camera("cam0", 0,   width=640, height=480, fourcc="MJPG", backend=backend)
    mc.add_camera("cam1", 2, width=640, height=480, fourcc="MJPG", backend=backend)
    mc.start()
    

    try:
        # 等待两路都出第一帧（快速定位设备问题）
        wait_ready(mc, ["cam0", "cam1"], timeout_s=3.0)

        while True:
            pair = mc.get_pair_synced("cam0","cam1", max_skew_ms=60, timeout_ms=300)
            if pair is None:
                # 打印节流诊断，略
                continue

            p0, p1 = pair
            raw0, raw1 = p0.image, p1.image

            # ---- cam0 ----
            tags0 = det0.detect_tags(raw0, estimate_pose=True)
            best0 = choose_best_tag(tags0)
            if best0:
                car.update_pose_from_tag(best0, cam_id="cam0")  # ★ 关键：告诉是 cam0
            vis0 = undistort_crop_via_cache(det0, raw0)
            if tags0:
                for d in tags0:
                    pts = d["corners"].astype(int)
                    cv2.polylines(vis0, [pts], True, (0,255,0), 2)
                    c = np.mean(pts, axis=0).astype(int)
                    cv2.putText(vis0, f"id:{d['tag_id']}", tuple(c), 0, 0.6, (0,255,0), 2)

            # ---- cam1 ----
            tags1 = det1.detect_tags(raw1, estimate_pose=True)
            best1 = choose_best_tag(tags1)
            if best1:
                car.update_pose_from_tag(best1, cam_id="cam1")  # ★ 关键：告诉是 cam1
            vis1 = undistort_crop_via_cache(det1, raw1)
            if tags1:
                for d in tags1:
                    pts = d["corners"].astype(int)
                    cv2.polylines(vis1, [pts], True, (0,255,0), 2)
                    c = np.mean(pts, axis=0).astype(int)
                    cv2.putText(vis1, f"id:{d['tag_id']}", tuple(c), 0, 0.6, (0,255,0), 2)

            # 显示车姿态（你已有）
            x, y, yaw = car.get_pose()
            for im in (vis0, vis1):
                cv2.putText(im, f"POSE x:{x:.2f} y:{y:.2f} yaw:{yaw:.1f}",
                            (10,30), 0, 0.8, (0,255,0), 2)

            cv2.imshow("cam0 | tag", vis0)
            cv2.imshow("cam1 | tag", vis1)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

    finally:
        mc.stop()
        cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("[ERROR] 无法打开摄像头")
    #     return
    # while True:
    #     ok, frame = cap.read()
    #     if not ok:
    #         time.sleep(0.01)
    #         continue
    #     tags = det.detect_tags(frame)
    #     best = choose_best_tag(tags)
    #     if best:
    #         car.update_pose_from_tag(best)
    #     x, y, yaw = car.get_pose()
    #     cv2.putText(frame, f"POSE x:{x:.2f} y:{y:.2f} yaw:{yaw:.1f}", (10,30), 0, 0.7, (0,255,0), 2)
    #     if tags:
    #         for dete in tags:
    #             pts = dete["corners"].astype(int)
    #             cv2.polylines(frame, [pts], True, (0,255,0), 2)
    #             c = np.mean(pts, axis=0).astype(int)
    #             cv2.putText(frame, f"id:{dete['tag_id']}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    #     cv2.imshow("Live", frame)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:
    #         break

# ====== 入口 ======
if __name__ == "__main__":
    main()
