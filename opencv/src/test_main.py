# src/main.py
import os
import sys
import time
import json
import math
import numpy as np
import cv2
import threading

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
from src.detect_dart import find_dart

# ========== 任务状态机 ==========

class State:
    INIT_CHECKS = 0
    INITIAL_LOCATE = 1
    GO_DART1 = 2
    GO_DART2 = 3
    ATTACK = 4
    DONE = 5
    FAIL = 6
    GRAB = 7
    
def start_heartbeat(link, interval=0.2):
    def run():
        while True:
            link.heartbeat(interval_s=interval)
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()

# ========== 一些函数 ==========

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

def get_tags(mc, det0, det1, car):
    pair = mc.get_pair_synced("cam0","cam1", max_skew_ms=60, timeout_ms=300)
    if pair is None:
        return None
    p0, p1 = pair
    raw0, raw1 = p0.image, p1.image

    tags0 = det0.detect_tags(raw0, estimate_pose=True) or []
    tags1 = det1.detect_tags(raw1, estimate_pose=True) or []
    
    all_tags = [(t, "cam0") for t in tags0] + [(t, "cam1") for t in tags1]

    if not all_tags:
        return None
    
    det, cam_id = min(
        all_tags,
        key=lambda x: float(np.linalg.norm(np.array(x[0]["pose_t"]).reshape(3)))
    )
    
    car.update_pose_from_tag(det, cam_id=cam_id)
    return {"best_cam": cam_id, "best_tag": det}

# ============== 移动相关 ==============
def attach_callbacks(link: SerialLink):
    link.on_ack   = lambda ok, f: print("ACK:", "OK" if ok else f)
    link.on_enc   = lambda enc:  print("ENC:", enc)
    link.on_text  = lambda s:    print("TXT:", s)
    link.on_fault = lambda fs:   print("FAULT:", fs)
    
class _AckWaiter:
    """
    等待一条 $A...# 回帧：
      - 不改协议，不需要 action_id
      - 不破坏你已绑定的 on_ack（会级联调用）
    """
    def __init__(self, link):
        self.link = link
        self.cv = threading.Condition()
        self.queue = []

        self._old_on_ack = link.on_ack  # 可能是 None（比如绑定了日志）

        def _hook(ok, fields):
            # 入队并唤醒等待者
            with self.cv:
                self.queue.append((ok, fields))
                self.cv.notify_all()
            # 级联旧回调
            if self._old_on_ack:
                try:
                    self._old_on_ack(ok, fields)
                except Exception:
                    pass

        link.on_ack = _hook

    def wait(self, timeout: float):
        """等待一条 ACK；返回 (ok:bool, fields:list) 或 None(超时)"""
        deadline = time.time() + float(timeout)
        with self.cv:
            while True:
                if self.queue:
                    return self.queue.pop(0)
                remain = deadline - time.time()
                if remain <= 0:
                    return None
                self.cv.wait(remain)
    
def _wrap_deg(a):
    """把角度规范到 (-180, 180]"""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def _world_vec_to_body(dx_w, dy_w, yaw_deg):
    """
    把世界系位移向量 [dx_w, dy_w] 旋转到车体系（以当前车头方向为 x 正，左为 y 正）
    等价于 R(-yaw) * [dx_w, dy_w]
    """
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    dx_b =  c * dx_w + s * dy_w
    dy_b = -s * dx_w + c * dy_w
    return dx_b, dy_b

def _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=6.0):
    w = _AckWaiter(link)
    link.send_vel_xy(dx_b, dy_b)
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def _send_rot_and_wait_ack(link, d_yaw, timeout_s=4.0):
    w = _AckWaiter(link)
    link.rotate(d_yaw)
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def go_to_ack_then_optional_tag_verify(
    car, goal, link,
    pos_tol=50.0,         # mm
    yaw_tol=5.0,          # deg
    move_timeout=6.0,
    rot_timeout=4.0,
    verify_window=0.8,    # DONE 后给这段时间扫描 tag
    max_corrections=2,    # 校验失败允许的纠正次数
    scan_fn=None          # e.g. lambda: get_tags(mc, det0, det1, car)
):
    """
    1) 计算相对车体 (dx_b,dy_b,d_yaw)，依次发送 MOVE / ROT，并等待各自 ACK
    2) 完成后尝试扫 tag：若能看到就用 car.get_pose() 与 goal 比较，达标 -> 结束；未达标 -> 再发一次移动
    3) 若扫不到 tag，则直接接受 DONE
    """
    def _rel_from_world():
        x0, y0, yaw0 = car.get_pose()          # 世界系 mm / deg
        x1, y1, yaw1 = goal
        dx_b, dy_b = _world_vec_to_body(x1 - x0, y1 - y0, yaw0)
        d_yaw = _wrap_deg(yaw1 - yaw0)
        return dx_b, dy_b, d_yaw

    def _try_verify_once():
        """
        verify_window 内多次尝试：
          - 若任何一次看到 tag（通过 scan_fn 触发 update_pose_from_tag），并发现已达标 -> True
          - 若窗口内完全没看到 tag -> None
          - 若看到 tag 但仍未达标 -> False
        """
        t0 = time.time()
        saw = False
        while time.time() - t0 < verify_window:
            if scan_fn:
                try:
                    scan_fn()  # 里面应调用 update_pose_from_tag
                except Exception:
                    pass
            x, y, yaw = car.get_pose()
            x1, y1, yaw1 = goal
            dpos = math.hypot(x1 - x, y1 - y)
            dyaw = abs(_wrap_deg(yaw1 - yaw))
            if dpos <= pos_tol and dyaw <= yaw_tol:
                return True
            time.sleep(0.05)
            saw = True
        return None if not saw else False

    corrections = 0
    while True:
        # 计算一次相对目标
        dx_b, dy_b, d_yaw = _rel_from_world()
        print(f"[CMD] rel: dx={dx_b:.1f} mm, dy={dy_b:.1f} mm, dyaw={d_yaw:.1f}°")

        # 1) 平移并等 ACK
        ok, info = _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=move_timeout)
        if not ok:
            print(f"[ERR] MOVE {info}")
            return {"ok": False, "stage": "move", "info": info}

        # 2) 旋转并等 ACK
        ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
        if not ok:
            print(f"[ERR] ROT {info}")
            return {"ok": False, "stage": "rot", "info": info}

        # 3) DONE 后尝试 tag 校验
        verdict = _try_verify_once()
        if verdict is None:
            # 没看到 tag：按你的策略，直接接受
            print("[INFO] DONE (no tag seen during verify window) -> accept.")
            return {"ok": True, "stage": "done_no_tag", "corrections": corrections}
        elif verdict is True:
            print("[OK] DONE and verified by tag.")
            return {"ok": True, "stage": "done_verified", "corrections": corrections}
        else:
            # 看到了 tag，但未达标，发一次纠正
            if corrections >= max_corrections:
                print("[WARN] Verified not at goal; correction limit reached -> stop.")
                return {"ok": False, "stage": "verify_fail", "corrections": corrections}
            corrections += 1
            print(f"[INFO] Verified not at goal; send correction #{corrections} ...")
            # while 循环继续，重新计算相对量并重发

def detect_sb(frame, roi, hsv_range, thresh=0.4, debug=False):
    """
    判断哨所指示灯是否已变色
    参数:
        frame: 输入帧 (BGR)
        roi: (x, y, w, h) ROI区域
        hsv_range: {"lower": [H,S,V], "upper": [H,S,V]} 目标颜色范围
        thresh: 判断阈值 (0~1)，ROI 区域内目标颜色像素比例超过此值判定为已变色
        debug: 是否显示调试窗口
    返回:
        bool: False=已变色（即哨所失效）, True=未变色
        float: 实际占比
    """
   # TODO: 检测哨兵
    x, y, w, h = roi
    roi_img = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_range["lower"], dtype=np.uint8)
    upper = np.array(hsv_range["upper"], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    ratio = cv2.countNonZero(mask) / (w * h)

    if debug:
        cv2.imshow("ROI", roi_img)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

    return ratio <= thresh, ratio

def attack(target, link):
    # TODO: 不同目标位姿也不一样，可以考虑在函数外把位姿调整好？
    # 转速在函数里确认
    if target == 0:
        print("[INFO] 攻击哨兵")
        link.send_shooter_rpm(1500)  # 设置转速
        time.sleep(1.0)               # 等待转速稳定
        link.shooter_fire()           # 触发发射
    elif target == 1:
        print("[INFO] 攻击大本营")
        link.send_shooter_rpm(2000)  # 设置转速
        time.sleep(1.0)               # 等待转速稳定
        link.shooter_fire()           # 触发发射
        
import time

def get_frames_burst(mc, cam_name, n=5, timeout_s=0.5, min_gap_ms=0):
    """
    从单路相机抓 n 帧（去重），超时返回已抓到的。
    min_gap_ms>0 时，保证相邻两帧的时间间隔至少这么多毫秒。
    """
    frames = []
    last_id = None
    last_ts = None
    t_end = time.time() + timeout_s

    while len(frames) < n and time.time() < t_end:
        pkt = mc.latest(cam_name)
        if pkt is None:
            time.sleep(0.005); continue

        # 去重 + 控制最小间隔
        if last_id is not None and pkt.frame_id == last_id:
            time.sleep(0.005); continue
        if min_gap_ms and last_ts is not None:
            dt_ms = (pkt.ts_ns - last_ts) / 1e6
            if dt_ms < min_gap_ms:
                time.sleep((min_gap_ms - dt_ms)/1000.0); continue

        frames.append(pkt.image)
        last_id = pkt.frame_id
        last_ts = pkt.ts_ns

        # 按估计 FPS 自适应小睡（更平滑）
        if pkt.fps and pkt.fps > 1e-3:
            time.sleep(1.0 / pkt.fps * 0.5)
        else:
            time.sleep(0.01)
    return frames

def get_dart(link, dart_id, mc, roi):
    # TODO: 还需要判断dart是否成功抓取？
    # 对于多次尝试抓取，可以试着每次加上偏移，但这样的话抓取飞镖就不是预设动作了，得和电控交流
    # 目前的流程：只尝试一次抓取，失败直接跳过
    if dart_id == 0:
        print("[INFO] 抓取常规飞镖")
        link.send_get_dart1()
        frames = get_frames_burst(mc, "cam1", n=5, timeout_s=1.0, min_gap_ms=100)
        if find_dart(frames, roi) == False:
            return True
        else:
            print("[ERR] 常规飞镖抓取失败")
            return False
    elif dart_id == 1:
        print("[INFO] 抓取战略飞镖")
        link.send_get_dart2()
        frames = get_frames_burst(mc, "cam1", n=5, timeout_s=1.0, min_gap_ms=100)
        if find_dart(frames, roi) == False:
            return True
        else:
            print("[ERR] 战略飞镖抓取失败")
            return False

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
    link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"])
    link.open()
    # start_heartbeat(link, interval=0.2) # 启动心跳线程

    # # cap = open_camera(cfg["camera_index"])
    # state = State.INIT
    # goal = cfg["goal"]

    # print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")

    # mc = MultiCam()
    # backend = cv2.CAP_DSHOW  # Windows使用DirectShow后端 # linux上直接使用默认赋值即可
    # mc.add_camera("cam0", 0,   width=640, height=480, fourcc="MJPG", backend=backend)
    # mc.add_camera("cam1", 2, width=640, height=480, fourcc="MJPG", backend=backend)
    # mc.start()
    
    # # TODO: 数据待更新
    # state = State.INIT_CHECKS
    # dart1_pos = [1, 1, 1, 1, 1]
    # dart2_pos = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # shaobing = True
    # scan_fn = lambda: get_tags(mc, det0, det1, car)
    # dart1_roi = cfg["dart1_roi"] # TODO: 待更新
    # dart2_roi = cfg["dart2_roi"] # TODO: 待更新
    # dart1_num = 3
    # dart2_num = 5
    # attach_callbacks(link)
    # led_roi = cfg["led_roi"]
    # last_state = None # TODO: 待更新
    # led_hsv_range = cfg["led_hsv_range"] # TODO: 待更新
    
    # wait_ready(mc, ["cam0", "cam1"], timeout_s=3.0)
    # print("[INFO] 两路相机已就绪")
    # link._send_bytes(b'$T#')
    
    link.send_vel_xy(500, 150)
    time.sleep(1)
    link.send_stop()
    time.sleep(1)
    link.query_encoders()
    time.sleep(1)
    link.send_shooter_rpm(1500)
    time.sleep(1)
    link.shooter_fire()
    time.sleep(1)
    link.arm_preset("GRAB")
    time.sleep(1)
    link.rotate(90)
    time.sleep(1)
    link.send_heartbeat()
    print("测试旋转90度并等待ACK...")
    res = _send_rot_and_wait_ack(link, 90, timeout_s=15.0)
    print(res)
    


# ====== 入口 ======
if __name__ == "__main__":
    main()
