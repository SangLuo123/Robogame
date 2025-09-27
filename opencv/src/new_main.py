# src/main.py
import os
import sys
import time
import json
import math
import numpy as np
import cv2
import threading
from enum import Enum, auto
import apriltag

# --- 保证能导入 src/ 与 package/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.car import RobotCar
from src.detector import Detector
from src.comm import SerialLink
from src.load import load_calib, load_config
from src.transform import build_T_robot_cam, build_tag_map
from src.multicam import MultiCam, _CamWorker
from src.detect_dart import find_dart
from src.camera import Camera

# ========== 任务状态机 ==========

class State(Enum):
    INIT_CHECKS = auto()
    INITIAL_LOCATE = auto()
    GO_DART1 = auto()
    GO_DART2 = auto()
    ATTACK1 = auto()
    ATTACK2 = auto()
    DONE = auto()
    FAIL = auto()

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
    # 什么时候更新位姿？
    # init_locate：刚开始更新位姿
    # 每次移动结束会更新
    # TODO: 下楼梯需不需要用到？
    # 需不需要考虑车的z坐标？感觉应该不用考虑，不考虑唯一有问题的是斜着，即上下楼梯时的位姿，但上下楼梯不需要测算位姿，
    # 故直接不考虑z坐标，将小车中心定位小车底座中心对应的地面
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
    x0, y0, yaw0 = car.get_pose()
    print(f"[INFO] 通过 {cam_id} 识别到标签 {det['tag_id']}，更新位姿为 x={x0:.1f} mm, y={y0:.1f} mm, yaw={yaw0:.1f}°")
    return {"best_cam": cam_id, "best_tag": det}

# ============== 移动相关 ==============
def attach_callbacks(link: SerialLink):
    link.on_ack   = lambda ok, f: print("ACK:", "OK" if ok else f)
    link.on_enc   = lambda ok, f:  print("ENC:", "OK" if ok else f)
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

        self._old_on_enc = link.on_enc  # 可能是 None（比如绑定了日志）

        def _hook(ok, fields):
            # 入队并唤醒等待者
            with self.cv:
                self.queue.append((ok, fields))
                self.cv.notify_all()
            # 级联旧回调
            if self._old_on_enc:
                try:
                    self._old_on_enc(ok, fields)
                except Exception:
                    pass
# $EOK#    $EERR# 
        link.on_enc = _hook

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
    # TODO: 逻辑？
    """
    把世界系位移向量 [dx_w, dy_w] 旋转到车体系（以当前车头方向为 x 正，左为 y 正）
    等价于 R(-yaw) * [dx_w, dy_w]
    """
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    dx_b =  c * dx_w + s * dy_w
    dy_b = -s * dx_w + c * dy_w
    return dx_b, dy_b

def _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=8.0):
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

def _send_grab(link, dart, timeout_s=25.0):
    w = _AckWaiter(link)
    link.arm_preset(dart)
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def _send_fire(link, rpm, timeout_s=8.0):
    w = _AckWaiter(link)
    link.send_shooter_rpm(rpm)
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
    max_corrections=1,    # 校验失败允许的纠正次数
    scan_fn=None,         # e.g. lambda: get_tags(mc, det0, det1, car)
    strategy="move_direct_then_rot" # 默认策略
):
    """
    1) 计算相对车体 (dx_b,dy_b,d_yaw)，依次发送 MOVE / ROT，并等待各自 ACK
    2) 完成后尝试扫 tag：若能看到就用 car.get_pose() 与 goal 比较，达标 -> 结束；未达标 -> 再发一次移动
    3) 若扫不到 tag，则直接接受 DONE
    """
    """
    strategy 可选值：
      - "rot_then_move_direct"  : 先旋转，再一次性平移(dx, dy)
      - "rot_then_move_x_then_y": 先旋转，再平移x，再平移y
      - "rot_then_move_y_then_x": 先旋转，再平移y，再平移x
      - "move_direct_then_rot"  : 一次性平移(dx, dy)，最后旋转
      - "move_x_then_y_then_rot": 先平移x，再平移y，最后旋转
      - "move_y_then_x_then_rot": 先平移y，再平移x，最后旋转
    这里的dx和dy均为车体系
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

        # === 根据 strategy 执行 ===
        if strategy == "rot_then_move_direct":
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}
            ok, info = _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=move_timeout)
            if not ok: return {"ok": False, "stage": "move", "info": info}

        elif strategy == "rot_then_move_x_then_y":
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}
            if abs(dx_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, dx_b, 0, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_x", "info": info}
            if abs(dy_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, 0, dy_b, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_y", "info": info}

        elif strategy == "rot_then_move_y_then_x":
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}
            if abs(dy_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, 0, dy_b, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_y", "info": info}
            if abs(dx_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, dx_b, 0, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_x", "info": info}

        elif strategy == "move_direct_then_rot":
            ok, info = _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=move_timeout)
            if not ok: return {"ok": False, "stage": "move", "info": info}
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}

        elif strategy == "move_x_then_y_then_rot":
            if abs(dx_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, dx_b, 0, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_x", "info": info}
            if abs(dy_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, 0, dy_b, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_y", "info": info}
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}

        elif strategy == "move_y_then_x_then_rot":
            if abs(dy_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, 0, dy_b, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_y", "info": info}
            if abs(dx_b) > 1.0:
                ok, info = _send_move_and_wait_ack(link, dx_b, 0, timeout_s=move_timeout)
                if not ok: return {"ok": False, "stage": "move_x", "info": info}
            ok, info = _send_rot_and_wait_ack(link, d_yaw, timeout_s=rot_timeout)
            if not ok: return {"ok": False, "stage": "rot", "info": info}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # DONE 后尝试 tag 校验
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
                return {"ok": False, "stage": "verify_fail", "corrections": corrections}
            corrections += 1
            print(f"[INFO] Verified not at goal; send correction #{corrections} ...")
            # while 循环继续，重新计算相对量并重发

def detect_sb(mc, camera_name="cam0", roi=None, flash_duration=5, flashes_per_sec=3):
    """
    检测哨所当前是否被打击一次（即闪烁模式：1s内闪烁三下，共5s）。
    
    :param mc: MultiCam实例，多摄像头管理器
    :param camera_name: 摄像头名称，默认为"cam0"
    :param roi: 灯的感兴趣区域 (tuple: (x, y, w, h))
    :param flash_duration: 闪烁总时长 (s)
    :param flashes_per_sec: 1s内闪烁次数
    :return: bool - 是否检测到一次完整的闪烁（打击）
    """
    if not mc or camera_name not in mc.workers:
        print(f"错误: 未找到摄像头 '{camera_name}'")
        return False
    
    # 检测参数
    brightness_threshold = 150  # 亮度阈值（根据实际灯调整）
    min_flash_count = flashes_per_sec  # 1秒内最少闪烁次数
    flash_window = 1.0  # 闪烁检测时间窗口(秒)
    
    # 状态变量
    flash_history = []  # 记录闪烁时间戳
    last_brightness = None
    start_time = time.time()
    detection_timeout = flash_duration + 2.0  # 检测超时时间
    
    print(f"开始检测哨所打击，超时时间: {detection_timeout}秒")
    
    while time.time() - start_time < detection_timeout:
        # 获取最新帧
        frame_packet = mc.latest(camera_name)
        if frame_packet is None:
            time.sleep(0.01)
            continue
        
        frame = frame_packet.image
        
        # 裁剪ROI区域
        if roi:
            x, y, w, h = roi
            # 确保ROI在图像范围内
            h, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h - 1))
            w = max(1, min(w, w_frame - x))
            h = max(1, min(h, h - y))
            region = frame[y:y+h, x:x+w]
        else:
            region = frame
        
        # 计算区域平均亮度
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        current_time = time.time()
        
        # 检测亮度变化（闪烁）
        if last_brightness is not None:
            # 检测从亮到暗或从暗到亮的变化
            bright_to_dark = (last_brightness > brightness_threshold and 
                             brightness <= brightness_threshold)
            dark_to_bright = (last_brightness <= brightness_threshold and 
                             brightness > brightness_threshold)
            
            if bright_to_dark or dark_to_bright:
                flash_history.append(current_time)
                print(f"检测到闪烁变化: {bright_to_dark and '亮→暗' or '暗→亮'}, 时间: {current_time - start_time:.2f}s")
        
        last_brightness = brightness
        
        # 清理过期的闪烁记录
        flash_history = [t for t in flash_history if current_time - t <= flash_duration]
        
        # 检查是否满足闪烁模式
        if len(flash_history) >= min_flash_count * 2:  # 每次闪烁有亮暗两次变化
            # 检查最近1秒内的闪烁次数
            recent_flashes = [t for t in flash_history if current_time - t <= flash_window]
            if len(recent_flashes) >= min_flash_count * 2:
                print(f"检测到有效闪烁模式: {len(recent_flashes)//2}次闪烁/秒")
                return True
        
        # 短暂延迟，避免过度占用CPU
        time.sleep(0.01)
    
    print("检测超时，未发现有效闪烁模式")
    return False

def attack(target, link):
    # TODO: 不同目标位姿也不一样，可以考虑在函数外把位姿调整好？
    # 转速在函数里确认
    if target == 0:
        print("[INFO] 攻击哨兵")
        link.send_shooter_rpm(120)  # 设置转速以及触发发射
        # time.sleep(1.0)               # 等待转速稳定
        # link.shooter_fire()           # 触发发射
    elif target == 1:
        print("[INFO] 攻击大本营")
        link.send_shooter_rpm(150)  # 设置转速
        # time.sleep(1.0)               # 等待转速稳定
        # link.shooter_fire()           # 触发发射

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

def get_dart(link, dart_id, mc, roi, max_attempts=1):
    """
    抓取飞镖函数，支持多次尝试
    
    Args:
        link: 通信链接对象
        dart_id: 飞镖类型 (0-常规飞镖, 1-战略飞镖)
        mc: 相机管理器对象
        roi: 感兴趣区域
        max_attempts: 最大尝试次数，默认为1次
        
    Returns:
        bool: 抓取成功返回True，失败返回False
    """
    # TODO: link.send_get_dart1()函数待实现
    # 对于多次尝试抓取，可以试着每次加上偏移，但这样的话抓取飞镖就不是预设动作了，得和电控交流
    dart_names = {0: "常规飞镖", 1: "战略飞镖"}
    dart_name = dart_names.get(dart_id, f"未知飞镖({dart_id})")
    
    for attempt in range(1, max_attempts + 1):
        print(f"[INFO] 第{attempt}次尝试抓取{dart_name}")
        
        # 发送抓取指令
        _send_grab(link, dart_id)
        
        # 获取图像帧并检测飞镖
        frames = get_frames_burst(mc, "cam1", n=5, timeout_s=1.0, min_gap_ms=100)
        
        # 检测飞镖是否还存在（不存在表示抓取成功）
        if find_dart(frames, roi) == False:
            print(f"[SUCCESS] 第{attempt}次抓取{dart_name}成功")
            return True
        else:
            print(f"[WARNING] 第{attempt}次抓取{dart_name}失败")
            
            # 如果不是最后一次尝试，可以添加一些延迟或调整策略
            if attempt < max_attempts:
                print(f"[INFO] 准备进行第{attempt+1}次尝试...")
                # 这里可以添加一些重试前的延迟或其他操作
                # time.sleep(0.5)  # 如果需要延迟
    
    print(f"[ERROR] 抓取{dart_name}失败，已尝试{max_attempts}次")
    return False

def start_heartbeat(link, interval=0.2):
    def run():
        while True:
            link.heartbeat(interval_s=interval)
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()

calibdown_path = os.path.join(ROOT, "calib", "calibdown.npz")
camera_matrix, dist_coeffs = load_calib(calibdown_path)

# 全局变量
latest_result = None
latest_frame = None
detection_lock = threading.Lock()
detection_running = False
detection_thread = None

def setup_apriltag_detector():
    options = apriltag.DetectorOptions(families="tag36h11",
                                      border=1,
                                      nthreads=4,
                                      quad_decimate=2.0,
                                      quad_blur=0.8,
                                      refine_edges=True,
                                      refine_decode=True,
                                      refine_pose=True,
                                      debug=False,
                                      quad_contours=True)
    detector = apriltag.Detector(options)
    return detector

detector = setup_apriltag_detector()

# 创建全局相机实例（确保只创建一个）
camera_instance = None

def get_camera():
    """获取全局相机实例"""
    global camera_instance
    if camera_instance is None:
        camera_instance = Camera("/dev/cam_down")
        # 预先初始化相机
        camera_instance.initialize()
    return camera_instance

def continuous_detection():
    """持续检测线程函数"""
    global latest_result, latest_frame, detection_running
    
    # 获取相机实例
    cam = get_camera()
    
    while detection_running:
        try:
            # 使用锁确保线程安全地获取图像
            with detection_lock:
                img = cam.get_frame()
            
            if img is None:
                time.sleep(0.01)
                continue
            
            # 检测tag
            result = detect_dart_in_frame(img)
            
            # 更新最新结果
            with detection_lock:
                latest_frame = img
                latest_result = result
            
            time.sleep(0.033)  # 约30fps
            
        except Exception as e:
            print(f"检测错误: {e}")
            time.sleep(0.1)

def start_continuous_detection():
    """开始持续检测"""
    global detection_running, detection_thread
    
    if detection_running:
        return
    
    # 预先初始化相机
    get_camera()
    
    detection_running = True
    detection_thread = threading.Thread(target=continuous_detection, daemon=True)
    detection_thread.start()
    print("开始持续检测AprilTag")

def stop_continuous_detection():
    """停止持续检测"""
    global detection_running, camera_instance
    
    detection_running = False
    if detection_thread:
        detection_thread.join(timeout=1.0)
    
    if camera_instance:
        camera_instance.release()
        camera_instance = None
    print("停止持续检测")

def detect_dart_in_frame(img):
    """在指定帧中检测飞镖"""
    # 直接转换为灰度图（不去畸变！）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测AprilTag（在原始图像上检测）
    results = detector.detect(gray)
    
    if not results:
        return None
    
    # 获取第一个检测到的tag
    tag = results[0]
    corners = tag.corners.astype(np.float32)
    
    # tag尺寸
    tag_size = 20.5  # 毫米
    
    # tag的3D角点坐标
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],  # 左下
        [tag_size/2, -tag_size/2, 0],   # 右下  
        [tag_size/2, tag_size/2, 0],    # 右上
        [-tag_size/2, tag_size/2, 0]    # 左上
    ], dtype=np.float32)
    
    # 重要：直接使用畸变参数进行solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    
    if not success:
        return None
    
    return {
        'rvec': rvec,
        'tvec': tvec,
        'tag_id': tag.tag_id,
        'corners': corners
    }

def get_latest_detection():
    """获取最新检测结果"""
    with detection_lock:
        return latest_result

def get_latest_frame():
    """获取最新帧"""
    with detection_lock:
        return latest_frame

def detect_dart():
    """兼容原有接口的检测函数"""
    return get_latest_detection()

def calculate_lateral_offset(tvec, target_offset_x=0):
    """计算横向移动距离"""
    if tvec is None:
        raise ValueError("tvec is None")
    
    tvec_arr = np.asarray(tvec).reshape(-1)
    if tvec_arr.size < 1:
        raise ValueError("tvec shape unexpected: " + str(tvec.shape))

    current_offset = float(tvec_arr[0])
    movement_needed = current_offset - target_offset_x
    
    return movement_needed

def cleanup():
    """程序退出时清理资源"""
    stop_continuous_detection()
    cv2.destroyAllWindows()

def main():
    config_path = os.path.join(ROOT, "data", "config.json")
    cfg = load_config(config_path)                     # 可换成 data/config.json
    calibup_path = os.path.join(ROOT, "calib", "calibup.npz")

    # 1) 相机标定参数
    K0, dist0 = load_calib(calibup_path)        # 你已有的函数，返回 (mtx, dist)

    # 2) 组装外参与地图
    T_robot_cam_0 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam0"])
    T_robot_cam_1 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam1"])
    tag_map = build_tag_map(cfg) # tag_map

    # 3) 模块初始化
    # car = RobotCar(tag_map)
    # car.set_camera_extrinsic("cam0", T_robot_cam_0)
    # car.set_camera_extrinsic("cam1", T_robot_cam_1)
    link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"])
    link.open()
    # start_heartbeat(link, interval=0.2) # 启动心跳线程

    # cap = open_camera(cfg["camera_index"])
    goal = cfg["goal"]

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")

    # mc = MultiCam()
    # w = 640
    # h = 480
    # print("[INFO] 读取相机标定参数完成, 值为：", K0, dist0, K1, dist1)
    # mapx0, mapy0, newK0 = _CamWorker.build_undistort_maps(K0, dist0, (w, h))
    # mapx1, mapy1, newK1 = _CamWorker.build_undistort_maps(K1, dist1, (w, h))
    # print("[INFO] 畸变校正映射计算完成, 值为：", newK0, newK1)
    # backend = cv2.CAP_DSHOW  # Windows使用DirectShow后端 # linux上直接使用默认赋值即可
    # mc.add_camera("cam0", "/dev/cam_up", width=w, height=h, undistort_maps=(mapx0, mapy0, newK0), fourcc="MJPG")
    # mc.add_camera("cam1", "/dev/cam_down", width=w, height=h, undistort_maps=(mapx1, mapy1, newK1), fourcc="MJPG")
    # mc.start()
    # det0 = Detector(
    #     camera_matrix=newK0,
    #     tag_size=cfg["tag_size"],
    #     hsv_params=cfg["hsv_range"]
    # )
    # det1 = Detector(
    #     camera_matrix=newK1,
    #     tag_size=cfg["tag_size"],
    #     hsv_params=cfg["hsv_range"]
    # )
    
    # TODO: 数据待更新
    state = State.INIT_CHECKS
    dart1_pos = [1, 1, 1, 1, 1]
    dart2_pos = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sb_hp = 2
    # scan_fn = lambda: get_tags(mc, det0, det1, car)
    dart1_roi = cfg["dart1_roi"] # TODO: 待更新
    dart2_roi = cfg["dart2_roi"] # TODO: 待更新
    dart1_num = 3
    dart2_num = 5
    attach_callbacks(link)
    led_roi = cfg["led_roi"] # TODO: 待更新
    last_state = State.INIT_CHECKS
    
    start_continuous_detection()
    while True:
        # 每次移动后失败怎么处理？
        if state == State.INIT_CHECKS:
            # 初始阶段，检测各模块
            # if camera.is_initialized == False:
            #     camera.initialize()
            print("[INFO] 相机已就绪")
            last_state = state
            state = State.INITIAL_LOCATE

        elif state == State.INITIAL_LOCATE:
            # ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=4.0)
            ok, info = _send_move_and_wait_ack(link, 1600, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            time.sleep(1.0)
            ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            time.sleep(1.0)
            ok, info = _send_move_and_wait_ack(link, 4000, 0, timeout_s=15.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            print("[INFO] 撞墙")
            ok, info = _send_move_and_wait_ack(link, 0, -1000, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 1000, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)            
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            last_state = state
            state = State.GO_DART1
            
                

        elif state == State.GO_DART1:
            dart_num = 3
            DART_TAG_IDS = set(range(7, 15))
            # ok, info = _send_move_and_wait_ack(link, 0, 230, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            for _ in range(dart_num): 
                flag = False
                while True:
                    # time.sleep(1)
                    result = detect_dart()
                    if result is None:
                        if flag == False:
                            print("[ERR] 未检测到飞镖")
                            # time.sleep(1)
                            # TODO:横着走多少
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        else:
                            continue
                    tvec = result['tvec']
                    tag_id = result['tag_id']
                    
                    if tag_id not in DART_TAG_IDS:
                        print(f"[INFO] 检测到非飞镖tag {tag_id}，跳过处理")
                        # time.sleep(1)
                        # TODO:横着走多少
                        if flag == False:
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                        continue
                    
                    print(f"检测到tag {tag_id}, 位置: {tvec.flatten()}")
                    flag = True
                    
                    # 计算移动量
                    movement = calculate_lateral_offset(tvec, target_offset_x=-71)
                    if abs(movement) < 4:
                        print("[INFO] 飞镖已在目标位置，无需移动")
                        break
                    print(f"[INFO] 需要横向移动 {-movement:.1f} mm")
                    if -movement > 0:
                        ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                    else:
                        ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                time.sleep(2)
                ok, info = _send_move_and_wait_ack(link, 150, 0, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                ok, info = _send_grab(link, "", timeout_s=25.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                tries = 3
                while tries > 0:
                    tries -= 1
                    result = detect_dart()
                    if result is None or result['tag_id'] not in DART_TAG_IDS:
                        tries = 0
                        continue
                    ok, info = _send_grab(link, "", timeout_s=25.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
            last_state = state
            state = State.ATTACK1

        elif state == State.GO_DART2:
            pass
                

        elif state == State.ATTACK1:
            # TODO: 具体定位
            # 后退
            ok, info = _send_move_and_wait_ack(link, -100, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 旋转
            ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            # # 后面撞墙
            # ok, info = _send_move_and_wait_ack(link, 0, -1500, timeout_s=6.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # # 右面撞墙
            # ok, info = _send_move_and_wait_ack(link, -1000, 0, timeout_s=6.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 发射
            fire_times = 3
            for _ in range(fire_times):
                ok, info = _send_fire(link, rpm=1200, timeout_s=8.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                time.sleep(1.0)
            last_state = state
            state = State.DONE
        
        elif state == State.ATTACK2:
            pass
        
        elif state == State.DONE:
            print("[INFO] 任务完成")
            break

        elif state == State.FAIL:
            link.send_stop()
            print("[ERR] 任务失败")
            break
        
        link.heartbeat(interval_s=0.2)
        print("[INFO] 上一个状态:", last_state.name, "当前状态:", state.name)

# ====== 入口 ======
if __name__ == "__main__":
    main()
    cleanup()
