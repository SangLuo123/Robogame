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

# ========== 任务状态机 ==========

class State(Enum):
    INIT_CHECKS = auto()
    INITIAL_LOCATE = auto()
    GO_DART1 = auto()
    GO_DART2 = auto()
    ATTACK = auto()
    DONE = auto()
    FAIL = auto()
    GRAB = auto()

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

def get_tags(mc, det1, car):
    # 什么时候更新位姿？
    # init_locate：刚开始更新位姿
    # 每次移动结束会更新
    # TODO: 下楼梯需不需要用到？
    # 需不需要考虑车的z坐标？感觉应该不用考虑，不考虑唯一有问题的是斜着，即上下楼梯时的位姿，但上下楼梯不需要测算位姿，
    # 故直接不考虑z坐标，将小车中心定位小车底座中心对应的地面
    pkt0 = mc.latest("cam1")
    pkt0 = pkt0.image
    
    all_positions = []
    all_distances = []
    valid_tags = []
    sample_count = 5  # 采样次数

    for _ in range(sample_count):
        tags1 = det1.detect_tags(pkt0, estimate_pose=True) or []
        if tags1:
            closest_tag = min(
                tags1,
                key=lambda t: float(np.linalg.norm(np.array(t["pose_t"]).reshape(3)))
            )
            position = np.array(closest_tag["pose_t"]).reshape(3)
            distance = float(np.linalg.norm(position))
            
            all_positions.append(position)
            all_distances.append(distance)
            valid_tags.append(closest_tag)
                
    
    if not valid_tags:
        return None
    # if tags1:
    # for d in tags1:
    #     pts = d["corners"].astype(int)
    #     cv2.polylines(pkt0, [pts], True, (0,255,0), 2)
    #     c = np.mean(pts, axis=0).astype(int)
    #     cv2.putText(pkt0, f"id:{d['tag_id']}", tuple(c), 0, 0.6, (0,255,0), 2)
    
    # 计算平均位姿和距离
    avg_position = np.mean(all_positions, axis=0)
    avg_distance = np.mean(all_distances)
    
    # 使用第一个tag的结构，替换为平均位姿
    best_tag = valid_tags[0].copy()
    best_tag["pose_t"] = avg_position.tolist()
    
    car.update_pose_from_tag(best_tag, cam_id="cam1")
    return {"best_cam": "cam1", "best_tag": best_tag, "distance": avg_distance}
        

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
        """等待一条 ENC；返回 (ok:bool, fields:list) 或 None(超时)"""
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

def _send_move_and_wait_ack(link, dx_b, dy_b, timeout_s=6.0):
    w = _AckWaiter(link)
    link.send_vel_xy(dx_b, dy_b)
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    print(f"MOVE ENC: {ok} {' '.join(map(str, fields))}")
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def _send_rot_and_wait_ack(link, d_yaw, timeout_s=4.0):
    w = _AckWaiter(link)
    link.rotate(d_yaw)
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    print(f"ROT ENC: {ok} {' '.join(map(str, fields))}")
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

def attack(target, link):
    # TODO: 不同目标位姿也不一样，可以考虑在函数外把位姿调整好？
    # 转速在函数里确认
    if target == 0:
        print("[INFO] 攻击哨兵")
        link.send_shooter_rpm(1500)  # 设置转速
        # time.sleep(1.0)               # 等待转速稳定
        # link.shooter_fire()           # 触发发射
    elif target == 1:
        print("[INFO] 攻击大本营")
        link.send_shooter_rpm(2000)  # 设置转速
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

def _send_grab(link, dart_id, timeout_s=10.0):
    w = _AckWaiter(link)
    if dart_id == 0:
        link.arm_preset("0")
    elif dart_id == 1:
        link.arm_preset("1")
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

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

def detect_apriltag_pose(frame, tag_size, camera_matrix, dist_coeffs, tag_family='tag36h11'):
    """
    检测AprilTag并计算相对位置
    
    Args:
        frame: 输入图像帧 (BGR格式)
        tag_size: 标签的实际物理尺寸（米）
        camera_matrix: 3x3相机内参矩阵
        dist_coeffs: 畸变系数
        tag_family: 标签类型，默认'tag36h11'
    
    Returns:
        List[dict]: 检测到的标签信息列表，每个包含id、位置、距离等信息
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 创建检测器
    detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))
    
    # 检测标签
    detections = detector.detect(gray)
    
    results = []
    
    # 定义标签的3D角点坐标（以标签中心为原点）
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],  # 左下
        [ tag_size/2, -tag_size/2, 0],  # 右下
        [ tag_size/2,  tag_size/2, 0],  # 右上
        [-tag_size/2,  tag_size/2, 0]   # 左上
    ], dtype=np.float32)
    

    # 检查检测结果的结构
    if detections is None or len(detections) == 0:
        return results
    
    # 根据不同的库返回格式进行处理
    # 方法1：如果 detections 是字典列表
    if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
        for detection in detections:
            if detection.get('hamming', 0) != 0:
                continue
                
            corners = np.array(detection.get('lb-rb-rt-lt', detection.get('corners')), dtype=np.float32)
            if corners is None or len(corners) != 4:
                continue
                
            success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
            if not success:
                continue
                
            # 计算距离和角度
            distance = np.linalg.norm(tvec)
            position = tvec.flatten()
            angle_x = np.arctan2(position[0], position[2]) * 180 / np.pi
            angle_y = np.arctan2(position[1], position[2]) * 180 / np.pi
            
            result = {
                'tag_id': detection.get('id', detection.get('tag_id', -1)),
                'corners': corners.astype(int),
                'center': np.mean(corners, axis=0).astype(int),
                'position': position.tolist(),
                'distance': float(distance),
                'angle_x': float(angle_x),
                'angle_y': float(angle_y),
                'rvec': rvec,
                'tvec': tvec
            }
            results.append(result)
    
    # 方法2：如果 detections 是特定格式的数组
    else:
        # 尝试不同的属性访问方式
        for detection in detections:
            try:
                # 尝试获取标签ID
                tag_id = getattr(detection, 'tag_id', getattr(detection, 'id', -1))
                
                # 尝试获取角点
                corners = getattr(detection, 'corners', None)
                if corners is None:
                    # 有些库使用不同的角点属性名
                    corners = getattr(detection, 'lb-rb-rt-lt', None)
                
                if corners is None:
                    continue
                    
                corners = np.array(corners, dtype=np.float32)
                
                # 使用solvePnP计算位姿
                success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
                if not success:
                    continue
                    
                # 计算距离和角度
                distance = np.linalg.norm(tvec)
                position = tvec.flatten()
                angle_x = np.arctan2(position[0], position[2]) * 180 / np.pi
                angle_y = np.arctan2(position[1], position[2]) * 180 / np.pi
                
                # 计算中心点
                center = np.mean(corners, axis=0).astype(int)
                
                result = {
                    'tag_id': tag_id,
                    'corners': corners.astype(int),
                    'center': center,
                    'position': position.tolist(),
                    'distance': float(distance),
                    'angle_x': float(angle_x),
                    'angle_y': float(angle_y),
                    'rvec': rvec,
                    'tvec': tvec
                }
                results.append(result)
                
            except Exception as e:
                print(f"处理检测结果时出错: {e}")
                continue
    
    return results

def calculate_relative_movement(current_result, goal_relative_position):
    """
    计算从当前位置到目标相对位置需要移动的距离
    
    Args:
        current_result: 当前检测到的AprilTag结果
        goal_relative_position: 目标相对位置 [x, y, z]（Tag在相机坐标系中的期望位置）
    
    Returns:
        dict: 包含移动信息的字典
    """
    # 当前Tag的相对位置（从相机到Tag的向量）
    current_tvec = current_result['tvec'].flatten()
    
    # 目标相对位置
    goal_pos = np.array(goal_relative_position)
    
    # 计算需要移动的向量
    # 注意：move_vector 是相机需要移动的方向
    move_vector = goal_pos - current_tvec
    
    # 计算总移动距离
    total_distance = np.linalg.norm(move_vector)
    
    # 分解到各轴的距离
    dx, dy, dz = move_vector
    
    # 计算移动方向角度（可选）
    angle_xy = np.arctan2(dy, dx) * 180 / np.pi  # 水平面角度
    angle_xz = np.arctan2(dz, dx) * 180 / np.pi  # 前后倾斜角度
    
    return {
        'current_relative_position': current_tvec.tolist(),
        'goal_relative_position': goal_pos.tolist(),
        'move_vector': move_vector.tolist(),
        'total_distance': float(total_distance),
        'move_x': float(dx),  # X方向移动量（向右为正）
        'move_y': float(dy),  # Y方向移动量（向下为正）
        'move_z': float(dz),  # Z方向移动量（向前为正）
        'angle_xy': float(angle_xy),  # 水平方向角度
        'angle_xz': float(angle_xz)   # 前后方向角度
    }


def main():
    config_path = os.path.join(ROOT, "data", "config.json")
    cfg = load_config(config_path)                     # 可换成 data/config.json
    # calibup_path = os.path.join(ROOT, "calib", "calibup.npz")
    calibdown_path = os.path.join(ROOT, "calib", "calibdown.npz")

    # 1) 相机标定参数
    # K0, dist0 = load_calib(calibup_path)        # 你已有的函数，返回 (mtx, dist)
    K1, dist1 = load_calib(calibdown_path)      # 你已有的函数，返回 (mtx, dist)

    # 2) 组装外参与地图
    # T_robot_cam_0 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam0"])
    T_robot_cam_1 = build_T_robot_cam(cfg["T_robot_cam_for"]["cam1"])
    tag_map = build_tag_map(cfg) # tag_map

    # 3) 模块初始化
    car = RobotCar(tag_map)
    # car.set_camera_extrinsic("cam0", T_robot_cam_0)
    car.set_camera_extrinsic("cam1", T_robot_cam_1)
    # link = SerialLink(port=cfg["serial_port"], baud=cfg["baud"])
    link = SerialLink(port="/dev/pts/5", baud=cfg["baud"])
    link.open()
    start_heartbeat(link, interval=0.2) # 启动心跳线程

    # cap = open_camera(cfg["camera_index"])
    goal = cfg["goal"]

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")

    mc = MultiCam()
    w = 640
    h = 480
    print("[INFO] 读取相机标定参数完成, 值为：", K1, dist1)
    # mapx0, mapy0, newK0 = _CamWorker.build_undistort_maps(K0, dist0, (w, h))
    mapx1, mapy1, newK1 = _CamWorker.build_undistort_maps(K1, dist1, (w, h))
    print("[INFO] 畸变校正映射计算完成, 值为：", newK1)
    # backend = cv2.CAP_DSHOW  # Windows使用DirectShow后端 # linux上直接使用默认赋值即可
    # mc.add_camera("cam0", "/dev/cam_up", width=w, height=h, undistort_maps=(mapx0, mapy0, newK0), fourcc="MJPG")
    mc.add_camera("cam1", "/dev/cam_down", width=w, height=h, undistort_maps=(mapx1, mapy1, newK1), fourcc="MJPG")
    mc.start()
    # det0 = Detector(
    #     camera_matrix=newK0,
    #     tag_size=cfg["tag_size"],
    #     hsv_params=cfg["hsv_range"]
    # )
    det1 = Detector(
        camera_matrix=newK1,
        tag_size=cfg["tag_size"],
        hsv_params=cfg["hsv_range"]
    )
    
    # TODO: 数据待更新
    state = State.INIT_CHECKS
    dart1_pos = [1, 1, 1, 1, 1]
    dart2_pos = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sb_hp = 2
    scan_fn = lambda: get_tags(mc, det1, car)
    dart1_roi = cfg["dart1_roi"] # TODO: 待更新
    dart2_roi = cfg["dart2_roi"] # TODO: 待更新
    dart1_num = 3
    dart2_num = 5
    attach_callbacks(link)
    last_state = State.INIT_CHECKS
    dist_coeffs = np.zeros((4, 1))
    while True:
        if state == State.INIT_CHECKS:
            # 初始阶段，检测各模块
            if wait_ready(mc, ["cam1"], timeout_s=3.0):
                print("[INFO] 两路相机已就绪")
                last_state = state
                state = State.INITIAL_LOCATE
            else:
                print("[ERR] 相机初始化失败")
                last_state = state
                state = State.FAIL

        elif state == State.INITIAL_LOCATE:
            d_x = 1500
            d_y = 500
            d_yaw = 90
            ok, info = _send_move_and_wait_ack(link, d_x, 0)
            if not ok:
                print(f"[ERR] MOVE {info}")
            ok, info = _send_move_and_wait_ack(link, 0, d_y)
            if not ok:
                print(f"[ERR] MOVE {info}")
            tag = get_tags(mc, det1, car)
            print("[INFO] tag:", tag)
            ok, info = _send_rot_and_wait_ack(link, d_yaw)
            if not ok:
                print(f"[ERR] MOVE {info}")
            tag = get_tags(mc, det1, car)
            print("[INFO] tag:", tag)
            last_state = state
            state = State.GO_DART1
                

        elif state == State.GO_DART1:
            """
            上一个状态的可能情况：
            State.INITIAL_LOCATE: 初始定位完成后，进行常规飞镖获取
            State.ATTACK: 攻击完成后，检查是否还有常规飞镖未获取，若有则继续获取
            State.GO_DART: 上一个常规飞镖获取失败，尝试下一个位置
            """
            if dart1_num <= 0:
                print("[INFO] 常规飞镖已全部获取")
                # TODO: 该情况应该只会在上个状态是打击的情况，不用更新last_state，否则干扰移动
                # last_state = state
                state = State.DONE
                continue
            dart1 = goal["dart1"]
            # first_one_index = None
            # for i, value in enumerate(dart1_pos):
            #     if value == 1:
            #         first_one_index = i
            #         break
            # if first_one_index is None:
            #     print("[INFO] 所有飞镖靶均已击中")
            #     # TODO: 如果全部遍历完，dart1_num早为0了，应该不会到这个if？
            #     # last_state = state
            #     state = State.GO_DART2
            #     continue
            # dart1[0] += first_one_index * 230  # TODO:每个存飞镖区间隔300mm
            # 不论上个状态是initial_locate还是attack，移动策略都一样，先平移再旋转，且先x更好
            # if last_state != State.ATTACK and last_state != State.INITIAL_LOCATE and last_state != State.GO_DART1:
            #     print("[WARN] 上一个状态非预期")
            #     # TODO:
            # 其实这个drot为0！！！
            res = go_to_ack_then_optional_tag_verify(
                car, dart1, link,
                pos_tol=50.0, yaw_tol=5.0,
                move_timeout=6.0, rot_timeout=4.0,
                verify_window=0.8,
                max_corrections=1,
                scan_fn=scan_fn,
                strategy="move_x_then_y_then_rot"
            )
            print("[RESULT]", res)
            # TODO:找飞镖，拿飞镖
            print("[INFO] 寻找常规飞镖…")
            dx = 300
            dy = 200
            ok, info = _send_move_and_wait_ack(link, dx, 0)
            if not ok:
                print(f"[ERR] MOVE {info}")
            ok, info = _send_move_and_wait_ack(link, 0, dy)
            if not ok:
                print(f"[ERR] MOVE {info}")
            print("[INFO] 移动完成，开始检测常规飞镖…")
            wait_ready(mc, ["cam1"], timeout_s=3.0)
            while True:
                frame = mc.latest("cam1")
                if frame is None:
                    print("[WARN] 读相机帧失败")
                    time.sleep(0.01)
                    continue
                img = frame.image
                cv2.imshow("down", img)
                key = cv2.waitKey(1) & 0xFF  # 添加这一行
                res = detect_apriltag_pose(img, cfg["tag_size0"], newK1, dist_coeffs)
                if res != []:
                    break
            goal1 = goal["grab_dart1"] # TODO:
            move_info = calculate_relative_movement(res, goal1)
            dx = move_info['move_x']
            dy = move_info['move_y']
            print(f"[INFO] 需要移动: dx={dx:.1f} mm, dy={dy:.1f} mm")
            ok, info = _send_move_and_wait_ack(link, dx, dy)
            if not ok:
                print(f"[ERR] MOVE {info}")
            last_state = state
            state = State.GRAB
            

        elif state == State.GO_DART2:
            """
            上一个状态的可能情况：
            State.ATTACK: 攻击完成后，检查是否还有常规飞镖未获取，若有则继续获取
            State.GO_DART1: 常规飞镖已全部获取，进行战略飞镖获取
            State.GO_DART2: 上一个战略飞镖获取失败，尝试下一个位置
            """
            if dart2_num <= 0:
                print("[INFO] 常规飞镖已全部获取")
                # TODO: 同上
                # last_state = state
                state = State.DONE
                continue
            dart2 = goal["dart2"]
            first_one_index = None
            for i, value in enumerate(dart2_pos):
                if value == 1:
                    first_one_index = i
                    break
            if first_one_index is None:
                print("[INFO] 所有飞镖靶均已击中")
                # TODO: 同上
                # last_state = state
                state = State.DONE
                continue
            dart2[1] -= first_one_index * 400  # TODO:每个存飞镖区间隔300mm
            # 到战略飞镖区的前一个状态只有ATTACK
            if last_state != State.ATTACK and last_state != State.GO_DART2:
                print("[WARN] 上一个状态非预期")
                # TODO:
            x0, y0, yaw0 = car.get_pose()
            goal_dart2 = (2100, y0, 180)
            res = go_to_ack_then_optional_tag_verify(
                car, goal_dart2, link,
                pos_tol=50.0, yaw_tol=5.0,
                move_timeout=6.0, rot_timeout=4.0,
                verify_window=0.8,
                max_corrections=1,
                scan_fn=scan_fn,
                strategy="rot_then_move_x_then_y"
            )
            print("[RESULT]", res)
            res = go_to_ack_then_optional_tag_verify(
                car, dart2, link,
                pos_tol=50.0, yaw_tol=5.0,
                move_timeout=6.0, rot_timeout=4.0,
                verify_window=0.8,
                max_corrections=1,
                scan_fn=scan_fn,
                strategy="rot_then_move_x_then_y"
            )
            # TODO:找飞镖，拿飞镖
            dart2_pos[first_one_index] = 0
            frames = get_frames_burst(mc, "cam1", n=5, timeout_s=1.0, min_gap_ms=100)
            if find_dart(frames, dart2_roi) == True:
                print("[INFO] 战略飞镖已找到，在第", first_one_index + 1, "个位置")
                last_state = state
                state = State.GRAB
            else:
                print("[INFO] 第", first_one_index + 1, "个位置战略飞镖未找到，尝试下一个位置")
                last_state = state
                state = State.GO_DART2
                
        elif state == State.GRAB:
            # 若未成功抓取，直接跳过
            # TODO: 考虑最后重新抓取？如果真有飞镖颜色一样，那得考虑，否则正常情况是能抓取成功
            # 该阶段不更新last_state，因为抓取成功后进入打击需要上个阶段的状态
            if last_state == State.GO_DART1:
                success = get_dart(link, 0, mc, dart1_roi) # dart1
                if success:
                    dart1_num -= 1
                    print("[INFO] 常规飞镖获取成功，剩余数量:", dart1_num)
                    # last_state = state
                    state = State.ATTACK
                else:
                    print("[ERR] 常规飞镖获取失败，尝试下一个位置")
                    # last_state = state
                    state = State.GO_DART1
            elif last_state == State.GO_DART2:
                success = get_dart(link, 1, mc, dart2_roi) # dart2
                if success:
                    dart2_num -= 1
                    print("[INFO] 战略飞镖获取成功，剩余数量:", dart2_num)
                    # last_state = state
                    state = State.ATTACK
                else:
                    print("[ERR] 战略飞镖获取失败，尝试下一个位置")
                    # last_state = state
                    state = State.GO_DART2

        elif state == State.ATTACK:
            
            if sb_hp > 0:
                d_x = -800
                ok, info = _send_move_and_wait_ack(link, d_x, 0)
                if not ok:
                    print(f"[ERR] MOVE {info}")
                goal_dart1 = goal["dart1"]
                res = go_to_ack_then_optional_tag_verify(
                    car, goal_dart1, link,
                    pos_tol=50.0, yaw_tol=5.0,
                    move_timeout=6.0, rot_timeout=4.0,
                    verify_window=0.8,
                    max_corrections=1,
                    scan_fn=scan_fn,
                    strategy="move_x_then_y_then_rot"
                )
                print("[RESULT]", res)
                d_yaw = 90
                ok, info = _send_rot_and_wait_ack(link, d_yaw)
                if not ok:
                    print(f"[ERR] ROT {info}")
                attack(0, link) # 攻击哨兵
                print("[INFO] 哨兵已攻击")
                sb_hp -= 1
            else:
                if last_state == State.GO_DART1:
                    d_yaw = 45
                    ok, info = _send_rot_and_wait_ack(link, d_yaw)
                    if not ok:
                        print(f"[ERR] ROT {info}")
                    attack(1, link) # 攻击大本营
                    print("[INFO] 大本营已攻击")
                elif last_state == State.GO_DART2:
                    pass
            
            last_state = state
            state = State.GO_DART1

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
