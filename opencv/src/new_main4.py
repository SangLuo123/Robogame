"""
预赛全流程，从头开始
常规飞镖打哨所
战略飞镖打大本营
"""
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

# from src.car import RobotCar
# from src.detector import Detector
from src.comm import SerialLink
from src.load import load_calib # , load_config
# from src.transform import build_T_robot_cam, build_tag_map
# from src.multicam import MultiCam, _CamWorker
# from src.detect_dart import find_dart
from src.camera import Camera

# ========== 任务状态机 ==========

class State(Enum):
    INIT_CHECKS = auto()
    INITIAL_LOCATE = auto()
    GO_DART1 = auto()
    GO_DART2 = auto()
    FIND_DART2 = auto()
    ATTACK1 = auto()
    ATTACK2 = auto()
    ATTACK3 = auto()
    GO_HIGH = auto()
    DONE = auto()
    FAIL = auto()

# ========== 一些函数 ==========
# 全局变量存储参数
dart1_num = 3
dart2_num = 5
config_file = os.path.join(ROOT, "src", "config.txt")  # 替换为你的文件路径


# 读取配置文件
def read_config():
    global dart1_num, dart2_num
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith('dart1_num:'):
                        dart1_num = int(line.split(':')[1].strip())
                    elif line.startswith('dart2_num:'):
                        dart2_num = int(line.split(':')[1].strip())
            print(f'读取配置: dart1_num={dart1_num}, dart2_num={dart2_num}')
        else:
            print('配置文件不存在，使用默认值')
            write_config()  # 创建默认配置文件
    except Exception as e:
        print(f'读取配置失败: {e}，使用默认值')
        
# 更新参数的函数
def update_parameters(new_dart1, new_dart2):
    global dart1_num, dart2_num
    dart1_num = new_dart1
    dart2_num = new_dart2
    write_config()  # 立即保存到文件
        
# 写入配置文件
def write_config():
    try:
        with open(config_file, 'w') as file:
            file.write(f'dart1_num: {dart1_num}\n')
            file.write(f'dart2_num: {dart2_num}\n')
        print(f'保存配置: dart1_num={dart1_num}, dart2_num={dart2_num}')
    except Exception as e:
        print(f'保存配置失败: {e}')

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

def _send_upstairs(link, timeout_s=10.0):
    w = _AckWaiter(link)
    link.send_upstairs()
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def _send_downstairs(link, timeout_s=10.0):
    w = _AckWaiter(link)
    link.send_downstairs()
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def _send_pick_dart(link, timeout_s=25.0):
    w = _AckWaiter(link)
    link.pick_dart()
    ev = w.wait(timeout_s)
    if ev is None:
        return False, "timeout"
    ok, fields = ev
    return (ok, "OK" if ok else ("ERR:" + ",".join(map(str, fields))))

def start_heartbeat(link, interval=0.2):
    def run():
        while True:
            link.heartbeat(interval_s=interval)
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()

calibdown_path = os.path.join(ROOT, "calib", "calibdown.npz")
camera_matrix1, dist_coeffs1 = load_calib(calibdown_path)
calibup_path = os.path.join(ROOT, "calib", "calibup.npz")
camera_matrix2, dist_coeffs2 = load_calib(calibup_path)

# 第一个摄像头的全局变量
latest_result_cam1 = None
latest_frame_cam1 = None
detection_lock_cam1 = threading.Lock()
detection_running_cam1 = False
detection_thread_cam1 = None

# 第二个摄像头的全局变量
latest_result_cam2 = None
latest_frame_cam2 = None
detection_lock_cam2 = threading.Lock()
detection_running_cam2 = False
detection_thread_cam2 = None

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


# 创建全局相机实例
camera_instance_cam1 = None
camera_instance_cam2 = None

def get_camera(camera_id):
    """获取全局相机实例"""
    global camera_instance_cam1, camera_instance_cam2
    # TODO: 索引有问题就修改这里
    if camera_id == 1:
        if camera_instance_cam1 is None:
            camera_instance_cam1 = Camera("/dev/cam_down")
            camera_instance_cam1.initialize()
        return camera_instance_cam1
    elif camera_id == 2:
        if camera_instance_cam2 is None:
            camera_instance_cam2 = Camera("/dev/cam_up")  # 假设第二个摄像头设备路径
            camera_instance_cam2.initialize()
        return camera_instance_cam2
    else:
        raise ValueError("camera_id must be 1 or 2")
    
def get_camera_params(camera_id):
    """获取对应摄像头的内参和畸变参数"""
    if camera_id == 1:
        return camera_matrix1, dist_coeffs1
    elif camera_id == 2:
        return camera_matrix2, dist_coeffs2
    else:
        raise ValueError("camera_id must be 1 or 2")

def continuous_detection(camera_id, tag_size):
    """持续检测线程函数"""
    global latest_result_cam1, latest_frame_cam1, detection_running_cam1
    global latest_result_cam2, latest_frame_cam2, detection_running_cam2
    
    local_detector = setup_apriltag_detector()
    # 获取对应摄像头的内参和畸变参数
    camera_matrix, dist_coeffs = get_camera_params(camera_id)
    
    # 获取相机实例
    cam = get_camera(camera_id)
    
    while (detection_running_cam1 if camera_id == 1 else detection_running_cam2):
        try:
            # 使用锁确保线程安全地获取图像
            if camera_id == 1:
                with detection_lock_cam1:
                    img = cam.get_frame()
            else:
                with detection_lock_cam2:
                    img = cam.get_frame()
            
            if img is None:
                time.sleep(0.01)
                continue
            
            # 检测tag（传入摄像头参数）
            result = detect_dart_in_frame(img, tag_size, camera_matrix, dist_coeffs, detector=local_detector)
            
            # 更新最新结果
            if camera_id == 1:
                with detection_lock_cam1:
                    latest_frame_cam1 = img
                    latest_result_cam1 = result
            else:
                with detection_lock_cam2:
                    latest_frame_cam2 = img
                    latest_result_cam2 = result
            
            time.sleep(0.033)  # 约30fps
            
        except Exception as e:
            print(f"摄像头{camera_id}检测错误: {e}")
            time.sleep(0.1)

def start_continuous_detection(camera_id=1, tag_size=20.5):
    """开始持续检测"""
    global detection_running_cam1, detection_thread_cam1
    global detection_running_cam2, detection_thread_cam2
    
    if camera_id == 1:
        if detection_running_cam1:
            return
        detection_running_cam1 = True
        detection_thread_cam1 = threading.Thread(
            target=continuous_detection, 
            args=(camera_id, tag_size), 
            daemon=True
        )
        detection_thread_cam1.start()
        print(f"开始摄像头{camera_id}持续检测AprilTag, tag_size: {tag_size}mm")
    else:
        if detection_running_cam2:
            return
        detection_running_cam2 = True
        detection_thread_cam2 = threading.Thread(
            target=continuous_detection, 
            args=(camera_id, tag_size), 
            daemon=True
        )
        detection_thread_cam2.start()
        print(f"开始摄像头{camera_id}持续检测AprilTag, tag_size: {tag_size}mm")

def stop_continuous_detection(camera_id=1, wait_timeout=2.0):
    """停止持续检测。等待线程退出后再释放 camera 实例。"""
    global detection_running_cam1, camera_instance_cam1, detection_thread_cam1
    global detection_running_cam2, camera_instance_cam2, detection_thread_cam2

    if camera_id == 1:
        detection_running_cam1 = False
        # 等待线程结束（轮询 join）
        if detection_thread_cam1:
            detection_thread_cam1.join(timeout=wait_timeout)
            if detection_thread_cam1.is_alive():
                print(f"警告: 摄像头{camera_id}检测线程未在 {wait_timeout}s 内退出")
            detection_thread_cam1 = None

        if camera_instance_cam1:
            try:
                camera_instance_cam1.release()
            except Exception as e:
                print(f"释放 camera1 时出错: {e}")
            camera_instance_cam1 = None
        print(f"停止摄像头{camera_id}持续检测")
    else:
        detection_running_cam2 = False
        if detection_thread_cam2:
            detection_thread_cam2.join(timeout=wait_timeout)
            if detection_thread_cam2.is_alive():
                print(f"警告: 摄像头{camera_id}检测线程未在 {wait_timeout}s 内退出")
            detection_thread_cam2 = None

        if camera_instance_cam2:
            try:
                camera_instance_cam2.release()
            except Exception as e:
                print(f"释放 camera2 时出错: {e}")
            camera_instance_cam2 = None
        print(f"停止摄像头{camera_id}持续检测")


def detect_dart_in_frame(img, tag_size, camera_matrix=None, dist_coeffs=None, detector=None):
    """在指定帧中检测飞镖"""
    # 如果没有传入相机参数，使用默认的第一个摄像头参数（保持向后兼容）
    if camera_matrix is None:
        camera_matrix = camera_matrix1
    if dist_coeffs is None:
        dist_coeffs = dist_coeffs1
    if detector is None:
        detector = setup_apriltag_detector()
    
    # 直接转换为灰度图（不去畸变！）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测AprilTag（在原始图像上检测）
    results = detector.detect(gray)
    
    if not results:
        return None
    
    # 获取第一个检测到的tag
    tag = results[0]
    corners = tag.corners.astype(np.float32)
    
    # tag的3D角点坐标
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],  # 左下
        [tag_size/2, -tag_size/2, 0],   # 右下  
        [tag_size/2, tag_size/2, 0],    # 右上
        [-tag_size/2, tag_size/2, 0]    # 左上
    ], dtype=np.float32)
    
    # 使用传入的相机参数进行solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    
    if not success:
        return None
    
    return {
        'rvec': rvec,
        'tvec': tvec,
        'tag_id': tag.tag_id,
        'corners': corners,
        'camera_matrix': camera_matrix,  # 可选：返回使用的相机参数
        'dist_coeffs': dist_coeffs       # 可选：返回使用的畸变参数
    }

def get_latest_detection(camera_id=1):
    """获取最新检测结果"""
    if camera_id == 1:
        with detection_lock_cam1:
            return latest_result_cam1
    else:
        with detection_lock_cam2:
            return latest_result_cam2

def get_latest_frame(camera_id=1):
    """获取最新帧"""
    if camera_id == 1:
        with detection_lock_cam1:
            return latest_frame_cam1
    else:
        with detection_lock_cam2:
            return latest_frame_cam2

def detect_dart(camera_id=1):
    """兼容原有接口的检测函数"""
    return get_latest_detection(camera_id)

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
    stop_continuous_detection(1)
    stop_continuous_detection(2)
    cv2.destroyAllWindows()

def main():
    # config_path = os.path.join(ROOT, "data", "config.json")
    # cfg = load_config(config_path)                     # 可换成 data/config.json

    global dart1_num, dart2_num
    # 模块初始化
    link = SerialLink(port="/dev/ttyUSB0", baud=115200)
    link.open()
    # start_heartbeat(link, interval=0.2) # 启动心跳线程

    print("[INFO] 启动完成，进入主循环… 按 ESC 退出。")
    
    read_config()  # 启动时读取配置文件
    print(f'当前参数: dart1_num={dart1_num}, dart2_num={dart2_num}')
    attach_callbacks(link)
    recover_flag = False
    if dart1_num > 0:
        state = State.INITIAL_LOCATE
        last_state = State.INITIAL_LOCATE
    else:
        state = State.GO_HIGH
        last_state = State.GO_HIGH
    
    if dart2_num != 5:
        recover_flag = True
    box_num = 3
    DART2_lr = 0 # 0-左 1-右
    dart2_grab = 0 # 抓了几个战略飞镖
    fire_dart1 = 0
    # TODO: 注意id1要映射到扫飞镖的摄像头，id2要映射到扫环境tag的摄像头
    start_continuous_detection(camera_id=1, tag_size=20.5)
    start_continuous_detection(camera_id=2, tag_size=300)
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
            ok, info = _send_move_and_wait_ack(link, 0, -1600, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 1000, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # ---------------------------下面不需要
            # ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 先走到大概位置
            # ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)            
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            last_state = state
            state = State.GO_DART1
            
                

        elif state == State.GO_DART1:
            ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            DART_TAG_IDS = {247, 248, 268, 270, 282, 297, 367, 509}
            # flag1 = False
            fire_dart1 = dart1_num
            grabed = 0
            first_fire = False
            for times in range(dart1_num):
                print(f"[INFO] 准备抓取第 {times+1} 个常规飞镖")
                # # 后退
                # ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                # time.sleep(0.5)
                # if flag1 == False:
                #     ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)
                #     if not ok:
                #         print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                #     flag1 = True
                #     time.sleep(2.0)
                # ----------------------------------
                # ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                flag_de = False
                move_num = 0
                is_detected = False
                # first_fire = False
                # dist = 150 + grabed * 150
                # ok, info = _send_move_and_wait_ack(link, 0, dist, timeout_s=4.0)            
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                # ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                while True:
                    # time.sleep(1)
                    result = detect_dart(1)
                    if result is None or result['tag_id'] not in DART_TAG_IDS:
                        if is_detected:
                            print("[WARN] 飞镖丢失，重新检测")
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        print("[ERR] 未检测到飞镖")
                        # time.sleep(1)
                        ok, info = _send_move_and_wait_ack(link, 0, 70, timeout_s=4.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                        ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                        move_num += 1
                        if move_num >= 5:
                            # 重新标定
                            ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            time.sleep(0.5)
                            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            move_num = 0
                        continue
                    tvec = result['tvec']
                    tag_id = result['tag_id']
                    move_num = 0
                    is_detected = True
                    
                    print(f"检测到tag {tag_id}, 位置: {tvec.flatten()}")

                    
                    # 计算移动量
                    movement = calculate_lateral_offset(tvec, target_offset_x=-70)
                    if abs(movement) < 2:
                        if flag_de == False:
                            flag_de = True
                            ok, info = _send_move_and_wait_ack(link, 400, 0, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        print("[INFO] 飞镖已在目标位置，无需移动")
                        break
                    print(f"[INFO] 需要横向移动 {-movement:.1f} mm")
                    if -movement > 0:
                        ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                    else:
                        ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                time.sleep(1.0)
                # ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                # ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
                # if not ok:
                #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                tries = 2 # 最多试两次
                ok, info = _send_grab(link, "", timeout_s=32.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                tries -= 1
                while tries > 0:
                    tries -= 1
                    result = detect_dart(1)
                    if result is None or result['tag_id'] not in DART_TAG_IDS:
                        tries = 0
                        continue
                    ok, info = _send_move_and_wait_ack(link, 85, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    ok, info = _send_grab(link, "", timeout_s=25.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                grabed += 1
                dart1_num -= 1
                update_parameters(dart1_num, dart2_num)
                # 发射
                print("[INFO] 准备发射")
                ok, info = _send_move_and_wait_ack(link, -100, 0, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                time.sleep(1.0)
                ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
                time.sleep(2.0)
                if first_fire == False:
                    ok, info = _send_move_and_wait_ack(link, 0, -200, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    ok, info = _send_fire(link, 131, timeout_s=8.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                    ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(1.0)
                    first_fire = True
                    ok, info = _send_rot_and_wait_ack(link, -90, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
                    time.sleep(1.0)
                    ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(1.0)
                else:
                    # TODO: 撞墙发射
                    ok, info = _send_move_and_wait_ack(link, 0, -700, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, 1600, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    print("[INFO] 发射")
                    ok, info = _send_fire(link, 134, timeout_s=8.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                    # 回到墙角
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, -300, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, 0, 300, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_rot_and_wait_ack(link, -90, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
                    time.sleep(2.0)
                    ok, info = _send_move_and_wait_ack(link, 0, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, 0, -1600, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    # 1
                    # ok, info = _send_move_and_wait_ack(link, -300, 0, timeout_s=4.0)
                    # if not ok:
                    #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    # time.sleep(0.5)
                    # ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=4.0)
                    # if not ok:
                    #     print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
                    # time.sleep(0.5)
                    # 2
                    ok, info = _send_move_and_wait_ack(link, 700, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, 0, -500, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(0.5)
                    ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    time.sleep(1.0)
                    ok, info = _send_move_and_wait_ack(link, -150, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    # flag_de = False
                    dist = 200 + grabed * 150
                    ok, info = _send_move_and_wait_ack(link, 0, dist, timeout_s=4.0)            
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    ok, info = _send_move_and_wait_ack(link, 1, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            print("[INFO] 战术飞镖发射完成")
            last_state = state
            state = State.DONE

        elif state == State.GO_DART2:
            """
            得先转180度
            """
            ok, info = _send_move_and_wait_ack(link, 0, 300, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, -300, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_rot_and_wait_ack(link, 180, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            print("[INFO] 准备上台阶")
            # 上楼梯前得先贴墙，直接往前贴可行？如果不准还是得再侧着贴另一边
            ok, info = _send_move_and_wait_ack(link, 2000, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 0, 400, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_upstairs(link, timeout_s=30.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 上楼梯失败: {info}")
            print("[INFO] 上楼梯完成，撞墙")
            ok, info = _send_move_and_wait_ack(link, 0, 2000, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 400, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            print("[INFO] 撞墙完成，准备抓取战略飞镖")
            # # 后退
            # ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # # 右移
            # ok, info = _send_move_and_wait_ack(link, 0, -200, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            last_state = state
            state = State.FIND_DART2
            
        elif state == State.FIND_DART2:
            """
            先抓三个，再抓两个，直接定位中间的tag进行打击
            """
            DART_TAG_IDS = {247, 248, 268, 270, 282, 297, 367, 509}
            if recover_flag == False:

            
            # range_num = min(box_num, dart2_num)
            # dart2_grab = range_num
                print(f"[INFO] 正常状态")
                flag2 = False
                if DART2_lr == 0:
                    dist = -1
                    range_num = 3
                    dart2_grab = 3
                else:
                    dist = 1
                    range_num = 2
                    dart2_grab = 2
                for _ in range(range_num):
                    ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    if flag2 == False:
                        if DART2_lr == 0:
                            ok, info = _send_move_and_wait_ack(link, 0, -200, timeout_s=4.0)
                        else:
                            ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                        flag2 = True
                    while True:
                        result = detect_dart(1)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            print("[ERR] 未检测到飞镖")
                            ok, info = _send_move_and_wait_ack(link, 0, dist, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        
                        tvec = result['tvec']
                        tag_id = result['tag_id']
                        
                        print(f"检测到tag {tag_id}, 位置: {tvec.flatten()}")
                        
                        # 计算移动量
                        # TODO: 位置不一样，具体值需要调
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
                    time.sleep(1.0)
                    ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    tries = 3 # 最多试三次
                    ok, info = _send_grab(link, "", timeout_s=25.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                    tries -= 1
                    while tries > 0:
                        tries -= 1
                        result = detect_dart(1)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            tries = 0
                            continue
                        ok, info = _send_pick_dart(link, timeout_s=25.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                    dart2_num -= 1
                    update_parameters(dart1_num, dart2_num)
            else:
                print(f"[INFO] 恢复状态")
                # flag2 = False
                ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                ok, info = _send_move_and_wait_ack(link, 0, -200, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                time.sleep(2.0)
                if dart2_num > 2:
                    range_num = dart2_num - 2
                    dart2_grab = range_num
                    next_state = State.ATTACK2
                else:
                    range_num = dart2_num
                    dart2_grab = range_num
                    next_state = State.ATTACK3
                for _ in range(range_num):
                    ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    # if flag2 == False:
                    #     ok, info = _send_move_and_wait_ack(link, 0, -200, timeout_s=4.0)
                    #     if not ok:
                    #         print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    #     flag2 = True
                    while True:
                        result = detect_dart(1)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            print("[ERR] 未检测到飞镖")
                            ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        
                        tvec = result['tvec']
                        tag_id = result['tag_id']
                        
                        print(f"检测到tag {tag_id}, 位置: {tvec.flatten()}")
                        
                        # 计算移动量
                        # TODO: 位置不一样，具体值需要调
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
                    time.sleep(1.0)
                    ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    tries = 3 # 最多试三次
                    ok, info = _send_grab(link, "", timeout_s=25.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                    tries -= 1
                    while tries > 0:
                        tries -= 1
                        result = detect_dart(1)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            tries = 0
                            continue
                        ok, info = _send_grab(link, "", timeout_s=25.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 抓取失败: {info}")
                    dart2_num -= 1
                    update_parameters(dart1_num, dart2_num)
                last_state = state
                state = next_state
                recover_flag = False
                continue
            last_state = state
            if DART2_lr == 0:
                state = State.ATTACK2
            else:
                state = State.ATTACK3

        elif state == State.ATTACK1:
            # TODO: 具体定位：如何走，发射转速如何
            # 后退
            ok, info = _send_move_and_wait_ack(link, -100, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 旋转
            ok, info = _send_rot_and_wait_ack(link, 90, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            # 后面撞墙
            ok, info = _send_move_and_wait_ack(link, 0, -500, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 右面撞墙
            ok, info = _send_move_and_wait_ack(link, 500, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            time.sleep(1.0)
            # 发射
            for _ in range(fire_dart1):
                ok, info = _send_fire(link, rpm=1200, timeout_s=8.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                time.sleep(2.0)
            last_state = state
            state = State.DONE
        
        elif state == State.ATTACK2:
            DART_TAG_IDS = set(range(5, 7))
            # 后退
            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 旋转
            ok, info = _send_rot_and_wait_ack(link, -180, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            while True:
                result = detect_dart(2)
                if result is None or result['tag_id'] not in DART_TAG_IDS:
                    print("[ERR] 未检测定位tag")
                    ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    continue
                tvec = result['tvec']
                tag_id = result['tag_id']
                print(f"检测到定位tag {tag_id}, 位置: {tvec.flatten()}")
                if tag_id == 5:
                    movement = calculate_lateral_offset(tvec, target_offset_x=0)
                    if abs(movement) < 4:
                        print("[INFO] 位置已在目标位置，无需移动")
                        break
                else:
                    while True:
                        DART_TAG_IDS = set(range(6, 7))
                        result = detect_dart(2)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            print("[ERR] 未检测定位tag")
                            ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        tvec = result['tvec']
                        tag_id = result['tag_id']
                        print(f"检测到定位tag {tag_id}, 位置: {tvec.flatten()}")
                        movement = calculate_lateral_offset(tvec, target_offset_x=0)
                        if abs(movement) < 4:
                            print("[INFO] 位置已在目标位置，无需移动")
                            break
                        print(f"[INFO] 需要横向移动 {-movement:.1f} mm")
                        if -movement > 0:
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                        else:
                            ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    break
            print(f"[INFO] 定位完成")
            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 发射
            for _ in range(dart2_grab):
                ok, info = _send_fire(link, rpm=1200, timeout_s=8.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                time.sleep(2.0)
            # 往右靠墙
            ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_rot_and_wait_ack(link, 180, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 0, -2000, timeout_s=8.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 290, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            DART2_lr = 1
            last_state = state
            state = State.FIND_DART2


        elif state == State.ATTACK3:
            DART_TAG_IDS = set(range(4, 6))
            # 后退
            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 旋转
            ok, info = _send_rot_and_wait_ack(link, -180, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            while True:
                result = detect_dart(2)
                if result is None or result['tag_id'] not in DART_TAG_IDS:
                    print("[ERR] 未检测定位tag")
                    ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                    if not ok:
                        print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    continue
                tvec = result['tvec']
                tag_id = result['tag_id']
                print(f"检测到定位tag {tag_id}, 位置: {tvec.flatten()}")
                if tag_id == 5:
                    movement = calculate_lateral_offset(tvec, target_offset_x=0)
                    if abs(movement) < 4:
                        print("[INFO] 位置已在目标位置，无需移动")
                        break
                else:
                    while True:
                        DART_TAG_IDS = set(range(4, 5))
                        result = detect_dart(2)
                        if result is None or result['tag_id'] not in DART_TAG_IDS:
                            print("[ERR] 未检测定位tag")
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                            if not ok:
                                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                            continue
                        tvec = result['tvec']
                        tag_id = result['tag_id']
                        print(f"检测到定位tag {tag_id}, 位置: {tvec.flatten()}")
                        movement = calculate_lateral_offset(tvec, target_offset_x=0)
                        if abs(movement) < 4:
                            print("[INFO] 位置已在目标位置，无需移动")
                            break
                        print(f"[INFO] 需要横向移动 {-movement:.1f} mm")
                        if -movement > 0:
                            ok, info = _send_move_and_wait_ack(link, 0, 1, timeout_s=4.0)
                        else:
                            ok, info = _send_move_and_wait_ack(link, 0, -1, timeout_s=4.0)
                        if not ok:
                            print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
                    break
            print(f"[INFO] 定位完成")
            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 发射
            fire_times = dart2_grab # 发射次数
            for _ in range(fire_times):
                ok, info = _send_fire(link, rpm=1200, timeout_s=8.0)
                if not ok:
                    print(f"[ERR] 阶段 {state.name} 发射失败: {info}")
                time.sleep(2.0)
            # # 往右靠墙
            # ok, info = _send_move_and_wait_ack(link, 200, 0, timeout_s=6.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # ok, info = _send_rot_and_wait_ack(link, 180, timeout_s=4.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            # ok, info = _send_move_and_wait_ack(link, 0, -2000, timeout_s=8.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # ok, info = _send_move_and_wait_ack(link, 300, 0, timeout_s=6.0)
            # if not ok:
            #     print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # DART2_lr = 1
            last_state = state
            state = State.DONE
            
        elif state == State.GO_HIGH:
            print("[INFO] 上台阶")
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
            # 离开墙进行转向
            ok, info = _send_move_and_wait_ack(link, -200, 0, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 0, 200, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_rot_and_wait_ack(link, -90, timeout_s=4.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 旋转失败: {info}")
            # 撞墙
            ok, info = _send_move_and_wait_ack(link, 0, 300, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 290, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            # 上台阶
            ok, info = _send_upstairs(link, timeout_s=30.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 上楼梯失败: {info}")
            print("[INFO] 上楼梯完成，撞墙")
            ok, info = _send_move_and_wait_ack(link, 0, 1000, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            ok, info = _send_move_and_wait_ack(link, 400, 0, timeout_s=6.0)
            if not ok:
                print(f"[ERR] 阶段 {state.name} 移动失败: {info}")
            print("[INFO] 撞墙完成，准备抓取战略飞镖")
            last_state = state
            state = State.FIND_DART2

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
