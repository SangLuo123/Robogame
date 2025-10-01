import cv2
import numpy as np
import threading
import time
import os
import sys
import apriltag

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from src.camera import Camera
from src.load import load_calib, load_config

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

# 显示相关的全局变量
display_running = False
display_thread = None

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

# 创建全局相机实例
camera_instance_cam1 = None
camera_instance_cam2 = None

def get_camera(camera_id):
    """获取全局相机实例"""
    global camera_instance_cam1, camera_instance_cam2
    
    if camera_id == 1:
        if camera_instance_cam1 is None:
            camera_instance_cam1 = Camera("/dev/cam_down")
            camera_instance_cam1.initialize()
        return camera_instance_cam1
    elif camera_id == 2:
        if camera_instance_cam2 is None:
            camera_instance_cam2 = Camera("/dev/cam_up")
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
            result = detect_dart_in_frame(img, tag_size, camera_matrix, dist_coeffs)
            
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

def stop_continuous_detection(camera_id=1):
    """停止持续检测"""
    global detection_running_cam1, camera_instance_cam1
    global detection_running_cam2, camera_instance_cam2
    
    if camera_id == 1:
        detection_running_cam1 = False
        if detection_thread_cam1:
            detection_thread_cam1.join(timeout=1.0)
        
        if camera_instance_cam1:
            camera_instance_cam1.release()
            camera_instance_cam1 = None
        print(f"停止摄像头{camera_id}持续检测")
    else:
        detection_running_cam2 = False
        if detection_thread_cam2:
            detection_thread_cam2.join(timeout=1.0)
        
        if camera_instance_cam2:
            camera_instance_cam2.release()
            camera_instance_cam2 = None
        print(f"停止摄像头{camera_id}持续检测")

def detect_dart_in_frame(img, tag_size, camera_matrix=None, dist_coeffs=None):
    """在指定帧中检测飞镖"""
    # 如果没有传入相机参数，使用默认的第一个摄像头参数（保持向后兼容）
    if camera_matrix is None:
        camera_matrix = camera_matrix1
    if dist_coeffs is None:
        dist_coeffs = dist_coeffs1
    
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
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }

def draw_detection_results(frame, result, camera_id):
    """在图像上绘制检测结果和位姿信息"""
    if result is None:
        # 没有检测到tag，显示提示信息
        cv2.putText(frame, f"Camera {camera_id}: No Tag Detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    # 绘制tag的边界框
    corners = result['corners'].astype(int)
    for i in range(4):
        cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
    
    # 绘制角点
    for corner in corners:
        cv2.circle(frame, tuple(corner), 5, (255, 0, 0), -1)
    
    # 绘制tag ID
    center = np.mean(corners, axis=0).astype(int)
    cv2.putText(frame, f"ID: {result['tag_id']}", 
               tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 提取位姿信息
    tvec = result['tvec'].flatten()
    rvec = result['rvec'].flatten()
    
    # 计算距离（欧几里得距离）
    distance = np.linalg.norm(tvec)
    
    # 显示位姿信息
    info_lines = [
        f"Camera {camera_id} - Tag {result['tag_id']}",
        f"Distance: {distance:.1f}mm",
        f"X: {tvec[0]:.1f}mm",
        f"Y: {tvec[1]:.1f}mm", 
        f"Z: {tvec[2]:.1f}mm",
        f"Rot: ({rvec[0]:.2f}, {rvec[1]:.2f}, {rvec[2]:.2f})"
    ]
    
    # 在图像左上角显示信息
    y_offset = 30
    for i, line in enumerate(info_lines):
        color = (0, 255, 0) if i == 0 else (255, 255, 255)  # 第一行用绿色，其他用白色
        font_size = 0.6 if i > 0 else 0.7
        cv2.putText(frame, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
        y_offset += 25
    
    return frame

def continuous_display():
    """持续显示线程函数"""
    global display_running
    
    window_name1 = "Camera 1 - AprilTag Detection"
    window_name2 = "Camera 2 - AprilTag Detection"
    
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
    
    while display_running:
        try:
            # 获取两个摄像头的最新帧和检测结果
            frame1 = get_latest_frame(1)
            result1 = get_latest_detection(1)
            frame2 = get_latest_frame(2) 
            result2 = get_latest_detection(2)
            
            display_frame1 = None
            display_frame2 = None
            
            # 处理摄像头1的显示
            if frame1 is not None:
                display_frame1 = frame1.copy()
                if result1 is not None:
                    display_frame1 = draw_detection_results(display_frame1, result1, 1)
                else:
                    cv2.putText(display_frame1, "Camera 1: No Tag Detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 处理摄像头2的显示
            if frame2 is not None:
                display_frame2 = frame2.copy()
                if result2 is not None:
                    display_frame2 = draw_detection_results(display_frame2, result2, 2)
                else:
                    cv2.putText(display_frame2, "Camera 2: No Tag Detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示图像
            if display_frame1 is not None:
                cv2.imshow(window_name1, display_frame1)
            if display_frame2 is not None:
                cv2.imshow(window_name2, display_frame2)
            
            # 检查按键输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 按q或ESC退出
                break
                
            time.sleep(0.03)  # 约30fps
            
        except Exception as e:
            print(f"显示错误: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()

def start_continuous_display():
    """开始持续显示"""
    global display_running, display_thread
    
    if display_running:
        return
        
    display_running = True
    display_thread = threading.Thread(target=continuous_display, daemon=True)
    display_thread.start()
    print("开始实时显示")

def stop_continuous_display():
    """停止持续显示"""
    global display_running
    display_running = False
    if display_thread:
        display_thread.join(timeout=1.0)
    print("停止实时显示")

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
    stop_continuous_display()
    stop_continuous_detection(1)
    stop_continuous_detection(2)
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    try:
        # 启动两个摄像头的检测
        start_continuous_detection(camera_id=1, tag_size=20.5)
        start_continuous_detection(camera_id=2, tag_size=15.0)
        
        # 启动显示
        start_continuous_display()
        
        # 等待显示线程结束
        if display_thread:
            display_thread.join()
            
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        cleanup()