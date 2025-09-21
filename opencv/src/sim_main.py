import cv2
import numpy as np
import apriltag
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from src.load import load_calib
from src.camera import Camera

# def main():
#     # 初始化摄像头（保持开启状态）
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("无法打开摄像头")
#         return
    
#     # 设置摄像头参数
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     cap.set(cv2.CAP_PROP_FPS, 30)
    
#     # 预热几帧
#     for _ in range(5):
#         cap.read()
    
#     print("按 'q' 键退出，按 's' 键保存当前帧")
    
#     while True:
#         # 读取帧
#         ret, frame = cap.read()
#         if not ret:
#             print("无法读取帧")
#             break
        
#         # 复制原始帧用于显示
#         display_frame = frame.copy()
        
#         try:
#             # 对当前帧进行检测
#             img_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
#             gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
            
#             # 检测AprilTag
#             results = detector.detect(gray)
            
#             if results:
#                 # 处理每个检测到的tag
#                 for i, tag in enumerate(results):
#                     corners = tag.corners.astype(np.float32)
                    
#                     # 在图像上绘制tag边界
#                     cv2.polylines(display_frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                    
#                     # 计算位姿
#                     tag_size = 0.1  # 单位：米
#                     object_points = np.array([
#                         [-tag_size/2, -tag_size/2, 0],
#                         [tag_size/2, -tag_size/2, 0], 
#                         [tag_size/2, tag_size/2, 0],
#                         [-tag_size/2, tag_size/2, 0]
#                     ], dtype=np.float32)
                    
#                     empty_dist_coeffs = np.zeros((5, 1), dtype=np.float32)
#                     success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, empty_dist_coeffs)
                    
#                     if success:
#                         # 绘制坐标系
#                         axis_length = 0.05
#                         axis_points = np.array([
#                             [0, 0, 0],
#                             [axis_length, 0, 0],
#                             [0, axis_length, 0],
#                             [0, 0, axis_length]
#                         ], dtype=np.float32)
                        
#                         img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, empty_dist_coeffs)
#                         img_points = img_points.astype(np.int32)
                        
#                         # 绘制坐标轴
#                         origin = tuple(img_points[0][0])
#                         cv2.line(display_frame, origin, tuple(img_points[1][0]), (0, 0, 255), 2)  # X轴-红
#                         cv2.line(display_frame, origin, tuple(img_points[2][0]), (0, 255, 0), 2)  # Y轴-绿
#                         cv2.line(display_frame, origin, tuple(img_points[3][0]), (255, 0, 0), 2)  # Z轴-蓝
                        
#                         # 显示tag信息和距离
#                         distance = np.linalg.norm(tvec)
#                         info_text = f"Tag {tag.tag_id}: Dist={distance:.2f}m, X={tvec[0][0]:.2f}m"
#                         cv2.putText(display_frame, info_text, (10, 30 + i*30), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
#                         # 显示角点坐标
#                         for j, corner in enumerate(corners):
#                             cv2.circle(display_frame, tuple(corner.astype(int)), 5, (255, 0, 255), -1)
#                             cv2.putText(display_frame, str(j), tuple(corner.astype(int)), 
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
#             # 显示FPS
#             fps_text = f"FPS: {int(1/(time.time() - start_time))}" if 'start_time' in locals() else "FPS: -"
#             cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
#         except Exception as e:
#             error_text = f"Error: {str(e)}"
#             cv2.putText(display_frame, error_text, (10, 60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # 显示图像
#         cv2.imshow('AprilTag Detection', display_frame)
        
#         # 记录时间用于计算FPS
#         start_time = time.time()
        
#         # 键盘控制
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             # 保存当前帧
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             filename = f"frame_{timestamp}.jpg"
#             cv2.imwrite(filename, frame)
#             print(f"已保存: {filename}")
    
#     # 释放资源
#     cap.release()
#     cv2.destroyAllWindows()

calibdown_path = os.path.join(ROOT, "calib", "calibdown.npz")

camera_matrix, dist_coeffs = load_calib(calibdown_path)

def setup_apriltag_detector():
    # 创建tag36h11检测器
    options = apriltag.DetectorOptions(families="tag36h11",
                                      border=1,
                                      nthreads=4,
                                      quad_decimate=1.0,
                                      quad_blur=0.0,
                                      refine_edges=True,
                                      refine_decode=False,
                                      refine_pose=False,
                                      debug=False,
                                      quad_contours=True)
    
    detector = apriltag.Detector(options)
    return detector

detector = setup_apriltag_detector()

camera = Camera("/dev/cam_down")

def get_image_from_camera():
    """获取相机图像"""
    return camera.get_frame()

def cleanup():
    """程序退出时清理资源"""
    camera.release()

def detect_dart():
    # 获取图像
    img = get_image_from_camera()
    
    # 去畸变（重要！）
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    
    # 检测AprilTag
    results = detector.detect(gray)
    
    if not results:
        return None
    
    # 获取第一个检测到的tag
    tag = results[0]
    
    # tag的4个角点（像素坐标）
    corners = tag.corners.astype(np.float32)
    
    # tag36h11的实际物理尺寸（需要根据你的tag大小调整）
    tag_size = 20.5  # 单位：毫米
    
    # tag的3D角点坐标（在tag坐标系中）
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],  # 左下
        [tag_size/2, -tag_size/2, 0],   # 右下  
        [tag_size/2, tag_size/2, 0],    # 右上
        [-tag_size/2, tag_size/2, 0]    # 左上
    ], dtype=np.float32)
    
    # 使用solvePnP计算精确位姿
    empty_dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, empty_dist_coeffs)
    
    if not success:
        return None
    
    return {
        'rvec': rvec,
        'tvec': tvec,
        'tag_id': tag.tag_id,
        'corners': corners
    }
    
def calculate_lateral_offset(tvec, target_offset_x=0):
    """
    计算横向移动距离，带调试信息
    """
    current_offset = tvec[0]  # 当前横向偏移
    movement_needed = current_offset - target_offset_x
    
    print(f"当前横向位置: {current_offset:.3f}mm")
    print(f"目标横向偏移: {target_offset_x:.3f}mm")
    print(f"需要移动: {movement_needed:.3f}mm ({'向右' if movement_needed > 0 else '向左'})")
    
    return movement_needed


# 更简单的版本（如果上面的太复杂）
def simple_main():
    
    print("按 'q' 键退出")
    
    while True:
        img = get_image_from_camera()
        if img is None:
            break

        display_frame = img.copy()

        try:
            # 使用你的detect_dart函数
            result = detect_dart()
            
            if result:
                # 绘制检测结果
                corners = result['corners']
                tag_id = result['tag_id']
                tvec = result['tvec']
                
                # 绘制边界
                cv2.polylines(display_frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                
                # 显示信息
                info_text = f"Tag {tag_id}: X={tvec[0][0]:.3f}mm, y={tvec[1][0]:.3f}mm, Z={tvec[2][0]:.3f}mm"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # movement = calculate_lateral_offset(tvec, target_offset_x=0)
                # print(f"建议移动: {movement:.3f}m")
                
        except Exception as e:
            cv2.putText(display_frame, f"Error: {e}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('AprilTag Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 选择运行哪个版本
    # main()  # 功能丰富的版本
    simple_main()  # 简单版本