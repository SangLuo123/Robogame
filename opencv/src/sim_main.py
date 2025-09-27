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
    """正确的检测方法：不去畸变，直接使用畸变参数"""
    # 获取图像（原始畸变图像）
    img = get_image_from_camera()
    
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
    
def calculate_lateral_offset(tvec, target_offset_x=0):
    """
    计算横向移动距离，带调试信息
    """
    if tvec is None:
        raise ValueError("tvec is None")
    
    tvec_arr = np.asarray(tvec).reshape(-1)   # -> shape (3,)
    if tvec_arr.size < 1:
        raise ValueError("tvec shape unexpected: " + str(tvec.shape))

    
    current_offset = float(tvec_arr[0])  # 当前横向偏移
    movement_needed = current_offset - target_offset_x
    
    # print(f"当前横向位置: {current_offset:.3f}mm")
    # print(f"目标横向偏移: {target_offset_x:.3f}mm")
    # print(f"需要移动: {movement_needed:.3f}mm ({'向右' if movement_needed > 0 else '向左'})")
    
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
            # print("1")
            
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
                # print("2")
                movement = calculate_lateral_offset(tvec, target_offset_x=-145)
                # print(f"建议向左（正方向）移动: {movement:.3f}m")
                text = f"y-right-move: -{movement:.3f}mm"
                cv2.putText(display_frame, text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        except Exception as e:
            # print("3")
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