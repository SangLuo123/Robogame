import cv2
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector

class Detector:
    def __init__(self, camera_matrix, dist_coeffs, tag_size, tag_families="tag36h11", hsv_params=None):
        """
        camera_matrix: 3x3 相机内参矩阵 (numpy array)
        dist_coeffs: 畸变系数 (numpy array)
        tag_size: AprilTag 实际边长 (米)
        tag_families: AprilTag 字典族
        hsv_params: dict, 飞镖颜色HSV范围等 {'lower':(h,s,v), 'upper':(h,s,v)}
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.hsv_params = hsv_params if hsv_params else {}
        
        fx, fy, cx, cy = float(camera_matrix[0,0]), float(camera_matrix[1,1]), float(camera_matrix[0,2]), float(camera_matrix[1,2])
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        # 初始化 AprilTag 检测器
        self.tag_detector = AprilTagDetector(
            families=tag_families,
            nthreads=2, # 多线程加速
            quad_decimate=1.5, # 图像金字塔下采样倍数。>1 表示先把图缩小再做边缘/四边形搜索。
            refine_edges=True # 是否对检测到的四边形角点做亚像素级边缘细化。开启（True）通常能提高角点定位精度 → 位姿估计更准，但会略微变慢。
        )

    def detect_tags(self, frame, estimate_pose=True):
        """
        检测 AprilTag，返回检测结果列表
        每个元素包含:
          tag_id: int
          pose_R: 3x3 numpy array
          pose_t: 3x1 numpy array
          corners: 4x2 numpy array
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.tag_detector.detect(
            gray,
            estimate_tag_pose=estimate_pose, # 是否直接在检测阶段估计位姿。
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=self.tag_size
        )

        detections = []
        for r in results:
            det = {
                "tag_id": r.tag_id, # 标签 ID
                "pose_R": r.pose_R, # 相机坐标系下tag的旋转矩阵
                "pose_t": r.pose_t, # 相机坐标系下tag的平移向量
                "corners": r.corners # tag 在图像坐标系下的 四个角点坐标 (4×2 像素点)。
            }
            detections.append(det)
        return detections

    def detect_dart(self, frame):
        """
        检测飞镖（颜色 + 轮廓）
        返回:
          center: (x,y) or None
          contour: 最大轮廓 or None
        """
        if not self.hsv_params:
            return None, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.hsv_params['lower'], dtype=np.uint8)
        upper = np.array(self.hsv_params['upper'], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作去噪
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:  # 太小 # TODO: 100是面积，得调整
            return None, None
        M = cv2.moments(cnt) # 计算矩
        if M["m00"] == 0:
            return None, None
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        return (cx, cy), cnt

    def detect_landmark(self, frame):
        """
        检测关键建筑（可以用模板匹配、特征点、颜色等）
        这里先留空
        """
        pass

    def draw_results(self, frame, tags=None, dart=None):
        """
        在画面上画检测结果
        tags: detect_tags() 的返回
        dart: (center, contour)
        """
        if tags:
            for det in tags:
                pts = det["corners"].astype(int)
                cv2.polylines(frame, [pts], True, (0,255,0), 2)
                c = np.mean(pts, axis=0).astype(int)
                cv2.putText(frame, f"id:{det['tag_id']}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if dart and dart[0]:
            cv2.drawContours(frame, [dart[1]], -1, (0,0,255), 2)
            cv2.circle(frame, dart[0], 5, (0,0,255), -1)
            cv2.putText(frame, "Dart", (dart[0][0]+5, dart[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return frame
