import cv2
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector

class Detector:
    def __init__(self, camera_matrix, tag_size, tag_families="tag36h11", hsv_params=None):
        """
        camera_matrix: 3x3 相机内参矩阵 (numpy array)
        dist_coeffs: 畸变系数 (numpy array)
        tag_size: AprilTag 实际边长 (米)
        tag_families: AprilTag 字典族
        hsv_params: dict, 飞镖颜色HSV范围等 {'lower':(h,s,v), 'upper':(h,s,v)}
        """
        self.camera_matrix = camera_matrix
        self.tag_size = tag_size
        self.hsv_params = hsv_params if hsv_params else {}
        
        fx, fy, cx, cy = float(camera_matrix[0,0]), float(camera_matrix[1,1]), float(camera_matrix[0,2]), float(camera_matrix[1,2])
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        # 初始化 AprilTag 检测器
        self.tag_detector = AprilTagDetector(
            families=tag_families,
            nthreads=2, # 多线程加速
            quad_decimate=1.0, # 图像金字塔下采样倍数。>1 表示先把图缩小再做边缘/四边形搜索。
            refine_edges=True # 是否对检测到的四边形角点做亚像素级边缘细化。开启（True）通常能提高角点定位精度 → 位姿估计更准，但会略微变慢。
        )

    def detect_tags(self, frame, estimate_pose=True):
        """
        输入:
            frame    : 已经去畸变的图像 (BGR 或 Gray)
            K_new    : 去畸变后对应的内参 (3x3)
            estimate_pose: 是否估计位姿
        返回:
            detections: 列表，每个元素包含
                tag_id   : int
                pose_R   : 3x3 numpy array
                pose_t   : 3x1 numpy array
                corners  : 4x2 numpy array (无畸变图像坐标系)
                reproj_err: float (像素单位)
        """
        # 1) 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        
        # 2) 内参拆分
        fx, fy = K_new[0, 0], K_new[1, 1]
        cx, cy = K_new[0, 2], K_new[1, 2]
        
        # 3) 在无畸变图像上检测 tag
        results = self.tag_detector.detect(
            gray,
            estimate_tag_pose=estimate_pose,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.tag_size
        )

        # 4) 组装结果
        detections = []
        for r in results:
            det = {
                "tag_id": int(r.tag_id),
                "pose_R": None,
                "pose_t": None,
                "corners": np.asarray(r.corners, dtype=np.float64),
                "reproj_err": None,
            }

            if estimate_pose:
                R = np.asarray(r.pose_R, dtype=np.float64)
                t = np.asarray(r.pose_t, dtype=np.float64).reshape(3, 1)
                det["pose_R"] = R
                det["pose_t"] = t

                # 5) 计算重投影误差
                # tag 在世界坐标系下的 4 个角点 (假设 tag 平面在 z=0)
                s = self.tag_size / 2.0
                obj_pts = np.array([
                    [-s, -s, 0],
                    [ s, -s, 0],
                    [ s,  s, 0],
                    [-s,  s, 0],
                ], dtype=np.float64)

                img_pts_proj, _ = cv2.projectPoints(
                    obj_pts,
                    cv2.Rodrigues(R)[0],  # rvec
                    t, K_new, None        # 无畸变参数
                )
                img_pts_proj = img_pts_proj.reshape(-1, 2)
                img_pts_obs = det["corners"]

                err = np.linalg.norm(img_pts_proj - img_pts_obs, axis=1).mean()
                print(f"[DEBUG] tag_id={det['tag_id']}, reproj_err={err:.2f}")
                det["reproj_err"] = float(err)

            detections.append(det)

        return detections
        
        # H, W = frame.shape[:2]
        
        # # 1) 读取原始内参与畸变（已保证存在）
        # K_orig = np.asarray(self.camera_matrix, dtype=np.float64)
        # dist = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        
        # # 2) 懒加载/缓存去畸变映射；alpha=0 裁黑边（可通过 self.undist_alpha=0~1 调整）
        # alpha = float(getattr(self, "undist_alpha", 0.0))  # 0=裁黑边, 1=尽量保视场
        # cache = getattr(self, "_ud_cache", None)
        # need_reinit = True
        # if cache is not None:
        #     same_size  = cache.get("size")  == (W, H)
        #     same_alpha = cache.get("alpha") == alpha
        #     same_K     = np.array_equal(cache.get("K_orig"), K_orig)
        #     same_dist  = np.array_equal(cache.get("dist"), dist)
        #     need_reinit = not (same_size and same_alpha and same_K and same_dist)

        # if need_reinit:
        #     K_new, valid_roi = cv2.getOptimalNewCameraMatrix(K_orig, dist, (W, H), alpha, (W, H))
        #     map1, map2 = cv2.initUndistortRectifyMap(K_orig, dist, None, K_new, (W, H), cv2.CV_16SC2)
        #     self._ud_cache = {
        #         "map1": map1, "map2": map2,
        #         "K_new": K_new, "valid_roi": valid_roi,
        #         "size": (W, H), "alpha": alpha,
        #         "K_orig": K_orig.copy(), "dist": dist.copy(),
        #     }

        # map1 = self._ud_cache["map1"]; map2 = self._ud_cache["map2"]
        # K_new = self._ud_cache["K_new"]
        # x0, y0, ww, hh = self._ud_cache["valid_roi"]  # 去畸变后有效区域

        # # 3) 去畸变 + 裁掉黑边
        # undist_full = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        # undist = undist_full[y0:y0+hh, x0:x0+ww]

        # # 4) 计算裁剪后对应的新内参（主点需减去裁剪偏移）
        # K_crop = K_new.copy()
        # K_crop[0, 2] -= x0
        # K_crop[1, 2] -= y0

        # # 存一下当前使用的内参与 ROI（可用于外部可视化/投影）
        # self._K_used = K_crop
        # self._undist_roi = (x0, y0, ww, hh)

        # fx, fy, cx, cy = map(float, (K_crop[0,0], K_crop[1,1], K_crop[0,2], K_crop[1,2]))

        # # 5) 在“无畸变+裁剪”的图上检测（针孔模型，畸变=0）
        # gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        # results = self.tag_detector.detect(
        #     gray,
        #     estimate_tag_pose=estimate_pose,
        #     camera_params=(fx, fy, cx, cy),   # ★关键：用更新后的参数
        #     tag_size=self.tag_size
        # )

        # # 6) 组装返回
        # detections = []
        # for r in results:
        #     detections.append({
        #         "tag_id": int(r.tag_id),
        #         "pose_R": None if not estimate_pose else np.asarray(r.pose_R, dtype=np.float64),
        #         "pose_t": None if not estimate_pose else np.asarray(r.pose_t, dtype=np.float64).reshape(3,1),
        #         # 角点在“去畸变+裁剪后的图像坐标系”
        #         "corners": np.asarray(r.corners, dtype=np.float64)
        #     })
        # return detections


    # def detect_dart(self, frame):
    #     """
    #     检测飞镖（颜色 + 轮廓）
    #     返回:
    #       center: (x,y) or None
    #       contour: 最大轮廓 or None
    #     """
    #     if not self.hsv_params:
    #         return None, None

    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     lower = np.array(self.hsv_params['lower'], dtype=np.uint8)
    #     upper = np.array(self.hsv_params['upper'], dtype=np.uint8)
    #     mask = cv2.inRange(hsv, lower, upper)

    #     # 形态学操作去噪
    #     kernel = np.ones((5,5), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return None, None
    #     cnt = max(contours, key=cv2.contourArea)
    #     if cv2.contourArea(cnt) < 100:  # 太小 # TODO: 100是面积，得调整
    #         return None, None
    #     M = cv2.moments(cnt) # 计算矩
    #     if M["m00"] == 0:
    #         return None, None
    #     cx = int(M["m10"]/M["m00"])
    #     cy = int(M["m01"]/M["m00"])
    #     return (cx, cy), cnt

    # def detect_landmark(self, frame):
    #     """
    #     检测关键建筑（可以用模板匹配、特征点、颜色等）
    #     这里先留空
    #     """
    #     pass

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
