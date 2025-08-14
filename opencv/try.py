import cv2
import numpy as np
from pupil_apriltags import Detector
from package import load_calib

# ===== 1) 相机参数（替换为你的标定结果）=====
npz_path = "mtx_dist/calib.npz"  # 替换为你的标定文件路径
mtx, dist = load_calib(npz_path)

fx, fy, cx, cy = float(mtx[0,0]), float(mtx[1,1]), float(mtx[0,2]), float(mtx[1,2])

# 标签实际边长（米），用尺子量内黑框的白色方块边长
tag_size = 0.12

# ===== 2) 构建检测器 =====
detector = Detector(
    families="tag36h11",       # 常见家族，你打印的很可能是这个
    nthreads=2,
    quad_decimate=1.5,         # >1 更快、鲁棒性稍降；近距离可调小到1.0
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

# ===== 3) 打开相机 =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("摄像头无法打开")

def euler_from_R(R):
    """
    将旋转矩阵换成 roll(x), pitch(y), yaw(z)（单位：度）
    坐标系：OpenCV摄像机系 x向右, y向下, z向前
    采用ZYX顺序（yaw->pitch->roll）
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
    else:
        # 近奇异情况的处理
        yaw   = np.degrees(np.arctan2(-R[0,1], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = 0.0
    return roll, pitch, yaw

axis_len = tag_size * 0.5  # 画坐标轴的长度

while True:
    ok, frame = cap.read()
    if not ok: 
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== 4) 检测 + 位姿估计 =====
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size
    )

    for det in results:
        # 角点顺序：四个点，已按AprilTag标准排好
        corners = det.corners.astype(np.int32)
        cv2.polylines(frame, [corners], True, (0,255,0), 2)

        # 文本：ID
        c = det.center.astype(int)
        cv2.putText(frame, f"id:{det.tag_id}", (c[0]+6, c[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 位姿（R,t）：以相机为原点，单位：米
        R = det.pose_R
        t = det.pose_t.reshape(3)

        # 距离（相机到标签中心）
        dist_m = np.linalg.norm(t)

        # 欧拉角（roll/pitch/yaw）
        roll, pitch, yaw = euler_from_R(R)

        # 在图上打印
        cv2.putText(frame, f"dist:{dist_m:.3f}m", (c[0]+6, c[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2)
        cv2.putText(frame, f"rpy:{roll:.1f},{pitch:.1f},{yaw:.1f}", (c[0]+6, c[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2)

        # 画 3D 坐标轴
        rvec, _ = cv2.Rodrigues(R)
        obj_axes = np.float32([[0,0,0],
                               [axis_len,0,0],
                               [0,axis_len,0],
                               [0,0,axis_len]])
        imgpts, _ = cv2.projectPoints(obj_axes, rvec, t, mtx, dist)
        p0, px, py, pz = imgpts.reshape(-1,2).astype(int)
        cv2.line(frame, p0, px, (0,0,255), 3)   # X-红
        cv2.line(frame, p0, py, (0,255,0), 3)   # Y-绿
        cv2.line(frame, p0, pz, (255,0,0), 3)   # Z-蓝

    cv2.imshow("AprilTag Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
