import numpy as np
import cv2

# 坐标变换
"""
注意乘的矩阵是谁相对谁，是否求逆
"""

def rt_to_T(R, t):
    """
    R: 3x3 (list/tuple/ndarray)
    t: 3   (list/tuple/ndarray)
    return: 4x4 ndarray
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """4x4 变换矩阵求逆"""
    Ri = T[:3,:3].T
    ti = -Ri @ T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = Ri
    Ti[:3, 3] = ti
    return Ti

# def compose(*Ts) -> np.ndarray:
#     """按顺序左乘：compose(A,B,C)=A·B·C"""
#     T = np.eye(4)
#     for X in Ts:
#         T = T @ X
#     return T

# def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
#     """OpenCV风格 rvec(3,), tvec(3,) 转 4x4"""
#     R, _ = cv2.Rodrigues(rvec.reshape(3,1))
#     return rt_to_T(R, tvec)

def T_to_xyyaw(T: np.ndarray):
    """提取2D位姿 (x,y,yaw_deg)（世界系：x前, y左, z上；yaw绕z）"""
    x, y = T[0,3], T[1,3]
    yaw = np.degrees(np.arctan2(T[1,0], T[0,0]))  # ZYX里绕z的角
    return float(x), float(y), float(yaw)

# def estimate_robot_pose_world(T_world_tag: np.ndarray,
#                               pose_R_cam_tag: np.ndarray,
#                               pose_t_cam_tag: np.ndarray,
#                               T_robot_cam: np.ndarray) -> np.ndarray:
#     """
#     返回 T_world_robot (4x4)
#     """
#     T_cam_tag = rt_to_T(pose_R_cam_tag, pose_t_cam_tag)         # 相机←tag
#     T_world_robot = compose(
#         T_world_tag,
#         T_inv(T_cam_tag),          # = tag←相机
#         T_inv(T_robot_cam)         # = 相机←机器人
#     )
#     return T_world_robot

# def pick_best_pose_world_robot(detections, tag_map_T_world_tag, T_robot_cam):
#     """
#     detections: 来自 pupil_apriltags.detect(...) 的结果列表
#       - 每个 det 有 det.tag_id, det.pose_R, det.pose_t
#     tag_map_T_world_tag: dict[int] -> 4x4, 预先标定好的世界←tag
#     返回: T_world_robot or None
#     """
#     candidates = []
#     for det in detections:
#         tag_id = det.tag_id
#         if tag_id not in tag_map_T_world_tag:
#             continue
#         T_world_tag = tag_map_T_world_tag[tag_id]
#         T_wr = estimate_robot_pose_world(T_world_tag, det.pose_R, det.pose_t, T_robot_cam)
#         # 可用相机到tag的距离作为置信度（越近越准）
#         dist = np.linalg.norm(det.pose_t.reshape(3))
#         candidates.append((dist, T_wr))
#     if not candidates:
#         return None
#     candidates.sort(key=lambda x: x[0])  # 取最近
#     return candidates[0][1]

####################################################

# # —— 初始化：准备固定外参和tag地图 ——
# # 例：相机与车中心完全对齐、相机光心在车中心上方0.25m
# T_robot_cam = rt_to_T(np.eye(3), np.array([0.0, 0.0, 0.25]))  # 机器人←相机

# # tag 地图（世界←tag），你需要实测或定义。举例：id=0 放在世界原点，面向 +X。
# def eulerZYX_to_R(roll, pitch, yaw):
#     cr, sr = np.cos(roll), np.sin(roll)
#     cp, sp = np.cos(pitch), np.sin(pitch)
#     cy, sy = np.cos(yaw), np.sin(yaw)
#     Rz = np.array([[ cy,-sy,0],[ sy, cy,0],[0,0,1]])
#     Ry = np.array([[ cp, 0, sp],[  0, 1, 0],[-sp,0, cp]])
#     Rx = np.array([[  1, 0,  0],[  0,cr,-sr],[  0,sr, cr]])
#     return Rz @ Ry @ Rx

# tag_map_T_world_tag = {
#     0: rt_to_T(eulerZYX_to_R(0,0,0), np.array([0.0, 0.0, 0.0])),
#     1: rt_to_T(eulerZYX_to_R(0,0,np.deg2rad(90)), np.array([2.0, 0.0, 0.0])),
#     # ... 继续填你的标签地图（米 & 弧度）
# }

# # —— 运行时：拿到 pupil_apriltags 的 detections 后 ——
# T_world_robot = pick_best_pose_world_robot(detections, tag_map_T_world_tag, T_robot_cam)
# if T_world_robot is not None:
#     x, y, yaw_deg = T_to_xyyaw(T_world_robot)
#     # x,y,yaw_deg 就是“小车中心在世界系”的 2D 位姿，可直接做运动决策
