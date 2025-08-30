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

def T_to_xyyaw(T: np.ndarray):
    """提取2D位姿 (x,y,yaw_deg)（世界系：x前, y左, z上；yaw绕z）"""
    x, y = T[0,3], T[1,3]
    yaw = np.degrees(np.arctan2(T[1,0], T[0,0]))
    yaw_x0 = yaw + 90.0
    # 归一化到 (-180, 180]
    if yaw_x0 <= -180.0: yaw_x0 += 360.0
    elif yaw_x0 > 180.0: yaw_x0 -= 360.0

    return float(x), float(y), float(yaw_x0)

def tag_T_from_world_corners(tl, tr, br, bl):
    """
    输入四个角点在世界坐标系下的 3D 坐标（单位 m），顺序：tl, tr, br, bl
    Tag 局部系约定:
      - 原点: Tag 中心
      - x 轴: tl->tr（上边向右）
      - y 轴: tl->bl（左边向下）
      - z 轴: x × y（右手系，指向相机/法线）
    返回 世界<-Tag 的 4x4 齐次矩阵
    """
    tl = np.asarray(tl, float)
    tr = np.asarray(tr, float)
    br = np.asarray(br, float)
    bl = np.asarray(bl, float)

    center = 0.25 * (tl + tr + br + bl)

    x_w = _normalize(tr - tl)
    y0  = _normalize(bl - tl)
    # 正交化 y，防止量测误差导致不垂直
    y_w = _normalize(y0 - np.dot(y0, x_w) * x_w)
    z_w = _normalize(np.cross(x_w, y_w))  # 右手系

    R_wt = np.column_stack((x_w, y_w, z_w))  # world <- tag
    T_wt = np.eye(4, dtype=float)
    T_wt[:3, :3] = R_wt
    T_wt[:3, 3]  = center
    return T_wt

def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def rotz_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def build_T_robot_cam(cam):
    R = cam["R"]
    t = cam["t"]
    return rt_to_T(R, t)

def build_tag_map(cfg):
    tag_map = {}
    for k, v in cfg["tag_map"].items():
        tag_id = int(k)
        T = tag_T_from_world_corners(v["tl"], v["tr"], v["br"], v["bl"])
        tag_map[tag_id] = T
    return tag_map