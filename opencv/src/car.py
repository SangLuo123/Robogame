import numpy as np
import cv2
from src.transform import rt_to_T, T_inv, T_to_xyyaw

# ======== 小车类 ========
class RobotCar:
    def __init__(self, T_robot_cam, tag_map):
        """
        T_robot_cam: 4x4, 机器人←相机的外参矩阵
        tag_map: dict[tag_id] = T_world_tag(4x4), 世界←tag 的位姿
        """
        self.pose_world = np.eye(4)  # 小车在世界系下的位姿矩阵（初始原点）
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # 度

        self.T_robot_cam = T_robot_cam
        self.tag_map = tag_map

    def update_pose_from_tag(self, det):
        # 传参det，根据单个tag进行更新，tag的选取在调用处完成
        """
        根据单个 AprilTag 检测结果更新小车位姿
        det: pupil_apriltags 检测结果，需包含 tag_id, pose_R, pose_t
        """
        # print("tag_id =", det["tag_id"], type(det["tag_id"]))
        # print("map keys =", list(self.tag_map.keys()), [type(k) for k in self.tag_map.keys()])
        # print("contains? ->", det["tag_id"] in self.tag_map)
        if det["tag_id"] not in self.tag_map:
            return False
        T_world_tag = self.tag_map[det["tag_id"]]
        T_cam_tag = rt_to_T(det["pose_R"], det["pose_t"])

        # 世界←小车
        T_world_robot = T_world_tag @ T_inv(T_cam_tag) @ T_inv(self.T_robot_cam)
        self.pose_world = T_world_robot
        self.x, self.y, self.yaw = T_to_xyyaw(T_world_robot)
        print(f"更新小车位姿: x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.1f}°")
        return True

    def get_pose(self):
        """返回 (x, y, yaw_deg)"""
        return self.x, self.y, self.yaw

    # # 有待更新
    # def compute_control_to_target(self, target_x, target_y, kv=0.5, ktheta=1.0):
    #     """
    #     根据当前位姿计算去目标点的简单控制指令 (vx, vy, omega)
    #     target_x, target_y: 目标点世界坐标（米）
    #     kv, ktheta: PID比例系数
    #     """
    #     # 更新为全向控制
    #     # dx = target_x - self.x
    #     # dy = target_y - self.y

    #     # # 目标方向角（世界系）
    #     # target_yaw = np.degrees(np.arctan2(dy, dx))
    #     # # 航向误差
    #     # yaw_err = (target_yaw - self.yaw + 180) % 360 - 180
    #     # # 距离误差
    #     # dist_err = np.hypot(dx, dy)

    #     # v = kv * dist_err  # 简单比例控制
    #     # omega = np.radians(ktheta * yaw_err)  # 转成弧度/秒
    #     # return v, omega

    def __repr__(self):
        return f"RobotCar(x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.1f}°)"