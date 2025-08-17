# src/main.py
import os, sys, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# 项目根路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.comm import SerialLink  # 你已更新的三通道速度协议

# ========== 小工具 ==========
def wrap_pi(a: float) -> float:
    """把角度差规约到 [-pi, pi)"""
    return (a + math.pi) % (2 * math.pi) - math.pi

def body_error(x: float, y: float, th: float, xg: float, yg: float) -> Tuple[float, float]:
    """世界误差旋到机体系：返回 (ex 前向, ey 左向)"""
    dx, dy = xg - x, yg - y
    c, s = math.cos(th), math.sin(th)
    ex =  c * dx + s * dy
    ey = -s * dx + c * dy
    return ex, ey

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interp1(xy_list: List[Tuple[float, float]], x: float) -> float:
    """
    一维线性插值：给定 [(x0,y0), (x1,y1), ...]，返回 x 的 y 值
    - x 超出范围时做首尾外推（线性）
    """
    pts = sorted(xy_list, key=lambda p: p[0])
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return lerp(y0, y1, t)
    if x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        t = 1.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return lerp(y0, y1, t)
    # 区间内
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return lerp(y0, y1, t)
    return pts[-1][1]

# ========== 控制器 ==========
@dataclass
class OmniCtrlCfg:
    kx: float = 1.0
    ky: float = 1.0
    kth: float = 2.0
    vxmax: float = 0.6
    vymax: float = 0.6
    wmax: float = 2.5
    pos_tol: float = 0.05          # 到点阈值（m）
    yaw_tol: float = math.radians(3.0)  # 到角阈值（rad）

def omni_controller(x, y, th, xg, yg, thg, cfg: OmniCtrlCfg):
    ex, ey = body_error(x, y, th, xg, yg)
    eth = wrap_pi(thg - th)
    vx = clamp(cfg.kx * ex, -cfg.vxmax, cfg.vxmax)
    vy = clamp(cfg.ky * ey, -cfg.vymax, cfg.vymax)
    w  = clamp(cfg.kth * eth, -cfg.wmax, cfg.wmax)
    pos_err = math.hypot(ex, ey)
    yaw_err = abs(eth)
    return vx, vy, w, pos_err, yaw_err

# ========== 位姿来源（占位，等你接入里程计/AprilTag融合） ==========
class PoseSource:
    """
    你需要在这里融合编码器/IMU/AprilTag，维护 (x,y,yaw_rad)，单位(m, m, rad)。
    现在先用占位数据，接口不变，方便以后直接接。
    """
    def __init__(self):
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0  # m, m, rad

    def update(self):
        # TODO: 接入你的检测器/里程计更新 (self.x, self.y, self.yaw)
        pass

    def get(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw

# ========== 瞄准/发射参数 ==========
@dataclass
class FireTables:
    """
    距离(m) -> RPM（示例表，需你现场标定后替换）
    建议每 0.5 m 一个点，目标“顶面开口中心”的水平距离。
    """
    base: List[Tuple[float, float]] = None      # 大本营表
    outpost: List[Tuple[float, float]] = None   # 哨所表

    def __post_init__(self):
        if self.base is None:
            # (距离m, RPM)
            self.base = [(2.0, 2800), (2.5, 3000), (3.0, 3200), (3.5, 3450), (4.0, 3700)]
        if self.outpost is None:
            self.outpost = [(2.0, 3000), (2.5, 3250), (3.0, 3500), (3.5, 3800), (4.0, 4100)]

def pick_rpm(tables: FireTables, dist_m: float, target: str) -> int:
    if target == "base":
        rpm = interp1(tables.base, dist_m)
    elif target == "outpost":
        rpm = interp1(tables.outpost, dist_m)
    else:
        rpm = interp1(tables.base, dist_m)
    return int(round(rpm))

# ========== 任务/状态机 ==========
class State:
    INIT = "INIT"
    TO_AMMO = "TO_AMMO"
    PICK = "PICK"
    TO_FIRE = "TO_FIRE"
    AIM = "AIM"
    FIRE = "FIRE"
    LOOP = "LOOP"
    ABORT = "ABORT"
    END = "END"

@dataclass
class Waypoint:
    x: float
    y: float
    yaw: Optional[float] = None  # 目标朝向（rad），None 表示不要求

@dataclass
class MissionCfg:
    # 关键位姿（示例，按你的场地坐标改）
    ammo_dock: Waypoint = Waypoint(0.8, 0.5, 0.0)      # 弹药库预抓取位
    fire_pos: Waypoint = Waypoint(2.5, 1.2, None)      # 发射位（可在远程区附近）
    fire_target: str = "base"                           # "base" 或 "outpost"
    # 控制器
    ctrl: OmniCtrlCfg = OmniCtrlCfg()
    # 时间/节拍
    aim_settle_time: float = 0.6     # 到位驻留时间（s）
    spinup_time: float = 0.7         # 飞轮到速时间（s）
    fire_burst: int = 1              # 单次发射枚数（1~3）

# ========== 主逻辑 ==========
class RobotMain:
    def __init__(self):
        self.pose = PoseSource()
        self.link = SerialLink(port="/dev/stm32", baud=115200)
        self.tables = FireTables()
        self.cfg = MissionCfg()
        self.state = State.INIT
        self.state_t0 = time.time()
        self.start_t = time.time()
        self.has_ammo = True   # 简化：是否还有飞镖（可根据抓取/发射次数更新）
        self.ammo_count = 3    # 示例：本方弹药库携带量
        self.fired = 0

        # 回调（可选）
        self.link.on_ack = lambda ok, f: None
        self.link.on_enc = lambda enc: None

    def enter(self, s: str):
        self.state = s
        self.state_t0 = time.time()
        print(f"[STATE] -> {s}")

    def time_in_state(self) -> float:
        return time.time() - self.state_t0

    # --- 导航到点 ---
    def goto_waypoint(self, wp: Waypoint) -> bool:
        x, y, th = self.pose.get()
        thg = wp.yaw if wp.yaw is not None else th
        vx, vy, w, perr, terr = omni_controller(
            x, y, th, wp.x, wp.y, thg, self.cfg.ctrl
        )
        # 到点与到角
        at_pos = perr < self.cfg.ctrl.pos_tol
        at_yaw = (terr < self.cfg.ctrl.yaw_tol) if (wp.yaw is not None) else True

        if at_pos and at_yaw:
            self.link.send_vel_xyw(0.0, 0.0, 0.0)
            return True
        else:
            self.link.send_vel_xyw(vx, vy, w)
            return False

    # --- 目标距离（水平） ---
    def distance_to_target(self, target_xy: Tuple[float, float]) -> float:
        x, y, _ = self.pose.get()
        tx, ty = target_xy
        return math.hypot(tx - x, ty - y)

    # --- 瞄准（平面角） ---
    def aim_heading(self, target_xy: Tuple[float, float]) -> bool:
        x, y, th = self.pose.get()
        tx, ty = target_xy
        thg = math.atan2(ty - y, tx - x)
        eth = wrap_pi(thg - th)
        w = clamp(self.cfg.ctrl.kth * eth, -self.cfg.ctrl.wmax, self.cfg.ctrl.wmax)
        # 只做原地转（vx, vy = 0）
        if abs(eth) < self.cfg.ctrl.yaw_tol:
            self.link.send_vel_xyw(0.0, 0.0, 0.0)
            return True
        else:
            self.link.send_vel_xyw(0.0, 0.0, w)
            return False

    # --- 主循环 ---
    def run(self):
        self.link.open()
        print("[INFO] Start main. Ctrl+C to exit.")
        try:
            while True:
                # 1) 全局超时（比赛 6 分钟），示例 5:50 停止动作
                if time.time() - self.start_t > 5.8 * 60:
                    self.enter(State.END)

                # 2) 更新位姿（接入你的定位）
                self.pose.update()

                # 3) 状态机
                if self.state == State.INIT:
                    # 自检（串口/IMU/相机等）——此处简化为 0.5s 后进入 TO_AMMO
                    if self.time_in_state() > 0.5:
                        self.enter(State.TO_AMMO)

                elif self.state == State.TO_AMMO:
                    if self.goto_waypoint(self.cfg.ammo_dock):
                        # 到了预抓取位，驻留一下防抖
                        if self.time_in_state() > self.cfg.aim_settle_time:
                            self.enter(State.PICK)

                elif self.state == State.PICK:
                    # 固定动作抓取（交给下位机）
                    # TODO: 若你要视觉微调，这里先做横移微调再抓取
                    self.link.arm_preset("GRAB")  # $CGRAB#
                    time.sleep(0.8)               # 等动作完成（也可等待 ACK）
                    # 简化：抓一次就 +1
                    self.ammo_count = max(0, self.ammo_count - 1)
                    self.has_ammo = self.ammo_count > 0
                    self.enter(State.TO_FIRE)

                elif self.state == State.TO_FIRE:
                    if self.goto_waypoint(self.cfg.fire_pos):
                        if self.time_in_state() > self.cfg.aim_settle_time:
                            self.enter(State.AIM)

                elif self.state == State.AIM:
                    # 瞄准目标中心（请把目标世界坐标填入）
                    # 示例：大本营/哨所的世界坐标，需要你标定后填写
                    target_xy = (3.5, 2.0) if self.cfg.fire_target == "base" else (4.2, 2.4)
                    if self.aim_heading(target_xy):
                        if self.time_in_state() > self.cfg.aim_settle_time:
                            self.enter(State.FIRE)

                elif self.state == State.FIRE:
                    # 1) 设定 RPM（按水平距离插值）
                    target_xy = (3.5, 2.0) if self.cfg.fire_target == "base" else (4.2, 2.4)
                    dist = self.distance_to_target(target_xy)
                    rpm = pick_rpm(self.tables, dist, self.cfg.fire_target)
                    self.link.send_shooter_rpm(rpm)
                    time.sleep(self.cfg.spinup_time)

                    # 2) 触发发射（单发/连发）
                    for _ in range(self.cfg.fire_burst):
                        self.link.shooter_fire()
                        time.sleep(0.15)

                    self.fired += self.cfg.fire_burst

                    # 3) 决策：回去补给 or 继续循环 or 结束
                    if self.ammo_count <= 0:
                        # 示例：发完一次就回去再抓
                        self.ammo_count = 3      # 演示：重置为还能再抓 3 发（真实应读取感知）
                        self.enter(State.TO_AMMO)
                    else:
                        self.enter(State.LOOP)

                elif self.state == State.LOOP:
                    # 这里可以插入“去拿战略飞镖 / 换位 / 尝试另一目标”的逻辑
                    # 演示：简单等待 0.5s 后再去发射位继续 AIM
                    if self.time_in_state() > 0.5:
                        self.enter(State.AIM)

                elif self.state == State.ABORT:
                    # 异常：急停，等待人工/按规则抬回启动区（6s）
                    self.link.send_stop()
                    if self.time_in_state() > 6.0:
                        self.enter(State.TO_AMMO)

                elif self.state == State.END:
                    self.link.send_stop()
                    break

                # 4) 心跳兜底
                self.link.heartbeat(interval_s=0.2, mode="last")

                # 5) 控制周期
                time.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            pass
        finally:
            self.link.send_stop()
            self.link.close()

# ===== 入口 =====
if __name__ == "__main__":
    RobotMain().run()
