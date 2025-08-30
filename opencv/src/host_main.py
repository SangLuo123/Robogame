# test_comm.py
import sys, time, math
from comm import SerialLink

# --------------- 参数（也可用命令行覆盖） ---------------
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 115200
MODE = sys.argv[2] if len(sys.argv) > 2 else "circle"   # circle / waypoints

def attach_callbacks(link: SerialLink):
    link.on_ack   = lambda ok, f: print("ACK:", "OK" if ok else f)
    link.on_enc   = lambda enc:  print("ENC:", enc)
    link.on_text  = lambda s:    print("TXT:", s)
    link.on_fault = lambda fs:   print("FAULT:", fs)

# ============= 模式 A：圆形目标点（相对车体） =============
def run_circle(link: SerialLink, A_mm=400.0, freq_hz=0.2):
    """
    在车体坐标系下，让目标点匀速绕圆：x=A*cos(ωt), y=A*sin(ωt)
    A_mm：半径（毫米），freq_hz：环绕频率
    """
    t0 = time.time()
    next_q = 0.0
    try:
        while True:
            t = time.time() - t0
            omega = 2 * math.pi * freq_hz
            x = A_mm * math.cos(omega * t)
            y = A_mm * math.sin(omega * t)

            # 下发“目标点（相对车体，mm）”
            link.send_vel_xy(x, y)

            # 1Hz 查询编码器
            if t >= next_q:
                link.query_encoders()
                next_q = t + 1.0

            # 兜底心跳（重发上一条 $Vx,y#）
            link.heartbeat(0.2, mode="last")

            time.sleep(0.05)   # 20Hz 主循环
    except KeyboardInterrupt:
        print("\n[TEST] circle: Ctrl-C")
    finally:
        link.send_stop()

# ============= 模式 B：路标点序列（相对车体） =============
def run_waypoints(link: SerialLink, repeats=3):
    """
    依次发送相对坐标路标点，单位 mm。
    每个路标保持 hold_s 秒。没有“到达反馈”时用时间近似。
    """
    waypoints = [
        ( 500,   0),  # 前 500mm
        (   0, 400),  # 右 400mm
        (-500,   0),  # 后 500mm
        (   0,-400),  # 左 400mm
    ]
    hold_s = 2.0
    next_q = 0.0
    try:
        for k in range(repeats):
            print(f"[TEST] Round {k+1}/{repeats}")
            for (x, y) in waypoints:
                print(f"  -> target (mm): ({x}, {y})")
                t_enter = time.time()
                while time.time() - t_enter < hold_s:
                    link.send_vel_xy(x, y)

                    # 每秒查一次编码器
                    t = time.time()
                    if t >= next_q:
                        link.query_encoders()
                        next_q = t + 1.0

                    link.heartbeat(0.2, mode="last")
                    time.sleep(0.05)

        # 可选：演示旋转（相对朝向，角度制）
        print("[TEST] rotate demo: 45 deg")
        link.rotate(45.0)
        time.sleep(1.0)
        print("[TEST] rotate demo: -45 deg")
        link.rotate(-45.0)
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[TEST] waypoints: Ctrl-C")
    finally:
        link.send_stop()

# ========================== 主程序 ==========================
if __name__ == "__main__":
    link = SerialLink(PORT, BAUD)
    link.open()
    attach_callbacks(link)

    try:
        if MODE == "waypoints":
            run_waypoints(link, repeats=3)
        else:
            run_circle(link, A_mm=400.0, freq_hz=0.2)
    finally:
        link.close()
