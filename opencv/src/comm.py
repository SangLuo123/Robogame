# src/comm.py
import serial
import threading
import time
import math
from typing import List, Tuple, Callable, Optional

# ========== 协议常量 ==========
FRAME_HEAD = ord('$')
FRAME_TAIL = ord('#')
ACK_TIMEOUT = 3  # 秒
MAX_RETRIES = 3   # 次

# ========== 协议工具类 ==========
class AsciiProtocol:
    """
    ASCII 帧协议: $<CMD><payload>#
      V: $V<x_mm>,<y_mm>#          (坐标，单位 mm，表示目标相对小车坐标系的位置)
      S: $S#                       (急停)
      Q: $Q#                       (查询编码器)
      E: $E<e1>,<e2>,...#          (下位机回复)
      A: $AOK# / $AERR,<code>#     (下位机回复)
      U: $U#                       (上楼)
      D: $D#                       (下楼)
      T: $T<rpm>#                  (设置发射器转速)
      G: $G<name>#                 (机械臂预设动作)
      R: $R<yaw_deg>#              (旋转角度)
      H: $H#                       (心跳)
    """

    # ---- 组包 ----
    @staticmethod
    def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
        """CRC-16-CCITT 算法"""
        crc = init
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                if (crc & 0x8000) != 0:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
                crc &= 0xFFFF
        
        # 当生成的CRC包含\x00时，进行偏移：连续加1直到两个字节都不为0
        while ((crc >> 8) & 0xFF) == 0 or (crc & 0xFF) == 0:
            crc = (crc + 1) & 0xFFFF
        
        return crc
    
    @staticmethod
    def build(cmd: str, fields: List[str] | None = None) -> bytes:
        assert len(cmd) == 1, "CMD 必须是单字符"
        payload = cmd if not fields else (cmd + ",".join(fields))
        core = b"$" + payload.encode("ascii")
        crc = AsciiProtocol.crc16_ccitt(core)
        msg = core + crc.to_bytes(2, 'big') + b"#"
        return msg

    @staticmethod
    def build_vel_xy(x: float, y: float) -> bytes:
        return AsciiProtocol.build("V", [f"{x:.0f}", f"{y:.0f}"])

    @staticmethod
    def build_stop() -> bytes:
        return AsciiProtocol.build("S")

    @staticmethod
    def build_query_enc() -> bytes:
        return AsciiProtocol.build("Q")

    @staticmethod
    def build_shooter_rpm(rpm: int) -> bytes:
        return AsciiProtocol.build("T", [str(int(rpm))])

    @staticmethod
    def build_arm_preset(name: str) -> bytes:
        return AsciiProtocol.build("G", [name])
    
    @staticmethod
    def build_rotate(yaw_deg: float) -> bytes:
        # 默认逆时针？
        return AsciiProtocol.build("R", [f"{yaw_deg:.0f}"])

    @staticmethod
    def build_upstairs() -> bytes:
        return AsciiProtocol.build("U")

    @staticmethod
    def build_downstairs() -> bytes:
        return AsciiProtocol.build("D")

    @staticmethod
    def build_heartbeat() -> bytes:
        return AsciiProtocol.build("H")

    # ---- 流式解析（状态机）----
    def __init__(self):
        self._in_frame = False
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[Tuple[str, List[str]]]:
        """把读到的字节流喂进来，吐出解析好的帧列表 [(cmd, [fields...]), ...]"""
        out: List[Tuple[str, List[str]]] = []
        for ch in data:
            if ch == FRAME_HEAD:
                self._in_frame = True
                self._buf.clear()
                continue
            if not self._in_frame:
                continue
            if ch == FRAME_TAIL:
                try:
                    s = self._buf.decode("ascii", errors="strict")
                    if s:
                        cmd = s[0]
                        fields = s[1:].split(",") if len(s) > 1 else []
                        fields = [f for f in fields if f != ""]
                        out.append((cmd, fields))
                except UnicodeDecodeError:
                    pass
                self._in_frame = False
                self._buf.clear()
            else:
                self._buf.append(ch)
        return out


# ========== 串口封装 ==========
class SerialLink:
    """
    OrangePi/PC <-> STM32 串口（ASCII 协议）
    - 发送：V/S/Q/P/T/C/R
    - 接收：A(ACK), E(Encoders) 等，分发到回调
    - 心跳：仅提供主循环兜底 heartbeat()，**不含独立心跳线程**
    """

    # 回调类型
    FrameCB = Optional[Callable[[str, List[str]], None]]
    AckCB   = Optional[Callable[[bool, List[str]], None]]   # (ok, fields)
    EncCB   = Optional[Callable[[bool, List[str]], None]]         # [e1,e2,...]
    TextCB  = Optional[Callable[[str], None]]               # 调试文本
    FaultCB = Optional[Callable[[List[str]], None]]         # 故障字段

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200, timeout: float = 0.02):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self.ser: Optional[serial.Serial] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.running = False
        self._tx_lock = threading.Lock()

        self.proto = AsciiProtocol()
        self.last_tx_time: float = 0.0              # 最近一次发送任意帧的时间
        self._last_cmd_xy: Tuple[float, float] = (0.0, 0.0)

        self._ack_event = threading.Event()
        self._ack_ok = False


        # 回调（按需绑定）
        self.on_frame: SerialLink.FrameCB = None
        self.on_ack:   SerialLink.AckCB   = None
        self.on_enc:   SerialLink.EncCB   = None
        self.on_text:  SerialLink.TextCB  = None
        self.on_fault: SerialLink.FaultCB = None

    # ---------- 基础 ----------
    def open(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self.running = True
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

    def close(self):
        self.running = False
        if self.rx_thread:
            self.rx_thread.join(timeout=0.3)
            self.rx_thread = None
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ---------- 发送 ----------
    def _send_bytes(self, msg: bytes, timeout: float = ACK_TIMEOUT, retries: int = MAX_RETRIES) -> bool:
        # TODO: 还是发送失败怎么办
        """
        发送 msg（已经包含 CRC 和 '#'），等待 ACK（b'\x06' 或 b'ACK'），超时重发。
        timeout 单位秒，retries 重试次数（含第一次发送）。
        返回 True 表示收到 ACK。
        """
        print(f"[DEBUG] 发送: {msg}")
        if not self.is_open() or self.ser is None:
            return False
        with self._tx_lock:
            for attempt in range(retries):
                try:
                    # 发送数据
                    self.ser.write(msg)
                    self.ser.flush()
                    self.last_tx_time = time.time()

                    # 等待 ACK
                    self._ack_event.clear()
                    start_time = time.time()
                    while True:
                        if self._ack_event.wait(timeout=timeout):
                            print(f"[DEBUG] 收到 ACK: {self._ack_ok}")
                            return self._ack_ok
                        if time.time() - start_time >= timeout:
                            print(f"Timeout waiting for ACK (attempt {attempt+1}/{retries})")
                            break

                except Exception as e:
                    print(f"Exception during send (attempt {attempt+1}/{retries}): {e}")

                if attempt < retries - 1:
                    time.sleep(0.05)
        return False
    def send_vel_xy(self, x: float, y: float):
        # 单位：mm，正前方为x，正左方为y
        """ 位移命令：$Vx,y# """
        self._last_cmd_xy = (float(x), float(y))
        self._send_bytes(self.proto.build_vel_xy(x, y))

    def send_stop(self):
        """急停：$S#"""
        self._send_bytes(self.proto.build_stop())

    def query_encoders(self):
        """查询编码器：$Q#"""
        self._send_bytes(self.proto.build_query_enc())

    def send_shooter_rpm(self, rpm: int):
        """设置发射器转速：$Trpm#"""
        self._send_bytes(self.proto.build_shooter_rpm(rpm))

    def send_upstairs(self):
        """上楼：$U#"""
        self._send_bytes(self.proto.build_upstairs())
    
    def send_downstairs(self):
        """下楼：$D#"""
        self._send_bytes(self.proto.build_downstairs())

    def arm_preset(self, name: str):
        """机械臂预设动作：$Gname#（如 GRAB/REL）"""
        self._send_bytes(self.proto.build_arm_preset(name))
        
    def rotate(self, yaw_deg: float):
        """旋转：$Ryaw_deg#"""
        self._send_bytes(self.proto.build_rotate(yaw_deg))
        
    def send_heartbeat(self):
        """心跳：$H#"""
        msg = self.proto.build_heartbeat()
        # print(f"[DEBUG] 发送心跳: {msg}")
        
        # 如果串口未打开，直接返回
        if not self.is_open() or self.ser is None:
            return False
        
        try:
            with self._tx_lock:
                self.ser.write(msg)
                self.ser.flush()
                self.last_tx_time = time.time()  # 如果需要记录最后发送时间
            return True
        except Exception as e:
            print(f"Exception during heartbeat send: {e}")
            return False

    # ---------- 接收线程 ----------
    def _rx_loop(self):
        if self.ser is None:
            return
        while self.running:
            try:
                if not self.is_open():
                    time.sleep(0.1)
                    continue
                # 读取可用数据
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    frames = self.proto.feed(data)
                    
                    for cmd, fields in frames:
                        # 通用帧回调
                        if self.on_frame:
                            self.on_frame(cmd, fields)

                        # 分类分发
                        if cmd == "A":  # ACK
                            ok = True
                            if len(fields) >= 1 and fields[0].upper().startswith("ERR"):
                                ok = False
                            self._ack_ok = ok
                            self._ack_event.set()   # 唤醒等待方
                            if self.on_ack:
                                self.on_ack(ok, fields)

                        elif cmd == "E":  # Encoders
                            ok = True
                            if len(fields) >= 1 and fields[0].upper().startswith("ERR"):
                                ok = False
                            if self.on_enc:
                                self.on_enc(ok, fields)

                        elif cmd == "T":  # Text（保留）
                            if self.on_text:
                                self.on_text(",".join(fields))

                        elif cmd == "F":  # Fault（保留）
                            if self.on_fault:
                                self.on_fault(fields)

                        else:
                            # 其他类型需要时再扩展
                            pass
                else:
                    time.sleep(0.01)  # 没有数据时短暂休眠

            except Exception as e:
                print(f"Exception in RX loop: {e}")
                time.sleep(0.05)

    # ---------- 心跳（主循环兜底） ----------
    def heartbeat(self, interval_s: float = 0.2):
        """
        在你的主控制循环里周期调用：
          - 若 interval_s 内未发送任何帧，则自动发送一条保活心跳 $H#
        """
        now = time.time()
        if now - self.last_tx_time > interval_s:
            self.send_heartbeat()
