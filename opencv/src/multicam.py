# multicam.py
import cv2
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass # 能够自动生成一些基础方法
class FramePacket:
    frame_id: int
    ts_ns: int            # monotonic time (ns)
    image: "cv2.Mat"
    fps: float            # 采集线程估算的FPS

def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)

class _CamWorker:
    """
    单路摄像头采集线程：只保留最新一帧（避免爆队列）。
    """
    def __init__(self, name: str, device, width=None, height=None,
                 fourcc: Optional[str]=None, fps: Optional[int]=None,
                 undistort_maps: Optional[Tuple]=None,
                 backend=cv2.CAP_V4L2, reopen_interval=1.0):
        self.name = name
        self.device = device
        self.width, self.height = width, height
        self.fourcc = fourcc
        self.fps_req = fps
        self.undistort_maps = undistort_maps
        self.backend = backend
        self.reopen_interval = reopen_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._frame_id = 0
        self._ema_fps = 0.0
        
        self.newK = None
        if undistort_maps is not None:
            if len(undistort_maps) == 3:
                self.undistort_maps = (undistort_maps[0], undistort_maps[1])
                self.newK = undistort_maps[2]
            else:
                self.undistort_maps = undistort_maps
    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ------------- public: 获取最新帧（非阻塞） -------------
    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    # ------------- internal -------------
    def _open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device, self.backend)
        if not self._cap or not self._cap.isOpened():
            return False

        # 可选设置：分辨率 / FOURCC / FPS / 缓存
        if self.width:  self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        if self.height: self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fourcc: self._cap.set(cv2.CAP_PROP_FOURCC, _fourcc(self.fourcc))
        if self.fps_req: self._cap.set(cv2.CAP_PROP_FPS, self.fps_req)

        # 尝试缩小相机内部缓冲
        try: self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        # 预热丢几帧
        for _ in range(3):
            self._cap.read()
        return True

    def _loop(self):
        last_open_try = 0.0
        last_ts = None
        while not self._stop_evt.is_set():
            # 保证已打开
            if not self._cap or not self._cap.isOpened():
                now = time.time()
                if now - last_open_try < self.reopen_interval:
                    time.sleep(0.05); continue
                last_open_try = now
                if not self._open():
                    time.sleep(self.reopen_interval)
                    continue

            ok, img = self._cap.read()
            t_ns = time.monotonic_ns()
            if not ok or img is None:
                # 掉线：释放稍后重连
                self._cap.release()
                self._cap = None
                time.sleep(self.reopen_interval)
                continue

            # 去畸变（若提供 mapx,mapy）
            if self.undistort_maps is not None:
                mapx, mapy = self.undistort_maps
                img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # FPS（EMA）
            if last_ts is not None:
                dt = (t_ns - last_ts) / 1e9
                if dt > 0:
                    inst = 1.0 / dt
                    if self._ema_fps == 0:
                        self._ema_fps = inst
                    else:
                        self._ema_fps = 0.9 * self._ema_fps + 0.1 * inst
            last_ts = t_ns

            self._frame_id += 1
            pkt = FramePacket(self._frame_id, t_ns, img, self._ema_fps)

            # 仅保留最新帧（覆盖）
            with self._lock:
                self._latest = pkt

    # ------------- util: 载入去畸变映射（静态方法） -------------
    @staticmethod
    def build_undistort_maps(K, dist, size: Tuple[int,int]):
        """
        K: 3x3, dist: (k1,k2,p1,p2[,k3 ...]), size: (w,h)
        """
        w, h = size
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
        return mapx, mapy, newK


class MultiCam:
    """
    多相机管理器：管理多路 _CamWorker，提供同步获取接口。
    """
    def __init__(self):
        self.workers: Dict[str, _CamWorker] = {}

    def add_camera(self, name: str, device, width=None, height=None,
                   fourcc: Optional[str]=None, fps: Optional[int]=None,
                   undistort_maps: Optional[Tuple]=None,
                   backend=cv2.CAP_V4L2):
        if name in self.workers:
            raise ValueError(f"camera '{name}' already exists")
        self.workers[name] = _CamWorker(
            name, device, width, height, fourcc, fps, undistort_maps, backend
        )
        return self

    def start(self):
        for w in self.workers.values():
            w.start()
        return self

    def stop(self):
        for w in self.workers.values():
            w.stop()

    def latest(self, name: str) -> Optional[FramePacket]:
        w = self.workers.get(name)
        return None if w is None else w.latest()

    def get_pair_synced(self, a: str, b: str,
                        max_skew_ms: float = 30.0,
                        timeout_ms: float = 300.0) -> Optional[Tuple[FramePacket, FramePacket]]:
        """
        在 timeout 内尝试拿到时间戳相差 <= max_skew_ms 的成对帧。
        """
        t_end = time.monotonic_ns() + int(timeout_ms * 1e6)
        while time.monotonic_ns() < t_end:
            pa = self.latest(a)
            pb = self.latest(b)
            if pa is None or pb is None:
                time.sleep(0.005); continue
            skew_ms = abs(pa.ts_ns - pb.ts_ns) / 1e6
            if skew_ms <= max_skew_ms:
                return pa, pb
            # 丢掉更旧的那一侧，等待下一帧
            if pa.ts_ns < pb.ts_ns:
                time.sleep(0.005)
            else:
                time.sleep(0.005)
        return None
