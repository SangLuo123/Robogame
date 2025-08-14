# undistorted_capture.py
# 用法示例：
#   from undistorted_capture import UndistortedCapture
#   cap = UndistortedCapture(0, calib_path="calib.npz", alpha=0.0)
#   while True:
#       ok, raw, undist = cap.read()
#       ...
import cv2
import numpy as np
import os
from typing import Tuple, Optional

class UndistortedCapture:
    """
    类似 cv2.VideoCapture，但 read() 会返回 (ok, 原图, 去畸变图)。
    其他接口基本与原生 VideoCapture 一致。
    """
    def __init__(self,
                 src=0,
                 calib_path: Optional[str] = "calib.npz",
                 alpha: float = 0.0,
                 autopen: bool = True):
        self._src = src
        self._cap = None
        self._alpha = float(alpha)

        self._have_calib = False
        self._K = None
        self._dist = None

        self._size: Optional[Tuple[int, int]] = None  # (w,h)
        self._newK = None
        self._map1 = None
        self._map2 = None

        self._load_calib(calib_path)
        if autopen:
            self.open(src)

    # ---------- 公共接口 ----------
    def open(self, src=None) -> bool:
        if src is not None:
            self._src = src
        self._cap = cv2.VideoCapture(self._src)
        # self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25=手动(具体取值与驱动有关) # 开/关自动曝光模式。
        # self._cap.set(cv2.CAP_PROP_EXPOSURE,    100)     # 具体数值按画面调 # 设置曝光时间（单位不是统一标准，取决于驱动）。
        # self._cap.set(cv2.CAP_PROP_AUTO_WB,     0)       # 关闭自动白平衡 # 开/关自动白平衡（WB = White Balance）。
        # self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500) # 需要驱动支持 # 设置色温（单位 Kelvin，常见范围 2800–6500K）。
        self._size = None
        self._map1 = self._map2 = None
        return self.isOpened()

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        """
        返回 (ok, raw_frame, undistorted_frame)。
        如果没有标定参数，undistorted_frame = raw_frame。
        """
        if not self.isOpened():
            return False, None, None

        ok, raw_frame = self._cap.read()
        if not ok or raw_frame is None:
            return ok, raw_frame, raw_frame

        h, w = raw_frame.shape[:2]
        size_now = (w, h)

        # 分辨率变化则重建映射表
        if self._size != size_now:
            self._size = size_now
            self._rebuild_maps()

        if self._map1 is not None and self._map2 is not None:
            und = cv2.remap(raw_frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
            return True, raw_frame, und
        else:
            return True, raw_frame, raw_frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._map1 = self._map2 = None
        self._size = None

    def get(self, propId):
        if not self.isOpened():
            return None
        return self._cap.get(propId)

    def set(self, propId, value):
        if not self.isOpened():
            return False
        ok = self._cap.set(propId, value)
        self._size = None
        self._map1 = self._map2 = None
        return ok

    # ---------- 内部方法 ----------
    def _load_calib(self, calib_path: Optional[str]):
        self._have_calib = False
        self._K = None
        self._dist = None
        if calib_path and os.path.exists(calib_path):
            try:
                data = np.load(calib_path, allow_pickle=True)
                if "mtx" in data and "dist" in data:
                    self._K = data["mtx"].astype(np.float32)
                    self._dist = data["dist"].astype(np.float32).reshape(-1)
                    self._have_calib = True
                    print(f"[UndistortedCapture] 已加载标定文件 '{calib_path}'")
                else:
                    print(f"[UndistortedCapture] '{calib_path}' 缺少 'mtx' 或 'dist'")
            except Exception as e:
                print(f"[UndistortedCapture] 读取 '{calib_path}' 失败：{e}")
        else:
            print(f"[UndistortedCapture] 未找到标定文件 '{calib_path}'，将直接返回原图")

    def _rebuild_maps(self):
        self._map1 = self._map2 = None
        if not self._have_calib or not self.isOpened() or self._size is None:
            return
        w, h = self._size
        try:
            newK, _ = cv2.getOptimalNewCameraMatrix(self._K, self._dist, (w, h), self._alpha)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(self._K, self._dist, None, newK, (w, h), cv2.CV_16SC2)
        except Exception as e:
            print(f"[UndistortedCapture] 构建映射表失败：{e}")
            self._map1 = self._map2 = None

# ---------- 测试演示 ----------
if __name__ == "__main__":
    cap = UndistortedCapture(0, calib_path="./MtxAndDist/calib.npz", alpha=0.0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit(1)

    print("按 ESC 退出")
    while True:
        ok, raw, und = cap.read()
        if not ok:
            break
        # 并排显示
        h, w = raw.shape[:2]
        disp_w = 640
        disp_h = int(disp_w * h / w)
        left = cv2.resize(raw, (disp_w, disp_h))
        right = cv2.resize(und, (disp_w, disp_h))
        combined = np.hstack((left, right))
        cv2.imshow("Raw | Undistorted", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
