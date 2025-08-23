import os
import cv2
import sys
import time
import numpy as np
from datetime import datetime
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ========= 可改参数 =========
CALIB_NPZ = os.path.join(ROOT, "calib", "calib.npz")   # 包含内参和畸变的 npz 文件
CAM_INDEX = 0                   # 摄像头索引
SAVE_DIR  = "captures"          # 保存目录
WINDOW    = "Undistorted (Space=Save, q/Esc=Quit)"
# ==========================

def load_calib(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "mtx" in data and "dist" in data:
        K, dist = data["mtx"], data["dist"]
    elif "camera_matrix" in data and "dist_coeff" in data:
        K, dist = data["camera_matrix"], data["dist_coeff"]
    else:
        raise ValueError(f"{npz_path} 中未找到 'mtx'/'dist' 或 'camera_matrix'/'dist_coeff'")
    return K.astype(np.float32), dist.astype(np.float32)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) 打开摄像头
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: 无法打开摄像头")
        return

    # 2) 读取一帧获取尺寸
    ok, frame0 = cap.read()
    if not ok:
        print("ERROR: 无法读取首帧")
        return
    h, w = frame0.shape[:2]

    # 3) 加载标定并构建去畸变映射
    try:
        K, dist = load_calib(CALIB_NPZ)
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)  # alpha=0 尽量裁掉黑边
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_16SC2)
        use_undistort = True
        x, y, w_roi, h_roi = roi
    except Exception as e:
        print(f"[WARN] 标定加载失败/不完整：{e}\n将直接显示原始画面（不去畸变）。")
        use_undistort = False
        x, y, w_roi, h_roi = 0, 0, w, h

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    t_last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if use_undistort:
            undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            view = undist[y:y+h_roi, x:x+w_roi]
        else:
            view = frame

        # 简单 FPS 叠加
        now = time.time()
        dt = now - t_last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t_last = now
        cv2.putText(view, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # Esc 或 q 退出
            break
        elif key == 32:            # 空格拍照
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(SAVE_DIR, f"cap_{ts}.png")
            # 保存去畸变后的图像（或原图）
            cv2.imwrite(path, view, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
