import cv2
import numpy as np

NPZ_PATH = "calib.npz"   # 改成你的npz路径

def load_calib(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "mtx" not in data or "dist" not in data:
        raise KeyError("npz 中必须包含 'mtx' 和 'dist'")
    mtx = data["mtx"].astype(np.float32)
    dist = data["dist"].astype(np.float32).reshape(-1)  # 展平为(5,)等
    return mtx, dist

def main():
    K, dist = load_calib(NPZ_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 先读一帧以确定当前分辨率
    ok, frame = cap.read()
    if not ok:
        print("无法读取摄像头画面")
        return
    h, w = frame.shape[:2]

    # 根据当前分辨率计算新的投影矩阵，并预计算映射表（速度快）
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)  # alpha=0 保守裁边
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

    print("按 ESC 退出，按 S 保存一张对比图到 undist_pair.png")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败")
            break

        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # 仅用于展示：缩放拼接对比图
        disp_w = 480
        disp_h = int(disp_w * h / w)
        left  = cv2.resize(frame, (disp_w, disp_h))
        right = cv2.resize(undistorted, (disp_w, disp_h))
        combined = np.hstack((left, right))
        cv2.imshow("Original | Undistorted", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key in (ord('s'), ord('S')):
            cv2.imwrite("undist_pair.png", combined)
            print("已保存 undist_pair.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
