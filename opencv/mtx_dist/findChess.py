import cv2
import numpy as np
import math

def onlineCalculate(pattern_size, square_size, camera_intrinsic, cam_id=0):
    """
    pattern_size: (cols, rows) 例如 (9, 6)
    square_size : 棋盘格每小格的实际边长，单位自定（cm 或 mm）
    camera_intrinsic: {"mtx": K, "dist": distCoeffs}
    cam_id: 摄像头编号
    """
    cols, rows = pattern_size
    # 1) 构造棋盘在世界坐标系下的 3D 角点（Z=0 的平面）
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)   # 按列优先、行优先都行，但必须与 findChessboardCorners 返回顺序一致
    objp *= float(square_size)

    K   = camera_intrinsic["mtx"]
    dist= camera_intrinsic["dist"]

    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        print("无法打开摄像头")
        return

    # 角点亚像素精修参数
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) 检测棋盘角点
        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # 3) 亚像素精修：提升 PnP 精度
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

            # 4) 求解位姿（很多点 → 选 ITERATIVE 更稳）
            success, rvec, tvec = cv2.solvePnP(
                objp, corners, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                cv2.putText(frame, "solvePnP failed", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow("pose", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # 5) 距离（与 square_size 同单位）
            dist_val = float(np.linalg.norm(tvec))

            # 6) 欧拉角（使用 RQDecomp3x3 更直观；单位：度）
            R, _ = cv2.Rodrigues(rvec)             # 旋转矩阵
            result = cv2.RQDecomp3x3(R)
            euler = result[-1]                       # 兼容不同 OpenCV 版本
            euler = np.array(euler).reshape(-1)   # 拍扁成一维
            if euler.size < 3:
                raise ValueError(f"Unexpected euler shape: {euler.shape}, value={euler}")
            pitch, yaw, roll = map(float, euler[:3])   # rx, ry, rz (deg)
            # 7) 画角点与坐标轴
            cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
            axis_len = 3 * float(square_size)      # 坐标轴长度（3 格）
            axis_3d = np.float32([
                [0, 0, 0],
                [axis_len, 0, 0],      # X - 红
                [0, axis_len, 0],      # Y - 绿
                [0, 0, -axis_len],     # Z - 蓝（指向相机为 -Z）
            ])
            axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
            p0 = tuple(axis_2d[0].ravel().astype(int))
            px = tuple(axis_2d[1].ravel().astype(int))
            py = tuple(axis_2d[2].ravel().astype(int))
            pz = tuple(axis_2d[3].ravel().astype(int))
            cv2.line(frame, p0, px, (0, 0, 255), 3)
            cv2.line(frame, p0, py, (0, 255, 0), 3)
            cv2.line(frame, p0, pz, (255, 0, 0), 3)

            # 8) 信息叠加
            txt1 = f"dist: {dist_val:.2f}  (same unit as square_size)"
            txt2 = f"yaw: {yaw:.2f}  pitch: {pitch:.2f}  roll: {roll:.2f}  (deg)"
            cv2.putText(frame, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unable to Detect Chessboard", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

    cam.release()
    cv2.destroyAllWindows()

def load_calib(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "mtx" not in data or "dist" not in data:
        raise KeyError("npz 中必须包含 'mtx' 和 'dist'")
    mtx = data["mtx"].astype(np.float32)
    dist = data["dist"].astype(np.float32).reshape(-1)  # 展平为(5,)等
    return mtx, dist

if __name__ == "__main__":
    mtx, dist = load_calib("calib.npz")
    camera_intrinsic = {
        "mtx": mtx,
        "dist": dist
    }

    onlineCalculate(pattern_size=(9,6), square_size=2.43, camera_intrinsic=camera_intrinsic)
