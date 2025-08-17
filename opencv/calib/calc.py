import numpy as np
import cv2
import glob
import math

height = 6 
width = 9
length = 24.3

# ! 修改矩阵参数（长、宽），修改格子实际宽度，修改文件路径
def calibrate(folder="/home/orangepi/Robogame/opencv/calib/img/*.png", save="calib.npz"):
    images = glob.glob(folder)
    if not images:
        raise FileNotFoundError("没有找到标定图，请检查路径或后缀")

    # 备好棋盘在世界坐标系的3D点
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp *= length

    obj_points, img_points = [], []
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    used = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])  # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, (width, height))
        if not ret:
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (5,5), (-1,-1), criteria
        )

        obj_points.append(objp)
        img_points.append(corners)
        used += 1

    if used < 8:
        raise RuntimeError(f"有效标定图太少：{used} 张。至少拍 10–15 张、角度/距离要多样。")

    # 标定
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    # 计算平均重投影误差，快速判断好坏（<0.5 像素通常不错）
    total_err = 0
    total_pts = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)
        total_err += err**2
        total_pts += len(img_points[i])
    mean_err = np.sqrt(total_err / total_pts)

    np.savez(save, mtx=mtx, dist=dist, img_size=img_size, rms=rms, mean_err=mean_err)
    print(f"[OK] 标定完成：RMS={rms:.4f}, mean reproj err={mean_err:.4f}, 用图={used} 张")
    print("K=\n", mtx, "\ndist=\n", dist)
    return mtx, dist, img_size

mtx, dist, img_size = calibrate()
print(Camera_Intrinsic)


# print(Camera_Intrinsic)

# # 读取相机内参和畸变参数
# mtx = Camera_Intrinsic["mtx"]
# dist = Camera_Intrinsic["dist"]

# # 读取测试图像
# img = cv2.imread("../img/10.png")

# # 去畸变
# undistorted = cv2.undistort(img, mtx, dist)

# # 显示或保存去畸变后的图像
# cv2.imshow("undistorted", undistorted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


