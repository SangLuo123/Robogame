import numpy as np
import cv2
import glob
import math

height = 6 
width = 9
length = 2.43

# ! 修改矩阵参数（长、宽），修改格子实际宽度，修改文件路径
def calcMtxAndDist():
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = length * objp  # 打印棋盘格一格的边长为2.6cm
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    images = glob.glob("/home/orangepi/Robogame/opencv/img/*.png")  # 黑白棋盘的图片路径

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (width, height), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.waitKey(1)
    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # 内参数矩阵
    camera_intrinsic = {"mtx": mtx, "dist": dist, }
    return objp, camera_intrinsic


def onlineCalculate(objp, camera_intrinsic):
    obj_points = objp  # 存储3D点
    img_points = []  # 存储2D点

    # 从摄像头获取视频图像
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:  # 画面中有棋盘格
            img_points = np.array(corners)
            cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
            # rvec: 旋转向量 tvec: 平移向量
            _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_intrinsic["mtx"], camera_intrinsic["dist"])  # 解算位姿
            distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)  # 计算距离
            rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵
            proj_matrix = np.hstack((rvec_matrix, tvec))  # hstack: 水平合并
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
            pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
            cv2.putText(frame, "dist: %.2fcm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (distance, yaw, pitch, roll),
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
                break
        else:  # 画面中没有棋盘格
            cv2.putText(frame, "Unable to Detect Chessboard", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 0, 255), 3)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
                break
    cv2.destroyAllWindows()



objp, Camera_Intrinsic = calcMtxAndDist()
onlineCalculate(objp, Camera_Intrinsic)
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


