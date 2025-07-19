import cv2
import numpy as np

# 你的内参矩阵
mtx = np.array([[583.34686119, 0., 304.37140818],
                [0., 584.6438033, 239.35915117],
                [0., 0., 1.]])

# 你的畸变参数
dist = np.array([[-0.45572681, 0.08221623, -0.00161879, 0.00369152, 0.63039585]])

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 去畸变
    undistorted = cv2.undistort(frame, mtx, dist)

    # 显示原图和去畸变图
    cv2.imshow('original', frame)
    cv2.imshow('undistored', undistorted)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按 ESC 键退出
        break

cap.release()
cv2.destroyAllWindows()
