# 测试detector.py
# 假设你有 calib.npz 存相机内参和畸变
calib = np.load("mtx_dist/calib.npz")
camera_matrix = calib["mtx"]
dist_coeffs = calib["dist"]

# 飞镖颜色 HSV 范围（示例）
hsv_params = {'lower': (0, 100, 100), 'upper': (10, 255, 255)}

# 初始化 Detector
detector = Detector(camera_matrix, dist_coeffs, tag_size=0.16, hsv_params=hsv_params)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 检测 AprilTag
    tags = detector.detect_tags(frame)

    # 2. 检测飞镖
    dart = detector.detect_dart(frame)

    # 3. 画结果
    vis = detector.draw_results(frame.copy(), tags=tags, dart=dart)
    cv2.imshow("Detection", vis)

    if cv2.waitKey(1) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()
