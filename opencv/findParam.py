import cv2
import numpy as np

hsv_ranges = [([15, 44, 59], [23, 255, 255])]

def detect_multicolor_object(hsv_ranges, camera_index=0):
    """
    使用多个 HSV 范围提取物体。

    参数：
    - hsv_ranges: 列表，每个元素为 ([H_min, S_min, V_min], [H_max, S_max, V_max])
    - camera_index: 摄像头编号
    """

    # ! 是否需要预处理
    def preprocess_frame(frame):
        # 1. 高斯模糊降噪
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. 白平衡（简单灰度世界法）
        result = frame.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3
        result[:, :, 0] *= avg_gray / avg_b
        result[:, :, 1] *= avg_gray / avg_g
        result[:, :, 2] *= avg_gray / avg_r
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 初始化合并掩码
        full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            full_mask = cv2.bitwise_or(full_mask, mask)

        result = cv2.bitwise_and(frame, frame, mask=full_mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Combined Mask", full_mask)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_multicolor_object(hsv_ranges, camera_index=0)  # 使用默认摄像头
