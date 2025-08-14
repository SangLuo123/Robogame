import cv2
import numpy as np

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

def preprocess_for_calib(frame):
    """A 方法：灰度 + 自适应对比度CLAHE + 轻微高斯"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25=手动(具体取值与驱动有关) # 开/关自动曝光模式。
# cap.set(cv2.CAP_PROP_EXPOSURE,    100)     # 具体数值按画面调 # 设置曝光时间（单位不是统一标准，取决于驱动）。
# cap.set(cv2.CAP_PROP_AUTO_WB,     0)       # 关闭自动白平衡 # 开/关自动白平衡（WB = White Balance）。
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500) # 需要驱动支持 # 设置色温（单位 Kelvin，常见范围 2800–6500K）。


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame2 = preprocess_frame(frame)
    gray2 = preprocess_for_calib(frame)
    gray2_bgr = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    frame_resized = cv2.resize(frame, (320, 240))
    frame2_resized = cv2.resize(frame2, (320, 240))
    gray2_resized = cv2.resize(gray2_bgr, (320, 240))

    row1 = np.hstack([frame_resized, frame2_resized])
    row2 = np.hstack([gray2_resized, gray2_resized])  # 示例
    combined = np.vstack([row1, row2])
    cv2.imshow("Combined View", combined)


    # cv2.imshow("Original", frame)
    # cv2.imshow("Processed", frame2)
    # cv2.imshow("Gray for Calibration", gray2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()