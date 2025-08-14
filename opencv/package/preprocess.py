import cv2
import numpy as np

def grayworld_white_balance(img_bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    灰度世界白平衡（快速、通用）。RGB 三通道都按均值归一，避免偏色。
    eps 防止除零。
    """
    f32 = img_bgr.astype(np.float32)
    b, g, r = f32[:, :, 0], f32[:, :, 1], f32[:, :, 2]
    avg_b = max(float(np.mean(b)), eps)
    avg_g = max(float(np.mean(g)), eps)
    avg_r = max(float(np.mean(r)), eps)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    f32[:, :, 0] *= (avg_gray / avg_b)
    f32[:, :, 1] *= (avg_gray / avg_g)
    f32[:, :, 2] *= (avg_gray / avg_r)
    return np.clip(f32, 0, 255).astype(np.uint8)

def preprocess_frame(
    frame_bgr: np.ndarray,
    *,
    downscale: float = 1.0,
    do_white_balance: bool = True,
    blur_ksize: int = 5
) -> np.ndarray:
    """
    通用预处理：
      1) 可选缩放（仅影响显示和计算速度，不改变语义）
      2) 高斯去噪
      3) 可选灰度世界白平衡
    """
    out = frame_bgr

    if downscale and downscale != 1.0:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w / downscale), int(h / downscale)))

    if blur_ksize and blur_ksize > 1:
        # 取奇数核
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)

    if do_white_balance:
        out = grayworld_white_balance(out)

    return out
