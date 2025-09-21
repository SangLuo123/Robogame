import cv2
import numpy as np
import os
import sys
# pip install labelImg
# labelImg  # 启动工具

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

def select_roi_interactive(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img_copy = img.copy()
    
    # 选择矩形 ROI
    roi_rect = cv2.selectROI("Select ROI", img, False)
    cv2.destroyAllWindows()
    
    # 可视化结果
    x, y, w, h = roi_rect
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Selected ROI", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return roi_rect

img_path = os.path.join(ROOT, "tools/img", "1.png")
roi_rect = select_roi_interactive(img_path)
print(f"ROI矩形坐标: {roi_rect}")  # (x, y, width, height)