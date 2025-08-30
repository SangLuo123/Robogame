import cv2
import numpy as np
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)



img_path = os.path.join(ROOT, "tools/img", "1.png")
frames = [cv2.imread(img_path)]  # 你自己的样片列表
hsv_thr = {'H_min':26, 'H_max':97, 'S_min':60, 'V_min':62}
img = cv2.imread(img_path)
roi_rect=(332, 285, 94, 94)

# 矩形ROI
roi = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# ---- 根据 hsv_thr 生成掩码并显示 ----
Hmin, Hmax = hsv_thr['H_min'], hsv_thr['H_max']
Smin, Vmin = hsv_thr['S_min'], hsv_thr['V_min']

# 1) 在 ROI 上做 HSV 阈值
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
mH = (H >= Hmin) & (H <= Hmax)     # 如果是红色跨界，改成: mH = (H>=Hmin) | (H<=Hmax)
mask = ((mH) & (S >= Smin) & (V >= Vmin)).astype(np.uint8) * 255

# （可选）形态学清理一点点小噪声（不需要就注释掉）
# ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)

cv2.imshow("HSV mask (ROI)", mask)

# 2) 把掩码“涂色”叠回到原图上，方便肉眼检查
overlay = img.copy()
x, y, w, h = roi_rect
color = np.zeros_like(roi)
color[mask > 0] = (0, 255, 0)                 # 掩码区域染成绿色
roi_vis = cv2.addWeighted(roi, 0.7, color, 0.3, 0)
overlay[y:y+h, x:x+w] = roi_vis               # 覆盖回原图
cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,255), 2)  # 画出ROI边框

cv2.imshow("Overlay on full image", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
