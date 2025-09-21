import cv2
import numpy as np
import os
import sys
import json
"""
该代码根据传入的ROI自动分析飞镖的HSV范围并计算ROI中目标飞镖的面积占比决定是否有飞镖
"""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ---------- 工具：形态学清理 ----------
def _refine_mask(mask, k_open=3, it_open=1, k_close=5, it_close=1):
    if k_open > 0 and it_open > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=it_open)
    if k_close > 0 and it_close > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=it_close)
    return mask

# ---------- 工具：高光掩膜 ----------
def _highlight_mask(bgr, v_high=220, s_low=40, rgb_high=240):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m1 = (V >= v_high) & (S <= s_low)
    B, G, R = cv2.split(bgr)
    m2 = (R >= rgb_high) & (G >= rgb_high) & (B >= rgb_high)
    return (m1 | m2).astype(np.uint8) * 255  # 255=高光

# ---------- A) 标定：从样片里估计 HSV 阈值 ----------
def calibrate_hsv_thresholds(frames, roi_rect=None, roi_poly=None,
                             keep_percent=95,   # 保留主体 95% 像素
                             h_pad=3, s_pad=10, v_pad=10,
                             s_min_floor=60, v_min_floor=60,
                             v_high=220, s_low=40, rgb_high=240):
    """
    frames: list of BGR images (同一孔的若干样片)
    roi_rect: (x,y,w,h) 或 None
    roi_poly: 多边形顶点列表 [(x,y),...] 或 None
    返回: dict {H_min,H_max,S_min,V_min} 以及一些统计信息
    """
    assert roi_rect or roi_poly, "需要传 ROI（矩形或多边形）"
    H_all, S_all, V_all = [], [], []

    for img in frames:
        if roi_rect:
            x,y,w,h = roi_rect
            crop = img[y:y+h, x:x+w]
            mask_roi = np.ones(crop.shape[:2], np.uint8) * 255
        else:
            mask_roi = np.zeros(img.shape[:2], np.uint8)
            pts = np.array(roi_poly, dtype=np.int32).reshape((-1,1,2))
            cv2.fillPoly(mask_roi, [pts], 255)
            crop = cv2.bitwise_and(img, img, mask=mask_roi)

        # 高光掩膜并排除
        hl = _highlight_mask(crop, v_high=v_high, s_low=s_low, rgb_high=rgb_high)
        valid_mask = cv2.bitwise_not(hl)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        H_all.append(H[valid_mask>0].astype(np.float32))
        S_all.append(S[valid_mask>0].astype(np.float32))
        V_all.append(V[valid_mask>0].astype(np.float32))

    Hs = np.concatenate(H_all) if len(H_all) else np.array([], np.float32)
    Ss = np.concatenate(S_all) if len(S_all) else np.array([], np.float32)
    Vs = np.concatenate(V_all) if len(V_all) else np.array([], np.float32)

    if len(Hs) < 100:
        raise RuntimeError("有效样本像素太少；检查 ROI 是否正确、光斑是否过多。")

    # 百分位（中间95%）
    low = (100 - keep_percent)/2.0
    high = 100 - low

    # H 对于黄色通常不跨0/179，所以直接取分位即可；若你以后做红色，需要做环形分布处理
    H_min = float(np.percentile(Hs, low))  - h_pad
    H_max = float(np.percentile(Hs, high)) + h_pad
    S_min = float(np.percentile(Ss, low))  - s_pad
    V_min = float(np.percentile(Vs, low))  - v_pad

    # 下限地板，避免过松
    S_min = max(S_min, s_min_floor)
    V_min = max(V_min, v_min_floor)
    H_min = max(H_min, 0)
    H_max = min(H_max, 179)

    thr = dict(H_min=H_min, H_max=H_max, S_min=S_min, V_min=V_min)
    stats = dict(n=len(Hs),
                 H_range=(float(np.min(Hs)), float(np.max(Hs))),
                 S_range=(float(np.min(Ss)), float(np.max(Ss))),
                 V_range=(float(np.min(Vs)), float(np.max(Vs))))
    return thr, stats

# ---------- B) 检测：在 ROI 内用阈值做分割并给面积占比 ----------
def detect_dart_in_roi(frame_bgr, roi_rect=None, roi_poly=None, hsv_thr=None,
                       v_high=220, s_low=40, rgb_high=240,
                       morph_open=3, morph_close=5,
                       area_ratio_thresh=0.18,  # TODO: 面积占比阈值，可以按孔调，给孔做一个map
                       return_debug=False):
    """
    返回: present(bool), ratio(float), out(dict可选)
    out 包含: mask(ROI同尺寸二值), overlay(可视化), bbox/contour 等
    """
    assert roi_rect or roi_poly, "需要传 ROI（矩形或多边形）"
    assert hsv_thr is not None, "需要传已标定的阈值 hsv_thr"

    if roi_rect:
        x,y,w,h = roi_rect
        roi = frame_bgr[y:y+h, x:x+w]
        mask_roi = np.ones((h, w), np.uint8) * 255
        roi_origin = (x, y)
    else:
        mask_roi = np.zeros(frame_bgr.shape[:2], np.uint8)
        pts = np.array(roi_poly, dtype=np.int32).reshape((-1,1,2))
        cv2.fillPoly(mask_roi, [pts], 255)
        roi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_roi)
        x,y,w,h = cv2.boundingRect(pts)
        roi = roi[y:y+h, x:x+w]
        mask_roi = mask_roi[y:y+h, x:x+w]
        roi_origin = (x, y)

    # 高光掩膜
    hl = _highlight_mask(roi, v_high=v_high, s_low=s_low, rgb_high=rgb_high)
    valid = cv2.bitwise_and(mask_roi, cv2.bitwise_not(hl))

    # HSV 阈值
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    mH = (H >= hsv_thr["H_min"]) & (H <= hsv_thr["H_max"])
    mS = (S >= hsv_thr["S_min"])
    mV = (V >= hsv_thr["V_min"])
    mask = (mH & mS & mV).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, valid)

    # 形态学清理
    mask = _refine_mask(mask, k_open=morph_open, it_open=1, k_close=morph_close, it_close=1)

    # 面积占比
    area = float(np.count_nonzero(mask))
    denom = float(np.count_nonzero(mask_roi))
    ratio = (area / denom) if denom > 0 else 0.0
    present = ratio >= area_ratio_thresh

    if not return_debug:
        return bool(present), float(ratio), None

    # 可视化
    overlay = frame_bgr.copy()
    if roi_rect:
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,255), 2)
    else:
        if roi_poly:
            if not roi_poly or len(roi_poly) < 3:
                raise ValueError("roi_poly 必须是非空的顶点列表，且至少包含 3 个点")
            pts = np.array(roi_poly, dtype=np.int32).reshape(-1, 2)  # 转换为 (n, 2) 的数组
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
    # 在原图上画 mask 轮廓
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No contours found in mask")
    if roi_origin is None:
        raise ValueError("roi_origin 必须定义为 (x, y) 坐标")
    roi_origin = np.array(roi_origin, dtype=np.int32).reshape(1, 1, 2)  # 转换为 (1, 1, 2)
    for c in contours:
        c2 = c + roi_origin  # 平移到原图坐标系
        cv2.polylines(overlay, [c2], True, (0,255,0), 2)

    out = dict(mask=mask, overlay=overlay, contours=contours, roi_origin=roi_origin)
    return bool(present), float(ratio), out

# ---------- C) 多帧投票（稳定判定） ----------
def vote_presence(present_flags, need_true=3):
    """
    present_flags: 最近 N 帧的布尔判定列表
    need_true: 至少多少帧为 True 才判定“有”
    """
    return sum(bool(x) for x in present_flags) >= int(need_true)

# 保存阈值到 JSON
def save_hsv_thresholds(hsv_thr, file_path):
    """
    将HSV阈值保存到JSON文件
    
    Args:
        hsv_thr: 阈值字典 {H_min, H_max, S_min, V_min}
        file_path: JSON文件保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(hsv_thr, f, indent=4, ensure_ascii=False)
    
    print(f"HSV阈值已保存到: {file_path}")
    
# 读取阈值从 JSON
def load_hsv_thresholds(file_path):
    """
    从JSON文件加载HSV阈值
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        dict: HSV阈值字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"阈值文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        hsv_thr = json.load(f)
    
    print(f"HSV阈值已从 {file_path} 加载")
    return hsv_thr

def find_dart(frames, roi):
    # TODO: 传样片
    # 传多个frame
    # img_path = os.path.join(ROOT, "tools/img", "1.png")
    # fnames = [cv2.imread(img_path)]  # 你自己的样片列表
    # 假设你已经采集了几张样片 frames（BGR），并知道该孔的 ROI
    # roi = (332, 285, 94, 94)  # 或 roi_poly=[(x1,y1),...]
    # file_path = os.path.join(ROOT, "data", "dart_hsv_thr.json")
    

    # hsv_thr, stats = calibrate_hsv_thresholds(fnames, roi_rect=roi)
    # print("建议阈值:", hsv_thr, "样本范围(H/S/V):", stats)
    # save_hsv_thresholds(hsv_thr, file_path)
    # 把 hsv_thr 存成 JSON，比赛时直接读

    hsv_thr, stats = calibrate_hsv_thresholds(frames, roi_rect=roi)
    print("建议阈值:", hsv_thr, "样本范围(H/S/V):", stats)
    
    # loaded_hsv_thr = load_hsv_thresholds(file_path)
    present_flags = []
    for _ in range(5):  # 连续5帧投票
        frame = frames[0]  # 你自己的实时帧
        present, ratio, out = detect_dart_in_roi(
            frame, roi_rect=roi, hsv_thr=hsv_thr,
            area_ratio_thresh=0.18, return_debug=True
        )
        print("该孔面积占比: %.3f, 判定有飞镖: %s" % (ratio, present))
        present_flags.append(present)
        # 可视化查看
        # cv2.imshow("overlay", out["overlay"])
        # cv2.imshow("mask", out["mask"])
        # cv2.waitKey(0)

    final_decision = vote_presence(present_flags, need_true=1)
    print("该孔是否有飞镖:", final_decision)
    return final_decision
